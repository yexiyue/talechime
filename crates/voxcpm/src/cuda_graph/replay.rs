//! Fixed-buffer CUDA recording, independent of the operation being recorded.
use candle_core::cuda_backend::{CudaDevice, CudaStorageSlice};
use candle_core::{CudaStorage, Device, Result, Storage, Tensor};
use cudarc::driver::{CudaGraph, CudaStream, DevicePtr, SyncOnDrop, sys};
use std::sync::Arc;

pub(super) fn driver_error(error: cudarc::driver::DriverError) -> candle_core::Error {
    candle_core::Error::Msg(format!("cuda driver: {error:?}"))
}

struct Buffer {
    tensor: Tensor,
    ptr: sys::CUdeviceptr,
    bytes: usize,
}

/// Output is borrowed scratch: consume it on the same stream before replaying.
/// The graph is destroyed before any buffers or captured parameters are dropped.
pub(super) struct Replay {
    graph: CudaGraph,
    stream: Arc<CudaStream>,
    output: Buffer,
    inputs: Vec<Buffer>,
    _parameters: Vec<Tensor>,
}

impl Replay {
    pub fn record(
        device: &CudaDevice,
        inputs: &[&Tensor],
        warmup: &Tensor,
        parameters: Vec<Tensor>,
        mut operation: impl FnMut(&[Tensor]) -> Result<Tensor>,
    ) -> Result<Self> {
        let stream = device.cuda_stream();
        let inputs = inputs
            .iter()
            .map(|input| {
                let buffer = Buffer::new(&stream, device, input)?;
                write(&stream, &buffer, input)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>>>()?;
        let output = Buffer::new(&stream, device, warmup)?;
        let tensors: Vec<_> = inputs.iter().map(|b| b.tensor.clone()).collect();
        stream
            .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(driver_error)?;
        let capture = Capture(&stream);
        // Free every capture allocation before ending the recording, including
        // the result. Only buffers allocated before capture survive a replay.
        let result = operation(&tensors)?;
        write(&stream, &output, &result)?;
        drop(result);
        let graph = stream
            .end_capture(sys::CUgraphInstantiate_flags_enum(0))
            .map_err(driver_error)?
            .ok_or_else(|| {
                candle_core::Error::Msg("cuda graph capture produced no graph".into())
            })?;
        drop(capture);
        Ok(Self {
            graph,
            stream,
            output,
            inputs,
            _parameters: parameters,
        })
    }

    pub fn launch(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != self.inputs.len() {
            candle_core::bail!("cuda graph input count changed");
        }
        for (buffer, input) in self.inputs.iter().zip(inputs) {
            write(&self.stream, buffer, input)?;
        }
        self.graph.launch().map_err(driver_error)?;
        Ok(self.output.tensor.clone())
    }
}

/// Always exit capture before buffers are dropped, including early errors.
struct Capture<'a>(&'a Arc<CudaStream>);
impl Drop for Capture<'_> {
    fn drop(&mut self) {
        match self.0.capture_status() {
            Ok(sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_NONE) => {}
            Ok(_) => {
                let _ = self.0.end_capture(sys::CUgraphInstantiate_flags_enum(0));
            }
            Err(error) => tracing::warn!(?error, "cuda graph capture status unavailable"),
        }
    }
}

impl Buffer {
    /// Allocates a fixed device buffer shaped like `like`.
    ///
    /// Candle allocates it, so the storage layout matches what the recorded
    /// kernels expect; the device pointer is then read back so replays can
    /// overwrite the same address.
    fn new(stream: &Arc<CudaStream>, device: &CudaDevice, like: &Tensor) -> Result<Self> {
        let tensor = Tensor::zeros(
            like.shape().clone(),
            like.dtype(),
            &Device::Cuda(device.clone()),
        )?;
        Self::from_tensor(stream, &tensor)
    }

    fn from_tensor(stream: &CudaStream, tensor: &Tensor) -> Result<Self> {
        let (ptr, bytes) = {
            let (storage, layout) = tensor.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                candle_core::bail!("cuda graph buffers must live on cuda");
            };
            if !layout.is_contiguous() {
                candle_core::bail!("cuda graph buffers must be contiguous");
            }
            let (ptr, _, _sync) = device_slice(storage, stream)?;
            let bytes = tensor.elem_count() * tensor.dtype().size_in_bytes();
            let offset = layout.start_offset() * tensor.dtype().size_in_bytes();
            (ptr + offset as u64, bytes)
        };
        Ok(Self {
            tensor: tensor.clone(),
            ptr,
            bytes,
        })
    }
}

pub(super) fn copy_into(stream: &CudaStream, destination: &Tensor, source: &Tensor) -> Result<()> {
    write(stream, &Buffer::from_tensor(stream, destination)?, source)
}

/// Overwrites a fixed buffer with the values produced this frame.
fn write(stream: &CudaStream, buffer: &Buffer, src: &Tensor) -> Result<()> {
    if !buffer.tensor.device().same_device(src.device()) {
        candle_core::bail!("cuda graph input device changed");
    }
    if buffer.tensor.dims() != src.dims() || buffer.tensor.dtype() != src.dtype() {
        candle_core::bail!("cuda graph buffer shape or dtype changed");
    }
    // Conditioning read from reference audio is a narrowed, transposed view into a
    // much larger storage, so it has to be materialized before its raw pointer and
    // element count can be used.
    let src = src.contiguous()?;
    let (storage, layout) = src.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("cuda graph inputs must live on cuda");
    };
    if !layout.is_contiguous() {
        candle_core::bail!("cuda graph input is not contiguous");
    }
    let (base, allocated, sync) = device_slice(storage, stream)?;
    let element_bytes = src.dtype().size_in_bytes();
    let offset = layout.start_offset() * element_bytes;
    let bytes = src.elem_count() * element_bytes;
    if offset > allocated || bytes > allocated - offset {
        candle_core::bail!("cuda graph input exceeds its allocation");
    }
    memcpy(stream, base + offset as u64, bytes, buffer, sync)
}

fn memcpy(
    stream: &CudaStream,
    source: sys::CUdeviceptr,
    bytes: usize,
    buffer: &Buffer,
    sync: SyncOnDrop<'_>,
) -> Result<()> {
    if bytes != buffer.bytes {
        candle_core::bail!("cuda graph buffer size changed");
    }
    // SAFETY: both pointers are device allocations of `bytes` bytes in the same
    // context, and the stream orders this copy after the kernels that produced the
    // source.
    let status =
        unsafe { sys::cuMemcpyDtoDAsync_v2(buffer.ptr, source, bytes, stream.cu_stream()) };
    // Keep the source alive until the copy has been scheduled.
    drop(sync);
    if status != sys::cudaError_enum::CUDA_SUCCESS {
        candle_core::bail!("cuMemcpyDtoDAsync failed: {status:?}");
    }
    Ok(())
}

/// Device pointer, byte length and the event guard that keeps the storage alive
/// until the copy using the pointer has been recorded.
///
/// The storage slice variants carry the concrete element type, so the pointer and
/// byte count come from the variant instead of from a dtype-to-type conversion
/// this crate cannot name.
fn device_slice<'a>(
    storage: &'a CudaStorage,
    stream: &'a CudaStream,
) -> Result<(sys::CUdeviceptr, usize, SyncOnDrop<'a>)> {
    let (ptr, bytes, sync) = match &storage.slice {
        CudaStorageSlice::F16(slice) => {
            let (ptr, sync) = slice.device_ptr(stream);
            (ptr, slice.len() * 2, sync)
        }
        CudaStorageSlice::BF16(slice) => {
            let (ptr, sync) = slice.device_ptr(stream);
            (ptr, slice.len() * 2, sync)
        }
        CudaStorageSlice::F32(slice) => {
            let (ptr, sync) = slice.device_ptr(stream);
            (ptr, slice.len() * 4, sync)
        }
        other => candle_core::bail!("cuda graph input storage {other:?} is not supported"),
    };
    Ok((ptr, bytes, sync))
}
