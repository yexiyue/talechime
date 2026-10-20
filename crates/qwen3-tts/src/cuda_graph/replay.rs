//! Private CUDA recorder, copied from Talechime’s former shared graph crate.
//! Kept local until upstream Candle provides a suitable replacement; preserve
//! buffer ownership and graph-before-buffer destruction when replacing it.
use candle_core::cuda_backend::{CudaDevice, CudaStorageSlice};
use candle_core::{CudaStorage, Device, Result, Storage, Tensor};
use cudarc::driver::{CudaGraph, CudaStream, DevicePtr, SyncOnDrop, sys};
use std::sync::Arc;

fn driver_error(error: cudarc::driver::DriverError) -> candle_core::Error {
    candle_core::Error::Msg(format!("cuda driver: {error:?}"))
}

/// Whether `stream` is still inside an unfinished capture.
///
/// A recording that ends inside capture leaves the stream in a capturing
/// state, which silently invalidates every later capture on it.
pub(crate) fn is_capturing(stream: &CudaStream) -> Result<bool> {
    Ok(stream.capture_status().map_err(driver_error)?
        != sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_NONE)
}

struct Buffer {
    tensor: Tensor,
    ptr: sys::CUdeviceptr,
    bytes: usize,
}

/// Output is borrowed scratch: consume it on the same stream before replaying.
/// The graph is destroyed before any buffers or captured parameters are dropped.
pub(crate) struct Replay {
    graph: CudaGraph,
    stream: Arc<CudaStream>,
    output: Buffer,
    inputs: Vec<Buffer>,
    _parameters: Vec<Tensor>,
}

impl Replay {
    /// Records `operation` once, with `inputs` copied into persistent buffers.
    ///
    /// `warmup` must be the result of the same operation on the same inputs:
    /// its value is both the output buffer's shape and this frame's result,
    /// because recording does not execute. `parameters` are tensors the graph
    /// reads that the caller would otherwise drop.
    pub(crate) fn record(
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
        // Prime the copy kernel's layout parameters too: an operation may
        // return a strided view that its eager warm-up never materialized.
        write(&stream, &output, warmup)?;
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

    /// Overwrites the persistent inputs and re-executes the recorded kernels.
    ///
    /// Returns the output buffer; it stays valid until the next launch.
    pub(crate) fn launch(&self, inputs: &[&Tensor]) -> Result<Tensor> {
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

/// Overwrites a fixed buffer with the values produced this frame.
fn write(stream: &CudaStream, buffer: &Buffer, src: &Tensor) -> Result<()> {
    if !buffer.tensor.device().same_device(src.device()) {
        candle_core::bail!("cuda graph input device changed");
    }
    if buffer.tensor.dims() != src.dims() || buffer.tensor.dtype() != src.dtype() {
        candle_core::bail!("cuda graph buffer shape or dtype changed");
    }
    // Conditioning read from reference audio can be a narrowed, transposed view
    // into a much larger storage, so it has to be materialized before its raw
    // pointer and element count can be used.
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
    if bytes != buffer.bytes {
        candle_core::bail!("cuda graph buffer size changed");
    }
    // SAFETY: both pointers are device allocations of `bytes` bytes in the same
    // context, and the stream orders this copy after the kernels that produced
    // the source.
    let status = unsafe {
        sys::cuMemcpyDtoDAsync_v2(buffer.ptr, base + offset as u64, bytes, stream.cu_stream())
    };
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
/// this crate cannot name. Integer codes are first-class: Qwen's acoustic codes
/// are `U32`, not floating point.
fn device_slice<'a>(
    storage: &'a CudaStorage,
    stream: &'a CudaStream,
) -> Result<(sys::CUdeviceptr, usize, SyncOnDrop<'a>)> {
    let (entry, bytes) = match &storage.slice {
        CudaStorageSlice::U8(slice) => (slice.device_ptr(stream), slice.len()),
        CudaStorageSlice::U32(slice) => (slice.device_ptr(stream), slice.len() * 4),
        CudaStorageSlice::I16(slice) => (slice.device_ptr(stream), slice.len() * 2),
        CudaStorageSlice::I32(slice) => (slice.device_ptr(stream), slice.len() * 4),
        CudaStorageSlice::I64(slice) => (slice.device_ptr(stream), slice.len() * 8),
        CudaStorageSlice::BF16(slice) => (slice.device_ptr(stream), slice.len() * 2),
        CudaStorageSlice::F16(slice) => (slice.device_ptr(stream), slice.len() * 2),
        CudaStorageSlice::F32(slice) => (slice.device_ptr(stream), slice.len() * 4),
        CudaStorageSlice::F64(slice) => (slice.device_ptr(stream), slice.len() * 8),
        CudaStorageSlice::F8E4M3(slice) => (slice.device_ptr(stream), slice.len()),
        CudaStorageSlice::F6E2M3(slice) => (slice.device_ptr(stream), slice.len()),
        CudaStorageSlice::F6E3M2(slice) => (slice.device_ptr(stream), slice.len()),
        CudaStorageSlice::F4(slice) => (slice.device_ptr(stream), slice.len()),
        CudaStorageSlice::F8E8M0(slice) => (slice.device_ptr(stream), slice.len()),
    };
    let (ptr, sync) = entry;
    Ok((ptr, bytes, sync))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    #[ignore = "requires CUDA; run serially"]
    fn strided_output_copy_is_warmed_before_capture() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let _guard = cuda.enable_cuda_graph_htod_cache();
        let input = Tensor::arange(0f32, (2 * 52 * 80) as f32, &device)?.reshape((2, 52, 80))?;
        let warm = input.transpose(1, 2)?;
        let replay = Replay::record(cuda, &[&input], &warm, vec![], |args| {
            args[0].transpose(1, 2)
        })?;
        for offset in [1.0, 3.0, 1.0] {
            let input = input.affine(1.0, offset)?;
            let expected = input.transpose(1, 2)?.flatten_all()?.to_vec1::<f32>()?;
            let actual = replay.launch(&[&input])?;
            assert_eq!(actual.dtype(), DType::F32);
            assert_eq!(actual.flatten_all()?.to_vec1::<f32>()?, expected);
        }
        Ok(())
    }
}
