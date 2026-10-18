//! Replayable CUDA graph for the per-frame conditional flow matching step.
//!
//! The acoustic LocDiT call runs once per generated frame with a constant
//! shape (two `mu` tokens, four conditioning tokens, 64 latent positions, batch
//! two for classifier-free guidance) and a constant RoPE offset of zero, so it
//! can be recorded once and replayed for every later frame.
//!
//! Replay reads its inputs from fixed buffers so the recorded addresses stay
//! valid: each frame copies the freshly computed conditioning, the previous
//! patch and the new noise into those buffers before launching. Candle's
//! functional API has no in-place tensor write, so those copies are cudarc
//! device-to-device memcpys issued on the same stream, which keeps them ordered
//! against both the producing ops and the replay.
//!
//! # Balanced allocation nodes
//!
//! Candle serves every allocation with cudarc's stream-ordered allocator, and an
//! allocation issued while the stream is capturing becomes an allocation node in
//! the graph. A graph that holds an allocation node without a matching free node
//! can only be launched once: every later `cuGraphLaunch` fails with
//! `CUDA_ERROR_INVALID_VALUE`, which shows up as an unusable graph rather than as
//! a capture error. Candle's functional style already drops each intermediate as
//! soon as the next op consumes it, so those allocations are paired inside the
//! recorded region. The step result is the one allocation that would outlive it,
//! so the recorded region copies the result into a buffer allocated beforehand
//! and frees it, leaving the graph balanced and its output address stable.
//!
//! # Cancellation
//!
//! The recorded region runs the step from start to finish, so cancellation is
//! polled before a capture and before a replay instead of inside the graph. That
//! makes the granularity one frame rather than one transformer layer; PCM is
//! still produced outside the graph, so a cancelled frame emits nothing.
//!
//! Any capture or replay failure permanently disables graphs for the owning model
//! and falls back to the ordinary eager path.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::cuda_backend::{CudaDevice, CudaStorageSlice};
use candle_core::{CudaStorage, DType, Device, Result, Storage, Tensor};
use cudarc::driver::sys;
use cudarc::driver::{CudaGraph, CudaStream, DevicePtr, SyncOnDrop};

use crate::acoustic::Acoustic;

/// How many specializations stay alive; each one owns a graph and its buffers.
const MAX_CAPTURED: usize = 4;

/// `VOXCPM_CUDA_GRAPH=0` forces the eager path even on CUDA.
fn enabled_by_env() -> bool {
    std::env::var_os("VOXCPM_CUDA_GRAPH").is_none_or(|value| value != "0")
}

/// candle does not implement `From<cudarc::DriverError>`, so keep the driver
/// message instead of collapsing it into an opaque error.
fn driver_error(error: cudarc::driver::DriverError) -> candle_core::Error {
    candle_core::Error::Msg(format!("cuda driver: {error:?}"))
}

/// Identifies one captured specialization. Every field is baked into the graph,
/// so a different value has to be recorded separately.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    steps: usize,
    /// `Options::cfg` becomes a scalar kernel argument inside the recorded
    /// region, so different guidance scales need different graphs.
    cfg: u64,
    dtype: DType,
}

/// A fixed device allocation whose address a recorded graph can bake in.
///
/// Replays read from these addresses, so a buffer is allocated once and then
/// only overwritten in place.
struct Buffer {
    tensor: Tensor,
    ptr: sys::CUdeviceptr,
    bytes: usize,
}

struct Captured {
    graph: CudaGraph,
    /// Replay output; the graph writes here on every launch.
    patch: Buffer,
    mu: Buffer,
    prefix: Buffer,
    noise: Buffer,
    used: u64,
}

pub(crate) struct CfmGraphs {
    stream: Arc<CudaStream>,
    device: CudaDevice,
    captured: HashMap<GraphKey, Captured>,
    clock: u64,
    /// Set once capture or replay fails; the model then stays on the eager path.
    disabled: bool,
}

impl CfmGraphs {
    pub fn new(device: &Device) -> Option<Self> {
        if !enabled_by_env() {
            return None;
        }
        let Device::Cuda(cuda) = device else {
            return None;
        };
        Some(Self {
            stream: cuda.cuda_stream(),
            device: cuda.clone(),
            captured: HashMap::new(),
            clock: 0,
            disabled: false,
        })
    }

    /// True once at least one specialization has been recorded and no failure has
    /// forced the model back to the eager path.
    pub fn active(&self) -> bool {
        !self.disabled && !self.captured.is_empty()
    }

    /// Runs one CFM step, recording it on the first call of each specialization.
    ///
    /// Recording a step does not execute it, so a cache miss runs the frame
    /// eagerly and only records the step for the frames after it. The first
    /// replay therefore writes into a buffer that predates the recording, and
    /// every later frame replays the same kernels on the same shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        acoustic: &mut Acoustic,
        mu: &Tensor,
        prefix: &Tensor,
        noise: &Tensor,
        steps: usize,
        cfg: f64,
        cancel: &impl Fn() -> bool,
    ) -> Result<Tensor> {
        if self.disabled {
            return acoustic.sample(mu, prefix, noise, steps, cfg, cancel);
        }
        let key = GraphKey {
            steps,
            cfg: cfg.to_bits(),
            dtype: mu.dtype(),
        };
        if self.captured.contains_key(&key) {
            match self.replay(mu, prefix, noise, &key, cancel) {
                Ok(patch) => return Ok(patch),
                Err(error) => {
                    // A cancelled frame is not a graph failure, so the graphs stay
                    // usable for the next request.
                    if cancel() {
                        return Err(error);
                    }
                    tracing::warn!(%error, "voxcpm cuda graph replay failed; using eager sampling");
                    self.disable();
                    return acoustic.sample(mu, prefix, noise, steps, cfg, cancel);
                }
            }
        }
        match self.record(acoustic, mu, prefix, noise, steps, cfg, cancel, key) {
            Ok(()) => acoustic.sample(mu, prefix, noise, steps, cfg, cancel),
            Err(error) => {
                if cancel() {
                    return Err(error);
                }
                tracing::warn!(%error, "voxcpm cuda graph record failed; using eager sampling");
                self.disable();
                acoustic.sample(mu, prefix, noise, steps, cfg, cancel)
            }
        }
    }

    fn replay(
        &mut self,
        mu: &Tensor,
        prefix: &Tensor,
        noise: &Tensor,
        key: &GraphKey,
        cancel: &impl Fn() -> bool,
    ) -> Result<Tensor> {
        // The graph carries no cancellation check, so the frame has to be known to
        // be wanted before it is launched.
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        self.clock += 1;
        let Self {
            stream,
            captured,
            clock,
            ..
        } = self;
        let entry = captured.get_mut(key).expect("checked by the caller");
        entry.used = *clock;
        write(stream, &entry.mu, mu)?;
        write(stream, &entry.prefix, prefix)?;
        write(stream, &entry.noise, noise)?;
        entry.graph.launch().map_err(driver_error)?;
        Ok(entry.patch.tensor.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        acoustic: &mut Acoustic,
        mu: &Tensor,
        prefix: &Tensor,
        noise: &Tensor,
        steps: usize,
        cfg: f64,
        cancel: &impl Fn() -> bool,
        key: GraphKey,
    ) -> Result<()> {
        // Candle only fills its host-to-device parameter caches while this guard is
        // held, and it refuses to upload anything uncached once the stream is
        // capturing. The warm-up therefore has to run inside the guard, on the
        // same shapes, so every upload the recording needs already has a cached
        // device copy.
        let _guard = self.device.enable_cuda_graph_htod_cache();
        // cuBLAS workspaces, quantized matmul kernels and the time schedule
        // allocate on first use, so warm them up before recording.
        let warmup = acoustic.sample(mu, prefix, noise, steps, cfg, cancel)?;
        // A cancellation that fires mid-recording would end it early and leave a
        // graph that computes only a prefix of the step, so a cancelled frame
        // never starts one.
        if cancel() {
            candle_core::bail!("generation cancelled");
        }

        // Every buffer has to exist before recording starts: a memcpy recorded
        // into the graph would replay from the address it captured and overwrite
        // the values written before each launch.
        let mu_buf = Buffer::new(&self.stream, &self.device, mu)?;
        let prefix_buf = Buffer::new(&self.stream, &self.device, prefix)?;
        let noise_buf = Buffer::new(&self.stream, &self.device, noise)?;
        let patch_buf = Buffer::new(&self.stream, &self.device, &warmup)?;
        write(&self.stream, &mu_buf, mu)?;
        write(&self.stream, &prefix_buf, prefix)?;
        write(&self.stream, &noise_buf, noise)?;
        drop(warmup);

        let recorded = (|| -> Result<Captured> {
            self.stream
                .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(driver_error)?;
            // The recorded region must run start to finish, so cancellation is
            // forced off inside it and polled by the caller instead.
            let patch = acoustic.sample(
                &mu_buf.tensor,
                &prefix_buf.tensor,
                &noise_buf.tensor,
                steps,
                cfg,
                &|| false,
            )?;
            // Hand the result to a buffer that predates the recording and release
            // the step's own allocation inside the region, so every allocation
            // node is matched by a free node and the graph stays replayable.
            write(&self.stream, &patch_buf, &patch)?;
            drop(patch);
            let graph = self
                .stream
                .end_capture(sys::CUgraphInstantiate_flags_enum(0))
                .map_err(driver_error)?;
            let Some(graph) = graph else {
                candle_core::bail!("cuda graph capture produced no graph");
            };
            Ok(Captured {
                graph,
                patch: patch_buf,
                mu: mu_buf,
                prefix: prefix_buf,
                noise: noise_buf,
                used: self.clock,
            })
        })();

        // A failed recording can leave the stream capturing, and the next frame
        // would then run inside a graph that is thrown away. Cleanup must not mask
        // the original failure.
        match self.stream.capture_status() {
            Ok(status)
                if status != sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_NONE =>
            {
                let _ = self
                    .stream
                    .end_capture(sys::CUgraphInstantiate_flags_enum(0));
            }
            Ok(_) => {}
            Err(error) => tracing::warn!(?error, "cuda graph capture status unavailable"),
        }

        let entry = recorded?;
        self.captured.insert(key, entry);
        while self.captured.len() > MAX_CAPTURED {
            let oldest = self
                .captured
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| *key);
            match oldest {
                Some(key) => {
                    self.captured.remove(&key);
                }
                None => break,
            }
        }
        Ok(())
    }

    fn disable(&mut self) {
        self.disabled = true;
        self.captured.clear();
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
        let (ptr, bytes) = {
            let (storage, layout) = tensor.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                candle_core::bail!("cuda graph buffers must live on cuda");
            };
            if !layout.is_contiguous() {
                candle_core::bail!("cuda graph buffers must be contiguous");
            }
            let (ptr, bytes, _sync) = device_slice(storage, stream)?;
            (ptr, bytes)
        };
        Ok(Self { tensor, ptr, bytes })
    }
}

/// Overwrites a fixed buffer with the values produced this frame.
fn write(stream: &CudaStream, buffer: &Buffer, src: &Tensor) -> Result<()> {
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
    let (ptr, bytes, sync) = device_slice(storage, stream)?;
    memcpy(stream, ptr, bytes, buffer, sync)
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
