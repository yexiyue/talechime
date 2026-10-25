//! Request-owned, stateless stage0 CUDA forward graph.
use super::*;

#[path = "cuda_graph_replay.rs"]
pub(super) mod replay;

/// Borrows the model so captured weights cannot outlive their owner. Constants
/// are copied into the graph inputs; no mutable history is shared across calls.
pub(super) struct ForwardGraph<'a> {
    graph: Option<replay::Replay>,
    model: &'a Stage0Model,
    disabled: bool,
}

impl<'a> ForwardGraph<'a> {
    pub(super) fn new(model: &'a Stage0Model) -> Self {
        Self {
            graph: None,
            model,
            disabled: false,
        }
    }

    pub(super) fn forward(&mut self, inputs: [&Tensor; 4], lengths: &[usize]) -> Result<Tensor> {
        if let Some(graph) = &self.graph {
            // A launch error terminates this request, rather than retrying on a
            // possibly damaged CUDA context. Cancellation stays between steps.
            return graph.launch(&inputs).map_err(Into::into);
        }
        let Device::Cuda(cuda) = inputs[0].device() else {
            unreachable!()
        };
        let _guard = cuda.enable_cuda_graph_htod_cache();
        let warm = eager(self.model, &inputs, lengths)?;
        if !self.disabled {
            let result = replay::Replay::record(cuda, &inputs, &warm, vec![], |args| {
                eager(
                    self.model,
                    &[&args[0], &args[1], &args[2], &args[3]],
                    lengths,
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
            });
            match result {
                Ok(graph) => {
                    self.graph = Some(graph);
                    #[cfg(test)]
                    super::cuda_graph_probe::recorded();
                }
                Err(error) => {
                    if replay::is_capturing(cuda.cuda_stream().as_ref())? {
                        return Err(error.into());
                    }
                    self.disabled = true;
                    tracing::warn!(%error,"OmniVoice CUDA forward capture failed; request stays eager");
                }
            }
        }
        // Capture records kernels without executing them. Deliver the actual
        // eager result of this step; first replay is the next denoise step.
        Ok(warm)
    }
}

fn eager(model: &Stage0Model, inputs: &[&Tensor; 4], lengths: &[usize]) -> Result<Tensor> {
    Ok(model
        .forward_with_text_cache(
            inputs[0],
            inputs[1],
            inputs[2],
            &BTreeSet::new(),
            Some(inputs[3]),
            Some(lengths),
        )?
        .logits)
}
