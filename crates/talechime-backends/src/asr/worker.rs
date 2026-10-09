use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{mpsc, oneshot};
use tts_core::verification::{ReadbackAudio, Recognition, RecognitionRequest, VerificationError};
pub(super) enum Model {
    Qwen,
    SenseVoice,
}
struct Request {
    audio: Arc<ReadbackAudio>,
    cancelled: Arc<AtomicBool>,
    run: Arc<ort::session::RunOptions>,
    _completion: Completion,
    result: oneshot::Sender<Result<String, VerificationError>>,
}
struct Owner {
    jobs: Option<mpsc::Sender<Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    identity: RecognizerIdentity,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
struct Completion(tokio::sync::watch::Sender<bool>);
impl Drop for Completion {
    fn drop(&mut self) {
        self.0.send_replace(true);
    }
}
struct Cancel(Arc<AtomicBool>, Arc<ort::session::RunOptions>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
        let _ = self.1.terminate();
    }
}
impl Recognizer for Owner {
    fn identity(&self) -> RecognizerIdentity {
        self.identity.clone()
    }
    fn transcribe(&self, audio: Arc<ReadbackAudio>) -> Recognition<'_> {
        self.request(audio).future
    }
    fn request(&self, audio: Arc<ReadbackAudio>) -> RecognitionRequest<'_> {
        let (done, completion) = tokio::sync::watch::channel(false);
        let work = Completion(done);
        let future = Box::pin(async move {
            let cancelled = Arc::new(AtomicBool::new(false));
            let run = Arc::new(
                ort::session::RunOptions::new()
                    .map_err(|e| VerificationError::Recognition(e.to_string()))?,
            );
            let _guard = Cancel(cancelled.clone(), run.clone());
            let (result, received) = oneshot::channel();
            self.jobs
                .as_ref()
                .ok_or_else(|| VerificationError::Recognition("ASR owner closed".into()))?
                .send(Request {
                    audio,
                    cancelled,
                    run,
                    result,
                    _completion: work,
                })
                .await
                .map_err(|e| VerificationError::Recognition(e.to_string()))?;
            received
                .await
                .map_err(|e| VerificationError::Recognition(e.to_string()))?
        });
        RecognitionRequest {
            future,
            completion: Some(completion),
        }
    }
}
enum Loaded {
    Qwen(Box<qwen3_asr::AsrInference>),
    SenseVoice(sensevoice::SenseVoice),
}
pub(super) async fn spawn(
    directory: PathBuf,
    variant: Model,
    threads: usize,
    identity: RecognizerIdentity,
) -> anyhow::Result<Rc<dyn Recognizer>> {
    let (jobs, mut received) = mpsc::channel::<Request>(1);
    let (ready, prepared) = oneshot::channel();
    // Model/device objects are constructed, used and dropped on this dedicated owner.
    let thread = std::thread::Builder::new()
        .name(format!("asr-{}", identity.family))
        .spawn(move || {
            let model = match variant {
                Model::Qwen => qwen3_asr::AsrInference::load(&directory, candle_core::Device::Cpu)
                    .map(|model| Loaded::Qwen(Box::new(model)))
                    .map_err(|e| e.to_string()),
                Model::SenseVoice => sensevoice::SenseVoice::load(&directory, threads)
                    .map(Loaded::SenseVoice)
                    .map_err(|e| e.to_string()),
            };
            let mut model = match model {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error));
                    return;
                }
            };
            if ready.send(Ok(())).is_err() {
                return;
            }
            while let Some(request) = received.blocking_recv() {
                if request.result.is_closed() || request.cancelled.load(Ordering::Relaxed) {
                    continue;
                }
                let result = (|| -> anyhow::Result<String> {
                    let samples = mono_16k(&request.audio)?;
                    // Exact silence must not become a generative ASR hallucination.
                    if samples.iter().all(|x| x.abs() <= 1e-7) {
                        return Ok(String::new());
                    }
                    match &mut model {
                        Loaded::Qwen(model) => {
                            let mut options = qwen3_asr::TranscribeOptions::default()
                                .with_language("Chinese")
                                .with_max_new_tokens(256);
                            options.cancelled = request.cancelled.clone();
                            Ok(model.transcribe_samples(&samples, options)?.text)
                        }
                        Loaded::SenseVoice(model) => {
                            model.transcribe(&samples, &request.cancelled, &request.run)
                        }
                    }
                })()
                .map_err(|e| VerificationError::Recognition(e.to_string()));
                let _ = request.result.send(result);
            }
        })?;
    let owner = Owner {
        jobs: Some(jobs),
        thread: Some(thread),
        identity,
    };
    prepared.await?.map_err(anyhow::Error::msg)?;
    Ok(Rc::new(owner))
}
fn mono_16k(audio: &ReadbackAudio) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        audio.sample_rate > 0
            && audio.channels > 0
            && !audio.samples.is_empty()
            && audio.samples.len().is_multiple_of(audio.channels as usize)
            && audio.samples.iter().all(|x| x.is_finite()),
        "invalid ASR PCM"
    );
    anyhow::ensure!(
        audio.samples.len() <= 4 * 1024 * 1024
            && audio.samples.len() / audio.channels as usize <= audio.sample_rate as usize * 30,
        "ASR PCM bound exceeded"
    );
    let mono: Vec<f32> = audio
        .samples
        .chunks_exact(audio.channels as usize)
        .map(|x| x.iter().sum::<f32>() / audio.channels as f32)
        .collect();
    if audio.sample_rate == 16000 {
        return Ok(mono);
    }
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    let parameters = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / audio.sample_rate as f64,
        2.0,
        parameters,
        mono.len(),
        1,
    )?;
    let delay = resampler.output_delay();
    let frames = (mono.len() as u64 * 16000).div_ceil(audio.sample_rate as u64) as usize;
    let mut output = resampler.process(&[mono], None)?.remove(0);
    output.extend(resampler.process_partial::<Vec<f32>>(None, None)?.remove(0));
    anyhow::ensure!(output.len() >= delay + frames, "incomplete resampling");
    Ok(output[delay..delay + frames].to_vec())
}
