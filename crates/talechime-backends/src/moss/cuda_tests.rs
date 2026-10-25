use super::*;
use diagnostics::GenerationEnd;
use std::time::Duration;

mod profile;

async fn collect(backend: &MossBackend, voice: &str) -> Vec<u32> {
    let (mut stream, report) = backend
        .stream_diagnosed("你好，欢迎使用叙铃。", voice, Some(42))
        .await
        .unwrap();
    let mut bits = Vec::new();
    let mut ended = false;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.recv())
        .await
        .unwrap()
    {
        match chunk.unwrap() {
            AudioChunk::Pcm(pcm) => {
                pcm.duration_ms().unwrap();
                assert_eq!((pcm.sample_rate, pcm.channels), (48000, 2));
                bits.extend(pcm.samples.into_iter().map(f32::to_bits));
            }
            AudioChunk::End => {
                ended = true;
                break;
            }
        }
    }
    assert!(ended && !bits.is_empty());
    assert_eq!(report.await.unwrap().end, GenerationEnd::Eos);
    bits
}

#[tokio::test]
#[ignore = "explicit CUDA model test; set TALECHIME_NANO_ONNX_ROOT and compatible ORT runtime"]
async fn real_cuda_queue_cancel_voice_switch_and_failure_reuse() {
    let directory = std::env::var_os("TALECHIME_NANO_ONNX_ROOT")
        .expect("explicit local ONNX model root required");
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = MossBackend::load_on(directory.into(), talechime_protocol::Device::Cuda)
                .await
                .unwrap();
            let expected = collect(&backend, "Weiguo").await;
            let (cancelled, report) = backend
                .stream_diagnosed("你好，欢迎使用叙铃。", "Weiguo", Some(42))
                .await
                .unwrap();
            drop(cancelled);
            assert_eq!(report.await.unwrap().end, GenerationEnd::Cancelled);
            assert_eq!(collect(&backend, "Weiguo").await, expected);

            let (mut stream, report) = backend
                .stream_diagnosed("你好，欢迎使用叙铃。", "Weiguo", Some(42))
                .await
                .unwrap();
            assert!(matches!(
                stream.recv().await.unwrap().unwrap(),
                AudioChunk::Pcm(_)
            ));
            // Observe a full capacity-one queue, without claiming the send is blocked.
            tokio::time::timeout(Duration::from_secs(30), async {
                while stream.len() != 1 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(stream.max_capacity(), 1);
            drop(stream);
            assert_eq!(report.await.unwrap().end, GenerationEnd::Cancelled);
            assert_eq!(collect(&backend, "Weiguo").await, expected);

            let (mut failed, report) = backend
                .stream_diagnosed("你好。", "missing-voice", Some(42))
                .await
                .unwrap();
            assert!(failed.recv().await.unwrap().is_err());
            assert_eq!(report.await.unwrap().end, GenerationEnd::InferenceFailure);
            drop(failed);
            assert_eq!(collect(&backend, "Weiguo").await, expected);
            let other = collect(&backend, "Adam").await;
            assert_ne!(other, expected);
            assert_eq!(collect(&backend, "Weiguo").await, expected);
            println!(
                "CUDA ONNX state reuse: {} identical Weiguo F32 bits",
                expected.len()
            );
        })
        .await;
}
