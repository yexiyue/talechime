# Readback models and frontend sources

Qwen3-ASR source attribution is in crates/qwen3-asr/SOURCE.md (MIT).
The official 0.6B HF model revision is pinned in qwen06.json. Its model card
README.md (Apache-2.0 metadata) is downloaded and verified alongside weights.

SenseVoiceSmall INT8 ONNX and tokens originate from
https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
at revision 2365baeacb507f821a0c8120fcee3d484dba7a07, pinned in sensevoice.json.
The downloaded LICENSE points to the independent FunASR model agreement;
SenseVoice source MIT does not relicense these model weights.

Frontend contract was checked against the sherpa-onnx Python spike
(see spike record for package versions and comparison). Runtime uses kaldi-fbank 0.1.0 (Apache-2.0), ORT rc.13 pinned by
workspace, and rubato 0.16.2 for resampling; no sherpa C++ library is linked.
The fixed fbank-golden.json was generated with kaldi-native-fbank 1.22.3 from
640 integer sawtooth samples ((i * 37) % 32768 - 16384), 16kHz/80 bins,
Hamming/dither=0/snip_edges=true/high_freq=0. It contains no user recording.

Weights and experimental recordings are not committed or bundled in releases.
