# Official Kokoro native inference

Acoustic reference: https://github.com/hexgrad/kokoro
Revision: `dfb907a02bba8152ca444717ca5d78747ccb4bec` (0.9.4).
Files: `kokoro/model.py`, `modules.py`, `istftnet.py`.
License: Apache-2.0, copied as `LICENSE`.

Talechime reimplements the official tensor layouts, inference operations and
single-segment behavior in Rust/Candle. It folds weight normalization during
development-only export, supports explicit excitation fixtures and bounds
duration expansion. No third-party Rust Kokoro implementation is used as a
source or correctness reference. Upstream derives neural modules from
https://github.com/yl4579/StyleTTS2 (see upstream source comments).

Chinese frontend reference: https://github.com/hexgrad/misaki
Revision: `fba1236595f2d2bf21d414ba6e57d25256afada3` (0.9.4), Apache-2.0.
Its `zh_frontend.py` and `tone_sandhi.py` adapt PaddleSpeech:
Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. Apache-2.0.
Frontend implementation/data acceptance is tracked separately from acoustics.

Official Chinese weight/config/voice source:
https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
Revision: `01e7505bd6a7a2ac4975463114c3a7650a9f7218`, Apache-2.0.
`kokoro-v1_1-zh.pth` SHA-256:
`b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b`.

Weights, generated audio, full real-model fixtures and external checkouts are
not distributed in this crate. Python is used only by development export tools;
native runtime inference does not invoke it. GPU features do not imply device
acceptance; see dated records under `docs/records/kokoro/`.

CPU moment reduction additionally follows PyTorch 2.7.1
`aten/src/ATen/native/cpu/moments_utils.h`, `layer_norm_kernel.cpp` and
`FlashAttentionKernel.cpp`:
https://github.com/pytorch/pytorch/tree/v2.7.1/aten/src/ATen/native/cpu
The BSD-style license is preserved as `licenses/PYTORCH-LICENSE`.

Optional contextual pronunciation uses the author's g2pW repository:
https://github.com/GitYCC/g2pW
Revision `b9328ce8129c4717a70f16a2688542a0197541dd`, Apache-2.0.
Author model: https://huggingface.co/YC-Chen/g2pW/tree/fb4784f7dce8466a53dd4f3552f5a4c5b657b90c
`G2PWModel-v2-onnx.zip` SHA-256:
`699f3c1fd7fb0e2c2d49ed2486826fd5bff233fee7759350a91c3b49aedc4ed2`.
Extracted `g2pw.onnx` SHA-256:
`bb40c8c7b5baa755b2acd317c6bc5a65e4af7b80c40a569247fbd76989299999`.
Tokenizer vocabulary: https://huggingface.co/google-bert/bert-base-chinese/tree/8f23c25b06e129b6c986331a13d8d025a92cf0ea
(Apache-2.0), SHA-256:
`45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c`.
Context exports retain the official g2pW license; the pinned BERT vocabulary is
used without separate BERT weights. g2pW predictions precede existing Misaki
sandhi; they do not silently alter the faithful `1.1-zh` baseline.
