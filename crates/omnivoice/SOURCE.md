# OmniVoice native inference

Source: https://github.com/FerrisMind/omnivoice-rs
Revision: 4c7b088294fd6f3a5fb9e3b1dfd3a77bc12f105b
Code license: Apache-2.0 (LICENSE).

Only the inference library and manifest assets are included. No CLI, server,
model downloader, private Candle fork, FlashAttention, Vulkan/WGPU or automatic
ASR dependency remains. All inference uses workspace Candle 0.11.0.
Reference transcripts are required; optional quantized embedding lookup uses
standard Candle dequantization. Debug/reference comparison helpers remain to
validate parity. Timing accumulation is disabled unless explicitly requested.

Weights: k2-fsa/OmniVoice c5fdb5ccb189668d56333f77ba2629f4cd7535f4.
The generator is Apache-2.0. The audio tokenizer has the separate BOSON HIGGS
AUDIO 2 COMMUNITY LICENSE AGREEMENT incorporating Llama 3 terms:
https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/audio_tokenizer/LICENSE
Do not describe the complete weight bundle as Apache-2.0.

## Windows CUDA ordered decoder specialization (2026-10-24)

The geometry-specialized kernel in `src/codec_layers/ordered_deconv.cu` follows
Candle's direct `conv_transpose1d` tap -> channel F32 accumulation order, with
explicit round-to-nearest FMA and specialized integer indexing. Reference:
`candle-kernels/src/conv.cu` at Candle fork revision
`2266099d92265112ecc0d56ecee1761e3cc2ac00` (Apache-2.0 OR MIT).
License copies accompany this crate as `LICENSE.candle-APACHE` and
`LICENSE.candle-MIT`; Talechime's adaptation is Apache-2.0.
CUDA builds compile PTX using the existing CUDA toolkit at build time. Runtime
inference loads that PTX through Candle; it does not compile using NVRTC.
CPU and Metal builds do not invoke the CUDA build step. Unvalidated geometry,
precision, layouts and devices below compute capability 8 retain Candle's path.
The original upstream import used Candle 0.9.2; Talechime uses workspace 0.11.0.
