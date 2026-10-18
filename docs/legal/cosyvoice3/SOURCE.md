# CosyVoice3 第三方来源声明

`crates/cosyvoice3` 的实现以 Fun-CosyVoice3 官方 PyTorch 实现为行为规范，模块结构参考社区
Rust 移植，全部代码在本仓库内按 Candle 0.11.0 重写并修复审计偏差
（见 docs/records/cosyvoice3/cosyvoice3-candle-audit-2026-10-17.md）。

| 来源 | 固定版本 | 许可 | 使用方式 |
| --- | --- | --- | --- |
| FunAudioLLM/CosyVoice | 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc | Apache-2.0 | 行为规范（Python 对照）；本地参考克隆不分发 |
| SpenserCai/cosyvoice3.rs | 0da2fdcc61c9e47e2239e1cd977e00f47cab924c | MIT OR Apache-2.0（本仓库取 Apache-2.0） | 模块划分与权重转换脚本参考，代码重写 |
| SpenserCai/candle 分支 cosyvoice_support | 6058747 | Apache-2.0 OR MIT | 其 models/cosyvoice 模块为移植底稿，代码重写 |
| shivammehta25/Matcha-TTS | main（子模块指针） | MIT | feat_extractor（mel）数学规范 |
| lucidrains/x-transformers | 2.11.24 | MIT | DiT RoPE 数学规范 |

权重转换脚本（tools/tts/cosyvoice3_convert_weights.py）派生自 SpenserCai/cosyvoice3.rs 的
scripts/convert_weights.py（Apache-2.0）。模型本体（llm.pt/flow.pt/hift.pt/*.onnx）不进入
Git，来源 FunAudioLLM/Fun-CosyVoice3-0.5B-2512，许可以模型发布页为准。
