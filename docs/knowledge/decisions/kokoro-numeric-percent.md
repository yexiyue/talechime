# Kokoro 数字全角百分号支持

日期：2026-11-13。状态：已实现，相关自动化与本机 CPU 真实缓存模型样本通过，人工听感及其他平台未单独复验。

CastGlean 真实章节在数字后的全角百分号上中止。Kokoro 数字规则此前只接受半角 %；本次在数字后同时接受 % 和 ％，沿用全角数字转换、负数、小数与精度限制，不执行通用 Unicode 折叠。只规范化执行文本，不改原文、范围和权重。

孤立的 你好％ 仍明确拒绝；原官方拒绝夹具保留，并通过普通及 g2pW 两条真实资源路径复验。新数值用例覆盖 50％、１２.５％、负百分数及中英混排。普通测试不下载模型。

Kokoro 推理身份提升为 01e7505-runtime-f32-v3-text-v5-loqui-neural-0.4.0；固定资源版本仍为 01e7505-runtime-f32-v3。宿主旧任务与音频保留，但旧语义计划不能混合续跑，需显式创建新版本。

本机 CPU、1.1-zh-g2pw、zf_003 通过 native_probe 对公开句子生成 6.375 秒、24kHz 单声道 float PCM，全部有限、峰值约 0.485、流正常结束。产物位于 ignored target/percent-acceptance；这是有效 PCM 证据，不是人工听感结论。源码与测试见 crates/kokoro/src/frontend/normalize.rs、crates/talechime-backends/src/registry/kokoro.rs 及 openspec/changes/support-kokoro-numeric-fullwidth-percent。

该结论补充此前官方对齐台账：数字后的全角百分数现在支持，孤立符号拒绝边界继续保留；旧台账及独立夹具不改写。
