# Qwen3-TTS 推理与验证

模型资源统一位于 `qwen/models/<model-id>/<revision>/`，默认模型为 `0.6b-customvoice`。目录由后端的模型身份解析，资源按固定清单准备与校验。

Qwen 的 CustomVoice、Base ICL 和 VoiceDesign 是不同模型路径。当前语言参数未指定或为 `auto` 时使用模型的自动语言前缀；CustomVoice 的 Eric 和 Dylan 在 `auto` 或中文条件下分别使用四川和北京方言前缀，显式其他语言保持其语言条件。VoiceStudio 共享设计入口默认使用 Auto；模型专用 `qwen::design::reference` 可通过 `Options.language` 显式选择语言。语言条件可传递不表示所有语言、音色或设备已完成听验。

实现来源分开记录：`crates/qwen3-tts/SOURCE.md` 的 `711ceee07cad92673f86de8997bdf54c30caa49f` 是 TrevorS 社区 Rust 移植；独立 Python 对照使用 QwenLM/Qwen3-TTS 官方 `022e286b98fbec7e1e916cb940cdf532cd9f488e`，`modeling_qwen3_tts.py` SHA-256 为 `25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0`。不将社区版本称为官方。

普通测试使用官方 Python 方法生成的合成权重 fixture，检查 CustomVoice、Design 和 Base ICL 的输入结构、语言前缀及不同参考长度；这些测试不下载模型。工具 `tools/tts/qwen_prefill_oracle.py` 接受经 SHA 校验的官方源码，并可通过 `--model` 使用本地固定 Base 权重生成独立 CPU 数值参考。被忽略的 `full_voice_clone_prefill_matches_real_official_inputs` 测试使用 `QWEN_PREFILL_MODEL` 和 `QWEN_PREFILL_ORACLE` 的绝对路径，执行完整 `voice_clone_prefill`，比较实际模型输入、尾部文本及 offset。真实权重、设备、官方对照和人工听验须分别记录；不能以合成 fixture 或单个变体的结果代替其他变体的实机门禁。

Base 的不同参考语言、CustomVoice 与 VoiceDesign 各自的真实模型和多语言听验仍需分别核验。完整前缀输入数值对照不等于 transformer 输出、生成内容、音频质量或 CUDA/Metal 验收。

2026-11-11 本机 CPU 实测：官方 Python 固定源码与 `Qwen/Qwen3-TTS-12Hz-1.7B-Base@fd4b254389122332181a7c3db7f27e918eec64e3`，使用合成 speaker embedding、参考 codec 和 token，直接执行 Rust `voice_clone_prefill`；Auto/2 帧、日语/8 帧、中文/5 帧三例的模型输入最大绝对误差 `2.03e-6`，尾部文本最大误差 `1.08e-6`，offset 均一致。此结果仅证明上述 CPU 输入条件，不含真实参考音频和 transformer 隐藏状态数值对照。


同日 Metal 补充：CustomVoice 0.6B / 1.7B 和 Base 1.7B 各 19/19 公开文本生成，
Base 0.6B 为 18/19（全角数字 Auto 截断）。两个 CustomVoice 规模的 Eric/Dylan
显式中文探针共 4/4 生成。Base 1.7B 中文参考三项生成；Base 0.6B 中文参考混读和
英文生成，日文 Auto 与全角数字显式中文仍截断。VoiceDesign 已完成本机文本和显式语言探针；完整人工听验
仍待完成，不能把已有音频当作年份产品读法已修复。
详见[逐变体验收](records/tts-core/official-backend-parity-2026-11-11.md#qwen-逐变体与参考语言)。


2026-11-12 补充：Base 0.6B 使用原英文参考并显式 `language=zh` 后，全角数字样本
生成成功；VoiceDesign 韩语 Auto 截断后，以 `language=ko` 生成成功。报告保留两次
原失败，不把后续显式条件当作默认 Auto 已改善。VoiceDesign 其余 17 项生成，
超出设计 100 字符限制的长段明确拒绝；中文/英文/日文三个显式语言设计样本均生成。
官方结构 fixture 已扩为原 11 条加 19×11 条，共 220 条；合成 token/embedding
边界仍成立，不等于所有 checkpoint 数值、语音或听感已通过。
