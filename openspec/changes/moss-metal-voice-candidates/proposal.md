# Proposal

## Why

单音色长篇需要可试听、可固定复用的旁白资产，而不是把无参考的 narrator 当成稳定人物。Mac 当前不能使用 moss-voicedesign；用户希望补上 Metal 能力，并准备较多候选集中试听。

## What Changes

- 为 moss-voicedesign 增加 macOS Metal 设备路径，沿用现有 feature、权重与 codec；先以真实模型验证内存、完成与取消，再对外声明支持。
- 保留 CUDA/BF16；Metal 使用现有 F16 策略，不静默回退 CPU/CUDA，不修改原生依赖版本。
- 通过现有 VoiceStudio 与音色库制作 32 个基础候选：8 画像 × Qwen/VoxCPM × 2 seed；MOSS Metal 门禁通过后追加 16 个，批次上限 48 个。
- 每项交付原始参考、准确转写、来源与参数、固定参考克隆试听、试听索引和待填写反馈。所有候选保持待人工验收，不自动替换默认音色。
- 支持串行、有界、可恢复的批次操作；单项失败留痕，MOSS 阻塞不阻止其余引擎完成。

## Capabilities

### New Capabilities

- `voice-design`：补充 MOSS Metal 的设备、资源、取消与实机验收要求。主 specs 当前为空，沿用已完成但未归档 voice-studio 变更中的同名能力路径，仅新增本次要求。
- `voice-candidate-review`：定义可追溯的候选批次、固定参考试听、恢复与人工选择交付。

### Modified Capabilities

无已归档主能力需要修改；不改 voice-studio 的既有音色库格式或设计命令语义。

## Impact

主要涉及 backends 的 design/moss.rs、moss/candle/design.rs 与现有 device/dtype/codec 路径；仅在实机证据发现必要缺口时局部调整计算实现。批次编排优先扩展 scripts/voice_studio_acceptance.py 或同目录专用操作工具，复用现有 Rust 库/CLI，不新增服务或通用任务平台。

音频、批次私有清单与反馈放在忽略的 target/voice-candidates/<batch-id>/；库资产沿用当前 VoiceLibrary 路径。公开代码只保留可分发画像、文本和操作说明，模型、生成音频不入 Git。与 Talechime #6 关联；不引入 CastGlean、线上 LLM、滚动/双参考或默认音色变更。
