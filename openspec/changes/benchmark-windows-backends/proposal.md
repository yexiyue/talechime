# Proposal

## Why

当前 Windows 性能数据分散在不同源码版本、模型变体和探针中，不能形成可复用的完整 CPU/CUDA 基线；部分权重仍位于 target 或旧目录。需要统一保留资源、完成当前主线所有 TTS 后端变体的真实测量，并交付可离线查看图表和试听音频的报告，作为 issue #5 后续优化与听感验证的依据。

## What Changes

- 建立覆盖六个当前 TTS 后端、所有公开模型变体及 Windows CPU/CUDA 适用路径的显式测试矩阵；实验模型、音色设计辅助模型、失败及不适用项分别登记。
- 将已有权重、转换资源与所需音色复制并校验到 `~/.talechime`，保留原件与现有数据；允许补下载缺失的固定 revision 权重。性能批次、日志和音频也在该目录长期保存，避免依赖 target。
- 提供开发侧资源盘点、串行基准、结果汇总与恢复执行工具；复用生产适配器，补齐必要测量入口，保持推理语义与产品 API 不变。
- 预编译 release 后测量短句、多段语料、实际支持的完整/流式/接续模式，分别报告模型加载、参考准备、热合成、首 PCM、RTF、内存及终止状态。
- 输出原始 JSON/CSV、图表、可离线打开的 HTML 试听报告及可打包目录；仓库报告保留结果、方法和限制，不提交模型、参考音频或生成音频。
- 当前只测 Windows 本机 CPU 与 NVIDIA CUDA；不开展模型优化、官方性能对照、Mac 测量或长期实际播放验收。

## Capabilities

### New Capabilities

- `benchmark-resources`: 开发基准使用的模型、转换资源和音色的身份校验、持久保存、冲突保护与复用。
- `backend-benchmarking`: 明确设备与模型覆盖范围的可恢复串行测量、原始结果、统计图表和离线试听报告。

### Modified Capabilities

无；当前 OpenSpec 主规格为空。现有合成、播放、设备门控及资源校验要求保持不变。

## Impact

涉及 `xtask/`、`crates/talechime-backends/examples/`、必要的模型探针入口及对应普通测试、`docs/development.md`、`docs/knowledge/tts-performance.md` 和新的 Windows 批次记录。默认资源根沿用 `talechime-core/src/paths.rs`，不新增产品路径迁移机制；本地生成物位于 `~/.talechime/benchmarks/windows/<run-id>/`，资源位于 `~/.talechime/resources/`。

本机盘点发现：Qwen Base06 在 target；Vox BF16 在旧 `.novel-tts`；IndexTTS 在相邻 index-tts 仓库；CosyVoice3 HiFT 与当前生产清单存在摘要冲突。必须明确处理这些门禁，不能放宽清单或把独立计算库测量当作生产适配器通过。关联：https://github.com/yexiyue/talechime/issues/5 。
