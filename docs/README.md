# Talechime 文档

当前使用与架构以本目录的指南和当前源码为准。`records/` 保存特定日期、版本和设备的历史验收与实验结论，不代表所有平台或模型均已通过验收。

## 使用与开发

| 文档 | 内容 |
| --- | --- |
| [项目 README](../README.md) | 安装、模型能力、CLI 与宿主集成 |
| [Rust 库入口](library.md) | 模型准备、直接 PCM、可选播放/检查点、控制与关闭 |
| [ASR 回读校验](readback.md) | 可选模型组、独立报告、逐片段门禁、重试与 CLI |
| [配置与手工搬迁](migration.md) | 数据归属、新目录与旧资源复用 |
| [架构与集成](architecture.md) | 协议、所有权、故障边界和多音色演进 |
| [开发与验证](development.md) | 工具链、CPU/GPU 验证及独立发行 |
| [许可证与来源](licenses.md) | 第三方许可、原生库声明与制品归档 |
| [品牌资产](brand.md) | 标志、封面和使用方式 |
| [迁移来源](../SOURCE.md) | TRNovel 拆分与固定来源记录 |

## 开发知识

- [工具链与平台](knowledge/toolchain.md)：原生依赖、设备探测、构建、平台限制和实验记录。
- [会话与宿主集成](knowledge/booksource.md)：从 TRNovel 提取的听书知识；原阅读器专属条目为历史背景。

## 模型与平台验收

- [生成参数与 seed 的协议及后端缝](records/generation-params-seam-2026-10-10.md)：ParameterSpec 目录、PlanRequest.params/seed、Backend 单请求入口与生产者 seed 策略。

- [四后端生成参数与 seed 接入](records/params-exposure-2026-10-10.md)：参数目录、类型化构造器、OmniVoice 原生语速自动路由与 VoxCPM 真模型复现验证。

- [其他模型接续](records/continuation-all-models-2026-10-10.md)：Qwen 1.7B Base、VoxCPM2、MOSS Local / Realtime 的格式与验证边界。

- [Nano Candle 验证与默认切换](records/moss-nano-candle-trial-2026-10-10.md)：原实验移植、官方实现对照、数值验证与试听确认；现为默认 Nano 实现。

- [三个小模型段落内接续](records/continuation-small-models-2026-10-10.md)：实现、固定种子开／关对照、decoder 数值验证及尚未完成的主观试听。

- 通用：[模型分层](records/tts-model-tiers-acceptance.md)、[Metal 效率](records/metal-tts-efficiency.md)、[ORT rc.13 升级](records/ort-rc13-upgrade.md)。
- MOSS：[Nano](records/moss-tts-acceptance.md)、[Candle](records/moss-candle-acceptance.md)、[macOS 1.7B 流式输出](records/moss-macos-streaming.md)、[连贯性与对齐](records/moss-continuity-acceptance.md)、[DirectML 实验](records/moss-directml-evaluation.md)。
- Qwen：[Candle 集成](records/qwen-tts-acceptance.md)。
- VoxCPM：[Candle 移植](records/voxcpm-candle-acceptance.md)、[macOS 性能](records/voxcpm-macos-performance.md)。
- ASR：[主线原生验收](records/asr-readback-mainline-2026-10-09.md)、[《蛊真人》验收（已中断）](records/guzhenren-readback-acceptance-2026-10-09.md)、[回读校验选型 spike](records/asr-spike-2026-10-09.md)：Qwen、SenseVoice、FireRed、Whisper 候选及当时未接入产品的边界；当前实现见主线记录。

## 播放、解耦与历史设计

- [连续播放与对齐](records/continuous-tts-acceptance.md)、[自适应缓冲](records/tts-buffering-acceptance.md)、[窗口跟随](records/tts-follow-acceptance.md)。
- [进程解耦验收](records/tts-acceptance.md)、[解耦审查](records/tts-review.md)。
- [最初多后端计划](records/tts-backend-plan.md)、[旧 Kokoro 基线](records/tts-baseline.md)：保留演进背景，退休后端不再是当前运行入口。
