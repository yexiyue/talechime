# Talechime 文档

当前使用与架构以本目录的指南和当前源码为准。`records/` 按主题分子目录保存特定日期、版本和设备的历史验收与实验结论，不代表所有平台或模型均已通过验收；归档规则见 [records/README.md](records/README.md)。

## 使用与开发

| 文档 | 内容 |
| --- | --- |
| [项目 README](../README.md) | 安装、模型能力、CLI 与宿主集成 |
| [Rust 库入口](library.md) | 模型准备、直接 PCM、可选播放/检查点、控制与关闭 |
| [ASR 回读校验](readback.md) | 可选模型组、独立报告、逐片段门禁、重试与 CLI |
| [配置与手工搬迁](migration.md) | 数据归属、新目录与旧资源复用 |
| [架构与集成](architecture.md) | 协议、所有权、故障边界和多音色演进 |
| [开发与验证](development.md) | 工具链、CPU/GPU 验证及独立发行 |
| [许可证与来源](licenses.md) | 第三方许可、原生库声明与制品归档（许可原文在 [legal/](legal/)） |
| [品牌资产](brand.md) | 标志、封面和使用方式 |
| [迁移来源](../SOURCE.md) | TRNovel 拆分与固定来源记录 |

## 开发知识

- [工具链与平台](knowledge/toolchain.md)：原生依赖、设备探测、构建、平台限制和实验记录。
- [TTS 性能优化方法](knowledge/tts-performance.md)：跨后端的测量、准确性门禁、模块边界与优化取舍。
- [会话与宿主集成](knowledge/booksource.md)：从 TRNovel 提取的听书知识；原阅读器专属条目为历史背景。
- [CosyVoice3 后端](cosyvoice3.md)：实验性零样本克隆、原生流式、固定本地转换文件与音色导入。
- [IndexTTS 2.5 后端](indextts.md)：实验性参考克隆、源码 feature、本地转换文件与设备边界。

## 历史记录（records/）

### 实施计划（plans/）

尚未实现或已完成的规划文档，保留演进背景。

- [独立生成与长文本表达：Candle 实施计划](records/plans/independent-tts-candle-plan.md)：候选模型可行性、试听筛选、分阶段移植、逐段表演计划与验收条件。
- [Kokoro G2P 前端替换计划](records/plans/kokoro-g2p-frontend-plan-2026-10-14.md)：以统一前端替换规则 G2P 的路线与阶段划分。
- [最初多后端计划](records/plans/tts-backend-plan.md)：退休后端不再是当前运行入口。

### TTS 核心（tts-core/）

与具体后端无关的引擎能力验收。

- [生成参数与 seed 的协议及后端缝](records/tts-core/generation-params-seam-2026-10-10.md)：ParameterSpec 目录、PlanRequest.params/seed、Backend 单请求入口与生产者 seed 策略。
- [四后端生成参数与 seed 接入](records/tts-core/params-exposure-2026-10-10.md)：参数目录、类型化构造器、OmniVoice 原生语速自动路由与 VoxCPM 真模型复现验证。
- [逐段风格的开放与组合边界](records/tts-core/style-exposure-2026-10-10.md)：VoxCPM/OmniVoice/MOSS style 接线、MOSS language 槽位与 VoxCPM style×接续的显式拒绝。
- [模型分层](records/tts-core/tts-model-tiers-acceptance.md)
- [连续播放与对齐](records/tts-core/continuous-tts-acceptance.md)、[自适应缓冲](records/tts-core/tts-buffering-acceptance.md)、[窗口跟随](records/tts-core/tts-follow-acceptance.md)
- [进程解耦验收](records/tts-core/tts-acceptance.md)、[解耦审查](records/tts-core/tts-review.md)
- [旧 Kokoro 基线](records/tts-core/tts-baseline.md)

### 后端移植与验收

- CosyVoice3：[复刻调研与社区审计](records/cosyvoice3/cosyvoice3-candle-audit-2026-10-17.md)（官方推理链路、ONNX 边界与 `crates/cosyvoice3` 移植决定）、[本机对照与修复](records/cosyvoice3/cosyvoice3-local-parity-2026-10-17.md)、[后端接入验收](records/cosyvoice3/cosyvoice3-backend-2026-10-17.md)、[后续实现审查](records/cosyvoice3/cosyvoice3-review-2026-10-17.md)、[性能与 Candle 优化](records/cosyvoice3/cosyvoice3-performance-2026-10-17.md)（Metal/CPU 热路径、Accelerate 与基准工具）。
- IndexTTS 2.5：[官方 / 社区对照与修复](records/indextts/indextts-official-community-review-2026-10-17.md)、[性能优化与计时修复](records/indextts/indextts-candle-performance-2026-10-17.md)（三轮：RTF 0.70→0.47 ABBA 均值；固定语料验证与人工试听边界分开记录）、[历史验收台账](records/indextts/indextts-candle-acceptance-2026-10-17.md)、[移植过程](records/indextts/indextts-candle-port.md)。
- MOSS：[Nano Candle 验证与默认切换](records/moss/moss-nano-candle-trial-2026-10-10.md)、[模型验收](records/moss/moss-tts-acceptance.md)、[Candle 验收](records/moss/moss-candle-acceptance.md)、[macOS 1.7B 流式输出](records/moss/moss-macos-streaming.md)、[连贯性与对齐](records/moss/moss-continuity-acceptance.md)、[DirectML 实验](records/moss/moss-directml-evaluation.md)。
- Qwen：[Candle 集成与官方对照](records/qwen/qwen-tts-acceptance.md)、[性能审查与准确性验证](records/qwen/qwen-performance-review-2026-10-18.md)、[Metal 性能优化](records/qwen/qwen-performance-optimization-2026-10-18.md)。
- VoxCPM：[Candle 移植](records/voxcpm/voxcpm-candle-acceptance.md)、[macOS 性能](records/voxcpm/voxcpm-macos-performance.md)、[CUDA 性能对照](records/voxcpm/voxcpm-performance-comparison-2026-10-18.md)、[优化审查与扩展](records/voxcpm/voxcpm-review-extension-2026-10-18.md)。

### ASR 回读（asr/）

- [主线原生验收](records/asr/asr-readback-mainline-2026-10-09.md)：当前实现基准。
- [ASR 语言自动检测与音频标签捕获](records/asr/asr-language-labels-2026-10-10.md)：Qwen3-ASR auto 语言、SenseVoice 情感/事件标签进报告。
- [回读校验选型 spike](records/asr/asr-spike-2026-10-09.md)：Qwen、SenseVoice、FireRed、Whisper 候选及当时未接入产品的边界。
- [《蛊真人》验收（已中断）](records/asr/guzhenren-readback-acceptance-2026-10-09.md)。

### 跨段接续（continuation/）

- [其他模型接续](records/continuation/continuation-all-models-2026-10-10.md)：Qwen 1.7B Base、VoxCPM2、MOSS Local / Realtime 的格式与验证边界。
- [接续与后续改动审查](records/continuation/continuation-review-2026-10-10.md)：八项修复、两处简化及本轮测试结果。
- [三个小模型段落内接续](records/continuation/continuation-small-models-2026-10-10.md)：实现、固定种子开／关对照、decoder 数值验证及尚未完成的主观试听。

### 音色漂移（voice-drift/）

- [漂移实验](records/voice-drift/voice-drift-experiment-2026-10-14.md)、[Qwen 与 MOSS 对照](records/voice-drift/voice-drift-qwen-moss-2026-10-14.md)。

### 平台与依赖（platform/）

- [Metal TTS 效率](records/platform/metal-tts-efficiency.md)、[ORT rc.13 升级](records/platform/ort-rc13-upgrade.md)。
