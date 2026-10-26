# 架构重构与历史兼容清理验证（2026-10-26）

基线为 main `a7431bf`。本次修改未发布、打标签或部署；包版本与 JSON Lines v7 不变。当前指南见 `docs/architecture.md` 与 `docs/library.md`。

## 实现边界

- Registry 通过各后端私有登记模块生成确切的 ResolvedModel。资源准备一次完成，校准、回退及恢复只重建模型实例；CLI/Engine 共享 PrepareOptions 服务。
- core 不再依赖 rodio/reqwest。facade 的 playback feature 控制默认音频设备，cli 显式启用它；下载、续传锁和下载错误归 backends 私有资源层。
- 分段请求显式携带音色和执行内时长估计。音色切换、取消后重启和两个执行交错均不共享学习状态。
- 删除历史目录迁移、旧配置驱动会话入口和 Resources 转发别名。单音色构造封闭计划，旧会话测试的有效断言迁入计划路径。
- 不删除磁盘旧文件，不改模型数学、权重、原生依赖版本或 GPU 策略。当前不支持的数据版本仍明确报错，不自动转换或覆盖。

## 本机验证

环境为 Apple Silicon macOS；普通测试不下载模型。本地日志位于 `target/architecture-refactor-20261026/`，不进入 Git。

| 检查 | 结果 |
| --- | --- |
| 全 workspace 扩展 CPU 组合：lib/tests/examples | 提交前串行全量复核通过（--test-threads=1）；42 项显式 ignored，包括要求真实权重、原生探针或性能构建凭据的用例 |
| 全 workspace 扩展 CPU 组合：all-targets Clippy，-D warnings | 通过 |
| workspace private-items/examples rustdoc，-D warnings | 通过 |
| 无默认 feature 的 facade library 测试 | 17 项通过；注入播放器，无默认音频/HTTP 依赖 |
| 无 feature、playback、cli、moss、qwen、voxcpm、omnivoice、cosyvoice3、moss-candle、moss-nano-candle、asr、indextts | 12 组 all-targets Clippy 全部通过 |
| 七组模型 Metal feature 加 ASR：all-targets 编译 | 通过；不是实机推理或音频验收 |
| Python 发布/架构测试 | 15 项通过；涵盖实际发布顺序、feature 选择及依赖边界 |
| cargo metadata 与 headless facade 的 normal 依赖树 | core、protocol、headless facade 无 rodio/cpal/reqwest |
| 默认 CLI build 和无模型 v7 握手 smoke | 通过 |
| cargo-dist 0.32.0 generate、generate --check、plan | 通过；重新生成的工作流内容不变，发行 feature 显式包含 cli |
| 许可证清单收集 | 通过；按发行依赖图收集，保留原生来源声明 |
| cargo fmt --all --check、git diff --check | 通过 |

扩展 CPU feature 组合为 `talechime/qwen,talechime/voxcpm,talechime/omnivoice,talechime/cosyvoice3,talechime/moss-candle,talechime/asr,talechime/indextts,talechime-backends/performance-tools`。
Metal 编译组合为 facade 的 `qwen-metal,voxcpm-metal,omnivoice-metal,cosyvoice3-metal,moss-candle-metal,moss-nano-candle-metal,indextts-metal,asr`。

新增确定性回归覆盖：默认模型与显式模型身份、目录查询无写入、显式设备失败不回退、自动缓存加速器失败回 CPU、恢复开关、失败校准不发布缓存、资源只准备一次、A→B→A 估计重置、交错执行隔离、取消/失败后重启、显式分段请求稳定、下载取消/续传，以及旧目录忽略与不支持检查点字节保留。

历史目录回归现在检查旧文件未导入且字节保留，不复用旧迁移流程的五秒耗时断言。功能测试不承担启动性能验收。

## 提交前审查与简化

以 `a7431bf` 为基线，规范轴检查 AGENTS 与开发约束，方案轴检查已批准的重构要求；两轴独立静态审查均未报告阻塞问题。补充执行路径检查发现两项 P2，均已修复：性能探针会误判动态分段；检查点读取拒绝未知版本，但原保存路径仍可能覆盖未知版本或损坏文件。

确定性回归 `adaptive_segmentation_uses_the_execution_voice_and_duration` 使用非默认音色和依赖时长估计的分段后端，修复前明确报 `consumer ranges differ from production segmentation`，修复后通过。探针改为有界记录实际执行时后端返回的原文范围，再与消费端范围比较，不再额外调用分段接口；记录与消费共享 1024 段证据上限。

检查点读取与持锁保存现在复用格式校验。`saving_preserves_unsupported_and_malformed_checkpoints` 在修复前复现覆盖问题，修复后确认报错且文件字节不变；当前格式仍允许明确重启后按新正文保存。

同时将探针包装器统一为 Segmentation 类型别名，简化范围校验和条件分支，并复用 producer 已有音色身份判断，删除重复的音色状态及字符串分配。分段学习、暂停/seek、取消与背压契约保持原义。

提交前并行复核曾在未修改的 `xtask` 资源导入测试遇到一次文件锁竞争；该测试单独复测及完整串行套件均通过，未跳过该测试。最终工作区 Clippy、rustdoc、格式检查及 15 项 Python 检查也通过。

## 未覆盖的验收

未运行真实模型推理、数值/PCM 对照、试听或性能基准。未完成 Windows/Linux 原生构建或 CUDA 编译；这些由对应平台 CI 和独立模型验收负责。确定性准备测试不代表真实 provider 恢复验收，Metal 编译不代表设备音频验收。
