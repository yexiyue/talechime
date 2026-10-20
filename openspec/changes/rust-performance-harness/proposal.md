# Proposal

## Why

现有后端性能探针和 Python 编排缺少统一的测量契约、诊断上下文及持续回归入口。建立 Rust 主导的开发性能工具，让 Windows 基线和后续 macOS CPU/Metal 实测复用同一套执行、证据与比较协议。

## What Changes

- 以 `cargo xtask perf` 提供计划、资源检查、构建、串行执行、恢复、比较和离线报告入口；测量与报告链路不依赖 Python。
- 采用轻量 xtask 调度明确 feature 的生产适配器探针子进程，隔离 CPU/CUDA/Metal 构建及模型生命周期。
- 定义跨后端阶段、首 PCM、背压、资源采样和终态指标；提供 tracing 关联及 metrics 聚合，原始逐轮结果保持为权威证据。
- 分离 baseline、trace、profile 运行；保留详细采集开销、采集完整性和原生 profiler 产物，防止诊断数字混入正式基线。
- 提供无模型微基准和固定硬件真实模型回归分层、可比性检查及离线 JSON/CSV/HTML/SVG/WAV 交付。
- 从首版 schema 与命令设计支持平台分区，预留 macOS CPU/Accelerate、Metal、统一内存及 Instruments 诊断；Mac 原生实现和实机验收后续另行补齐，未验证能力显式标记 deferred。
- Python 仅允许作为已有上游模型转换的显式准备依赖；不重写所有历史对照工具，不以框架建设触发全量模型下载或测量。

## Capabilities

### New Capabilities

- `performance-execution`: Rust 开发入口、平台计划、进程隔离、冻结身份、恢复和离线证据交付。
- `performance-observability`: 测量边界、请求关联、阶段诊断、聚合指标和采集开销隔离。
- `performance-regression`: 可比性门禁、分层微基准与真实模型回归判定。

### Modified Capabilities

无。主规格当前为空；进行中的 `benchmark-windows-backends` 定义 Windows 覆盖及资源行为，不将其误列为已存在的主规格。

## Impact

规划涉及未来的 `xtask/`、Cargo alias/workspace、开发探针、必要埋点及文档/CI。xtask 不发布、不进入发行安装包，不增加产品命令或改变 JSON Lines v7；推理与音频对象继续由原 owner 持有。资源根沿用 AppPaths，不建立第二套迁移规则。

与 `benchmark-windows-backends` 的关系：本 change 提供共用基础设施，前者消费框架完成 Windows 全量批次；前者的 Python 编排/报告实现任务在实施前需要协调调整。本次不改写该 change，不重复定义其 28 格矩阵、资源导入规则或全量执行任务。macOS 原生采集与真实模型证据不作为本 change 完成条件，但平台契约、缺失状态、扩展位置和后续验收清单必须交付。
