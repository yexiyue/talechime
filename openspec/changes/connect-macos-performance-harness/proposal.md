# Proposal

## Why

Rust 性能框架已交付共用探针、证据、统计和报告，但 macOS 的 runner、executor、进程生命周期及原生采集仍被平台门禁阻止。先完成 Mac 工具接通，才能让后续独立基线任务复用可信执行链路。

## What Changes

- 为 macOS 接入有界串行执行、进程组隔离、超时、CLI 取消、子孙进程回收和恢复，复用现有 Rust runner 与生产探针。
- 接入原生 RSS/可用 OS 高水位、机器/系统身份、CPU math 与线程条件记录；不可获取的指标保留 null 和原因。
- 明确 Metal host PCM 完成边界与统一内存含义；按能力实证解除 deferred，不把工具接通标成模型性能已验证。
- 验证 plan → build → run → resume → compare/report/pack 的免模型闭环，并以已有已校验资源完成 CPU 与 Metal 短 smoke。
- 完整基线报告另立任务：本 change 不补齐模型、不跑全变体正式矩阵、不作性能排名、真实模型观测开销或听感验收；Instruments 保持可选 deferred。

## Capabilities

### New Capabilities

- `macos-performance-runtime`: Mac 性能工具的原生生命周期、采集、能力激活与接线验证契约。

### Modified Capabilities

无。正式 specs 库目前为空；复用未归档 `rust-performance-harness` change 中 performance-execution、performance-observability、performance-regression 的共用契约，本能力只补充 Mac 平台行为，不另建 runner 或重述共用协议。

## Impact

- 预期涉及 `xtask/src/perf/{process,executor,cancellation,runner,sampling,environment,plan,profiling}.rs`、Mac 平台模块、相关进程/CLI/入口测试与性能 CI。
- 必要时在现有 backends 性能探针中补充有效构建/math 条件和计时契约；保持 Registry、LocalSet、owner 和有界流所有权。
- 仅增加必要的 target-gated 原生依赖；不修改模型算术、v7 协议、产品 executable、发行配置或资源校验/迁移规则。Windows 行为保持原契约，其他未实现平台继续显式拒绝。
