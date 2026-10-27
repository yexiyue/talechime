# Kokoro 后端发行接入与验收（2026-10-27）

## 需求与审查范围

来源：本任务用户要求“先接入到我们后端，然后审查和简化代码，提交推送发包”，
并明确选择应用安装包和 Rust 库同时发布。用户确认审查基点为 `45b82b0`。

- R1：生产 Kokoro Backend / Registry / Engine 使用原生模型及已验证优化；不得只接入实验探针。
- R2：标准 CPU 发行包包含 `1.1-zh` 和 `1.1-zh-g2pw`；macOS 额外编入 Metal 解码路径；模型资源仍显式转换与准备，不打包权重、不静默下载。
- R3：保持音频、取消/复用、模型所有权、当前数据格式与 JSON Lines v7 约束；不恢复滚动、双参考或旧 MOSS 实现。源码与官方对照和性能证据分别保存。
- R4：发布安装包和按依赖顺序发布 Rust 库；版本体现自上次发布以来的公共 API 移除。保持既有标签与已发布版本的身份，不混入工作区无关实验改动。
- R5：按 code-review 的 Standards / Spec 两轴并行审查，简化新增代码，并完成格式、测试、Clippy、rustdoc、打包与归档的必要验证。
- R6：发布包包含所需第三方许可；原生数值/性能验收不替代 Windows/Linux/CUDA 实机音频验证。

## 实现

Kokoro 计算库已由 `crates/talechime-backends/src/kokoro/runtime.rs` 生产路径调用，
Registry 与 facade 已提供两种模型及 feature；本次将 `kokoro-g2pw` 加入标准 CPU
配置，`kokoro-metal` 加入 macOS 配置，并将计算库加入库发布依赖顺序。
归档 smoke 会查询真实可执行文件的模型目录并检查两种模型及 PyTorch 许可副本。

应用/facade 为 0.6.1，core 为 0.6.0，backends/protocol 为 0.4.0，MOSS/VoxCPM 为 0.2.0，
Kokoro 首发 0.1.0；其余依赖图使用新的补丁版本。协议 wire version 仍为 v7。
发布身份检查要求根 Cargo.toml 改变后不复用旧包版本，故整个依赖图统一登记本次批次。

## 证据

- [官方移植与真模型验收](kokoro-native-validation-2026-10-27.md)
- [CPU 缓冲、归一化与向量优化](kokoro-cpu-buffer-vector-2026-10-27.md)

## Standards

双轴独立审查及增量复查后：0 项硬性违反，0 项剩余可选建议。已将重复的三输入
存储借用闭包提取为私有 helper，保持读锁顺序与作用域；归档 smoke 删除了入口
已拒绝的 TRNovel 不可达分支。协议目录测试按 backend ID 查找 MOSS，并覆盖两种
Kokoro 模型，避免依赖目录首项。

## Spec

0 项需求偏差。生产推理路径、标准包 feature、模型目录、显式资源边界与许可收集
均有直接检查；无关 VoxCPM 源码及叙述实验未进入发布快照。

## 本地验证

- 独立检出的扩展 CPU 工作区测试：877 项通过，44 项显式真模型等测试保持忽略。
- 全工作区 Clippy `-D warnings`、严格 rustdoc、纯库构建与格式检查通过。
- cli、cli+moss、cli+moss-nano-candle、cli+kokoro、cli+kokoro-g2pw 五种最小 CLI 组合均通过测试。
- Kokoro 在 registry Candle 0.11 下以 g2pw/metal 打包 dry-run 通过。
- 简化后的五组 CPU/Metal 官方数值门禁通过，十份完整 PCM 与优化基线逐字节相同；
  opt-in 两项真模型 SNR 为 43.1115 / 77.0372 dB，门槛仍为 40 / 60 dB。
- Mac 实际 0.6.0 归档、校验和、本地安装器、双 Kokoro 模型目录、许可、帮助与 v7
  无模型握手检查通过。Windows/Linux 归档由原生发布 CI 验证；不宣称其模型音频验收。
- 构建与日志保留在本地 `target/release-preparation/`。验证快照不包含无关工作区改动。

## 发布入口

应用由 `talechime-v0.6.1` 触发；facade 补丁由 `talechime-libs-v0.6.1` 触发，其余库复用已成功发布的 0.6.0 批次，分别运行原生
归档与依赖顺序发布流程。上传、平台 smoke 与发布结果以服务端记录为准：

- [0.6.1 安装包](https://github.com/yexiyue/talechime/releases/tag/talechime-v0.6.1)
- [库发布工作流](https://github.com/yexiyue/talechime/actions/workflows/publish-libraries.yml)
- [talechime 0.6.1](https://crates.io/crates/talechime/0.6.1)
- [talechime-kokoro 0.1.0](https://crates.io/crates/talechime-kokoro/0.1.0)

Standards：0 项；Spec：0 项；两轴均无剩余最高严重度问题。

## 0.6.0 远端失败与补丁恢复

Rust 库批次已全部成功发布。应用发布的最小 `cli` 测试在干净 CI 用户目录下失败：
非法 UTF-8/缺失文件用例没有提供配置，先触发“没有可用后端”。本地用户配置掩盖了问题。
测试现使用临时当前格式配置、HOME、模型与检查点目录，并验证配置不变且不创建资源目录。

cargo-dist 0.32.0 生成的 host 条件接受跳过的构建与 smoke，导致只有清单的空应用
Release 被公开。已将其转回草稿，保留原标签；不改写已发布 crate。应用/facade 补丁
改为 0.6.1，模型计算库版本保持不变。

`.github/scripts/generate-release.py` 先运行固定版本 cargo-dist，再按配置生成严格
host needs/if：CPU CI、性能契约、两级构建及归档 smoke 均须 success。生成器在 CPU CI
检查；回归测试覆盖每个前置作业的失败、跳过、取消及缺失作业/模板变动时拒绝生成。
安装包仅在这一门禁全部满足后才发布。

补丁本地验收：五种最小 CLI 组合全部通过，非法文件用例以临时 HOME/配置执行；
19 项 Python 回归、真实固定版本生成检查、dist plan、格式、Clippy `-D warnings`
和严格 rustdoc 均通过。两轴增量复查均无剩余发现。补丁不改变推理数值路径。
