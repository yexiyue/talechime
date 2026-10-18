# Candle 第三方来源声明

工作区的 `candle-core` / `candle-nn` / `candle-transformers` / `candle-metal-kernels` 通过根
`Cargo.toml` 的 `[patch.crates-io]` 指向 Talechime 的 Candle fork。

| 来源 | 固定版本 | 许可 | 使用方式 |
| --- | --- | --- | --- |
| huggingface/candle | `31f35b147389700ed2a178ee66a91c3cc25cc80d`（上游 tag `0.11.0`，"Bump candle version to 0.11.0 (#3658)"） | Apache-2.0 OR MIT | 全工作区统一的 Candle 计算运行时 |
| yexiyue/candle（fork） | 分支 `talechime-perf`，基线同上 | Apache-2.0 OR MIT（与上游一致，未改许可） | 承载可上游化的性能补丁 |

工作区固定 fork revision 为 `bafb03db81c3eae48e63ff28d6b30f5a6576ba38`。
`candle-metal-kernels` 也必须 patch：否则平台探测使用 registry 包，而 fork 的
core/nn 使用 git 子 crate，形成同版本双来源。CUDA kernels 与 candle-ug 已由
fork 内部路径依赖解析到同一 revision，不另写重复 patch。

许可全文见本目录 `LICENSE-APACHE` 与 `LICENSE-MIT`，副本同时随
`crates/voxcpm` 分发（`LICENSE.candle-APACHE` / `LICENSE.candle-MIT`），
以便 cargo-dist 的 `prepare-notices.py` 自动收集。

## Fork 策略

- fork 保持上游目录结构与 crate 版本号 `0.11.0` 不变，因此工作区的
  `=0.11.0` 依赖约束照常成立。
- fork 内的补丁必须是**行为级、可上游**的改动：不新增/删除/改变公开 API。
  原因：`cargo publish` 会剥离 `[patch]`，本仓库发布的 candle 消费库
  （`voxcpm` / `indextts` / `cosyvoice3` / `moss-tts` / `qwen3-tts` /
  `omnivoice` / `talechime-backends`）在下游会解析 registry 版本；若代码
  依赖 fork 独有 API，发布版本将无法编译。
- 每个补丁在 fork 的 `FORK.md` 中登记：上游 issue/PR 链接、改动摘要。
- 若将来确实需要新增 candle API（判据：图级 KV 预分配写入等无法用现有公开
  API 表达的能力），则把受影响的库标记 `publish = false`，直到该补丁被上游
  合并后恢复。

## 发布时移除 patch

`cargo publish --dry-run --locked` 若因 patch 解析失败，发布流程在干净工作树
中临时注释根 `Cargo.toml` 的 `[patch.crates-io]` 区块后重试，发布完成再恢复。
该步骤只影响本地打包，不修改 `publish-libraries.py` 的逻辑。

2026-10-18 实测：`cargo publish --dry-run --locked --allow-dirty -p voxcpm`
在**保留** `[patch.crates-io]` 的情况下直接通过（打包时 patch 被忽略，依赖按
registry 版本解析），因此该回退步骤目前未被触发，保留作为兜底。

## fork 是否真的被用到

截至 2026-10-18，fork 相对上游**没有任何功能性改动**（仅新增 `FORK.md`）。
每帧 CFM 的 CUDA Graph 只用 candle-core 0.11 / cudarc 0.19 的公开 API 实现，
未新增也未修改 candle 的 API（见
`docs/records/voxcpm/voxcpm-performance-comparison-2026-10-18.md` §6.4）。
因此轨道 B（`publish = false`）尚未触发；`[patch.crates-io]` 目前只是把 candle
来源固定到可审计的 fork rev 上，改变包的来源身份，不改变上游运行时实现。

## 退出条件

当 `FORK.md` 中登记的补丁全部被上游合并或放弃后，删除 `[patch.crates-io]`、
刷新 `Cargo.lock`，并归档或删除 fork 仓库。
