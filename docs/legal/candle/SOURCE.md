# Candle 第三方来源声明

工作区的 `candle-core` / `candle-nn` / `candle-transformers` 通过根
`Cargo.toml` 的 `[patch.crates-io]` 指向 Talechime 的 Candle fork。

| 来源 | 固定版本 | 许可 | 使用方式 |
| --- | --- | --- | --- |
| huggingface/candle | `31f35b147389700ed2a178ee66a91c3cc25cc80d`（上游 tag `0.11.0`，"Bump candle version to 0.11.0 (#3658)"） | Apache-2.0 OR MIT | 全工作区统一的 Candle 计算运行时 |
| yexiyue/candle（fork） | 分支 `talechime-perf`，基线同上 | Apache-2.0 OR MIT（与上游一致，未改许可） | 承载可上游化的性能补丁 |

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

## 退出条件

当 `FORK.md` 中登记的补丁全部被上游合并或放弃后，删除 `[patch.crates-io]`、
刷新 `Cargo.lock`，并归档或删除 fork 仓库。
