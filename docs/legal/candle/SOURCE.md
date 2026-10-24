# Candle 第三方来源声明

工作区的 `candle-core` / `candle-nn` / `candle-transformers` / `candle-metal-kernels` 通过根
`Cargo.toml` 的 `[patch.crates-io]` 指向 Talechime 的 Candle fork。

| 来源 | 固定版本 | 许可 | 使用方式 |
| --- | --- | --- | --- |
| huggingface/candle | `31f35b147389700ed2a178ee66a91c3cc25cc80d`（上游 tag `0.11.0`，"Bump candle version to 0.11.0 (#3658)"） | Apache-2.0 OR MIT | 全工作区统一的 Candle 计算运行时 |
| yexiyue/candle（fork） | 分支 `talechime-perf`，基线同上 | Apache-2.0 OR MIT（与上游一致，未改许可） | 承载可上游化的性能补丁 |

工作区固定 fork revision 为 `884b359895a16c8fbd3647789b1f7c7974894708`。
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

工作区固定 rev `884b359895a16c8fbd3647789b1f7c7974894708` 相对上游包含以下行为补丁及一项休眠实验内核，登记在 fork 的 `FORK.md`：

| 补丁 | 位置 | 性质 | 实测 |
| --- | --- | --- | --- |
| skinny-M GEMM 用 32 行 tile | `candle-metal-kernels::mlx_gemm::select_tile_config` | 行为级，不改公开 API | VoxCPM LocDiT `m=22`：21.71 → 18.99 ms/帧（-12.5%），WAV 逐字节一致 |
| 单通道共享 FIR 直卷 | Metal conv1d 的连续 F32 / 单输入输出通道分支 | 行为级，不改 core 公开 API | IndexTTS 固定输入误差 ≤9.5e-7；其余布局回退 |
| rank-4 transpose(1,2) 连续化 | Metal copy 的精确 stride 签名分支 | 纯复制，不改 core 公开 API | CosyVoice3 / IndexTTS 转置，位级门；见下方记录 |
| 二元广播快路径 | Metal binary 的 F32 rank-3 广播分支 | 行为级，不改 core 公开 API | 广播与窄视图位级门；其余布局回退 |
| dense direct conv1d | Metal kernels 内核与 launcher | **未接入调度的休眠实验** | CosyVoice3 实测慢于 im2col，不能计为生产收益 |

其中 skinny-M 补丁来自上游 #3313 引入的 `if m < 16 { return TILE_32_32_16_2_2 }` 守卫：
`16 <= m <= 32` 时 `ceil(m/32) == ceil(m/64) == 1`，即用 32 行 tile 不增加 M-tile
数量，却被送进 64 行 tile 多算 32–42 行 padding。修正为 `m <= 32`。

因此轨道 B（`publish = false`）仍**未触发**：`voxcpm` 等 crate 继续以
`publish = true` 发布，只是下游从 crates.io 安装时会解析 registry 版 Candle，
拿不到这些补丁的性能收益，直到相应补丁被上游合并。

宿主融合内核使用 Candle 0.11.0 的公开 custom-op / Metal API，与上述 fork 内部
launcher 分开维护。详细实现登记见固定 revision 的 `FORK.md`；验证边界见
[转置记录](../../records/cosyvoice3/cosyvoice3-metal-transpose-copy-2026-10-24.md)、
[广播记录](../../records/cosyvoice3/cosyvoice3-metal-broadcast-binary-2026-10-24.md) 与
[宿主融合记录](../../records/indextts/indextts-metal-hosted-kernels-2026-10-24.md)。

## 退出条件

当 `FORK.md` 中登记的补丁全部被上游合并或放弃后，删除 `[patch.crates-io]`、
刷新 `Cargo.lock`，并归档或删除 fork 仓库。
