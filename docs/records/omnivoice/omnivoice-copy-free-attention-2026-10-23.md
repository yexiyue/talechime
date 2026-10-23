# OmniVoice stage0 免拷贝注意力路径（2026-10-23）

> 审计更正：原驱动的编译监控不可靠。下列历史计时保留原值；候选是否保留及
> 与官方的比较，以后附独立复核为准。免拷贝仅指减少 q/k 布局物化，仍会分配归一化和旋转输出。

本轮把 stage0 注意力的 q/k 从"先转成头主序再归一化/旋转"改为"在投影本来就有的
序列优先布局上归一化/旋转"，**去掉了每层两次布局拷贝**。历史整链短文本 ABBA
**3.0384 s → 2.7143 s（-10.67%）**，进程均值区间不重叠，输出与改动前
**逐字节一致**。

本轮另记录若干离线候选；它们的局部计时不能替代生产路径验证。

## 身份与环境

- 源码基线：`a43e46a`，分支 `codex/omnivoice-hotspot`。
- Candle：固定 `2266099d92265112ecc0d56ecee1761e3cc2ac00`，本轮没有改依赖或内核。
- 官方对照参考：`k2-fsa/OmniVoice` rev `08be0b4`，本地 `/Users/yexiyue/.cache/talechime-ref/OmniVoice`。
- 实测机器：Apple M4 Pro，12 CPU 核 / 16 GPU 核 / 24 GiB，macOS 26.6.2（25G83）。
- 模型：生产清单 revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`，本机现有权重，
  不下载、不转换；stage0 F16，stage1 F32。
- 线程：Rayon / VecLib / OMP 各 8；推理串行；先完成编译再测量。
- 语料：`target/perf/omni-input/corpus-short.txt`（音频 5.400 s）、
  `corpus-long.txt`（音频 51.12 s）。
- 全部 release、无插桩（`observation_mode=off`），WAV 写出在计时区外。

二进制 SHA256：

| 侧 | SHA256 |
| --- | --- |
| 原路径 `probe-base` | `f5c6356b8972424866eb9269b5c87cd44fbde7f4e5f752d7f99553b06b1ff9a2` |
| 候选 `probe-candidate` | `26279fccad493c55c7b9004d76f744e0432e2407a8a5675e97d6a779adaee5c9` |

原路径二进制摘要与上一轮 probe-pointwise 相同，只能证明原路径产物身份；
候选的源码对应关系另外通过当前源码构建和测试复核。

## 一、GEMM 微基准：提交与同步完成的墙钟时间

新增离线探针 `crates/omnivoice/examples/gemm_merge_probe.rs`。除计时外它把
**主机提交时间**与 **同步完成墙钟时间**分开打印。

`M=314`（短句真实形状：batch 2 × 序列 157）下的单次 GEMM：

| 形状 | 提交 | 同步完成墙钟时间 |
| --- | ---: | ---: |
| N=1024 K=1024 | 6.5 µs | 312.8 µs |
| N=2048 K=1024 | 5.1 µs | 260.5 µs |
| N=4096 K=1024 | 7.5 µs | 501.5 µs |
| N=3072 K=1024 | 7.0 µs | 391.6 µs |
| N=6144 K=1024 | 17.4 µs | 752.7 µs |
| N=1024 K=3072 | 4.2 µs | 445.0 µs |
| N=8200 K=1024 | 7.9 µs | 1001.6 µs |

这些独立 GEMM 的提交时间相对较短，但同步墙钟不是 GPU timestamp。
不能据此判定整个生产 forward 仅受 GPU 限制，也不能排除融合和减少分派的收益。

结论二：`N=1024` 的形状只有约 2.7–3.9 TFLOP/s，而 `N=4096` 有 5.25 TFLOP/s，
小 N 的 GEMM 效率明显偏低。

一个 28 层 step 的投影 GEMM 总量：

| 配置 | 提交 | 同步完成墙钟时间 |
| --- | ---: | ---: |
| 分开投影 M=314 | 0.61 ms | 68.13 ms |
| 合并 qkv / gate_up M=314 | 0.43 ms | 65.81 ms |

合并投影逐位等价（见第三节），但只省约 1.9 ms/step，本轮不实现。

## 二、再定位：真正的浪费是注意力子路径的布局拷贝

`crates/omnivoice/examples/attention_layout_probe.rs` 在真实形状上逐项测量
stage0 注意力子路径：

| 操作 | 提交 | 同步完成墙钟时间 |
| --- | ---: | ---: |
| `q` 视图 `flatten(0,2)`（拷贝） | 3.2 µs | 293.7 µs |
| `k` 视图 `flatten(0,2)`（拷贝） | 2.5 µs | 138.0 µs |
| rms_norm（连续 `[B*H,S,D]`） | 3.8 µs | 54.9 µs |
| rms_norm（带 stride 的 `[B,H,S,D]` 视图） | 46.6 µs | 917.6 µs |
| rope（连续 `[B,H,S,D]`） | 3.6 µs | 69.1 µs |
| sdpa（连续 q/k/v） | 4.2 µs | 97.6 µs |
| sdpa（带 stride 的 q/k/v） | 3.2 µs | 95.9 µs |
| `context.transpose(1,2).contiguous()` | 2.9 µs | 339.1 µs |
| `context.permute(0,2,1,3).contiguous()` | 2.5 µs | 338.0 µs |
| `context` 沿序列 narrow+cat | 950.5 µs | 1598.6 µs |

要点：

1. **每层两次投影拷贝 293.7 + 138.0 µs，28 层合计约 12.1 ms/step**。
   它们存在的原因只是：融合 rope kernel 只接受连续的 `[B,H,S,D]`，而投影输出
   是连续的 `[B,S,H*D]`。
2. 带 stride 的 rms_norm 反而更慢（917.6 µs）且与连续路径有 25/643072 个元素
   不同，不能拿来替代拷贝。
3. 融合 rope kernel 直接拒绝带 stride 的输入（`xs has to be contiguous in rope`）。
4. sdpa **接受**带 stride 的 q/k/v，且与连续输入逐位一致（0/643072 元素不同），
   甚至略快（95.9 vs 97.6 µs）。
5. `context` 的转置拷贝 339.1 µs/层（约 9.5 ms/step）是剩下最大的一笔拷贝，
   沿序列 narrow+cat 反而慢 4.7 倍，本轮不处理。

## 三、改动：序列优先的 rope（`rope_thd`）

固定 Candle fork 的 `candle-nn/src/rotary_emb.rs` 同时提供两个融合 kernel：

- `rope`：输入必须是连续的 `[B,H,S,D]`，cos/sin 按序列位置索引；
- `rope_thd`：输入必须是连续的 `[B,S,H,D]`，cos/sin 同样按序列位置索引。

两者的算术完全相同（`dst[i1] = src[i1]*cos - src[i2]*sin`，
`dst[i2] = src[i1]*sin + src[i2]*cos`，同一份 half-split 约定），
`rope_thd` 的 metal kernel 用 `i_t = (i_bth / h) % t` 取序列位置。

生产改动只在 `crates/omnivoice/src/stage0_qwen3.rs`：

- Metal：投影输出 reshape 成 `[B,S,H,D]`（连续视图，零拷贝）→ 融合 rms_norm →
  `rope_thd` → `transpose(1,2)` 得到 `[B,H,S,D]` 视图，交给 sdpa。
- 其他设备：保持原路径（头主序 + `rope`），CPU 输出不受影响。

因此 Metal 上 q/k 不再有任何布局拷贝，sdpa 读的 q/k 是带 stride 的视图。

### 逐位等价证据

`crates/omnivoice/examples/attention_copyfree_probe.rs`：

| 比较 | 结果 |
| --- | --- |
| q：原路径 vs 免拷贝路径 | 0 / 643072 元素不同 |
| k：原路径 vs 免拷贝路径 | 0 / 321536 元素不同 |
| sdpa（无 mask）：原路径 vs 免拷贝路径 | 0 / 643072 元素不同 |
| sdpa（有 mask）：原路径 vs 免拷贝路径 | 0 / 643072 元素不同 |

同探针的成本对比（同一循环内，避免跨次分配偏差）：

| 路径 | 同步完成墙钟时间 |
| --- | ---: |
| 原 q 路径（拷贝 + 归一化 + rope） | 655.7 µs |
| 免拷贝 q 路径（归一化 + rope_thd + 转置视图） | 155.6 µs |

新增单元测试（Metal feature 门控）：

- `sequence_first_rope_matches_head_major_rope`：16 头与 8 头两种形状下，
  序列优先路径与头主序路径的 q/k 位模式完全相同。
- `strided_attention_matches_contiguous_attention`：有/无 mask 两种情况下，
  sdpa 读带 stride 的 q/k/v 与读连续张量结果完全相同。

## 四、真实模型输出对照

候选二进制与冻结基线逐字节一致：

| 产物 | 基线摘要 | 候选摘要 |
| --- | --- | --- |
| 短文本 codes dump | `d1f037724f6a674505e3e0f52c5ae23d3c341d95cefb6c7ad74d7a7015d829c3` | 同左 |
| 长文本 codes dump | `69dd8d8476dcb38aef411a588461245ae39026dc9159c4d6af63d5bdbb42b29c` | 同左 |
| 短文本 F32 WAV | `2eadc756eb4482fe320efe678ef8a67312ce79dcfa6d6f4dbf41ed4642be324e` | 同左 |
| 长文本 WAV（24 个） | 逐文件比较 | 24/24 逐字节相同 |
| CPU F32 WAV | `0bfc4c0cb1011211eca51f15fb9bfcf433e427bcbd7d598349281195d5a599df` | 同左 |

codes dump 含参考准备与正文生成；WAV 同时比较整段与各切分段的 F32 与 PCM16。
CPU 侧沿用未改动的分支，输出与上一轮记录的 CPU 基线一致。

## 五、整链验收

协议：无插桩 release、独立进程、ABBAABBA 顺序（A1 B2 B3 A4 A5 B6 B7 A8）、
每进程 1 预热 + 5 热测、只统计热测轮、短文本、32 步、seed 42、narrator。
旧驱动每 0.5 秒采样进程，但错误地对完整路径做前缀匹配；空结果不能证明
没有编译干扰。这组性能结果保留为历史记录，需由修正监控后的复测独立确认。

| 组 | 各进程热测均值 (s) | mean-of-means | 区间 | RTF |
| --- | --- | ---: | --- | ---: |
| A 原路径 | 3.03937 / 3.03615 / 3.04113 / 3.03685 | **3.03837 s** | [3.03615, 3.04113] | 0.5627 |
| B 免拷贝 | 2.71599 / 2.71758 / 2.70848 / 2.71496 | **2.71425 s** | [2.70848, 2.71758] | 0.5026 |

**-10.67%，两组区间不重叠。** 40 个热轮的 F32 WAV 摘要全部是
`2eadc756eb4482fe320efe678ef8a67312ce79dcfa6d6f4dbf41ed4642be324e`，A、B 两侧一致。

相对上一轮记录的官方同条件短句 RTF 0.635，我们这一轮是 0.5026；官方数字来自
不同时间、不同构建与不同进程，不构成同条件对照，不能说明当前领先官方，不作为排名结论。

## 六、被否掉的方向

### 前缀 KV 缓存与拆分 CFG：尚未实现验证

stage0 的 cond 行前 `c_len - target_len` 个位置（短句 27 个）在 32 步里恒定，
但输入 token 不变不代表深层 K/V 不变：双向注意力下，前缀表示会受到变化中的音频目标影响。
不能直接套用自回归前缀 KV cache。下表只是 GEMM 行数微基准：

| 每 step 的位置数 | 28 层投影 GEMM |
| --- | ---: |
| M=314（现状，CFG 两行合批） | 68.13 ms |
| M=287 | 66.68 ms |
| M=260（两行都只算目标段） | 64.92 ms |
| M=157（只有 cond 行） | 41.45 ms |
| M=130（只有 uncond 行） | 40.38 ms |

拆分两行的投影微基准耗时相加高于合批；这提示可能降低 GEMM 效率，
但不是完整 forward 实测。M=260 也不是已经实现的前缀缓存方案，不能由此推算
缓存可节约的真实时间。本轮没有实现这两个候选。

### 合并 q/k/v 与 gate/up 投影：仅完成合成输入微基准

离线探针所用输入的逐位等价已验证（0 个元素不同），不等于真实权重整链验收，28 层 step 从 67.72 ms 降到 65.81 ms，
约 -1.9 ms/step（约 forward 的 1.8%）。要动权重装载与量化分支，
尚未完成真实权重输出与整链验证，本轮不作为已确认优化保留。

### 音频头只算目标段

音频头 `[314,1024]×[1024,8200]` 约 1010 µs/step；换成 M=260 仍是 1003 µs，
M=130 才是 635 µs。这个投影不是行数受限，省不到时间。

## 七、离线诊断线索（不等于生产耗时分解）

| 位置 | 量级 | 判断 |
| --- | ---: | --- |
| stage0 投影 GEMM | 合成 28 层 GEMM 组约 65.8–68 ms | 主要成本；独立微基准结果；不能据此穷尽生产优化方式 |
| `context` 转置拷贝 | 独立复制操作约 339 µs | sdpa 输出布局固定为 `[B,H,S,D]`，o_proj 需要 `[B,S,H*D]`；测过的替代写法更慢，尚未证明必须修改 fork |
| 融合 rms_norm | 独立操作约 54.9 µs | 已是融合 kernel |
| 融合 rope | 独立操作约 69.1 µs | 已是融合 kernel |
| sdpa | 独立操作约 97.6 µs | 已是融合 kernel |
| 采样打分（3 次 log_softmax + mask + argmax/gather） | 约 2.6 ms/step | 固定 fork 没有融合 log_softmax；用 softmax+log 改写会改变算术 |
| stage1 七抽头膨胀卷积 | 2.40 / 4.63 / 8.41 / 7.70 / 11.38 ms 每块 | stage1 最大单笔；下一轮候选 |

## 八、质量门禁

| 门禁 | 结果 |
| --- | --- |
| `cargo fmt --all --check` / `git diff --check` | 通过 |
| `cargo test --locked -p talechime-omnivoice --features metal --lib` | 34 passed |
| `cargo test --locked -p talechime-omnivoice --no-default-features` | 23 passed |
| 本 crate，Metal，Clippy all-targets / `-D warnings` | 通过 |
| 本 crate，Metal + `candle-core/accelerate`，Clippy all-targets / `-D warnings` | 通过 |
| 本 crate rustdoc，Metal / no-deps / document-private-items / examples | 通过 |

普通测试不下载模型；真模型验证只读取本机既有资源。没有改会话、取消或采样行为，
没有新增运行时开关，没有 push、合并、发包或部署。

## 复现与产物

```sh
cargo build --release --locked -p talechime-omnivoice --features metal \
  --example gemm_merge_probe --example attention_layout_probe --example attention_copyfree_probe
target/release/examples/gemm_merge_probe
target/release/examples/attention_layout_probe
target/release/examples/attention_copyfree_probe

cargo build --release --locked -p talechime-backends \
  --no-default-features --features omnivoice-metal,performance-tools --example performance_probe
```

原始数据在忽略目录 `target/perf/omni-round2/`：

- `probe-base` / `probe-candidate`：两个独立 release 二进制。
- `abba/`：ABBA 八进程逐轮计时、每轮 WAV 摘要与 `summary.json`（含编译活动采样）。
- `parity-short*` / `parity-long*` / `parity-long2*`：codes dump 与 WAV 摘要。
- `cpu-out/`：CPU 回归输出。
- `abba_driver.py`：ABBA 驱动脚本（顺序、预热/热测轮数、编译活动采样、统计）。

## 独立审计后的保留结论

见 [官方对比复核](omnivoice-official-comparison-audit-2026-10-23.md)。重新构建当前源码，
新二进制 SHA256 为 `8cfb1b4c82d0bbc56013da5cac0f10ed34add04bf21b104fd2c629b1d303f0ef`，
与历史候选二进制分别标识，不混用产物身份。新产物重新验证短/长完整 codes 及
28 个 F32/PCM16 WAV，与冻结基线均逐字节一致；Metal 34、CPU 23 tests 通过，
Clippy、rustdoc 和格式检查通过。

修正监控后进行三路 ABCCBA、每进程 1 预热 + 3 热轮：a43e46a 组均值
3.02383 s，新候选 2.67203 s（减少 11.63%），固定官方源码 MPS F16 路径 3.23450 s。
每侧两个进程，未观察到 Cargo/实际编译器活动，仍非独占系统或锁频基准。
这是本轮完成保留的依据。历史微基准不作为前缀缓存、合并投影或生产瓶颈的充分证明。
