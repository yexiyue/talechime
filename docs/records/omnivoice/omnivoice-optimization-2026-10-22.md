# OmniVoice 0.6B 融合优化与验收（2026-10-22，Apple M4 Pro）

本记录接续 [热点定位与官方对照](omnivoice-hotspot-and-official-2026-10-22.md)：
把那轮给出的三项优化候选（融合 SDPA、融合 rms_norm、device 端 top-k）逐项落地，
给出数值 parity、ABBA 验收、CPU provider 回归验证，以及**"逐位输出等价"不可达**
的量化证据。

**结论先行**：融合 SDPA + 融合 rms_norm 已落地，短语料 RTF 1.088 → 0.783（**-28.0%**，
两次独立 ABBA 分别为 -28.04% 与 -26.64%，两组区间均不重叠）；
device 端 top-k 实测收益上限 0.18%，按 KISS/YAGNI **不实现**；
**离散输出已改变**（离散 code 差异 80.7%），合并前**必须人工试听**，本轮试听与
ASR 回读**尚未执行**。

## 一、结论摘要

| 项 | 结果 | 证据级别 |
| --- | --- | --- |
| 步 1 融合 SDPA | 已实现（Metal 走 `candle_nn::ops::sdpa`，非 Metal 保留手写路径）；Metal 测试 `fused_sdpa_matches_manual_attention` 通过 | 本地测试 |
| 步 2 融合 rms_norm | 已实现（Metal / 非 f16/bf16 走融合 kernel，其余保留 `forward_diff`）；Metal 测试 `fused_rms_norm_tracks_f64_host_reference` 通过，融合版与 layer-norm 路径对 f64 参考的误差**相同** | 本地测试 |
| 步 3 device 端 top-k | **不实现**。Metal→CPU→Metal 回退的全量 argsort 实测 0.20 ms/step，占每步 ~110 ms 的 **0.18%** | 本地测量 |
| 端到端性能 | 短语料 ABBA 8 进程（1 预热 + 5 热测）：A 5.8775 s / B 4.2294 s，**-28.04%**，RTF 1.0884 → 0.7832；区间 [5.5759, 5.9801] vs [4.0152, 4.3096] 不重叠 | 本地测量 |
| 复现性 | 第二次完整 ABBA（最终提交代码的独立二进制）：**-26.64%**，区间同样不重叠 | 本地测量 |
| 长语料 | A 55.69–60.25 s / B 41.89 s，**约 -28.6%**（音频 50.99 s vs 51.12 s） | 本地测量 |
| CPU provider | **未受影响**：CPU 路径输出与改动前**逐字节一致**（`0bfc4c0c…`） | 本地测量 |
| 输出等价 | **不成立**。融合后离散 code 与基线差异 80.7%（排除参考预热段），音频 100 ms 包络相关 0.97、样本级相关 0.41 | 本地测量 |
| 人工试听 | **未执行**（pending） | 真实业务验收 |
| ASR 回读 | **未执行**（本机 `~/.talechime/resources` 无 ASR 模型） | 真实业务验收 |

## 二、身份与环境

| 项 | 值 |
| --- | --- |
| 仓库 / 分支 | `talechime`，`codex/omnivoice-hotspot`（基线提交 `d2282dd`，本轮改动未 push） |
| 机器 | Apple M4 Pro（12 CPU 核、16 GPU 核、24 GiB 统一内存），macOS 26.6.2 |
| 依赖 | Candle 统一版，fork `github.com/yexiyue/candle` rev `2266099d92265112ecc0d56ecee1761e3cc2ac00`（含 `metal-sdpa` kernel） |
| 权重 | `~/.talechime/resources/omnivoice/models/0.6b/c5fdb5ccb189668d56333f77ba2629f4cd7535f4`（生产清单） |
| 参数 | `num_step=32`、`guidance_scale=2.0`、`position_temperature=5.0`、`class_temperature=0.0`、`t_shift=0.1`、`layer_penalty_factor=5.0`、`seed=42`、voice `narrator` |
| 语料 | 短语料 `target/perf/omni-input/corpus-short.txt`（音频 5.400 s）；长语料 `corpus-long.txt`（音频 50.99 s） |
| 二进制 | A 侧 `target/perf/omni-opt/probe-base`（基线源码构建），B 侧 `target/perf/omni-opt/probe-final`（本轮源码构建）。两者都是 `--release`，同一 `Cargo.lock`、同一权重 |

设备分工：`stage0` 走 Metal + F16，`stage1`（audio tokenizer / vocoder）走 Metal + F32。
所有测量都在**同一台机器**上前后对比；A/B 是两个**独立构建的二进制**，不是同一
二进制上的运行时开关，避免开关本身的分支与缓存差异污染结论。

## 三、实现改动

### 3.1 步 1：融合 SDPA（`crates/omnivoice/src/stage0_qwen3.rs`）

原先 attention 是手写的：`q.matmul(k_t)` 物化完整 `[B, H, S, S]` 分数矩阵 →
`broadcast_add(mask)` → `softmax_last_dim` → `matmul(v)`。S=287 时每层 2.57 ms，
其中 mask 广播加单独占 0.35 ms（要写出 5.3 MB）。

现在拆成两个自由函数，按**设备**分派：

- `attention_sdpa()`：走 `candle_nn::ops::sdpa`，不物化分数矩阵。
  mask 用 `expand` 得到 `[B, H, S, K]` 并让 head 维 stride 为 0，
  共享同一份 `[B, 1, S, K]` 掩码，不复制。
- `attention_manual()`：原手写路径，**非 Metal 设备保留**
  （candle 的 `sdpa` 没有 CPU 实现，强行调用会直接失败）。

分派点是 `q.device().is_metal()`，不是环境变量，因此**没有留开关**，
也不存在"忘记关开关"的误用风险。

### 3.2 步 2：融合 rms_norm（`crates/omnivoice/src/stage0_qwen3.rs`）

`apply_hf_compatible_rms_norm()` 改名为 `apply_rms_norm()`，规则变成：

- Metal，**或** dtype 不是 f16/bf16 → 融合 `norm.forward()`；
- 其余（CPU / CUDA 的 f16、bf16）→ 保留 `norm.forward_diff()`（layer-norm 展开），
  维持 HuggingFace `Qwen3RMSNorm` 的数值语义。

上一轮的 `OMNIVOICE_ABLATE=rms,mask` 消融开关随本轮改动**一起删除**
（`docs/knowledge/tts-performance.md` 要求"失败实验删除代码和开关"，
且消融会改变数值，不能留在生产路径上）。

### 3.3 步 3：device 端 top-k —— 测了，不做

官方用 `torch.topk(scores.flatten(), k)` 在 device 端取 top-k；
我们用全量 `argsort_descending`，Metal 且列数 > 1024 时回退 CPU 排序再拷回。
新增诊断探针 `stage0_argsort_sync`（`Device::synchronize()` 夹住 argsort 调用）实测：

| 阶段（Metal，短语料真实序列 S=287，1 预热 + 2 热测） | 基线 | 优化后 |
| --- | ---: | ---: |
| `stage0_forward_sync`（每步 forward） | 154.72 ms | **105.36 ms** |
| `stage0_logits_cast_sync` | 1.35 ms | 1.32 ms |
| `stage0_predict_sync` | 3.72 ms | 3.80 ms |
| `stage0_update_sync`（含 argsort） | 0.53 ms | 0.72 ms |
| `stage0_argsort_sync`（argsort 单独） | — | **0.20 ms** |
| `stage0_writeback_sync` | 0.36 ms | 0.38 ms |

argsort 只花 0.20 ms/step，占每步 ~110 ms 的 **0.18%**、占 update 阶段的 28%。
收益上限远低于 10% 门槛，而实现 device 端 top-k 会改变采样语义、
需要额外的正确性论证，按 KISS/YAGNI **本轮不实现**，只保留测量证据。

## 四、数值 parity

### 4.1 融合 SDPA vs 手写 attention（Metal 测试）

`fused_sdpa_matches_manual_attention`：B=2、H=16、KV=8、S=287、dim=128、F16，
掩码形状与 stage0 实际使用一致（`[B, 1, S, S]`，加性，运行 dtype）。
断言 `max_abs < 0.05` 且相对 RMS 误差 ≤ 1%——通过。

`fused_sdpa_matches_manual_attention_with_masked_rows`：掩码整行屏蔽
（`-inf`）的场景，同样通过。

### 4.2 融合 rms_norm vs f64 host 参考（Metal 测试）

`fused_rms_norm_tracks_f64_host_reference`：rows=64、hidden=1024、eps=1e-6、
F16 激活。融合 kernel 与 layer-norm 路径**误差相同**：

| 路径 | max_abs | relative RMS |
| --- | ---: | ---: |
| `forward_diff`（layer-norm 展开） | 0.001643 | 2.916e-4 |
| 融合 `forward` | 0.001643 | 2.916e-4 |

两者都受 F16 **输出舍入**限制（相对精度 2^-11 ≈ 4.88e-4），不是融合 kernel
额外引入的误差。**融合 rms_norm 没有精度损失。**

## 五、性能验收

### 5.1 验收协议

- 无插桩、`--release`、同一台机器、同一权重、同一 seed、同一语料。
- 8 进程 ABBAABBA 顺序（A1 B2 B3 A4 A5 B6 B7 A8），每进程 1 预热 + 5 热测，
  只统计热测轮。
- A、B 各用**独立构建的二进制**（`probe-base` / `probe-final`）。
- 门槛：无插桩 ABBA ≥ 10%，且两组区间不重叠。

### 5.2 短语料结果（两次独立 ABBA）

第一次（探索期，同一二进制的 env 开关版；输出 sha 与最终版一致）：

| 组 | 各进程热测均值 (s) | mean-of-means | 区间 | RTF |
| --- | --- | ---: | --- | ---: |
| A（基线） | 5.9801 / 5.9775 / 5.9764 / 5.5759 | **5.8775** | [5.5759, 5.9801] | 1.0884 |
| B（融合） | 4.3024 / 4.3096 / 4.2905 / 4.0152 | **4.2294** | [4.0152, 4.3096] | 0.7832 |

**-28.04%，两组区间不重叠。** 输出 sha：A 全部 `687c30a2…`，B 全部 `6c0f2845…`，
跨进程完全确定。

第二次（最终提交代码的独立二进制，`abba2/`）：

| 组 | 各进程热测均值 (s) | mean-of-means | 区间 | RTF |
| --- | --- | ---: | --- | ---: |
| A（基线） | 5.7767 / 6.0044 / 5.9990 / 6.5716 | **6.0879** | [5.7767, 6.5716] | 1.1274 |
| B（融合） | 4.2017 / 4.3242 / 4.3277 / 5.0117 | **4.4663** | [4.2017, 5.0117] | 0.8271 |

**-26.64%，两组区间不重叠。** 两次 ABBA 的绝对值有 ~3% 的机器级漂移
（后段进程普遍偏慢/偏快，方向不固定，属持续负载下的运行间波动），
但改善幅度、区间分离和输出确定性三次一致复现。

### 5.3 长语料结果

| 组 | 热测均值 (s) | 音频 (s) | RTF |
| --- | --- | ---: | ---: |
| A（基线） | 60.249 / 60.174 / 58.391 / 55.693 | 50.99 | 1.150 |
| B（融合） | 41.885 / 41.890 | 51.12 | 0.819 |

**约 -28.6%。** 音频长度从 50.99 s 变成 51.12 s（+0.13 s），说明融合改变了
离散 token 序列的分段边界——这是数值改动的直接后果，不是测量误差。

### 5.4 与官方的关系

官方 `k2-fsa/OmniVoice` rev `08be0b4` 的短句 RTF 是 **0.635**（每步 forward
96.6 ms）。本轮之后我们短句 RTF 0.783–0.827，每步 forward（带探针同步）
105 ms。**我们仍然慢于官方**，不能声称已经追平；剩余差距主要在
logits 物化策略、stage1 设备放置与切分粒度上，属于后续独立课题。

## 六、为什么"逐位输出等价"不可达（量化证据）

stage0 是**扩散采样器**：每一步的离散 token 会反馈进下一步的输入，
所以任何超过浮点舍入量级的数值扰动都会放大成离散输出的差异。

探索期临时加了一个 attention scale 扰动探针（把 scale 乘成 `(1+eps)`），
配合 `OMNIVOICE_CODES_DUMP` 导出最终离散 code 逐位比较。该探针**已在合入前删除**，
数据保留在 `target/perf/omni-opt/chaos2-*.bin`：

| 相对扰动 | 离散 code 变化 | 结论 |
| ---: | ---: | --- |
| 0（基线） | — | 参照 |
| 1e-8 | **0 / 6240（0.0%）** | 逐位一致 |
| 1e-7 | 4992 / 6240（**80.0%**） | 已完全发散 |
| 1e-6 | 3468 / 6240（55.6%） | 发散 |
| 1e-5 | 4092 / 6240（65.6%） | 发散 |
| 1e-4 | 4122 / 6240（66.1%） | 发散 |

（口径：一次生成的 7 个 segment，去掉参考预热段后共 6240 个离散 code。）

**关键推论**：发散阈值落在 **1e-8 与 1e-7 之间**，而 F16 的相对舍入是
**4.88e-4**——比阈值高 **3～4 个数量级**。任何 F16 级、哪怕是"等价改写"的
stage0 数值改动，都必然改变离散输出。

因此"输出逐位一致"这个验收门槛对**任何有意义的优化**都不可达；
正确做法是：数值 parity 在算子层证明（第四节的 Metal 测试），
端到端接受"离散输出改变"，并把**人工试听 + ASR 回读**作为放行门槛。

> 记录纪律：探索期曾有一版探针把 scale 乘成 `0` 而不是 `1+eps`，
> 导致第一版 ABBA 与第一次 chaos 测量全部作废（作废数据保留在
> `target/perf/omni-opt/abba-invalid-scale/`，**不得引用**）。本记录里的
> 所有数字都来自修复后的重跑。

## 七、输出差异（音频层面）

离散 code 差异（相对基线 `manual/diff`，排除参考预热段，6240 个 code）：

| 配置 | code 差异 |
| --- | ---: |
| manual / 融合 rms | 3792（60.8%） |
| SDPA / layer-norm rms | 4992（80.0%） |
| SDPA / 融合 rms（本轮） | 5034（80.7%） |

音频层面（24 kHz F32，同一语料）：

| 语料 | 时长 A / B | peak A / B | 100 ms 包络相关 | 样本级相关 |
| --- | --- | --- | ---: | ---: |
| 短语料 | 5.400 s / 5.400 s | 0.5000 / 0.5000 | **0.9664** | 0.4080 |
| 长语料 | 50.990 s / 51.120 s | 0.6081 / 0.6469 | **0.6244** | 0.0079 |

短语的时长与峰值完全一致、100 ms 包络相关 0.97，说明**节奏与响度结构保持**；
样本级相关只有 0.41，是因为扩散采样器在**每个采样步**都换了一个离散 token，
波形级相位就整体错开了——这是采样器行为，不是"音质退化"。
长语料的相关更低（0.62 / 0.01），因为长文本被切成 11 段，任一段的切分边界
变化都会让后续段落整体错位。

**但"结构一致"不能替代人工试听。** 本机没有 ASR 模型
（`~/.talechime/resources` 下只有 `moss` / `omnivoice` / `qwen` / `voxcpm`），
无法做机器回读，所以试听与回读都还是 pending。

## 八、CPU provider 回归

融合 SDPA **没有 CPU 实现**，因此 CPU 路径必须继续走手写 attention。
用 `device=cpu`（release 二进制，1 轮短料）对比改动前后的二进制：

| 版本 | 输出 WAV sha256 | 合成耗时 |
| --- | --- | ---: |
| 基线（改动前） | `0bfc4c0cb1011211eca51f15fb9bfcf433e427bcbd7d598349281195d5a599df` | 22.831 s |
| 融合后 | `0bfc4c0cb1011211eca51f15fb9bfcf433e427bcbd7d598349281195d5a599df` | 22.415 s |

**逐字节一致**，CPU 路径未被融合改动影响。

## 九、质量门禁

全部通过（工作区 `codex/omnivoice-hotspot`）：

| 门禁 | 命令 | 结果 |
| --- | --- | --- |
| 格式 | `cargo fmt --all --check` | 通过 |
| 单元/集成测试（CI 组合） | `cargo test --locked --workspace --features talechime/qwen,talechime/voxcpm,talechime/omnivoice,talechime/cosyvoice3,talechime/indextts,talechime/moss-candle,talechime/asr --lib --tests --examples` | 通过 |
| Metal 测试 | `cargo test -p talechime-omnivoice --features metal` | 通过，26 passed |
| Clippy（CI 组合） | `cargo clippy --locked --workspace --all-targets --features … -- -D warnings` | 通过 |
| Clippy（Metal 组合） | `cargo clippy --locked -p talechime --no-default-features --features cli,qwen-metal,voxcpm-metal,omnivoice-metal,cosyvoice3-metal,moss-candle-metal,moss-nano-candle-metal --all-targets -- -D warnings` | 通过 |
| Clippy（OmniVoice crate + Metal 测试） | `cargo clippy --locked -p talechime-omnivoice --features metal --all-targets -- -D warnings` | 通过 |
| 文档 | `cargo doc --locked --no-deps --document-private-items --workspace --features … --examples` | 通过 |

## 十、未完成项

- **人工试听 A/B**（放行门槛，见第十一节的 WAV 路径）。
- **ASR 回读**（本机无 ASR 模型，需要外部模型或换机器）。
- 官方长文本 6 轮连续运行的 ~8% 波动仍未定位。
- stage1 放 CPU 是否值得（跨设备不做等价承诺）。
- Metal allocation 采集、Instruments profile 未做。
- device 端 top-k：已测，收益 0.18%，按 KISS/YAGNI 不做（见 3.3）。

## 十一、复现方式与原始数据

### 11.1 试听用 A/B 音频（绝对路径）

| 用途 | 路径 |
| --- | --- |
| 短语料 基线 A | `/Volumes/yexiyue/omnivoice-hotspot/talechime/target/perf/omni-opt/abba2/A1/round-1-full-f32.wav` |
| 短语料 融合 B | `/Volumes/yexiyue/omnivoice-hotspot/talechime/target/perf/omni-opt/abba2/B2/round-1-full-f32.wav` |
| 长语料 基线 A | `/Volumes/yexiyue/omnivoice-hotspot/talechime/target/perf/omni-opt/abba2/long-LA/round-1-full-f32.wav` |
| 长语料 融合 B | `/Volumes/yexiyue/omnivoice-hotspot/talechime/target/perf/omni-opt/abba2/long-LBfinal/round-1-full-f32.wav` |

### 11.2 原始数据目录

| 数据 | 路径 |
| --- | --- |
| 第一次 ABBA | `target/perf/omni-opt/abba/{A1,B2,B3,A4,A5,B6,B7,A8}.out` |
| 第二次 ABBA（最终代码） | `target/perf/omni-opt/abba2/{A1,B2,B3,A4,A5,B6,B7,A8}/out.jsonl` |
| 长语料 A / B | `target/perf/omni-opt/abba2/long-LA/`、`target/perf/omni-opt/abba2/long-LBfinal/` |
| 两个二进制 | `target/perf/omni-opt/probe-base`、`target/perf/omni-opt/probe-final` |
| 逐阶段探针（基线 / 优化后） | `target/perf/omni-opt/syncprobe-baseline.err`、`target/perf/omni-opt/probe-argsort3.err` |
| 离散 code dump | `target/perf/omni-opt/codes-{manual-diff,manual-fused,sdpa-diff,sdpa-fused}.bin` |
| chaos 实验（修复后） | `target/perf/omni-opt/chaos2-{0,1e-8,1e-7,1e-6,1e-5,1e-4}.bin` |
| CPU 回归 | `target/perf/omni-opt/cpu-{base,new}.out` |
| 作废数据（禁止引用） | `target/perf/omni-opt/abba-invalid-scale/`、`target/perf/omni-opt/chaos-*.bin` |

### 11.3 保留的诊断开关（默认关闭）

- `OMNIVOICE_SYNC_PROBE=1`：逐阶段 GPU 同步计时 + 金丝雀 GEMM + 段长打印 +
  `stage0_argsort_sync`。
- `OMNIVOICE_CODES_DUMP=<path>`：把每次生成的最终离散 code 追加写入文件，
  用于两条数值路径逐 token 比较。
- `OMNIVOICE_STAGE_TIME=1`：stage 级耗时输出。

三个开关都**只在诊断时打开**，验收测量全程关闭。

### 11.4 合并与发布

- 本轮改动在 `codex/omnivoice-hotspot` 分支，**未 push、未合并、未发包**。
- 合并前需要：人工试听 A/B + （有 ASR 模型后）机器回读。
- MOSS nano 走 ORT 是另一条独立路线，不在本记录范围内。
