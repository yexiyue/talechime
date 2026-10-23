# OmniVoice 0.6B 热点定位与官方对照（2026-10-22，Apple M4 Pro）

> 2026-10-23 更正：直接观测模型层输入后确认，短句的实际 attention 序列长为
> 157，CFG 两路沿 batch 维拼接，不是 cond=157 + target=130 = 287。本文 S=287
> 微基准是独立合成形状，不能用于该短句的逐层成本外推；端到端和 forward 原始
> 测量仍有效。后续固定 codes 实验定位到 stage1 反卷积的通用慢路径。

本记录接续 [OmniVoice 0.6B 性能基线](omnivoice-performance-2026-10-22.md)：把基线里的
三个“假设”逐条换成测量结论，拉取官方 `k2-fsa/OmniVoice` 做同条件对照，并给出下一轮
优化的优先级。本轮**没有改动任何生产默认路径**，也没有做输出保持型优化验收；
人工试听与 ASR 回读仍未执行。

## 一、结论摘要

1. **我们比官方慢 1.6×（短句）**：同一段短文本、同一 seed、同一 32 步，
   官方 MPS 每步 forward 96.6 ms，我们 Metal 每步 forward 157 ms。
   官方短句 RTF 0.635，我们 1.111。
2. **差距集中在 attention 与 rms_norm，不在 GEMM**：同形状 GEMM 上
   Candle Metal 与 PyTorch MPS 基本持平（M=2800 时我们低 8–10%）；
   但我们的 attention（含 mask 广播）每层 2.57 ms，融合 SDPA 只要 0.39 ms；
   我们的 rms_norm 走
   手写分解 170–196 µs/次，融合算子只要 12.6 µs。
3. **基线记录里“第二代起 forward 退化 2.6×”的结论是错的**（见第七节）。
   真实原因是测量框架的第一次生成用的是 9 字参考预热文本（target_len 48），
   之后才是语料文本（target_len 130）。序列长度不同，不是进程退化。
4. **长文本上我们与官方持平**：我们 RTF 1.176，官方 1.163–1.279（同进程 6 轮
   的官方数据波动 8%，见第四节）。长文本总 forward 时间两边都是 ~56–58 s，
   差别只在切分粒度（我们 11 段 × 32 步，官方 4 段 × 32 步）。
5. 下一轮的收益排序：**融合 SDPA（估 -20%~-27%）> 融合 rms_norm（实测 -19%）
   > mask 广播路径（实测 -8%）> device 端 top-k 替代全量 argsort（未测）**。

## 二、身份与环境

| 项 | 我们（Talechime） | 官方（OmniVoice） |
| --- | --- | --- |
| 仓库 | `talechime`，提交 `ec3b862`（本轮改动在 `codex/omnivoice-hotspot`） | `github.com/k2-fsa/OmniVoice`，revision `08be0b4ccbac3e13e374e86fbfead4b4cac343e2` |
| 依赖 | Candle 统一版，fork `github.com/yexiyue/candle` rev `2266099d92265112ecc0d56ecee1761e3cc2ac00` | torch 2.8.0 / transformers 5.3.0 / Python 3.12.14 |
| 设备路径 | Metal（stage0 F16，stage1/audio F32） | MPS（参数与 buffer 分布 cpu+mps，dtype float16） |
| 权重 | `~/.talechime/resources/omnivoice/models/0.6b/c5fdb5ccb189668d56333f77ba2629f4cd7535f4` | 同一份权重目录 |
| 参数 | `num_step=32`、`guidance_scale=2.0`、`position_temperature=5.0`、`class_temperature=0.0`、`t_shift=0.1`、`layer_penalty_factor=5.0`、`seed=42` | 同左 |
| 音色 | `narrator`（模型内建） | 官方默认 auto voice（narrator） |
| 模型结构 | Qwen3 0.6B：28 层 / 16 头 / 8 KV 头 / hidden 1024 / ffn 3072 / head_dim 128；audio codebook 8×1025 | 同一份 config.json |

机器与系统：Apple M4 Pro（12 CPU 核、16 GPU 核、24 GiB 统一内存）、macOS 26.6.2。
两边都是同一台机器、同一份权重、同一段文本、同一 seed。

**口径警告**：PyTorch 与 Rust 的随机序列不同，相同 seed 不代表输出逐样本相同；
CPU/MPS/Metal 的归约与累加顺序也不同。本记录的对照只用于**性能口径**，
不构成“数值等价”的证据。

官方可运行副本放在本地盘 `/Users/yexiyue/.cache/talechime-ref/OmniVoice`
（原始拉取在慢盘 `/Volumes/yexiyue/talechime/runs/reference/OmniVoice`）。
慢盘上 Python 导入极慢（`import sympy` 8.5 s，首次 `import omnivoice` 62 s），
venv 必须放本地盘，否则基准会被磁盘 I/O 主导。

## 三、官方实现与我们的实现差异

| 位置 | 官方（PyTorch） | 我们（Candle/Rust） | 影响 |
| --- | --- | --- | --- |
| 注意力 | transformers 的融合 SDPA（`omnivoice/models/omnivoice.py` 走 `_attn_implementation="sdpa"`，也支持 flex_attention 分支） | `crates/omnivoice/src/stage0_qwen3.rs` 手写 `q.matmul(k_t)` → 物化完整 `[B,H,S,S]` 分数 → `broadcast_add(mask)` → `softmax_last_dim` → `matmul(v)` | **最大差距来源** |
| GQA | 2026-10-23 更正：本地 transformers 5.3.0 的 SDPA 在有 attention mask 时也会 `repeat_kv`；不能把官方此路径称为原生 GQA | 当时 `repeat_kv` 物化 8→16 头；后续 Metal 改用原生 GQA | 见后续独立实测 |
| flash attention | 有 `omnivoice_flashinfer.py` 分支 | `flash_seqlens` 参数被完全忽略，`try_flash_attention_per_row` 是返回 `None` 的桩 | Metal 上永远没有融合注意力 |
| 位置选择 | `torch.topk(scores.flatten(), k)`（device 端） | `argsort_descending` 全量 argsort，Metal 且列数 > 1024 时回退 CPU 排序再拷回 | 每次生成一次主机往返 |
| logits 精度 | 每步把整批 `.to(torch.float32)` | 只 cast 目标切片 | **我们更省**：官方每步把 `[2,8,287,1025]`（约 4.7M 元素、18.8 MB F32）整批物化，我们只 cast 目标切片 |
| audio tokenizer / vocoder | MPS 下强制放 CPU（`tokenizer_device = "cpu" if str(model.device).startswith("mps")`） | stage1 跑在 Metal + F32 | 我们 stage1 更贵（短句 615 ms，长文占 ~4 s） |
| 切分粒度 | 长文本切成 4 段 × 32 步 | 长文本切成 11 段 × 32 步 | 总 forward 时间接近，但每步序列长度差 3–4 倍 |

反过来，我们有两处比官方好：logits 不做全量 F32 物化；stage0 用 F16 而不是 F32。

## 四、性能对照

### 4.1 端到端 RTF

官方为 1 次预热 + 5 次热测（`bench_macos_mps.py`，WAV 导出在计时外，
计时区前后 `torch.mps.synchronize()`）；我们沿用基线记录的
3 进程 × (1 预热 + 5 热测) 协议。

| 语料 | 官方 MPS | 我们 Metal（基线记录 `ec3b862`） | 我们 Metal（本轮） |
| --- | ---: | ---: | ---: |
| 短句 | 3.423 s / 音频 5.390 s / **RTF 0.635**（0.634–0.638） | 5.999 s / 5.400 s / **RTF 1.1110**（1.1048–1.1158） | 同上 |
| 长文 | 65.66 s / 51.334 s / **RTF 1.279**（1.251–1.316） | 59.98 s / 50.99 s / **RTF 1.1763**（1.1402–1.2250） | 同上 |

长文官方另有一组 1 预热 + 2 热测的 profiling 运行：59.71 / 60.45 s，
RTF 1.163 / 1.177。官方长文本在 6 轮连续跑（65.7 s/轮）与 3 轮（60.1 s/轮）之间
有 ~8% 差距，属于持续 MPS 负载下的运行间波动，未定位。
**我们的长文 RTF 1.176 落在官方两个区间之间，不能声称长文本我们更快。**

### 4.2 每步分解

官方用 `profile_forward_mps.py` 在 `model.llm.forward` 与 `model.forward` 上打点；
我们用新增的 `OMNIVOICE_SYNC_PROBE=1`（`Device::synchronize()` 夹住每个阶段）。
两者都是“GPU 执行时间”，不是命令提交时间。

短语料（target_len 130 / cond_len 157 / 序列长 S=287 / 32 步）：

| 阶段 | 官方 MPS | 我们 Metal | 比值 |
| --- | ---: | ---: | ---: |
| forward 合计 | 96.0–97.3 ms（中位 96.0） | 155–162 ms | 1.62× |
| ├ LLM 骨干 | 94.3–94.7 ms | 未单独切分 | — |
| └ audio head | 1.7–2.5 ms | 未单独切分 | — |
| logits cast | 含在 forward 内（整批 F32） | 1.21–1.70 ms | 我们更省 |
| 采样 / 更新 | 5.2–6.2 ms | predict 3.24–4.31 + update 0.42–0.67 + writeback 0.30–0.50 ≈ 5.9 ms | 持平 |
| 单步合计 | ~102 ms | ~163 ms | 1.6× |
| 32 步 stage0 | ~3.26 s | 5.11 s | — |
| stage1（vocoder） | 官方在 CPU，合计 < 0.2 s | 615 ms（Metal F32） | 我们更贵 |

长语料：我们 11 段 × 32 步 = 384 次 forward，总 forward 时间 55.2 s
（每步 63.8–238.6 ms，随段长变化）；官方 4 段 × 32 步 = 128 次 forward，
总 forward 57.95 s（每步均值 453 ms，中位 491 ms）。
**总计算量两边相同，切分粒度不同。**

## 五、热点定位

### 5.1 方法

- 新增 `OMNIVOICE_SYNC_PROBE=1`：在 `stage0_forward` / `logits_cast` / `predict` /
  `update` / `token_writeback` 前后插 `Device::synchronize()`，
  打印 `[stage] name=... ms=...`；同时打印每步的 `enqueue_ms`（命令提交时间）
  与一个固定形状的“金丝雀”GEMM，用来区分“GPU 变慢”和“CPU 提交变慢”。
- 新增 `OMNIVOICE_ABLATE=rms,mask`：在真实模型上分别切换到融合 rms_norm、
  跳过 mask 广播加，用来量出单项收益。
- 新增 `crates/omnivoice/examples/gemm_probe.rs`：Metal 逐算子微基准，
  形状与官方 `bench_gemm_mps.py` 对齐，用于两边同形状对比。

三个开关都默认关闭；`OMNIVOICE_ABLATE` 会改变数值，**不得用于验收测量**。

### 5.2 GEMM 不是瓶颈

同形状 F16 GEMM，Candle Metal（本轮两次运行）vs PyTorch MPS：

| 形状 (M×N×K) | Candle Metal | MPS |
| --- | ---: | ---: |
| 300×1024×1024 | 219–248 µs（2.5–2.9 TFLOP/s） | 348 µs（1.81） |
| 300×3072×1024 | 400–420 µs（4.5–4.7） | 390 µs（4.84） |
| 300×1024×3072 | 457–467 µs（4.0–4.1） | 340 µs（5.56） |
| 300×8200×1024 | 1023–1078 µs（4.7–4.9） | 969 µs（5.20） |
| 2800×1024×1024 | 1101–1133 µs（5.2–5.3） | 988 µs（5.94） |
| 2800×3072×1024 | 3385–3411 µs（5.2） | 2949 µs（5.97） |
| 2800×1024×3072 | 3262–3378 µs（5.2–5.4） | 2965 µs（5.94） |
| 2800×8200×1024 | 8951–9099 µs（5.2–5.3） | 7915 µs（5.94） |

短句形状我们不慢，长句形状我们低 8–10%。按 28 层估算，短语料的
GEMM 理论下限约 44 ms/step，而实测 forward 157 ms——**GEMM 最多解释三成**。

### 5.3 逐算子微基准（Candle Metal，F16）

`seq=150, tokens=300`：

| 算子 | 耗时 |
| --- | ---: |
| rms_norm（`RmsNorm::forward_diff`，当前路径） | 170–196 µs |
| rms_norm（融合 `ops::rms_norm`） | **12.6–12.8 µs** |
| silu（`broadcast_mul`） | 6.0 µs |
| silu（`Activation::Silu`，[300,3072]） | 153–159 µs |
| rope | 60.5–62.0 µs |
| repeat_kv | 15.7–16.6 µs |
| transpose + contiguous | 135.8–137.9 µs |
| broadcast_add `[2,1,S,S]→[2,16,S,S]` | **351–362 µs** |
| broadcast_add（同形状 full） | 10.3–10.6 µs |
| attention 手写（无 mask） | 502–531 µs |
| attention 手写（有 mask） | 844–869 µs |
| **attention 融合 SDPA**（`candle_nn::ops::sdpa`） | **155–270 µs** |

`seq=287`（真实语料序列长）：

| 算子 | 耗时 |
| --- | ---: |
| attention 手写（无 mask） | 1168 µs |
| attention 手写（含 mask 广播加） | **2569 µs** |
| attention 融合 SDPA（含 mask） | **386 µs** |

mask 广播把每层 attention 的成本抬高了 1.4 ms（`[2,1,S,S] → [2,16,S,S]` 的
`broadcast_add` 要写出 5.3 MB），这正是第六节里 `mask` 消融能省 8% 的来源。

融合 SDPA 在本仓库锁定的 candle fork（rev `2266099`）里已经可用：
`candle_nn::ops::sdpa` → `CustomOp3("metal-sdpa")` → `call_sdpa_full`，
支持加性 mask 与 head 维 stride；限制是要求 mask 形状恰好是
`[bs, qheads, qseq, kseq]` 且 dtype 与 q 相同。
2026-10-23 更正：`q_head == k_head` 检查的是 head dimension，不是头数；
固定 fork 的 `call_sdpa_full` 原生支持 GQA，K/V 不需要先 `repeat_kv`。

### 5.4 真实模型消融（Metal，短语料，forward 均值）

消融在真实测量轮（target_len 130 / S=287）上做，round0 是参考预热文本，
不参与对比：

| 配置 | round1 | round2 | 相对基线 |
| --- | ---: | ---: | ---: |
| base | 155.1 ms | 155.6 ms | — |
| `rms`（融合 rms_norm） | 126.1 ms | 126.4 ms | **-18.7%** |
| `mask`（跳过 mask 广播加） | 142.8 ms | 143.0 ms | **-8.1%** |
| `rms` + `mask` | 118.1 ms | 126.9 ms | -19%（噪声偏大） |

消融只用于定位收益量级，**不是验收**：它没有做无插桩 release ABBA、
没有验证输出等价、也没有数值 parity 检查。

## 六、热点小结（按预估收益）

| 候选 | 证据 | 预估收益（相对当前 157 ms/step） | 风险 |
| --- | --- | ---: | --- |
| attention 换成 `candle_nn::ops::sdpa`（融合 Metal kernel） | 逐算子 2569 → 386 µs/层（-85%）；28 层 | **-61 ms/step（-39%）**，forward 157 → ~96 ms/step | 数值语义变化（融合累加顺序）、mask 需 expand 到 `[B,H,S,S]`；当时实现重复 K/V，内核本身无需重复 |
| rms_norm 换成融合算子 | 真实模型消融 -18.7% | **-29 ms/step** | 与 HF `Qwen3RMSNorm` 的数值 parity 必须先验证 |
| mask 广播路径 | 逐算子 351 µs vs full 10.6 µs；消融 -8.1% | **-12 ms/step** | SDPA 方案会自动覆盖这一项 |
| device 端 top-k 替代全量 argsort | 代码路径存在；官方用 `torch.topk` | 未测 | 采样语义必须等价 |
| stage1（vocoder）放 CPU（对齐官方） | 官方 MPS 下强制 CPU | 短句 stage1 615 ms → 更低 | 会把负载搬到 CPU，Metal F32 路径与 CPU 数值不同 |
| 减少日志量级（长文切分粒度） | 我们 11 段 vs 官方 4 段 | 总 forward 时间基本不变 | 属于切分策略变更，会改变输出 |

单是融合 SDPA 就预估把每步从 157 ms 降到 ~96 ms（与官方每步 96.6 ms 持平）；
再叠加融合 rms_norm 预估到 ~70 ms 量级，短句 RTF 从 1.111 降到 ~0.65–0.80，
**但这是叠加估计，必须以正式 ABBA 实测为准**。

## 七、更正：“第二代起 forward 退化 2.6×”是测量假象

上一轮排查得到的现象是：进程内第一次生成 84 ms/step，之后两次 160 ms/step，
看起来像“第二次生成起 GPU 变慢 2.6×”，并伴随若干已排除项（金丝雀 GEMM 稳定、
CPU 提交时间稳定、stage1 放 CPU 无效、去掉 residency `set.commit()` 无效、
各轮 WAV 字节一致）。

本轮用新增的长度打印直接定位到原因：

```
[stage] name=stage0_lengths target_len=[48]  cond_len=[63]   codebooks=8   ← 参考预热
[stage] name=stage0_lengths target_len=[130] cond_len=[157]  codebooks=8   ← 语料轮 1
[stage] name=stage0_lengths target_len=[130] cond_len=[157]  codebooks=8   ← 语料轮 2
```

测量框架的 `prepare_reference` 会先用 `SHORT_CORPUS = "你好，欢迎使用叙铃。"`
做一次**不计时**的预热合成，它生成的 token 序列只有 48 帧（S=111），
而语料文本是 130 帧（S=287）。**序列长度差 2.6 倍，forward 时间正好差 2.6 倍。**

决定性验证：`num_step=128` 的单次生成里，128 次 forward **全部**是 60 ms 量级，
随后那一代才开始 157 ms——说明触发条件既不是 forward 次数，也不是累计时间，
更不是 GPU 降频（金丝雀 GEMM 在两段都是 ~3.3 ms）。
长文同理：11 段的每步均值从 63.8 ms（短段）到 238.6 ms（长段），
随段长单调变化。

结论：**没有进程级退化**。之前那些“已排除项”仍然成立（金丝雀稳定、
CPU 提交稳定、输出未污染），但它们本来就不需要解释——序列长度本来就不同。
后续所有性能结论都必须**只比较同一 target_len / 同一段长**的轮次。

## 八、未完成项

- 人工试听与 ASR 回读（仍然 pending；本轮没有做任何输出改变，风险低）。
- 融合 SDPA / 融合 rms_norm 的实现与**无插桩 release ABBA**（≥10%、两组区间不重叠、
  输出等价）验收。
- HF `Qwen3RMSNorm` 与 candle 融合 rms_norm 的数值 parity（独立小张量 + 真实权重）。
- 官方长文本 6 轮连续运行的 8% 波动未定位。
- stage1 放 CPU 是否值得（会把负载从 Metal 挪到 CPU，跨设备不做等价承诺）。
- Metal allocation 采集、Instruments profile 仍未做。
- device 端 top-k 未测收益。

## 九、建议的下一步顺序

1. **先做融合 SDPA**：收益最大且有现成 Metal kernel。
   验收口径：真实权重逐阶段数值对比 + 无插桩 ABBA + WAV 摘要一致。
2. **再做融合 rms_norm**：收益已被真实模型消融验证（-18.7%），但必须先过 HF parity。
3. 然后处理 mask 广播（若 SDPA 已覆盖则跳过）。
4. 最后评估 device 端 top-k。
5. 以上都不改变时，再考虑对齐官方的 stage1 设备放置与切分粒度。

## 十、复现方式与原始数据

本轮新增的诊断入口（默认关闭）：

- `OMNIVOICE_SYNC_PROBE=1`：逐阶段 GPU 同步计时 + 金丝雀 GEMM + 段长打印。
- `OMNIVOICE_ABLATE=rms,mask`：真实模型消融（会改变数值，禁止用于验收）。
- `cargo run --release -p talechime-omnivoice --features metal --example gemm_probe`：
  逐算子微基准。

原始数据（忽略目录）：

- 官方：`target/perf/official-omni-mps-short.json`、`official-omni-mps-long.json`。
- 我们的探针输出：`target/perf/omni-syncprobe/{lenprobe,s128,longprobe,ab_base,ab_rms,ab_mask,ab_rms_mask}/`。
- 官方 profiling 输出（短/长）：本轮会话内重跑，逐行 JSON 见记录正文表格。

官方脚本保存在本地盘副本里：`bench_macos_mps.py`、`profile_forward_mps.py`、
`bench_gemm_mps.py`。这三个脚本属于测量工具，不是官方仓库的一部分，
记录时与官方 revision 分开标注。
