# CosyVoice3 DiT 注意力 rope_i+sdpa 融合（2026-10-24）

official-parity goal §三.2 第一轮：对生产流式路径 Flow/DiT 做 Metal kernel 级逐算子
profiling，归因组件差距，落地第一批优化候选并以 ABBA 验收。方法沿用 IndexTTS
已验证的「ignored 固定形状探针 + 数值门 + ABBA」管道。

## 一、探针与归因（`flow::dit::tests::attention_stage_timing_metal`，ignored）

真实几何：B=2（CFG）、heads=16、dim=1024、head_dim=64、ff inner=2048、chunk=50，
流式形状 T∈{128,512}；权重/输入 randn（x σ=1/32 → 投影后 σ≈1，与真实 post-AdaLN
量级一致）；rope 表来自真实 `RotaryEmbedding`，mask 来自生产 `chunk_attention_mask`。
计时 warmup 5 + 50 轮、循环内 `synchronize`（debug build，GPU kernel 时间主导）。

T=512 逐算子（单 block）：

| 段 | 旧 | 候选 | Δ |
| --- | ---: | ---: | --- |
| qkv_gemm_x3 | 2.15 ms | — | 健康（~3 TFLOPS） |
| rope_old_x2 → rope_new_x2 | 0.81 | 0.36 | −55% |
| split_x3（reshape+transpose+contiguous） | 1.57 | — | 待 fork 融合 |
| attn_manual_mask → attn_sdpa_mask | **5.05** | **0.64** | **−87%** |
| out_proj / ffn | 1.29 / 2.47 | — | GEMM 健康 |
| attention 整体 | 10.07 | 5.45 | −46% |

**病根 = 物化注意力**：手写 q@kᵀ 物化 [2,16,T,T]（T=512 时 33.5 MB），mask
broadcast_add + softmax 再各读写一遍，单 block ~200 MB 流量；sdpa 融合内核直接
−87%。rope 旧链（广播 mul/rotate_half/cat 五次半宽 pass + 全宽 cat）次之。T=128
时 mask 项 0.37→0.10，量级一致（T² 物化）。

## 二、语义侦察修正（重要）

- `RotaryEmbedding::new(cfg.dim_head)`：**partial RoPE** —— 只旋转扁平
  (B,T,1024) 的前 64 通道（reshape 后即 head 0），其余 960 通道直通；表为
  pairwise 重复 [1,T,64]（cat 于新增尾轴），rotate_half 为交错配对。
  因此 **rope_i 不能对 (B,H,T,D) 全头应用**（各头频率不同），只能对 64 宽切片以
  (B,1,T,64) + (T,32) 唯一频率表应用。此前侦察记录「rope_i 可全维迁移」是误读，
  本轮探针数值门（rel 6.9e-8）修正之。
- rope_i 唯一频率表提取：`dup.reshape(1,T,32,2).narrow(-1,0,1).squeeze…contiguous()`
  （每对取偶位）。sdpa float mask 语义 `scores += log2(e)·mask`：0/−inf 加性
  精确；mask 必须精确 [B,heads,T,T] 同 dtype（`prepare_solve` 每次 solve 展开一次，
  T=512 时 33.5 MB，10 步 Euler 摊薄）。softcap=1.0 为内核约定的"禁用"。

## 三、数值门（全部通过）

| 门 | rel | 阈值 |
| --- | ---: | ---: |
| rope_new vs rope_old（同输入同表） | 6.9e-8 | 1e-6 |
| sdpa vs manual（无 mask） | 1.6e-6 | 3e-6（T=512 分母 512 项的自然舍入底） |
| sdpa_masked vs manual_masked | 1.2e-6 | 1e-5 |
| 生产 `Attention::forward` 新旧两分支 | 1.0e-6 | 1e-5 |

探针量级教训：σ=1 输入使 post-scale scores 达 σ≈10³，ULP 级求和序差异翻转
近并列 softmax 项，"rel 1.8e-4"是量级伪影非语义错误；改 σ=1/32 后门即有分辨力。

## 四、生产改动（`crates/cosyvoice3/src/flow/dit.rs`）

- 新增 `RopeTables`：`prepare_solve` 一次构造；Metal 附带 (T,32) 唯一表，
  CPU 保持 pairwise 重复表。
- `Attention::forward` 增加 `is_metal` 分支：切片+`rope_i`+cat 重组 → sdpa
  （含 mask）；CPU 与未展开 mask 的诊断路径（`block_detail`）走原链**逐位不变**，
  CPU 数值锚不受影响。`PreparedDiT` 新增 `mask_bias_heads`（Metal 每 solve 展开
  一次）。
- 测试：36/36 CPU 全绿；clippy 0；`cargo fmt` 已过。全 F32 精度不变
  （仅浮点求和序差异，见 §三）。

## 五、ABBA 验收（生产流式最终路径，短句）

协议与 2026-10-24 最终路径复测一致：`performance_probe`、native_stream、
`production_synthesis_stream_consumer_start_to_completed`、seed 42、custom:bench、
corpus sha `f29b49d8…`、RAYON=12；8 槽 A1 B2 B3 A4 A5 B6 B7 A8 串行
（1 预热 + 5 热测/槽），release 双臂
（A=`ac38b8c9…` HEAD dit.rs / B=`655901dd…` 本轮改动）。
证据：`target/perf/cv3-attn-abba/`（request-{slot}.json、{slot}/stdout.jsonl）。

| 臂 | 均值(槽均值) | 区间 | 首 PCM |
| --- | ---: | --- | ---: |
| A（旧） | 1.4760 | [1.4150, 1.6542] | 2.49 s |
| B（新） | 1.2911 | [1.278, 1.3298] | 2.16 s |

**−12.5%，区间不重叠，接受。** 无噪声槽稳定值 1.415→1.278（−9.7%）；B7 末轮
1.535 与 A8 全槽 1.654 为尾部系统噪声（仍不重叠）。臂内 6 轮输出 sha 一致（A、B
各 1 个 sha，两臂间不同属预期浮点路径差异）。

## 六、对账与下一步

- 每步节省对账：attention 单 block −4.6 ms × 22 block × ~10 Euler 步 ≈ −1.0 s，
  与端到端 10.04→~9.0 s 完全吻合 —— Flow/DiT 内的注意力优化已**足额兑现**。
- 流式 RTF 1.29 仍 >1.0：剩余时间在 Flow/DiT 之外（LLM AR token 生成、
  HiFiGAN、逐段零样本开销），Flow 内尚余 split_x3（1.57 ms/block，需 fork
  融合 transpose+head0-rope 内核，与 IndexTTS follow-up 同族）。
- 候选排序：① 流水线级 profiling 定位 LLM/vocoder/分段占比（下一步）；
  ② fork 融合 split+rope 内核；③ CPU 组件差距 4.5–5.6× 受控复测
  （accelerate 启用后旧数字已过期，脚本 runs/cv3-parity/）。

## 七、流水线级 profiling（同日后续轮）

新增 `streaming.rs` 内建分段计时（`CV3_STREAM_PROFILE=1` 启用，禁用零开销；
prefill/llm/flow/hift/other 五桶，逐块累计行 + 会话汇总；生产计时契约不含
它）。流水线严格串行（LLM 步→flow 重解→HiFT 重解码），墙钟归属无重叠。
探针 = ABBA B 臂 + 本插桩，`target/perf/cv3-attn-abba/profile-{short,long}/`。

| 语料 | 墙钟 | LLM | Flow | HiFT | 会话外（逐段前端/条件） | 音频 | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 短句（3 块/1 会话） | 8.93 s | 1.93 s (22%) | **4.78 s (54%)** | 2.18 s (25%) | ~0.04 s | 7.10 s | 1.278 |
| 长文（每轮 10 会话×3–6 块） | 106.0 s | 18.9 s (18%) | **58.4 s (56%)** | 27.2 s (26%) | ~1.6 s | 61.66 s | **1.72** |

- **长文 RTF 2.012→1.72**（旧臂未复测，按 −12.5% ABBA 推算一致）。
  逐块 flow 增量 1088→+1487→+2209 ms（短句）/ 至 +4.2 s（长文尾块）——
  O(n²) 前缀重解 + 整 mel 重解码（**官方语义**，模块文档明示不得改增量
  缓存）。
- NFE=10（yaml `flow.cfm.n_timesteps`，官方同值）——无步数捷径。
- Flow 内部预算（T=512 单 block，§一）：GEMM 5.9 ms（69%，健康）、
  split/转置拷贝 ~2.1 ms（24%，**8 GB/s 级非连续拷贝 = 下一个病灶**）、
  sdpa 0.64、rope 0.36。×22 block ×10 步。
- 逐段零样本开销实证：长文 10 会话共 ~1.6 s 会话外（≈160 ms/段，参考
  特征重提取为主），占比 ~1.5%——**不是**主要矛盾。

### RTF<1.0 的剩余路径（短句需再 −1.7 s）

1. **fork 分块转置内核**（可融合 head0-rope）：split_x3 1.57→~0.3 ms +
  sdpa 出口转置 ~0.5→~0.1 ms，估短句 −0.9~−1.1 s；与 IndexTTS follow-up
  同一内核家族（那边已登记 FORK 轮）。**病灶已定量**（探针新增
  `tpose_copy` 段）：单次 (2,16,512,64)→转置连续化 4 MB 拷贝 0.69 ms
  ≈ **11.6 GB/s 有效带宽**（M4 Pro ~4%），每次 attention 共 4 次此类
  拷贝（split×3 + 出口×1）≈ 剩余 Flow 的 ~30%；分块转置内核可达
  ~50–100 GB/s，单拷贝 −0.5 ms/block 量级。
2. **HiFT Metal 卷积轮**（先探针归因）：2026-10-17 §三 实测 Metal HiFT
   2× 慢于 Python CPU，估短句 −0.7~−1.1 s。
3. 两者合计 ≈ RTF 0.98–1.05——**紧**；需先做①再判。长文 1.72 受官方
   O(n²) 语义约束，<1.0 需语义级增量缓存（另行立项，不在本 goal 语义
   等价约束内）。

### CPU 组件差距更正（goal §三.2 第 (2) 项证据已在案）

2026-10-17 记录 §11（2026-10-18 工作，同一受控 184-tok fixture）已含
accelerate 复测：Flow/DiT CPU **13.9→4.8 s vs 官方 4.05 s = 1.19×（≤2×
达成）**，逐 token 端到端 CPU 首次反超官方（43.6 vs 45.6 ms/tok）。此前
goal 文本"受控复测未做"为引用遗漏。残余：CPU HiFT 1.03 s vs 0.326 s
（3.2×，卷积 im2col 类开销；Metal 档为快速路径，CPU HiFT 不在本 goal
点名项内，登记不优化）。
