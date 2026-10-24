# IndexTTS DiT 注意力链路优化：融合 rope_i + SDPA（2026-10-24）

[官方对照 parity goal](../plans/official-parity-optimization-goal-2026-10-24.md)
IndexTTS 主线第二轮：定位 CFM estimator（名义 6.4s vs 官方 MPS 3.21s）的差距
来源为 **RoPE 广播元素级链（52%）+ 手写注意力（21%）**，GEMM 全部健康。改用
candle 上游现成的融合 `rope_i` 与 `sdpa` Metal kernel（**本轮零 fork 改动**），
**通过 ABBA 八进程验收（-15.3%，区间不重叠）**。

## 一、归因（逐算子，ignored 探针 `forward_block_stage_timing_metal`，B=2/T=1024/F32）

| 阶段 | ms/call | 说明 |
| --- | ---: | --- |
| qkv_gemm | 0.75 | (2048,512)@(512,1536) ≈ 3.5 TFLOPS，接近峰值 |
| **旧 chain（split+apply_rope）** | **8.44** | narrow×3 拷贝 1.21 + **apply_rope×2 5.87** + transpose×3 1.72 |
| attn_manual | 3.67 | kt 拷贝 + q@kt + affine + softmax + probs@v |
| ffn_gemm | 3.22 | w1/w3/w2 ≈ 4 TFLOPS |
| wo_gemm | 0.34 | |

- apply_rope 的 stride-2 视图（`(…,half,2)` narrow）使 8 个 broadcast
  kernel 全部落入非连续慢路径：4M 元素仅 ~5.5 GB/s 有效带宽（GPU ~270GB/s
  的 2%）。官方 PyTorch（gpt_fast `apply_rotary_emb` + `F.scaled_dot_product_attention`）
  为零拷贝 view + 融合 kernel。
- 每块 ×13 blocks ×25 Euler 步 = 325 calls：rope 链 5.87ms/call ≈ 1.9s，
  注意力手写多出 ≈0.57s，与 6.4−3.21≈3.2s 总差距吻合（其余为 6 次拷贝
  与 launch 开销）。
- **教训**：run 级 `INDEXTTS_PROFILE` 嵌套 scope 未及时 drop 会重复计数
  （首轮 dit.qkv "71%" 实为守卫活到函数尾）；同步开销还会把短算子放大
  ~36%。逐算子归因以独立探针为准，run 级 span 只看相对分布。

## 二、改动（全部应用层，上游公开 API，[patch] 剥离后仍可编译）

`crates/indextts/src/dit.rs` `forward_block`：

1. **split**：`narrow → contiguous → reshape(B,T,h,d) → transpose(1,2) →
   contiguous` 得 (B,H,T,D)——(h,d) 结构必须先在 token-major 显式分出再转置。
2. **RoPE**：`candle_nn::rotary_emb::rope_i`（interleaved 变体，与官方
   `apply_rotary_emb` 配对一致；Metal 走 `reduce.metal` 的融合 `ropei`
   kernel，每线程一个 pair）。`rope_tables` 改产 (T, half) 2-D 表。
3. **注意力**：Metal 分派 `candle_nn::ops::sdpa`（融合 kernel，约束
   head_dim=64/q_seq>1/无 mask/F32 全满足）；CPU 保留手写路径（sdpa 无
   CPU 实现）。

## 三、证据链

### 3.1 数值门禁

- 旧链 vs 新链（同 qkv 固定输入）：v 精确 0；q/k max_abs 7.6e-6 @ |x|≈110–118
  → **相对误差 6.5–7.1e-8（亚 ULP）**，纯乘法舍入差异。
- sdpa vs 手写 vs f64 真值（T=1024）：manual 4.03e-6 / fused 3.10e-6——
  融合路径离真值**更近**，互差 5.5e-6 与各自 f32 误差同阶，非语义差异。
- `rope_i_matches_reference`（CPU 精确 0.0；Metal 2.4e-7）、
  `split_orders_match_metal`（CPU/Metal 双端 0.0，真实形状）常驻测试。
- CPU 全量测试通过（rope_i CPU 路径 + 手写注意力分支被覆盖），clippy 0。

### 3.2 探针计时（chain 与注意力）

| 项 | 旧 | 新 |
| --- | ---: | ---: |
| split+rope 链 | 8.44 ms | **2.85 ms**（split×3 2.87 + rope_i×2 0.33） |
| 注意力 | 3.67 ms | **1.87 ms**（sdpa 1.97×） |

### 3.3 ABBA 八进程（A=HEAD 旧链 `2a58caa2`，B=新链 `95f7f80a`，1 预热+5 热测 × 8 slot 串行）

| | mean-of-means RTF | 进程均值区间 |
| --- | ---: | --- |
| A（旧链） | 1.9798 | [1.8848, 2.1636] |
| B（rope_i+sdpa） | **1.6776** | [1.6130, 1.8374] |

**-15.3%，区间不重叠**（A 最小 1.8848 > B 最大 1.8374），≥10% 门槛通过。
A 基线较 FIR 轮记录的 2.08–2.12 偏低（当日机器状态漂移），ABBA 交错内部
自洽。B 臂 rms/过零率与 A 同分布。

### 3.4 run-level 新分布（`INDEXTTS_PROFILE=1`，插桩口径）

cfm.estimator 8.71 → **6.14s**（插桩含 ~30% 同步放大）：dit.qkv 0.35 /
dit.split 0.88 / dit.rope 0.17 / dit.attn_sdpa 0.53（旧 attn_scores 1.86）/
dit.attn_out 0.21 / dit.ffn 1.55 / dit.adaln 0.53。插桩轮 RTF 2.42。

## 四、陷阱与事故登记

- **head-major 谬误**：`narrow→transpose(1,2)→contiguous` 得到的 (B,H,T)
  行内是 T 个元素、没有 d 维，再 reshape (B,H,T,D) 属张冠李戴（q 差 154，
  CPU 也错——先用 CPU 仲裁排除 Metal 嫌疑才定位到）。修复为 reshape 先行。
- **外部卷写入丢失**：/Volumes/yexiyue（APFS 外置盘）在 10:44 前后丢失约
  1 分钟的写入（stash pop 落地、probe-before、mkdir 全部回滚，/tmp 备份
  幸存）。已从备份恢复并重建双臂；此后写后 `sync` 并逐步验证落盘。
  ABBA/probe 产物均为恢复后重建所得。

## 五、剩余差距与下一轮候选

对照官方 MPS CFM 3.21s：本轮后名义 estimator ≈4.3–4.7s（由 ABBA 差值
−1.8s/轮 与插桩差值 −2.57s 推算）。候选按预期收益排序：

1. **融合 split+rope kernel**（fork 下一轮）：单核读 qkv 直写 roped
   (B,H,T,D)×2 + v，消掉 6 次跨步拷贝（split 0.88s + attn_out/rope 部分），
   预计 −0.9s；是收敛到官方水平的关键一步。
2. **BigVGAN res_conv 1.43 / up_conv 0.81 / snake 0.75**（插桩口径，共 ~3s）：
   真实 conv/激活，需逐个归因（snake 为逐元素三角函数）。
3. GPT 阶段 3.3s 与 ffn 1.55s：GEMM 已近峰值，收益有限。

人工试听验收仍挂起（全后端共通）；模型级 WAV 对照因 Metal 采样跨进程
非确定性不可用作等价门禁（见 FIR 记录），以固定输入探针为准。
