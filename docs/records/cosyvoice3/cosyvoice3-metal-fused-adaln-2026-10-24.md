# CosyVoice3 DiT 逐步在位归因与融合调制 LayerNorm（2026-10-24）

official-parity goal §三.2 第四轮。前一轮修正五桶后 Flow 占 62%，但其内部
"块内 GEMM ~1.5s vs 实测 ~2.4s"的 ~0.9s 差值悬而未决。本轮以 release 构建
单步在位探针归因，锁定并消除 **手写 7-pass LayerNorm 链**，ABBA 验收。

## 一、逐步在位探针（`CV3_DIT_PROFILE=1`，dit.rs 内建，零开销门控）

`forward_prepared` 六区计时（embed/adaln/attn/门控链/ffn/out + 整步），
**每区先 `synchronize` 再停表**（前两轮教训：异步队列吸收）。release 构建、
生产路径、最终块 seq=534、22 层 × NFE=10，每步稳定值：

| 区 | ms/步 | 每块 | 说明 |
| --- | ---: | ---: | --- |
| attn | 119.6 | 5.4 | qkv/out GEMM + sdpa + rope + split，GEMM 主导 |
| ffn | 86.4 | 3.9 | 2 GEMM + gelu |
| **门控链** | **54.6** | 2.5 | 残差×2 + **ff_norm 手写 LN** + scale/shift 2 pass |
| **adaln** | **38.6** | 1.75 | **手写 LN（7-pass）** + scale/shift 2 pass |
| embed | 23.9 | — | Linear + **分组因果卷积 k31/groups16**（每组循环病） |
| out | 1.7 | — | — |

**派发缝隙 = 0.0ms**（区和=步墙钟）——三候选中"派发批化"出局。归因结论：
两处手写 LN（每 block 2 次）+ 广播 pass 是差值主体。手写链
`mean→sub→sqr→mean→+eps→sqrt→div` = 2 归约 + 5 全宽 pass，而 candle-nn 有
融合 `ops::layer_norm` CustomOp（cpu+metal 双实现）。

## 二、实现（`modulated_layer_norm`，embeddings.rs）

- 融合内核要求 1-D alpha/beta，而 AdaLN 调制是**逐 batch** 的（CFG 双分支
  t 不同）→ **按分支调用**：`scale.narrow(b).reshape(dim)` 折入 alpha/beta，
  LN+调制一次内核完成；B=1 免 cat。
- **仅 Metal 走融合**；CPU 分支保留参考两遍链（CPU 锚位级不变，沿 sdpa
  dispatch 先例）；`block_detail` 诊断路径不动。
- 数值注意：融合内核方差是**一遍式**（E[x²]−mean²），参考链两遍式——f32
  下 ~2e-7 相对差（门控证实），远小于已接受的非位级包络（F0 混沌项 1e-3+）。

## 三、数值门（`dit::tests::modulated_ln_matches_reference_metal`，ignored）

x ~ N(0.3σ, σ)（σ∈{1,8}，带均值偏移）、scale/shift ~ N(0,0.5)、
(2,{534,128},1024)：Metal 融合 vs 手写链 **rel 1.8–2.0e-7**（阈值 1e-5），
CPU 分支即参考链（rel=0，自校准）。全套 36+36 测试绿、clippy 0、
attention 探针门全部原值通过（LN 不在注意力链内）。

## 四、效果与 ABBA

逐步（同机同温内对比）：adaln 38.6→**15.1ms（−61%）**、门控链 54.6→39
（−29%），步墙钟 −12%（绝对值被机身热态漂移掩盖，见下）。

**ABBA 两轮被机身热态噪声否决的过程（留档）**：A1B2B3A4… 顺序下槽位间
热漂 ±25%（B3 1.68 峰值、B6/B7 却 0.983），区间重叠判 NO（首轮 −13.9%、
冷却重跑 −10.0%）。**改 ABABABAB 严格交替**（每对共享热态）：A 槽超稳
（1.355–1.388），B4 首跑被外部干扰污染（轮间 1.30–1.40 乱跳，其余槽
±0.005），单独重跑后：

| 臂 | 均值 | 区间 | 成对 Δ |
| --- | ---: | --- | --- |
| A（转置轮） | 1.3651 | [1.3550, 1.3880] | — |
| B（+融合 LN） | **1.1189** | [0.9815, 1.1650] | 中位对 −14.1%，整体 **−18.0%** |

**区间不重叠，接受。** 首 PCM 2.33→2.04s。证据
`target/perf/cv3-ln-abba2/`（含被否决的 cv3-ln-abba/ 两轮与 B4 污染轮）。

**机身双态发现**：持续负载态 B=1.165，短空闲后冷态 B=**0.982（<1.0！）**
（B4 重跑 5 轮 0.980–0.983 极稳）。即 RTF<1.0 在冷态已达；持续负载口径
（final-path 协议：3 进程×6 轮连跑）仍 1.165，需再 −~1.2s。热态差异属
机身功耗管理（持续 GPU 负载降频），非代码可控。

## 五、下一候选（已在位探针定量）

1. **门控链剩余 39ms/步**：4 个广播 pass（(2,534,1024)×(2,1,1024)）实测
   ~0.9GB/s 有效带宽——疑似 candle 二元广播算子慢路径（与 transpose2 同族，
   fork 轮候选：UniformBlocks 化的二元内核或融合 mul-add）。
2. **embed 24–28ms/步**：分组因果卷积（k31/groups16）每组循环小 GEMM——
   skill 已登记的 grouped conv 病，可用窗口堆叠/批量 matmul。
3. attn/ffn GEMM 主导区 ~3 TFLOPS（M4 Pro 实用峰附近），tiling 余量有限。
