# CosyVoice3 DiT 第七轮：adaln 边界融合（块尾产下一块 modLN 平面）（2026-10-24）

承接 `cosyvoice3-fused-dit-customops-2026-10-24.md`（第六轮合并 ABAB −4.95%，
<10% 线登记待合并）。本轮指令：块尾内核顺带产出下一块 attn_norm 的 modLN
平面（next scale/shift 逐步预计算）、22×affine 批量化（估 −2.5~−3.5%），
合并 ABAB 验收 ≥10%。

## 一、结论先行

1. **边界 LN 融合落地**：尾内核升级为 `cv3_gate_tail_modln_f32`——除
   `out = z + gate_mlp⊙ff` 外同遍产出
   `modLN(out, scale_next+1, shift_next)` 平面（下一块的注意力输入；末块
   则为 norm_out，直接喂 proj_out，norm_out 段归零）。上一轮的门数值门扩展
   覆盖新平面（rel 1.3e-7，门 ≤1e-5）。
2. **affine 批量化未做，有意**：把 22 块 (1024→6144) 权重栈成单 GEMM 需
   复制 553MB 常驻权重（每块 25MB×22），换 ~1–2ms/步的派发开销——内存
   代价不成比例，登记不做。affine 本身 22×3 派发保留。
3. **合并 ABAB（A=LN 基线，B=rbcast+四内核）：−5.11% 区间不重叠、四对
   −4.99~−5.19%**——仍 <10% 线。逐步墙钟在热平价下 275→249.5ms
   （−9.3%，gemm 64.1 vs 64.5 同态），但 **DiT 只占 chunk 的 ~45%**
   （Flow 3.66s 中 DiT ≈3.3s，chunk 总 6.45s），端到端回收被结构份额封顶。
4. 修正后的剩余地图（seq 534 每步）：gemm 64 近峰、sdpa 45 为 candle 全
   flash 内核（在线 softmax+MMA 分块，查证非朴素实现）、embed 24（直卷已
   证伪）、rot 14.8、gate 13.1、adaln 7.3、ffn 79 近峰——可再榨的元素级
   地板合计 ~−40ms/步（≈−7% chunk），每轮收益递减。

## 二、实现要点

- `AdaLayerNormZero::mods()` / `AdaLayerNormZeroFinal::mods_raw()`：仅仿射
  （无 LN），供边界链路取原始 (B,1,6D)/(B,1,2D) 嵌入。注意 final 的通道
  序是 [scale, shift]（与 attn_norm 的 [shift, scale,…] 相反），末块打包时
  重排成 AdaLayerNorm 序并补零到 6D。
- `DiTBlock::forward_act` 重构：签名增 `carry: Option<(Tensor, &Tensor)>`
  （上一块尾平面 1 + 本块原始嵌入；把两者捆绑使非法状态不可表示）与
  `emb_next: Option<&Tensor>`；返回 `(残差承载, Option<下一块 normed>)`。
  携带时跳过 attn_norm 的 LN（仿射由调用方预算）。CPU/非 Metal 路径与旧
  算子链逐位不变（cfg 门）。
- `DiT::forward_prepared` 循环重构：每步为块 i 预取 `emb_{i+1}`（末块取
  final 伪嵌入），块间传递 `(normed, emb)`；末块直接 `proj_out(plane1)`，
  norm_out 段归零（out 区 0.5→0.3ms）。
- 两个坑：
  1. candle `Tensor::cat` 在本形状偶尔走转置回退路径返回非连续布局
     （strides (6D,12D,1)），内核要求行主序——打包后 `.contiguous()` 归一
     （连续时零拷贝）。
  2. `mods_raw` 返回 (B,1,2D) 三维，`dims2()` 运行时炸——用 `dim(0)`。

## 三、逐步区域（seq 534，热平价 gemm 64.1 vs 基线 64.5）

| 区 | LN 基线 | 第六轮(归一) | 本轮 | 备注 |
|---|---|---|---|---|
| wall | 275–278 | ~252–256 | **249.5–252** | −9.3% vs 基线 |
| adaln | 10.3 | ~11(热) | **7.3–7.5** | 块间 LN 已并入尾内核 |
| gate | 26.5 | ~15 | **13.1** | |
| rot | 18.7 | ~15 | **14.8** | |
| ffn | 85.6 | ~77 | **79.4** | 尾内核含双平面写 |
| embed | 24 | 24 | 24.1 | 不变 |
| out | 0.5 | 0.5 | **0.3** | norm_out 归零 |

seq 250 同比 wall 136.9→124–130（−9.3~9.5%）。

## 四、ABAB 验收（target/perf/cv3-dc-abba/，8 槽×5 热）

A 槽 0.9824/0.9856/0.9864/0.9860 → 0.9851 [0.9822, 0.9880]
B 槽 0.9334/0.9351/0.9355/0.9348 → 0.9347 [0.9332, 0.9362]
**B−A = −5.11%，区间不重叠，四对 −4.99/−5.12/−5.15/−5.19。**

<10% 门槛——按协议维持「已落地待合并」登记（第五~七轮累计 vs LN 基线
−5.11% 端到端 / DiT 步 −9.3%）。持续负载口径另测（§五）。

## 五、持续负载与 chunk 结构

流水线桶（B 二进制，CV3_STREAM_PROFILE）：chunk3 累计 llm 1930 / flow
3663 / hift 851 ms（总 ~6.45s，RTF 0.935 冷态）。Flow≈90% 为 DiT 步——
无"glue"大桶可省；端到端回收封顶于 DiT 的 chunk 份额（~45%）+ LLM/HiFT
不变。持续负载连跑（15×(1 warmup+5 hot) 背靠背）：见 goal §三.2 更新。

## 六、回归

cv3 36+36（默认/metal）、indextts 107、四门（含扩展后的门控区门）全绿；
clippy 0（两 feature 组合）；fmt 过。非 metal 构建路径保持逐位旧链。

## 七、下一步判定

- 单轮 10% 门槛已连续三轮不可达（−4.5%/−4.95%/−5.11%）——收益递减 +
  份额封顶。剩余已定量候选：元素级地板 ~−40ms/步合计（rot 14.8/gate
  13.1/adaln 7.3/embed 24，地板估 ~15–20ms）≈ −7% chunk，需 3+ 轮；sdpa
  为全 flash 内核无易果；GEMM 近峰。
- 持续口径 <1.0 的可行性：冷态 0.935×双态系数 ~1.13–1.18 ≈ 1.06–1.10——
  **本轮后仍 >1.0**。要收口需上述全部地板轮 + 或双态系数缓解（散热/能效
  设置类，非代码）。
