# IndexTTS 宿主 custom-op 四内核轮：split+rope / convT gather / GEMM-direct / fir_up+snake（2026-10-24）

[超过官方 goal](../plans/indextts-exceed-official-goal-2026-10-24.md) 执行清单
第 1–3 项的落地轮。全部走 **talechime 侧 custom-op 宿主路径**
（`crates/indextts/src/metal_fused.rs`，cv3 模板），**零 candle fork 改动**
（goal 第 1 项预判的优先级反转成立：rope 记录写"fork 轮"时宿主路径尚未发现）。
四个内核、一个验收单元（门禁 §三.3 子阈值合并验收条款）。

## 一、内核清单与数值门（全部 ignored Metal 测试，`--test-threads=1`）

| 内核 | 替代的链 | 数值门 | 结果 |
| --- | --- | --- | --- |
| `itb_qkv_split_rope_f32` | split×3（6 次跨步拷贝）+ rope_i×2 | **位级**（vs 生产链，T=1024/37 双形状，q/k/v 三平面 + sdpa 视图路径） | max_abs 全 0 |
| `itb_convt_gather_f32` | candle `conv_transpose1d`（padding≠0 落 naive 标量核） | ≤1e-5 相对（vs `ConvTranspose1d`，六级真实形状） | rel ≤2.6e-6 |
| `itb_im2col1d` + GEMM 直出 | candle conv1d 的 strided-B bmm + 转置回拷 | ≤1e-5 相对 | **位级 0.000e0**（六级 24 组合） |
| `itb_fir_up_snake_f32` | pad 物化 + 双相位 conv + snake 五算子链 | ≤1e-5 相对（vs `fir_upsample`+`apply_activation`，六级+奇数 T） | rel ≤4.3e-7 |

rope 表达式逐字对齐 candle `ropei` 内核（cv3 已验证位级可行的同款表达式）；
convt gather 的 tap 顺序与 naive 核一致、c_in 归约序为 matmul 序；
snake 的 a/scale 在加载期预计算（与 `apply_activation` 同式）。

## 二、区域探针（ignored 测试，B=1 真实形状，M4 Pro）

1. **split+rope**（`forward_block_stage_timing_metal` 新增 `chain_fused` 段）：
   生产链 1.344 ms → 融合 **0.157 ms**（8.6×；单次 dispatch 消 8 launches）。
2. **convT gather**（六级）：340.6/350.0/94.1/49.9/28.0/16.8 ms →
   **2.73/2.83/2.61/2.17/2.05/1.95 ms**（L0/L1 各 ~125×；合计 879→14.3 ms）。
3. **conv1d GEMM-direct**：全级仅 −5~10%（例 L0 3.87→3.33，L4 4.49→3.61）。
   **归因结论：多通道 conv1d 的地板是 im2col 物化本身**（k× 写放大 + 每 call
   新 buffer 分配），GEMM 编排不是瓶颈；直卷内核（threadgroup x-tile 常驻）
   是下一步候选，本轮不做（见 §五）。
4. **fir_up+snake**（109 次调用的形状）：chain 4.5–8.7 ms → 融合
   **0.77–0.96 ms**（~8×；消 pad 物化、2 conv、5 元素算子）。

## 三、端到端（ABABABAB 8 进程 × 1 预热 + 5 热测，同机同语料同参考）

- 第一段（仅 split+rope，单轮验收）：A=rope 轮基线 mean-of-means **1.9610**
  [1.8512, 2.0754]，B=**1.7906** [1.6938, 1.9607]，**−8.69%**，区间重叠 →
  按门禁 §三.3 判**子阈值**，并入本轮合并验收。
- **合并验收（四内核全开）**：A=rope 轮基线 mean-of-means **1.8175**
  [1.7387, 1.9098]（slot 1.8795/1.9098/1.7421/1.7387），B=**1.3568**
  [1.3155, 1.4194]（slot 1.4194/1.3756/1.3155/1.3167）——**−25.35%，
  区间不重叠**（A 最小 1.7387 > B 最大 1.4194），≥10% 单轮验收通过
  （含吸收第一段的子阈值项）。
- 我方背靠背 16 连跑（单进程，1 预热 + 15 热测）：mean **1.2093**
  [1.1910, 1.2441]，首 5 轮均值 1.2042 = 末 5 轮 1.2041，**无降频爬升**。

## 三点五、与官方的同日对照（判定口径：官方只取稳态轮）

- 官方 MPS 当日单次复测（1 预热 + 5 轮）：[20.34 惰性编译, 3.57,
  **1.1528, 1.1206, 1.1082**]——较上午 parity 记录的 1.42–1.50 快 ~27%
  （官方自身同日漂移，惰性编译深浅为主要嫌疑）。
- **O-T-O-T 四窗口交错**（O=官方 1 预热+10 轮，T=我方 1 预热+5 轮，
  21:26–21:43）：O1 稳态（丢 2 轮惰性编译）mean **1.1447** [1.067, 1.205]；
  O2 稳态 mean **1.4467** [1.378, 1.629]；T1 **1.2080**；T2 **1.4051**。
  **两臂同步漂移、配对方向翻转**：窗口 1 我方慢 5.5%，窗口 2 我方反超
  2.9%；全量 T 1.3066 vs O 1.2957（+0.84%），区间重叠。
- **判定（goal §二.1 口径）= parity，未达"超过"**：我方不满足"更快且
  区间不重叠"。官方自身同日漂移 ~27%（上午 1.42–1.50、晚间两窗口
  1.14/1.45），单窗口结论不可用；同机同窗口两臂 ±5% 内互有胜负。
  起点为落后 ~15%（1.68 vs 1.42–1.50）。

## 四、插桩分布（INDEXTTS_PROFILE=1，3 轮）

融合前基线：bigvgan 4.09–4.36s（res_conv 1.44–1.62 + snake 0.75 +
up_conv 0.86–0.91 + fir_up 0.62 + fir_down 0.33）；cfm 5.02–5.20s
（estimator 4.98–5.15）；gpt 3.36–3.44s。conv 项 2.3s + snake 0.75s 与
goal §一 预估吻合。

融合后（同日复测）：**bigvgan 1.99s**（up_conv **0.023** ←0.87、
**fir_up_snake 0.213** ←1.37、res_conv 1.361 ←1.5、fir_down 0.303）；
cfm 5.11s（estimator 5.07：ffn 1.47 / adaln 0.39 / attn_sdpa 0.38 /
qkv 0.35 / attn_out 0.22 / split_rope_fused 0.25）；gpt 3.44s。
BigVGAN 一项对齐官方 1.00s 还剩 res_conv（直卷 threadgroup-tile 内核
候选，本轮不做）；**CFM −1.86s 成为最大剩余缺口，升级为下一轮主线**
（goal 清单第 4 项条款触发）。

## 五、教训

- **convT padding 陷阱**：candle Metal `conv_transpose1d` 的 col2im 快路径
  仅 padding==0；BigVGAN 全部六级 padding=(k−s)/2>0 → 全部落入 naive
  per-output 标量核（L0 0.015 TFLOPS）。同类症状：任何带 padding 的
  ConvT 都值得先查分派路径再谈优化。
- **越界写症状是数值污染不是崩溃**：gather 内核 x 维 div_ceil(256) 后未
  guard，l_out=2108 时 196 线程越界写踩相邻 buffer，表现为对拍 rel 0.85
  （CPU 标量模拟全对定位到 Metal 侧）。凡非整除 grid 必须 `if (pos.x >= n) return;`。
- **Metal 内核三元类型**：`(odd ? 0 : half_k)` int/size_t 不匹配编译失败；
  先算 `phase_row` 再乘。
- **探针先于动手**：res_conv 若直接照 cv3 直卷经验移植会重蹈 fork 轮
  `conv1d_dense` 3-4× 慢的覆辙；本轮 GEMM-direct 小收益证明 im2col 是地板。

## 六、门禁与原始数据

- talechime：fmt/clippy(metal all-targets) 0 告警；CPU 全量 108 过 + 17
  ignored；metal 108 过 + 26 ignored（新增 5 个门/探针测试）。
- 普通测试不下载模型；ignored 探针仅用随机张量。
- 数据：`target/perf/itb-splitrope-abba/`（两段 ABAB、probe 二进制、
  analyze.py）、`target/perf/itb-profile-splitrope/`（融合前插桩）、
  `target/perf/itb-profile-after3/`（融合后插桩）、`target/perf/itb-final/`
  （16 连跑 sustain + O-T-O-T 交错 t1/t2 与官方窗口日志）、
  `runs/indextts-parity/official-mps-{o1,o2}.json`（交错官方窗口）、
  `official_rtf_ab.py`（交错变体脚本）。探针输出在测试日志。

## 七、CFM/wavenet 续轮（同日晚间，goal 清单第 4 项升级主线）

### 7.1 归因：cfm.estimator 的未插桩 2.0s

补齐 `dit.wavenet`/`wn.*`/`dit.norm`/`dit.long_skip`/`dit.final_head`/
`dit.conv2` 插桩后，此前 estimator 5.07s 中 dit.* 各段仅 ~3.06s 的缺口
定位到 **WaveNet 终层 2.29s**（wn.conv k5 1.05 + wn.act 0.58 +
wn.res_skip 0.47 + wn.pad 0.14），此前完全未归因。

### 7.2 bias 丢失事故（在案未流入任何已验收发布）

审查发现 §一 的 convT gather 与 GEMM-direct 首版**漏传 bias**——真实
BigVGAN 六级 up_conv 与 res_conv 全带 bias，数值门测试自建参照时也写了
`bias=None`，故门未拦截。修复：两个宿主函数与 `ResConv`/up_conv 分派补
`bias`（GEMM 后元素加，与 candle conv 路径同构位确定）；数值门改带 bias
参照重验（convT rel≤2.0e-6、GEMM-direct 位级，计时不变）。
**教训：自建参照必须复刻真实调用的全部参数（bias/dilation/padding），
否则门只证明"错的对拍错"。** 同日 ABAB §三 的 −25.35% 是带 bug 二进制
测的性能结论（性能不变性成立），数值结论作废，已用修复二进制重跑 §7.4。

### 7.3 wavenet 融合与一次回退

- **回退**：wavenet k5 conv 走 GEMM-direct、k1 conv 走裸 GEMM 的分派
  **无收益反退**（wn.conv 1.05→1.29s）：candle conv 的批 GEMM 在 rhs
  广播时把 batch 折进 M（mlx_gemm collapse 条件：A 批连续 + B 批
  stride-0），我方"直出 (c_out,l) 布局"把广播换到 lhs 侧丢了 collapse，
  B=2 时两粒 (m,k)@(k,n) 不如一粒 (2m,k)@(k,n)。**BigVGAN 的 B=1 是
  gemmdirect 的唯一受益场景，B≥2 的 token-major 批别用直出布局。**
- **落地 `itb_wn_gate_act_f32`**：门控激活
  `tanh(v0)·sigmoid(v1)`（v=x_in+g 切片）单内核，表达式逐字取 candle
  （`precise::tanh`、`1/(1+exp(-x))`）→ 位级；替换 broadcast_add+
  narrow×2+tanh+sigmoid+mul 六内核三中间体。
- **落地 `itb_wn_res_skip_tail_f32`**：res/skip 尾双平面单内核
  （plane0=x+res、plane1=output+skip），替换 narrow×2+add×2 四内核；
  平面 1 偏移是 `b_len·H·T + plane`（首个版本写成 `plane+H·T` 在 B=2
  越界踩 plane0，数值门当场拦截）。
- 插桩：wn.act 0.576→**0.133s**（4.3×）；dit.wavenet 2.51→**1.57s**；
  **cfm.estimator 4.39s**（对官方 3.21 差 −1.18s，续轮前 −2.9s）。
  同窗口 gpt 2.96 / bigvgan 1.79（机器态较前窗偏快，交错内自洽为准）。
- 剩余：wn.conv 0.91 + wn.res_skip 0.39 为 GEMM/conv 地板（同 7.3
  collapse 论），wn.pad 0.10 可并入后续内核。

### 7.4 修复后合并 ABAB（A=rope 轮基线，B=四内核+bias 修复+wavenet 融合）

A mean-of-means **1.7891** [1.6099, 2.0413]（slot 1.6099/1.6147/2.0413/
1.8904），B **1.3390** [1.1766, 1.5150]（slot 1.1766/1.3095/1.5150/1.3550）
——**−25.16%，区间不重叠**（A 最小 1.6099 > B 最大 1.5150），验收通过；
较带 bug 版 −25.35% 性能结论复现（差值在机器态噪声内），数值侧修复。
B 最快槽 1.1766 已低于官方上午窗口下沿 1.4165——最终判定见 §7.5 交错。

### 7.5 最终判定（wavenet 续轮后，O-T-O-T 第二次交错）

- 窗口 3（快态）：O3 稳态 **1.1862** [1.1255, 1.2993] vs T3 **1.1936**
  ——官方快 0.6%，打平。
- 窗口 4（慢态）：O4 稳态 **1.3849** [1.3313, 1.4366] vs T4 **1.3280**
  ——**我方快 4.1%**。
- 聚合：T 1.2608 [1.1767, 1.3564] vs O 1.2855 [1.1255, 1.4366]，T 快
  1.92%，区间重叠。
- **判定 = parity（略领先）**：较上午第一次交错（+0.84% 落后、配对
  方向翻转）已反转为两窗口不败+聚合领先，但仍不满足"更快且区间不
  重叠"的"超过"口径。剩余定量缺口 = cfm.estimator −1.18s（ffn 1.47
  GEMM 地板为主）+ BigVGAN res_conv −0.3s（B=1 直卷候选）+ GPT +0~0.3s，
  已按 goal §四"定位后本轮不做"登记于清单第 4 项。
- 原始数据：`target/perf/itb-final/{t3,t4,o3.log,o4.log}`、
  `runs/indextts-parity/official-mps-{o3,o4}.json`（22:51–23:07）。
