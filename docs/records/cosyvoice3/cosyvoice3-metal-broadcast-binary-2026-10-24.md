# candle fork 广播二元快路径（2026-10-24）

official-parity goal §三.2 第五轮（fork `15af86dd`，分支 talechime-perf，
FORK.md 已登记）。第四轮在位探针把门控链剩余 ~39ms/步列为候选，本轮
定位为 candle Metal 二元算子的广播慢路径并消除——**单元与微基准全绿，
但短句端到端 -4.5%（冷态），低于 ABBA ≥10% 门槛，暂记"已落地未单独
验收"，与 embed 分组卷积合并验收**。

## 一、病灶

`broadcast_mul((B,T,D), (B,1,D))`：candle 先把 rhs `broadcast_as` 成
stride-0 视图（strides `(D,0,1)` 或窄视图 `(6D,0,1)`），`binary()` 调度
矩阵落入 `rhs_contiguous=false` → `_rstrided` 内核——对 rhs 每元素做
多维 div/mod 索引链（binary.metal `binary_kernel_strided` +
`strided_indexer`）。门控链实测 ~0.9 GB/s 有效带宽。同病：LinT2d 的
bias 加法 `(D,)→(B,T,D)`（qkv/out/ffn 每块 5 次）与 AdaLN 窄视图调制。

## 二、fork 改动（rev `15af86dd`，behavior-only）

- `binary.metal`：`binary_kernel_rbcast`——rhs 索引坍缩为
  `(i/group_len)·src_stride + i % block_len`；`init_binary_k` 宏为全部
  已有 op×dtype 追加 `_rbcast` 实例。
- `kernels/binary.rs`：`call_binary_rbcast` 启动器。
- `candle-core/metal_backend` `binary()`：lhs 连续、双 f32 3-D、rhs
  strides `[s,0,1]`（s%D==0，含窄视图）或 `[0,0,s]` 时走新内核；内核
  读的偏移就是被校验 strides 的直接推论（构造性正确）。
- 门：`binary_rbcast_matches_cpu_bitwise`（4 形状 × mul/add × 两种广播
  + 窄视图情形，to_bits 逐位相等，一次通过）+ 微基准
  `broadcast_mul_stage_timing`：**bmul_rowview 0.40ms / bmul_row
  0.21 / bmul_bias 0.19 / 同形加 0.043**（(2,534,1024)，旧路径 ~10ms
  量级，≈25×）。

## 三、端到端（ABABABAB 交替，短句）

A=第四轮二进制（fork `5a7d251f`+LN 融合）vs B=+rbcast（fork `15af86dd`）。
证据 `target/perf/cv3-rb-abba/`。冷态成对 Δ：A1/B2 **−4.8%**、A5/B6
−1.7%、A7/B8 **−4.5%**（A3/B4 对被热尖峰污染，A3 轮间 1.46–1.71 乱跳）。
整体 −7.9% 区间重叠——**不满足 ≥10% 单独验收**，但节省真实（~0.37s，
与微基准推算一致）。

- 回归：cv3 36+36、indextts 107、fork 全部 metal 门（含 transpose2/
  FIR/rbcast 位级）、attention 探针门原值、LN 门原值——全绿。
- 逐步（T=534）：门控区 39→29ms（严格版首测，热态归一后与冷态一致）、
  adaln/attn/ffn 的 bias 广播同享（微基准证实）。

## 四、剩余与合并验收计划

持续负载口径现状 ≈1.11（ABAB B 槽），冷态极值 0.98。剩余候选：

1. **embed 分组因果卷积**（k31/groups16，24–28ms/步 ≈0.4–0.5s）：病灶
   =im2col 物化 136MB（traffic 主导，与分组数无关），必须融合直卷内核
   （threadgroup 铺瓦），估 −0.4s；**与 rbcast 合并后 ABBA（合计估
   ~9%，贴线）**。
2. attn 区剩余 ~45ms/步 的 rotate 链派发（窄视+连续化+rope_i+cat ×2 ×22
   块）——已登记的"融合 split+rope"fork 内核（IndexTTS 同款）。
3. GEMM 近峰（~3 TFLOPS），tiling 余量有限。

判定风险提示：合并估 ~9% 贴 10% 线，若 ABBA 不过线则并入 rotate 融合
轮再合并验收。机身双态依旧（持续负载降频），绝对数字必须注明热态。
