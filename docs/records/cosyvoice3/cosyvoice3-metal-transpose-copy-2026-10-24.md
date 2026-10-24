# candle fork 转置连续化行块内核（2026-10-24）

official-parity goal §三.2 第二轮（fork 轮）：上一轮流水线归因定位的
「(B,T,H,D)↔(B,H,T,D) 转置连续化拷贝 11.6 GB/s（M4 Pro ~4%）」病灶，
在 candle fork 内以一个 behavior-only Metal 内核消除。cv3 与 IndexTTS
共用（dispatch 在 `copy_strided_src` 通用层，应用代码零改动）。

## 一、病灶与归因

- `Tensor::contiguous()` → `MetalStorage::copy_strided_src`。转置视图的
  `strided_blocks()` 返回 `MultipleBlocks`（x1 行翻转的 wrap stride 打破
  UniformBlocks 均匀性，layout.rs L249 算法实测核对）→ 回退逐元素
  strided gather 内核。
- 探针定量（前一轮 §七）：单次 4 MB 拷贝 0.69 ms ≈ 11.6 GB/s；每次
  attention 4 次（split×3 + sdpa 出口×1）≈ 剩余 Flow 的 ~30%。

## 二、fork 改动（rev `5a7d251f`，分支 talechime-perf，FORK.md 已登记）

- `candle-metal-kernels/src/metal_src/unary.metal`：
  `transpose2_contiguous_f32(rows, d4, x1x2, x2_len, s0, s2, src, dst)`，
  grid `(d/4, rows)`，每线程搬一个 float4 块（16B 对齐要求内置）。
- `kernels/unary.rs`：`transpose2::F32` 常量 + `call_transpose2_contiguous`
  （x1x2 = s0/d）。
- `candle-core/src/metal_backend/mod.rs` `copy_strided_src`：精确签名
  `ndim==4 && F32 && strides==(d1·d2·d3, d3, d1·d3, 1) && d3%4==0 &&
  两偏移 %4==0` 时走新内核；其余形状原路径不变。内核读的偏移就是被
  校验 strides 的直接推论——任何命中签名的布局都正确（构造性正确）。
- 纯拷贝无算术：位级一致由构造保证，另以 7 形状×双向（含退化 1,1,1,4）
  的 fork 内单测 `transpose2_contiguous_matches_cpu_bitwise` 门住
  （to_bits 逐元素相等）。

**调试教训**：初版内核读偏移漏加 `idx.x*4`（每 chunk 都读行首 4 元素），
位级门第一形状即红；用「值=下标」的微型用例（(1,2,2,8)）三秒定位。
位级门的分辨力就是为这种错设计的——语义错会 1e-2 量级爆红，而这类
偏移错直接逐位不符。

## 三、探针回归（`attention_stage_timing_metal`，fork 712d0017→5a7d251f）

| 段 | 旧 | 新 | Δ |
| --- | ---: | ---: | --- |
| tpose_copy（单次 4MB 转置拷贝） | 0.690 ms | **0.071 ms** | **9.7×（≈113 GB/s）** |
| split_x3 | 1.57 ms | 0.23 ms | 7× |
| out_proj（含出口转置） | 1.47 ms | 0.74 ms | 2× |
| chain_old | 1.86 ms | 0.73 ms | 2.5× |
| attention 整体（Metal 分支） | 5.45 ms | **3.53 ms** | −35% |

T=128 同向（tpose 0.196→0.059）。全部数值门与上一轮逐位同值
（rope 6.9e-8 / sdpa ~1.5e-6 / attn_total 1.06e-6）——内核只动拷贝路径。

测试：cv3 36+36（默认/metal）、indextts 107 + dit 探针全绿；indextts
6 个 ignored 失败为环境门 oracle 测试（`INDEXTTS_ORACLE_JSON`/
`INDEXTTS_TIKTOKEN_FILE` 未设），fork 前后同样失败，非本轮回归。

## 四、ABBA 验收（生产流式短句）

臂定义：A = `probe-attn`（rope_i+sdpa 轮二进制 `655901dd`，fork
`712d0017`）vs B = `probe-transpose`（同代码 + fork `5a7d251f`，
`77906f47`）。同 request/seed/corpus，8 槽 A1 B2 B3 A4 A5 B6 B7 A8。
证据：`target/perf/cv3-tpose-abba/`。

| 臂 | 均值(槽均值) | 区间 | 首 PCM |
| --- | ---: | --- | ---: |
| A | 1.4888 | [1.3662, 1.5824] | 2.42 s |
| B | **1.3139** | [1.3103, 1.3231] | 2.19 s |

**−11.7%，区间不重叠，接受。** 本轮 A 臂槽内热漂移大（A1 从 1.274 爬到
1.48；A4/A5/A8 热稳态 1.47–1.58），B 全程 1.310–1.323 极稳——热稳态
口径下 A≈1.5 vs B≈1.31（−13%）。跨轮绝对值受机身热状态影响：上一轮
B（=本轮 A）冷态曾测 1.278；本轮 B 热稳态 1.311、A 冷态首轮 1.274
（与上轮一致，二进制同一）。**结论：热对齐时本轮 B 预计 ~1.24–1.27，
但本轮无可争议的数字是热稳态 −13% 与接受协议 −11.7%。**

## 五、对账与剩余

- 逐块节省与探针一致（split+出口 4 拷贝 ×9.7×）。短句 RTF 仍 >1.0
  （热稳态 1.31）：剩余 HiFT 25%（2.18 s）+ LLM 22%（1.93 s）+ Flow
  54% 内的不可约部分。
- IndexTTS 侧本轮只回归未单独 ABBA（其 split 链路自动受益；其下一轮
  fork 融合 split+rope 内核的预估收益已被本内核吃掉大半——登记修正：
  indextts follow-up 剩余价值主要在 rope 融合与 BigVGAN）。

## 六、HiFT 探针归因（同轮修正：「HiFT 卷积轮」的前提被证伪）

新增 `CV3_HIFT_PROFILE=1` 内建分段计时（generator.rs 七段 + f0
condnet/classifier 细分 + conv_post/istft 细分；profiling 模式下每段
先 `synchronize` 再计时——**Metal 异步队列下，不同步的"段耗时"只是
入队耗时，首个含 CPU 回读的段会吸干全部排队 GPU 工作**：首测 istft
"578ms" 实为整条 GAN 卷积链的异步执行）。短句最终块（360 mel 帧，
HiFT 总 1084 ms）真实分布：

| 段 | ms | 占比 |
| --- | ---: | ---: |
| f0（CPU f64 condnet 卷积链） | **537** | **50%** |
| res（mel 支 resblocks 融合，Metal conv） | **317** | 29% |
| sbranch（source downs+resblocks） | 140 | 13% |
| up（上采样卷积） | 66 | 6% |
| post 含 istft | 16 | 1% |

- **病灶 1 = f0 condnet**：官方语义要求 f64（torch 官方路径 .double()），
  我们已在 CPU f64 执行（f0_predictor.rs L59-61），慢的是 candle CPU
  f64 conv1d 本身（classifier 0ms、传输可忽略）。官方 CPU 全 HiFT 同负载
  仅 326 ms——差距在 candle 的 CPU f64 卷积实现。候选：fork 内 candle
  CPU conv1d f64 走 im2col+dgemm（Accelerate 覆盖 f64 BLAS）。
- **病灶 2 = resblocks/sbranch/up 的 Metal conv1d**（523 ms 合计）：
  IndexTTS BigVGAN follow-up 同族（res_conv/up_conv/snake）。
- istft/conv_post/source 几乎免费（GEMM 短小 + 本轮转置内核顺带加速了
  istft 内 (B,16,T)→(B,T,16) 相邻转置的兄弟模式之外的部分——该
  rank-3 模式未命中 rank-4 内核，仍走通用路径但量小）。
- 估算：f0 路径修复 −0.3~0.4 s、resblock conv 轮 −0.15~0.25 s（短句
  热态 1.31 → ~1.15–1.2）——**仍不够 <1.0**，还需 LLM 轮（1.93 s）。
  下一轮首选 f0 CPU f64 conv（fork，双后端无依赖），次选 resblock。

## 七、f0 轮前提证伪（同日后续：更细归因推翻 §六的病灶 1）

§六把 537ms 记给 f0 condnet（CPU f64 卷积）。进一步细分后发现 **f0 的
真算力开销只有 ~10ms**：`mv（mel Metal→CPU 传输）` 占了 596ms——
`to_device` 是同步点，**吸干了 flow solve 留在 Metal 队列里的 GPU 工作**。
"CPU 端、以设备传输开头的段"同样会吸收队列：逐段"先同步再停表"只修了
GPU 段，段首的传输仍会替前段买单（教训并入 metal.md）。

在 `CV3_STREAM_PROFILE` 的 flow 段尾补 `synchronize` 后，短句修正五桶：

| 桶 | 修正后 | §六时误记 |
| --- | ---: | ---: |
| Flow | **5.17 s（62%）** | 4.78 s（少计 ~0.6s） |
| LLM | 1.95 s（23%） | 1.93 s |
| HiFT | **0.94 s（11%）** | 2.18 s（多计 f0/istft） |
| 逐段条件/其他 | ~0.05 s | — |

- **f0（CPU f64 卷积）与 istft 双双出罪**（各 ~10ms/16ms 量级）；candle
  CPU conv1d 本就走 im2col+GEMM、accelerate 覆盖 dgemm（fork
  cpu_backend L2753/L1520 核对），原「im2col+dgemm」轮没有立项价值。
- HiFT 真实病灶只剩 GAN 卷积链（res 317 + sbranch 140 + up 66 @360帧，
  现占 HiFT 的 ~88%）。
- **主战场回到 Flow（62%）**：块内 GEMM ~6.8ms/block（T≈576，~3 TFLOPS）
  ×22×10 ≈ 1.5s，实测 chunk3 单块流 ~2.4s——差值 ~0.9s 为逐 op 派发/
  AdaLN 元素级链/每 solve 固定件，**下一轮应做 release 构建下的
  forward_prepared 单步在位探针**（此前逐算子探针是 debug 构建，CPU 侧
  派发开销失真），再定 GEMM tiling vs 元素级融合 vs 派发批化的靶点。
- LLM 1.95s（10.8ms/tok）已比官方 CPU 快 ~2×，每 token 采样回读是
  AR 依赖性固有，余量有限。
