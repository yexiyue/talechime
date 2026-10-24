# CosyVoice3 DiT 融合内核第六轮：custom-op 宿主架构 + 三内核 + embed 直卷证伪（2026-10-24）

承接 `cosyvoice3-metal-broadcast-binary-2026-10-24.md`（第五轮 rbcast 已落地未
单验）。本轮指令：实施 embed 分组因果卷积（k31/g16）融合直卷 Metal 内核（估
−0.4s）并与 rbcast 合并 ABAB 验收；不过线则并入 attn 区 rotate 链融合。

## 一、结论先行

1. **embed 直卷前提被证伪**：candle 的 im2col+GEMM 在此形状上并不慢——im2col
   中间缓冲实际 L2 常驻，GEMM 计算受限。两版直卷内核（朴素逐输出、8 宽 l 分
   块）在 M4 Pro 上都比 im2col 慢 3–4×（embed 区 24→88/105ms/步）。派发已从
   fork 移除，内核+launcher+门保留为休眠基础设施（fork `884b3598`，FORK.md
   注记「NOT dispatched」）。
2. **发现并落地 talechime 侧 custom-op 宿主架构**：candle 的 `CustomOp3::
   metal_fwd` 直接给 `&MetalStorage`，配合公开的 `MetalDevice`（command_
   encoder/new_buffer_builder/kernels）、`MetalStorage::new`、
   `apply_op3_no_bwd`，融合内核可以完全住在 talechime 里，**零 fork 派发改
   动**。这取代了此前"每轮都要在 fork 里找现有算子挂钩"的约束。
3. **三个融合内核落地**（`crates/cosyvoice3/src/flow/metal_fused.rs`）：
   rope+头拆分（位级一致门）、门控残差+modLN 双平面、块尾门控加。合并
   ABAB（A=LN 轮基线，B=rbcast+三内核）**−4.95% 区间不重叠、四对
   −4.8~−5.1%**——真实但低于 10% 门槛，登记「已落地待合并」，下一轮=
   adaln 边界融合（块间 LN 并入尾内核 + 22×affine 批化，估 −2.5~−3.5%）。

## 二、embed 直卷证伪过程

- 朴素版（每线程算一个输出，c_in×k=1984 MAC）：seq 534 embed 24→88ms。
  病灶=每线程 64 行离散读（~9GB 有效读放大），memory-latency 受限。
- 8 宽 l 分块版（`acc[8]` 寄存器复用）：更差（105–112ms）。病灶=**内层循环
  界是运行期值 `j < n_out`，编译器无法展开，acc[] 溢出到 local memory**——
  Metal 经典陷阱。
- 两版都过了 CPU 参照数值门（≤1e-5，含 chunk 视图与转置视图）——门只能证
  数值，证不了性能；region 探针才是性能门。
- 修正归因：原「136MB im2col 物化浪费」估算忽略了 L2 常驻与 GEMM 计算受限。
  **教训：以 traffic 推算收益前先确认缓存驻留与算力对比。**

## 三、custom-op 宿主架构（本轮最大方法论产出）

蜡烛公开面足够把自定义 Metal kernel 全部放应用侧：

| 需求 | 公开 API |
|---|---|
| 原始缓冲 | `CustomOp3::metal_fwd(&MetalStorage, &Layout, ×3)` |
| 设备/编码器 | `MetalDevice::{command_encoder, new_buffer_builder, metal_device, id}`（`Device::Metal(d)` 可从 Tensor 拿） |
| 结果张量 | `MetalStorage::new(Arc<Buffer>, dev, el, dtype)` + 返回 Shape |
| 挂接 | `Tensor::apply_op3_no_bwd`（candle-nn rope_i 即此模式） |
| 自带源码编译 | `metal_device().new_library_with_source` + `new_compute_pipeline_state_with_function`（进程内 OnceLock 缓存，键=(name, DeviceId)） |
| 依赖 | cosyvoice3 增 optional `candle-metal-kernels`（set_param/Output）+ `objc2-metal`（仅 MTLSize；与 candle 同 pin 0.3.1，零新增编译） |

三个踩坑（都已写进模块头注释与 skill）：

1. **输出缓冲必须用 `Output::new` 绑定**：普通输入式绑定绕过 candle 的 hazard
   追踪，回读与内核竞态，读到全零（且不报错）。
2. **禁止运行期非 2 幕 `%`/`/` 分解 tid**：GPU 整数除法是仿真指令序列（~30
   周期/线程），简单内核里反客为主（微基准 4×）。改 3-D grid
   `(channels, seq, batch)` + `threads_per_grid`，全部分解退化为移位/掩码。
3. **MSL 属性参数类型必须同形**：`uint j` 与 `uint3` 混用编译报错——标量
   改从 `uint3 tpos.x` 取。

## 四、三个内核与数值契约

生产链语义（`flow/dit.rs` Attention Metal 分支 + DiTBlock::forward_act）：

1. `cv3_rope_headsplit_f32`：partial 交织 RoPE（仅头 0，64/1024 flat 通道）+
   头拆分转置，一格融合 slice-copy/rope_i/cat/transpose 四派发。旋转表达式
   顺序与 rope_i 内核逐字相同 → **位级一致**（门 `fused_rope_headsplit_
   matches_chain_metal`，T=534/128 max_abs=0）。微基准 0.097 vs 0.276ms/调用。
2. `cv3_gate_residual_modln_f32`：`z = x + gate_msa⊙attn` 与
   `modLN(z, scale+1, shift)` 单遍双平面输出 `(2,B,T,D)`——plane0=FFN 输入、
   plane1=残差基（`narrow(0,i,1).squeeze(0)` 零拷贝视图）。调制直接从原始
   `(B,1,6D)` AdaLN 嵌入按通道偏移读（2D/3D/4D），免打包内核。行归约
   threadgroup 树状（D=1024=线程组宽）；输出用 `fma(norm, sc, sh)`、
   `rsqrt(var+1e-6)` 对齐 candle layer_norm；两动量方差 vs Welford ~1.3e-7
   相对差；fma 收缩差 ~4.8e-8（门 ≤1e-6/1e-5）。
3. `cv3_gate_residual_add_f32`：块尾 `out = z + gate_mlp⊙ff`（gate_mlp @5D）。

`AdaLayerNormZero::forward_act` 增返回原始 emb（`(norm, gate_msa, shift,
scale, gate_mlp, emb_raw)`）；CPU/非 Metal 走原算子链不变。

## 五、归因与验收

attn 子拆分探针（CV3_DIT_PROFILE 扩展，先同步再停表）：seq 534
attn 128.9 = gemm 64.5 + **rot 18.7** + sdpa 45.1（含 to_out）。融合后（同日
gemm 通胀 ~1.12–1.2 归一）：gate 26.5→~15–17（−11ms）、ffn 内尾 −4~9、
rot −2~3、步墙钟 275→~252–256（−8~−10%，seq 534）。

ABABABAB（`target/perf/cv3-dc-abba/`，A=LN 轮基线探针、B=rbcast+三内核，
1 warmup+5 hot×8 槽，RAYON=12）：

- A 槽 0.9812/0.9843/0.9856/0.9847 → mean 0.9840 [0.9809, 0.9870]
- B 槽 0.9338/0.9345/0.9372/0.9355 → mean 0.9353 [0.9330, 0.9376]
- **B−A = −4.95%，区间不重叠，四对 −4.83/−5.06/−4.91/−4.99**（极稳）
- 本批为冷态口径（批间空隙）；持续负载口径需另测。低于 10% 门槛 → 按
  协议登记「已落地待合并」，不单独记验收。

## 六、回归

- cv3 36+36（默认/metal lib）、indextts 107、四门（attention_stage_timing/
  modulated_ln/两个新融合门）全绿；clippy 0（两种 feature 组合）；fmt 过。
- 非 metal feature 构建补齐 `#[cfg]` 门（本轮曾漏，只在 metal 构建下编译过）。
- fork `884b3598` = transpose2 + rbcast +（休眠）conv1d_dense；workspace
  [patch] 四行同步该 rev。

## 七、剩余候选（下一轮）

1. **adaln 边界融合**（已定量）：块尾内核顺带产出下一块 attn_norm 的
   modLN 平面（next scale/shift 可逐步预计算）+ 22×(affine GEMM+silu) 批成
   单 GEMM/步——adaln 10.3ms/步中 LN 部分可消，估 −2.5~−3.5%。与本轮
   −4.95% 合并有望过 10% 线。
2. sdpa 桶 ~29ms/步（seq 534，去掉 to_out 后）：flash 级内核=大工程，未立项。
3. embed 24ms/步：直卷已证伪；剩余为 GEMM+mish+残差，收益薄。
4. gemm/ffn 近峰（~2.3 TFLOPS 有效），不动。
