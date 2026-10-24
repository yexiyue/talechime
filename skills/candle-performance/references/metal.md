# Metal

## 判断真实成本

先查正在使用的 Candle revision 的 Metal 算子实现和目标设备能力。
大词表排序、grouped conv 的每组循环、小张量 command encoding、布局物化和
CPU 读回后的队列同步是不同问题，分别做最小复现。
不套用 CUDA 的 fast path 或调度阈值，也不因统一内存就假定主机往返没有同步成本。

**分段计时必须"先同步再停表"**：Metal 命令队列是异步的，只 enqueue 的段
（纯 GPU op 链）测出的"耗时"只是入队耗时；队列里第一个含 CPU 回读
（`to_device`/`to_vec1`）的段会吸干全部排队的 GPU 工作——cv3 HiFT 首测
把整条 GAN 卷积链 578ms 记到了 istft 头上，逐段加 `synchronize` 后 istft
实为 16ms。**CPU 端、以设备传输开头的段同样会替前段买单**：cv3 f0 段以
`mel.to_device(Cpu)` 开头，flow 留在队列里的 ~0.6s GPU 工作全记进了 f0
"CPU f64 卷积"（真算力仅 ~10ms）——归因探针要在**每段开始前**（含前段
结束后）同步，而不是只给 GPU 段收尾同步；跨 CPU/GPU 混合流水线尤其如此
（参考 cv3 `CV3_HIFT_PROFILE`/`CV3_STREAM_PROFILE` 的最终实现）。

涉及 GQA/RoPE、codec 转置卷积、1×1 卷积或贪心 confidence 的布局优化时，
参照 [OmniVoice Metal 经验](omnivoice-metal.md)：先核消费者支持的 stride 和原生
快路径条件，再决定是否物化；其中收益与回退条件是版本相关案例，不是通用阈值。

**转置视图的 `contiguous()` 是隐形慢路径**：`transpose(1,2)` 后的
rank-4 布局（如注意力 (B,T,H,D)↔(B,H,T,D)）使 `strided_blocks()` 返回
`MultipleBlocks`，回退逐元素 gather（实测 11.6 GB/s ≈ 4% 带宽）；candle
fork `5a7d251f` 起，精确 stride 签名 `(d1·d2·d3, d3, d1·d3, 1)` 的 F32
拷贝走 `transpose2_contiguous_f32` 行块内核（≈113 GB/s，9.7×）。fork 外
（upstream 0.11.0）此模式仍是慢路径——项目内若见类似形态先查布局签名。

## 排序与采样

Candle 历史版本的大词表排序曾出现线程组/索引问题；这不是所有新版本或所有 Metal
设备的永久限制。核对 kernel 的线程数、padding、真实词表和返回索引，不只测试 4/32
列；验证稀疏 logits、同分、非排序词表 ID 和过滤后至少有候选。

Qwen 曾小词表测试通过、3072 词表过滤失败，真实生成反复选 token 0 并触达帧上限。
CPU 排名策略与一次 logits 回传是其已验证修复，不是所有模型必须使用 CPU 采样。
改变 top-p 边界属于准确性修复，另立性能基线。

若 trace 的回传 span 很长，先测已经同步完成的 logits 回传。Qwen 的历史约 19 ms
主要包含前序队列等待；不能据此承诺把 12 KiB 复制改小就会提速。
查 cumsum/排序等 API 的当前实现；矩阵乘三角矩阵的 CDF 可能比顺序扫描更贵，但
更换累加顺序也可能改变 token，必须先遵守主 skill 的数值门槛。

## 卷积与布局

grouped conv 若逐通道分派，先定位真实调用几何，再评估窗口堆叠和批量 matmul。
Qwen 的局部替换仅适用于 unit stride/dilation、每组单输入、groups=channels。
验证因果左 padding、kernel 方向、bias 广播、短输入、多 batch 和浮点归约。
不合格配置明确走 stock；性能阈值按当前设备重新测量。

逐 tap 广播相加可能改变求和顺序，即使误差很小也不能冒充逐位一致。
transpose/narrow 的非连续布局只在必要的消费边界物化，实测连续化成本和下游内核
选择；删除 contiguous 不是天然优化。一次设备 tensor 简化若无明确收益，只能作为
单独的可读性改动解释，不作性能宣传。

广播元素级链一旦含 stride-2 视图（如交错 RoPE 的 `(…,half,2)` narrow），
全部 kernel 落入非连续慢路径——IndexTTS 实测 ~2% 带宽、占 block 52%。
先查 candle-nn 的 CustomOp 融合实现再自己写 kernel：`rotary_emb::rope_i`
（interleaved，Metal 融合 `ropei`，cos/sin 为 (T,half) 2-D）与
`ops::sdpa`（head_dim 32–512、F32/F16/BF16；CPU 均无实现，需设备分派）
都是上游公开 API，换调用即得融合路径。[B,T,H] 想直接转 [B,H,T,D] 时
注意：先 transpose 得到的行内只有 T 个元素、没有 d 维，必须先 reshape
分出 (h,d) 再 transpose。

`conv_transpose1d` 的 Metal col2im 快路径**仅 padding==0 分派**；带 padding
的 ConvT（上采样 (k-u)/2>0 是常态）全部落入 naive per-output 标量核——
IndexTTS L0 (1536→768,k8,s4) 仅 0.015 TFLOPS（340ms）。k%stride==0 且
stride 为 2 幂时用 GEMM+gather 分解在应用侧重建：`(l_in,c_in)@
(c_in,c_out·k)` 一次健康 GEMM + 每输出恰 k/stride 次读的散射核
（~125×，rel≤2.6e-6）。多通道 `conv1d` 换直出布局 GEMM 仅 −5~10%：
im2col 物化（k× 写放大 + 每 call 新 buffer）才是地板——**改 conv 编排前
先拿等形纯 GEMM 探针对拍**，GEMM 与 conv 的差值才是编排的可用空间。

----
## 跨实现官方对照：当日 O-T-O-T 窗口交错，配对方向才是结论（2026-10-24，IndexTTS）

官方（PyTorch/MPS）自身同日可漂移 ~27%（上午稳态 1.42–1.50、晚间同机
1.11–1.15 与 1.38–1.63 两窗口——MPS 惰性编译深浅是主要嫌疑），隔日或单窗口
读数对"谁更快"的判定完全不可用。可行协议：同日 O-T-O-T 窗口交错（O=官方
进程 1 预热+10 轮取丢 2 轮惰性编译的稳态，T=我方 1 预热+5 热测），**看配对
窗口的方向**：IndexTTS 四内核轮窗口 1 我方慢 5.5%、窗口 2 反超 2.9%（两臂
同步进入慢态），全量区间重叠 → 判定 parity 而非"超过"。官方进程加载 ~200s
是交错成本，两窗口足够翻转一次机器状态；结论必须附全部槽值与窗口时刻。

## 平台验证

依赖与源码测试都按 macOS + feature 门控。编译通过、设备枚举成功、实际 kernel
执行和端到端输出是不同层次。无 GPU 的原生 CI 不强行构造 Metal 设备。
macOS 统一内存峰值、进程 RSS 与独占 GPU 显存不可混为一个指标。

## 先重建成本模型，再选优化方向

从 CUDA 移植结论前先核四件事：派发下限、进程 CPU 占用、GGUF/权重真实 dtype、
Candle Metal 的 kernel 分派分支。

- 派发下限：极小张量的单次逐元素算子，异步流水、末尾只同步一次，取多次最优。
  它是主机端编码的下界，约 1.5 µs/算子（M4 Pro，candle 0.11）。
- 进程 CPU 占用与异步重叠：Candle Metal 默认累计算子后提交 command buffer，主机
  编码可与 GPU 执行重叠。`算子数 × 派发下限` 只给出主机编码量级，不能用
  `1 - 主机编码/墙钟` 反推精确的 GPU 带宽占比。
- 权重字节必须从文件头或加载代码核实，不能凭模块名推断：例如 VoxCPM 的
  `VoxCPM2-Acoustic-F16.gguf` 内含 F16 的 `residual_lm`，而 `VoxCPM2-BaseLM-Q8_0.gguf`
  的 Q8_0 每 32 个权重占 34 字节（约 1.0625 B/参数，不是 0.5 B）。
- `权重字节 / 模块耗时` 只能做同口径相对比较：若某个小模块算出的“带宽”高于芯片
  规格，说明存在缓存命中、计时边界或重叠，不能写成 DRAM 带宽实测。

**同一模型内两个同层形状子模块的对比最有诊断价值**：VoxCPM 的 LocDiT（`b=2,t=11`，
collapse 后 `m=22`）与 LocalEncoder（`cls + 4 patch`，`m=5`）权重形状和 dtype 相同，
但每层次耗时差约 1.4 倍。阅读 `candle-metal-kernels::mlx_gemm` 可见：`m < 16` 走
`32x32x16` tile，`m >= 16` 在 M4 Pro（`applegpu_g16s`，归入 `Max`）的 F16 `nt`
路径走 `64x32x32` tile。这类问题应归入底层 kernel tile 选择，不在模型层造特判。

Metal 有 Indirect Command Buffer，但 Candle 没有公开的 Metal 图式录制/回放边界；
在主机编码不占关键路径时，也不应把 CUDA Graph 思路直接搬到 Metal。

## thin-M GEMM 与权重布局

- 做 M 扫描（如 `m=1/4/5/11/16/22/32/44`）时，结合 `select_tile_config` 看平台期：
  `{4, 11}` 同属 `bm=32` 的 1 个 M tile，`{22, 44}` 同属 `bm=64` 的 1 个 M tile。
- **检查启发式的分界是否落在 tile 边界上**。`if m < 16 { small tile }` 这类守卫
  容易选错阈值：`16..=32` 同样只需 1 个 `bm=32` tile，却被送进 `bm=64` 多算
  32–42 行 padding。判据是 `ceil(m/bm_small) == ceil(m/bm_large)` 的区间，
  这只是 M 方向的候选依据，N/K tile、线程组和 dtype 仍须实测，不能据此断言整体零代价。VoxCPM LocDiT `m=22` 修前后 21.71 → 18.99 ms/帧。
- **不要把 `(out, in)` 权重盲目预转置成 `(in, out)` 连续**。`QMatMul::Tensor` 交给
  Candle 的是 `w.t()` 视图，`broadcast_matmul` 只在广播时才 `contiguous()`，
  该视图会作为 `b_trans=true`（`nt`）进入 `mlx_gemm`；预转置会切成 `nn` 并改变
  tile 选择，实测 LocDiT 层慢 59%（711 → 1133 µs）。
- 合成微基准常复用同一组权重并跑满功率循环，既可能命中缓存，也可能触发功耗限制：
  同一 LocDiT 层在真实模型里约 206 µs，而在探针里是 711 µs。**合成数据只用于同探针
  交错臂间比较**，不用于归因、带宽结论或对外性能数字。

## 用模块级 profile 做归因，用端到端 A/B 做验收

改动命中某个模块时，**先看该模块的时间是否按预期变化**，再看端到端：

- 模块级（`VOXCPM_BENCH_PROFILE=1` 等）信噪比高得多，能在一次短跑内确认因果。
  副作用是边界同步会放大总时长，所以它只用来归因，不用来报吞吐。
  逐算子 span 要注意守卫生命周期：guard 不及时 drop 会把后续嵌套 span
  重复计入（IndexTTS 首测 dit.qkv "71%" 即此假象），且逐 span 同步对短算子
  放大可达 ~36%——**逐算子归因以固定形状的独立探针为准**，run 级 span 只看
  相对分布。
- 若模块降了 X% 而它占帧成本 P%，端到端预期约 `X% × P%`。实测吻合
  （CFM -12.5% × 55.4% ≈ 端到端 -6.9%）说明改动确实落在预期位置；
  若对不上，说明还有别的分支没命中。
- **未被改动的模块应当保持不变**（±0.5% 内）。这是补丁作用域正确的对照，
  比"整体变快了"更有信息量。

## 官方实现不是自动的同条件 oracle

- 先记录官方源在目标设备的**实际** dtype、权重格式、公共 API 的缓存边界和输出长度。
  VoxCPM 固定官方源码在 MPS 上把 BF16/FP16 自动改为 F32；官方非流式 `generate()` 还会
  在每段内部准备参考。它与 Candle F16、已缓存参考的测量不构成 Rust/Python 的 runtime A/B。还必须核对 continuation 与隔离 identity 是否同时开启、badcase 重试是否改变实际 seed。
- `optimize()` / `torch.compile` 被调用不等于有效加速。VoxCPM 官方实现的 optimize 明确只
  允许 CUDA，在 MPS 上捕获错误后回退 eager；没有有效编译组件时，不创建 eager/optimize
  吞吐对照，也不据此迁移生产路径。
- 设备注册不足以证明生成算子走该设备。开发探针可在 BaseLM prefill/step、ResidualLM
  prefill/step、编码器、CFM estimator 与 AudioVAE decode 的实际张量边界记录 device；这种
  hook 只作路径审计，不能放进无插桩吞吐数字。

## 应用侧自定义 Metal 内核：CustomOp3 宿主路径（2026-10-24，cv3 第六轮）

融合内核**不必进 fork 找现有算子挂钩**——candle 公开面足够把它们全部放在
应用 crate 里（此路径已被 candle-nn 自身的 rope_i 采用）：

- `impl CustomOp3`，在 `metal_fwd(&MetalStorage, &Layout, ×3)` 里拿原始缓冲；
  设备从 `Tensor::device()` 的 `Device::Metal(d)` 取。`MetalDevice::{command_
  encoder, new_buffer_builder, metal_device}`、`MetalStorage::new`、
  `Tensor::apply_op3_no_bwd` 全公开。自有 .metal 源码用
  `metal_device().new_library_with_source` 编译，进程内缓存（键含 DeviceId）。
- 三个硬坑（每个都实测踩过）：
  1. **内核写出的缓冲必须用 `candle_metal_kernels::Output::new` 绑定**。
     普通输入式元组绑定绕过 hazard 追踪，回读与内核竞态——**读回全零且不报
     错**，极像"内核没跑"。
  2. **禁止用运行期非 2 幕常数 `%`/`/` 分解 tid**。GPU 整数除法是仿真指令
     (~30 周期/线程)，简单内核里反客为主（微基准 4×）。用 3-D grid
     `(channels, seq, batch)` + `[[threads_per_grid]]`，分解退化为移位/掩码。
  3. MSL 属性参数类型必须同形：标量 `uint` 与 `uint3` 混用编译报错，标量从
     `uint3 tpos.x` 取。
- 双平面输出技巧：一个内核写 `(2,B,T,D)`（FFN 输入 + 残差基），消费侧
  `narrow(0,i,1).squeeze(0)` 是零拷贝连续视图。
- 调制参数若本就是某大张量的窄切片（如 AdaLN 的 `(B,1,6D)`），让内核按
  通道偏移直读原始张量，免打包内核。
- fma 收缩：内核里 `x + g*y` 会被收缩成一条 fma（一次舍入），与 CPU 链
  mul+add（两次舍入）有 ~5e-8 相对差——数值门按 1e-6 设，别追位级。

## 用 traffic 估算内核收益前，先问缓存驻留与算力比（2026-10-24，cv3 直卷证伪）

"中间物化 N MB → 消掉它省 N/BW" 的推算曾两轮失真：
- im2col 缓冲（~136MB 估算）实际 L2 常驻、GEMM 计算受限，直卷内核（朴素与
  8 宽分块两版）反而慢 3–4×；且分块版的 `acc[8]` 被运行期循环界挡住展开，
  溢出 local memory 更慢。**先拿 region 探针确认该算子真实占比，再写内核。**
- 数值门只能证数值（两版直卷都 ≤1e-5 过门），性能门必须另做（region 探针
  或微基准交错臂）。

## 「持续负载」读数是环境变量，不是二进制属性（2026-10-24，cv3 第七轮）

同一二进制、同一 ABAB 槽协议，不同日期的"持续态"读数差 ±15%（A 基线当日
1.119–1.365、隔日 0.982–0.987）——机身降频取决于当日热背景而非代码。判定
"<1.0" 类阈值时：当日做连续负载连跑（≥15 次背靠背、~60s 满载）确认 RTF 无
爬升，并把双态幅度如实登记，而不是引用历史槽值。

另：candle `Tensor::cat` 偶尔走转置回退分支返回非连续布局（strides
(6D,12D,1)），喂给要求行主序的自定义内核前 `.contiguous()` 归一（连续时
零拷贝）。
