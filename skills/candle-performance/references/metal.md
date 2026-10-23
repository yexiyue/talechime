# Metal

## 判断真实成本

先查正在使用的 Candle revision 的 Metal 算子实现和目标设备能力。
大词表排序、grouped conv 的每组循环、小张量 command encoding、布局物化和
CPU 读回后的队列同步是不同问题，分别做最小复现。
不套用 CUDA 的 fast path 或调度阈值，也不因统一内存就假定主机往返没有同步成本。

涉及 GQA/RoPE、codec 转置卷积、1×1 卷积或贪心 confidence 的布局优化时，
参照 [OmniVoice Metal 经验](omnivoice-metal.md)：先核消费者支持的 stride 和原生
快路径条件，再决定是否物化；其中收益与回退条件是版本相关案例，不是通用阈值。

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
