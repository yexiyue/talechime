# macOS Accelerate / AMX

OmniVoice 的真实输入归因、受限 GQA/投影/卷积布局及失败候选见
[CPU 案例](omnivoice-cpu.md)，不要将 Metal 的收益直接外推到 CPU。

## 先确定调用路径

本 skill 中的 AMX 优化入口主要指 Apple Silicon CPU 的 Accelerate BLAS 路径，
不是 Metal GPU、Neural Engine，也不是直接编写 AMX 指令。
启用 Candle accelerate feature 能说明选择了系统 BLAS 接线；不能仅凭 feature 或
吞吐改善证明某个 shape 实际由 AMX 执行。具体硬件分派随芯片、系统、dtype 和形状变化，
需要相应 profiler/实现证据；没有证据就称作“Accelerate CPU 路径”。

检查当前 Candle 的 CPU matmul、BLAS 支持 dtype 和实际加载/链接。F32/F64 matmul
路径的收益不能直接推广到量化 matvec、低精度、逐元素算子或 Metal。
不要混淆 Apple Accelerate framework 与 Hugging Face 的 Python accelerate 包。

在 macOS target 依赖上激活，而非把 Apple framework feature 无条件加给所有目标。
例如目标项目已有 workspace Candle 时：

```toml
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { workspace = true, features = ["accelerate"] }
```

这是接线示例，不是自动启用默认项的授权。Cargo feature 在同一构建中会合并：某个
消费者启用 accelerate 后，共用 Candle 的其他 CPU 模型也可能改变数值路径。
单包测试不足，增加实际产品 feature 组合的 CPU 回归与最终链接验证。

## 线程与实验矩阵

分别比较纯 Rust GEMM 与 Accelerate，明确 dtype、线程配置和相同输入；加载与首次
BLAS 初始化分开记录。CPU 控制组件输入应固定，端到端还要报告 token 数是否改变。

Rayon 与 BLAS 可以形成嵌套并行。区分 RAYON_NUM_THREADS、BLAS 自身线程策略和应用
并发；先测默认策略，再改变一个维度。VECLIB_MAXIMUM_THREADS 的配置效果按目标
系统/BLAS API 确认，不能把限制 Rayon 当作限制整个 BLAS。

CosyVoice3 M4 Pro 案例中，默认 BLAS 策略配 Rayon 8 优于其限 BLAS 线程方案，
另一次不约束 Rayon 的组合明显变慢。这只支持调查争用，不支持“永远设 8”或
“BLAS 永远不能限线程”。使用前核对当前 Apple threading API 的适用性。

VoxCPM AudioVAE 的 macOS 回放试验说明，CPU/GPU 统一内存不消除设备边界：应把
Metal patch 完成、F16→F32、CPU decode 和 PCM 就绪全部放进混合路径的计时。该试验把
真实 Metal patch 与生产 PCM 一次性落盘，普通 CPU 与 Accelerate CPU 用独立
`CARGO_TARGET_DIR` 重放；Accelerate 使 CPU F32 时间减少 37.59%，但与 Metal F16 的
两进程范围仍重叠，未创建 worker 或流水线。串行替代无优势不能排除与剩余 GPU 工作重叠的流水线；需再测 CPU 每帧成本与剩余 GPU 工作，并验证并发资源争用，再考虑
Rayon 矩阵；不能把 CPU 内部加速等同为端到端协同收益。

## 数值与结论边界

更换 BLAS 可能改变归约、FMA 与舍入，先按数值变化方案处理。不要用同 token 数、
局部容差或有限样本 hash 证明所有输入严格等价。TTS 的 F0/相位和迭代采样可能放大
局部误差；用户接受某例 CPU 默认切换也不代表已完成全部试听或授权其他模型同样切换。

参考：Apple [Accelerate 概览](https://developer.apple.com/accelerate/)、
[BLAS_THREADING](https://developer.apple.com/documentation/accelerate/blas_threading)。
这些资料支持 BLAS/线程接口的判断，不证明某个 Candle 算子实际用了 AMX。

## Kokoro：先测 GEMM 前的窗口展开

M4 Pro / Candle 0.11 fork `884b359` / CPU F32 的 Kokoro 案例中，普通卷积约
6.14 s，其中 im2col 分配/逐点展开约 4.71 s，Accelerate GEMM 约 1.27 s。
已经接通 BLAS 时，仍应分开测复制、窗口、bias 初始化与 GEMM，不能把整个卷积
时间都归于矩阵乘法，也不能仅凭源码中的权重复制就认定它是主因。

本例保留原有收缩次序、SGEMM M/N/K 与 beta=1，只把 stride=1 的每个 tap 时间行
改为连续片段复制和 padding 清零；1×1/stride=1 输入本身就是同布局窗口，直接复用。
这与改变 GEMM 几何的布局优化不同，仍须覆盖非对齐长度、多 batch、非连续输入、
dilation/stride，并验证真实 PCM。宽卷积仍物化完整窗口，不能称为零复制。

归一化可先尝试只并行独立行，行内保留原归约和 affine 次序；避免为提速改变 F0
敏感的舍入。此例保留小张量串行分支，保持模型/设备对象在 owner，仅把原始切片
交给同步 Rayon 任务。组合后公共 Engine 约减少 48.5% 耗时，PCM 不变；不是对其他
模型、线程配置或平台的收益承诺。方法、各候选交错数据与回归见
[Kokoro CPU 优化记录](../../../docs/records/kokoro/kokoro-cpu-optimization-2026-10-27.md)。

## Kokoro：热点迁移后的存储借用与融合

窗口展开改进后须重新归因。此例剩余 Snake 约 0.73 s、转置卷积约 0.48 s，
InstanceNorm 约 0.42 s（含它的 AdaIN 约 0.59 s）；LeakyReLU 仅约 0.015 s。
不能继续按优化前热点排名决定工作，也不要为很小的算子引入专用框架。

- 连续 CPU F32 Tensor 可在 `storage_and_layout` 的读锁存活期间借用切片，
  注意 contiguous offset；同步 BLAS 返回后才能释放。非连续布局继续物化。
  不必先建立跨请求权重缓存；后者还会引入失效、设备和所有权问题。
- 融合先核对实际原语。Candle CPU `Tensor::sin` 使用向量 `Sin::f32_vec`，
  Accelerate 配置下不能用标量 `f32::sin` 代替并宣称逐位相同。
  本例先乘 alpha、调用同长度向量 sin、再保留 square/divide/add 舍入。
  首版额外复制输入，整链只小幅改善；改为带读锁借用输入并复用输出缓冲。
- AdaIN 可去掉 Tensor 中转及广播中间数组，保留 InstanceNorm 的统计、
  learned affine、style gamma 加一、乘法和加法各自的舍入，不合并 affine 系数。
- 转置卷积只在独立输出行之间重排工作，沿用 Candle 原 `VecOps::vec_dot`
  和每个输出的 kernel-tap 累加次序。先核实后端分支：padding=0 的 COL2IM/GEMM
  与带 padding 的 dot 路径不同，不能无条件替换。边界单元测试还需区分参考实现
  本身的无符号 extent 下溢和候选的数值差异。

这些是 M4 Pro / `884b359` / F32 的实现经验，必须配合完整 PCM、非连续布局、
偏移、多 batch 和重复/取消复用验证。正式交错数据与最终保留方案见
[Kokoro 同机官方对照](../../../docs/records/kokoro/kokoro-cpu-parity-2026-10-27.md)。

本轮正式交错结果：长文原生 4.7264 s / 官方组合 4.3347 s，差 9.0%；短句
0.4630 / 0.4192 s，差 10.4%，后者尚未满足 10% parity 门槛。公共 Engine 本轮
RTF 0.11006→0.09623，耗时再减约 12.6%；组件边界和 Engine 边界分开报告。
48 份正式原生 PCM、10 份 CPU/Metal 官方夹具输出及两种 Engine 模型均保持
各自基线的完整 PCM 字节；不是跨框架逐位相同的声明。

## Kokoro：缓冲、保序多行统计与并行向量原语

后续实验继续固定 `884b359` / M4 Pro / F32。源头 profile 显示激励信号只有
约 0.034 s / 4.095 s 声学时间，完全消除也不足 1%；先用这个收益上限排除低占比
方向。普通卷积仍约 2.013 s、Snake 0.500 s，是更值得测的候选。

- im2col 可按模型所有权复用一个高水位 Vec，每次完整重建当前使用区间，包含
  padding。用 NaN 污染旧缓冲，交错大小、stride/dilation、音色及取消复用，再检查
  完整 PCM；不要把上次数据作为输入。按需 reserve_exact 避免几何扩容过量保留。
  本语料实测 decoder/encoder 合计保留约 297.3 MiB，必须同时报告容量与进程峰值
  RSS；它们不是相同指标，释放模型也不代表分配器立即归还全部物理页。
- AdaIN 可合并逐元素遍历，仍依次完成 learned affine 和 style multiply/add。
  统计若受单条 F64 累加链限制，可交错四个独立行的累加器，每个行内索引仍严格
  递增，不拆分归约、不合并部分和。不要把它写成「允许改变归约顺序的 SIMD」。
- 向量原语保留后还可测调度。本例对至少 256K 元素的 Snake sin 采用 64K 分块，
  每块仍调用 Sin::f32_vec；不能据此改用标量 sin 或推广到所有向量库/版本。
  单元覆盖整块及单元素尾部，真实 CPU/Metal 夹具与完整 PCM 再逐字节比较。
  同步 Rayon 任务只接收原始切片，Tensor/设备对象及读锁仍由 owner 持有。
- 线程扫描也要覆盖官方。本例官方 torch intra=8 比 4 明显更快，原生 12 线程
  反而更慢；不能沿用低线程官方成绩宣称超过。最终新旧原生都用 BLAS/Rayon=8/8，
  官方 intra=8/inter=1；不在库中强行设置全局环境或线程池。
- 断盘、睡眠或异常中断的运行整段作废并保留原因，恢复后重新完成交错序列；
  不把中断期间的墙钟时间混入均值，也不只删除一个慢轮次。

本轮最终数据、内存对照、门禁及全部条件见
[Kokoro 缓冲与向量优化记录](../../../docs/records/kokoro/kokoro-cpu-buffer-vector-2026-10-27.md)。

本轮八线程正式 ABCCBA：长文原生 RTF 0.06522、官方 0.06637，原生均值快约
1.7% 且两进程均值范围不重叠；短句 RTF 0.07010 / 0.06944，原生慢约 1.0%，
范围重叠。公共 Engine ABBA 从 0.08486 降至 0.07259，耗时约减 14.5%。这是本机
有限语料的结果，不称作统计显著性证明。组件进程峰值 RSS 约 3009–3013 MiB →
2512–2513 MiB：复用容量增加不能直接推断进程峰值增加，也不把组合峰值差单独
归因于某一个缓冲优化。48 份正式 PCM、十组数值夹具及两种 Engine 模型均保持
各自原生基线的完整字节；没有放宽 40/60 dB 数值门槛。
