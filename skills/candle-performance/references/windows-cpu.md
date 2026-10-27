# Windows CPU：先保留实际计算路径

## 跨平台基线

先检查 `cfg` 和实际算子调用。macOS 的 Accelerate affine GEMM 与 Windows 的
纯 Rust GEMM 不是同一实现；不要为复用 Mac 快路径直接更换 BLAS、GEMM 几何或
把 bias 从独立加法融合进 beta=1。即使 dtype 都是 F32，也可能改变 F0 和最终 PCM。

独立官方数值门禁、仓库内新旧输出门禁和性能门禁分别报告。基线对官方已失败时，
保存失败数值与中间特征，继续逐位验证新旧实现；提速与输出不变不代表修复了跨框架
准确性。谐波激励会放大小幅 F0 差异，应比较完整 PCM，不能只比较编码器误差。
若目标包含胜过官方或另一平台，新旧实现 A/B 只是其中一项验收。另建同机官方
对照；跨机器还要核对语料长度、分段、前端模式和计时边界。不能将 GPU 的生产
RTF 与另一机器 CPU 的组件 RTF 混排，或把对旧版的改善写成已胜过官方。
若固定官方中间特征的解码器也失败，就不能把根因归结为编码器 F0；保留该独立
门禁并继续隔离解码/激励差异，不能用新旧逐位相同替代官方准确性验收。

## 卷积窗口与调度

确认窗口构造是否串行，先固定现有 GEMM 的维度、layout、K 顺序和 bias 舍入。
在相同窗口矩阵内按独立 batch/time 行并行，保留每个元素的读取位置与零 padding；
同步任务仅借用原始 CPU storage，模型/设备对象及 storage lease 留在 owner。
小张量保留串行填充，阈值与线程数需要在真实输入上筛选。

检查 GEMM 与 Rayon 各自的线程默认值。本例 Candle fork `884b359` 的
`utils::get_num_threads()` 在未设置 `RAYON_NUM_THREADS` 时采用物理核数；
不能把显式 16 线程当成“未设置环境的默认运行”。显式环境用于控制变量，
不在生产库中重置调用方的全局线程池。

测试应包含多 batch、stride/dilation、短输入/宽 padding、偶数 kernel、非零 storage
offset 和 strided 输入/权重。算子逐位通过后，再检查长短文、速度与音色切换、同 owner
取消恢复及完整 PCM。模型内窗口缓存与并行填充是不同候选；未复用容量时不要宣称
降低了分配次数或峰值内存。

把窗口构造、GEMM 和输出布局物化分开计时。矩阵乘之后的 `transpose/contiguous`
可能仍是串行的大复制；保持矩阵几何不变，按独立输出通道和缓存时间块并行复制，
可单独验证数据移动收益。通道组不能跨 batch，短尾时间块、部分通道组、非零 offset、
strided 输入以及 NaN/负零的位模式都需覆盖。单通道等已连续的结果保留无复制路径。

外层时间分块后再调用同一 GEMM，也可能改变内核、K 分块或尾块路径。必须补充大 K、
小输出通道与很短尾块的逐位检查；两段模型通过不代表所有分块都保持输出。

有 padding 的转置卷积可能落入逐 tap 调度。可按互不重叠的输出行调度全部 tap，
继续使用原来的 vector-dot kernel 与同一行内 tap 累加顺序。不要把这项调度优化
推广到无 padding 的 GEMM/col2im 路径、groups 或未测 dtype。

## Candle feature、SIMD 与 native 库

先读锁定版本的 feature 定义和合并后的实际图。`accelerate` 是 macOS 的系统库路径；
Windows 可试 `candle-core/mkl` / `candle-nn/mkl`，但它不是 oneDNN 卷积开关。
MKL 会改变 GEMM 及部分向量数学，feature 还可能合并影响其他模型。逐项核对整数
ABI（LP64/ILP64）、OpenMP、CRT、运行时 DLL、来源/许可及干净机器运行，不能把
“静态链接 MKL”推导成“没有任何额外 DLL”。先隔离构建，再决定是否暴露产品 feature。

gemm 0.19 的 `x86-v4` 编译 AVX-512 微内核并运行时选择；它与 Candle 自有 `cpu::avx`
受编译期 `target_feature=avx2` 门控是两件事。前者已有 FMA 运行时路径时，不能说
整个 F32 GEMM 是标量。单独 target 目录测试 `+avx2,+fma`，不能先写全局配置；
它提高 CPU 硬件要求，不等同完整 x86-64-v3，也不能在发行路径使用 `target-cpu=native`。
SIMD 宽度、内核和归约变化都要比较完整 PCM；支持指令不保证本模型 RTF 变好。

混合 Rayon 与 BLAS/OpenMP 时，分别筛查线程数和等待策略，例如
`KMP_BLOCKTIME=0` / `OMP_WAIT_POLICY=PASSIVE`。保存对照后判断是否发生竞争，
不因网上报告过超订就把它当作本机退化根因，不在库中重置全局线程池。
用 oneDNN/MKL verbose 核对真实调用与版本，日志中的厂商措辞不等于实际 CPU 厂商
或已证实的 ISA。verbose 运行只作归因，正式计时关闭它。

## 官方资源与探针准备

Windows Git 的 `core.autocrlf` 会让官方导出工具的逐文件 SHA 校验失败，即使
`git diff` 看起来干净。新建官方参考 checkout 时使用
`git -c core.autocrlf=false clone ...`，保留固定 revision 的原始 LF 字节。
已有工作目录先检查本地改动；不通过归一化后再 hash 或删除来源检查绕过校验。
这只是参考资源准备，不是推理优化。

复用性能探针前，分别核对 catalog、离线资源验证与生产加载三条接线。catalog
登记了后端，不代表探针的 `verify`/`load` 已实现；失败应补齐生产清单校验，不使用
fixture 模式冒充真模型。固定官方 ID/style/激励的声学探针排除前端、随机数生成与
加载；生产 SynthesisStream 到 Completed 的计时另列，不能把两者混为同一 RTF。

具体设备、版本、保留候选与未通过项见
[Kokoro Windows 实测记录](../../../docs/records/kokoro/kokoro-windows-performance-2026-10-27.md)。
feature、SIMD、BLAS 与失败分块候选见
[第二轮调查](../../../docs/records/kokoro/kokoro-windows-cpu-round2-2026-10-27.md)，其中包含
上游文档/issue 入口及本机适用边界，不把外部模型的倍数收益当作当前结果。
