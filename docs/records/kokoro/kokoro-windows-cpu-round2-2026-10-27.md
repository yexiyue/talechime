# Kokoro Windows CPU：算子归因、SIMD 与 BLAS 调查（2026-10-27 至 28）

起点是[第一轮 Windows 优化](kokoro-windows-performance-2026-10-27.md)的本地候选，
基于 `0d3ce5a`，模型、Candle revision、资源和输入沿用该轮。目标是继续缩小与同机
官方的 CPU 差距；相对旧实现的提速不等于达到官方或跨机器 Mac 的验收目标。

## 先确认本机实际路径

- Windows 当前没有启用 Accelerate；它来自各模型的 macOS target 依赖，不是
  Windows 上已经启用但失效的开关。锁定 Candle 的 CPU 替代路径是 `mkl`。
- 默认 F32 GEMM 已通过 gemm 运行时分派至 FMA 内核，不能称为整个 CPU 后端标量。
  但 gemm 0.19.0 的 AVX-512 微内核受 `x86-v4` Cargo feature 保护，默认未编入。
  [版本 feature 表](https://docs.rs/crate/gemm/0.19.0/features)与本地源码一致。
- Candle 自有 `cpu::avx` 的编译期 `target_feature=avx2` 门控与上述运行时分派不同。
  `VecOps` 的点积/归约需要单独核对；编入更宽 GEMM 不会自动启用这些实现。
  [上游 issue #3707](https://github.com/huggingface/candle/issues/3707)也区分两种分派，
  但其量化矩阵问题不是当前 Kokoro F32 的热点证据。
- 官方运行日志实际显示 oneDNN 3.7.1、OpenMP 8 线程、F32 AVX-512 直接卷积、
  blocked memory layout 和 BRGEMM LSTM。不是据 PyTorch 安装了 oneDNN 就推断命中。
  日志中的 ISA 名含 Intel，不能据其字符串把这台 AMD CPU 误写成 Intel。

## 归因

在未改矩阵计算的起点上单独插桩，长文探针的完整执行序列累计宽卷积组件为：

| 组件 | 累计秒数 | 宽卷积三项合计占比 |
| --- | ---: | ---: |
| 窗口展开 | 1.5458 | 29.8% |
| GEMM | 2.2729 | 43.9% |
| 输出 transpose/contiguous | 1.3598 | 26.3% |

这些数值包含预热、热轮及恢复调用，仅用于归因，不当作正式延迟；嵌套的 linear/
conv 等计时不相加。宽卷积还不是整模型全部成本。因此“换 BLAS 就能消除差距”
不是成立的归因，输出布局物化也需要处理。

## 已筛选与撤回的候选

| 候选 | 观察 | 决定 |
| --- | --- | --- |
| gemm `x86-v4`，运行时 AVX-512 | 短文两热轮约 0.887/0.869 s，长文约 3.149/3.219 s；相对起点未证明明确收益，完整 PCM 改变 | 撤回 feature/依赖改动 |
| 交换矩阵乘方向以直接输出 channel-first | 算子逐位测试失败 | 未扩大到性能验收 |
| 512 时间步窗口分块、块内串行 GEMM | 两段 PCM 相同，但短文筛查约 0.833/0.849 s，未优于较简单的输出转置候选；后续边界检查发现风险 | 撤回 |
| 2048 时间步及内部窗口快速复制 | 短文约 1.179/1.141 s、长文约 3.354/3.271 s；小输出通道与短尾块的补充测试失败 | 撤回；未保留直接 gemm 依赖 |
| 单独简化内部窗口读取/复制 | PCM 相同，短文约 0.907/0.914 s、长文约 3.353/3.328 s，未支持保留 | 撤回 |
| 已安装 NumPy OpenBLAS 宽卷积原型 | 8 线程长文约 3.535/3.656 s；1/4 线程也未优于纯 Rust 候选，且完整 PCM 改变 | 仅保留忽略目录的调查产物 |

上述均为筛查，不混入正式均值。时间步分块会改变 GEMM 的有效维度；完整 K 顺序
相同仍不能保证各尾块采用相同内核或 K 分块。两段真实模型输出通过，不能替代
大 K、小输出通道、短尾块的算子边界检查。没有放宽逐位断言。

OpenBLAS 原型复用了本地 NumPy 2.4.6 的 OpenBLAS 0.3.31.188.0 / ILP64 /
DYNAMIC_ARCH，实际配置字符串为 SkylakeX。不是“所有 Windows OpenBLAS 都更慢”的
结论，也没有把其 DLL 纳入产品。首次原型逐调用加载/卸载 DLL，测量受构造/线程池
生命周期干扰，已中断并排除；后续改为整个独立探针进程只加载一次。

## 保留的输出布局优化及正式对照

在 Windows 私有 `cpu_conv::ChannelFirst` 中按不跨 batch 的独立输出通道组调度，
每组再按 64 时间步复制，减少串行跨步物化。通道组最多 16 个通道，并按当前 Rayon
池大小分组；不修改全局池。单通道或单时间步的已连续结果仍走原无复制路径。
输入、权重、GEMM 几何与 K 顺序、独立 bias 加法保持原路径，没有引入窗口缓存、
直接 GEMM 依赖、BLAS DLL、全局 Rust flags 或新的产品 feature。

正式声学按原版→候选→官方→官方→候选→原版；生产整链按 ABBA。短、长文分别
运行独立进程，各 1 预热 + 5 热轮，全部保留，共 20 进程、100 热轮、20 预热。
每个版本/输入/入口各两个进程，先算进程均值再平均。硬件与资源同第一轮，全部
Rayon=8；官方 torch intra=8/inter=1，OMP/MKL/OPENBLAS=8，关闭 verbose。先完成
编译再串行计时，未锁频；进程均值范围不是置信区间，也不是统计显著性证明。

| 生产整链 | 原版秒数 / RTF | 候选秒数 / RTF | 原版 / 候选进程均值秒数范围 |
| --- | --- | --- | --- |
| 短文，2.760 s 音频 | 0.8790 / 0.31848 | 0.8537 / 0.30932 | 0.8742–0.8838 / 0.8216–0.8859 |
| 长文，10.920 s 音频 | 3.2121 / 0.29415 | 2.9548 / 0.27059 | 3.2041–3.2202 / 2.9466–2.9631 |

本次生产长文耗时减少 **8.0%**。短文均值减少约 2.9%，但范围重叠，不宣称明确
提速。新旧完整 PCM 分别在各自输入内逐字节相同，没有缩短输出或减少工作量。
本轮控制组是第一轮优化后的 Windows 实现，不能把两轮改善百分比直接相加。

| 固定输入声学 | 原版秒数 / RTF | 候选秒数 / RTF | 同机官方秒数 / RTF |
| --- | --- | --- | --- |
| 短文，3.375 s 音频 | 0.9217 / 0.27308 | 0.9410 / 0.27880 | 0.3350 / 0.09927 |
| 长文，11.575 s 音频 | 3.4115 / 0.29473 | 3.3203 / 0.28685 | 1.4891 / 0.12864 |

声学原版短文进程均值范围 0.8637–0.9796 s，候选 0.9374–0.9445 s；长文原版
3.1819–3.6412 s，候选 3.3168–3.3238 s。基线波动明显，两者范围重叠，不据声学
均值宣称明确收益；原生与官方的完整整数时长及输出长度相同。候选仍约为官方声学
耗时的 2.81 倍（短文）/2.23 倍（长文），尚未达到胜过官方或 Mac 的目标。

原生进程峰值工作集约 630/916 MiB（声学短/长）、638/1043 MiB（生产短/长），
没有明显改变。官方 Python 的父级内存采样只捕捉到 venv 启动器约 5.6 MiB，**不是
官方模型内存**，原始值保留但排除内存比较；内部 forward 计时不受此采样范围影响。
本轮不宣称减少内存。

## Candle 的 MKL feature 实机试验

锁定 fork 的 `candle-core/mkl` 与 `candle-nn/mkl` 已成功完成 Windows release
隔离构建。`intel-mkl-src 0.8.1` 使用 Candle 工作区指定的 `mkl-static-lp64-iomp`，
自动获取固定 `windows/...:2022.0-3038006115` 包。此试验保存独立 lockfile，完成后
恢复产品 lockfile；没有向产品默认 feature 或全局 Rust flags 加入 MKL。

MKL 核心为静态库，Intel OpenMP 的 `libiomp5md.lib` 是导入库，探针运行仍需
`libiomp5md.dll`。本轮显式复用了本地 Torch 附带的该 DLL，不把“链接成功”当作
无需运行时依赖的干净机器验收。来源与许可要求见
[intel-mkl-src](https://github.com/rust-math/intel-mkl-src)。

同一长文、Rayon=8、MKL_DYNAMIC=FALSE 的两热轮筛查：

| MKL/OMP 线程 | 等待配置 | 声学秒数 |
| --- | --- | --- |
| 8 | 默认 | 4.377 / 4.498 |
| 8 | KMP_BLOCKTIME=0、OMP_WAIT_POLICY=PASSIVE | 4.595 / 4.526 |
| 4 | 同上 | 5.404 / 5.288 |
| 1 | 同上 | 7.315 / 7.294 |

在这台 Ryzen 9700X、此 MKL 构建及当前算子形状下，没有支持切换的性能证据。
未把已知厂商分派历史直接当成本机变慢根因，也未通过伪装 CPU 厂商的方式绕过。
线程池竞争只是需要排查的假设；本例被动等待没有解决退化。
[faer issue #108](https://github.com/sarah-quinones/faer-rs/issues/108)提供了调查线索，
不代表已经证明本例是同一原因。

`MKL_VERBOSE` 确认实际 oneMKL 2022.0 / build 20211112、SGEMM、NThr=8、Dyn=0。
头部没有提供足以确认本机 ISA 分派的证据，未据厂商名称推断它实际使用 SSE 或 AVX。
短文对官方 PCM SNR 为 26.1823 dB，长文 23.6803 dB，仍未达到既有 40 dB 门槛。

MKL 会改变 GEMM 和部分向量数学路径，Cargo feature 还会合并到共同 Candle 实例。
不能假定它只影响 Kokoro，或认为启用同为 F32 的后端自然保持 PCM。性能、数值、
多模型 feature 组合与发行包依赖必须分别通过。

## AVX2/FMA 编译基线试验

按建议用 `RUSTFLAGS="-C target-feature=+avx2,+fma"` 和单独 target 目录构建同一
最终候选，未写入 `.cargo/config.toml`。短文两热轮 0.875/0.870 s，长文
3.101/3.209 s，没有支持提高发行 CPU 基线的明确性能证据。完整 PCM 与原版不同，
对官方 SNR 短文 25.2609 dB、长文 22.8120 dB，未修复原有准确性问题。

`+avx2,+fma` 是编译期硬要求，不能当成自动运行时回退，也不等同于完整的
x86-64-v3 feature 集。Kokoro 上热点占比不支持把其他量化/attention 模型的收益
直接转成这里的 RTF 预期；本轮不修改通用发行基线，不宣称其他模型无收益。

## 检查及交付边界

最终 Kokoro lib 普通测试 17 通过、2 忽略，包括通道组与时间块的并行尾部、跨 batch、
strided/offset、NaN 与负零位模式。相关三包的 CLI+CUDA+g2pW+performance-tools
all-targets Clippy 在 `-D warnings` 下通过，CPU/g2pW 私有 rustdoc 严格检查通过。
普通测试没有下载模型。

真实模型仍限 `1.1-zh`。CPU/CUDA 各验证标准短文、长文、快/慢速、男声五组夹具，
完整 PCM 和整数时长与第一轮对应 Windows 基线一致；同 owner 重复与中途取消恢复
通过。公共 Engine 的 CPU/CUDA 取消、音色/长度切换后恢复原 pinned 请求也通过。
最终 CLI 的 v7 hello/config/shutdown 冒烟通过。未扩大到 MOSS 或其他模型的性能结论。

最终 CLI 以 `cli,kokoro-cuda,kokoro-g2pw` 构建到 `target/release/talechime.exe`，冻结
副本位于本轮证据目录 `talechime-final.exe`；不含 MKL 或 AVX2/FMA 的全局编译要求。
既有 CUDA 13 `LNK4098` 信息保留，没有通过屏蔽 linker 警告解决。

官方原有 40/60 dB 数值门禁问题未修复；本轮没有放宽门槛。人工试听、实际播放、
干净机器安装以及 Mac/Linux 回归未完成，不据逐位输出保持推断这些验收已通过。
源码、实验结论及技能反馈随本次代码提交保存；本轮未进行发行发布。

## 后续可验证方向

[oneDNN 卷积说明](https://uxlfoundation.github.io/oneDNN/dev_guide_convolution.html)
提供直接卷积与内部布局选择，和本轮官方日志对应。只启用 Candle `mkl` 不会把
其 im2col 卷积替换成 oneDNN 的 primitive；这是两条不同路线。若继续接入，需要
固定库版本、F32 math mode、可变长度 primitive/scratchpad 的模型所有权与上界，
先过独立官方解码/整模型门禁，再扩大性能测试。当前尚未接入 oneDNN。

[RTen 的卷积源码](https://github.com/robertknight/rten/blob/main/src/ops/conv.rs)
也提供按输入布局/形状选择算法的研究入口；虚拟窗口与 GEMM 打包融合需要内核级
支持，不能把本轮外层分块原型称为实现了这类算法。未复制其代码或迁移模型后端。

[Candle issue #3871](https://github.com/huggingface/candle/issues/3871)的广播权重复制
问题已核对调用链：当前宽卷积使用显式广播 view 后的 `matmul`，不是该报告的
`broadcast_matmul` 入口，不能套用其倍数收益。

所有筛查、插桩、试验性 DLL 适配器及失败日志保存在 Git 忽略目录
`target/kokoro-windows-cpu-round2-20261027/`。模型资源复用上一轮，不重复下载。
`evidence-manifest.json` 与 `source-snapshot/` 冻结本轮源码、二进制、资源、原始报告
和试验 native 库的摘要；`audit.py` 复核 100 个热轮的 PCM 及生产导出资产。
这些是本地实验产物，不冒充完整 xtask 矩阵审计或发行收据。
