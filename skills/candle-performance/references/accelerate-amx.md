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
