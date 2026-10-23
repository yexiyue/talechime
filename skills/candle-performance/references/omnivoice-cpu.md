# OmniVoice CPU：从真实张量定位到受限布局优化

截至 2026-10-24 的 M4 Pro 案例，Candle fork
`2266099d92265112ecc0d56ecee1761e3cc2ac00`，stage0 与 decoder 均 CPU F32、Accelerate。
用于提出实验，不是其他芯片、BLAS、模型或形状的性能及位级保证。
Metal 的结论见 [独立经验](omnivoice-metal.md)，通用验收沿用主 skill。

## 先区分 feature、实现与底层算子

macOS target 依赖启用 Accelerate，Cargo feature 合并后核验最终二进制动态链接。
Apple Accelerate 不能用于 AMD Windows/Linux；Hugging Face accelerate 包不是系统 BLAS。
链接存在不等于所有算子使用 BLAS，也不等于某个几何实际使用 AMX。
Rayon、VecLib、OMP 环境分别记录；本例请求 8 线程，不作为所有项目的固定最优值。

官方 Python 调度 ATen，Rust 调度 Candle；先查真实 attention 路由和 BLAS 调用，
不能把语言开销、没有启用加速、未优化布局或硬件上限混成一个原因。
本例官方 CPU 使用 Flash attention，但试过的 Candle CPU Flash 更慢且改变输出，未保留。
Accelerate GEMM 与 SiLU 向量函数均已接通，仍存在窗口展开、输出复制和激活差距。

## 用同一真实输入分开算术与产品差距

短文本实际 CFG 为 B2/S157，Q/KV 为 16/8 头、head_dim128，MLP 为 1024/3072；
CFG 拼接 batch，不能把序列翻倍。相同文本和 seed 不等于跨框架相同的隐藏态或 codes。
产品端同时报耗时、音频长度与 RTF；算子归因另重放相同 F32 值和不可变权重。

捕获能力放在私有 `cfg(test)` 模块，不扩展公共调试结构或默认推理开销。
本例覆盖 step0/15/31 × layer0/13/27；这些是局部取样，不是全模型全部输入的证明。
归档保存完整张量、shape、stride、offset 与 hash。Safetensors 会连续化值：重放必须
由真实上游重新计算布局，或明确重建原 stride；加载后的 contiguous gate/up 不代表生产布局。
重放先与原路径逐位比较，再计时；组件、完整层和下游消费者重叠，不能相加。

## 已保留路径与必须保留的边界

| 路径 | 减少的成本 | 本例适用条件与回退 |
| --- | --- | --- |
| 手写 attention 的转置 K 视图 | 避免 K 转置后的物化 | CPU F32 Accelerate、连续 K、head_dim128；其他走原复制 |
| grouped GQA | 避免 KV 重复并减少 BLAS 调用 | 连续 Q/K、2:1 头比、S>=8、dim128、同 batch/序列；不能照搬 Metal SDPA 接口 |
| K/V 投影权重转置缓存 | 移除每次投影的布局处理 | 连续不可变 CPU F32、1024×1024；短/非连续输入、非连续权重及 VarMap 活权重回退；包含额外长期权重存储成本 |
| 合并 gate/up 投影 | 少一次 GEMM 调用 | CPU F32 Accelerate、SiLU、1024/3072、连续不可变权重、连续 rank3 输入且 S>=8；短输入等回退，原权重视图共享合并缓冲 |
| greedy confidence gather | 复用 argmax，少扫描词表一次 | CPU F32 连续 log_probs；随机采样保留分布最大置信度，非连续/其他 dtype 保留原归约 |
| decoder 1×1 直接 NCL GEMM | 避免 im2col 与输出复制 | dense、连续 CPU F32、方阵通道32/64/128/256/512、标准1×1配置、T>=8且为8倍数 |
| decoder 7-tap NCL 窗口与 GEMM | 长连续行物化窗口，直接输出 NCL | 同样受限通道/时间格点，kernel7/stride1/padding3/dilation1/groups1；dilation3/9等回退 |

缓存活权重会过期；通过不可变参数筛选及更新后回归验证防止此问题。
合并后仍持有一份原权重加一份合并权重，会延长重复存储寿命；本例回退投影只保留合并缓冲的行视图。
不同操作的回退条件不是同一概念，不为减少几行判断而合并成宽泛的“CPU 快路径”。

## 卷积：连续化方向比是否使用 GEMM 更重要

固定 CPU Conv1d 原路径为 NLC im2col → GEMM → NCL 输出复制。
1×1 可直接 `W × X`；7-tap 仍需窗口，按 `[B,C,tap,T]` 物化后 `W × columns`。
保留 `[input_channel,tap]` 收缩索引次序，但 BLAS 几何改变仍可能改变舍入。
不受限的 CPU 1×1 在 T=13 曾出现位差，因此保留对齐格点与回退，未放宽容差。

每个候选计时包含 bias、所有布局转换和后续 Snake/残差/1×1 消费者。
隔离 NLC 窗口控制不等于 Candle 私有 im2col 的直接分段计时；只用于解释布局差异。
最终用相同 codes 完整 decode 验收，包括极短输入与长文本所有分段。
7-tap 仍分配完整 7 倍窗口和 padding，不能称为零复制或从算子收益推断峰值内存下降。

## 已排除的候选与结果边界

真实 MLP 的 SiLU×up 明显慢于官方，但显式 gate/up 连续化包含复制后反而慢：
三进程完整 MLP+残差分别慢约 2%–5%，未接入生产。Synthetic probe 的失败也不能仅凭
“换成真实输入”就改称成功，必须重新证明完整消费者收益。
预转置输出头/部分投影、MLP 自定义 SiLU×up、CPU Flash 等历史失败候选不设运行时开关。

两个 decoder 优化分轮验证：1×1 固定130帧约 419.9→345.5 ms，随后7-tap约
356.8→297.5 ms。各轮基线不同，不能相加百分比或直接把两个绝对数拼成配对收益。
短长完整 codes 与28个 WAV 保持一致，但整链增量仍受波动影响；局部收益不能冒充追平官方。

## 原始证据入口

移出本仓库时以下链接可能不可用，上述决策不依赖携带真实模型或张量。

- [CPU 真实输入归因](../../../docs/records/omnivoice/omnivoice-cpu-real-input-2026-10-23.md)
- [K/V 布局缓存](../../../docs/records/omnivoice/omnivoice-cpu-kv-layout-2026-10-23.md)
- [MLP 合并](../../../docs/records/omnivoice/omnivoice-cpu-mlp-2026-10-23.md)
- [真实 MLP 连续化撤回](../../../docs/records/omnivoice/omnivoice-cpu-mlp-layout-2026-10-24.md)
- [CPU 1×1](../../../docs/records/omnivoice/omnivoice-cpu-pointwise-2026-10-24.md)
- [CPU 7-tap](../../../docs/records/omnivoice/omnivoice-cpu-conv7-2026-10-24.md)
