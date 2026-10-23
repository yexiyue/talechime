# OmniVoice CPU / Metal 优化现状

截至 2026-10-24，以下为当前生产源码与已验收边界。历史候选、撤回项和阶段百分比
不代表默认同时启用。方法见 [性能门禁](tts-performance.md)，可迁移经验见
[Candle 技能](../../skills/candle-performance/SKILL.md)。

## 执行路径

| 环节 | CPU | Metal |
| --- | --- | --- |
| 构建 | `omnivoice`；macOS target 默认链接 Accelerate | `omnivoice-metal`；Accelerate 不替代 GPU 内核 |
| 默认精度 | stage0 / decoder F32 | stage0 F16 / decoder F32 |
| attention | 手写 softmax；受限 grouped GQA 与 K 转置视图 | 融合 SDPA、原生 GQA、sequence-first RMSNorm/RoPE，随后用转置视图 |
| K/V 投影 | 受限连续不可变权重缓存转置；短/非连续输入等回退 | 原投影布局 |
| MLP | 受限合并 gate/up，回退权重共享合并缓冲；其他路径原实现 | 原 MLP |
| greedy confidence | F32 连续 log_probs 复用 argmax gather | 复用 argmax gather；随机采样仍取分布最大置信度 |
| decoder 转置卷积 | 原路径 | 受限零 padding col2im 后裁剪与 bias |
| decoder 1×1 | 对齐时间长度、受限方阵通道的直接 NCL GEMM | dense F32 连续输入/权重的直接 NCL GEMM |
| decoder 7-tap | 受限 dilation=1 的 NCL 窗口/GEMM | 原卷积；Metal 候选未保留 |

私有实现负责适用条件和回退，参考音频编码不使用 decoder 特化。
CPU/CUDA/GGUF lazy 及不匹配配置不能套用 Metal 结论，CPU 特化也不能推广到任意 F32 形状。
macOS Accelerate 不可用于 AMD Windows/Linux；后者使用各自的 CPU 构建路径。
源码入口为 `stage0_qwen3.rs`、`stage0_cpu_linear.rs`、`stage0_model.rs`、`codec_layers.rs`。
CPU、Metal、scoring 回归测试在同名目录的具名子模块，真实模型探针显式 ignored。

## 性能证据与解释范围

| 路径 / 对照 | 主要观察 | 限制与记录 |
| --- | --- | --- |
| Metal 与同期官方 MPS | Rust 2.672 s / RTF .495；官方 3.235 s / RTF .600 | M4 Pro、短文本、F16 主干；输出长度与 RNG 不完全相同，是产品比较，非相同张量或音质等价。[复核](../records/omnivoice/omnivoice-official-comparison-audit-2026-10-23.md) |
| CPU 与官方 CPU | 优化 decoder 前的产品比较 Rust 7.021 s / 官方 6.236 s | 音频长度、seed 派生和宿主处理不同；decoder 优化后没有重新配对官方，不报新的排名。[归因](../records/omnivoice/omnivoice-cpu-real-input-2026-10-23.md) |
| CPU 1×1 新旧实现 | 固定130帧 decoder 419.901→345.533 ms，耗时减少17.71% | 不是整链百分比；短长完整输出一致，整链增量仍有波动。[记录](../records/omnivoice/omnivoice-cpu-pointwise-2026-10-24.md) |
| CPU 7-tap 新旧实现 | 固定130帧 decoder 356.828→297.541 ms，耗时减少16.62% | 与上一轮基线不同，不相加；整链两组观测0.46%/1.23%，不确认稳定增量。[记录](../records/omnivoice/omnivoice-cpu-conv7-2026-10-24.md) |

固定模型为 OmniVoice 0.6B revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`，
Candle fork `2266099d92265112ecc0d56ecee1761e3cc2ac00`。各记录附实际输入、轮次、
权重和二进制哈希，不把不同 revision 的结果拼成一条配对曲线。

这些优化证明旧路径存在实现开销：手写算子未自动融合、重复 KV、布局物化、通用 im2col
及错过 col2im 快路径。已链接 Accelerate 或启用 Metal 不能排除这些开销，也不能证明 CPU
已经到硬件上限。Python 与 Rust 只是调度不同原生算子，不能按语言判断性能。

## 数值与撤回边界

初期 Metal SDPA/RMSNorm 及转置卷积变换曾改变数值，按当时授权单独记录，
不能声称它们与最早 `main` 的 codes/WAV 逐位相同。
随后 GQA、RoPE 布局、greedy gather、decoder pointwise，以及 CPU 已保留路径，
各自对其上一验收基线验证完整输出；不能据此推出与官方 Python 逐位或音质等价。

Snake 常量缓存、Metal 7-tap、自定义 SiLU×up、CPU Flash、某些投影/输出头布局、
真实 MLP gate/up 连续化等没有取得可保留的净收益或未通过数值门禁，只留离线记录。
尤其真实 MLP 连续化让完整消费者慢约2%–5%，不因为某个子算子少几次调用就默认保留。
后续优先项为 CPU dilation=3/9 的真实窗口布局与消费者，尚未默认启用。

## 诊断与维护

吞吐与诊断分开：`OMNIVOICE_SYNC_PROBE` 会插入屏障和 GPU canary，codes dump 会导出
完整结果；它们不用于默认吞吐。canary 张量归对应 runtime/device，不能跨设备全局缓存。
捕获/重放的 `cfg(test)` 路径不进入生产；归因测量必须调用实际生产入口，保留旧路径时
明确标为控制组。重构优先保持已有运算顺序和设备选择，不把不同快路径回退条件合并。

复现命令与归档约定见 [离线性能工具](../../scripts/perf/README.md)。
全链路审查与简化见 [2026-10-24 审查记录](../records/omnivoice/omnivoice-optimization-review-2026-10-24.md)。
