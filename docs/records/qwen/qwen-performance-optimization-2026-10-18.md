# Qwen3-TTS Metal 性能优化（2026-10-18）

本记录延续 `qwen-performance-review-2026-10-18.md` 的实现内前后对照原则：
官方实现只作为同机开发对照，不把跨实现的输出差异解释成速度收益；所有合入路径
必须在固定输入下给出等价证据。测试设备为 Apple M4 Pro，模型为 Qwen 0.6B Base
ICL / Metal BF16，文本“山风吹过松林，星光照亮归途。两个人沿着小路慢慢往前走。”，
seed 42，10 帧流式块。原始证据保存在 Git 忽略目录 `runs/qwen-review/`。

## 起点

| 指标 | Rust Metal | 官方 PyTorch MPS BF16 |
| --- | ---: | ---: |
| 完整合成中位数 | 4.3487 s | 20.5737 s |
| 流式合成中位数 | 10.1550 s | — |
| 首 PCM 中位数 | 1.0753 s | — |

官方对照使用显式匹配语义采样、空历史 repetition penalty guard、1 轮预热和 5 轮测量。
双方随机数实现、精度传播与 token 数不同，因此只列绝对值，不宣称逐算子同精度或统一框架提速。

## 阶段 1：细分计时

新增 feature-gated tracing span，覆盖 CodePredictor 的 cache reset / prefill /
每个 acoustic step，以及 12Hz codec quantizer、pre-conv、transformer、upsample
和 decoder blocks。采样侧最终包含 dtype 转换、GPU→CPU 回传、temperature、
top-k、top-p、softmax、CDF/选择与 CPU→GPU 回传 span；早期诊断结论先用于定位，
随后补入生产 profiling feature。

两轮完整 + 流式的 trace 显示：`codec_upsample_stages` 约 9.64 s（20 个 chunk，
均值 482 ms），语义采样约 4.54 s（172 次，均值 26.4 ms），CodePredictor 内层约
3.64 s，cache reset 每帧仅约 0.001 ms。Metal tracing 对异步命令提交是方向性证据，
不能当作逐 kernel GPU 独占时间。

## 阶段 2：语义采样

已尝试把 batch=1 / vocab=3072 的 top-k、top-p、softmax、CDF 和 PCG 选择融合成
一次 CPU 行处理。小测试通过，但真实生成与基线首个语义 token 即发散，并连续生成到
2048 帧导致 Metal buffer 分配失败；重新应用仍复现。补充分 trace 后确认主要成本是
GPU→CPU logits 回传约 19.2 ms/次和 CPU CDF 约 5.6 ms/次；CPU softmax 与选择合计不足
0.03 ms。

另一个安全实验把 BF16 logits 先回传再转 F32。语义 codes 与基线相同，但回传均值
仅从 19.205 ms 到 19.172 ms，CPU F32 转换升至 5.63 ms，总采样约 25.21 ms 未改善，
已回退。融合补丁保留在 `runs/qwen-review/optimization-stage2/rejected-fused-sampler.patch`，
在找到首个差异的精确浮点原因前不重试。

## 阶段 3：CodePredictor KV cache

Metal/CUDA 原实现使用 `PreAllocKVCache`。将 Metal 临时切换为 Concat cache 后，
五轮输出与 PreAlloc 完全一致：

| cache | 完整 / s | 流式 / s | 首 PCM / s |
| --- | ---: | ---: | ---: |
| PreAlloc（保留） | 4.3879 | 10.1980 | 1.0947 |
| Concat（拒绝） | 4.2414 | 10.9409 | 1.0497 |

Concat 仅首 PCM 快约 45 ms，但完整慢 3.3%、流式慢 7.3%，因此保留 PreAlloc。
cache reset 不是瓶颈。下一步的小 projection / `slice_assign` 优化仍需固定隐藏态和
acoustic codes 等价，但当前 trace 中 CodePredictor 总占比已低于 codec 与采样。

## 阶段 4：Metal depthwise fast path

上游复核：<https://github.com/huggingface/candle/issues/3389> 仍为 open；
<https://github.com/huggingface/candle/pull/3531> 仍为 open / 未合并，且 fast path
仅限 CUDA。Candle 0.11 的 Metal grouped `conv1d` 仍按 group 拆成多个小 im2col/matmul。
Qwen 12Hz decoder 的两个 pre-upsample ConvNeXt 每个包含 depthwise causal conv
（`groups=channels`，kernel=7）。

新增本地 `DepthwiseCausalConv1d`：仅当 unit stride / dilation、每组单输入且
`groups=channels` 时加载；仅在 Metal 上把 7 个时间窗 stack 成 `[batch*channels,
length, kernel]`，一次 batched matmul 完成 per-channel reduction，CPU/CUDA 及不满足
条件时保持原 Candle 路径。固定输入单测覆盖多 batch、短
长度和长度为 1 的边界，并通过 `forward` 的 CPU 与原生 Metal 路由比较。
断言为逐元素相同；batched matmul 保持 stock grouped conv 的求和顺序。

| 版本 | 完整 / s | 流式 / s | 首 PCM / s |
| --- | ---: | ---: | ---: |
| Stage 3 PreAlloc | 4.3879 | 10.1980 | 1.0947 |
| depthwise batched-matmul fast path | 3.9792 | 5.4465 | 0.8013 |

五轮中位数分别改善 9.3%、46.6%、26.8%。86 帧、165120 samples、正常 EOS 和有限
PCM 均保持；codes、整段 PCM 和流式 PCM 与 PreAlloc 基线在 6 轮（预热 + 5 测量）中
SHA-256 全一致：codes `5a4b8fb3…4c9a28`，full `bbd6adf3…4b02485`，stream
`21f4993a…7991`。首个广播逐 tap 相加版本虽更快但改变求和顺序（PCM max abs
`4.5076e-7`），不满足本性能优化要求，已拒绝并改用上述等价 matmul 形态。该证据不宣称
听感等价；本次未重做人工试听。

## 增量解码状态与后续瓶颈

完整增量 12Hz state（causal conv tail、72 帧 K/V/RoPE、upsample/decoder 上下文）
暂不实现。depthwise 修复移除了最大异常开销后，新的 trace 中 upsample stages 合计仅
约 0.036 s（20 chunk），decoder blocks 合计约 1.246 s（该 trace 来自被拒绝的广播
实验，仅作方向性证据），采样仍是主要前端瓶颈。
此时引入复杂状态缓存的收益与风险不匹配。

补充三条中文文本、各 1 轮预热 + 5 轮测量，均正常 EOS 且 PCM 有限：

| 文本 | 帧 / samples | 完整 / s | 流式 / s | 首 PCM / s |
| --- | ---: | ---: | ---: | ---: |
| “今晚风很轻。”（短句补充） | 16 / 30720 | 1.0493 | 1.2317 | 0.8000 |
| “山风吹过松林，星光照亮归途。”（约 15 字） | 40 / 76800 | 2.0417 | 2.5871 | 0.7986 |
| 约 60 字长句 | 224 / 430080 | 10.0942 | 14.5775 | 0.8035 |

短句首 PCM 由固定模型准备与首个 codec 块主导，depthwise 修复仍将首块控制在约
0.8 s；长句完整路径 RTF 约 0.564，流式 RTF 约 0.813。官方对照未在这两条新文本上
重复运行，不能把本表与官方速度比较。
三条 Rust 文本的固定 seed codes/full PCM/stream PCM 在各自预热与测量轮内 SHA-256 均一致。
起点短句 hashes 见阶段 4；中句 codes/full/stream 为 `15e5219a…5468`、
`255dd688…71d8`、`34231aa2…900a1`，长句为 `517e1fcc…9285`、`8b91b8cc…e4d0a`、
`18b26a13…74d4a`。trace 汇总未导出到这些目录；阶段 1/4 的 trace 已覆盖瓶颈定位，
不把本表伪装成逐阶段 trace 矩阵。

## 阶段 6：readback 与 codec 卷积诊断

孤立探针使用已提交的 `[1, 3072]` logits fixture，并在每次采样前后同步 Metal。
F32 p50 约 `0.122–0.139 ms`；BF16 样本计时包含 F32→BF16 转换和 6 KiB 回传，不能当作纯传输。即使如此，两者都说明表观 `19 ms` 不是 12 KiB 传输带宽。
Candle 0.11 的 `MetalStorage::to_cpu` 会提交并等待当前 command buffer；Metal 队列
FIFO，因此生产 trace 中的 `sampling_gpu_to_cpu_transfer` p50 `19.23 ms` 主要是在
等待此前排队的 logits 相关 GPU 工作。重新按 B/E 事件配对后，CPU→GPU 单 token
返回 p50 仅 `0.018 ms`；早先“返回也约 19 ms”是汇总单位误读。生产采样中
`sampling_cdf_and_selection` p50 `5.76 ms`，来自 Candle `cumsum`（矩阵乘三角矩阵）
及其后 mask / where / argmin 全词表小算子。

新增 `qwen_conv_probe` 按真实 0.6B 12Hz decoder 权重几何构造 dense causal conv 与
transposed conv 微基准。268 帧、1 次预热 + 10 轮测量时，6 组 residual dense conv
p50 合计约 `570 ms`，4 个 decoder transposed conv 约 `130 ms`，其余卷积约
`188 ms`；合计约 `887 ms`。该 isolated fixture 只覆盖卷积调用本身，不包含
Snake/nonlinearity、残差组合和调度间隙，因此低于 full decode；54 帧输入下同一
集合 p50 约 `186 ms`。固定 codes 的 `context25` trace 显示非首块 `codec_decoder_blocks`
约 `15.5 ms`，前端/量化等阶段约 `2.4 ms`；慢因是每个 10 帧块重复计算 79 帧，而非
transposed conv 单点异常。证据在 `runs/qwen-review/optimization-stage6/`。

## 阶段 7：CPU CDF/选择等价优化

只替换 `multinomial_sample` 内的 CDF 构建与选择：直接读取 CPU 概率行，按原
vocabulary 顺序用 f32 累加，返回首个 `cumulative >= threshold` 的索引。temperature、
top-k、top-p、softmax、PCG 随机数、阈值比较和 CPU/Metal 统一采样策略均不变。
该实现不宣称与 Candle `cumsum` 的矩阵乘内部舍入逐位等价；当前证据是共同 4 轮真实模型 codes/full/stream PCM 逐字节一致。阶段 7 最初把无命中 fallback 设为最后一个 vocabulary index；后续完整复审用旧 tensor 选择路径复现后确认，旧版全候选相等时的 argmin 实际返回 0，现已修正并增加独立对照回归。若后续固定输入出现首个差异，应回退本优化并记录差异阈值。

| 版本 | 完整 / s | 流式 / s | 首 PCM / s |
| --- | ---: | ---: | ---: |
| 旧 full-tensor CDF | 9.5446 | 12.5395 | 1.0144 |
| sequential CPU CDF | 6.6953 | 9.6648 | 0.7937 |

上表只作非配对、方向性观察，不作为性能收益结论：优化侧 5 轮先跑，旧版 3 轮后跑，
轮数与机器状态不同。可归因证据是
`sampling_cdf_and_selection` p50 从 `5.76 ms` 降到 `0.002 ms`；总 `sampling` p50
约 `22.27 ms`，仍被等待 logits 的同步边界主导。优化侧与旧版在共同 4 轮（含预热）
中 codes、整段 PCM、流式 PCM 逐字节一致；trace 文件在
`runs/qwen-review/optimization-stage7/`。

## 阶段 8：additive token suppression

将复用的 suppression mask 从 boolean `where_cond` 改为预构建的
`[1, vocab]` 0/-inf additive mask，每帧一次 `broadcast_add` 替代 boolean broadcast、
负无穷 broadcast 和 `where_cond` 三个小 GPU kernel。保留 token、EOS 和 min-new-token
语义。短句同机 A/B：各 1 轮预热 + 5 轮测量，旧/新 codes、整段 PCM、流式 PCM
逐字节一致；流式中位数 `7.026 s -> 6.087 s`，首 PCM `0.967 s -> 0.760 s`。
整段中位数 `4.741 s -> 4.180 s`。A/B 的旧版后跑，且旧版轮间波动更大，因此该表
只作为同输出等价下的方向性观察。长句 203 帧仅在新实现上验证 6 轮输出稳定与正常
EOS；后半段机器负载升高，耗时未用于比较。试听文件见
`runs/qwen-review/optimization-stage8/additive-suppression/short/`。

复审后仅做等价简化：suppression mask 直接构建 F32 数据，repetition penalty 两条
公开路径合并到同一实现，readback 探针把计时辅助逻辑移出主循环并把不变式校验移出
采样区间。这些简化不改变生产输出或 trace 字段。

用户已确认本阶段新旧对照音频“听感没问题”。由于两侧文件逐字节相同，该试听确认的是
优化没有损伤当前样本听感，不扩展为全模型质量结论。

## 最终汇总

当前可与官方对照的是同一 M4 Pro、Qwen 0.6B Base ICL、同一中文短句与参考音频、
seed 42、显式匹配语义采样（top-k 50 / top-p 0.9 / temperature 0.9 /
repetition penalty 1.05，声学贪心）、空历史 repetition guard、模型准备和导出均在
计时外、各 1 轮预热 + 5 轮测量的非流式路径。官方为 PyTorch 2.7.1 / Transformers
4.57.3 MPS BF16；Talechime 为 Candle Metal BF16 Talker/CodePredictor 与 F32 codec。
两边随机数实现、浮点传播和输出长度不同，因此这是同输入条件观察，不是同一 token
工作负载的配对加速。

| 路径 | 合成中位数 / s | 音频 / s | RTF | 首 PCM / s |
| --- | ---: | ---: | ---: | ---: |
| 官方 PyTorch MPS BF16 | 20.5737 | 6.96 | 2.9560 | — |
| Talechime Rust Metal 完整 | 4.1796 | 8.16 | 0.5122 | — |
| Talechime Rust Metal 流式（10 帧） | 6.0868 | 8.16 | 0.7459 | 0.7597 |

在上述边界内，完整路径 RTF 低约 82.7%、合成耗时低约 79.7%；流式合成耗时低约
70.4%。这些比例不表示音质、内容完整性、CUDA/Windows 或其他设备的普遍优势。
官方原始证据在 `runs/qwen-review/fair-comparison-20261019/official/summary.json`；
最终 Rust 证据在 `optimization-stage8/additive-suppression/short/rounds.json`。

阶段内可归因收益不能跨阶段相加：Stage 4 的 depthwise fast path 在固定输出下将
流式中位数 `10.1980 s -> 5.4465 s`、首 PCM `1.0947 s -> 0.8013 s`，且 codes/full/
stream PCM 逐字节一致；Stage 7/8 分别有各自 A/B 与输出等价证据，但后续正确性修复
会改变输出和工作负载，因此不能用起点与最终表的差值宣称全链路总提速。后续仍应优先
定位 talker 28 层小 kernel 与 codec 增量 residual state。

后续优先级：

1. 采样若要进一步加速，需要减少 logits 产生路径中已排队的小 GPU kernel，或实现
   位等价 GPU 采样；不能用 dtype/传输微调替代该证据。
   当前 trace 的语义循环中，每个生成帧包含 15 次 CodePredictor acoustic step、一次
   talker decode 和一次 logits 回传同步；`talker_step` p50 约 `5.08 ms`，回传等待
   p50 约 `22.31 ms`。下一轮优先审视 talker 单步和 logits penalty / suppression
   的小 kernel 提交流水，而不是继续压缩 12 KiB CPU 采样本身。
2. codec 侧优先评估完整增量 residual causal conv 状态；transposed conv 只占约 6%
   卷积时间，先做局部 transposed conv 优化收益不足。
3. CodePredictor 仅在小张量分派仍有明显占比时优化，固定隐藏态下 15 个 acoustic
   codes 必须完全一致。

阶段 8 后，剩余主要疑点在 talker 单步 28 层小 kernel 串行提交；需要先为 layer /
attention / MLP 增加只读计时，再评估 RoPE 小张量、Q/K norm、projection 合并或
custom Metal kernel。任何合并必须证明固定 hidden 与 codes 逐位一致。

## 完整复审与职责收拢

审查范围为 `59696d5...9d0765a` 的 Qwen 正确性与性能改动；本轮修改以
`9d0765aa97c2f3954d7bc27bf62b65fa50a1b259` 为对照。规范与计划符合性分别审查，
修改后再次独立复核。没有引入新的采样算法、Metal kernel 或增量 KV/conv 缓存。

### 规范与架构审查

- depthwise 实现及其等价测试移回 `codec/causal_conv.rs`。删除原来为 ConvNeXt
  暴露卷积表示的六个 getter，stock/fast 路径共享左侧 padding；窗口、matmul 和 bias
  的顺序不变。ConvNeXt 只组合卷积、归一化、MLP 与残差。
- `synthesis/streaming/decoder.rs` 拥有 pending frames、25 帧 codes 历史、打包、
  PCM 裁剪和成功后提交。会话继续拥有自回归生成、EOS 与取消。注入的单次解码闭包
  使用静态分派，使普通单测能覆盖真实的状态提交路径，不需构造模型或加载权重。
  这里缓存的是 codes，上下文仍完整重算，不属于已延期的增量 codec 实现。
- `sample` 负责设备传输，`sample_cpu` 负责统一采样策略，移除一次递归调用和重复
  dtype span。随机数调用数量、top-k/top-p/softmax 和正常 CDF 选择顺序不变。
- 计时探针只在 Metal 上运行附加 readback 诊断，不再阻止 CPU/CUDA 合成计时。
  本轮未进行 CPU/CUDA 真模型性能验收，不能由入口可运行推断已验收。

按四类设计原则复核：

| 原则 | 本轮约束 |
| --- | --- |
| KISS | 保留现有算子与成功路径，只拆出有独立职责的状态和设备边界 |
| DRY | 共用 causal padding 与 repetition penalty 计算，回归测试使用独立旧实现作对照 |
| YAGNI | 不引入通用推理框架、动态 trait、插件路由或未验证的增量缓存 |
| SOLID | 卷积拥有卷积表示，会话拥有生成生命周期，decoder 拥有块解码状态；测试依赖窄解码接口 |

### 计划与准确性审查

两项实际回归先由新增测试复现失败，再修复：

1. CDF 无命中时返回末 token，偏离旧版 argmin 的 0。现用原 tensor
   `cumsum/ge/where/argmin` 路径核验零概率、全 NaN 和概率和不足的 fallback，
   并验证 RNG 状态推进一致。另以 vocab 4/3072、两行分布和 512 个 seed
   对照旧 Candle CDF；这仍不构成所有浮点分布的舍入等价证明。
2. repetition penalty 公共路径简化时丢失 mask 广播，导致 `[1, vocab]` mask
   对多 batch logits 报 shape mismatch。恢复元数据广播，测试同时检查正负 logits、
   未惩罚位置和两条公开调用路径的一致性。

新增 decoder 事务测试覆盖解码失败后原帧/历史保留并可重试，以及 10 帧后 2 帧尾块
只输出新 PCM、不重复历史。既有参考顺序及 25 帧历史推进测试随模块迁移保留。
计时探针额外拒绝达到帧上限的 EOS 未证实结果，并检查整段/流式帧数、样本数和
每帧 1920 samples 覆盖；整段 PCM 与流式 PCM 各自对照旧版，不要求两种解码方式互相逐字节相同。

macOS Accelerate 是先前已经记录的 CPU 数值边界；本轮与 `9d0765a` 对照时两侧
都启用它。当前 Metal 等价证据不能证明 Accelerate 与旧纯 Rust GEMM 的 CPU
codes/PCM 逐位相同，不能宣称已满足这项跨 CPU 实现的等价要求。

本轮原始构建、测试、逐轮耗时和 hashes 保存于
`runs/qwen-review/architecture-review-20261018/`；真模型结果与最终检查补充如下。

### 本轮真实模型对照

同一 M4 Pro、0.6B Base / Metal BF16、seed 42、默认显式生成配置、10 帧块，
使用 `reference-cross/fixed-prompt.json`。A 从 `git archive 9d0765a` 的不可变源码
单独 release 构建，B 为本轮实现；两侧同 feature，未启用 profiling。
模型加载、文件导出和附加 readback 诊断在计时外；各进程先预热整段和流式路径。
测量期间没有本任务的并行编译或推理。二进制、prompt hash 与完整文本见
`architecture-review-20261018/manifest.json`。

主样本按 A1/B1/B2/A2 串行，每进程 1 次预热 + 2 次测量，两侧各 4 个测量值；
中句和新增长句分别 A/B 各 1 次预热 + 1 次测量，只作补充回归。

| 文本 / 帧数 | 整段 A → B / s | 流式 A → B / s | 首 PCM A → B / s |
| --- | ---: | ---: | ---: |
| 主样本 / 86 | 3.8210 → 3.7201 | 5.5375 → 5.2787 | 0.8566 → 0.7901 |
| 约 15 字 / 40 | 2.0893 → 2.1025 | 2.8520 → 2.9229 | 0.9165 → 0.9568 |
| 约 60 字 / 163 | 7.6971 → 7.7252 | 11.2027 → 11.2435 | 0.9210 → 0.8657 |

三文本各自所有预热及测量轮的 codes/full PCM/stream PCM hash 全一致；所有输出
finite、正常 EOS，整段与流式帧数/样本数相等，流式 chunks 无样本丢失或重复。
主样本仍为阶段 4 的 `5a4b8fb3…4c9a28` / `bbd6adf3…4b02485` /
`21f4993a…7991`；新增长句为 `d61ea2b5…7b5b28` / `7f651864…654b11` /
`77281c9f…48cbb`。完整 hashes 与逐轮记录在 `comparison-rounds.json`、`summary.json`。

本轮以输出不回归、主样本 ABBA 未见性能损伤作为验收；中句和长句单轮耗时升幅
约 0.4%–2.5%，不足以判断稳定差异，不把主样本下降宣称为新增提速。
本轮主样本为 86 帧，不与上文 102 帧的最终官方对照表计算增减，也未重新运行官方实现。
PCM 与本轮前的文件相同，未重复要求人工试听；数值证据只覆盖这些本机输入。

真实后端额外执行一次“收到首块 PCM 后销毁接收端，再在同一模型上发起新请求”，
并与独立加载模型的正常请求比较：两者都正常 End、165120 个有限 samples，PCM
SHA-256 均为 `21f4993a…7991`。接收端销毁耗时小于 1 ms；这不是 GPU 在 1 ms 内
停止的证明。同模型复用请求 5.293 s、独立请求 5.341 s，仅用于验收正常复用，
不用于速度比较；结果见 `cancel-reuse-summary.json`。

本轮普通 Qwen 测试 220 passed / 3 ignored，后端测试 43 passed；显式执行原生
Metal vocab 4/3072 采样对照 1 passed。Qwen/backends Metal + profiling 的 Clippy
（`-D warnings`）、rustdoc（`-D warnings`）、格式及 diff 检查通过，`Cargo.lock` 未变。
本轮没有重跑全 workspace；前述范围以外的既有 workspace 结果见历次验证。

## 历次验证

- `cargo test --locked -p talechime-qwen3-tts --features metal,profiling --tests --examples`：
  216 passed，3 ignored（复审后新增 CDF 边界/fallback 测试；最终简化后重跑通过）。
- depthwise / sampling 定向测试：29 passed，2 ignored，包含原生 Metal 路由测试。
- depthwise fast path CPU/Metal 路由等价测试：2 passed。
- 显式原生 Metal 3072 词表采样测试：1 passed。
- `cargo clippy --locked -p talechime-qwen3-tts --features metal --all-targets -- -D warnings`：
  通过。
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked -p talechime-qwen3-tts --features metal --no-deps`：
  通过。
- workspace 串行 lib/tests/examples：61 个测试二进制合计 687 passed，24 ignored，
  0 failed；仅跳过既有
  `default_home_migration_keeps_hello_lightweight_and_legacy_voices_visible`
  5 秒时限测试。
- `cargo fmt --all -- --check`、`git diff --check`：通过。
- 阶段 7 采样定向测试：30 passed，1 ignored；旧/新共同 4 轮固定 seed 输出逐字节一致。
- 真模型：1 轮预热 + 5 轮测量；输出文件见
  `runs/qwen-review/optimization-stage3/baseline-prealloc/`、`concat/`、
  `optimization-stage4/depthwise-matmul-final/`（被拒绝的广播实验在 `depthwise/`）。

尚未执行 CUDA/Windows；本阶段人工试听已完成，三文本 Rust 矩阵已完成，官方 guard
只覆盖上述短句。
本记录中的性能比例只适用于上述本机输入和计时边界。
