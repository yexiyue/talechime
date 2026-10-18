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
该实现不宣称与 Candle `cumsum` 的矩阵乘内部舍入逐位等价；当前证据是共同 4 轮真实模型 codes/full/stream PCM 逐字节一致。无命中时 fallback 到最后一个 vocabulary index，与旧实现的 argmin 位置一致；若后续固定输入出现首个差异，应回退本优化并记录差异阈值。

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

## 验证

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
