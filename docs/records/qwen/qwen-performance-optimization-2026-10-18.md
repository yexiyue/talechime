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

后续优先级：

1. 定位 GPU→CPU logits 回传为何约 19 ms，并找出首个 fused sampler 差异 token 的
   浮点原因；否则不改采样语义。
2. 为 decoder block 的 dense causal conv / transposed conv 建立独立微基准，再评估
   官方增量状态方案。
3. CodePredictor 仅在小张量分派仍有明显占比时优化，固定隐藏态下 15 个 acoustic
   codes 必须完全一致。

## 验证

- `cargo test --locked -p talechime-qwen3-tts --features metal,profiling --tests --examples`：
  213 passed，3 ignored（复审后新增 Metal 路由测试）。
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
- 真模型：1 轮预热 + 5 轮测量；输出文件见
  `runs/qwen-review/optimization-stage3/baseline-prealloc/`、`concat/`、
  `optimization-stage4/depthwise-matmul-final/`（被拒绝的广播实验在 `depthwise/`）。

尚未执行 CUDA/Windows 或新的人工试听；三文本 Rust 矩阵已完成，官方 guard 只覆盖起点短句。
本记录中的性能比例只适用于上述本机输入和计时边界。
