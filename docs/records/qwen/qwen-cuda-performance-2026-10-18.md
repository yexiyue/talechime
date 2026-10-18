# Qwen CUDA 性能调研

本文是最初 0.6B 调研记录；同日后续已补齐 1.7B Base 权重并执行独立 CUDA
验收，见 [1.7B 验收](qwen17-cuda-acceptance-2026-10-18.md)。下文“1.7B 缺权重”
描述的是当时状态，不代表当前缓存。

## 基线与边界

本轮在合并主线 `ca18cd9` 后，检查 Qwen 的 CUDA 路径；不把前一轮 Metal 的收益
直接推广到 CUDA。Windows / RTX 5070 12 GB / CUDA 13.0，Candle 0.11.0 fork revision
`bafb03db81c3eae48e63ff28d6b30f5a6576ba38`，cudarc 0.19.10。
模型为本地完整 Qwen 0.6B Base，Talker/CodePredictor BF16，codec 保留 checkpoint F32。
两份权重 SHA-256 均与仓库 `models/0.6b-base.json` 固定清单一致，模型 revision 为
`5d83992436eae1d760afd27aff78a71d676296fc`。
1.7B 目录缺少主权重，本轮不下载也不把它记为验收对象。

开发探针只读 `target/drift-experiment/round3/qwen-base06` 与已有 ICL prompt；模型、
参考缓存摘要保存在忽略目录 `target/tmp/voxcpm-opt/qwen-cuda/model-hashes.json`。
参考缓存含 1024 维 speaker embedding、126 帧参考 codes 和 28 个参考文本 token。
文本为“山风吹过松林，星光照亮归途。”，seed 42、chunk_frames 10；语义 temperature
0.9、top-k 50、top-p 0.9、repetition_penalty 1.05，声学 argmax，min_new_tokens 2、
max_length 2048、EOS 2150。流式后续块保留 25 帧左上下文，不改变首块参考处理。
完整路径与流式分别计时，入口/出口
设备同步，导出 F32LE/WAV/codes 在计时外。每进程独立加载，一轮完整+流式预热后测量。

请求 SM 时钟 2700 MHz，运行中观测约 2692 MHz；Windows 实际切到高性能计划，结束后
恢复原计划并解除锁频。初始低负载快照仍可能只有 1290 MHz，不能称所有时刻恒为请求值。
桌面 GPU 进程驻留，非完全独占。没有复测官方 PyTorch，也不比较跨平台速度排名。

## CUDA 基础验证

- 新增显式忽略的 CUDA sampler 对照，复用 Metal 的共同测试逻辑：4/3072 词表、
  三个 top-p、16 个 seed；CUDA 与 CPU 策略相同，1 passed。不加载模型或改变采样策略。
- Qwen/Vox CUDA 组合普通库测试：229 passed / 13 ignored；Vox 的四个真模型图测试
  另行串行运行，4 passed，覆盖噪声/schedule/LRU、view 复制、录制失败与 decoder 状态重置。
- main 的顺序 F32 CDF 不保证与旧 GEMM cumsum 逐位一致。Windows 旧测试的相邻 token
  差异与修正边界见 [主线合并记录](../platform/tts-optimization-merge-2026-10-18.md)。

## 首轮 profiling

`qwen_timing_probe` 使用 `cuda,profiling` release 构建；一轮预热、一轮测量。
测量轮完整 3.1961 s、流式 3.7411 s、首 PCM 0.7484 s；两条路径均为 59 帧、113280
samples（4.72 s 音频），EOS，有限 PCM，流式六块覆盖全部 samples。该单轮带 profiling
结果只用于调查，不作为正式无插桩性能结论。
预热与测量的 codes、完整 F32 PCM、流式 F32 PCM 各自 SHA-256 一致；完整与流式 PCM
不要求彼此一致。这证明该输入的固定 seed 复现，不作为跨设备质量或试听验收。

主机 trace 的包含式 span 汇总包含两轮完整+流式，不能相加嵌套 span：

| API 边界 | 次数 | 平均墙钟 |
| --- | ---: | ---: |
| CodePredictor | 236 | 34.256 ms/帧 |
| 其中 14 个 acoustic step | 3304 | 2.272 ms/步 |
| CodePredictor prefill | 236 | 2.314 ms/帧 |
| Talker step | 236 | 13.168 ms/帧 |
| 语义采样 logits 回传 | 240 | 0.198 ms/步 |
| codec upsample stages | 14 | 77.965 ms/次 |

这些 span 可能包含此前 GPU 排队、同步与提交，**不是 GPU kernel 独占耗时或计算占比**。
但该路径没有显示 Metal 上旧 CDF 的毫秒级 CPU 开销：当前采样 CDF/selection 约
0.002 ms/步，CPU top-k/top-p 各约 0.007/0.017 ms。继续重写语义 sampler 的优先级低于
CodePredictor；不能为追求“全 GPU”恢复已拒绝的采样方案。

## 优化候选与架构

先筛选 CodePredictor 的结果打包：15 次 `slice_assign` 在 Candle 内部建立 mask、pad
和 where，并不是真正的原地单元素写。候选仅保留 15 个已生成的设备 tensor，末尾一次
cat，不改 Transformer、argmax、KV、codebook 顺序、归约和采样。结果与收益单独记录。

已完成无 profiling 的独立四进程 ABBA 初筛，每进程一轮预热、两轮热测，固定上述输入。
这不是正式八进程矩阵；初筛未过门槛，不再扩大测量或保留运行时开关。

| 指标 | 基线进程均值的均值 / s | 一次 cat 候选 / s | 变化 |
| --- | ---: | ---: | ---: |
| 完整合成 | 3.09672 | 3.12886 | 慢 1.04% |
| 流式合成 | 3.67450 | 3.65744 | 快 0.46% |
| 首 PCM | 0.73805 | 0.75957 | 慢 2.92% |

完整进程均值区间：基线 [3.06753, 3.12591]，候选 [3.11707, 3.14066]；
流式区间：基线 [3.64451, 3.70449]，候选 [3.65673, 3.65815]。区间重叠，远未达到
10% 收益，**候选已撤回**，最终 CodePredictor 源码与合并后的 main 完全相同。
初筛基线的完整/流式 RTF 分别约 0.6561/0.7785，不能把这组数值与 Metal 的不同
输出帧数、机器或官方测量直接比较。

四个进程（含预热）的 12 份 codes、12 份完整 F32 PCM、12 份流式 F32 PCM、
24 份 WAV，各类型内部 SHA-256 完全一致。都是 59 帧、113280 samples、EOS、有限
输出，证明打包候选保持本输入结果，不能证明它有性能收益或所有语料质量等价。
全部输出在 `target/tmp/voxcpm-opt/qwen-cuda/screen-*`，汇总 `screen-summary.json`；
拒绝的补丁 `packing-candidate.patch` 和两个预编译探针也仅留在忽略目录。

撤回后重新构建标准路径下的 CUDA release 探针，再做一轮预热加一轮回归测量：
10 份 codes/原始 PCM/WAV 制品与基线对应类型逐字节一致，仍为 59 帧、正常 EOS、
有限输出。该运行仅确认最终源码和二进制，未用于重新挑选更快的基线；证据在 `final/`。

后续更有价值的候选是完整 CodePredictor 的 15 步图：每帧内部长度固定为 prefill 2
加 14 步，主机 offset 可以在同一录制中展开，而 Talker 的跨帧动态 KV 不适合直接照搬。
图与 KV 缓冲应归每个生成 session 的专用状态，不能放进共享 `Qwen3TTS` 的全局可变
缓存。真正实现第二个使用者后才抽取通用 recorder；模型 schedule、缓存和失败策略
继续私有，不增加只有一个实现的 trait。

图的推进语义、scratch 输出别名、U32 code 输出、取消/失败后的缓存复用，都必须先做
不同隐藏态的 eager 对照，再验证完整/流式 PCM、EOS、跨请求复用与正式 ABBA。该图
目前是候选，尚未实现，不称为已加速能力。MRoPE 表/位置复用和 codec 卷积为后续候选；
cuDNN feature 会改变算术路径，必须与默认 stock CUDA 独立建立输出与性能证据。

复现入口：

```powershell
cargo build --locked --release -p talechime-qwen3-tts --features cuda,profiling --example qwen_timing_probe -j 1
# MODEL/PROMPT 为已有本地文件，OUTPUT 是新的忽略目录。
target/release/examples/qwen_timing_probe.exe $MODEL $PROMPT '山风吹过松林，星光照亮归途。' cuda $OUTPUT 1 10
```

无插桩计时用 `--features cuda` 重新构建；trace 在调用进程的当前目录写入，必须切到
忽略目录后运行。原始 profiling、主机 span 汇总、逐轮与设备快照在
`target/tmp/voxcpm-opt/qwen-cuda/profile/`。
