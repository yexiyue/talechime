# OmniVoice 后续热点与 7-tap 布局初筛（2026-10-23）

基线 `6a7c0a5`，已包含免 Q/K 布局复制。没有修改生产计算路径；本轮新增离线
`conv7_layout_probe`，排除两个变慢的通用卷积候选，整理下一轮实验优先级。

## 当前阶段诊断

复用上一轮独立构建并验证的 `target/perf/omni-audit/probe-current`，SHA256
`8cfb1b4c82d0bbc56013da5cac0f10ed34add04bf21b104fd2c629b1d303f0ef`。
M4 Pro / 24 GiB / macOS 26.6.2，固定 0.6b 权重 revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4`、Candle `2266099`。
stage0 Metal F16、decoder Metal F32；Rayon / VecLib / OMP 各 8。

短文本、seed 42、narrator、默认 32 步，1 预热 + 1 热轮。启用
`OMNIVOICE_SYNC_PROBE=1` 和 `OMNIVOICE_STAGE_TIME=1`，正式吞吐不使用这些设置。
只取最后一个 130 帧正文调用的 32 步，排除 48 帧参考准备和正文预热：

| 阶段 | 每步同步墙钟均值 |
| --- | ---: |
| stage0 forward | 76.511 ms |
| logits 转型 | 1.417 ms |
| predict / scoring | 4.131 ms |
| 更新（含下面的排序） | 0.810 ms |
| 排序 | 0.211 ms |
| 写回 | 0.558 ms |

该次完整 decoder 诊断耗时 170.87 ms。同步探针还每步运行额外 canary GEMM，
均值 3.543 ms；它不是生产计算。不能相加上述耗时形成生产占比，也不能与
上一轮无插桩的 2.672 s 直接相减声称退化。没有插入逐层同步。
日志 `target/perf/omni-next/profile.err`；原始轮次、阶段摘要和编译活动采样均保留。

## 7-tap：两个通用布局候选均不保留

固定 fork 的 Metal `conv1d` 为专用 im2col → GEMM → 输出转置物化。
残差单元 7-tap 的收缩索引顺序是 `[input_channel, tap]`。
用 `pad_with_zeros + unfold` 构造同一个窗口，然后比较：

1. 原 Candle Conv1d。
2. BTC 窗口 `[B,T,C*7]` × 转置权重，输出转回连续 NCL。
3. NCL 窗口 `[B,C*7,T]`，用权重左乘，直接产出 NCL。

两种候选都包含 padding、窗口物化、GEMM 和 bias。另测后续真实权重 Snake，
没有通过把布局成本转移到下一算子来隐藏代价。没有改卷积求和为逐 tap 累加，
没有降低 dtype 或替换近似激活函数。

离线输入为确定性合成激活，权重读取本机已安装模型，**不等于真实模型中间激活**。
帧数 130、batch 1，覆盖 5 块 × dilation 1/3/9 的 15 层。每个独立进程内逐层
按 ABCCBA 交错，每组 2 次预热 + 4 次计时，仅组边界同步。
三次独立进程，无本任务编译/推理重叠；每 0.5 s 按进程 basename 采样，
本轮阶段诊断与三进程均未观测到 cargo/rustc/clang/ld/clippy-driver。
未独占机器、未锁频，未排除全部 GUI/系统活动。

以下为各进程 15 层的等权均值，单位 ms；不是整个 decoder 的耗时：

| 路径 | 进程 1 / 2 / 3 | 带 Snake 进程 1 / 2 / 3 |
| --- | --- | --- |
| 原路径 | 7.154 / 7.084 / 7.197 | 8.508 / 8.433 / 8.664 |
| BTC | 15.630 / 15.583 / 15.764 | 16.890 / 16.896 / 17.034 |
| NCL | 8.550 / 8.496 / 8.568 | 9.867 / 9.781 / 9.888 |

卷积全部元素零 bit 差异，但 NCL 仍慢约 19.5%，BTC 慢约 119%。带 Snake 也变慢。
因此两个候选都不进入生产；只保留复现探针。没有为它们跑整链，亦不宣称整链变化。
这只拒绝这两种通用展开写法，不拒绝专用内核、不同 tile 或其他卷积算法。

窗口展开的最大张量是 `32 * 124800 * 7 * 4 = 111820800` bytes（约 112 MB）。
显式 NCL 候选最后一块窗口物化明显贵于它自己的预物化 GEMM；这一诊断不能直接
充当原路径 im2col 的独立计时。组件探针复用固定窗口/权重，缓存条件也不同。
下一轮需要单独测原专用 im2col 和输出复制，再判断是否值得改底层实现。

尚未覆盖该候选的短帧、多 batch、非连续输入、完整 decoder/长文本和取消复用；
因为候选已变慢，本轮不再扩大验收。没有修改任何依赖、公共 API、采样和资源清单。

## 下一轮优先级

| 优先级 | 候选 | 当前依据与第一道门禁 |
| --- | --- | --- |
| 1 | 7-tap 专用展开 / 输出布局 | 有真实层局部成本与大展开张量；先分离原 im2col/复制，验证更高效的展开实现。通用 unfold 已失败，不重复尝试；涉及专用内核时另立可审查实验 |
| 2 | attention context 写回布局 | Q/K 已免复制，但 SDPA 的 `[B,H,S,D]` 输出仍需物化到 `[B,S,H,D]`；先核查输出写入机制，验证同样计算、不同写入布局，不由独立复制微秒数外推整链收益 |
| 3 | scoring 词表 mask / 小算子链 | 每步仍有三次 log_softmax 和 mask 拼接。先测仅默认贪心、mask 位于词表末端时用有效词表视图替代拼接，保持完整 tokens/confidence；若涉及融合则逐操作保持舍入，禁止代数消去 log_softmax 冒充逐位一致 |
| 4 | 合并 gate/up，再评估 Q/K/V 投影 | 仅有历史合成微基准线索，收益尚未由真实权重证明。先单层真实权重、完整消费与位模式对照，再决定是否承担权重布局及量化分支维护成本；本轮未实施 |
| 5 | 请求内静态 mask / 索引的重复构造 | audio_mask 的 I64 转换、selection mask 等仍逐步准备；规模小，先测是否在关键路径，不能只因重复就增加常驻缓存 |

排序现约 0.21 ms/步，先不改 top-k 算法。Snake 常量缓存、两种 logits 转型已经被
前轮实测否定，不重新包装成新优化。双向去噪的深层 K/V 会随音频改变，不能套用
自回归前缀 KV cache。减少步数、改精度和近似激活不属于本轮保持输出的优化。
优先级综合成本、证据和维护风险，不承诺任何未经测量的收益比例。

## 复现与检查

```sh
cargo build --release --locked -p talechime-omnivoice --features metal --example conv7_layout_probe
target/release/examples/conv7_layout_probe MODEL 130
```

原始逐组数据 `target/perf/omni-next/conv7-{1,2,3}.jsonl`，
逐进程摘要 `conv7-summary.json`，二进制/源码/锁文件及数据哈希 `manifest.json`。
本轮没有重测官方，没有更新官方相对速度；未推送、合并、发包或部署。

检查通过：Metal lib 34 tests、CPU lib 23 tests，Metal all-targets Clippy
`-D warnings`，Metal library/examples rustdoc `-D warnings`，格式与 diff 检查。
普通测试未下载模型；本轮只提交离线探针和记录，不保留失败候选的运行时开关。
