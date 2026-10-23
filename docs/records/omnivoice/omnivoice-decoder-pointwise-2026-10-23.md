# OmniVoice decoder：Snake 缓存与 1×1 卷积（2026-10-23）

本轮保留 decoder 专用的通道优先 1×1 GEMM，固定 codes 解码从 176.20 降到
161.85 ms（-8.15%）。Snake 常量缓存撤回，保留 GEMM 原方向的另一候选也未采用。
整链观测均值变化约 -0.37%，小于进程波动，不能据此确认整链提速。

## 身份与测量边界

- 源码基线 `7c17e0e3de8767223cc307425471ef527758e605`，分支 `codex/omnivoice-hotspot`。
- M4 Pro，24 GiB，macOS 26.6.2；stage0 Metal F16，decoder Metal F32。
- Candle 固定 `2266099d92265112ecc0d56ecee1761e3cc2ac00`，没有改依赖或内核。
- 本机既有 0.6b 权重 revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`，不下载、不转换。
- Rayon / VecLib / OMP 各设为 8，推理串行，先完成编译再测量。
- `target/perf/omni-decoder/manifest.json` 保存原/候选二进制、两个 safetensors、
  配置、输入、源码和脚本的 SHA256；没有把模型、音频或正文提交到仓库。

decoder 独立二进制启用 `metal`；整链两侧均为
`--no-default-features --features omnivoice-metal,performance-tools`。
模型加载、WAV 导出不计入热解码；首轮 decode 另记，包含懒加载及内核预热，
不是操作系统/Metal shader cache 完全冷启动。第一次基线 1 帧首调 4.27 s，后续进程
已共享系统缓存，不能与后面的首调直接相减声称启动提速。

关键文件 SHA256（完整清单另见 manifest）：

| 文件 | SHA256 |
| --- | --- |
| decoder 权重 `audio_tokenizer/model.safetensors` | `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d` |
| 原 decoder 二进制 `stage1-base` | `476e32d393e447fe7b23853a4bcf0e4db7a1961583b456da472969d4cdb0d4b1` |
| 最终 decoder 二进制 `stage1-pointwise` | `6b47cd28d8bbc9f607b478220d1593df767dcfa558302e46597a8ba375517e16` |
| 原整链二进制 `probe-base` | `6a0b4bb94d3afdc68f41ddde781d21ca014eb96bdc902376309d2b3be7834ace` |
| 最终整链二进制 `probe-pointwise` | `f5c6356b8972424866eb9269b5c87cd44fbde7f4e5f752d7f99553b06b1ff9a2` |

## 1. 当前基线与固定输入

固定 codes 从已有通过验证的产物裁取，每个 codebook 独立截取，不改变行排列。
另从长文本 codes dump 取最长的 179 帧真实分段。每个基线进程 1 预热 + 5 热轮：

| 帧数 | 采样率 | 样本数 | 基线热轮均值 |
| --- | --- | --- | --- |
| 1 | 24000 | 960 | 6.464 ms，存在启动后逐轮下降 |
| 7 | 24000 | 6720 | 12.168 ms |
| 130 | 24000 | 124800 | 176.359 ms |
| 179 | 24000 | 171840 | 247.399 ms |

这一组用于建立输入和输出参照；优化百分比只用后面的同时间窗口配对基准。

## 2. Snake 常量缓存：输出相同，收益未确认，撤回

试验在 Metal F32 decoder 加载时计算 `(alpha + 1e-9).recip()`，forward 复用；
CPU、其他 dtype、参考音频编码不动。单元对照覆盖 1/7/130/179 长度、batch 1/2、
连续与非连续输入、零/正/负 alpha 及重复调用，输出 F32 位模式相同。
四组固定 codes 的完整 raw F32 WAV 也逐字节一致。

离线探针在三个独立进程对五个上采样块入口的真实 Snake 权重做 ABBA。
130 帧输入下各块原/缓存均值（µs）为：

| 块 | 原路径 | 缓存 |
| --- | --- | --- |
| 0 | 226.997 | 237.753 |
| 1 | 372.747 | 380.655 |
| 2 | 830.712 | 852.976 |
| 3 | 1642.465 | 1630.036 |
| 4 | 1574.706 | 1548.991 |

补测 1、7 帧亦有正负变化，没有稳定的局部改善。完整 130 帧 decoder ABBA：
A 两进程 176.921 / 176.736 ms，B 两进程 176.405 / 175.819 ms；
组均值 176.828 → 176.112 ms（约 -0.40%）。这个幅度与本轮基线漂移同量级，
不能确认归因于缓存；不因某个均值略低就保留额外缓存状态。

**撤回不是因为低于某个百分比门槛，而是缺少稳定收益证据。** 生产源码已恢复原 Snake，
候选补丁保存在忽略目录 `snake-cache.patch`，没有运行时开关或遗留缓存字段。

## 3. 1×1 卷积与布局：选用直接 NCL GEMM

固定 Candle Metal conv1d 的原路径为 im2col → GEMM → 输出 transpose copy。
真实权重离线探针覆盖五块、每块三个残差单元的 15 个 1×1 层，分别比较：

- 原路径：Candle Conv1d。
- BTC 候选：输入转为连续 BTC，保持原 GEMM 方向，输出转回连续 NCL，再加 bias。
- NCL 候选：`weight[Co,Ci] × input[B,Ci,T]`，直接得到连续 `[B,Co,T]`，再加 bias。

三进程内部按 ABCCBA 交错；每组 3 次预热、10 次计时，组边界同步。
每格为该进程所有 15 层两组样本的等权平均，单位 µs：

| 路径 | 算子进程 1/2/3 | 算子均值 | 带消费均值 |
| --- | --- | --- | --- |
| 原路径 | 1682.609 / 1685.884 / 1835.824 | 1734.772 | 3106.032 |
| BTC 候选 | 1993.714 / 2021.251 / 2152.839 | 2055.934 | 3365.721 |
| NCL 候选 | 653.353 / 657.857 / 718.528 | 676.579 | 2029.954 |

三进程所有真实层的比较均为零 bit 差异。NCL 候选算子样本约 -61.0%，
带消费约 -34.6%；BTC 候选更慢，未进入生产路径。
消费探针用相同形状做残差相加后再执行真实权重 Snake，是合成的下游消费场景，
不是从模型截获的完整残差轨迹。它包含后续计算成本，不能代替整段解码验收。

7-tap 卷积只做诊断，五块各三个 dilation 层的等权均值约
2.40 / 4.63 / 8.41 / 7.70 / 11.38 ms；还记录了输入 transpose/contiguous。
这些单独同步的算子耗时不相加作为生产 decoder 时间。本轮没有改 7-tap 算法。

### 生产路由与回退

只有 `stage1_model` 的残差单元 conv2 显式调用 crate 内 decoder 专用方法。
公共 `Module::forward` 仍用原路径，参考音频编码不因此切换算法。
匹配 dense Metal F32、kernel/stride/dilation/groups=1、padding=0、输入和权重连续的
配置才直接 GEMM；其他设备、dtype、非连续输入、量化权重及其他卷积配置回退。
不新增公开 API、模型文件、共享框架或运行时实验开关。

### 完整 decoder 配对

固定 8×130 codes，ABBA，每进程 1 预热 + 5 热轮。以下均为 ms：

| 顺序 | 路径 | 5 个热轮 | 热均值 | 首调（非全冷） |
| --- | --- | --- | --- | --- |
| A1 | 原路径 | 175.406 / 175.382 / 175.531 / 176.614 / 178.088 | 176.204 | 205.929 |
| B2 | NCL GEMM | 161.373 / 161.471 / 162.065 / 162.817 / 161.024 | 161.750 | 182.304 |
| B3 | NCL GEMM | 161.145 / 162.684 / 162.623 / 161.640 / 161.668 | 161.952 | 183.522 |
| A4 | 原路径 | 178.206 / 176.584 / 175.155 / 175.562 / 175.503 | 176.202 | 197.625 |

先算每进程均值，再算组均值：**176.203 → 161.851 ms，减少 14.352 ms（8.15%）**。
这是本轮保留依据；它独立于已经撤回的 Snake 缓存。

## 4. 整链及输出验证

短文本默认 narrator、seed 42、32 步，ABBA，每进程 1 预热 + 3 热轮，诊断全部关闭：

| 顺序 | 热轮耗时（s） | 进程均值（s） |
| --- | --- | --- |
| A1 | 3.02608 / 3.08638 / 3.09528 | 3.06925 |
| B2 | 3.09243 / 3.09248 / 3.07312 | 3.08601 |
| B3 | 2.97162 / 2.95854 / 2.94459 | 2.95825 |
| A4 | 2.97237 / 2.97054 / 3.05020 | 2.99770 |

组均值 3.03348 → 3.02213 s，观测差约 -0.37%。两侧进程范围明显交叠，
差异小于波动，**不确认整链提速**。所有热轮 F32 WAV SHA256 相同：
`2eadc756eb4482fe320efe678ef8a67312ce79dcfa6d6f4dbf41ed4642be324e`。
本轮正式局部、固定 codes 和整链计时均未观察到编译器活动；GUI/系统负载及频率未锁定。
不把局部比例与前几轮直接相加，没有重新测官方整链，也不更新官方相对速度。

1/7/130/179 帧候选与冻结基线的完整 raw F32 WAV 均逐字节一致。
正文短/长文本另启 codes dump 做完整对照，耗时不用于吞吐结论；基线产物先核对原摘要
SHA256 后复用，生成配置、权重及二进制与 `7c17e0e` 对应。

最终正文验收结果：短文本 codes dump SHA256 为
`d1f037724f6a674505e3e0f52c5ae23d3c341d95cefb6c7ad74d7a7015d829c3`，
长文本为 `69dd8d8476dcb38aef411a588461245ae39026dc9159c4d6af63d5bdbb42b29c`，
均与冻结基线一致；短文本 4/4 WAV、长文本 24/24 WAV 逐字节一致。
codes 包含参考准备和全部正文分段，不仅比较最终 PCM16。

## 5. 最终检查与交付

| 检查 | 结果 |
| --- | --- |
| `cargo fmt --all --check` / `git diff --check` | 通过 |
| Metal crate lib tests | 32 passed |
| 无默认 feature 的 CPU lib tests | 23 passed |
| Metal + Accelerate，Clippy all-targets / `-D warnings` | 通过 |
| Metal rustdoc，library + examples / `-D warnings` | 通过 |

新增测试覆盖奇数通道、矩形投影、有无 bias、batch 1/2、1/7/130/179 长度、
连续/非连续布局和重复调用；stride、padding、dilation、groups、CPU、F16 的回退
与原路径比较。一般 `Module::forward` 仍与原卷积结果一致。
Snake 试验测试随撤回补丁留在忽略目录，不给生产代码保留失败候选。

普通测试不下载模型。没有改会话、取消或采样行为，没有 push、合并、发包或部署。

## 复现与产物

```sh
cargo build --release --locked -p talechime-omnivoice --features metal --example stage1_probe --example decoder_ops_probe
target/release/examples/decoder_ops_probe MODEL 130
target/release/examples/stage1_probe MODEL CODES_JSON metal OUTPUT_DIR 5
```

原始数据位于 `target/perf/omni-decoder/`：`baseline-*`、`ops-*`、
`decoder-*`（Snake）、`pointwise-decoder-*`、`pointwise-e2e-*`、`pointwise-full-*`，
以及每轮独立的 `.activity.json`。三种输入帧数的算子日志、逐轮计时与文件摘要均保留。
