# OmniVoice 原生 7-tap 展开与输出布局实验（2026-10-23）

基线 `c9bb028`。本轮只提交离线探针、候选补丁与记录，不改变项目依赖或生产计算。
原生 im2col 不变、改变 GEMM 方向以省输出复制的候选，在固定 codes decoder ABBA
中从 162.680 降到 154.642 ms（约 -4.94%），完整 F32 WAV 不变。
另一个专用 NCL 展开候选变慢，mask 视图候选无稳定收益，均不进入生产。

## 身份与隔离

M4 Pro / 24 GiB / macOS 26.6.2。权重 revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4`，stage0 Metal F16，decoder Metal F32。
Candle 基线 `2266099d92265112ecc0d56ecee1761e3cc2ac00`，Rayon / VecLib / OMP 各 8。
沿用既有固定 1/7/130/179 帧 codes，不下载或转换资源。

Tensor 公开 API 无法单独调用原生 im2col。离线 scratch manifest 直接引用已经锁定的
`candle-metal-kernels`，没有给项目添加依赖或修改 Cargo.lock。解码候选另复制一份
Candle 到忽略目录，以 scratch manifest 的 path patch 接入；共享 Cargo checkout 未改。
复制版 package 增加显式 workspace 路径以解决嵌套工作区继承，不改变其他算术。
本轮 release 配置仍为 opt-level 3 / LTO / panic abort，先编译再串行推理。

原始文件、脚本、二进制在 `target/perf/omni-native/`，以下数字只引用对应的新实验。
历史首组 `native-*` 三进程都捕获到了外部编译，保留但不用于最终性能依据。
随后 `quiet-*` 三进程全部未观测到编译。进程监控使用 basename，包含 cargo、
cargo-clippy、rustc、clang、ld、clippy-driver；采样间隔 0.5 s。
机器未锁频、未独占，没有排除全部系统/GUI 干扰。

## 1. 原生阶段分离

`examples/support/conv7_native_probe.rs` 使用真实残差层权重与确定性合成激活。
它是独立 manifest 的入口，不由普通 examples 自动发现，也不在普通测试中下载模型。
原生 `call_im2col1d_strided` 的调用参数、输出布局、dtype 与固定 fork 一致；
不再用通用 unfold 的耗时冒充原生展开。

每个进程覆盖 5 块 × dilation 1/3/9 的 15 层，batch 1、130 帧几何。每层顺序
ABCCBA，每组 2 次预热 + 4 次热测，组边界同步：A 原 Conv1d；B 重建原生
im2col/GEMM/输出复制；C 同一原生 im2col，但 GEMM 直接写 NCL。
所有卷积输出零 bit 差异，另将后续真实权重 Snake 纳入消费测量。

| 路径 | 三进程各 15 层等权均值（ms） | 带 Snake（ms） |
| --- | --- | --- |
| 原路径 A | 7.607 / 7.284 / 7.168 | 9.076 / 8.752 / 8.646 |
| 重建原路径 B | 7.535 / 7.265 / 7.208 | 8.993 / 8.692 / 8.619 |
| 直接 NCL 输出 C | 7.022 / 6.698 / 6.676 | 8.348 / 8.061 / 8.089 |

以下为原生组件隔离测量的三进程/三 dilation/两个测量位置等权平均，单位 ms。
GEMM 列混合了两种矩阵方向，输出列分别包含复制+bias 或仅 bias；只用于定位，
不作为原完整 conv 的精确占比，不相加模拟生产 decoder。

| 块 | 原生 im2col（含分配） | 预展开 GEMM | 输出处理 |
| --- | ---: | ---: | ---: |
| 0 | 1.164 | 0.949 | 0.291 |
| 1 | 3.084 | 1.078 | 0.575 |
| 2 | 7.035 | 1.115 | 1.032 |
| 3 | 6.537 | 0.684 | 0.962 |
| 4 | 9.629 | 0.900 | 1.365 |

这些是同步墙钟，不是 GPU timestamp。预展开 GEMM 复用固定输入与权重，工作集
和完整调用不同；不能把这张表直接换算成 DRAM 带宽或整链优化上限。

## 2. 直接输出 NCL：局部通过，保留为离线候选

保持 im2col `[B,T,Cin*7]` 不变，以 `weight[B,Cout,Cin*7] × col.transpose(1,2)`
产生连续 NCL，省掉输出转置物化及其零初始化。bias 和后续 Snake 仍计入。
补丁为同目录 `omnivoice-conv7-ncl-candidate-2026-10-23.patch`。

隔离 Candle 补丁仅在 Metal F32、kernel 7、stride 1、padding=3*dilation、
Cin=Cout、连续权重时启用。其他条件保留原路径。这只是实验匹配范围，
不是已验收的通用 Candle fast path；生产接入仍需限制在 decoder 私有入口。

完整固定 130 帧 decoder，ABBA，各进程 1 次预热 + 5 次热轮：

| 顺序 | 热均值（ms） | F32 WAV |
| --- | ---: | --- |
| A1 | 163.588 | 相同 |
| B2 | 154.094 | 相同 |
| B3 | 155.190 | 相同 |
| A4 | 161.772 | 相同 |

两侧 mean-of-means **162.680 → 154.642 ms，-4.94%**。四进程未观测到编译。
130 帧原始 WAV SHA256 `2019972a03d3fb0a46d2709263350e84dda6199a0bd1bd0875b9c8078756197f`。
1/7/179 帧各再运行 A/B 完整解码，样本数分别 960/6720/171840，WAV 全部逐字节一致。
首调含懒加载/系统 shader cache，未将首调差异当作冷启动收益。

另以仅第 2 节补丁构建真实适配器 performance_probe，对短/长文本核对冻结基线摘要：
完整 codes 和全部 28 个分段/整段 F32/PCM16 WAV 逐字节一致，包含样本数与采样率。
短文本 codes SHA256 `d1f037724f6a674505e3e0f52c5ae23d3c341d95cefb6c7ad74d7a7015d829c3`，
长文本 `69dd8d8476dcb38aef411a588461245ae39026dc9159c4d6af63d5bdbb42b29c`。
这不是用固定 seed 代替输出比较，也不代表和官方逐位/音质等价。

随后无插桩短文本 ABBA、各 1 预热 + 3 热轮，仅完成 A1/B2 后，B2 捕获到外部 Cargo，
启动第三个进程前仍有编译，因此停止该组性能验收。原始 `e2e-1-A` / `e2e-2-B`
全部保留，不给出未完成/受干扰组的整链提速比例；本轮没有可信整链提速结论。

## 3. 专用 NCL 展开：完整解码变慢，撤回

另在隔离 Candle 副本新增纯复制的 NCL im2col shader，输出 `[B,Cin*7,T]`，
以时间轴连续写入、相同 `[input_channel,tap]` 收缩索引顺序做 GEMM。
没有改 dtype、padding/dilation 语义或逐 tap 分块累加。

独立 ABBA，1 预热 + 5 热轮：A 167.839 / 170.863 ms，B 185.850 / 185.951 ms，
组均值 **169.351 → 185.901 ms，慢 9.77%**。无编译观测。
四种帧数 raw F32 WAV 仍逐字节一致；因为完整解码已变慢，不再扩大验收。
没有将该 shader 保留在项目或候选补丁中，副本恢复到第 2 节版本后再构建整链。
不能仅凭连续访存推断收益；尚未把该退化单独归因于 shader 或 GEMM tile。
这一组只能和自己的 A 对照，不能与上一组较低的基线直接比较。

## 4. 贪心 mask 视图：无可重复收益，撤回

原路径在三次 log_softmax 后拼接被屏蔽的最后一列。试验只用有效词表视图做
argmax，同时从原连续 log_probs gather confidence，保持归一化和浮点算术不动。

最初直接从窄视图 gather confidence 的探针未通过数值检查：固定 Metal gather
只传递源 dims、未传源 strides，不能将这种非连续源当作支持的布局。
原生产路径的 mask 拼接产生连续张量，没有为这个失败候选改动生产代码。
修正为从原连续张量 gather 后，通过完整 tokens/confidence 位模式检查再计时。

三个独立进程各 ABBA，每组 100 次：完整 scoring 原/视图均值（µs）分别为
2671.790 / 2667.037、2625.854 / 2696.203、2694.569 / 2680.747。
变化为 -0.18%、+2.68%、-0.51%，缺少可重复收益；未进入生产，无整链提速结论。
失败的首个版本日志与修正版分开保存。`scoring_probe` 保留正确的离线复现分支。

## 接入边界

用户此前要求“不改变公共 API、资源清单、采样参数或依赖”，项目继续遵守这一边界。
第 2 节的收益足以保留具体候选，不以默认整链百分比门槛拒绝它；但当前实现依赖
Candle 内部 im2col 入口，不能直接在现有模型 API 中表达，所以尚未启用到生产。
后续若允许接入，可比较模型私有 Metal 实现直接引用现有固定版本 kernel crate，
或更新固定 Candle 补丁两种方式；需保留 decoder 限定入口及完整回退，不扩大到所有模型。
当前没有修改依赖清单、会话/取消、参考编码策略或采样参数，没有 push、合并或发布。

## 离线复现

原生探针需要单独的 manifest，避免给模型库增加直接依赖。例如在
`target/perf/omni-native-repro/Cargo.toml` 放入下面的配置，并复制仓库 Cargo.lock 到
同目录作为初始锁文件；`cargo build --offline` 只调整该 scratch 锁文件。
四个 Candle patch 与项目固定 revision 相同。

```toml
[package]
name = "conv7-native-audit"
version = "0.0.0"
edition = "2024"
[workspace]
[[bin]]
name = "conv7-native-audit"
path = "../../../crates/omnivoice/examples/support/conv7_native_probe.rs"
[dependencies]
candle-core = { version = "=0.11.0", features = ["metal"] }
candle-nn = { version = "=0.11.0", features = ["metal"] }
candle-metal-kernels = "=0.11.0"
serde_json = "1"
talechime-omnivoice = { path = "../../../crates/omnivoice", features = ["metal"] }
[profile.release]
lto = true
opt-level = 3
panic = "abort"
[patch.crates-io]
candle-core = { git = "https://github.com/yexiyue/candle", rev = "2266099d92265112ecc0d56ecee1761e3cc2ac00" }
candle-nn = { git = "https://github.com/yexiyue/candle", rev = "2266099d92265112ecc0d56ecee1761e3cc2ac00" }
candle-transformers = { git = "https://github.com/yexiyue/candle", rev = "2266099d92265112ecc0d56ecee1761e3cc2ac00" }
candle-metal-kernels = { git = "https://github.com/yexiyue/candle", rev = "2266099d92265112ecc0d56ecee1761e3cc2ac00" }
```

```sh
cargo build --release --offline --manifest-path target/perf/omni-native-repro/Cargo.toml
target/perf/omni-native-repro/target/release/conv7-native-audit MODEL 130
```

如需重复完整 decoder 候选，另复制固定 Candle 源码副本，应用同目录候选 patch，
将 scratch manifest 的四个 patch 改为该副本的对应 crate 路径，再以仓库
`examples/stage1_probe.rs` 为入口构建 A/B。复制版含 workspace 继承的 package 需
显式指定 `workspace = ".."`，避免继承外层 scratch workspace。
不要在共享 Cargo checkout 或项目根依赖上直接应用实验。

逐进程原始数据：`quiet-{1,2,3}.jsonl`、`decoder-abba.json`、`decoder-parity.json`、
`ncl-decoder-abba.json`、`ncl-decoder-parity.json`、`mask-fixed-{1,2,3}.txt`，以及
`full-parity.json`；完整源码、二进制、权重/输入、脚本和数据哈希在 `manifest.json`。
保留候选尚未覆盖所有 batch、非连续布局、空/异常形状、CPU/CUDA/dtype 路由及取消复用，
不将当前验证描述为通用后端补丁的完整验收。

本轮重新检查模型权重、配置及四份 codes 的 SHA256 与既有 manifest 一致。
主要二进制身份（完整列表见忽略目录 manifest）：

| 二进制 | SHA256 |
| --- | --- |
| 固定 codes 原版 | `24b441f49419659be6871c1176e2bf951ef720534f1e851e535859bf2e7601b8` |
| 固定 codes 输出布局候选 | `60ba3fcafcbbc36cc87f307fc9614e98da3fa962fb805a3131345ddca3df6444` |
| 整链输出布局候选 | `e8feeef0710c4a95285e566c2e782c776a8a1606507fcfeaf1191edc75ac782f` |

工程检查：项目 Metal lib 34 tests、CPU lib 23 tests，Metal + Accelerate all-targets
Clippy `-D warnings`，Metal library/examples rustdoc `-D warnings`，格式及 diff 检查通过。
独立原生探针另经 scratch Clippy `-D warnings` 检查；路径版 Candle 的既有依赖警告
保留在日志，不宣称对 Candle 全库执行了无警告验收。复现 manifest 通过 metadata，
候选 patch 通过固定 Candle 原版的 `git apply --check`。
