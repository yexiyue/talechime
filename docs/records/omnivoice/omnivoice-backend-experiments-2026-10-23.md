# OmniVoice 剩余性能差距实验（2026-10-23，M4 Pro）

本记录验证三个问题：CPU Accelerate 是否缺失、stage1 的设备选择是否合理、
融合后与官方的剩余差距来自哪里。接续 [融合实验](omnivoice-optimization-2026-10-22.md)。

## 身份与范围

- 源码基线：`6ebc58b886e95aa2f6f4003a7731b8a9b034490b`，`codex/omnivoice-hotspot`。
- Candle：`2266099d92265112ecc0d56ecee1761e3cc2ac00`，统一 0.11.0。
- 官方：`k2-fsa/OmniVoice`，`08be0b4ccbac3e13e374e86fbfead4b4cac343e2`，
  本地 `/Users/yexiyue/.cache/talechime-ref/OmniVoice`；torch 2.8.0 / transformers 5.3.0。
- 实测机器：Apple M4 Pro，12 CPU 核 / 16 GPU 核 / 24 GiB，macOS 26.6.2（25G83）。
- 模型：生产清单 revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`，本机现有权重，
  不下载、不转换；stage0 F16，stage1 F32。
- 三个线程环境变量统一为 8：`RAYON_NUM_THREADS`、`VECLIB_MAXIMUM_THREADS`、
  `OMP_NUM_THREADS`。官方报告 intra-op 8、inter-op 12；这不等于两套运行时线程调度完全相同。
- 全部 release；先构建，再串行测量，WAV 写出和数值比较在计时区外。
- 诊断归因、固定输入的解码比较、端到端 TTS 比较分开。相同 seed 不代表跨框架 codes 一样。

新增离线探针：`crates/omnivoice/examples/stage1_probe.rs`、`conv_transpose_probe.rs`。
初始整链候选通过临时补丁构建，基线补丁保存在
[实验补丁](omnivoice-col2im-candidate-2026-10-23.patch)。用户随后明确认可约 8% 的
整链收益，本轮按有记录的例外保留 Metal 实现，不机械套用默认 10% 门槛。
最终实现仅对 contiguous 输入/权重的 dense Metal F32、groups=1、dilation=1、
padding>0 且 output_padding<=padding 启用；其他路径调用原实现。
设备分工、默认 feature、CPU/CUDA/GGUF 路径不变，未合并或发布。

## 1. 固定 codes：CPU、Accelerate、Metal、官方 CPU

输入取自上轮融合版 codes dump 的第二个记录（跳过参考预热）：8 码本 × 130 帧，
同一份 JSON 输入传给四条路径，排除 stage0 随机数与输出长度干扰。

- 输入 JSON SHA256：`a6aa33dfa343347c34455038fa12f006a9001b561cdf6d6cb0844d885e7b7c44`。
- 源 dump SHA256：`a31cb6b25f2ebc0b0e598355c713ac3af03be5c10967bcf46d3c11397dce073c`。
- 每个配置 3 个独立进程，每进程 1 次冷启动/预热 + 5 次热测。
- 轮换顺序：Metal / CPU / Accelerate / 官方；反向；CPU / Metal / 官方 / Accelerate。
- 计时边界：`Stage1RuntimePlan::decode_raw`，包括 codes 搬移、完整解码与输出回读；
  官方对应 `audio_tokenizer.decode(...).audio_values.cpu()`，不含后处理。
- 全部输出相同长度：124800 样本 / 24 kHz / 5.20 s 原始 PCM。

| 路径 | 三进程均值的均值 | 进程均值范围 |
| --- | ---: | --- |
| Candle Metal F32 | 597.47 ms | 595.82–599.71 ms |
| Candle CPU 默认 | 667.48 ms | 643.21–686.91 ms |
| Candle CPU + Accelerate | 478.16 ms | 461.11–507.86 ms |
| 官方 PyTorch CPU F32 | 135.02 ms | 134.10–135.85 ms |

依赖树和二进制链接确认：默认探针没有启用 `accelerate-src`；显式构建
`--features metal,candle-core/accelerate` 后启用并链接 Accelerate.framework。
官方 PyTorch build config 的 `BLAS_INFO=accelerate`。这证明 BLAS 构建路径，
不证明每个算子都走 Accelerate，更不证明使用了某个具体 AMX kernel。

结论：Accelerate 确实缺失，对这组 CPU 解码改善约 28.4%，但直接切到 CPU +
Accelerate 仍远慢于官方。它不会替换 Metal 上的主干运算。

### 固定输入数值比较

以官方 CPU F32 的原始 PCM 为参考，比较完整 124800 个样本：

| 路径 | max_abs | relative RMS | SNR |
| --- | ---: | ---: | ---: |
| Candle Metal | 3.93e-6 | 1.42e-6 | 116.96 dB |
| Candle CPU 默认 | 6.20e-6 | 1.48e-6 | 116.58 dB |
| Candle CPU + Accelerate | 3.97e-6 | 1.27e-6 | 117.91 dB |

本样本未发现解码布局、长度或大幅数值错误；不外推为所有音色、长度或精度的正确性保证。

## 2. 定位并验证反卷积慢路径

源码链：`stage1_model::DecoderBlock` → `CodecConvTranspose1d` → Candle
`conv_transpose1d`。本模型五个上采样层采用 stride `[8,5,4,2,3]`，
padding 分别 `[4,3,2,1,2]`，output_padding `[0,1,0,0,1]`。

Candle CPU 与 Metal 的快速 GEMM + col2im 路径要求：kernel contiguous、dilation=1、
padding=0、output_padding=0。当前五层的非零 padding 全部使其落入通用实现。
Accelerate 只能改善其覆盖的矩阵乘，不能把通用反卷积自动改成 GEMM。

实验改写：保持原权重、bias、stride，先用零 padding / 零 output_padding 反卷积，
再从左侧 `padding` 位置裁剪，保留长度：

```text
full_length - 2 * padding + output_padding
```

只对本次验证的 groups=1、dilation=1、output_padding<=padding 情况启用；
裁剪后再加相同 bias。odd stride 5、3 对应的 output_padding=1 也包含在验证中。
数学等价不代表浮点逐位等价。

### 真实权重逐层探针

固定合成输入、真实五层权重，分别预热两条路径，再做 5 轮同步计时。

| 层 / Metal 输入形状 | 原路径均值 | 零 padding 后裁剪 | 相对 RMS 误差 |
| --- | ---: | ---: | ---: |
| 0 / `[1,1024,130]` | 150.63 ms | 1.60 ms | 1.04e-7 |
| 1 / `[1,512,1040]` | 117.88 ms | 2.34 ms | 8.25e-8 |
| 2 / `[1,256,5200]` | 96.57 ms | 3.84 ms | 5.70e-8 |
| 3 / `[1,128,20800]` | 27.62 ms | 3.33 ms | 7.20e-8 |
| 4 / `[1,64,41600]` | 35.49 ms | 3.97 ms | 3.92e-8 |

约 428 ms → 15 ms，与完整 stage1 的约 418 ms 改善量级吻合，因果证据强于
仅观察 CPU/Metal 差异。这是具体算子实现路径造成的效率问题，未发现功能性错误。

### 完整固定 codes 候选（筛选协议）

| 路径 | 原路径（三进程） | 候选（1 预热 + 5 热测，单进程） |
| --- | ---: | ---: |
| Metal F32 | 597.47 ms | 179.81 ms |
| CPU 默认 | 667.48 ms | 597.81 ms |
| CPU + Accelerate | 478.16 ms | 397.90 ms |

候选 Metal 相对原路径约 -69.9%。候选与官方 PCM 的 max_abs 约 3.99e-6，
SNR 118.03 dB，长度相同。CPU + Accelerate 仍慢于候选 Metal，不支持把 stage1
默认搬到 CPU。该候选改变浮点累加顺序，不宣称 WAV 逐字节保持。

## 3. 端到端交错复测

三配置各 4 个独立进程，每进程 1 预热 + 5 热轮；顺序为
`A B C B A C A B C B A C`，去掉官方 C 后为 ABBAABBA。A/B 使用同一个
候选 release 二进制，只有实验开关不同：A 为原路径，B 为零 padding 后裁剪；
C 是官方 MPS，解码仍在官方默认 CPU。模型、短文本和采样参数匹配，seed=42，
但跨框架随机数不相同。

| 配置 | 进程热均值 (s) | mean-of-means | RTF | 进程均值范围 |
| --- | --- | ---: | ---: | --- |
| A 原路径 | 4.0204 / 4.0385 / 4.1426 / 4.1601 | 4.0904 s | 0.7575 | 4.0204–4.1601 s |
| B 反卷积候选 | 3.6831 / 3.6604 / 3.7846 / 3.7853 | 3.7283 s | 0.6904 | 3.6604–3.7853 s |
| C 官方 | 3.3010 / 3.2770 / 3.3497 / 3.3784 | 3.3265 s | 0.6172 | 3.2770–3.3784 s |

观察到候选整链 -8.85%，A/B 进程均值范围不重叠；RTF 相对官方仍高 11.87%。
不能把 stage1 的 -69.9% 当作端到端收益。

运行中观测到其他项目的 cargo/rustc 活动（包括首对 A/B，后段再次出现），
本轮没有主动编译或并行推理，但不能宣称设备独占。这组整链结果标为有干扰的筛选证据。
即使暂不考虑干扰，8.85% 也未达到仓库默认 10% 的保持型优化门槛；而且此候选并非
逐位保持，因此还需另立数值变化验收。用户随后认可这档收益，最终限定 Metal F32
保留算子改写；CPU 切换与 Accelerate 默认均未引入。

### 编译安静时段复核与保留决定

在上述运行结束后，再做 ABBA 四进程、各 1 预热 + 3 热轮。每 0.5 秒采样进程名，
未观测到 cargo/rustc/clippy/clang/ld；桌面进程仍在，不能等同于独占 GPU 或锁频。

| 组 | 两个进程的热均值 | mean-of-means |
| --- | --- | ---: |
| A 原路径 Metal | 4.2119 / 4.1686 s | 4.1902 s |
| B 候选 Metal | 3.8218 / 3.8043 s | 3.8130 s |

改善 **9.00%**，两组进程均值范围不重叠，与前一组约 8.85% 的方向和量级一致。
用户明确表示“能改善 8% 感觉也不错了”，因此这次按用户偏好保留这一收益，
不将默认 10% 门槛误当作无法例外的限制。它仍是本分支、本机的已验证结果，
不等于跨设备、全语料听感或发行验收。

## 4. 融合后 stage0 的归因边界

官方额外带同步诊断的一次热轮：32 个 forward，均值 97.67 ms，其中 LLM 95.85 ms；
解码 113.31 ms，后处理 6.74 ms。该轮整链 3.286 s，音频 5.40 s。

Rust 粗粒度同步诊断（每侧一轮语料生成，另有不同短文本的参考预热）：

| 阶段 | A 原路径 | B 反卷积候选 |
| --- | ---: | ---: |
| 每步 forward，32 步 | 104.99 ms | 105.46 ms |
| 每步 logits cast | 1.32 ms | 1.34 ms |
| 每步预测/打分 | 3.78 ms | 3.76 ms |
| 每步 update（含 argsort） | 0.66 ms | 0.69 ms |
| 每步 argsort，包含于 update | 0.187 ms | 0.189 ms |
| 每步 writeback | 0.36 ms | 0.39 ms |
| stage1 解码 | 603.51 ms | 186.66 ms |

两侧导出的最终 codes（参考预热 + 语料，共 11400 字节）逐字节一致：候选只改变
解码算术，没有改变 stage0 采样轨迹。两侧 stage0 基本不变，解码单独减少约 417 ms。
诊断额外包含约 102 ms 金丝雀 GEMM 和同步开销，不与无插桩整链直接相加对账。

剩余差距的排查方向是 stage0 forward 与预测/打分/cast 等小算子链；目前只测到了
阶段差异，尚未定位到具体一个 stage0 kernel，不能将约 8 ms/步的 forward 差距
全部归因于某个算子。argsort 每步不到 0.2 ms，仍然不值得优先重写。

Rust 临时逐子模块同步探针确认实际层输入 `[2,157,1024]`：CFG 两路沿 batch 拼接，
不是 cond=157 + target=130 得到的 S=287。此前 S=287 微基准是真实的合成形状测量，
但不能标作短句真实形状，按它推导的逐层成本外推撤回。

逐子模块插入大量同步后，正常约 100 ms 的 forward 被显著拉长。该细粒度探针只用于
确认形状和定位候选，不能把同步后的子模块时间当成生产执行占比；也不能据此直接
宣称某个新融合可节省多少。探针补丁保留在忽略目录，正式源码已恢复。

旧记录中“所有有意义优化都无法保持逐位输出”“包络相关证明无音质退化”“剩余
主要是 logits 和切分”等结论超出证据，均已更正。HF 真实权重 parity 与人工听感
仍需独立验收，不能由随机小张量测试替代。

## 复现与产物

```sh
cargo build --release --locked -p talechime-omnivoice --features metal --example stage1_probe --example conv_transpose_probe
cargo build --release --locked -p talechime-omnivoice --features metal,candle-core/accelerate --example stage1_probe --example conv_transpose_probe
```

两种构建的二进制分别保存后再运行；不能覆盖后混用，不能测量期间编译。
`stage1_probe MODEL CODES_JSON cpu|metal OUTPUT_DIR HOT_ROUNDS`。
`conv_transpose_probe MODEL cpu|metal [FRAMES]`，默认 130 帧；每层比较完整输出，
相对 RMS 超过 1e-4 时失败。两个探针都只读取本地模型，普通测试不下载资源。

整链临时补丁从当前分支基线应用，构建独立二进制后通过
`OMNIVOICE_COL2IM=0/1` 切换。此开关只存在于基线实验补丁；最终库按设备/dtype/布局/参数自动路由，没有新增环境开关。
复现补丁时应使用 `6ebc58b` 基线，不能重复应用到已包含最终实现的源码。

原始目录（Git 忽略）：`target/perf/omni-followup/`。

- `environment.json`、`artifact-manifest.json`：代码、输入、二进制、脚本摘要。
- `accelerate-features.txt`、`accelerate-linkage.txt`：依赖与链接证据。
- `stage1-summary.jsonl`、`stage1-{cycle}-{path}.jsonl`：四路径原始轮次。
- `stage1-parity.json`、`col2im-parity-screen.json`、各目录 `raw.wav` / `raw.f32`：数值对照。
- `conv-{metal,cpu,accelerate}.jsonl`：真实权重五层对照。
- `stage0-profile.patch`、`profile-rust.err`、`profile-official.json`：诊断归因。
- `e2e-summary.jsonl`、`e2e-statistics.json`、`e2e-*.jsonl`：交错整链数据。
- `quiet-summary.jsonl`、`quiet-statistics.json`：无编译观测的 ABBA 复核。
- `profile-coarse-{A,B}.err`、`profile-{A,B}-codes.bin`：粗粒度计时与采样轨迹对照。
- 官方脚本 `official_stage1.py`、`official_profile.py`；它们是本地实验脚本，不是官方源码。

## 最终实现与验证

`codec_layers::forward_dense_conv_transpose1d` 只为符合条件的 Metal F32 dense 层
选择零 padding + 裁剪，其余回到原 Candle 层。裁剪长度用 checked subtraction，
越界返回错误；右侧裁剪量为 padding - output_padding，bias 在裁剪后相同位置相加。

普通测试不加载模型：Metal 测试用独立 f64 主机反卷积参考，覆盖 batch=1/2、
输入帧=1/7/130、stride=2/3/4/5/8、有/无 bias，共 60 组合，新路径全部对照独立参考；
其中 40 组再与原 Candle 路径比较。包含奇数 stride 的 output_padding=1。
一帧时原 Candle debug 长度公式在加 kernel 宽度前先减 padding，出现 usize 中间值
下溢（`candle-core/src/conv.rs:46`）；这 20 组使用独立 f64 参考验证新路径，
不调用会 panic 的原 debug 路径。本轮没有修改 Candle 依赖源码；release 真实权重
探针对照仍覆盖一帧。CPU 测试确认其继续走原路径。
真实权重探针另在 1 帧、7 帧的完整五级上采样链做 CPU/Metal 边界复核：20 项全部通过，
相对 RMS 误差不超过约 1.1e-7。130 帧五层及完整 codes 与官方的对照见前文。

最终质量门禁全部通过：

| 门禁 | 结果 |
| --- | --- |
| `cargo fmt --all --check` | 通过 |
| `cargo test --locked -p talechime-omnivoice --features metal` | 28 passed |
| `cargo test --locked -p talechime-omnivoice --no-default-features` | 22 passed |
| 本 crate，Metal + `candle-core/accelerate`，Clippy all-targets / `-D warnings` | 通过 |
| talechime 多后端 Metal feature 组合，Clippy all-targets / `-D warnings` | 通过 |
| 本 crate rustdoc，Metal / no-deps / document-private-items / examples | 通过 |
| 最终 release performance_probe 真模型复核，1 预热 + 5 热轮 | 5/5 WAV SHA256 与实验 B 相同 |

最终 WAV SHA256：`2eadc756eb4482fe320efe678ef8a67312ce79dcfa6d6f4dbf41ed4642be324e`。
最终 release 二进制 SHA256：`abeafab3ec894b842950ff8a92e2313446c8d12d8ddd9f89fff7b9577adab5f2`。
这次输出复核的热测均值 3.638 s，仅作为最终实现复现候选输出的证据；没有对应
同时段原路径配对，不能拿它重算改善幅度。性能结论仍用前述交错与 ABBA 数据。

日志保存在 `final-*.log`、`build-final-metal.log`、`final-verify-summary.json`。
没有改变设备策略或默认启用 Accelerate，没有 push、合并、部署或发包。
