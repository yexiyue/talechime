# VoxCPM macOS AudioVAE Accelerate CPU 协同验证（2026-10-19）

结论：**CPU 串行替代未发现稳定优势；混合流水线尚未验证，生产 Metal 路径不变。**

Accelerate CPU 使 AudioVAE 的 CPU F32 回放较普通 CPU F32 快 37.59%，但包含
Metal patch 就绪、F32 转换、CPU 解码和 PCM 就绪的完整边界上，和同一
Accelerate-enabled 构建中的 Metal F16 没有稳定优势。两组逐进程均值区间重叠，
且远未达到端到端至少 10% 的门槛。没有 profiler 证据，本记录只称
“Accelerate CPU 路径”，不声称其中的 GEMM 使用了 AMX。

## 边界与实现

- 实验代码是 `voxcpm` 的 macOS、测试专用私有模块：
  `codec-experiment` 只编入回放入口，`accelerate-experiment` 再开启
  `candle-core/accelerate`。默认 feature、生产 API、产品配置和调度均未改变。
- `Model::generate_inner` 只有一个静态分派的私有 patch 回调；生产入口传入可内联
  的 no-op。测试在 CFM sample 后、AudioVAE 解码前捕获真实 Metal F16 patch。
- 捕获与回放分离。捕获保存 f32le patch、生产 PCM、分段与 EOS 到
  `target/tmp/voxcpm-metal/codec-capture/`（受 `/target/` 忽略）；普通 CPU 与
  Accelerate CPU 的独立 release target directory 重放同一份工件。
- 每个回放进程按段 `Codec::reset()`，每个路径有 1 轮预热和 2 轮热回放。计时分别
  记录 handoff、decode、PCM 就绪及总时间；加载和 `ru_maxrss` 高水位单列。

## 环境与固定输入

- `cd1089b30b73682bf0c5f72c995cc6d4c4cb9411`，工作树有本轮之前的 Candle
  skinny-M 试验改动；Candle fork `2266099d92265112ecc0d56ecee1761e3cc2ac00`。
- Apple M4 Pro（Mac16,8），24 GB 统一内存；macOS 26.6.2 (25G83)；
  Rust 1.99.0。
- 本机已有 VoxCPM 2B Q8 模型及 `voice_dongqing` 参考音频；语料为
  `tools/tts/acceptance-corpus.txt`，用生产 `preprocess_text(_, 180)` 切成 11 段。
  参考加载复用生产的 16 kHz sinc 重采样参数。
- 捕获不计性能：11/11 段 EOS，共 364 个 patch、2,795,520 个 48 kHz PCM 样本；
  模型加载 2509 ms、参考准备 1639 ms、捕获 58663 ms。

## 初筛：默认线程，四进程 ABBA

顺序为 `plain → accelerate → accelerate → plain`，每进程之间冷却 20 秒。所有
进程串行运行，未并行编译或推理。表中的数值是每进程两轮热回放均值，单位 ms；
“Metal”是同一进程中相同落盘 patch 的 Metal F16 decode + PCM 就绪。

| 进程 | CPU 路径 | CPU F32 总时间 | Metal F16 总时间 | CPU 相对 Metal |
| --- | --- | ---: | ---: | ---: |
| plain-01 | 普通 GEMM | 16410.053 | 10865.392 | +51.03% |
| accelerate-01 | Accelerate | 9795.258 | 9531.620 | +2.77% |
| accelerate-02 | Accelerate | 10206.857 | 10506.254 | -2.85% |
| plain-02 | 普通 GEMM | 15636.891 | 10595.924 | +47.57% |

- 普通 CPU F32：均值 16023.472 ms，范围 [15636.891, 16410.053] ms。
- Accelerate CPU F32：均值 10001.057 ms，范围 [9795.258, 10206.857] ms，较普通
  CPU 少 37.585% 时间。
- 在两个 Accelerate 进程中，CPU F32 均值 10001.057 ms，Metal F16 均值
  10018.937 ms；两者范围分别为 [9795.258, 10206.857] 和
  [9531.620, 10506.254] ms，**重叠**。逐进程方向还发生反转，不能作为候选收益。

这不是完整 VoxCPM 端到端时间，更不能证明长时间播放无断流；它只是串行解码替代的成本测试。CPU 路径连同交接没有稳定快过 Metal 解码，因此没有继续串行替代的正式 ABBA，也未扫描 Rayon、创建 CPU worker、容量 1 队列或取消状态机。

**2026-10-19 审查更正**：原计划还允许“CPU 每帧耗时低于剩余 GPU 生成工作”时验证流水线。此处 Accelerate 约 10.0 s / 364 patch = 27.5 ms/patch，整体生成约 58 s，尚不能排除重叠空间。固定落盘 patch 回放未测量生产者/消费者同时运行的竞争，不能证明流水线无收益。混合串行原型、流水线、队列阻塞取消和正式端到端 ABBA 均为未验证范围；此前将其表述为“排除混合架构”过强。

## 数值与状态门禁

- Metal F16 回放与捕获期间的生产 PCM：两个热轮都是逐位一致。
- Metal F32 / CPU F32：两个热轮都保持 2,795,520 个样本、全有限、长度一致。
  Accelerate CPU 的误差为 max abs `0.0007121302`、mean abs
  `0.0000145947`、RMSE `0.0000236478`、SNR `56.8623 dB`；只报告，不认定听感可接受。
- 捕获索引保留所有 11 个 EOS、分段边界与 patch 数；每个 replay 都重置 codec，且两轮
  输出的误差统计相同。没有创建异步 worker 或队列，因而队列阻塞取消的验收不适用；
  既有库测试继续覆盖取消、EOS、`max_frames`、空文本和非法采样。

## 复现

测试二进制的工作目录不是仓库根，因此所有输入和工件路径须为绝对路径。先构建两个
独立 release target，再捕获一次，最后按上表的 ABBA 顺序运行 replay；不要在 replay
期间编译。

```bash
root=$(pwd -P)
model=/Users/yexiyue/.talechime/resources/voxcpm/models/2b-q8_0/169f64d8b98bbaab1761e4ca3a83e6af653456cc
ref=$model/voices/voxcpm/2b-q8_0/169f64d8b98bbaab1761e4ca3a83e6af653456cc/voice_dongqing/reference.wav

CARGO_TARGET_DIR=$root/target/voxcpm-codec-plain \
  cargo test --release --locked -p voxcpm --no-default-features --features 'metal,codec-experiment' --no-run
CARGO_TARGET_DIR=$root/target/voxcpm-codec-accelerate \
  cargo test --release --locked -p voxcpm --no-default-features --features 'metal,accelerate-experiment' --no-run

CARGO_TARGET_DIR=$root/target/voxcpm-codec-plain \
VOXCPM_EXPERIMENT_MODE=capture VOXCPM_EXPERIMENT_MODEL=$model \
VOXCPM_EXPERIMENT_REFERENCE=$ref VOXCPM_EXPERIMENT_TRANSCRIPT=$root/target/tmp/voxcpm-metal/ref-transcript.txt \
VOXCPM_EXPERIMENT_CORPUS=$root/tools/tts/acceptance-corpus.txt \
VOXCPM_EXPERIMENT_CAPTURE=$root/target/tmp/voxcpm-metal/codec-capture \
  cargo test --release --locked -p voxcpm --no-default-features --features 'metal,codec-experiment' \
  macos_accelerate_codec_experiment -- --ignored --nocapture
```

对 replay 使用相同的 `VOXCPM_EXPERIMENT_MODEL` 与
`VOXCPM_EXPERIMENT_CAPTURE`，并设置唯一的 `VOXCPM_EXPERIMENT_OUT`；普通路径使用
`target/voxcpm-codec-plain` 与 `metal,codec-experiment`，Accelerate 路径使用
`target/voxcpm-codec-accelerate` 与 `metal,accelerate-experiment`。每次结果的
`summary.json` 保留所有热轮、质量指标、feature、线程环境、加载时间及峰值 RSS。

## 未验证范围

- 未用 Instruments、Metal System Trace 或 AMX 计数器，因此不判断 CPU F32 GEMM 的
  微架构分派。
- 未做听音、ASR 回读或长期播放验收；F32 路径的数值差异不自动等于音质可接受。
- 未验证 CUDA、Intel Mac、其他 Apple Silicon、其他模型、其他 batch/codec 形状；本机
  的短进程结果也不外推为持续负载表现。
