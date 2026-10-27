<p align="center">
  <img src="assets/brand/readme-cover.png" alt="Talechime · 叙铃 — Let stories ring." width="100%" />
</p>

# Talechime · 叙铃

**让故事响起来。**

Talechime 是用 Rust 编写的本地语音合成与长文播放引擎。它提供命令行程序、可嵌入的 Rust 库，以及供阅读器调用的 JSON Lines worker。宿主负责获取正文、识别角色和绑定音色；Talechime 负责合成、播放、取消和按实际播放进度恢复。

[快速开始](#快速开始) · [TTS 后端](#tts-后端) · [与官方实现的性能对比](#与官方实现的性能对比) · [文档索引](docs/README.md) · [架构](docs/architecture.md) · [Rust 库](docs/library.md)

## 能做什么

- 朗读本地 UTF-8 文件，暂停、继续、停止、调节音量和速度，并从实际播放位置恢复。
- 流式接收 PCM，以有界队列和自适应预缓冲连续播放；也可等整章合成成功后再播放。
- 在同一模型内执行增量多音色朗读计划。原文范围使用 UTF-8 字节偏移和正文摘要，不额外运行对齐模型。
- 导入参考音频、管理音色，并通过 Rust API 或 JSON Lines v7 接入应用。
- 按需启用 Qwen3-ASR 与 SenseVoiceSmall 回读报告或逐段校验；默认关闭。

正文和参考音频由本地模型处理。模型权重按需准备，安装包不包含权重；CosyVoice3 需要先转换并放置本地模型文件。

## 0.5.0 更新

本版加入 CosyVoice3 实验后端与风格控制，并提供 IndexTTS 2.5 的源码构建与情绪参数；
同时合入多后端 CPU、Metal、CUDA 优化，修复 Qwen 分块解码的局部噪音及 CUDA 图布局差异，
校正 VoxCPM2 的单段时长预算。协议仍为 JSON Lines v7，用户数据仍位于 `~/.talechime/`。
完整变化见[更新日志](CHANGELOG.md)；安装包包含的后端与平台限制见下文。

## 快速开始

### 安装

发行包提供 Apple Silicon macOS、x86_64 Linux GNU 和 x86_64 Windows MSVC 的 `talechime` 程序：[查看 Releases](https://github.com/yexiyue/talechime/releases)。macOS / Linux 可使用安装脚本或 Homebrew：

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/yexiyue/talechime/releases/latest/download/talechime-installer.sh | sh
brew install yexiyue/tap/talechime
```

Windows PowerShell：

```powershell
irm https://github.com/yexiyue/talechime/releases/latest/download/talechime-installer.ps1 | iex
```

macOS 标准包要求 macOS 15+；Linux 包基于 Ubuntu 24.04，要求 glibc 2.39+、相应 libstdc++ 和 ALSA 运行库；Windows 需要 Visual C++ Redistributable。标准包在 macOS 编入各模型的 Metal 路径，在 Linux / Windows 使用 CPU；CUDA 需从源码编译相应 feature。实际可选设备以程序报告为准。

### 从源码运行

需要 Rust 1.89+ 和平台链接工具。Linux 还需 ALSA、OpenSSL 开发库及 pkg-config；CUDA 构建需要匹配的 Toolkit。

```sh
git clone https://github.com/yexiyue/talechime.git
cd talechime
cargo run --release -- chapter.txt
```

默认构建使用 MOSS Nano Candle。按 `Space` 暂停或继续，`s` 停止，`q` 退出。`--restart` 从头朗读；`--backend`、`--model`、`--tts-device` 指定模型和设备：

```sh
cargo run --release -- --restart chapter.txt
cargo run --release -- --backend moss --model nano-candle --tts-device cpu chapter.txt
```

首次朗读会下载并校验所选模型，可能需要数 GB 空间和较长时间。`--help`、协议握手和普通音色目录查询不下载权重。

## TTS 后端

“可选”表示适配器已接入 Registry；具体设备需要相应编译 feature、模型资源和实机支持。音色和功能以模型能力目录为准。

| 后端 / `--backend` | 模型或实现 | 音色与输出 | 构建和资源边界 |
| --- | --- | --- | --- |
| MOSS / `moss` | Nano Candle `nano-candle`（默认）；Realtime `realtime-1.7b` | Nano 内置或固定导入音色；两档均不提供 style/instruction/language | Nano 默认 CPU，Metal / CUDA 需显式 feature；Realtime 为需 `moss-candle-*` 的 GPU 模型 |
| Kokoro / `kokoro` | `1.1-zh`；可选 `1.1-zh-g2pw`，源码实验移植 | 官方中文预置音色，24 kHz 分段 PCM；支持 speed，无克隆/风格/原生流式 | Candle F32，CPU/Metal 已做官方对照，CUDA 待实机验证；须先[转换官方资源](docs/kokoro.md)，未随发行包发布 |
| Qwen3-TTS / `qwen` | 0.6B、1.7B CustomVoice；0.6B、1.7B Base | CustomVoice 预置音色；1.7B CustomVoice 支持风格；Base 可导入参考音频克隆 | Candle CPU；Metal / CUDA 需 `qwen-metal` / `qwen-cuda` |
| VoxCPM2 / `voxcpm` | 2B Q8 GGUF；另有实验 BF16 路径 | 固定参考音色、音色设计与原生流式 PCM | Candle CPU；Metal / CUDA 需 `voxcpm-metal` / `voxcpm-cuda` |
| OmniVoice / `omnivoice` | 0.6B | 参考音色、音色设计、风格和生成期语速；按语义段输出 | Candle CPU；Metal / CUDA 需 `omnivoice-metal` / `omnivoice-cuda`；当前不是模型原生实时流式 |
| CosyVoice3 / `cosyvoice3` | 0.5B，实验性 | 导入参考音色克隆，支持风格指令，24 kHz 原生流式输出 | Candle + ORT CPU 前端；可启用 Metal / CUDA；须先[转换官方权重并放入本地目录](docs/cosyvoice3.md)，不自动下载 |
| IndexTTS 2.5 / `indextts` | 2.5，实验性 | 中文或英文参考克隆，支持命名情绪/向量/情绪参考，22.05 kHz 整段输出 | Candle CPU；Metal / CUDA 需显式 feature 和[本地转换文件](docs/indextts.md)；标准包暂未编入 |

例如在 Apple Silicon 上编译 Qwen Metal，在 NVIDIA 设备上编译 Qwen CUDA：

```sh
cargo build --release --locked -p talechime --features qwen-metal
target/release/talechime --backend qwen --model 1.7b-customvoice --tts-device metal chapter.txt

cargo build --release --locked -p talechime --no-default-features --features cli,qwen-cuda --bin talechime
target/release/talechime --backend qwen --model 1.7b-customvoice --tts-device cuda chapter.txt
```

MOSS 两档均使用 Candle；各模型通过自己的 feature 启用 GPU。详细编译组合见[开发说明](docs/development.md)，验收边界见[模型验收记录](docs/records/tts-core/tts-model-tiers-acceptance.md)。

### 音色和生成参数

音色资产存放在独立的音色库（`~/.talechime/voices/`），与任何后端/模型解耦：
同一条音色可被所有支持参考克隆的模型使用，首次引用时自动物化进所选模型。

```sh
talechime models                             # 模型、设备、内建音色和参数
talechime models --backend qwen --json        # 完整能力与参数约束
talechime voices list                        # 库内音色（与 --backend 无关）
talechime voices engines                     # 可用设计引擎与代价
talechime voices design --name "我的朗读音色" --description "低沉温和的中年男声"
talechime voices import reference.wav --name "导入音色" --text "参考音频的准确文字"
talechime voices export my-voice --output my-voice.voice
talechime voices import-bundle my-voice.voice
talechime --voice my-voice chapter.txt       # 库音色直接朗读，隐式物化

talechime --backend voxcpm --param steps=32 --param temperature=0.9 --seed 7 chapter.txt
```

参考音频应附准确转写。设计引擎的自动选择优先复用已就绪的朗读模型（零额外下载），也可 `--engine` 显式指定。各模型只接受其能力目录声明的参数；非法名称、类型或范围会报错。[参数与风格记录](docs/records/tts-core/params-exposure-2026-10-10.md)与[音色库重构记录](docs/records/tts-core/voice-studio-2026-10-26.md)列出当前接口。

## 与官方实现的性能对比

性能记录涉及两台不同机器：**Windows 11 / Ryzen 7 9700X / RTX 5070（约 12 GB 显存）**，
以及 **MacBook Pro / M4 Pro（12 核 CPU、16 核 GPU、24 GB 统一内存）**。
下文分别标注 CUDA、Metal 或 CPU Accelerate 路径，收益不能跨设备叠加。
当前优先[继续优化已有后端的双设备性能 #5](https://github.com/yexiyue/talechime/issues/5)；
所有后端使用各自官方固定参考或预置音色流程；宿主滚动参考与双参考实验已移除，见[固定参考合成](docs/reference-synthesis.md)。
环境采集要求及历史结果边界见[性能方法](docs/knowledge/tts-performance.md#当前优先级与测试设备)。

以下数字来自仓库内的真实模型记录，以官方 Python / PyTorch 实现为基线。**RTF = 合成耗时 ÷ 输出音频时长，越低越快**。跨框架即使使用相同 seed，也可能生成不同长度的音频；表中变化是所述测试条件下的观察值，不代表音质、内容完整性或所有设备的速度。

| 模型与条件 | 官方 RTF | Talechime / Rust RTF | 相对官方 | 证据与限制 |
| --- | ---: | ---: | --- | --- |
| Qwen3-TTS 0.6B Base ICL，M4 Pro：官方 MPS BF16 vs Rust Metal BF16，同一文本/参考/seed/匹配采样，各 1 轮预热 + 5 轮热模型测量，非流式 | 2.956 | 0.512 | RTF 低约 82.7%；合成耗时低约 79.7% | [性能记录](docs/records/qwen/qwen-performance-optimization-2026-10-18.md)：双方随机数、浮点传播与输出长度不同，是同输入条件观察而非同一 token 负载配对；Rust 流式 RTF 0.746，首 PCM 0.760 s，官方未提供流式首 PCM |
| CosyVoice3 0.5B，M4 Pro CPU，`long` 样例，热模型非流式 | 1.17 | 1.089 | RTF 约低 7% | [本机报告](docs/records/cosyvoice3/cosyvoice3-performance-2026-10-17.md)：Rust 为 10 月 18 日默认启用 Accelerate 后的单组测量；官方与 Rust 采样 token 数不同，尚无最终版本的配对多轮官方复测 |
| IndexTTS 2.5，RTX 5070 CUDA，同语料 6 段 × 2 轮 | 0.45577 | 0.42137 | 整体 RTF 低 7.55%；排除首条后低 1.53% | [串行 ABBA 对照](docs/records/indextts/indextts-candle-performance-2026-10-17.md)：测于独立计算库；接入后端的实机性能尚未复测，稳态差距不能外推为稳定优势 |

CosyVoice3 的 Rust Metal 没有对应的官方 MPS 实现可作同设备对照；[报告](docs/records/cosyvoice3/cosyvoice3-performance-2026-10-17.md)另列 Metal 和官方 CPU 的绝对结果，不能把设备差异算成移植提速。Qwen 官方对照使用显式匹配采样和空历史 repetition guard；官方源码补丁只用于可运行的 MPS 基准与 EOS/来源记录，不改变 Talechime 实现，历史失效比较保留在[审查记录](docs/records/qwen/qwen-performance-review-2026-10-18.md)。MOSS 和 VoxCPM2 目前缺少同条件的**官方实现**性能基准，因此不报“比官方快多少”。VoxCPM2 的[原生 llama.cpp-omni 对照](docs/records/voxcpm/voxcpm-candle-acceptance.md)与官方 PyTorch 是不同基线。

OmniVoice 已有 [CPU / Metal 执行路径与对照记录](docs/knowledge/omnivoice-performance.md)。2026-10-23 的 M4 Pro 短文本同期比较为 Rust Metal 2.672 s、官方 MPS 3.235 s；两端生成长度与随机轨迹不同，是特定条件下的产品表现，不能外推所有输入或音质等价。CPU 后续 decoder 优化有独立固定 codes 收益，尚无最终组合版本与官方的新配对排名。

**仓库内优化（Qwen3-TTS 0.6B Base ICL，RTX 5070 CUDA）**：上表是“对官方”口径；CUDA 上另有一项仓库内 A/B——CodePredictor 整帧 CUDA 图（`qwen-cuda` 默认启用，`QWEN_CP_GRAPH=0` 关闭，profiling 走 eager）。ABBA 八进程、锁频、丢弃预热，每次生成 59 帧 / 4.72 s 音频，两组区间均不重叠，codes、原始 PCM 与 WAV 与 eager 逐字节一致：

| 指标 | eager | 启用帧图 | 改善 |
| --- | ---: | ---: | ---: |
| 完整合成耗时 / RTF | 3.0840 s / 0.653 | 1.9113 s / 0.405 | -38.0% |
| 流式合成耗时 / RTF | 3.6529 s / 0.774 | 2.4353 s / 0.516 | -33.3% |
| 首 PCM | 0.7688 s | 0.5663 s | -26.3% |

这是同一二进制 eager 与图回放之间的比较，不代表比官方 CUDA/PyTorch 快多少，也不外推到其他模型或设备；口径、锁频采样与未复验项见[记录](docs/records/qwen/qwen-cp-graph-2026-10-18.md)。

后续在 **CP 图已开启**的条件下，流式解码图将耗时从 **2.4240 s 降至 2.1140 s（减少 12.8%，RTF 0.514→0.448）**，完整合成与首 PCM 没有明确变化。八进程、各 1 预热 + 5 热轮，240 份产物逐字节一致；CUDA 默认开启，`QWEN_DECODE_GRAPH=0` 关闭。该增量实验使用平衡电源计划，不能和前表不同条件的数字直接叠加；[设计、审查与原始口径](docs/records/qwen/qwen-decoder-graph-2026-10-18.md)另列。

2026-10-25 为修复分块局部噪音，Qwen 流式 decoder 改为会话持有逐层状态，CUDA 图范围
调整为无状态声码器。上述解码图数字来自修改前的历史路径，新路径的 CUDA 性能尚未复测；
[噪音修复记录](docs/records/qwen/qwen-streaming-noise-2026-10-25.md)包含同码隔离与 Metal 数值验证。
Windows CUDA 的新声码器图已通过 Base 0.6B codes/PCM 逐位对照及取消复用，
见[合并审查记录](docs/records/platform/windows-main-review-2026-10-25.md)；这不替代新路径的性能或人工试听验收。

**1.7B Base ICL 也已独立验收**：同一 RTX 5070 上，eager→两项图开启，完整 **2.8463→1.8713 s（减少 34.3%，RTF 0.684→0.450）**、流式 **3.3973→2.0698 s（减少 39.1%，RTF 0.817→0.498）**、首 PCM **0.7665→0.6138 s**。12 个独立进程比较 eager／CP／两图三种配置，各 1 预热＋5 热轮，360 份产物逐字节一致；模型按固定清单保存于 `~/.talechime`。这里只验收 1.7B Base，不外推到 CustomVoice/VoiceDesign；[完整报告](docs/records/qwen/qwen17-cuda-acceptance-2026-10-18.md)包含配置、显存与磁盘清理记录。

## 接入应用

对外 Rust 包统一使用 `talechime` 品牌：模型库包括 `talechime-moss-tts`、
`talechime-qwen3-tts`、`talechime-qwen3-asr`、`talechime-voxcpm`、
`talechime-omnivoice`、`talechime-cosyvoice3` 和 `talechime-indextts`。
源码目录保留短名，Rust 导入名统一使用 `talechime_` 前缀；GPU 平台配置及图回放作为内部实现维护，不单独发布辅助包。

crates.io 的统一入口为 `talechime = "0.5.0"`；嵌入时关闭默认 feature，再按需选择模型。
只接入 worker 协议的宿主使用 `talechime-protocol = "0.3.1"`。示例与各包版本见
[Rust 库入口](docs/library.md)及[库发布说明](docs/development.md#rust-库发布)。

Rust 应用可直接使用 `talechime::Engine` 的 `prepare`、`synthesize`、`synthesize_pcm` 和 `listen`。直接合成无需播放器或检查点；朗读计划可在同一模型内切换音色。无模型示例：

```sh
cargo run --locked -p talechime --example synthesize
cargo run --locked -p talechime --example listen_plan
```

阅读器可把 `talechime` 作为独立 worker 启动，并通过 stdin/stdout 交换 UTF-8 JSON Lines；stderr 用于日志。协议当前为 **v7**，只有基于计划的 start 路径。宿主应检查版本、请求和会话 ID、正文摘要及事件序号，避免旧事件推进新会话。[协议说明](crates/talechime-protocol/README.md)和[架构说明](docs/architecture.md)有完整契约。

```sh
talechime --protocol
```

```json
{"protocol_version":7,"request_id":"hello-1","session_id":null,"type":"hello"}
```

CLI 也能以单音色播放整章，或载入与正文完全一致的封闭计划：

```sh
talechime chapter.txt --after-chapter
talechime chapter.txt --plan chapter-plan.json
```

Talechime 不获取章节、不分析角色，也不决定下一章。TRNovel 等宿主保留这些职责；CastGlean 的标注接入仍在规划中。[Rust 库用法](docs/library.md)和[回读校验](docs/readback.md)说明对应接口。

## 接入与命令行技能

根目录 `skills/` 提供两个面向使用方的独立技能，可复制各技能目录到所用 Agent 的技能目录：

- [talechime-integration](skills/talechime-integration/SKILL.md)：Rust API、JSON Lines v7、音色、播放及取消的接入约束。
- [talechime-cli](skills/talechime-cli/SKILL.md)：按当前构建查询能力、管理音色、朗读和排错。

完整命令说明见 [CLI 指南](docs/cli.md)，Rust 接入见 [库入口](docs/library.md)。技能跟随本仓库源码，使用旧发行包时先核对实际帮助与 API。

## 数据与开发

| 数据 | 默认路径 |
| --- | --- |
| 配置 | `~/.talechime/config.json` |
| 播放检查点 | `~/.talechime/checkpoints/` |
| 音色库（设计/导入音色的唯一事实源） | `~/.talechime/voices/` |
| 模型及可再生音色缓存 | `~/.talechime/resources/` |

可用 `--config`、`--checkpoint-dir`、`--model-dir` 指定自定义路径。程序只使用当前目录与格式，不扫描或迁移旧 `novel-tts` 数据；旧文件保留原位，详见[数据目录说明](docs/migration.md)。

开发命令、CPU / 原生 CI 与实机验收边界见[开发说明](docs/development.md)。普通测试不下载模型。项目源码采用 MIT；第三方源码和模型各有授权，见[许可说明](docs/licenses.md)。模型权重、用户音频和小说正文不随仓库分发。
