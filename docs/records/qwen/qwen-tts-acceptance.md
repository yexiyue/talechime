# Qwen3-TTS Candle 集成验收（2026-10-06）

## 范围与依赖

集成前的 MOSS、连续播放和默认关闭对齐改动已提交为 `f2de440`。本变更增加可选 `qwen` 后端，不改变 MOSS 默认值。采用 [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) 的原生 Rust/Candle 实现，固定 revision `711ceee07cad92673f86de8997bdf54c30caa49f`，Candle 0.9.2。无需 Python 推理环境。

模型为官方 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`，revision `85e237c12c027371202489a0ec509ded67b5e4b5`，8 个文件合计 2,498,383,173 bytes。manifest 固定大小和 SHA-256，资源保存在 `~/.novel-tts/qwen/`，与 Qwen 强制对齐资源分开。9 个预置音色，默认福叔；首版语言策略为中文和英文，未接入参考音频克隆、VoiceDesign 或 Candle CUDA。

CPU 与 macOS Metal 使用后端自己的设备目录。Metal 同时启用 Candle core、nn、transformers 的算子支持：单独启用 core 可以加载模型，但实际 RMSNorm 推理会失败。本次已复现并修复该构建问题。依赖仅针对 macOS，其他 target 不报告 Metal。现有 ORT/Kokoro 版本保持不变。

协议升级为 v4；阅读器与 worker 需同步升级。阅读器不依赖 Candle、具体后端、core 播放库或 ORT。MOSS/Kokoro/Qwen 共用模型无关的有界 PCM、播放和检查点机制。

## 实测环境与性能

Apple M4 Pro、24 GiB 内存，release 构建。固定文本、福叔和随机种子 42；CPU F32 与 Metal BF16 的输出帧数可能不同，不宣称 PCM 一致。

| 设备 | 模型加载 | 首 PCM | 完整生成 | 音频时长 | 最大 RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1.855 s | 2.956 s | 11.819 s | 3.68 s | 约 5.05 GB |
| Metal | 0.546 s | 0.896 s | 3.787 s | 3.52 s | 约 5.08 GB |

输入为“你好，欢迎收听。”，结果和试听在 `target/qwen-acceptance/{cpu,metal}-seed42.{json,wav}`。这组短文本仅为诊断；自动设备选择另行执行完整链路 3 次预热、5 次测量：

| 设备 | 平均完整生成 | 平均首 PCM |
| --- | ---: | ---: |
| CPU | 19.9253 s | 2.5965 s |
| Metal | 13.2949 s | 0.8946 s |

完整生成降低约 33%，首 PCM 降低约 65.5%，通过至少 15% 收益且首 PCM 不回退超过 10% 的门槛，auto 选择 Metal。首次校准到模型就绪约 289 s；缓存命中后的实际 worker 准备约 8.07 s（包含校验，且与其他验证并发）。缓存按硬件、模型和运行库隔离，并为不同模型使用不同文件名。

## 自动验证

- 完整 workspace 279 项测试、Clippy warnings deny、rustfmt、rustdoc warnings deny。
- 8 组构建：Qwen CPU、Qwen Metal、Qwen + alignment、Qwen + alignment + Metal、MOSS + Qwen、Kokoro + Qwen、全部后端与设备、无 TTS 阅读器。
- Qwen 单后端协议测试：设备与后端校验、9 音色目录，无需下载模型。
- 真实模型测试：首 PCM 为 24 kHz、取消接收后释放当前生成，下一请求正常 EOS；真实测试耗时约 13 s。
- EOS 单元测试：达到帧数上限、空音频不会发布完成事件；上游 `next_chunk()==None` 不能单独作为正常完成证明。
- 原文 UTF-8 映射、软换行合并、硬边界、装饰过滤、长文本字符边界切分。
- 模型文件全部 SHA-256 校验；关闭 alignment 时未创建对齐任务或加载对齐模型。

## 限制与未验收项

上游为实验实现，小块 codec 解码的听感仍需人工试听。正常 EOS 不能证明每个字都已朗读；中文、英文、混合文本、长文本与块边界的人工内容和音质验收继续保留。NVIDIA、Linux、Windows 实机未验收；本次不提供 Qwen CUDA。

所选上游目前通过固定 Git revision 引入，尚无对应 crates.io 包。独立发布 novel-tts-backends 前需要解决依赖分发；本次不宣称 crates.io 发布可用。现有 MOSS/Kokoro feature 和默认行为保留。

长段落 Metal 探针：“山风吹得血袍飘荡……”正常 EOS，首 PCM 0.893 s、生成 18.882 s、音频 17.68 s；模型加载 0.606 s。试听与指标在 `target/qwen-acceptance/story-metal.{wav,json}`。这仅验证生成完成，内容与听感仍需人工核对。

VHS 已检查后端 Qwen、Serena 音色切换、Auto → Metal、模型就绪，以及实际 Playing 状态和绿色片段高亮。暂停与退出后检查点保留在完成位置。录制在 `target/qwen-acceptance/vhs/qwen.gif`，可复用脚本为 `docs/tapes/qwen-tts.tape`。

## 本地库与 CUDA 接入（2026-10-07）

推理源码迁入 `crates/qwen3-tts`，来源固定 revision 不变。推理用 tokenizers 禁用 `esaxx_fast`，Windows Qwen + ORT CUDA 构建不再要求 `/MD` 环境变量。增加 `qwen-cuda`，原 ONNX `cuda` feature 改为 `ort-cuda`；GPU 设备目录与加载共享本地设备创建入口。移除上游 Hub、CLI、手写 fused PTX 和 Flash Attention，残差归一化使用 Candle 自带设备算子。

本机 CUDA 12.9 Update 1 已安装到 `D:\dev-tools\cuda\v12.9`，官方组件 SHA-256 全部校验通过。nvcc 12.9.86 与 MSVC 14.44 编译的 `sm_120` 内核在 RTX 5070 实际运行成功，官方 deviceQuery / vectorAdd 通过。`cargo build --locked -j 2 -p novel-tts --no-default-features --features qwen-cuda` 通过，worker 协议探针确认 Qwen `compiled` 与 `available` 均包含 CUDA。日志在 `target/qwen-local-cuda-build.log` 与 `target/cuda-install-check/worker-probe.log`。

初次 CUDA-only 构建产生 LNK4098；默认 MOSS + Qwen CUDA 组合进一步暴露 Candle MOE `/MT` 与 ORT `/MD` 的 LNK2038。将 MSVC x64 编译器目录与 CUDA bin 写入系统 PATH，配置系统 `NVCC_PREPEND_FLAGS=-Xcompiler=/MD` 并重建 candle-kernels 后，无需 PowerShell 激活脚本的 `cargo build --locked -j 2 -p novel-tts -F qwen-cuda` 通过，未产生链接警告。dumpbin 确认 MOE 对象的 RuntimeLibrary 为 MD_DynamicRelease；日志在 `target/qwen-local-cuda-default-build.log`。没有使用 `/NODEFAULTLIB` 屏蔽。

2026-10-07 后续已完成 Windows RTX 5070 的 0.6B/1.7B CustomVoice、Base/VoiceDesign 实际生成、正常 EOS、取消后重试以及 1.7B CustomVoice 30 分钟实际播放。RTF 分别约 0.758、0.716、0.796；30 分钟 RTF 约 0.758，显存未见持续增长。当前协议为 v5，新增模型策略与完整证据见 `tts-model-tiers-acceptance.md`。人工听感、Linux CUDA 与本次新增模型的 macOS Metal 实机仍待验收。OpenSpec 中将这些验收项分别记录，历史 Metal 测量保留为历史数据。

本地 portable feature 集的 workspace lib/tests/examples 回归通过（488 次测试通过，1 次忽略，其中包括既有协议测试对子进程的测试调用）；Clippy `-D warnings`、rustfmt 和 rustdoc `-D warnings` 通过。检查 reader 基础依赖树没有 Candle/ORT/rodio/core，Qwen-only 没有 ORT，esaxx-rs 未启用 cpp。通用 CI/lefthook 改用显式 feature 集；独立 qwen-cuda.yml 使用 CUDA 12.8.1 devel 环境和 CC 120 编译检查，该工作流尚未触发或验证。

## 官方实现对照与本机初测（2026-10-18，比较结论已撤回）

本轮后续修正了参考 encoder RoPE、codec 注意力窗口、流式历史、隐藏采样覆盖及低精度 codebook 累加。用户最终反馈本轮样本“听感差不多了”，此前确认流式不再突变音色；仅限该样本的人工接受，不作为全模型质量或官方数值等价结论。官方本机 Torch 2.7.1 MPS 的空历史重复惩罚异常与受保护的新对照分别保留，详见下述性能审查的最终章节。

后续审查确认旧矩阵的主模型 top-p、声学采样策略不一致，且官方侧未做预热与显式 MPS 同步。以下保留原始观测值用于追溯，不作为公平速度比较；修订与验收见 [性能审查](qwen-performance-review-2026-10-18.md)。

- 官方源码固定为 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`，克隆在 git 忽略的 `runs/reference/Qwen3-TTS/`。官方路径是 PyTorch/Transformers：主 Talker 生成语义 token，CodePredictor 每帧生成 15 个声学 token，最后 12Hz tokenizer decoder 输出 24 kHz PCM。
- 对照条件：Qwen 0.6B Base ICL，官方 revision `022e286…`，同一参考音频与转写，中文文本“山风吹过松林，星光照亮归途。两个人沿着小路慢慢往前走。”，seed 42；Rust 主模型 top-p=0.9、声学贪心，官方主模型 top-p=1.0、声学随机采样。3 个独立进程均是模型加载后的首个请求，官方脚本当时未显式同步 MPS。Apple M4 Pro，macOS 26.x。
- Rust/Candle Metal BF16 三次：完整生成 9.100 / 9.158 / 9.126 s，音频均 6.88 s，RTF 1.323 / 1.331 / 1.326；首 PCM 约 1.15–1.17 s。官方 PyTorch MPS F32 三次：24.280 / 24.150 / 23.806 s，音频均 6.88 s，RTF 3.529 / 3.510 / 3.460。两侧生成音频样本数一致。该结果仅代表本次 Qwen MPS F32 与 Rust Metal BF16 的组合，不等同于“Rust 比 PyTorch 快”的普遍结论。
- Rust CPU 单次（F32/Accelerate）总生成 10.388 s、音频 6.48 s、RTF 1.603；关闭 Accelerate 的 gemm 对照为 15.903 s、RTF 2.454。CPU 与 Metal 浮点差异使输出长度不同。官方 CPU 未完成同条件矩阵，不能列入对比。
- 官方仓库当前 tip 的本地权重缺少 `speech_tokenizer/preprocessor_config.json`，且 wrapper 以 `raw_audio=` 调用 Transformers 4.57.3 的 Wav2Vec2 positional 参数。为运行对照，本地未跟踪副本补了等价的 Wav2Vec2 feature extractor 配置，并把调用改为 positional + padding/attention mask；`tools/tts/qwen3_tts_bench.py` 会记录该限制。此补丁只用于基准，不改变 Talechime 实现。
- 汇总见 `runs/qwen-bench/summary.json`；目录还保存 Python/Rust 逐轮 JSON、WAV、`/usr/bin/time` 输出和 Rust 逐块诊断日志。`docs/learning/tts-primer.md` 补充了三段结构的性能直觉。
- 后续 profiling trace（同输入、热模型、Metal BF16）显示流式路径的主要开销为 CodePredictor 约 1.92 s、主 Talker 步进约 0.68 s、采样约 2.82 s、9 次分块 codec 解码约 1.30 s、ICL prefill 约 0.046 s。非流式整段解码路径为 4.88–4.97 s；流式路径为 9.24–9.42 s；当时对“每 chunk 重放参考 codec”的解释错误，源码只在首块使用参考 codes，后续块独立解码。`crates/qwen3-tts/src/synthesis/streaming.rs` 已把每帧 acoustic codes 留在 GPU、每 10 帧 bulk 传输，避免每帧 GPU→CPU 同步；对照三组流式中位数从 9.787 s 降至 9.254 s，非流式中位数从 4.994 s 降至 4.839 s。采样只保留原始词表位置过滤语义；一次“排序后直接 top-p”试验改变 token 位置导致连续生成，已拒绝并入。
- 追加官方 MPS BF16 矩阵：3 个独立进程为 21.744 / 17.613 / 17.665 s，音频均 6.08 s，RTF 3.576 / 2.897 / 2.905（第一轮为冷后端离群值，未与热路径 F32 混合取均值）。该路径生成有限音频，但输出样本数不同于 F32/Rust 的 6.88 s，不能作为配对速度结论；`tools/tts/qwen3_tts_bench.py` 已支持显式 `--dtype` 并记录 dtype 状态。
