# IndexTTS 2.5 Mac 首测（2026-10-24，M4 Pro）

issue #5 / Mac goal 第 4 步：IndexTTS 在 Mac 上的第一次真实模型测量。
**边界声明（最重要）**：本测量走 `crates/indextts` 的开发例程
`examples/corpus.rs`（与 2026-10-17 Windows 首轮同一方法学），**不是统一协议
生产探针**——统一 harness 的资源预检按设计拒绝 indextts（无生产 SHA 清单，
`performance.rs` 明示 "Do not invent an oracle"）。计时边界为例程内单段合成
`Instant`（含文本正则化，不含模型加载；load 单列），与生产请求边界不可比。

## 一、资源准备（本轮完成）

- 按 `crates/indextts/SOURCE.md` 的 pinned revision 下载全部来源：
  IndexTeam/IndexTTS-2.5 `c39ce5ba`（主 checkpoint）、facebook/w2v-bert-2.0
  `da985ba`、funasr/campplus `e4b6ede`、nvidia/bigvgan_v2_22khz_80band_256x
  `633ff70`（`hf download --revision`，共 7.7 G，收据目录
  `runs/indextts-checkpoints/`）。
- 用固定 Git 快照 `e591c94` 的 `tools/tts/indextts_export_safetensors.py`
  （cv3-venv torch 2.7.1 + safetensors + omegaconf，CPU）转换：
  **gpt 456 / codec 243 / s2mel 284 / campplus 937 / bigvgan 783 张量，
  feat1 (73,192)、feat2 (73,1280)——与 Windows 记录逐项一致**（跨机转换对齐）。
  转换产物 6.2 G 放入 `~/.talechime/resources/indextts/models/2.5/c39ce5ba-candle-v1/`。
- 音色：手工按 `VoiceStore::import` 的落盘格式创建 `voice_bench`
  （reference.wav + voice.json，转写「希望你以后能够做的比我还好呦。」），
  **与其他后端基线共用同一 bench 参考音色**。无固定 SHA 清单属已知设计，
  来源身份由 pinned 上游 revision 核验（docs/indextts.md 口径）。
- 例程 `examples/support/probe.rs` 的 device 解析缺 metal 臂（Windows 开发
  工具遗留），补 `"metal" => Device::new_metal(0)?`（镜像 cuda；dev-only，
  门禁见 §五）。

## 二、测量条件

语料：与其他 Mac 基线相同短句（`target/perf/omni-input/corpus-short.txt`，
sha `f29b49d8…`）；seed 42；4 轮（无独立预热轮，首轮含首次运行效应，
RTF 逐轮列出）；F32；Rayon 默认（未显式冻结——例程边界，弱于统一协议的
线程冻结，如实登记）。探针二进制：`--features metal --example corpus`，
release/locked。

## 三、结果

| 设备 | 逐轮 RTF | 汇总 RTF | load | 状态 |
| --- | --- | ---: | ---: | --- |
| Metal | 2.33 / 2.31 / 2.26 / 2.26 | **2.289** | 9.08 s | 4/4 成功 |
| CPU（Accelerate） | 4.85 / 4.82 / 4.74 / 4.74 | **4.789** | 0.77 s | 4/4 成功 |

音频时长 5.67–6.11 s/轮（同 seed 下长度有小幅波动，非逐位确定——与官方
采样行为一致，未做输出等价声明）。原始数据：
`target/perf/itb-example-{metal,cpu}/`（index.json + 逐段 WAV）。

**结论：IndexTTS 2.5 在 M4 Pro 上 Metal RTF ≈ 2.3、CPU ≈ 4.8，均高于实时线。**
CPU 超过 goal §三 的 RTF≥3 参考线，登记为不优化。Metal 是唯一候选优化路径；
本轮为首测（无热点剖析），优化立项需要先做统一边界的 profiling（GPT 循环 /
CFM / codec / BigVGAN 分阶段，Windows 侧已有同结构数据可对照）。

## 四、与统一协议的关系（登记为部分覆盖）

- 已完成：资源准备与身份核验、双设备有效语音首测（非空有限 PCM、4/4 成功、
  WAV 已落盘待试听）。
- 未完成：统一协议 3×6 正式基线（被 harness 资源预检的设计性阻塞挡住）；
  长语料；人工试听；与官方 Python 的同机对照。
- 后续若要走统一协议，需要先设计 indextts 的资源身份方案（转换 receipt 的
  机器可读核验），那是一次显式的工具改动，不在本记录内擅自发明。

## 五、改动与门禁

本轮代码改动仅一处：`crates/indextts/examples/support/probe.rs` 增加 metal
设备臂（dev-only 例程）。门禁结果见提交前检查（fmt / crate Clippy
all-targets / crate 测试 / rustdoc）——在本记录落盘后、Qwen CPU 对照测量
结束后串行执行，避免与基准抢 CPU；结果登记于 goal 文档勾选备注。
