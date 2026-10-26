# MOSS 档位移除决策（2026-10-26）

## 决策与范围

产品仅保留 `nano-candle` 和 `realtime-1.7b`。删除 Local 1.7B、Nano ONNX（registry ID `nano`）及 DirectML 实验探针，不提供替代别名、自动迁移或协议协商。VoiceDesign 仅保留为辅助参考音色生成器，不是第三个产品档位。

删除的实现包括 Local scheduler/manifest、ONNX session runtime/manifest/metadata、DirectML session 和 example、专属 CUDA 测试与历史数值探针，以及仅用于退休路径的性能参考脚本。剩余模型的固定 revision、manifest 和许可证保留；不改变其模型数学、权重、PCM/EOS 或 GPU 策略。

## 依据与已接受代价

- **Local 1.7B**：Mac 24GB 统一内存上初始化即 OOM、整平台不可用，见[macOS 基线 §三](moss-macos-performance-baseline-2026-10-24.md)。Windows 上 RTF≈0.99 贴线，见[Windows 生产记录](moss17-windows-production-2026-10-24.md)；官方常驻对照因容量下界 **13.22GB > 12.79GB** 显存被永久拒绝，见[官方 codec 契约](moss17-windows-official-codec-contract-2026-10-25.md)。它是 MOSS 唯一带 style/instruction/language 的产品档位；移除即失去这些能力，用户已接受。辅助 VoiceDesign 的描述输入不构成朗读风格能力。
- **Nano ONNX**：自 2026-10-10 人工听感确认后，Nano 默认已切换 Candle，见[Candle 试用记录](moss-nano-candle-trial-2026-10-10.md)。Windows CUDA 上 ONNX **0.694s vs Candle 0.461s**，见[Windows 记录](moss-nano-onnx-windows-2026-10-24.md)；Mac 上墙钟 **1.402 vs 1.417s** 平手，见[macOS 基线](moss-macos-performance-baseline-2026-10-24.md)，该记录明确 **RTF 不可比**。ONNX 独自承担静态 ORT 分发缺 sm120 内核、动态 ORT 加载机制和未验收打包的维护成本。
- **DirectML**：从未成为产品 Device，仅为实验入口；评估结论为 iGPU 慢于 CPU，见[DirectML 评估](moss-directml-evaluation.md)。

以上历史记录保持原始结论，不回写为当前能力。

## 行为与数据

能力目录只列保留的已编译档位。默认构建选择 `nano-candle`；仅启用 `moss` 的构建选择 `realtime-1.7b`。Realtime 仍仅开放已编译且可用的 GPU，纯 CPU 构建可查询目录，显式准备无可用设备时失败。

旧 `nano`、`local-1b`、`local-1.7b` 请求在模型解析时返回包含所请求身份与本构建支持列表的 `unsupported or retired model` 错误。已有 MOSS 配置在启动时先校验，即使 CLI 传入覆盖模型也不把它改写成新模型。协议 v7 不变，不增加协商或回退。用户可自行编辑配置后重试。

不扫描、复制、删除或改写用户的旧资源、音色及检查点。Nano 导入改用原生 Candle CPU encoder，48kHz stereo 交错 PCM，仍执行原 WAV 时长、非静音、采样和音色 ID 校验及原子不覆盖保存。其音色现位于独立的 `moss/models/nano-candle/44502f80dbf9743528fa921cc544d662c685ebec/voices`；旧 `moss/voices` 留在原处，不读取或自动迁移。Realtime 音色目录独立。

## Feature 与发行

- `moss` 启用 Candle 1.7B 基础路径；`moss-nano-candle` 在此基础上启用 Nano 并将其作为默认。
- 保留 `moss-candle`、两个档位各自的 `-metal` / `-cuda` 和 Realtime reference-anchor 实验开关。
- 删除 `directml-probe`、`ort-dynamic`、`ort-coreml`、`ort-cuda` 及 facade 转发。后两者经核查仅服务退休 Nano provider；ASR SenseVoice 与 CosyVoice 前端/数值对照仍保留各自 CPU ORT 依赖。根工作区 ORT pin 不变。
- MOSS feature 不再启用 `dep:ort`；保留 Candle/text 使用的 sentencepiece-rs、hound、rubato 等依赖，删除退休 runtime 独用的后端 rand、Windows D3D 依赖。
- CI 单 feature 及原生 GPU 编译组合均使用保留的 Candle 路径；标准发行已有两个 Candle 档位，无需引入新协议或依赖版本。

普通验证不下载模型，编译和夹具不替代真实音频、数值与性能验收。CUDA/Windows 及真实模型导入仍须在对应环境另行验收。

## 本次本地验证

macOS 上全程使用离线 Cargo，不下载模型：

- `cargo build --offline -p talechime` 通过；默认 v7 Hello 的 MOSS 目录精确为 `nano-candle`、`realtime-1.7b`，隔离配置/资源/检查点目录未创建文件。
- 扩展 CPU workspace 的 `--lib --tests --examples -- --test-threads=1`：**825 passed、0 failed、42 ignored**。覆盖 Qwen、VoxCPM、OmniVoice、CosyVoice、MOSS、ASR、IndexTTS 和 performance-tools。真实模型测试保持忽略或环境门控。
- 纯 `moss` 后端库测试 **17 passed**；Nano + reference-anchor **39 passed**；默认协议进程测试 **7 passed**，纯 `cli,moss` 协议进程测试 **6 passed**。退休配置失败不覆盖、旧目录不迁移、音色目录隔离、WAV stereo 交错均有回归覆盖。
- 扩展 CPU workspace、无模型、纯 `cli,moss` 和 MOSS Metal/reference-anchor 的 all-targets Clippy 均 `-D warnings` 通过；扩展 CPU 与纯 MOSS rustdoc 均 `-D warnings` 通过；Metal all-targets 编译通过。
- `cargo fmt --all --check`、16 个发布/架构脚本测试、cargo-dist 0.32.0 `generate` / `generate --check` 通过。现有发行 feature 与原生矩阵已是两个 Candle 路径，重新生成未产生发行 workflow 差异。
- `crates/` DirectML 零命中，ONNX 命中仅为 ASR/CosyVoice；纯 `moss` normal 依赖树无 ORT。提取后的 Nano prompt tokens 与内置 voice codes 与原资产逐值一致；根 ORT pin 和既有历史记录无改动。

未执行 Windows/Linux 原生构建、CUDA 编译、真实模型合成、音色导入音频/数值对照或性能测试；本次不据夹具宣称这些验收通过。未发布库、创建标签或部署。
