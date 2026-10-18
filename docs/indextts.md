# IndexTTS 2.5 后端

`indextts` / 模型 `2.5` 已接入 Talechime Registry、Rust Engine、CLI 与 JSON Lines v7。当前为实验能力：参考音频克隆、中文或英文文本、22.05 kHz 单声道输出。模型一次生成完整语义段，`native_streaming=false`；不提供逐段风格、执行级接续或自动音色设计。播放层仍可调整速度。

## 构建与资源

CPU 使用 `indextts` feature；macOS Metal 使用 `indextts-metal`；Linux / Windows CUDA 使用 `indextts-cuda-cudnn`，需要匹配的 CUDA 和 cuDNN。CUDA 采用 cuDNN 卷积路径，避免纯 Candle CUDA 分组卷积的已知布局限制。标准安装包目前不包含此实验后端，须从源码编译。

```sh
cargo build --release --locked -p talechime --features indextts
# macOS
cargo build --release --locked -p talechime --features indextts-metal
# Linux / Windows，具备 CUDA 与 cuDNN 工具链
cargo build --release --locked -p talechime --features indextts-cuda-cudnn
```

后端只读取本地转换文件，不自动下载官方权重或启动 Python。使用固定的 [IndexTTS 2.5 上游来源](../crates/indextts/SOURCE.md)和 [`tools/tts/indextts_export_safetensors.py`](../tools/tts/indextts_export_safetensors.py) 转换；转换环境与步骤见[移植记录](records/indextts/indextts-candle-port.md)。以 `--model-dir ROOT` 为资源根，将转换输出放入：

```text
ROOT/indextts/models/2.5/c39ce5ba-candle-v1/
```

该目录需包含 `config.json`、`gpt.safetensors`、`codec.safetensors`、`s2mel.safetensors`、`campplus.safetensors`、`bigvgan.safetensors`、`bigvgan.config.json`、`feats.safetensors`、`multilingual_zh_ja_yue_char_del.tiktoken`，以及 `w2v-bert-2.0/{model.safetensors,config.json}`。准备入口检查文件非空，模型加载再校验结构；目前没有已发布转换产物的固定 SHA-256 清单，来源身份须由操作者按上游 revision 核验。模型权重和用户音频不提交仓库。

## 导入并朗读

```sh
target/release/talechime --backend indextts --model 2.5 --model-dir ROOT \
  voices import reader --name 朗读者 --text '参考音频的准确转写。' reference.wav
target/release/talechime --backend indextts --model 2.5 --model-dir ROOT voices list
target/release/talechime --backend indextts --model 2.5 --model-dir ROOT \
  --voice custom:reader --tts-device cpu chapter.txt
# 英文正文：显式设置语言
target/release/talechime --backend indextts --model 2.5 --model-dir ROOT \
  --voice custom:reader --param language=en english.txt
```

参考 WAV 必须有效、非静音，时长 1–15 秒，并提供准确转写。音色按后端、模型和转换版本隔离。库使用 `import_reference_voice` 导入音色，然后通过 `Engine::prepare` 选择同一资源根与模型；音频缓存身份暂不声明，因此替换参考文件不会命中旧合成缓存。

单段音频只有在模型正常返回、样本有效后才发布 PCM 和 End。消费者丢弃接收端会在语义 token、扩散步与模型阶段边界停止计算；单个同步算子执行期间仍需等它返回。停止、失败和空音频都不能推进播放完成检查点。普通测试不下载模型；[官方性能对照](records/indextts/indextts-candle-performance-2026-10-17.md)来自独立计算库，尚未重新做集成后端的实机试听、跨平台运行及长期播放验收。
