# IndexTTS 2.5 后端

`indextts` / 模型 `2.5` 已接入 Talechime Registry、Rust Engine、CLI 与 JSON Lines v7。当前为实验能力：参考音频克隆、中文或英文文本、22.05 kHz 单声道输出。模型一次生成完整语义段，`native_streaming=false`；支持命名情绪风格与情绪控制参数；不提供执行级接续或自动音色设计。播放层仍可调整速度。

## 构建与资源

CPU 使用 `indextts` feature；macOS Metal 使用 `indextts-metal`；Linux / Windows CUDA 使用 `indextts-cuda-cudnn`，需要匹配的 CUDA 和 cuDNN。CUDA 采用 cuDNN 卷积路径，避免纯 Candle CUDA 分组卷积的已知布局限制。标准安装包目前不包含此实验后端，须从源码编译。

```sh
cargo build --release --locked -p talechime --features indextts
# macOS
cargo build --release --locked -p talechime --features indextts-metal
# Linux / Windows，具备 CUDA 与 cuDNN 工具链
cargo build --release --locked -p talechime --features indextts-cuda-cudnn
```

后端只读取本地转换文件，不自动下载官方权重或启动 Python。使用固定的 [IndexTTS 2.5 上游来源](../crates/indextts/SOURCE.md)和 [历史转换脚本](https://github.com/yexiyue/talechime/blob/e591c94/tools/tts/indextts_export_safetensors.py) 转换（脚本已从当前树删除，按需从上述固定 Git 快照取回）；转换环境与步骤见[移植记录](records/indextts/indextts-candle-port.md)。以 `--model-dir ROOT` 为资源根，将转换输出放入：

```text
ROOT/indextts/models/2.5/c39ce5ba-candle-v1/
```

该目录需包含 `config.json`、`gpt.safetensors`、`codec.safetensors`、`s2mel.safetensors`、`campplus.safetensors`、`bigvgan.safetensors`、`bigvgan.config.json`、`feats.safetensors`、`multilingual_zh_ja_yue_char_del.tiktoken`，以及 `w2v-bert-2.0/{model.safetensors,config.json}`。准备入口检查文件非空，模型加载再校验结构。开发性能工具另外使用已核验转换版本的 [固定摘要清单](../crates/talechime-backends/src/indextts/conversion-manifest.json)，校验全部文件且拒绝逃出自定义资源根的联接；序列化不同但数值相同的自定义转换不会自动获此性能收据。来源与逐张量核验见 [Windows 调查](records/indextts/indextts-windows-production-2026-10-24.md)。不提供转换权重下载，模型权重和用户音频不提交仓库。

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

## 情绪控制

`--style` / `SpeechSpan.style` 接受以下命名情绪，不是任意自然语言指令：
`happy`（高兴）、`angry`（愤怒）、`sad`（悲伤）、`afraid`（恐惧）、
`disgusted`（厌恶）、`melancholic`（低落）、`surprised`（惊讶）、`calm`（平静）。
英文不区分大小写，也接受表内中文名称。

```sh
talechime --backend indextts --voice custom:reader --style sad --param emo_alpha=0.6 chapter.txt
# 混合情绪；未指定的分量为零
talechime --backend indextts --voice custom:reader --param emo_sad=0.5 --param emo_calm=0.2 chapter.txt
# 独立情绪参考，使用同一 IndexTTS 模型下已导入的录音
talechime --backend indextts --voice custom:reader --param emo_voice=custom:emotion --param emo_alpha=0.7 chapter.txt
```

八个分量为 `emo_happy`、`emo_angry`、`emo_sad`、`emo_afraid`、`emo_disgusted`、
`emo_melancholic`、`emo_surprised`、`emo_calm`，各取 0..=1；总和大于 0.8 时调用
模型的官方归一化规则缩放到 0.8。未传任一分量时不启用向量，保持原参考音频情绪。
`emo_alpha` 为 0..=1 强度，默认 1；`emo_use_random=true` 可为情绪向量随机选择
锚点（受 seed 控制，可能降低相似度），默认 false。

命名 style、显式向量、`emo_voice` 三种来源互斥；强度必须配合其中一种来源，
随机锚点只适用于 style/向量。非法参数或冲突显式报错。情绪参考只改变表达，
说话人身份仍由 `--voice` 指定；每次请求清除前次情绪参考，未设置时恢复默认。

Rust 通过 `talechime::IndexTtsParams` 的 `emotion_vector([f32; 8])`、
`emotion_strength(f64)`、`emotion_voice(id)`、`emotion_random(bool)` 构造参数，
再用 `to_generation_params()` 放入 `SynthesisOptions.params` 或 `PlanSessionOptions.params`。
向量顺序即上文八种情绪顺序。模型的文本自动情绪分析没有作为参数开放；本次不引入额外分析模型。
接线和无模型测试不代替真实模型情绪听感验收。
