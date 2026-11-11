# IndexTTS 2.5 后端

当前工作区使用 `c39ce5ba-runtime-f32-front-v6` 固定配方准备及校验资源；本轮改动尚未发行。
下文涉及 `c39ce5ba-native-f32-v2` 的原生构建与耗时为源配方验证记录。
固定分发、许可与两类配方边界见[公开成品记录](records/resources/runtime-distribution-2026-11-05.md)。

`indextts` / 模型 `2.5` 已接入 Talechime Registry、Rust Engine、CLI 与 JSON Lines v7。当前为实验能力：参考音频克隆、中文/英文/日文/西文/阿拉伯文文本、22.05 kHz 单声道输出。模型一次生成完整语义段，`native_streaming=false`；支持命名情绪风格、情绪控制参数和官方 `duration_factor` 时长倍率（0.5–2，数值越大越慢）；不提供执行级接续或自动音色设计。时长倍率的真实模型听感尚未验证；播放层也可调整速度。

`text_normalization` 默认开启，中英使用固定官方平台 FST，西语使用固定 NeMo FST；调用方可设为 `false` 跳过 TN。关闭它不会跳过模型所需的标点、大小写、读音标注和日语分词。

适配器接受 `language=zh/en/ja/es/ar`，默认 zh。日语与阿语 TN 透传；日语使用固定 UniDic 分词。各语言的生成与听验分别记录，不能由参数可接受推断听感通过。

## 构建与资源

CPU 使用 `indextts` feature；macOS Metal 使用 `indextts-metal`；Linux / Windows CUDA 使用 `indextts-cuda-cudnn`，需要匹配的 CUDA 和 cuDNN。CUDA 采用 cuDNN 卷积路径，避免纯 Candle CUDA 分组卷积的已知布局限制。标准安装包目前不包含此实验后端，须从源码编译。

```sh
cargo build --release --locked -p talechime --features indextts
# macOS
cargo build --release --locked -p talechime --features indextts-metal
# Linux / Windows，具备 CUDA 与 cuDNN 工具链
cargo build --release --locked -p talechime --features indextts-cuda-cudnn
```

SDK 通过固定配方原生准备资源，不启动 Python、PyTorch 或 MLX。先调用
`ResolvedModel::native_preparation_plan()` 展示缺失下载量、待转换组件和峰值额外空间，
确认指纹后调用 `prepare_native_resources`；仅加载已就绪资源使用 `load_existing`，不会隐式下载。
完整 API 与取消语义见[原生准备说明](native-resource-preparation.md)。

当前开发资源版本目录为 `ROOT/indextts/models/2.5/c39ce5ba-runtime-f32-front-v6/`。
配方包含固定 UniDic 词典、平台中英/西语 FST 及 QwenEmotion 六件资源；规模和逐件摘要以编译清单为准。
此前 v3 的 14 个输入、13 个输出和 6,676,865,606 字节是 2026-11-05 历史基线。
GPT、codec、s2mel、CAMPPlus、BigVGAN 保持官方 F32；特征矩阵及 mean/sqrt(var)
由固定官方输入生成。词表、BigVGAN 配置和 w2v 文件直接验证复用。
源构建依据固化在 `indextts/native-assets.json` 和 `indextts/native-sources.json`，成品输入使用 `indextts/distribution-inputs.json`；
发布时验证每个输出与 `preparation.json`，缺件、混用版本或摘要不符均不能就绪。

2026-11-05 实测：2,707 个张量与独立 PyTorch 导出逐字节一致；在不含 Python 命令的
PATH 中完成完整准备及就绪复验，用时 173.89 秒。社区 FP16 资源的名称/布局及精度损失
已逐项核验，未用于本配方。历史 feats 摘要差异仍保留记录，未改写旧基线。
详见[来源与逐张量记录](records/resources/indextts-community-2026-11-05.md)。
CPU/Metal 真实短文本合成已生成有效 WAV；人工听感和其他平台验收仍未完成，详见[补充记录](records/resources/native-preparation-2026-11-05.md)。

参考音色位于 `ROOT/voices/indextts/` 对应模型子目录，由引擎音色存储管理，
不写入不可变的模型资源版本。权重、生成音频和转换工作目录不进入 Git。

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

参考 WAV 必须有效、非静音，时长 1–15 秒，并提供准确转写。音色按后端和模型隔离，与资源转换版本分开保存。库使用 `import_reference_voice` 导入音色，然后通过 `Engine::prepare` 选择同一资源根与模型；音频缓存身份暂不声明，因此替换参考文件不会命中旧合成缓存。

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

情绪来源按官方优先级选择：文本分类 > style/显式向量 > `emo_voice`。
未启用文本分类时 style 与显式向量互斥；高优先级来源覆盖情绪参考。
强度须配合一种来源；随机锚点适用于向量（含文本分类结果）。非法参数显式报错。情绪参考只改变表达，
说话人身份仍由 `--voice` 指定；每次请求清除前次情绪参考，未设置时恢复默认。

Rust 通过 `talechime::IndexTtsParams` 的 `emotion_vector([f32; 8])`、
`emotion_strength(f64)`、`emotion_voice(id)`、`emotion_random(bool)` 构造参数，
再用 `to_generation_params()` 放入 `SynthesisOptions.params` 或 `PlanSessionOptions.params`。
向量顺序即上文八种情绪顺序。`emotion_text(text)` 启用固定 QwenEmotion；
CLI 使用 `--param use_emo_text=true --param emo_text=高兴地说话`，省略 emo_text 时分析当前正文。
直接加载前验证六件固定资源，缺件或摘要错误明确失败。文本分类结果按官方直接使用，
不执行调用方显式向量的 0.8 缩放。
接线和无模型测试不代替真实模型情绪听感验收。

## 固定文本前端（2026-11-11，未发行）

适配器接受 `language=ja`，使用固定 unidic-lite 1.0.8 / UniDic 2.1.2
词典及 Vibrato 0.5.3，匹配官方 `JapaneseG2PProcessor(g2p_ratio=0)`：
保留原字和连续 ASCII 空格，分词表面形式之间插入空格；不把日语年份展开为中文。
先执行官方清理/大小写及读音标注，再分词和恢复特殊标记。
所有 14 条独立官方分词及完整前端对照在 macOS CPU 通过。
词典与算法身份进入 front-v6 资源版本；缺件或摘要不符明确失败，不使用近似分词。

中英规范化已替换近似数字规则，使用固定 wetext 0.1.0 FST 与默认配置。
20 条官方 `TextNormalizer` 对照通过，包含日期、金额、型号、百分比、拼音保护和全角数字。
这是 macOS/Windows 的官方 WeText 基线；本机只执行 macOS CPU，
Linux WeTextProcessing 1.3.0 的独立基线另有 20 条对照通过。
来源、SHA 与许可证见 [SOURCE.md](../crates/indextts/SOURCE.md) 及 native-sources.json。

真实日语适配器四条样本已生成有效 WAV（旧 front-v5），人工听验待确认；
front-v6 全矩阵及完整情绪/时长生成继续由
[能力补齐变更](../openspec/changes/finish-official-backend-parity/tasks.md) 跟踪。

本轮追加：固定 Linux WeTextProcessing 1.3.0 和西语 NeMo 1.2.0 语法已接入，
运行不调用 Python。两套中英基线各 20 条、西语 16 条官方对照已在 macOS 的
原生 FST 执行器通过；Linux oracle 来自隔离 Linux 容器，这不是 Windows/Linux
完整应用实机验收。压缩语法包约 5 MB，含 Apache-2.0 许可证和逐件摘要，
复现入口为 tools/tts/indextts_fst_export.py；源码树身份记录在 native-sources.json。
适配器现接受 zh/en/ja/es/ar；西语归一化后按官方转为大写，阿语 TN 透传。
QwenEmotion 在 macOS CPU 上通过 12 条独立输出转换 fixtures 与 4 条真实官方模型对照：
提示词 token 与八维结果一致。此证据不等于 Metal/CUDA 数值、情绪声音或全部能力听验通过。
