# 韵律标注模型选型：免训练现成件调研

日期：2026-10-29。背景：韵律规划路线（[语气衔接调研](prosody-coherence-research-2026-10-29.md)路线三）确定后，需要现成可推理的语义/情绪标注模型作为原型标注器、蒸馏教师与评测对照；用户明确不自己训练。本轮为桌面调研，未下载权重、未实测中文小说语料效果。

## 结论

免训练候选按四个用途给出，全部可在现有硬件（M4 Pro 24 GB / RTX 5070 12 GB）轻量运行，BERT 系走 Candle 或 ORT 均成熟：

| 用途 | 首选 | 备选 | 规模 |
| --- | --- | --- | --- |
| 段落级离散情绪标签（主对照） | 达摩院 StructBERT 七分类 `iic/nlp_structbert_emotion-classification_chinese-base` | 同系列 tiny/large；Erlangshen-110M-Sentiment（正负二分类，有 ONNX 版） | base ≈110M |
| 情绪强度（映射 IndexTTS `emotion_strength`） | 无可靠现成件；用 LLM 输出离散档位 | `liudev/chinese-roberta-novel-regression`（中文小说情绪回归，仅参考件） | ≈110M |
| LLM 零样本逐段演绎指令（原型主力） | Qwen3-0.6B/1.7B/4B-Instruct，参照 Alexandria 的 schema | 任意 OpenAI 兼容服务 | 1.7B bf16 ≈3.4 GB |
| 语义断点排序（issue #3） | `BAAI/bge-small-zh-v1.5`（维持原计划） | `gte-small-zh`（MIT，更小更快，综合分略低） | ≈24M |

## 逐项证据

### 达摩院 StructBERT 七分类（首选离散标签）

[ModelScope `iic/nlp_structbert_emotion-classification_chinese-base`](https://www.modelscope.cn/models/iic/nlp_structbert_emotion-classification_chinese-base/summary)：恐惧、愤怒、厌恶、喜好、悲伤、高兴、惊喜七类，官方口径准确率 90%+，另有 [tiny](https://www.modelscope.cn/models/damo/nlp_structbert_emotion-classification_chinese-tiny) 与 large 版，同系列还有[情感二分类](https://www.modelscope.cn/models/iic/nlp_structbert_sentiment-classification_chinese-base/summary)。七类粒度与朗读叙事主情绪吻合，可直接映射 CosyVoice3 Instruct / VoxCPM description 的情绪档。注意：训练分布以对话/评论为主，小说叙述文本上的实际表现需用本产品语料小样本核验，90%+ 是其原数据集口径。

### Erlangshen 系列（极性校验）

[Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment)（2022）：正负二分类 ~96–97%，有社区 [ONNX 导出版](https://huggingface.co/onnx-community/Erlangshen-Roberta-110M-Sentiment-ONNX)。二郎神系列**没有**现成多情绪分类版本（SMP2020/CPED 风格需自行微调，不符合免训练前提），只能当极性维度用。

### 情绪强度回归（缺口）

`liudev/chinese-roberta-novel-regression`：chinese-roberta-wwm-ext 底座（与 CastGlean 计划共享底座）、中文小说情绪强度单标量回归、Apache-2.0。但模型卡无训练数据与指标说明、月下载个位数，只可作为实验对照件，不作为教师或主判。**强度维度当前没有可靠免训练件**，建议由 LLM 打标时输出离散档位（强/中/弱）代替连续回归。

### LLM 零样本打标（原型主力 + 蒸馏主教师）

[Alexandria](https://github.com/Finrandojin/alexandria-audiobook)（MIT）是本路线已被验证的开源同型：整书由 LLM 转为 `{speaker, text, instruct}` JSON——instruct 为 2–3 句直接喂 TTS 的演绎指令（"Startled and fearful…"），非言语声写成可读文本；按说话人分组（~500 字符）合成；跨块传角色名册与最近 3 条条目维持连续性（纯文本上下文滚动，无音频回灌）；另有第二遍 "Review Script" 自动修正说话人误归属、过度切分与非法 instruct；TTS 端用 Qwen3-TTS（与本产品后端同型）。它推荐非思考类 LLM（思考模型需禁 `<think>` 才能稳定出 JSON）。其 schema、双遍修正与名册机制可直接作为本产品原型与对照样本；Qwen3 小模型零样本中文情绪打标无公开基准，原型期需人工抽检小样本。

### 语义断点（issue #3 维持原计划）

[bge-small-zh](https://www.datalearner.com/ai-models/pretrained-models/bge-small-zh)（~24M/512 维，CPU 友好）与 [gte-small-zh](https://www.datalearner.com/ai-models/pretrained-models/Gegeral-Text-Embeddings-Small)（MIT）均为轻量候选；2026 年社区对比（如[繁中实测](https://ihower.tw/blog/12167-embedding-models)、[BGE vs GTE](https://faun.dev)）倾向 BGE 综合分更高、GTE 体积速度占优。原计划选 bge-small-zh-v1.5 仍合理；piccolo-base 成绩更好但更大，Conan 系无小杯。

## 对 Talechime 的建议

1. 原型期双通道对照：LLM（Qwen3-1.7B/4B-Instruct，Alexandria 式 schema）打 `instruct` 标签 + StructBERT 七分类独立打离散情绪；两者不一致的段落自动标记为疑点（延续"自动指标只标记不选胜者"纪律），人工抽检。
2. 极性维度加 Erlangshen 二分类做廉价校验（有 ONNX 版，走现有 ORT 线）。
3. 强度档位由 LLM 离散输出，映射 IndexTTS `emotion_strength`；novel-regression 只作旁证。
4. 蒸馏期（CastGlean P0-P5）：LLM 打标为主教师，StructBERT 七分类为副教师与评测锚点；评测先行，在授权小说语料上先建人工标注小金标集。
5. 所有候选在小说叙述文本上的真实效果未经核验，原型第一步即用本产品语料跑小样本对照。

本轮未修改 OpenSpec 任务、生产默认或代码。
