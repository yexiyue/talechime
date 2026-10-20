# TTS 入门：从零到 2026 主流格局

> 个人学习笔记。日期：2026-10-16。内容为通用 TTS 知识 + 2026 年格局调研，末节把 talechime 手里的模型对号入座。
> 讨论背景：评估用 Candle 复刻 CosyVoice 3.0 的可行性时补的基础课。

## 一、所有 TTS 都是这个骨架：前端 → 声学模型 → 声码器

```mermaid
flowchart LR
    A[原始文本] --> B[文本前端<br/>正则化 TN + 注音 G2P]
    B --> C[声学模型<br/>产出中间表示]
    C --> D[声码器<br/>中间表示 → 波形]
    D --> E[PCM 音频]
```

**文本前端**：把"3.14kg"变成"三点一四公斤"（正则化，TN）；把"银行/行李"里两个"行"标上不同读音（注音，G2P）。
前端决定读音，后面所有环节只是照着执行——这就是 talechime"前端瓶颈"的含义：Kokoro、ZipVoice 这类模型的读音错误都源自这一层。

**声学模型**：把处理好的文本变成"中间表示"。中间表示有两种，这是区分技术路线的关键：

- **mel 频谱**：把声音按"时间 × 频率"切成小格子，每格记能量，像声音的指纹底片。人耳关心的音色信息大部分在里面，数据量比原始波形小几十倍。
- **离散语音 token**：2023 年后兴起。用神经网络（神经编解码器，codec）把 1 秒音频压成一串"词汇表里的编号"，1 秒语音约 12~75 个 token（CosyVoice 每秒 25 个；talechime 里 qwen3-tts 的 codec 是 12Hz）。压缩原理像套娃：第一层 token 抓语义内容，后面几层逐级补音色细节。

语音一旦变成 token，"生成语音"就变成了"预测下几个 token"——这正好是语言模型（LLM）的老本行。这是 TTS 界 2023 年后大变局的起点。

**声码器**：把 mel 或 token 还原成能播放的波形（HiFiGAN、Vocos 这些名字就是干这个的）。纯卷积网络，各家通用，没有路线之争。

## 二、AR（自回归）是什么

**AR = autoregressive：一次只生成一个 token，每个新 token 都以"之前所有输出"为条件。**

类比：接龙写小说，一个字一个字往外写，每写一个字都把前文重读一遍。工程上用 KV 缓存避免真的重读——把读过的部分记在提词卡上，只处理新字。GPT 生成文字就是这个工作方式；VALL-E（微软，2023）第一个把它搬进 TTS，之后 CosyVoice、Qwen3-TTS、MOSS、VoxCPM 全是这条路。

**优点**：

- 韵律自然。"下一个音怎么念"参考了全部前文，停顿、语气、情绪起伏像人念稿一样有全局感。
- 天然流式。写出几十个 token 就能开始出声，首包延迟低。
- 能复用整个 LLM 工具链（采样、top-p、重复惩罚都是 LLM 的原装概念）。

**毛病**：

- 慢。串行依赖无法并行：10 秒语音 ≈ 250 个 token ≈ 250 次前向计算。
- 会滚雪球。错一个 token 可能带偏后面一串——"复读机"现象是 AR 的经典病。
- 时长不定。念多长由模型自己决定，可能跳词漏字（验收时盯 EOS 行为就是在防这个）。

## 三、非 AR（并行生成）：扩散 / 流匹配一族

**核心思想：不一个字一个字写，而是从一团噪声里把整段语音"雕"出来。** 先给一张纯噪声的"画布"，模型每一步把整幅画面同时细化一点，固定迭代 N 步（通常 10 步左右）出成品。类比：雕塑家对着大理石一遍遍整体打磨，逐渐显形——而不是从脚趾开始逐厘米雕刻。

- **扩散（diffusion）**：老一点的说法，逐步去噪。
- **流匹配（flow matching，FM）**：同一思想的升级数学路线——学一个"从噪声指向目标的速度场"，训练更稳、步数更少。2024-2026 年的新主流；ZipVoice、F5-TTS 和 CosyVoice 的声码段都是 FM。DiT（Diffusion Transformer）就是用 Transformer 当这个"雕刻手"。

**和生图领域的关系**：扩散模型确实是生图先火起来的——Stable Diffusion、Midjourney、DALL-E 2 全是"从噪声画出一幅画"，TTS 是把这个思想借来"雕语音"。有趣的是升级版的传播方向反了过来：Meta 的 Voicebox（2023）先在语音界把流匹配用火，生图界才跟进——Flux、Stable Diffusion 3 都是 2024 年改用流匹配路线的。另一个差别：生图没法 AR，图像几十万个像素逐个生成太慢，所以生图界天然全是并行生成；语音能压缩成每秒几十个 token 的短序列，AR 和并行两条路都走得通——这道"选择题"是语音圈独享的。

**优点**：步数固定所以快且时长可控、不会复读、整段并行生成。
**缺点**：要预先知道"画布多大"（时长预测是独立难题）；流式要切块做，切缝处理是技术活；文本与声音的对应全靠注意力去学，精细控制弱。

一句话记两边的性格：**AR 像人念稿——自然、有灵气，但会口误；非 AR 像排版印刷——稳定齐整，但偶尔呆板。**

## 四、2026 主流格局：四大范式

| 范式 | 代表模型 | 性格 | 典型用途 |
|---|---|---|---|
| ① 轻量并行声学模型（VITS/StyleTTS2 系，GAN） | Kokoro、Piper | 音素输入、CPU 实时、体积小；无克隆、表现力弱 | 边缘/草稿档 |
| ② 纯 AR token（LLM 直改） | VALL-E（开山）、Orpheus、Dia、Sesame CSM；闭源：ElevenLabs、MiniMax Speech | 韵律最强、流式好；慢、会复读 | 商业旗舰、研究 |
| ③ AR token + 流匹配声码器（混合，当前开源主流） | CosyVoice 1/2/3、VoxCPM、MOSS、Qwen3-TTS、Fish/OpenAudio、IndexTTS-2 | "大脑用 AR 想韵律，双手用流匹配快速上色"：AR 出语义 token，FM 刷成 mel，声码器出波形 | 全能主力 |
| ④ 纯并行扩散/流匹配（DiT） | F5-TTS、E2-TTS、ZipVoice、MegaTTS3、VibeVoice | 克隆快、稳定、结构简单；精细控制弱 | 快速克隆、批量合成 |

③为什么是主流：两个部件各干擅长的事——AR 负责"说什么、什么节奏"（全局韵律最强），FM 负责"声音长什么样"（并行速度最快）。

（史前史一句话带过：统计参数合成与拼接合成（unit selection）是神经 TTS 之前的时代，博物馆藏品。）

**前沿趋势**：

- 规模化 + RL 后训练：CosyVoice 3 论文（100 万小时数据、1.5B、可微奖励模型）代表这个方向。
- 流式优先：双向流、chunk-aware 流匹配（首包 150ms 级）。
- 新混合体在学术阶段：离散流匹配（DiFlow）、AR+FM 统一建模（UniVoice）。
- 多模态大模型内置 TTS：Qwen3-Omni 家族，Qwen3-TTS 就是从那里独立出来的。

## 五、talechime 的档位表翻译成范式语言

| 档位 | 模型 | 范式 | 为什么是它 |
|---|---|---|---|
| CPU 草稿 | Kokoro | ① | 草稿要快、稳、免费——①的全部优点，无情感可容忍 |
| GPU 轻量（候选） | ZipVoice | ④ | FM 少步并行，听感好于①；音素前端债未清 |
| GPU 草稿+终稿 | Qwen3-TTS / MOSS / VoxCPM，未来可能 + CosyVoice 3 | ③ | 终稿要自然、可克隆、可控——③的全家桶 |
| 云端终稿 | MiniMax Speech | ③（闭源版） | 同范式换算力，质量顶配 |

规律：**档位的高低，本质是"读音大脑和韵律大脑放在哪一层"的差别**——①的大脑是规则前端（最弱）、③的大脑是 0.5B LLM（最强）、④介于中间靠注意力硬学。所以 CosyVoice 3 的拼音纠正标签（pronunciation inpainting）价值在于：给③的大脑配上 talechime 自建的神经消歧前端，把两端优势叠起来。

## 参考来源

- [SiliconFlow：2026 最佳开源 TTS](https://www.siliconflow.com/articles/best-open-source-text-to-speech-models)
- [BentoML：The Best Open-Source TTS Models in 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [OpenVoxAI：2026 本地与云 TTS 评测](https://openvoxai.com/blog/best-tts-models-local-cloud-2026)
- [Dataroot Labs：Text-to-Speech 模型综述](https://datarootlabs.com/blog/text-to-speech-models)
- [Emergent Mind：Flow Matching TTS 专题](https://www.emergentmind.com/topics/flow-matching-based-tts-model)
- [Reddit r/LocalLLM：2026 开源 TTS 讨论](https://www.reddit.com/r/LocalLLMA/comments/1uh2xyh/what_is_the_best_opensource_tts_model_right_now)

## Qwen3-TTS 性能补充（2026-10-18）

Qwen3-TTS 是“主 Talker + 子 CodePredictor + 12Hz codec 解码器”的结构：主模型逐帧采样语义 token；每一帧再由 CodePredictor 顺序预测 15 个声学 token；最后把帧交给 codec 解成 PCM。因此每个音频帧不是一次 Transformer 前向，而是约一次主模型前向加 15 次 CodePredictor 前向。优化应先分清这三段，再比较整体 RTF。
