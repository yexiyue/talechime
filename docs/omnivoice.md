# OmniVoice 推理与验证

当前固定模型为 0.6B，声学模型和 codec 由资源清单固定。普通合成、参考音色及设计
使用同一原生管线；宿主继续传原文，不在公共层强制中文或展开年份。

未指定 `language` 或设为 `auto` 时传递官方的空语言条件；显式值经模型语言表校验，
可使用代码或受支持的名称。字符可编码、音频存在或输入条件对照通过，都不表示该
语言的朗读质量已通过。参考音色显式绑定公开参考 WAV 与转写，不能由参考语言
推断目标语言或把未知目标语言改成中文。

设计和 style 使用固定指令标签，支持性别、年龄、音调、耳语及口音/方言等分类，
例如 `male, middle-aged, low pitch` 或 `男，中年，低音调`。
同一分类冲突、未知标签、混用互斥口音/方言均明确报错；不接受通用自由描述
`A calm middle-aged male voice, speaking clearly.`。完整标签和校验归模型的
`frontend::voice_design` 模块，不能静默忽略不认识的描述。

公开生成参数为 `num_step`（1–4096，默认 32）、`guidance_scale`（0–10，默认 2）、
`language` 和 `speed`（0.5–2）。原生 speed 改变目标 token 数；不是生成后的重采样。
可选官方 `normalize_text` 默认关闭，本适配器没有把它声明成额外公开控制。

2026-11-11 至 12 日的本机验证：

- 固定官方 Python `08be0b4ccbac3e13e374e86fbfead4b4cac343e2` 的输入条件 AST：
  原有 7 项及 19 条公开文本 × 7 种条件，共 140 项离线对照通过。使用合成 byte
  tokenizer/参考码，只证明输入结构，不冒充 checkpoint token 或权重数值等价。
- 官方时长规则覆盖同一 19 条文本、无/有参考及三档速度，共 114 项对照通过。
- Metal 基础文本 19/19 生成。中/英文参考、设计、显式语言、快/慢语速及生成参数
  共 21/21 控制样本生成；最初通用自由描述的三项拒绝记录保留，改用有效标签后重跑。
- 混读样本 speed=1.3 为 2.92 s，speed=0.7 为 5.28 s；这些是生成记录，表达、
  读音、相似度和连续性仍待人工听验。CPU/CUDA 真模型本轮未执行，CUDA 后续另排。

固定来源与社区移植的区别见 [SOURCE](../crates/omnivoice/SOURCE.md)；执行路径及历史
数值/性能限制见[性能指南](knowledge/omnivoice-performance.md)。本轮完整机器证据和
试听入口由[能力验收](records/tts-core/official-backend-parity-2026-11-11.md)维护。
