# 滚动上下文与长篇单音色稳定性：上游调研

日期：2026-10-26。范围为 Nano、VoxCPM2、Qwen3-TTS Base、OmniVoice、CosyVoice3、IndexTTS 2.5 和 MOSS Realtime 的公开 issue、PR、文档及当前官方代码。未下载模型、改推理默认或执行新一轮真实模型实验。

## 结论与边界

目前查到的材料支持优先验证“固定可信参考 + 模型预算内的语义组段”，并不支持把上一块生成音频一直滚动回灌作为通用长篇方案。固定参考也可能有片内漂移和块间语气差异，不能据此承诺所有模型整章稳定。

必须区分三件事：宿主每次用生成 PCM 更新参考；模型内部在一次生成中延续 tokens、latents 或 KV；每次独立生成但复用同一份音色参考。前两者都可能有误差积累，但机制不同，不能把原生流式解码或跨轮会话统称为滚动 WAV。

证据分级：官方代码证明接口与执行路径；维护者回复提供建议或承认限制；issue 作者提供现象与假设；PR 作者的本地试验不等于合入或普遍质量保证。自动关闭和机器人回复不视为问题已修复。

## 逐模型证据

### MOSS Nano

[Issue #65](https://github.com/OpenMOSS/MOSS-TTS-Nano/issues/65)报告文本切片后音色不一致和衔接音调跳变。协作者首先建议用干净、稳定的同一参考验证跨块稳定性；滚动上一块尾部只是可尝试的另一种策略，未提供长篇对照结果。该 issue 虽已关闭，不能理解为滚动漂移已有修复。

当前[官方 infer.py](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/infer.py)把 `voice_clone` 设为默认，与 `continuation` 分开，voice-clone 另有句子切分预算（当前默认 75 text tokens）。[官方 README](https://github.com/OpenMOSS/MOSS-TTS-Nano)也把 voice clone 作为推荐工作流。该值是上游当前配置，不直接搬成 Talechime 的中文字符预算，需匹配 pinned 模型与 prompt 模板。

[Issue #100](https://github.com/OpenMOSS/MOSS-TTS-Nano/issues/100)另报告 ONNX 增量解码相对整段解码的音高跳变，目前页面无维护者修复结论。它是另一类解码路径问题，不能用来证明本项目 Candle 滚动漂移的根因。需要固定同一份生成 codes 回放两种解码，才能隔离生成差异。

本项目源码已将无滚动的参考 codes 放在 user reference，而 continuation 将生成 codes 放入 assistant prefix。这不只是换了录音，也切换了输入模式。此前 Nano 固定/多句方案也被用户否定，故上游建议只能支撑一个有具体假设的新实验，不能撤销既有淘汰结论。

### VoxCPM2

[Issue #372](https://github.com/OpenBMB/VoxCPM/issues/372)报告普通 continuation 和 reference+continuation 在多个推理框架中均有长文本失真，贡献者确认长文本问题尚未完全解决。该报告主要是长单次生成，不能直接等同于宿主跨请求滚动；与本地 Windows/Mac 反馈共同提示，应排查模型条件稳定性，不能只归因于 Metal。

[PR #313](https://github.com/OpenBMB/VoxCPM/pull/313)提议语义分段、固定外部参考或冻结首个生成片段；截至查阅仍 Open。[PR #391](https://github.com/OpenBMB/VoxCPM/pull/391)也用同一 prompt cache 合成短块，截至查阅仍 Open。两者都不是已发布的根治补丁。

[Issue #302](https://github.com/OpenBMB/VoxCPM/issues/302)作者后续明确修正：固定锚点分块只是缓解，个别块末尾仍可能漂移；连续注入说话人身份、分离身份与韵律是研究方向，简单混合 latent 可能同时固定不需要的音质特征。这些诊断属于作者假设，不是维护者确认的统一根因。

当前 Talechime 保留用户认可的默认生成音色冻结参考，不滚动、不补静音。普通滚动和双参考滚动均有负面反馈。尚不能只因换固定参考改善，就断言 VAE 重编码是唯一原因。

### Qwen3-TTS Base

[Issue #80](https://github.com/QwenLM/Qwen3-TTS/issues/80)报告即使使用参考克隆，也可能在局部或整段变成其他声音；作者更换参考后仍有反例。固定参考是一条基线，不是绝对保证。

[Issue #341](https://github.com/QwenLM/Qwen3-TTS/issues/341)报告 ICL 输出在目标文字前多读参考尾部内容；其他使用者称换掉特定参考尾句后改善。尚无可核验的通用模型修复，不据此采用输出削头或重试掩盖问题。

[官方 prompt API](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)支持：固定音频、转写、codes、speaker embedding 的 ICL；或 `x_vector_only_mode=True`，只使用说话人向量，忽略参考 codes/转写。后者是一个可隔离声学上下文影响的实验候选，不能预先宣称音色更像、韵律更好。

本项目 `qwen/runtime.rs` 当前有上下文时从生成 PCM 重建 transient clone prompt。优先改实验方式为固定原始完整 prompt，并分别比较两种 Base 尺寸；是否额外试 speaker-only，按短样结果决定。

### OmniVoice

[Issue #44](https://github.com/k2-fsa/OmniVoice/issues/44)中协作者说明，voice design 的文字描述不能唯一固定跨次生成声音；要跨次一致，应使用 voice cloning。官方长文设计流程会先生成第一块，再把这块作为后续参考，而非不断替换成最新一块。

[官方参数文档](https://github.com/k2-fsa/OmniVoice/blob/master/docs/generation-parameters.md)提供约 15 秒目标分段、约 30 秒触发切分的配置。这是上游默认估计时长，并非本项目应直接采用的安全上限。[Issue #64](https://github.com/k2-fsa/OmniVoice/issues/64)仍有长片段中途加速报告，说明分段与参考选择均需验收。

建议冻结一次设计或导入的可信参考，复用同一 clone prompt，在预算内合并相邻句子，保持文本驱动的语气变化。遵从用户要求，本项目实验不引入该文档中的默认补静音或淡入淡出。

### CosyVoice3

[Issue #1932](https://github.com/QwenAudio/CosyVoice/issues/1932)报告跨文本块的语速、音调与韵律不一致。回复明确自称自动答疑机器人，issue 后因 stale 自动关闭，因此不能当作维护者证实的修复。

[官方普通 zero-shot 路径](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py)先分文本，各段复用同一 prompt speech 和 prompt text。[原生 inference_bistream](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/llm/llm.py)则可在单次调用中接收文本 Generator 并保留该生成过程的状态；它不是每块合成完再回灌 WAV。

建议固定参考的语义组段为基线；原生文本流连续会话可以作为单独的接口与生命周期实验，不能由 issue 中的机器人推荐直接认定已在本项目支持。旧版 300M 的流式音色变化报告也不能代替 CosyVoice3 的实际验收。

### IndexTTS 2.5

[官方 infer_v2_5.py](https://github.com/index-tts/index-tts/blob/main/indextts/infer_v2_5.py)对参考条件缓存后按 tokens 切分目标文字，每段复用 speaker/style/mel 条件，默认主流程没有把每段新生成 WAV 替换为下一段身份参考。

[Issue #775](https://github.com/index-tts/index-tts/issues/775)报告英文长段末句对齐坍塌，作者缩小 token 分段预算后改善；这是内容和长度敏感性报告，不是已证实的滚动音色 bug，也不支持直接给中文设相同阈值。

建议维持固定 speaker 条件，并单独约束情绪参考/强度；较大语义块与内容正确性一起验收。现有无跨请求 continuation 能力不必为全模型一致性而补上滚动 WAV。

### MOSS Realtime

[官方 model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)有原生文本/声学历史建模，并给出 `session.push_text` / `end_text` 示例。它提供了比外部滚动 WAV 更直接的状态连续实验方向，但面向实时多轮交互，不是中文整章单旁白稳定性的保证。

本轮未找到足以确认“本项目滚动音色变化已有官方修复”的 issue。固定参考继续作为基线；原生会话须单独验证状态上限、文本/codec 历史、取消及自然边界重置，不能宣传为零漂移。

## 对 Talechime 的建议

1. 参考身份固定：原始录音，或经试听选定后冻结的生成锚点。不能把“当前音色”逐块重新定义为最新生成结果。
2. 连续性先从组段获取：在模型预算内把语义相关的相邻句子合并；不退回每个很短的句子都独立合成，也不把整章塞成一次无限生成。
3. 优先复用原生参考 codes/features/embedding，避免为了更新上下文反复走 PCM→重采样→编码。即便完全免除重编码，自生成 tokens/latents 仍可能漂移，不能视为根治。
4. 若模型明确支持原生会话，单独验证文本流/KV/codes 连续；否则固定参考独立请求仍是有效候选，不为统一抽象强行补滚动。
5. 对已经发生多读漏读的片段，保留正文/音频失配证据，不能继续把它当作“准确文字+语音”的可靠上下文。ASR 和声纹只标记疑点，不自动替代听感验收。
6. 不把缩短滚动链、固定 seed、降温、latent 混合或补静音列为通用修复。当前 Nano 短链已有负面反馈；本轮用户明确不要补静音。

下一批优先级：VoxCPM 固定参考完整章验证；Nano 仅在有新参考或明确的固定-clone/组段假设时重启小样；Qwen 两种 Base 各自固定完整 prompt，再视反馈试 speaker-only；其余模型保留固定参考基线。以上为调研后的实验建议，未自动修改 OpenSpec 任务或产品默认。
