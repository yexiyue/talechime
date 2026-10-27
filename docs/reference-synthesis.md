# 固定参考合成

2026-10-27 起，所有后端仅使用各自官方的固定参考、预置音色或自然生成路径。生成结果不回灌为下一片段的参考；滚动 PCM、双参考及身份锚定实验接口、重锚预算和开关均已删除。

## 各模型的路径

| 后端 / 模型 | 保留的合成条件 | 官方依据 |
| --- | --- | --- |
| MOSS Nano Candle | 固定内置或导入 codes，使用 voice-clone user reference 模板；不把生成 codes 注入 assistant prefix | [官方 infer.py](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/infer.py) 的默认 `voice_clone`，固定实现模板 revision 仍由资源清单约束 |
| Kokoro v1.1-zh（源码实验） | 官方预置中文 voice style 表，按当前音素长度选行；可选 g2pW 只改变前端多音字读音 | [官方模型入口](https://github.com/hexgrad/kokoro/blob/dfb907a02bba8152ca444717ca5d78747ccb4bec/kokoro/model.py)与[当前移植边界](kokoro.md) |
| MOSS Realtime 1.7B | 所选固定参考，独立请求；保留单请求内部原生文字/音频调度和流式解码 | [官方模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md) 的参考与 single-turn 用法 |
| Qwen 0.6B / 1.7B Base | 从导入原始录音和准确转写建立固定完整 ICL prompt，重复使用同一音色缓存 | [官方 create_voice_clone_prompt](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py) |
| Qwen CustomVoice | 官方预置 speaker；1.7B 可带其支持的 style | [官方模型用法](https://github.com/QwenLM/Qwen3-TTS) |
| VoxCPM2 Q8 / BF16 | 固定单参考录音和转写，或无参考自然生成；保留官方 `(description)text` 风格格式 | [官方 core.py](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/core.py) 的单 prompt 输入；本产品不提供 ref+continuation 双输入 |
| OmniVoice | 固定 VoiceClonePrompt 或官方 instruct 生成；模型自身的长文本切分不更新为最新片段 | [官方稳定音色建议 #44](https://github.com/k2-fsa/OmniVoice/issues/44)、[长文参数](https://github.com/k2-fsa/OmniVoice/blob/master/docs/generation-parameters.md) |
| CosyVoice3 | 固定原始录音、准确转写和说话人条件，按语义分段及原生流式输出 | [官方 zero-shot 实现](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py) |
| IndexTTS 2.5 | 固定 speaker/style/mel 条件；情绪条件依该模型已有参数显式提供 | [官方 infer_v2_5.py](https://github.com/index-tts/index-tts/blob/main/indextts/infer_v2_5.py) |

自然生成或文字音色描述不是固定说话人身份的承诺。需要整篇同一设计音色时，先通过 `voices design` 生成一次、试听后选用该资产；后续请求复用这份固定录音。音色设计辅助模型不进入朗读模型目录，不偷偷用第一块或最新块更新音色。

## 连续朗读与状态

语义组段继续由模型的 token/时长预算约束。相同音色的相邻句子可在预算内合成，不要求每个短句单独起读；也不把整章作为无界单次生成。分块与 UTF-8 原文范围继续绑定。

一次生成内部的 transformer KV、卷积历史、codec overlap 和原生 PCM streaming 状态保持不变。这里删除的是宿主跨请求生成参考，并未取消正常流式解码。MOSS/CosyVoice 的原生跨轮或文本流 session 是独立能力，本产品没有新增该接口来替代旧滚动功能。

播放、导出和内容门禁只消费生成音频。门禁重试改变当前片段的采样 seed，不改变固定参考。原有边界处理仅裁剪多余静音，响度对齐有界；宿主不向参考或输出额外补静音，不新增削头、去噪或调速掩盖问题；模型原有停顿和官方处理保持其自身语义。

## 接口与存储

- `SegmentRequest` 仅包含 text、voice、style、seed、params；不再导出 `SpeechContext`。
- `PlanRequest`、`PlanSessionOptions`、`SynthesisOptions` 不再有滚动开关、边界或时长预算。旧 wire 字段由既有未知字段校验明确拒绝；旧 CLI 参数不再解析。协议仍为 v7，不提供兼容旧策略的分支。
- 删除双参考/身份锚定 Cargo features、模型实验方法和滚动对照探针；性能探针请求也不再接受旧滚动字段。
- 章节缓存使用新的固定参考策略键，旧生成内容不会作为当前缓存命中。用户原始录音、音色库、模型权重、TRNovel 数据和历史实验音频均不迁移、不删除。

## 验收边界

官方支持的输入模式不代表所有参考、长度或模型都通过长篇听感验收。历史 Nano 负面反馈继续保留；VoxCPM 固定生成参考的 106.72 秒试听只证明该样本得到认可，不自动成为整章结论。

本次普通测试验证接口删除、固定参考模板、分段和参数/seed、内容门禁、取消和缓存行为，不下载权重或读取小说。真实模型与平台检查单独记录在[实施记录](records/voice-drift/official-reference-synthesis-2026-10-27.md)。此前滚动与双参考的依据及失败见[上游调研](records/voice-drift/rolling-context-research-2026-10-26.md)。
