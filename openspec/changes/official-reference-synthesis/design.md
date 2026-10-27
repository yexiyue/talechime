# Design

## Contract

每个合成片段的条件仅来自用户选定的固定音色和该片段文字/风格/参数。生成 PCM 只走播放、导出、回读，不成为下一请求的音色或语气参考。SegmentRequest 不携带 SpeechContext；删除上一段 PCM 收集与重锚预算。无 rolling/dual/reference-anchor feature、CLI 或 wire 字段。

## Model policy

Nano 固定已有 reference codes，使用官方 voice_clone user-reference 模板；Qwen Base 缓存原始 voice prompt；VoxCPM 保留官方单参考（录音+准确转写）及自然音色，无分离双参考；OmniVoice 保留官方固定克隆与音色设计；MOSS Realtime 固定用户 reference codes，无宿主跨请求历史；CosyVoice3 和 IndexTTS 原有固定身份路径不变。不引入新权重、speaker-only 或未合入的 long-form PR。

## Boundaries

保留原生单请求 transformer/codec 历史及取消重置。保持原文范围、语义组段、内容门禁、响度/边界处理及实际播放进度职责。移除字段后 PlanRequest 继续 deny_unknown_fields；旧入口明确失败。缓存 identity 增加固定参考策略版本，避免复用旧滚动生成的章节。

## Evidence

依据 docs/records/voice-drift/rolling-context-research-2026-10-26.md 中官方源码与本轮重新核验的 Nano infer.py、VoxCPM core.py 和 Qwen API。官方提供低层 continuation 并不要求本产品实现滚动回灌。已被用户否定的双参考和缩短链路不再保留为实验候选。

## Validation

协议旧字段拒绝；公开请求不再传递前段音频；单/多音色、内容门禁、取消、章节缓存和 fixture 性能工具回归；默认及扩展 CPU 和可用 Metal feature 的测试、Clippy、rustdoc。实际资源就绪则生成固定参考短样，缺失设备不声称验证。历史实验保留，开发执行入口退役，旧任务不伪造完成。
