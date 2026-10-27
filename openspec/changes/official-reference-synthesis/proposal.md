# Proposal

## Why

用户要求所有后端采用官方推荐的合成/固定参考方式，移除已出现音色漂移和朦胧感的自建滚动上下文及双参考策略。移除入口和实现，不携带兼容层。

## What Changes

- 删除宿主级滚动音频、双参考与身份锚定实验接口、能力和配置。
- Nano 使用官方固定 voice-clone 模板；Qwen Base 使用固定导入 prompt、CustomVoice 使用官方 speaker；VoxCPM 使用固定导入参考或自然生成；OmniVoice 使用固定 clone prompt/官方设计流程；MOSS Realtime 使用固定参考的独立请求；CosyVoice3、IndexTTS 维持官方固定参考流程。
- 保留单次生成内部的原生 KV/codec/streaming 状态、语义分段、取消、播放与校验。
- 清理缓存键、CLI、协议、开发工具、文档与测试。保留历史音频和用户音色库。

## Capabilities

### New Capabilities

- `reference-synthesis`: 官方固定参考合成及移除滚动入口。

## Impact

涉及 protocol/core/backends/model experimental APIs/CLI/xtask/docs。协议仍 v7；旧滚动字段作为未知字段拒绝，不迁移或删除用户资产。无发布或远程推送。
