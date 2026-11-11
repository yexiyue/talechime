# Fish API 核对记录

核验日期：2026-11-11。依据 [OpenAPI](https://docs.fish.audio/api-reference/openapi.json)、[Raw API skill](https://docs.fish.audio/.well-known/agent-skills/fish-audio-api/SKILL.md) 和 [定价与限流](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits)。这是文档核验，不是真实服务验收。

- `POST /v1/tts` 的 `model` header 支持 `s1`、`s2-pro`、`s2.1-pro`、`s2.1-pro-free`、`drama-3`；省略或未知值回退付费 `s2.1-pro`。`s1` 即将退役，首版不暴露它。
- TTS 支持 JSON 和 MessagePack，内联参考的二进制使用 MessagePack；音频格式为 mp3、wav、pcm、opus。成功结果是流式音频字节，非 JSON 包装。
- `GET /model` 的 `page_size` 为 1–100、`page_number` 从 1 开始，过滤参数是 `title`、`tag`、`language`，自有范围为 `self=true`。`POST /model` 使用 multipart，`type=tts`、`train_mode=fast`、`title` 和 1–20 个 `voices`；`texts` 对应样本。以私有模式显式创建。
- OpenAPI 描述公开创建请求会降级为 private，而 skill 文档对默认 visibility、音频增强默认值的描述与克隆指南不一致；实现显式设置 private，不依赖未确认默认值，也不提供公开创建。
- Starter 文档并发上限为 5，按账号共享；免费模型价格为每百万 UTF-8 字节 0 美元，服务政策可能变化。限额和价格不应硬编码为授权或永久保证。
- 设计没有要求精确 PCM 采样编码或云端音色自动发布；这些细节须在实现前再次检查 schema，真实服务及试听仍未验证。
