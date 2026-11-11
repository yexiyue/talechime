# Proposal

## Why

`talechime-cloud-tts` 当前只有 Fish provider。HeyGen 已提供独立的 v3 语音模型、音色目录与 HTTP 音频流，适合让 Rust 宿主在不引入数字人、视频和本地推理依赖的前提下选择第二个云服务。

## What Changes

- 新增与 `fish` 并列的 `heygen` Rust 模块，直接使用 HTTPS API 与调用方提供的凭据。
- 显式选择 HeyGen Voice、Turbo 或 Turbo Free；提供 PCM16 与 MP3 HTTP 流式合成，不自动切换模型或重试可能计费的请求。
- 支持公开/自有音色的游标分页、公开目录过滤及详情查询，保留状态和服务元数据。
- 返回有界、拉取式音频流及格式信息，区分失败、取消和传输完成；保留请求结果与费用的不确定性。
- 提供无凭据的 mock 测试与 Rust 示例，明确真实服务、试听和性能验证的不同边界。
- 本轮不包含音色创建/克隆/设计、整段 URL 下载接口、默认 WAV/SSE、WebSocket、时间戳对齐、视频、CLI/协议或宿主集成；也不引入通用 provider trait。

## Capabilities

### New Capabilities

- `heygen-cloud-speech`: HeyGen 的独立音色发现、显式模型合成及有界 HTTP 音频流生命周期。

### Modified Capabilities

无。当前 `openspec/specs/` 无主规格；已实现 Fish 的 `cloud-speech` 与 `cloud-voices` 仍位于 `add-cloud-tts` 变更中，本提案不修改或归档该变更。

## Impact

涉及 `crates/talechime-cloud-tts/src/heygen.rs` 及其具名子模块、`src/lib.rs`、测试、示例与当前能力文档。优先复用 workspace 中已有 reqwest/serde/bytes 依赖；不依赖 HeyGen CLI、外部技能、本地模型、播放器或其他项目的业务类型。保留现有 `fish` 公共 API 和行为。凭据由宿主注入，不读取或提交 `.env`。
