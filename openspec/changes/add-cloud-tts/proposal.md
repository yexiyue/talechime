# Proposal

## Why

CastGlean 需要不依赖本地模型的云端合成、音色发现和参考克隆能力，同时保留对不同服务商的扩展空间。Fish Audio 提供公开音色、参考音频与远端音色模型，可作为首个真实 provider 验证抽象。

## What Changes

- 增加可独立依赖的 Rust 云 TTS crate，提供有界合成流和 Fish provider。
- 支持 Fish 公开及自有音色检索、查询、即时参考克隆和显式创建、查询持久克隆。
- 保留音色身份、请求参数、费用和失败的不确定性；显式选择模型，避免意外回退到付费模型。
- 本地参考音频由调用方提供，不改变本地音色库格式；提供对接指引。
- 不包含 CastGlean 集成、播放适配、CLI/协议修改、WebSocket、云端音色设计或其他 provider；这些作为后续候选。

## Capabilities

### New Capabilities
- `cloud-speech`: 独立云端合成、流生命周期与 Fish 请求安全边界。
- `cloud-voices`: Fish 音色发现及即时和持久参考克隆。

### Modified Capabilities

无。

## Impact

新增 workspace crate、HTTP 和 MessagePack 依赖及 Rust 公共 API；需要 mock 服务测试、独立依赖构建、文档和单独记录的真实 API 验证。现有本地音色和播放保持独立，宿主负责角色与项目绑定。
