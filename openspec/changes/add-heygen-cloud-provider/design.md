# Design

## Context

动机见 [proposal.md](proposal.md)。当前 `src/lib.rs` 仅导出 `fish`；Fish 的 client、speech、voices 为具名私有模块，依赖 workspace reqwest/serde/bytes；其流由调用方持有 HTTP Response，已有本地 TCP mock 覆盖截断、取消、drop 和敏感错误。Fish 类型均位于 provider 命名空间，暂无通用 provider trait。该实现作为行为参照，不要求把 HeyGen 参数塞进 Fish 类型。

资料核对于 2026-11-11：
- [TTS 指南](https://developers.heygen.com/docs/voices/heygen-voice-speech.md)：HTTP 流端点、输出模式、终态限制、模型及语言。
- [流 API](https://developers.heygen.com/reference/stream-speech)与[合成 API](https://developers.heygen.com/reference/generate-model-speech)：实施时再次核验具体输出组合与响应头。
- [音色指南](https://developers.heygen.com/docs/voices/overview.md)：公开/自有范围、过滤、游标与状态。
- 用户提供的 [Video Agent Sessions](https://developers.heygen.com/reference/list-video-agent-sessions) 属于视频会话，不是本次语音调用链。

这些是文档依据，不是 provider 的真实服务验收。此前仅 CLI 认证通过，不证明 Rust 合成、音色查询或音频质量。

## Goals / Non-Goals

**Goals:** 加入可独立使用的 Rust HTTP provider；让模型、音频格式和错误契约保持明确；复用现有工具链并保留 Fish 行为。

**Non-Goals:** 不建立插件注册中心、统一业务音色库、自动分段、重试调度或播放适配；范围边界以 proposal 为准。

## Decisions

### 1. Provider 模块及最小公共 API

新增 `src/heygen.rs`，其下 `client.rs`、`speech.rs`、`voices.rs`。公开 provider 自有 Client、ApiKey、Model、SpeechRequest、AudioFormat、AudioStream、StreamState、VoiceQuery、VoiceRecord/VoicePage、Error。名称仅为实施方向，最终 Rust 命名可按邻近代码调整，但不能改动规格行为。请求以远端 ID 指定音色；克隆音色只有在调用方已有 ID 时可引用。

继续使用已有 workspace HTTP、序列化和字节依赖，不通过 CLI 进程转发。拒绝现在抽取通用 trait 或移动 Fish 公共类型：两家身份、格式和分页参数不同，直接增加并列模块更容易保留兼容性。只在实施中确有相同行为且无需改公共 API 时考虑小型私有复用，不开展跨 provider 重构。

### 2. 固定两种原始字节输出，跳过 SSE

`POST /v3/models/audio/tts/stream` 显式设置 `output_format` 和 `with_timestamps=false`：PCM16 使用 24000 Hz，MP3 使用 44100 Hz/128000 bps，均为单声道。二者用枚举表达，避免开放任意格式组合。默认不设置 output_format 的 SSE 含独立 WAV，不可当连续文件直接拼接，首版不实现该协议分支。

收到响应头后校验 Content-Type（忽略合法参数）、`X-Audio-Sample-Rate` 和 `X-Audio-Channels`；缺失、非法、与请求不符时直接释放响应并失败。流公开请求格式及服务确认格式；`Content-Language` 与 `X-Request-Id` 可作为可选元数据保留，不将远端 header 无条件写日志。正文使用 Rust `chars().count()` 校验 1–5000 个标量值并拒绝纯空白；可选语言只接受官方支持的基础语言代码，不将 `zh-CN` 静默改写为 `zh`。首版不暴露可能被瞬时音色忽略的专业声线参数。

替代方案为先合成 URL 再下载，会增加二次请求、签名 URL 与下载认证边界；不如直接流契合现有库用途。

### 3. 响应生命周期与错误

拉取式 Response 无后台任务和额外队列。累计音频受 max_audio_bytes 限制；单块超出剩余预算时不交付，响应立即释放。取消/完成/失败集中关闭响应，后续读取保持终态。网络块可跨越采样边界，调用方负责解码/组帧。HTTP 库检测到截断返回 Transport，空输出或格式/容量错误为 InvalidResponse；已交付部分数据后失败不提供“完整音频”结果。

重要限制：原始字节流没有 SSE 的 `[DONE]`。HTTP 正常 EOF 不能证明模型说完正文，特别是服务端以正常关闭结束异常请求时。库只报告传输结束，不恢复对齐、不判断正文覆盖或实际播放完成。

密钥使用隐藏 Debug 的包装类型和 `X-Api-Key`，不读取环境或 `.env`；默认 HTTPS `api.heygen.com`，可控 HTTPS endpoint/loopback HTTP 用于测试，拒绝 userinfo/query/fragment，关闭重定向及自动重试。401/403/429 单独分类，其余已收到 HTTP 响应保留状态码（含 402/404/409）；可能计费请求发送失败或 5xx 保守映射 OutcomeUnknown，音频 body 读取错误则为 Transport。所有分类不携带正文或原始 reqwest 错误文本。只读目录 JSON 采用固定 4 MiB 上限；错误体不读取。

### 4. 音色目录

显式 public/private 范围，单页 1–50，游标由调用方保存并在相同条件下续传；不自动遍历全部页面。公开目录支持 language/gender/age/query，自有目录携带这些过滤时本地拒绝。详情 ID 经 path segment 编码，拒绝空白及点路径。保留 voice_id、name、mode、status、language、preview_audio_url 和未知字段；不把未知或 PENDING 状态转成 ACTIVE，也不因为音色查询成功承诺所选模型兼容。服务拒绝合成时明确报错，不自动换模型或音色。

### 5. 验证与文档

新增 HeyGen mock 套件，独立验证认证头、模型/格式请求、响应头、输入校验、错误、预算、慢消费者 drop 与目录游标。运行现有 Fish 测试确保兼容；示例凭据由环境显式传入并使用固定非私人短句。普通测试无外网、无模型；真实探针只在明确请求后执行，默认显式选择免费档，记录日期与当时政策，不能因为免费临时开放而自动切付费。

## Risks / Trade-offs

- [文档/实际响应头漂移] → 实施前核验 schema；缺失信息安全失败，必要时通过新的规格决定兼容策略，不猜格式。
- [免费档容量和期限变化] → 模型显式选择；文档当前优惠至 2026-11-30，不写成代码中的永久权限或可用性保证。
- [原始流正常结束但内容不完整] → 明确传输完成限制；试听、文本完整性与性能分别验收。
- [Windows 网络依赖代理] → 宿主提供 HTTPS_PROXY 等进程配置，不把系统代理地址或密钥写进库、规格和日志。
- [单次 HTTP chunk 内存由网络层决定] → 本库不额外无界累计；总交付上限不宣称约束 HTTP 库内部所有分配。

## Migration Plan

加法导出 heygen 模块并同步 crate README、根 README/架构的当前能力描述与知识索引。无需迁移配置、本地音色或用户目录，也不新增本地数据格式。撤销时移除新增模块/导出及文档即可，不删除任何远端或本地音色。原 Fish 规格仍保留于 add-cloud-tts，本变更不以归档它为前置条件。


## 2026-11-11 实施核验

流 API 的 output_format 明确包含 PCM 24000 与 MP3 44100/128000，原始响应头与指南一致。音色列表返回顶层 data 数组/has_more/next_token，详情以 data 对象包装；测试据此构造独立 HTTP 夹具。当前参考页列表上限为 100、流 text 为 10000，指南仍描述公开页 50/文本 5000；本库保留批准规格的 50/5000 子集，不提高公共限额。均为文档核验，不是在线服务验证。
