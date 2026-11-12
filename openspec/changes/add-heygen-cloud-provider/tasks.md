# Tasks

## 1. Provider 与安全请求边界

- [x] 1.1 对照设计链接的当前 HeyGen schema 核验 PCM 24000、MP3 44100/128000、响应头和音色 JSON 字段，将核验日期与来源补入设计；以请求/响应夹具固定这些字段，不访问真实服务作为普通测试。
- [x] 1.2 新增 heygen 具名模块、显式 Model、隐藏 Debug 的 ApiKey 与错误类型并导出；以单测验证三种模型、未知/缺失模型与凭据校验，cargo check 和依赖树确认无模型/播放器/CLI 依赖。
- [x] 1.3 实现 X-Api-Key、HTTPS/loopback endpoint、不跟随重定向及不自动重试；mock 验证 401/403/402/404/409/429/5xx、断连结果未知、敏感错误不泄漏和没有第二次请求。
- [x] 1.4 为公开入口写 rustdoc，说明凭据注入、代理由宿主配置、结果未知与付费边界；以 rustdoc 和错误/Debug 脱敏测试验证。

## 2. 音色查询

- [x] 2.1 实现公开/自有目录、单页 1–50 与游标及公开过滤校验；mock 验证下一页保留过滤条件、自有范围不支持过滤、非法分页/过滤在网络前失败。
- [x] 2.2 实现详情 ID 的 path segment 编码、有界 JSON 读取和元数据保留；mock 验证含特殊字符 ID、未知/PENDING/FAILED 状态、扩展字段、超大及损坏 JSON、无权限/不存在不换音色。
- [x] 2.3 增加 HeyGen 音色查询 Rust 示例与 crate README 使用说明；以本地 mock 检验示例路径，记录远端 ID、语言、状态与模型兼容性边界，不读写本地音色库。

## 3. HTTP 原始音频流

- [x] 3.1 实现显式 PCM16/MP3 请求、1–5000 Unicode 标量值文本、非空白 ID、可选基础语言和非零预算校验，固定 with_timestamps=false；mock 检查请求体、中文边界、非法输入无请求及模型不回退。
- [x] 3.2 实现响应 Content-Type/采样率/声道检查与请求/实际格式元数据；mock 验证 PCM16/MP3 正常头、缺失/冲突头及 200 SSE/JSON 被拒绝。
- [x] 3.3 实现累计有界拉取、Completed/Failed/Cancelled 终态和 cancel/drop 释放；mock 覆盖空流、预算恰好/超限、截断 Transport、慢消费者、连接关闭和终态重复读取，无后台无界队列。
- [x] 3.4 增加固定非私人中文短句合成示例及格式说明，mock 验证示例写出的字节；同步 crate README、根 README 与架构当前能力，说明原始流 EOF、采样组帧与实际播放边界。

## 4. 集成验证与证据

- [x] 4.1 运行 cargo fmt --all --check、cargo test --locked -p talechime-cloud-tts --all-targets、cargo clippy --locked -p talechime-cloud-tts --all-targets -- -D warnings 及警告视为错误的 rustdoc；确认 Fish 现有用例通过且所有普通测试无凭据/无外网。
- [x] 4.2 交付独立验收记录并更新知识库索引，明确普通测试、CLI 认证与 Rust 实际合成不是同一证据；如未得到真实测试请求，明确标记 API 合成、解码、试听、延迟及费用未验证，不能以 mock 勾选为真实验收通过。
- [x] 4.3 检查公开文档、示例、任务和规格一致，运行 openspec validate add-heygen-cloud-provider --strict 与 git diff --check；确认只增加 HeyGen provider、未改变 Fish API、未提交凭据/音频或本地运行产物。

## Workflow follow-up

- 真实合成需用户另行明确请求；执行时优先显式免费档、公开音色和固定短句，记录当日政策与验证范围，不创建音色、不自动切付费。人工试听与性能基准分别记录。
- 实现完成并通过审查后，再按用户请求执行归档或提交推送。
