# talechime-protocol

独立的 JSON Lines 协议 DTO、编码、请求校验、来源身份和 UTF-8 原文范围。此 crate 不依赖模型、音频设备或进程运行时。

## 协议 v6

每个请求包含 `protocol_version`、唯一 `request_id`、可为空的 `session_id` 和 `type`。带参数的命令使用 `payload`；公开 Rust DTO 是字段定义的来源。

```json
{"protocol_version":6,"request_id":"hello-1","session_id":null,"type":"hello"}
```

客户端先发送 `hello`，核对 Ready 中的模型与编译设备，再使用 get_status/get_config/update_config、prepare_model/cancel_prepare、start/pause/resume/stop/seek/shutdown。未知协议版本在握手阶段拒绝；v6 不兼容此前版本。

v6 删除了 `alignment_enabled`、`alignment_device`、`alignment_status`、`sentence_started` 和 `sentence_finished`。配置更新拒绝未知字段。当前进度以实际播放的片段为单位。

## 身份与完成语义

响应带 request_id；异步事件的 request_id 为空。所有消息带 instance_id 和递增 sequence，会话事件另带 session_id。客户端校验进程、会话与序号，忽略旧进程或旧会话的事件。

TextRange 是同一正文快照的 UTF-8 左闭右开字节范围，必须与 text_hash 一起校验。JSON 转义保留正文换行，不能把范围套用到另一次规范化后的文本。

`accepted` 表示命令已接受。`segment_started` / `segment_finished` 跟随实际播放，完成的片段才提交检查点；`session_ended` 的原因分为 completed/cancelled/failed。模型生成完成不能替代播放完成。

## 设备与缓冲

`device_status` 报告 TTS 的 compiled、available、selected 与原因；`tts_device` 可为 auto/cpu/coreml/cuda/metal，实际支持由所选模型决定。Auto 经过校准后选择设备，显式设备不可用时返回错误。

`session_state=buffering` 表示自动预缓冲或恢复，区别于用户暂停。`buffer_status` 携带原始音频余量 buffered_ms、按倍率换算的 target_ms 和 underruns；缓冲事件不提交检查点。

单行编码上限为 16 MiB（不含行尾）。读取方须在解码前限制缓冲大小。普通协议测试不下载模型；数值、听感和实时性由独立模型验收验证。
