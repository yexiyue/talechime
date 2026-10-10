# talechime-protocol

独立的 JSON Lines DTO、编码、请求校验、来源身份和 UTF-8 原文范围。不依赖模型、音频设备或进程运行时。

## 协议 v7

v7 采用一个计划执行入口，不提供旧版本适配或扩展协商。单音色就是一个完整声音范围并封闭的计划。宿主集成留待 Talechime API 和 CastGlean 分别稳定后进行。

每个请求包含 `protocol_version`、唯一 `request_id`、可为空的 `session_id` 和 `type`。带参数的命令使用 `payload`，公开 Rust DTO 定义字段。

```json
{"protocol_version":7,"request_id":"hello-1","session_id":null,"type":"hello"}
```

先 `hello` 查询模型目录，然后 `prepare_model` 等待 `model_ready`。模型身份使用所选能力中的 backend/model；计划身份精确匹配，`null` 不代表任意模型。查询与配置不准备模型，显式准备可能下载权重。

| 命令 | payload / 作用 |
| --- | --- |
| `start` | `PlanRequest`：完整正文、摘要、来源、backend/model、固定 voices、初始 spans、sealed、playback、resume_byte、restore_checkpoint |
| `append` | `{ "spans": [...] }`：从接受终点连续追加，整批原子接受 |
| `seal` | 无：要求覆盖全文，输入封闭不等于完成播放 |
| `fail_input` | `{ "message": "分析失败" }`：失败终止，不能以 seal 替代 |
| `get_progress` | 无：返回 `progress` 快照，含输入状态和三个连续终点 |
| `pause` / `resume` / `stop` | 无：实际播放控制；stop 产生 cancelled |
| `seek` | `{ "byte": 0, "new_session_id": "新的唯一ID" }`：保留计划/暂停状态，从原文重新生成 |
| `get_status` / `get_config` | 无：查询状态或持久默认配置 |
| `update_config` | `ConfigPatch`：音色/风格只作为未来计划默认值；音量/速度影响活动播放；模型/设备变更取消执行并释放模型 |
| `cancel_prepare` / `shutdown` | 无：取消准备或关闭 worker |

所有计划/播放命令要求 envelope 中非空 session_id。start 使用新 ID；后续命令使用活动 ID。start 全部验证后才替换旧执行。坏范围/摘要/音色、提前或重复 seal、旧 ID 都明确报错，非法追加不会使已有计划失败。

VoiceSpan 为 `{ "range": { "start": 0, "end": 3 }, "voice": "声音ID", "style": null }`，不含角色 ID。`playback` 为 `streaming` 或 `after_chapter_ready`。一次完整提交设置 sealed=true，增量提交设置 sealed=false，再追加与 seal；空章可直接 sealed=true。

`progress` 含 `input_state`（open/sealed/failed）、accepted_end、generated_end、played_end、waiting_for_input、chapter_ready。只在查询时返回，不需要高频事件推送。整章模式只有封闭且全部音频生成、暂存校验成功后 chapter_ready=true；生成期间 played_end 不推进。

## 身份、背压与完成

响应带 request_id；异步事件的 request_id 为空。消息带 instance_id 和严格递增 sequence，会话事件带 session_id。宿主须核对身份、序号、正文摘要，旧执行事件不能推进新执行。

TextRange 是同一份完整正文的 UTF-8 左闭右开字节范围。正文不再次规范化。`accepted` 只确认输入或控制操作；实际播放发布 segment_started / segment_finished。session_ended 的 completed/cancelled/failed 含义不同，只有有效生成与实际播放耗尽才 completed。

事件和 stdout 队列有界，宿主应持续消费 stdout 并排空 stderr。等待控制命令时 worker 继续排空会话事件，避免生命周期消息堵塞；慢宿主仍会施加背压。断开/关闭会取消并等待拥有的执行，不伪造 completed。

## 设备、容量与暂存

设备选择为 auto/cpu/coreml/cuda/metal，支持情况由实际模型决定。Auto 沿用 worker 的校准流程；显式设备不可用明确报错。音频预缓冲与用户暂停分开，buffer_status 不推进检查点。对齐已移除。

单行编码上限 16 MiB（不含行尾），解码前限制缓冲。每次 append 最多 4096 个范围；每执行最多 65536 个范围；声音集合最多 256 个。暂存采用 core 默认执行私有目录和 1 GiB / 一百万记录限额，失败不播放部分章节。暂存不是长期缓存，seek 会重新生成。

普通测试不下载模型。数值、听感和平台实时性由独立真实模型验收证明。

`PlanRequest.continuation` 默认 `true`；设为 `false` 可关闭本执行的段落内前段文字/音频条件。
`Capabilities.continuation` 表示所选模型是否实现这一能力，缺省为 `false`。协议仍为 v7。
参考不跨执行、seek、恢复或章节保存；具体边界与限额见[库说明](../../docs/library.md#段落内接续)。

`Capabilities.parameters` 是所选模型声明的生成参数目录（名称、类型/范围、默认值与描述）；
`PlanRequest.params` 按名称携带本次执行的设定值，`PlanRequest.seed` 可选地钉住采样以便
复现。未在目录中声明、类型不符或越界的参数在会话启动时被显式拒绝，不会静默忽略。
两者均为 v7 内的加法式可选字段。当前目录：VoxCPM2（steps/cfg/temperature/max_duration）、
Qwen（采样参数/language/max_duration/chunk_frames）、OmniVoice（num_step/guidance_scale/
language/speed）、MOSS Local/Realtime（instruction/max_duration）；Nano 与 ONNX 目录为空，
仅消费 seed。详见[参数接入记录](../../docs/records/params-exposure-2026-10-10.md)。
