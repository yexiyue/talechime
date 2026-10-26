# JSON Lines v7 worker 接入流程

目标是：宿主启动一个本地 worker，把文本计划交给它，控制播放并等到真实播放完成。
worker 不向宿主输出 PCM。完整顺序为：**启动 → 握手 → 选模型/声音 → 配置 → 准备 →
提交计划 → 消费事件/控制 → 关闭**。

## 1. 启动进程，同时建立两条输出读取任务

```sh
talechime --protocol --config ./speech-data/config.json --model-dir ./speech-data/resources --checkpoint-dir ./speech-data/checkpoints
```

路径由宿主选择；本例音色库为 ./speech-data/voices/。启动后持续读取 stdout 的 JSON
消息，同时排空 stderr 日志。不要等“发请求”之后才读输出，也不要假设下一行一定是
当前请求的响应，异步事件会穿插其中。

**本步结果：** 一个仍在运行的进程，以及持续工作的 stdout/stderr 消费任务。

## 2. 五秒内握手，拿到本次构建的模型目录

```json
{"protocol_version":7,"request_id":"hello-1","session_id":null,"type":"hello"}
```

等待 ready。其 payload 本身就是模型能力数组，不是 `{"models": [...]}`。
空数组表示无合成后端，仍可查询配置和管理资产，但不能开始朗读。

**本步结果：** 知道这次能选择哪些准确的 backend/model、内建声音、设备和参数。

## 3. 选择模型和声音

从 ready 中选择模型。声音有两种来源：直接选该模型的内建 voice，或使用库音色。
库音色通过 list_voices 查询；如果需要设计新声音，先发送 design_voice：

```json
{"protocol_version":7,"request_id":"design-1","session_id":null,"type":"design_voice","payload":{"name":"朗读者","description":"低沉温和的男声","id":"reader"}}
```

name 在协议中必填。设计不要求先 prepare_model；它会准备设计引擎自身资源，可能下载
并执行推理。等待 voice_ready，取 payload.voice 作为后续声音 id；accepted 不能代替
设计完成。使用库音色时，合成模型必须支持 cloning。

**本步结果：** 选定准确的 backend/model、可用设备，以及一个可用于本次计划的声音 id。
下面用 Nano 的 Weiguo 内建声音演示；实际接入要替换成当前目录和用户选择。

## 4. 读取配置 revision，再保存本次选择

先发 get_config，等待 config，读取 payload.revision。然后发送 update_config：

```json
{"protocol_version":7,"request_id":"config-1","session_id":null,"type":"update_config","payload":{"expected_revision":0,"backend":"moss","model":"nano-candle","voice":"Weiguo","tts_device":"cpu"}}
```

例中的 expected_revision=0 仅适用于刚读到 revision 为 0 的配置。正常代码始终使用
读取结果，并等待 config_changed；遇到 revision_conflict，重新读取并按宿主当前意图
重建 patch，不能循环强行覆盖。

**本步结果：** worker 的合成选择与宿主打算提交的计划一致。

## 5. 准备模型，等到可执行

```json
{"protocol_version":7,"request_id":"prepare-1","session_id":null,"type":"prepare_model"}
```

持续接收 model_progress/resource_state/device_status，直到 model_ready；错误则停止
本次启动流程并交给宿主处理。这一步可能下载模型。设计、模型准备和播放互斥，资源
命令有界排队，不要因为等待时间长而重复提交已接受的请求。

**本步结果：** 所选模型准备成功，可以提交播放计划。

## 6. 提交一份完整计划

先固定原文快照，对其 UTF-8 字节计算 SHA-256 小写十六进制。声音范围也使用同一份
快照的 UTF-8 字节偏移。例如「你好」占 6 字节，完整单音色计划如下：

```json
{
  "protocol_version": 7,
  "request_id": "start-1",
  "session_id": "chapter-1",
  "type": "start",
  "payload": {
    "source": {
      "namespace": "my-app",
      "book": "book-1",
      "chapter": "chapter-1"
    },
    "text": "你好",
    "text_hash": "670d9743542cae3ea7ebe36af56bd53648b0a1126162e78d81a32934a711302e",
    "backend": "moss",
    "model": "nano-candle",
    "voices": [
      "Weiguo"
    ],
    "spans": [
      {
        "range": {
          "start": 0,
          "end": 6
        },
        "voice": "Weiguo",
        "style": null
      }
    ],
    "sealed": true,
    "playback": "streaming",
    "resume_byte": 0,
    "restore_checkpoint": false
  }
}
```

backend/model 必须与已准备模型的准确身份一致，model=null 不表示任意模型。
request_id 每次唯一；start 使用全新的 session_id，后续控制使用该执行 id。
resume_byte=0、restore_checkpoint=false 表示明确从头。

**本步结果：** 收到 accepted，说明输入已接收；尚不能认为文本已经生成或播完。

增量接入时改用 sealed=false，再发 append 扩展连续前缀，覆盖全文后 seal。分析失败
发 fail_input。不要靠提前 seal 或断流冒充正常输入结束。初次接入先用上面的完整计划，
不必同时实现增量协议。

## 7. 持续消费事件，按需控制播放

start 之后继续读取消息。响应按 request_id 关联；异步事件 request_id 为空。
按 instance_id、严格递增 sequence、session_id 区分实例和执行，过滤旧执行事件。

| 宿主要做什么 | 发送什么 | 注意什么 |
| --- | --- | --- |
| 暂停/继续 | pause / resume | 使用活动 session_id |
| 查询进度 | get_progress | 分清 accepted_end、generated_end、played_end |
| 停止本次朗读 | stop | 结束原因是 cancelled，不是 completed |
| 跳转 | seek，payload 含 byte 和 new_session_id | 后续控制改用新 id |
| 取消设计/准备 | cancel_prepare | 若完成赢得竞态，仍可能返回 voice_ready |

等待 session_ended 并检查 payload.reason。只有 completed 才说明真正播完，允许自动
下一章。accepted、chapter_ready、generated_end 和 EOF 都不能替代这个判定。

## 8. 关闭进程

发送 shutdown，持续读取剩余输出到 EOF，然后等待子进程退出。关闭、取消与失败有
各自状态，不替它们生成 Completed。已成功设计的音色独立保存在库中；只有用户要求
删除时才发 remove_voice，payload 为 `{"voice":"reader"}`。

## 传输边界

每行一个 UTF-8 JSON 对象，单帧最多 16 MiB（不含换行）；声音集合最多 256；每批
append 最多 4096 spans，每执行最多 65536。长文本仍要在该上限内提供完整正文快照。
JS 端不能用字符串 length 代替 UTF-8 字节长度；若宿主规范化正文，应在计算 hash、
分析、标注和提交之前统一完成，不能让这些阶段各用不同的文本。
