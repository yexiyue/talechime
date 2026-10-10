# ASR 回读校验

回读校验是可选的内容门禁，默认关闭。库支持独立报告、合成仅报告和校验后交付三种用法；报告判断内容差异，不代表音质、韵律或播放完成。当前内置模型组是 **Qwen3-ASR 0.6B（主识别）+ SenseVoiceSmall INT8（复核）**，两者使用 CPU，按中文回读配置，显式准备后复用。库通过 `asr` feature 启用内置模型；自定义 `Recognizer` 不要求这个 feature。标准发行包包含内置组，但首次准备仍需下载约 1.82 GB 文件，权重不随包分发。

## 库调用

`prepare_readback` 只准备 ASR，`Engine::prepare` 只准备 TTS。准备调用可以下载固定 revision 的文件并验证大小、SHA-256；合成和报告调用不下载。宿主传入自己的资源目录，模型组留在同一个 LocalSet。

```rust,no_run
use talechime::*;
# async fn example(mut engine: Engine) -> Result<(), EngineError> {
let verifier = prepare_readback(ReadbackModelOptions::new("./models"), |_| {}).await?;
engine.set_verifier(verifier.clone())?; // Engine 空闲时设置，随后复用
let options = VerificationOptions {
    policy: VerificationPolicy::Gate { max_retries: 1, strict_suspect: false },
    ..Default::default()
};
let voice = engine.capabilities()?.default_voice;
let mut stream = engine.synthesize_verified("你好，世界。", &voice, None, options)?;
while let Some(item) = stream.next().await {
    match item? {
        SynthesisItem::Verification(report) => println!("{:?}", report.verdict),
        SynthesisItem::Audio(audio) => { /* 交给宿主音频接收端，再释放预算许可 */ }
    }
}
assert_eq!(stream.state(), SynthesisState::Completed);
engine.close().await?;
# Ok(())
# }
```

仅要 PCM 时仍可用 `recv()`，最近一份报告通过 `last_report()` 获取；要完整的每次尝试证据，用 `next()` 消费报告与 PCM 的有序流。监听计划使用 `PlanSessionOptions.verification`，报告从 `Event::Verification` 投递。旧 `synthesize` 方法仍默认关闭。仅记录采用 `VerificationPolicy::ReportOnly`，不自动重试。

已经有音频的宿主直接调用 `verifier.report(ReadbackRequest { source, range, spoken_text, backend, model, voice, style, attempt }, &pcm, &options).await`。这是报告接口，不需要 Engine、TTS、播放器或检查点；`options.policy` 不触发重合成。原文快照必须匹配摘要，`range` 是 UTF-8 字节范围，`spoken_text` 是该范围实际送入 TTS 的预处理文本。ASR 只接收音频，不接收正文或热词提示。超时/执行失败写入 `Unverified` 证据；无效输入、容量超限直接返回错误。超时后需要确认原生任务退出时可调用 `verifier.settled().await`，只等待该 Verifier 的报告请求。

## 交付与重试

主识别归一化后相同即通过；有差异才启动不同家族的复核。归一化覆盖大小写、全半角、标点、繁简及明确的数字读法；保留小数、负号、百分比区别，不猜测“一百二”等简称。同音字和数字歧义保留为疑点，两个模型一致的替换也不自动视为 TTS 错误。共同支持的缺失或原文短语重复才确认为内容异常。

| 判定 | 仅报告 | 门禁 |
| --- | --- | --- |
| `passed` | 交付 | 交付 |
| `suspect` | 交付并记录 | 默认交付并记录；`strict_suspect=true` 阻止 |
| `confirmed_error` | 交付并记录 | 保持正文、音色、风格重合成同一实际片段，耗尽后失败 |
| `unverified` | 交付并记录 | 失败，不冒充通过 |

开启时先收齐一个实际模型片段，再回读并交付。流式计划仍逐片段前进，但增加片段收集与回读延迟；默认关闭时保持原有 TTS 流式行为。重试只替换尚未发布的该片段 PCM，不修改计划、不切换 TTS、不撤回已交付音频。`AfterChapterReady` 全章暂存且通过才播放，后续片段失败时整章不播放。生成完成始终与播放完成分开。

每个片段最多 512 字符、30 秒、16 MiB float PCM；可降低限额。每个 ASR 超时默认 30 秒，可设为 1–300000 ms；重试 0–3 次。输入、报告、控制及 PCM 通道有界，消费报告同样施加背压。显式 `cancel/close` 等待本地生产者及原生在途任务退出；Drop 请求取消，不代表异步清理完成。ASR 原生对象由各自线程构造、调用、销毁。每次执行单独跟踪原生请求的完成回执，共享模型和报告缓存也不会让取消等待其他执行；校验关闭时不参与 ASR 等待。自定义原生识别器通过 `Recognizer::request` 提供 `RecognitionRequest.completion`，在请求真正清理结束后置 true 或关闭发送端；纯异步识别器只实现 `transcribe` 即可。

报告绑定原文摘要、字节范围、实际朗读文本、TTS/音色/风格、实际 PCM 哈希、ASR revision/实现及规则版本。缓存仅保留报告，默认 32 项，可设置 0–256；音频不缓存，超时/执行失败不缓存。同一音频重复生成可命中证据，`attempt` 仍独立记录。繁简、数字或预处理改变映射时，差异范围保守回落到整个实际原文片段，不伪造精确字节位置；报告还保留归一化朗读文本中的字节位置用于复核，避免将不同位置的同词缺失混为一致。重试单位始终是实际模型片段。

## CLI 与 JSON Lines

```sh
# 原有合成播放，校验开关与报告输出均显式指定
talechime chapter.txt --verify gate --verification-report attempts.jsonl
# 只回读已有 PCM16 WAV；不启动 TTS 或播放
talechime --model-dir ./models verify segment.txt segment.wav --report report.json
# v7 worker 在 prepare_model 阶段同时准备 ASR
talechime --protocol --readback-models
```

`--verify report` 仅报告；`gate` 默认一次重试、疑点可交付。`verify` 子命令输出单份 JSON 报告，内容异常本身不使报告命令失败。CLI 只有显式路径才落盘证据，默认不保存音频或报告。

v7 `PlanRequest` 可添加以下 `verification` 字段；省略即关闭，仍只有计划式 start 入口。worker 需要 `--readback-models`，未准备时拒绝启用校验的计划；没有兼容协商或自动下载。事件 `verification` 携带同一报告 DTO。

```json
{
  "verification": {
    "policy": { "mode": "gate", "max_retries": 1, "strict_suspect": false },
    "timeout_ms": 30000,
    "max_segment_ms": 30000,
    "max_segment_bytes": 16777216
  }
}
```

## 验证与候选

选型和原生运行证据见[主线验收记录](records/asr-readback-mainline-2026-10-09.md)，候选对照见[选型 spike](records/asr-spike-2026-10-09.md)。Whisper large-v3-turbo、Qwen 1.7B、FireRed 保留实验候选，未作为内置可切换适配器交付。宿主可以用其他 `Recognizer` 组合，只要求明确模型身份与不同家族复核。小样本不证明自然漏读召回率或人工听感。


## 语言自动检测与音频标签

Qwen3-ASR 主识别使用模型自身的语言自动检测（此前强制 Chinese，与生成侧开放的
language 参数不兼容）；`max_new_tokens` 恢复为 512 以覆盖较长非中文段。SenseVoice
复核在解码时捕获 `<|…|>` 情感/事件/语言标签，作为 `ReadbackEvidence.labels`
（v7 加法式字段）进入报告与 `asr_readback` 探针输出；文本比对仍不使用这些标签。
这为逐段风格标注（读取侧情感标签反哺生成侧 style 指令）提供数据源，尚未接入
任何自动策略。
