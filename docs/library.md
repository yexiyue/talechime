# Rust 库入口

`talechime` 包现在同时提供 library 和原有 executable。应用可只调用库，不需要解析 CLI 配置、发送 JSON Lines 或手动装配 Registry/SessionManager。首期一个 Engine 固定一个已准备的 backend/model/device，同时只拥有一个执行；多音色限同模型内。

## 准备和直接合成

在现有 workspace 外通过 path 依赖试用，模型 feature 沿用现有名称：

```toml
[dependencies]
talechime = { version = "0.5.0", default-features = false, features = ["moss-nano-candle"] }
tokio = { version = "1", features = ["rt", "macros", "time"] }
```

JSON Lines 宿主只需 `talechime-protocol = "0.3.1"`（协议 v7），模型和音频继续留在独立进程。
计算库的发布包名为 `talechime-qwen3-tts` 和 `talechime-qwen3-asr`，Rust library 名仍为
`qwen3_tts` / `qwen3_asr`；工作区使用 package 别名保留现有导入。

```rust,no_run
use talechime::{Engine, ModelOptions, SynthesisState, run_local};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_local(async {
        let options = ModelOptions::new("moss", "./my-app-models");
        let mut engine = Engine::prepare(options, |event| eprintln!("{event:?}")).await?;
        let voice = engine.capabilities()?.default_voice;
        let mut stream = engine.synthesize("你好，世界。", &voice, None)?;
        while let Some(block) = stream.recv().await {
            let block = block?;
            // Feed a host-owned sink here, then release the block's budget permit.
            println!("{} samples for {:?}", block.pcm().samples.len(), block.range());
        }
        assert_eq!(stream.state(), SynthesisState::Completed);
        engine.close().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    }).await
}
```

准备是显式操作，可能下载指定模型；资源父路径由宿主传入，不读取或写入全局 CLI 配置。默认 CPU，可显式选择 Registry 报告可用的设备；`Auto` 需由宿主先解析，装配层按选择调用既有模型准备流程。实际模型身份由准备后的 capabilities 返回。直接合成不打开音频设备、不写检查点或暂存文件；仍使用同一声音校验、模型分段、边界静音与原文范围规则。

`SpeechAudio` 持有 PCM 和预算许可，提供只读 `pcm()`、`range()`。保留块会施加背压，释放后归还预算；队列以 30 秒/16 MiB 限制，原始输入块在边界处理前也检查同一限制。宿主自行复制/缓存 PCM 的内存由宿主管理。简单收集使用 `engine.synthesize_pcm(text, voice, style, max_bytes)`；限制必须非零，超限或跨片段格式改变返回错误并取消执行，不返回截断音频。空白/空文本可以正常生成完成，但收集辅助方法报 `EmptyAudio`。

有效 PCM 前缀之后仍可能合成失败。`recv()` 返回错误后保持 Failed；None 也可能来自取消，不能仅靠流耗尽认定成功，需检查 `SynthesisState::Completed`。生成完成不表示播放完成。

## 可选回读校验

显式 `prepare_readback` 后用 `engine.set_verifier` 装配一次，按次 `synthesize_verified(..., VerificationOptions)` 或设置 `PlanSessionOptions.verification`；默认关闭。`Verifier::report` 支持只有已有音频的报告场景，不需要 TTS 或播放。完整策略、事件、取消与示例见[回读校验](readback.md)。

## 构造计划

```rust,no_run
# use talechime::*;
# fn plan(engine: &Engine, source: SourceSnapshot) -> Result<SpeechPlan, EngineError> {
let mut plan = engine.plan(source, vec!["A".into(), "B".into()], PlaybackPolicy::Streaming)?;
plan.append(vec![SpeechSpan::new(TextRange { start: 0, end: 3 }, "A", None)])
    .map_err(SessionError::from)?;
// 可直接启动开放计划，随后通过控制句柄追加和封闭。
# Ok(plan)
# }
```

`Engine::plan` 自动绑定本 Engine 已准备的准确模型身份和能力，宿主只提供正文快照、允许的音色和策略。单音色直接使用 `single_voice_plan(source, voice, style, policy)`，得到已封闭的完整计划；空正文也会检查音色和风格。完整多音色计划仍由同一路径 append 全部范围再 seal，不维护第二套执行逻辑。

## 播放和计划控制

```rust,no_run
# use talechime::*;
# async fn attach(engine: &Engine, chapter: SpeechPlan) -> Result<(), EngineError> {
let mut listening = engine.listen(ListeningOptions::default())?;
let control = listening.control();
control.start("chapter-1", chapter, PlanSessionOptions::default()).await?;
while let Some(event) = listening.recv().await {
    if let Event::SessionEnded { reason, .. } = event.event {
        println!("{reason:?}");
        break;
    }
}
listening.close().await?;
# Ok(())
# }
```

`listen` 才打开默认音频设备；高级宿主通过 `listen_with` 注入 Playback。宿主已有有界事件队列时，用 `listen_to(options, events)`；同时注入播放器则用 `listen_with_events(player, options, events)`。这两个入口返回 `ListeningSession`，会话直接投递到宿主队列，没有转发任务或第二个事件队列。宿主掌握队列容量（最多 4096），`ListeningOptions::event_capacity` 只控制库自建队列。`ListeningOptions::checkpoints` 默认 None，不建立磁盘检查点；恢复检查点必须提供路径，否则在替换旧执行前报错。显式目录仍采用既有检查点格式和原文校验。

`ListeningHandle` 有界控制队列最多 16 项，每个操作返回确认或错误。`start / append / seal / fail_input / pause / resume / seek / configure_playback / progress / status / stop` 直接复用 SessionManager。旧 ID 操作不能影响新执行；stop 明确取消当前执行、保留监听装配，close 则结束整个 Listening 所有权。原有多音色、增量输入和 AfterChapterReady 暂存规则见[架构说明](architecture.md)。

事件队列容量为 1..=4096，默认 64；宿主须在等待控制时并行消费事件。事件保持原有可靠有界背压，慢观察者仍会影响执行；高频观察快照/协议调度属于后续阶段，不宣称已解决。`Listening::close` 独立于事件消费：关闭投递、丢弃尚未消费事件、取消并等待生成/读回和在途 I/O，之后释放 Engine 预约；不会把行政关闭当 Completed。丢弃 Listening 会请求相同清理，但 Drop 不代表异步清理已经完成。`ListeningSession::close` 同样绕过观察者压力并等待清理，已经投递的事件由宿主保留和消费。

## 执行环境和关闭

```mermaid
flowchart LR
    H[宿主线程与 Tokio runtime] --> L[run_local / 已有 LocalSet]
    L --> E[Engine / 已准备 Backend]
    E --> S[直接 PCM 流]
    E --> A[Listening 本地 owner]
    C[其他任务或线程] -->|有界控制及确认| A
    A --> P[可选播放器 / 检查点 / 暂存]
```

库不创建会话后台线程或全局 runtime；`run_local` 在宿主已有 runtime 内建立 LocalSet，高级宿主可以使用自己的 LocalSet。Engine、PcmStream 和 Listening 保留 Rc 所有权，不能跨线程移动。`ListeningHandle`、`CancellationHandle` 为 Send + Sync，控制不会移动模型和音频对象。控制方法需要所属 LocalSet 保持运行。

直接流可以克隆 `cancellation()` 请求跨线程取消；这只是请求，流的 `cancel().await` 才释放接收端/排队 PCM 并等待生产者清理。`CancellationHandle::closed` 等待本地生产者释放资源，不能推断宿主所有 PCM 消费已完成。丢弃直接流后 Engine 保持 Busy，直到取消生产者完成；Engine::close 会等待这类已丢弃流。尚存活的流或 Listening 必须先显式关闭，Engine::close 返回 Busy，不偷偷结束宿主拥有的执行。

准备好的 Engine 可以在执行结束后复用模型；Engine::close 释放模型所有权。原生适配器的 owner 负责关闭请求通道和等待线程退出；MOSS Nano 本阶段补齐了与其他生产适配器一致的线程持有/初始化取消顺序。自定义 Backend 或 Playback 的额外所有者与异步资源仍由提供方管理，外部 Rc 不会被强行销毁。

## 无模型示例和边界

```sh
cargo run --locked -p talechime --example synthesize
cargo run --locked -p talechime --example listen_text
cargo run --locked -p talechime --example listen_plan
cargo run --locked -p talechime --example host_thread
```

上述示例用确定性 Backend/Playback，验证直接合成、单音色、多音色和宿主自建线程，不下载模型、不打开真实音频设备、不写用户目录。`prepare_model` 是明确会准备/可能下载模型的示例，普通测试只编译，不运行；使用 `cargo run --locked -p talechime --example prepare_model -- BACKEND RESOURCE_DIR [MODEL]` 手动执行。

库沿用现有模型 feature；`default-features = false` 可仅启用所需适配器或注入自定义 backend。`cli` feature 控制 executable 与 clap/crossterm 等 CLI 依赖；嵌入库设置 default-features=false，再选择所需模型，或直接注入自定义 Backend。播放器依赖仍参与 core 编译，运行时可不装配。CLI/worker 现在复用这个库入口，协议 v7 只有计划 start 路径，无旧版本协商层。音色资源版本/租约保护、真实模型听感和平台性能不能由本阶段夹具验证推断。

首期交付顺序是 Talechime API 与确定性验收，再等待 CastGlean 标注契约完成，最后共同接入 TRNovel；当前不把阅读器兼容作为库接口约束。

## 生成参数、seed 与原生语速

`PlanSessionOptions` 与 `SynthesisOptions` 携带 `params: GenerationParams` 与
`seed: SeedPolicy`。参数按准备后模型的能力目录（`Capabilities.parameters`，每项
`ParameterSpec` 含名称/类型/范围/默认/描述）在会话启动时校验：未声明、类型不符或
越界均显式拒绝，不静默忽略。协议侧 `PlanRequest.params` / `PlanRequest.seed` 是同一
数据的 v7 加法式字段。

```rust,ignore
let mut params = talechime::GenerationParams::new();
params.insert("steps", talechime_protocol::ParamValue::Int(32));
let options = talechime::PlanSessionOptions {
    params,
    seed: talechime::SeedPolicy::Pinned(7), // 默认 Auto：每次尝试独立随机
    ..Default::default()
};
```

`SeedPolicy::Auto` 为每次合成尝试解析新随机数，回读门禁重试因此换新采样，可逃逸采样
性错读；`Pinned(u64)` 按 hash(seed, 合成序号, 重试次数) 派生，同计划可复现且重试仍
变化。Pinned 复现以同一工具链与相同分段为前提（后端时长学习可能改变分段边界）。
每后端的类型化构造器（`VoxCpmParams` / `QwenParams` / `OmniVoiceParams` /
`MossParams` / `IndexTtsParams`，`to_generation_params()` 转换）随对应后端 feature 从本 crate 导出。

会话 `speed` 在模型声明 `speed` 参数（当前仅 OmniVoice）且宿主未显式设置时自动路由
到生成期无损变速，播放 sink 置 1.0；运行期 `configure_playback` 的改速按
新值÷原生值 换算 sink，不叠加。未声明 `speed` 的后端维持播放端变速。

自定义 Backend 实现者现在实现单个入口 `fn stream(&self, request: SegmentRequest)
-> Streaming`；请求携带 text/voice/style/context、本次尝试的具体 `seed` 与已校验的
`params`。不支持逐段风格或接续的后端用 `request.reject_unsupported(style, continuation)`
显式拒绝，不要静默忽略。

## 风格与情绪

逐段 `SpeechSpan.style` 与直接合成的 style 由各模型解释：Qwen 1.7B CustomVoice、
MOSS Local、CosyVoice3 使用指令；VoxCPM2 使用官方描述前缀；OmniVoice 使用其受限
属性词表。IndexTTS 接受八种命名情绪，并通过 `IndexTtsParams` / 能力目录开放
情绪向量、强度、随机锚点与独立情绪参考音色。未设置时沿用各模型默认表达。
控制方法与边界见 [CosyVoice3](cosyvoice3.md#逐段风格) 和 [IndexTTS](indextts.md#情绪控制)。
这些能力不要求开启接续，能力声明也不等同于所有情绪或语言的听感验收。

## 段落内接续

`PlanSessionOptions::continuation` 与 `SynthesisOptions::continuation` 默认开启；
配套的 `continuation_boundary`（协议 `PlanRequest.continuation_boundary`，v7 加法字段，
缺省 `paragraph`）决定哪些原文结构会清空链：`paragraph` 空行/装饰线/标题均清空（旧行为），
`scene` 仅标题与装饰线清空、空段落间隔保持接续，`none` 不因原文结构清空。
自链生成会随跳数累积音色漂移，因此 `continuation_span`（协议
`PlanRequest.continuation_span`，秒，1..=3600）限制单条链的累计音频时长：预算耗尽后在
下一个段落间隔重锚到静态音色参考，没有段落间隔的流在两倍预算处中途重锚；缺省 45 秒。
`Engine::synthesize_with_options` / `synthesize_pcm_with_options` 接收直接合成选项；
原便利入口使用默认设置。一次调用内部滚动使用上一段完整的生成文字与原始 PCM，
独立调用不共享历史。只有能力目录 `continuation=true` 的 OmniVoice、Qwen **0.6B / 1.7B Base**、
VoxCPM2、MOSS Nano / Local / Realtime 使用这一条件；Qwen CustomVoice / VoiceDesign 不支持。

```rust,ignore
let mut stream = engine.synthesize_with_options(
    text, voice, None,
    talechime::SynthesisOptions { continuation: false, ..Default::default() },
)?;
```

软换行、生成分块与相同音色/风格的连续 span 保留一段参考。按 `continuation_boundary`
命中的结构行、音色或风格变化清空；新执行、seek、恢复、停止和失败也清空。两种播放策略遵循相同规则。
参考是静音边界处理、音量、响度对齐与变速之前的完整音频，只接受 1–20 秒、最多 2048 UTF-8
文字字节、非静音且有限的 PCM。前段与候选参考合计最多 16 MiB，超限整段丢弃并记录原因。
只有显式 End 和成功交付的段落可以更新参考；门禁重试始终使用同一前段。
回读发现异常但仍交付的音频不作为下一段参考。接续编码或推理失败会终止合成。
交付侧在同一次执行内做有界响度对齐：慢基线跟踪有声段响度，逐包以约 ±3 dB 限幅、
线性斜坡的增益逼近基线，低于语音门限的停顿不参与跟踪，峰值超过 0.99 时整包回缩。

Qwen `0.6b-base` / `1.7b-base` 必须先导入带准确转写的参考音色，其资源与 CustomVoice
隔离，默认模型仍为 CustomVoice。接续不写入临时音色、不自动迁移或生成预置参考。
能力标记表示实现已提供，听感改善须试听对照音频后单独验收。

## 整章缓存

封闭的 `AfterChapterReady` 计划在确定性条件下可复用持久化整章音频：从正文第 0 字节开始、
采样 seed 为 `Pinned`、回读策略为 Off，且 Backend 提供 `cache_identity()`。
身份须覆盖实现版本、实际设备、模型与预置音色资源；不能保证稳定身份的 Backend 返回
`None`。`custom:` 音色暂不缓存，避免同名参考资源变化后错误重放。
`ChapterCacheOptions` 指定目录与字节预算（默认 8 GiB）；worker 的标准目录为
`~/.talechime/cache/chapters`。SHA-256 键覆盖 core 版本、Backend 资源身份、后端/模型、
正文摘要、seed、接续设置、生成参数与逐段音色/风格。原生生成语速属于生成参数，参与键；
仅播放侧音量与速度不参与键。
命中经过完整校验后跳过合成直接重放，生成进度已完成，播放进度仍按实际消费推进。
未命中时，每次执行持有独立临时 spool；全章成功校验后以硬链接原子发布，不覆盖
其他 worker 的已发布结果；执行保留独立路径与打开句柄，淘汰发布条目不影响播放。
候选持有执行文件锁；正常取消或失败移除该执行候选，强制退出遗留的已标记候选在下一次
缓存淘汰扫描时回收，活跃候选保留。目录扫描与音频校验在阻塞线程
执行，mtime 淘汰将已发布文件降到预算内。发布竞争或失败时仍播放当前成功生成的音频。
Auto seed、开放增量计划、断点续播及开启回读的执行使用一次性 staging。

当前六个模型适配器提供身份，包含适配器版本、具体实现、固定模型资源 revision、实际设备、
操作系统、架构与资源目录；MOSS Local / Realtime 同时纳入 codec revision。
Qwen 和 MOSS ONNX 若启用非 CPU 设备故障恢复，执行设备可能切换，因此不使用整章缓存。
缓存冻结首次成功执行的完整音频与分段，运行期间学习的时长估计不会使它失效。
固定 seed 支持重新合成时尽力复现，不能保证跨设备或不同执行状态的 PCM 完全相同。

当前 MOSS Candle/Nano、VoxCPM、OmniVoice 和稳定设备的 Qwen/ONNX 实现提供资源身份。
启用 CPU 自动恢复的非 CPU Qwen/ONNX 实例保持一次性 staging，避免运行中设备改变后
误命中缓存。缓存冻结首次完成执行的分段与音频，后续自适应时长估计不会重分已有条目。
