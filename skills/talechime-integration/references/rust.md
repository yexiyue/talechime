# Rust 接入流程

最终应做到：应用交入短文本和声音 id，拿到完整 PCM 或真正完成的播放事件，然后可以
复用或关闭 Engine。按下面顺序先跑通单音色，再扩展多音色与增量输入。

## 1. 配好依赖和执行环境

使用当前 checkout 时，依赖 `crates/talechime` 的绝对 path，或固定 Git revision。确认
发布包包含需要的 API 后再使用 registry version。嵌入时选 `default-features = false` 和
所需模型，例如 `moss-nano-candle`；要让库打开扬声器时额外加 `playback`。

Tokio 至少启用 rt、macros、time。应用保有 runtime，用 `run_local` 或自己的 LocalSet
运行后面的操作。不要把 Engine/Listening 放进要求 Send 的 `tokio::spawn`。
GPU feature 按模型和平台选择，不同时开启全部 GPU feature。

**本步结果：** 依赖能编译，模型对象将留在一个持续运行的 LocalSet 内。

## 2. 选模型和设备

调用 `model_catalog(resources)`，从结果选准确 backend/model 和 available_devices。
同时查看 voices、cloning 和 parameters，供下一步选声音。这里只读目录，不下载权重。

`ModelOptions::new` 默认 CPU；其他设备需要显式设置。Auto 的校准与恢复由 calibrate /
recovery 控制。设备错误不应触发接入方暗中换模型或下载另一套权重。

**本步结果：** 得到一份明确的 ModelOptions，知道所选模型能接受什么声音和参数。

## 3. 选声音，并统一数据路径

已有内建声音时，记下目录里的 id，直接进入准备模型。需要用户音色时，先创建
`VoiceStudio::at(resources, voices)`，选一种方式：

- 已有录音：`import_voice(Some("reader"), "朗读者", wav_path, transcript)`。
- 文字设计：`design(description).await`；要名称、进度或指定引擎，改用 design_with。
- 已有库音色：通过 list/get 找到其 id。

创建完成会返回 VoiceAsset；把 asset.id 留给后面的合成调用。设计前可用 engines() 查询
可用设备和下载代价。导入转写必须准确；使用库音色的合成模型必须支持 cloning。

**本步结果：** 得到声音 id；Engine 的 resources 和 voices 与 VoiceStudio 使用同一组目录。
只需把库 id 直接交给合成；通常不必先调用 install_voice_for。

## 4. 准备 Engine

调用 `Engine::prepare(options, progress).await`，转发进度并处理错误。这个调用可能下载
所选合成模型，成功返回后再开始执行。若目录中没有声音且没有提供库音色，应先返回
上一步，而不是猜一个 id。

**本步结果：** 持有一个可执行、可复用的 Engine。下面选择 PCM 或播放其中一条路径。

## 5A. 需要 PCM：启动流并持续消费

调用 synthesize(...).await，循环 recv()，把 block.pcm() 交给宿主播放器、编码器或文件
写入器，使用 block.range() 关联原文。块消费后释放，避免预算背压堵住生产者。
短文本也可用 synthesize_pcm(..., max_bytes)，字节限额必须非零且有限。

消费结束后检查 `SynthesisState::Completed`。取消用 stream.cancel().await；不要仅凭
recv() 返回 None 就认为生成成功。一次合成结束后可以复用同一个 Engine。

## 5B. 需要播放：建计划，再启动 Listening

先建立 `SourceSnapshot::new(source_id, exact_text, &text_hash(exact_text))`，再选计划形式：

| 文本何时确定 | 构造方式 | 何时封闭 |
| --- | --- | --- |
| 完整单音色 | `single_voice_plan(...).await` | 已封闭 |
| 多音色，或声音范围将陆续产生 | `plan(source, voices, policy).await`，再 append | 覆盖全文后 seal |

Streaming 边生成边播放；AfterChapterReady 等封闭且整章生成成功后再播放。
然后调用 `engine.listen(options)`，取 control 句柄并 start。需要宿主播放器时用
listen_with；已有事件队列时用 listen_to/listen_with_events。

启动后，事件消费与控制确认并发进行。开放计划继续从 control.append 追加连续范围，
最后 control.seal；库音色仍使用原库 id。范围使用同一正文的 UTF-8 字节偏移，不能用
字符数。分析失败调用 fail_input，不能通过 seal 掩盖失败。

持续消费事件，直到 SessionEnded；只有 reason=Completed 才算真正播完。pause/resume /
stop 控制当前执行；seek 必须提供新 id，随后控制使用新 id。需要持久恢复时显式设置
ListeningOptions.checkpoints；默认不保存检查点，PlanSessionOptions 默认不恢复。

## 6. 结束本次使用

PCM 流先收完或 cancel；播放先 close Listening，然后再 `engine.close().await`。
如果要继续使用相同模型，可以保留 Engine，不必每句重新加载。
Engine::close 不强行销毁仍然活跃的 Listening；Drop 只是请求取消，不等于清理已经完成。

## 最小完整示例：内建音色到 PCM

示例要求编入 moss-nano-candle 并在本机可用 CPU。运行它可能下载模型；编译验证本身
不会下载模型。它展示上述发现、准备、执行、检查和关闭的顺序。

```rust,no_run
use talechime::{Engine, ModelOptions, SynthesisState, model_catalog, run_local};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_local(async {
        let root = std::path::PathBuf::from("./my-app-speech");
        let resources = root.join("resources");
        let models = model_catalog(&resources)?; // 只读，不准备模型
        let selected = models.iter().find(|model| {
            model.capabilities.backend == "moss"
                && model.capabilities.model.as_deref() == Some("nano-candle")
                && model.available_devices.contains(&talechime::Device::Cpu)
        }).ok_or("build with moss-nano-candle and select an available device")?;
        let mut options = ModelOptions::new(&selected.capabilities.backend, &resources);
        options.model = selected.capabilities.model.clone();
        options.voices = Some(root.join("voices"));
        let mut engine = Engine::prepare(options, |event| eprintln!("{event:?}")).await?;
        let voice = engine.capabilities()?.default_voice;
        let mut stream = engine.synthesize("你好。", &voice, None).await?;
        while let Some(block) = stream.recv().await {
            let block = block?;
            // 将 block.pcm() 送入宿主 sink；用 block.range() 关联原文。
            println!("{} samples", block.pcm().samples.len());
        }
        if stream.state() != SynthesisState::Completed {
            return Err("synthesis did not complete".into());
        }
        engine.close().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    }).await
}
```

## 跑通后再加的能力

生成参数通过 synthesize_with_options / synthesize_pcm_with_options 的 SynthesisOptions
传入，包括 GenerationParams、SeedPolicy、接续和回读。具体参数只从所选模型的
Capabilities.parameters 读取，不把 steps、language、speed 当成通用参数。

音色资产的 rename/remove/export/import_bundle 与生成独立；显式
install_voice_for(backend, model, id).await 适合提前准备目标模型参考。用户资源不应直接
写进缓存目录，音色搬运包也不等于生成结果的 WAV。
