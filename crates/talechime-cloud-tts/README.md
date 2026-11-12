# talechime-cloud-tts

独立 Rust 云语音客户端，当前提供 Fish Audio HTTP 合成、音色发现以及即时和持久参考克隆；不依赖本地模型或播放设备。宿主负责章节、角色、音色绑定及播放。

## 合成

设置 `FISH_API_KEY` 与 `FISH_VOICE_ID`，运行 `cargo run -p talechime-cloud-tts --example fish_synthesize -- output.mp3`。示例显式选用 `s2.1-pro-free`，不会自动切换付费模型；运行前应确认当前服务条款、可用性及价格。环境变量由调用方提供，crate 不自动读取 `.env`。

网络需要代理时由宿主配置 `HTTPS_PROXY`；Windows 系统代理不一定自动传给 Rust 客户端，排查记录见[知识库](../../docs/knowledge/lessons/fish-audio-windows-proxy.md)。

`Client::synthesize` 返回拉取式 `AudioStream`，以调用方指定的最大字节数限制响应。网络块不是音频帧；音频格式与可选采样率是请求值，未显式指定的采样率由服务端选择。`Completed` 表示非空 HTTP 响应正常结束，既不证明内容完整，也不代表音频已播放。取消或丢弃响应只释放本地连接，不保证服务端停止生成或免除费用。

## 音色与参考录音

`search_voices` 和 `get_voice` 获取公开及账户音色。远端 ID 属于 Fish，可能因账号、权限、删除或服务变化而失效；宿主应保存 provider 与 ID，不应将它伪装成可离线使用的本地音色资产。可用状态由服务返回，未知状态会保留。

`Voice::References` 每次上传调用方提供的录音和精确转写进行即时克隆，不创建持久音色。`create_voice` 显式上传样本并创建私有云端音色，返回 ID 与状态；调用方可查询状态。远端创建在网络中断时可能已经成功，客户端返回结果未知且不会自动重试。删除本地音色不会删除远端音色，本 crate 不修改本地音色库。

文本、参考录音和元数据会发送给 Fish；使用前应确认内容与声音的授权、服务方数据处理及适用条款。API key 由宿主安全保存，不能放入日志、项目文件或版本控制。价格与免费额度可能变化，以 [Fish 定价](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits)为准。不要把 HTTP 成功或测试合成当作音质、授权或生产 SLA 的证明。

普通测试使用本地模拟 HTTP 服务，不访问 Fish。真实 API、人工试听、延迟与费用观察需单独记录。

`clone_probe` 是会创建远端音色的显式实测示例。它在创建前独占创建输出目录中的 `remote-voice-id.txt`，避免覆盖已有 ID；失败时可能留下空文件，此时远端结果仍可能未知，应先核对账户，不能直接重跑创建。


## HeyGen provider

`heygen::Client` 直接调用 v3 HTTP API，不依赖 HeyGen CLI 或视频/数字人技能。支持公开/自有音色的游标分页、公开目录语言/性别/年龄/文本过滤，以及远端 ID 详情查询；原始状态和扩展元数据保留，已有克隆音色可通过远端 ID 引用，本轮不创建、设计或删除音色。

设置 `HEYGEN_API_KEY` 与 `HEYGEN_VOICE_ID` 后使用：

```sh
cargo run --locked -p talechime-cloud-tts --example heygen_list_voices
cargo run --locked -p talechime-cloud-tts --example heygen_synthesize -- output.mp3
```

示例由宿主进程环境读取凭据，再显式传给客户端；库和示例均不自动加载 `.env`。需要代理时由调用方配置 `HTTPS_PROXY`。示例显式选择 `Model::TurboFree`；另外支持 `Voice1` 与 `Turbo`，不采用服务端默认模型，不切换付费档、不自动重试。免费期限、容量和账号权限以 [HeyGen 当前文档](https://developers.heygen.com/docs/voices/heygen-voice-speech.md)为准。

`SpeechRequest` 要求非空白文本（最多 5000 个 Unicode 标量值）、远端 voice_id、显式格式及正数 `max_audio_bytes`，可指定基础语言代码如 `zh`。`AudioFormat::Pcm16` 是单声道 24 kHz signed little-endian PCM16、没有 WAV 头；`Mp3` 是单声道 44.1 kHz/128 kbps。响应类型、采样率与声道经服务响应头校验，`format`/`sample_rate` 是请求值，`metadata` 是服务端确认值。网络块不保证采样对齐或音频帧边界。

流有累计字节预算，超限块不交付；`cancel`/drop 释放本地响应。`Completed` 只表示非空 HTTP body 正常结束：原始流没有 SSE 的完成标记，无法检测所有正常 EOF 下的内容缺失，也不证明播放完成。无效/超大响应、传输错误与结果未知区分；服务错误保留 402/404/409 等状态码，不输出服务错误正文。没有默认 WAV/SSE、时间戳或 WebSocket 支持。

音色列表每页 1–50 条，继续分页时传回 `next_token` 并保留相同过滤；自有范围不接受公开过滤。PENDING、FAILED 和未知状态不映射为可用，不保证每条音色兼容所选模型。本地音色删除不影响远端身份。文本会上传至 HeyGen，调用方负责授权、密钥和费用；当前真实服务/解码/试听/性能边界见 [实施验收记录](../../docs/records/tts-core/heygen-cloud-provider-2026-11-11.md)。
