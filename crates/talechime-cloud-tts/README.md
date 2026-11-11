# talechime-cloud-tts

独立 Rust 云语音客户端，当前提供 Fish Audio HTTP 合成、音色发现以及即时和持久参考克隆；不依赖本地模型或播放设备。宿主负责章节、角色、音色绑定及播放。

## 合成

设置 `FISH_API_KEY` 与 `FISH_VOICE_ID`，运行 `cargo run -p talechime-cloud-tts --example synthesize -- output.mp3`。示例显式选用 `s2.1-pro-free`，不会自动切换付费模型；运行前应确认当前服务条款、可用性及价格。环境变量由调用方提供，crate 不自动读取 `.env`。

网络需要代理时由宿主配置 `HTTPS_PROXY`；Windows 系统代理不一定自动传给 Rust 客户端，排查记录见[知识库](../../docs/knowledge/lessons/fish-audio-windows-proxy.md)。

`Client::synthesize` 返回拉取式 `AudioStream`，以调用方指定的最大字节数限制响应。网络块不是音频帧；音频格式与可选采样率是请求值，未显式指定的采样率由服务端选择。`Completed` 表示非空 HTTP 响应正常结束，既不证明内容完整，也不代表音频已播放。取消或丢弃响应只释放本地连接，不保证服务端停止生成或免除费用。

## 音色与参考录音

`search_voices` 和 `get_voice` 获取公开及账户音色。远端 ID 属于 Fish，可能因账号、权限、删除或服务变化而失效；宿主应保存 provider 与 ID，不应将它伪装成可离线使用的本地音色资产。可用状态由服务返回，未知状态会保留。

`Voice::References` 每次上传调用方提供的录音和精确转写进行即时克隆，不创建持久音色。`create_voice` 显式上传样本并创建私有云端音色，返回 ID 与状态；调用方可查询状态。远端创建在网络中断时可能已经成功，客户端返回结果未知且不会自动重试。删除本地音色不会删除远端音色，本 crate 不修改本地音色库。

文本、参考录音和元数据会发送给 Fish；使用前应确认内容与声音的授权、服务方数据处理及适用条款。API key 由宿主安全保存，不能放入日志、项目文件或版本控制。价格与免费额度可能变化，以 [Fish 定价](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits)为准。不要把 HTTP 成功或测试合成当作音质、授权或生产 SLA 的证明。

普通测试使用本地模拟 HTTP 服务，不访问 Fish。真实 API、人工试听、延迟与费用观察需单独记录。

`clone_probe` 是会创建远端音色的显式实测示例。它在创建前独占创建输出目录中的 `remote-voice-id.txt`，避免覆盖已有 ID；失败时可能留下空文件，此时远端结果仍可能未知，应先核对账户，不能直接重跑创建。
