# HeyGen cloud provider 实施验收（2026-11-11）

平台：Windows，本工作区开发版 `talechime-cloud-tts`；使用锁定依赖，未新增原生或模型依赖。范围为独立 HeyGen provider，不包含 CLI/协议、宿主、克隆或播放集成。

## API 与设计依据

[OpenSpec 设计](../../../openspec/changes/add-heygen-cloud-provider/design.md)记录官方来源与当日 schema 差异：本库使用 5000 字符/50 条单页子集；原始流明确指定 output_format，拒绝 SSE/JSON 成功响应。公开接口保留三种显式模型，不自动切付费或重复合成。

## 普通验证

本地 TCP HTTP mock 覆盖两种格式及请求字段、中文 5000 字符边界、目录游标和过滤/详情包装、状态与扩展字段、格式头缺失/冲突、响应上限、空流、截断、取消/drop、重定向和不自动重试、敏感错误。示例查询与写音频路径也使用 mock；不需要凭据、不连接 HeyGen、不下载模型。

集成命令：`cargo check --locked -p talechime-cloud-tts`、依赖树检查、`cargo test --locked -p talechime-cloud-tts --all-targets`、`cargo fmt --all --check`、crate all-targets Clippy（-D warnings）及 no-deps/private-items/examples rustdoc（RUSTDOCFLAGS=-D warnings）。上述命令均已通过：27 项测试（2 个单测、13 个 Fish HTTP、9 个 HeyGen HTTP、3 个示例测试），Clippy/rustdoc 无警告，格式检查通过。

## 未验证边界

此前 HeyGen CLI 认证曾通过系统代理成功，这只证明当时凭据能认证。Rust provider 的真实音色查询、PCM/MP3 合成与响应头、音频解码、人工试听、文本完整性、延迟/性能、账单费用及其他操作系统均未验证。本轮未执行真实合成，也未上传参考音频或创建远端音色；普通测试不能替代这些验收。

原始流不存在显式 DONE：即使 HTTP 正常 EOF，也不能证明语音内容完整。单块由 HTTP 库分配，本库预算限制累计交付，不宣称约束所有底层网络缓冲。取消不保证远端停止处理或免除费用。


## 同日真实服务补充：双云试听样本

用户明确请求两云生成试听后，使用独立、Git 忽略的本地探针对 Fish 与 HeyGen 发送相同原创中文短段，均使用公开音色与显式免费模型；没有创建或克隆音色。通过进程系统代理配置调用当前 Rust provider，而非 CLI 合成。探针是忽略目录内独立 Cargo workspace，使用独立生成的 Cargo.lock（reqwest 0.13.5、tokio 1.54.1、serde_json 1.0.154），不等同于主工作区锁定依赖下的真实服务验收。

- HeyGen `heygen-voice-1-turbo-free`：公开瞬时音色，目录查询成功；MP3 请求及格式响应头校验通过，返回 274181 字节，单次合成/拉取约 2.594 秒；ffprobe 检出 44.1 kHz、单声道 MP3，17.136313 秒。
- Fish `s2.1-pro-free`：公开中文目录查询成功；返回 356936 字节，单次合成/拉取约 9.932 秒；ffprobe 检出 44.1 kHz、单声道 MP3，22.308500 秒。
- 两份文件经 ffmpeg 全量解码至空输出均无错误。音频、正文、远端 ID 和探针仅在本机忽略的 output/cloud-listening-20261111/，不纳入 Git。

这是不同公开音色的单次同文试听样本，不是同音色、同输出时长或多轮性能基准；耗时不含目录查询和编译，不是首音频延迟。尚未得到用户听感反馈，内容完整性、音色质量、人工试听、计费账单、PCM 输出及其他平台仍未验收。上述成功补充仅更新此前“Rust 音色查询/MP3 合成/解码未验证”的边界，不追溯替代原先的普通测试或 CLI 认证证据。


## 同日提交审查补充

规范审查未发现阻塞问题；规格审查发现格式头重复时只检查第一个值的问题，现要求 Content-Type、X-Audio-Sample-Rate、X-Audio-Channels 各恰好一个，重复或冲突直接拒绝。已将三类重复冲突头纳入现有 mock 回归。元数据校验收拢到私有构造方法，Fish 公共 API 和生成行为未改变；修正知识索引、根 README 的证据入口，并注明真实探针的独立依赖锁边界。
