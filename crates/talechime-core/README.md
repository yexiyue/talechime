# talechime-core

会话核心，供 `talechime` 的 CLI 和 JSON Lines 入口共同使用。阅读器只依赖 `talechime-protocol`，不链接本库。

## API 迁移

原 `novel-tts` 库更名为 `talechime-core`。消费者在 Cargo.toml 中使用 `tts-core = { package = "talechime-core", version = "0.3.0" }`，Rust 引用为 `tts_core`；本次改名尚未发布，当前工作区可使用对应 path 依赖。

此次重构移除 `NovelTTS`、`ChapterTTS`、`Player` 和无界音频 queue API，不再公开 Kokoro 或 rodio 类型。使用 `backend::Backend`、`player::Playback` 和 `session::SessionManager`；位置为 `TextRange` 原文 UTF-8 字节范围，结束原因是 completed/cancelled/failed。示例见 `../talechime-backends/examples/moss.rs`。

核心不包含模型 feature 或推理依赖；音频输出使用 rodio 0.21.1。MOSS 与 GPU 模型适配器及固定 ort 版本位于 talechime-backends。Backend::stream 是统一接口，输出 PCM 块和显式 End；synthesize 仅是离线收集辅助方法。分段由后端提供，核心拥有原文范围、缓冲预算、取消和实际播放完成检查点。

`SessionManager` 和播放器运行于 Tokio LocalSet。设备只归播放线程，模型由专用推理线程构建、使用和销毁；线程之间只传拥有所有权的正文与 PCM。暂停停止消费；预取和播放受 30 秒、16 MiB 音频预算约束，播放消费后归还额度。模型推理自身的临时工作内存不属于音频队列预算。单个 PCM 块超过预算会失败；流式段可超过总预算，播放消耗后继续生成，不静默截断。

速度 0.5..2 和音量 0..10 由播放层处理。任何合成错误都停止会话。start、seek、切音色替换会话，旧会话取消，不跳过失败片段。合成结束不等于播放结束，片段进度来自播放器原始音频帧时钟，检查点只保存实际播放的片段边界。播放器时钟反映混音器消费，硬件输出缓冲延迟尚未补偿。

## 配置与恢复

唯一配置写入方是听书程序，默认路径为 `~/.talechime/config.json`。普通读取和 Drop 都不保存；首次激活才保存所选默认配置。更新校验后用临时文件原子替换，保留未知字段，短文件锁及 revision 防止并发覆盖。损坏文件、未知 backend、非法设置报错，不重置文件。revision_conflict 后重新查询设置再主动修改。旧主目录和退休后端不自动迁移。

具体模型资源由后端 crate 管理。检查点在 `~/.talechime/checkpoints/`，文件名是来源 ID 摘要；内容带 schema_version=1、来源、原文 SHA-256、resume_byte、completed 和更新时间。CLI 使用 cli-file 命名空间，阅读器使用 reader，彼此不覆盖。正文变化、非 UTF-8 边界或未知版本报错；损坏数据也报错。用户可通过 CLI `--restart` 或面板「从本章开头重新播放」显式忽略旧恢复点。

每段播放开始前保存起点，实际播放结束后保存终点；整章播放耗尽后才保存完成。取消会等待正在提交的检查点事务，避免旧会话覆盖新位置。故障后最多重复最近未完成片段；不承诺采样级恢复。读取检查点不会自动播放。

## 连续合成与播放

`BoundarySilence` 只裁剪块边界，不改变块内停顿。

生产者连续预取，FIFO 播放标记记录已入队的起止时间，不在每块末尾等待 sink 排空。暂停停止推进检查点；跳转、换音色、停止会话取消旧任务并隔离旧帧计数器。

后端通过 `paragraph_end` 判定语义段落边界；core 不将单个换行一律解释为段落尾。会话采用片段高亮和片段播放完成检查点。错误/不完整流不会提交当前片段完成或继续后文。
