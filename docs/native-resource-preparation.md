# 固定 Hub 资源准备

当前工作区使用固定仓库、commit、文件名、大小和 SHA-256 盘点共享 Hugging Face 缓存，并直接加载已验证文件。缺失或损坏时调用统一下载接口，不在用户机器转换模型或组装模型目录。Kokoro、CosyVoice3、IndexTTS、MOSS、Qwen、VoxCPM、OmniVoice 及识别资源分别使用编译时清单；模型可以跨仓库引用文件。

以下示意库调用；异步模型对象及推理对象须留在其 owner 线程：

```rust,no_run
# async fn example(registry: &talechime::Registry) -> anyhow::Result<()> {
let status = registry.resource_status("kokoro", Some("1.1-zh-g2pw")).await?;
let model = registry.resolve("kokoro", Some("1.1-zh-g2pw"))?;
let (progress, mut events) = tokio::sync::mpsc::channel(32);
let download = model.prepare_resources(progress);
let drain = async { while let Some(_event) = events.recv().await {} };
let (result, ()) = tokio::join!(download, drain);
result?;
# let _ = status;
# Ok(())
# }
```

`resource_status` 离线校验完整性并报告缺失字节；宿主负责展示空间、取得用户的下载确认和维护任务状态。`prepare_resources` 下载并校验固定文件；`load_existing` 仅加载已就绪资源，不发起下载。进度事件可能被合并，任务返回值与重新盘点是终态依据。取消时丢弃下载 future 并等待退出，暂停前检查 `Downloads::checkpoint_failed()`；已完成文件和客户端断点保留。

需要确认具体下载集合的宿主使用 `resource_inventory` 取得缺失字节和不透明指纹，并在开始下载前重新盘点比较指纹。指纹覆盖每项固定身份和就绪状态，因此缺失文件互换但总大小不变也会要求重新确认。

`Registry::with_storage_paths` 分别接收应用持久数据、应用缓存和 `DownloadConfig` 中的 Hub 缓存绝对路径，拒绝重叠。音色和参考编码保存在应用数据，校准在应用缓存。`Registry::clear_calibration_cache` 只移除应用可再生校准记录，不删除共享模型或持久音色。共享缓存可能被其他应用使用，本次不提供共享模型文件删除接口。

本页描述工作区源码，不代表已发布版本。旧原生转换流程和相关 API 已移除；历史背景见 [2026-11-05 记录](records/resources/native-preparation-2026-11-05.md)，直接缓存加载与一款真实模型的验证见 [2026-11-12 记录](records/resources/hf-cache-direct-2026-11-12.md)。其他设备、平台和人工听感另行验收。
