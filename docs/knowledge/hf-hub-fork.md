# hf-hub fork 与上游下载能力

2026-11-05 核对。当前维护仓库为 [yexiyue/hf-hub](https://github.com/yexiyue/hf-hub)，HTTP 补丁分支 `codex/resumable-http-downloads`，消费者以工作区 `Cargo.toml` 中的完整 Git `rev` 固定版本。TaleChime 不再保留 hf-hub vendor 副本；共享缓存、锁、blob 与快照仍由 hf-hub 管理。

## 源码与边界

HTTP 补丁基于官方 1.0.0 对应提交 `dcf865adacf7cbb750038d6521deb16c670ffd42`；本次迁移不升级上游基线。完整历史、Apache-2.0 许可证及补丁清单保存在 fork 的 `FORK.md` 与 crate 内 `SOURCE.md`。不依赖宿主的 workspace patch，不把下载实现复制回应用。

- 单文件显式禁用 Xet，按标准 HTTP 读取并报告真实文件字节。
- 元数据缺失的绝对 HEAD 跳转、凭据隔离、禁止 HTTPS 降级及跳转上限。
- 保留官方缓存发布路径和文件锁；Range 续传、状态/范围/长度和可用 SHA-256 校验。
- 每文件最多四路、全局四路数据请求，持久化分片断点，签名 URL 刷新时验证内容身份。
- 应用任务、暂停状态、重启恢复和速率/ETA 由宿主负责；不向宿主协议暴露 HF/Xet 内部类型。

Git 依赖解决维护与构建复现，不等于 SDK 可以发布到 crates.io。Cargo 发布会移除 Git 来源；发布前须采用提供这些 API 的上游正式版本，或独立命名的补丁 crate，并更新发布流程。当前未发布补丁 crate 或新版 SDK。

## 上游已有需求

| 需求 | 当前证据 | 处理建议 |
| --- | --- | --- |
| Rust HTTP 中断续传 | [Issue #196](https://github.com/huggingface/hf-hub/issues/196)、[PR #170](https://github.com/huggingface/hf-hub/pull/170) 均为 open | 补充 HTTP 分片断点与验证证据；不重复开续传 issue |
| 镜像缺失 ETag | [Issue #163](https://github.com/huggingface/hf-hub/issues/163) 为 open；维护者要求复测 main/1.0.0 | 补充无 ETag 的绝对 308 复现，与已处理的相对跳转区分 |
| 中途 body 失败 | [PR #227](https://github.com/huggingface/hf-hub/pull/227) 为 open；[Issue #229](https://github.com/huggingface/hf-hub/issues/229) 针对 stream | 整次重试与断点续传是不同需求，分开评审 |
| Xet future 取消后 poller 存活 | [PR #205](https://github.com/huggingface/hf-hub/pull/205) 为 open | 不重复提交同一生命周期修复 |
| 下载时网络有流量、文件仍为零 | Rust main `473de9430b0fbc544fb1d91102180eb713f8dc87` 的 poller 只转发逻辑重建字段，上传已转发网络字段 | 单独提交下载网络统计 PR，与 HTTP 续传补丁分开 |

PR #170 的维护者指出 Xet 文件可能乱序写入、有空洞，因此不能按文件长度追加续传。当前 HTTP 分片断点显式记录每片已保存位置，不推断 Xet 文件的完成范围。

## Python 对照

以已发布 [huggingface_hub v2.4.0](https://github.com/huggingface/huggingface_hub/releases/tag/v2.4.0) 为准核对源码：

- [PR #4400](https://github.com/huggingface/huggingface_hub/pull/4400) 已合并网络传输/文件重建双进度。[实现](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/utils/_xet_progress_reporting.py) 分别消费 `total_transfer_bytes_completed` 与 `total_bytes_completed`，不能把前者冒充文件完成百分比。
- 早期 [Issue #4058](https://github.com/huggingface/huggingface_hub/issues/4058) 与 PR #4059 仍显示 open，不表示当前仍没有修复；Xet 侧旧 [PR #791](https://github.com/huggingface/xet-core/pull/791) 已关闭且未合并，后续方案是 #4400。
- [`snapshot_download(max_workers=8)`](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/_snapshot_download.py) 控制并行文件数，不等于一个 HTTP 文件有八路分片。Xet 自行调度范围请求。
- [`http_get`](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/file_download.py) 能在同一次调用的网络异常后，按已收到字节发 Range 重试，并处理服务端忽略 Range 的情况。但 `_download_to_tmp_and_move` 使用唯一临时文件，失败/中断后在 `finally` 中清理，不能据此宣称支持应用重启后继续未完成文件。
- Python 高层下载函数不提供桌面应用持久化的“Paused 任务列表”。应用任务层仍有独立职责。

## Xet 修复范围

上游 [PR #240](https://github.com/huggingface/hf-hub/pull/240) 已提交：新增下载聚合事件中的网络字节、当前已发现的网络总量与网络速率，保持现有文件重建统计含义；私有 Xet 适配函数负责映射，CLI 和示例负责展示。成功结束前停止并等待 poller，再发送最终聚合快照，覆盖低于轮询间隔的短下载。网络传输完成不代替文件重建成功。修复已包含在当前 fork 中。

同一固定 MOSS Realtime 权重、独立临时缓存、90 秒 / 8 MiB 网络字节预算的真实探针，在 9.251 秒观察到 10,139,690 网络字节、重建字节仍为零；首次观察到网络进度约 7.501 秒。这验证新增事件可报告真实传输，不是完整文件验收。预算轮询会有少量超额。

这解决进度不可见，不承诺解决 CDN/TLS 链路故障或提升实际吞吐。应用默认 HTTP 路径及其暂停续传仍保留；完整 Xet 权重下载、模型推理和 Windows 实机不由短时进度探针证明。
