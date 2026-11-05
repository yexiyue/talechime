# hf-hub 本地补丁

来源：官方 crates.io `hf-hub 1.0.0`，上游提交 `dcf865adacf7cbb750038d6521deb16c670ffd42`，仓库 https://github.com/huggingface/hf-hub ，Apache-2.0（见 LICENSE）。保留发布包的源码、测试与清单。

补丁只服务单文件缓存下载：增加 `download_file().use_xet(false)`，默认仍为 true。TaleChime 显式选用标准 HTTP；保留上游共享缓存命名、锁与快照指针。HTTP 缓存分支补充 Range 续传、锁后重新检查、状态/Content-Range/总长度校验，服务器忽略 Range 时截断重写；不发布半成品。真实下载复现 Xet 长期无文件重建进度，并曾出现 CDN TLS handshake EOF。进一步对照确认网络接收与重建完成进度不同，不能把文件进度为零解释为没有网络数据；见[诊断记录](../../docs/records/downloads/xet-mirror-diagnosis-2026-11-05.md)。标准 HTTP 按流式读取报告字节。没有关闭证书验证。

此源码通过工作区 path 依赖消费，不需要宿主额外 patch；当前依赖 crates.io 同版本的发布构建不包含本地 API，发布前必须先迁移到提供该能力的上游版本或发布独立补丁包，不可直接发布当前 TaleChime 包。

镜像补丁：元数据 HEAD 支持无 ETag 的绝对 308/其他跳转；拿到携带 ETag 的 Hub→CDN 跳转后停止，避免误把 CDN 元数据当作 Hub 元数据。跨 origin 删除 Authorization、Cookie、Proxy-Authorization（含客户端默认头），禁止 HTTPS 降级，跳转有次数上限。新增本地 HTTP 测试覆盖镜像跳回源站、凭据隔离及不误追踪 CDN HEAD。

HTTP 分片扩展：`http_transfer.rs` 调度与校验、私有 `http_transfer/source.rs` 解析/刷新地址、`http_transfer/checkpoint.rs` 保存分片断点；大文件按 4 MiB 分片，每文件连接数可设 1–4、全局最多 4 个活跃下载请求（TaleChime 默认每文件 2）。保留原有 blob 锁及发布路径，Range 不支持时退回单流；校验响应范围/长度和可用的 SHA-256 ETag，不发布半成品。地址过期刷新检查 ETag/总大小，签名 URL 不写断点。取消时补存断点，写入失败可由 `download_checkpoint_failed()` 读取。网络连接与读超时分别为 15/90 秒，排队不计读超时。实现与实测见[并发下载验收](../../docs/records/downloads/parallel-download-acceptance-2026-11-05.md)。这些扩展同样受上述本地补丁发布限制约束。
