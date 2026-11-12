# Windows 音色临时文件与独立租约

日期：2026-11-07。状态：Windows 已验证；macOS/Linux 未在本轮运行。适用：当前 `talechime-core` 音色库 v2，无模型、协议或公开 API 变更。

CastGlean 的音色保存、重复保存和音色包检查测试在 Windows 报文件锁错误 33。原因是 [store.rs](../../../crates/talechime-core/src/voices/store.rs) 的 `temporary()` 对内容文件自身上锁，保存或 ZIP 读取随后从路径重开；Windows 的文件锁会阻止另一句柄读取，即使在同一进程。

临时内容改为使用已有 Scratch 目录，独立 `.lease` 保护目录，`Temporary` 同时拥有内容文件与目录所有者；保留旧的当前格式废弃临时文件回收，不迁移历史资产。引用持有期间可从路径或 `reopen()` 读取，释放后清理内容及目录。

Windows 还会拒绝重命名包含锁定文件的暂存目录。因此正式发布在持有库锁时释放暂存 lease 后重命名，暂存回收也持有同一库锁，避免释放到发布之间被回收。废弃 Scratch 目录已经通过独占 lease 确认没有活动所有者；删除前释放回收者自己的锁。唯一 Scratch 目录不会被新任务接管，并发回收者重复删除只会得到不存在，不会删除活动任务。

验证：`cargo test -p talechime-core --lib --offline` 70 项通过，含内容可重开、活动 lease 不清理、释放后回收、部分发布失败和幂等重导入；相关 Clippy 严格检查与 rustdoc 通过。相邻 CastGlean 的完整宿主回归也已通过，原三项文件锁失败转为通过。审查后将两个回收测试的 Windows 目录时间设置合并为测试专用 helper，并再次通过上述 70 项测试、Clippy 与 rustdoc。测试使用合成 WAV，不需要模型、私人录音或网络。未发布 SDK 或应用版本。
