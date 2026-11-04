# 音色传输 v2 验证

2026-11-04，源码分支 `codex/voice-library-batch-transfer`。仅交付源码，不发布新版本。

- macOS core 单元 69 项、传输集成 20 项通过；涵盖 ZIP 严格格式、物理重复条目、资源预算、输入冻结、冲突矩阵、稳定新身份重试、原子发布、取消、写盘故障注入和子进程异常退出。
- 256 音色导出、检查、导入完整往返在 `ulimit -n 64` 下通过。批量 WAV 快照共享目录租约，冻结后不持有逐条文件句柄；活跃暂存不会被清理，崩溃遗留可回收。
- Linux Docker `rust:1.95` 在只读源码挂载及独立 target/cargo 卷运行相同 69 + 20 项测试通过。
- facade/CLI 公开传输集成 2 项通过；模型无关 API 不需要 Registry 或网络。CLI 和其他无默认模型消费者测试通过。
- 扩展 CPU 后端消费者测试 95 项通过；对应 facade Clippy、core/facade/CLI Clippy（`-D warnings`）、fmt 和严格 rustdoc 通过。
- CastGlean macOS 隔离原生应用验证两包 20 选五、五项单包导出、空库回导、重启持久化、损坏输入移除、重复导入、同 ID 声音冲突、新身份导入、大 WAV 取消及 20 MiB 测试卷真实磁盘写满。失败不破坏旧目标。

Windows 未验收；Linux 图形界面未验收。容器文件系统测试不能证明所有桌面外部程序占用及文件选择器行为。全预算 1 GiB 维度使用收紧预算测试边界，原生大 WAV 实测为 63 MiB。

模块职责：asset 负责合法资产与摘要，library 是无模型公开入口，store 隐藏记录/锁/租约/原子发布，package 隐藏 ZIP DTO，transfer 编排预览/计划/提交与报告。facade 和 CLI 复用同一接口。跨模型资产不包含模型缓存或角色绑定，来源不构成强制后端绑定。旧单项包/API/CLI 分支已删除，无兼容读取或自动迁移。
