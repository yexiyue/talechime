# Fish Audio：Windows 系统代理与 Rust 客户端

2026-11-11，Windows 工作区开发版 `talechime-cloud-tts`，在此机器上 PowerShell 走系统代理能查询 Fish，而 Rust 客户端和 curl 直接连接超时。系统代理并未自动成为进程的 `HTTPS_PROXY`。在该进程显式设置 `HTTPS_PROXY` 后，Fish 免费模型短句合成成功。来源与未验收项见[实测记录](../../records/tts-core/fish-cloud-smoke-2026-11-11.md)。

宿主应根据本机网络配置提供代理环境变量或其他合适配置；不要记录代理凭据。此结论只适用于该机器及当时网络，不保证所有 Windows 系统都有相同行为。请求一旦发出后网络失败，远端处理与计费可能未知，不应自动重试。
