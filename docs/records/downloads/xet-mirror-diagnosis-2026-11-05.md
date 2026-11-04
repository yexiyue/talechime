# Xet 与镜像真实下载诊断

2026-11-05，macOS，本机网络与系统代理配置保持原样。仅使用公开仓库，不传私有令牌；每次独立 `/tmp` 缓存，最多 120 秒，观察到至少 64 MiB 文件进度即取消（单次进度跃迁可超过阈值）。没有下载完整模型或加载推理。并发实验共享带宽，不能据此比较绝对速度。

## 对象与逐段验证

仓库 `OpenMOSS-Team/MOSS-TTS-Realtime`，revision `75682787d8e2fcc73faca37ba2931453ca9c4022`，`model.safetensors` 为 4,663,931,664 字节。

官方 HEAD 返回 302，携带 X-Linked-Etag、X-Xet-Hash 和正确大小；Xet read-token 为 200，CAS 主机是 `cas-server.xethub.hf.co`。`/v1/reconstructions/{hash}` 返回 200，数据块主机为 `us.aws.cdn.hf.co`。使用服务端给出的签名地址和 Range 单独读取数据块，返回 206、收到 1,045,676 字节。令牌、签名地址和请求头仅在诊断进程内存中使用，不写入记录。

HF-Mirror HEAD 返回 308、Location 主机为 `huggingface.co`，没有 ETag 或 Xet 元数据；它的 read-token 接口返回 200，CAS 仍为同一官方主机。此固定文件的镜像入口把请求转回官方，没有在这次请求中提供独立的大文件分发路径。镜像站[使用说明](https://hf-mirror.com/)也建议在 xet-read-token 404 时改用 HTTP，但本次不是 404。

## 实际下载对照

| 路径 | 首次正数文件进度 | 停止时观察到的文件进度 |
| --- | --- | --- |
| 官方、默认 Xet | 120 秒内没有 | 0 |
| 镜像、默认 Xet，修正 308 后 | 120 秒内没有 | 0 |
| 镜像、标准 HTTP，修正 308 后 | 16.41 秒 | 120 秒 13,565,412 字节 |
| 官方、Xet V1、小批量 | 15.73 秒 | 约 69 秒达到 73,000,000 字节并取消 |
| 官方、Xet V2、小批量 | 17.94 秒 | 约 87 秒达到 70,000,000 字节并取消 |

小批量对照仅在诊断进程设置：`HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE=1mb`、`HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE=1mb`、`HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER=1mb`，分别令 `HF_XET_CLIENT_RECONSTRUCTION_API_VERSION=1` 或 `2`。未更改用户或应用的全局环境、端点配置。

默认 Xet 镜像实验在文件进度仍为零时，`nettop` 对该下载进程抽样累计网络接收 167,993,439 字节、发送 28,695 字节。这是进程网络计数，包含协议开销，不能当作已重建模型大小；但足以排除“完全没有收到网络数据”。

## 修正先前结论

早期确实捕获过 `us.aws.cdn.hf.co/xorbs/default/...` 的 `tls handshake eof`。这只能证明曾有连接失败，不能证明之后所有请求都失败，更不能把 0 B 文件进度当成零网络接收。

Xet 1.8.0 源码中，`xet-data/file_reconstruction/reconstruction_terms/xorb_block.rs` 的网络回调更新 transfer bytes，`file_reconstruction/run_state.rs::report_bytes_written` 更新文件 completed bytes。hf-hub 1.0.0 的 `spawn_download_progress_poller` 对下载只转发 completed bytes，没有转发独立的网络接收计数。默认最小重建请求为 256 MB、最小预取为 1 GB；缩小批量后两个 API 版本都能开始报告文件进度。因此，已经确认的体验问题是网络传输与文件重建进度脱节；默认批量/缓冲是本次对照支持的主要影响因素。不能把一次 TLS 错误当作唯一根因，也不能证明 TLS/代理的具体故障原因。

应用继续使用已验证的标准 HTTP 流式进度；本次没有自动改回 Xet，也没有修改用户的镜像设置。若以后启用 Xet，应分别公开网络活动与文件重建阶段，并使用网络活动判断超时，不能用仅文件重建进度的 90 秒超时规则套用 Xet。

## 镜像修复与回归

hf-hub 原逻辑只跟随相对 HEAD 跳转，镜像的绝对 308 导致 `missing ETag header`。补丁跟随缺少元数据的绝对跳转，保留携带 ETag 的 CDN 跳转边界；跨 origin 清除凭据，禁止 HTTPS 降级，跳转次数有界。没有关闭证书校验。小型诊断日志只记录警告类别，不输出签名 URL、令牌或请求头。

本地 HTTP Hub 集成测试新增镜像跨 origin 308（确认目标没有 Authorization）及携带元数据时不发 CDN HEAD，连同既有缓存/续传测试共 5 项；普通单元 29 项。配套 CastGlean 回归与代码检查记录见其 `docs/desktop-studio/model-progress-validation.md`。公网结果不是完整模型下载、推理或 Windows 验收。
