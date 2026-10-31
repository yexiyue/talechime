# MOSS Metal 设计与旁白候选批次

日期：2026-10-31。变更：`moss-metal-voice-candidates`，关联 Talechime #6。

## 构建与边界

以 main `ca07f66616d8a6181617892ff2edf4c0d13d94d3` 为基础，保留现有三个 VoxCPM 性能文件的未提交改动；初始 diff 保存于 `target/voice-candidates/20261031/initial-dirty.patch`，已通过逐字节比较确认未被覆盖。本构建含工作区改动，不冒充纯 main。

显式组合为 `cli,qwen-metal,voxcpm-metal,moss-candle-metal`。设计 Metal 使用 F16，MOSS CUDA 保留 BF16；没有改原生依赖、调度或采样数学。音频与日志放在 Git 忽略目录。

## MOSS 实机结果

新增 Metal 设备路径及加载同步/取消边界，编译和无模型设备校验通过。真实门禁失败：`invalid MOSS logits`，44.43 秒退出；最大进程 RSS 7,843,446,784 字节，macOS time 报告 peak memory footprint 18,375,027,344 字节（不是独立 GPU 显存测量）。未产出库音色；没有裁剪、修补或入库半成品。

这是数值失败，不是成功设计，也不是已经复现 OOM。不据此宣称 MOSS Metal 可用；两 seed、真实取消后成功复用和跨后端听感仍未完成。批次中的 16 个 MOSS 候选按预定分支标记 blocked，其余候选继续。根因仍需定位，不能把强制 CUDA 移除等同于完整支持。

## 资源准备

Hugging Face 直连超时，使用系统已配置代理仍较慢；最终从官方发布者 ModelScope 的对应模型副本下载，所有文件必须同时满足仓库 manifest 的 size 和固定 SHA-256，完成后才放入既定 revision 目录。未更改资源清单或将 master 内容直接视为固定版本。MOSS/Qwen 设计权重均完成校验，详细记录见 `resource-preparation.log`。

## 真实联调发现

- hound 的 32-bit float WAV 使用 WAVE_FORMAT_EXTENSIBLE；初版批次检查器不认识该子格式，误报已成功保存的设计。已补充格式支持和回归测试，保留原始 WAV，不重新设计已入库项。
- 首次在 Backend 加载后隐式物化音色，会遇到已冻结音色目录不含新 custom id 的错误。此次试听工具改为先通过公开 VoiceStudio 安装，再准备 Backend。未改核心隐式物化行为；这是已知后续缺口，不能用离线替身测试声称它已被修复。
- VoxCPM 固定参考 smoke 交付 19.13 秒音频，总耗时 24.24 秒（包含准备/物化，不是纯热推理 RTF），证明这条显式安装路径可运行，不代表整章听感。

## 检查

MOSS 无加速 feature 的 backends 库测试 26 项通过；Metal 专项 2 项通过；所选 feature 的 facade 库测试 9 项通过，包含取消不入库与资源接口。Python 批次测试 7 项通过，覆盖矩阵、扩展 WAV、无效音频、同名冲突、入库后导出失败恢复和超时。fmt、相关 Clippy/rustdoc 通过；CUDA 未在此 Mac 上验收。

## 候选交付

本次有效执行目录为 `target/voice-candidates/20261031/review-v2/`；较早 `review/` 保留解析器/隐式物化失败证据。库候选 ID 保持不变，设计来源、参考音频原字节与 fingerprint 在恢复时核验。

最终 Qwen/VoxCPM 各 16 项，共 32 个候选全部完成设计、导出和共同后端克隆；另 16 个 MOSS 候选明确阻塞。64 个音频链接及 32 个音色包均通过完整性核对；参考总长 193.76 秒，试听总长 720.94 秒。最终状态保存在 manifest.json 和 audit.json。所有候选及转写听感均待人工核对，未更改用户默认旁白；本记录不构成完整小说整章验收。

机器实测为 Apple M4 Pro、24 GB、macOS 26.6.2。最终工程检查包含 CPU 26 项、Metal 专项 2 项、facade 9 项、Python 7 项，以及 Clippy、fmt、rustdoc 含 examples 与 OpenSpec 严格校验。参考转写目前保存设计输入文本，其实际逐字正确性与长听偏好仍待用户试听；未将这一人工环节标记完成。

## 后续主线修复

本记录中的“先安装、后准备”是候选批次当时使用的绕过方案。随后已修复 VoxCPM 的首次隐式物化目录刷新，并移除 example 中的提前安装步骤；两款用户选定参考均通过正常入口生成多段基线。新证据和仍待完成的听感验收见 [单音色主线首轮验证](../voice-drift/single-voice-mainline-2026-10-31.md)。MOSS 数值门禁仍未通过。
