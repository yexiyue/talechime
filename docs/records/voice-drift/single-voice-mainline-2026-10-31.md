# 单音色主线首轮验证（2026-10-31）

## 范围与选择

用户选定御姐第一款 `vcchar1031-yujie-cold`（冷艳低音）和萝莉／甜萌第一款 `vcchar1031-cute-soft`（软糯轻甜），暂缓继续扩音色。它们来自本地 Qwen VoiceDesign 原创候选，不是所有安装都具备的内建音色；未修改全局默认配置。

后续顺序见 [单音色路线图](../../single-voice-roadmap.md)。本轮先修首次使用，再生成无规划多段基线，不代表整章、人工听感或实际播放验收完成。

## 首次使用修复

后端加载时冻结的音色目录看不到随后由音色库物化的新参考，导致正常首次调用可能报 `voice custom:hero is unavailable`。先将测试替身改成冻结目录并复现失败，再添加元数据刷新钩子：直接合成及计划构建在物化新音色后通知后端刷新。

VoxCPM 实现只更新音色 ID 和显示名，完整读取成功后才发布；损坏目录返回错误并保留旧快照，不重载推理模型。本轮原生实现与实机验收范围为 VoxCPM，其他后端仍需逐项检查。

## 无模型验证

- 核心库 78 项、后端 48 项、门面库 9 项测试通过，共 135 项；包含直接合成／计划入口的冻结目录回归，以及 VoxCPM 后安装音色和错误原子性检查。
- `cargo fmt --all`、差异空白检查通过。
- 选定 CLI、Qwen Metal、VoxCPM Metal、MOSS Candle Metal feature 组合的全目标 Clippy（`-D warnings`）、rustdoc（`-D warnings`）通过，release CLI 与基线 example 编译通过。
- 普通测试未下载或加载模型。MOSS 编译通过不代表其 VoiceDesign 数值问题已解决。

详细日志保存在忽略目录 `target/single-voice-mainline/20261031/`，包括失败复现 `regression-before.log`、通过结果 `tests.log` / `core-tests.log`、`clippy-final.log`、`rustdoc.log` 和 `build.log`。

## 真实模型基线

设备：Apple M4 Pro / 24 GB，macOS 26.6.2。使用现有 VoxCPM `2b-q8_0` / Metal / seed 7，同一份原创六段文本，无 LLM 规划，无 style。通过正常库入口隐式物化，不再提前显式安装作为绕过方案。

| 参考 | 生成音频 | 合成耗时 | 后端加载时已有音色 |
| --- | ---: | ---: | --- |
| 御姐 · 冷艳低音 | 90.68 秒 | 93.04 秒 | 否 |
| 甜萌 · 软糯轻甜 | 91.83 秒 | 92.11 秒 | 否 |

两次调用成功输出 48 kHz WAV，证明这两款参考的 VoxCPM 冷首次使用链路可用。正文 SHA-256：`57b40cf372ef5bb047cc3e4e486bd2956283437a684eb7c64b0928dba49eecbd`。

运行目录的 `selected.json` 记录参考指纹；`baseline.json` 记录参数、音频摘要与耗时；`index.html` 提供参考音频和连续朗读对照。模型、参考音频和生成音频均未提交仓库。

这些证据仅证明生成完成。未执行实际扬声器播放完成验证，也未完成漏读／重复、长段声线稳定性、衔接和表达的人工验收；六段样例不能替代整章结论。接下来审计组段与风格边界，再做人工规划对照和完整章节验证。
