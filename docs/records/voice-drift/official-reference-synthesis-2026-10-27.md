# 官方固定参考路径收敛

用户决定所有后端仅采用官方推荐路径，删除此前的滚动参考、双参考和身份锚定实验。实施变更为 `official-reference-synthesis`；此前 `single-voice-narration-experiments` 停止推进，未执行的实验不标完成。上游依据与模型策略见[固定参考合成](../../reference-synthesis.md)和[调研](rolling-context-research-2026-10-26.md)。

## 已实施范围

- protocol/core/facade 删除 `SpeechContext`、跨块 PCM 收集、重锚边界/预算、capability 和 start/options 字段。`SegmentRequest` 仅包含当前 text、voice、style、seed、params。
- 所有适配器删除从生成 PCM 重新建立 prompt/音色的分支。Nano 固定官方 voice-clone user-reference codes；Qwen Base 固定原始完整 ICL prompt；VoxCPM 固定单参考；OmniVoice 固定 clone prompt；Realtime 固定 reference codes。CosyVoice3 和 IndexTTS 原有固定条件保留。各自预置音色/自然生成与辅助音色设计流程不变。
- 删除 `experimental-dual-reference`、`experimental-moss-reference-anchor`，VoxCPM `DualReference`/`generate_dual` 和 MOSS 上一轮输入构造。模型单请求内部 KV、采样与 codec/流式历史不变。
- 删除 rolling/dual narration runner 和 `continuation_probe`，更新性能 probe 请求和归因字段。旧字段/CLI 参数明确拒绝，不保留兼容开关。
- 章节缓存 key 使用 `chapter-cache-v3-fixed-reference`，不会命中旧滚动音频。原录音、用户音色库、模型资源、小说和历史音频不删除或迁移。
- 既有语义分段、原文范围、门禁重试、取消、背压、播放检查点、静音裁剪及有界响度处理保留；不添加静音，不增加去噪/削头/调速策略。

## 代码与验收对应

| 要求 | 代码或测试证据 |
| --- | --- |
| 新请求不能携带前段 PCM | `talechime-core/src/backend.rs`、`session/producer.rs`；已删除 `continuation.rs` |
| 旧 wire/CLI 策略被拒绝 | protocol `removed_reference_strategy_fields_are_rejected`；CLI `removed_reference_strategy_flags_are_rejected`；能力 JSON 不再导出该字段 |
| 固定模型输入格式 | Nano `fixed_voice_clone_matches_pinned_official_builder`；其他适配器的固定缓存/模型原始路径与扩展编译 |
| 多音色、seek、恢复与门禁仍有效 | `fixed_reference_synthesis.rs`、`fixed_voice_style_assignments_survive_seek_and_recovery_in_both_strategies` 及既有会话测试 |
| 缓存和取消不失效 | core 章节缓存、满队列取消、暂存及 facade 生命周期测试；原生固定参考状态测试保留 |
| 开发工具不再启动旧策略 | 已删除 examples/runner/feature；performance fixture 与严格请求 schema 回归 |

扩展检查还修正了现有 `qwen_design_reference` 开发例程漏传的可选 seed 参数，使用 `None` 保持该例程的既有默认行为。

Metal 全量检查发现既有 Qwen 七 tap depthwise 卷积的矩阵乘法快路径测试错误要求与 grouped conv 逐位相同。800 个 F32 样本最大绝对差约 `4e-7`、RMSE 约 `6.34e-8`，生产卷积实现未改。测试保留 CPU 逐位检查，Metal 改为逐样本有限性及七 tap 加偏置的 F32 舍入界 `8*EPSILON*(1+abs(expected))`，三个形状均检查；单独 CPU/Metal 两项已通过。初次失败日志保留为 `metal-tests.log`，最终完整重跑另存 `metal-tests-final.log`。

## 验证状态

所有日志、构建和真实短样位于忽略目录 `target/official-reference-synthesis/`。

- 默认 workspace lib/tests/examples：790 passed，0 failed，41 ignored。
- 全部后端 + ASR + performance-tools 的扩展 CPU feature 回归：852 passed，0 failed，42 ignored。
- 全部可用 Metal feature 的 workspace all-targets Clippy：通过，`-D warnings`。
- 全部可用 Metal feature 的 workspace lib/tests/examples 最终回归：870 passed，0 failed，60 ignored。修正测试后另跑 Qwen Metal tests Clippy，通过。
- CI 的 12 种独立 feature Clippy、3 种最小 CLI 测试组合、2 种 MOSS catalog 检查、扩展 CPU Clippy、包含 private items/examples 的 rustdoc（`RUSTDOCFLAGS=-D warnings`）、CLI build、fmt/diff 检查均通过。Python 架构/发布配置测试 16 项通过；系统 Python 3.9 缺少 tomllib 的首次环境失败保留记录，改用已有的现代 Python 后通过，没有修改脚本或安装依赖。
- 最终加强的 seek/recovery 回归分别核验两种播放策略在初次生成、seek 和恢复后的完整新请求序列，而不是匹配此前已发生的调用；单项重跑通过。
- `source-audit.json` 检查 425 个 Rust/manifest 文件，旧滚动/双参考/锚定类型、方法、feature 均不存在，旧 runner 已删除。已构建 CLI 的 help 不含旧开关，三个旧参数均返回 unexpected argument；协议回归另验证旧字段拒绝。
- 普通测试未下载模型或读取小说；ignored/env-gated 测试的计数不是模型已执行证明。

## 真实模型入口检查

机器为 Apple M4 Pro / arm64 / macOS 26.6.2。使用最终 release `performance_probe`，binary SHA-256 为 `c1c71c2b84bb9efbc3c214c49deed11e32a777d4b9a683f12870fb1fa181fd39`。资源经生产 manifest 校验；仅使用已有本地权重，自写两段中文，不使用小说或新下载。模型串行执行，生成时无并发编译。

| 条件 | 实际输出 | 核验 |
| --- | --- | --- |
| Nano Candle / CPU / Weiguo 固定预置 codes | 4 段，23.60 秒 | 两次完整 round 均成功，完整 PCM 摘要完全一致 |
| VoxCPM2 Q8 / Metal / 先前默认生成录音冻结为固定导入参考 | 2 段，16.90 秒 | 两次完整 round 均成功，完整 PCM 摘要完全一致 |

VoxCPM 的参考安装在实验私有资源副本中，没有写入用户音色库。所有音频资产大小及摘要逐一核验，prepared capabilities 不含滚动能力。`smoke-results.json`、`smoke-verification.json` 与两个事件流保留请求、资源、输出和终态身份。模型目录内原有 89 个用户音色文件的前后摘要完全一致，用户音色库也保持原状。

这些结果证明上述真实模型路径能正常结束并复现，不证明逐字无误、听感通过或整章稳定。Qwen、OmniVoice、CosyVoice3、IndexTTS、Realtime 本轮完成代码/feature/相关普通测试检查，没有重新生成各模型的真实长篇音频。

本轮不会把官方推荐输入模式等同于长篇听感通过，也不把 Mac 编译或短样推断为 Windows/Linux/CUDA 验收。源码尚未提交、推送或发布。
