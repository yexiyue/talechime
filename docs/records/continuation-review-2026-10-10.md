# 接续、参数、风格与回读改动审查

审查基线为 `ce79e82`，覆盖至 `8f62eb8` 的六个提交，以及本轮修复。
按 code-review 的 Standards / Spec 两轴分别审查，再按 simplify 做局部简化。

## Standards

没有确认的文档规范硬性违规。两项低风险启发式建议已处理：

- MOSS 的码本数量由 `Mode::codebooks()` 声明，参考音色与临时接续编码复用。
- 数值参数的整数扩宽由 `ParamValue::as_f64()` 统一，各模型保留校验、默认值与精度选择。

## Spec

以下八项已修复；严重等级描述修复前的影响。

| 等级 | 问题 | 修复与验证 |
| --- | --- | --- |
| P1 | VoxCPM 将缺省 style 判为非空风格，拒绝普通接续 | 先规范化 style；回归覆盖 None 与空白风格的接续 |
| P2 | Realtime 宣称支持 instruction/style，但推理拒绝 | 按固定官方格式撤回两项声明；Local 保留，测试能力及参数目录 |
| P2 | 参数错误预览在非 UTF-8 边界截断，引发 panic | 按字符边界截取最多 30 字节；中文和 emoji 的非法参数返回诊断 |
| P2 | 原生语速在 seek 后丢失运行时倍率 | 恢复执行保留当前有效速度；测试显式生成参数优先级与运行时改速后 seek |
| P2 | 旧库 update_settings 在原生语速上叠加播放变速 | 与 configure_playback 一样除以原生速度；测试仅改音量和改速度 |
| P2 | CLI 丢弃计划文件的 params/seed | 保留合并后的设置并传入会话；测试计划值、CLI 覆盖和缺省值，通用校验移至准备前 |
| P2 | MOSS 将 12.5 Hz codec 按 25 Hz 换算，时长上限翻倍 | 采用 24000/1920，默认 375 帧为 30 秒；测试解码样本数与时长上限 |
| P3 | Local 目录有 language，类型化构造器却缺少入口 | MossParams 补齐 language 的字段、方法与转换；Realtime 仍拒绝该参数 |

ASR 记录也明确：切换语言自动检测后，当前中文与非中文行为均需重新验收，不能用旧强制中文结论证明新路径。

## 验证

本轮普通测试没有下载模型。

- 默认 workspace lib/tests/examples：462 通过、6 忽略。
- 扩展 CPU（qwen/voxcpm/omnivoice/moss-candle/asr）：491 通过、6 忽略。
- 无默认 feature 的 talechime library 集成测试：17 通过。
- rustfmt、git diff --check、扩展 CPU workspace Clippy、Metal 组合 Clippy：通过，warnings denied。
- 扩展 CPU workspace rustdoc：通过，warnings denied。

日志位于忽略目录 `target/review-*.log`。没有重新做真实模型试听；此前 Local 初始化 OOM、
Qwen 1.7B Base 与 Vox BF16 未运行的验证边界仍然存在。

既有 `7705e1f` 的[原生 CI](https://github.com/yexiyue/talechime/actions/runs/38025399643)：
CUDA 编译与链接通过；Metal release 链接失败，缓存包含已不存在的 Xcode 15.4 路径及
ORT 缓存路径，导致找不到 clang_rt.osx。本轮本地 Metal Clippy 通过不替代该远端链接检查。
