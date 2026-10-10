# 其他模型的执行级接续

## 范围与官方依据

在 `ce79e82` 的接续生命周期上增加 Qwen 1.7B Base、VoxCPM2 Q8 / 原始 BF16、
MOSS Local 1.7B / Realtime 1.7B。OmniVoice、Qwen 0.6B Base 与 Nano 保持原有路径。
Qwen CustomVoice / VoiceDesign 不支持音频 prompt，能力标记保持关闭。

- [Qwen 官方 Base 接口](https://github.com/QwenLM/Qwen3-TTS)：两种 Base 共享 VoiceClonePrompt / ICL 路径。
- [VoxCPM2 固定源码](https://github.com/OpenBMB/VoxCPM/blob/f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629/src/voxcpm/model/voxcpm2.py)：prompt_text / prompt_wav 配对，左侧补齐 encoder 音频，转写与目标文字拼接；流式 decoder 只交付新 latent。
- [Local 固定处理器](https://github.com/OpenMOSS/MOSS-TTS/blob/934d6826b084c46a0d033402174d5f8ac4ed2519/moss_tts_local/processing_moss_tts.py)：联合转写 user 消息，无 reference，assistant 音频前缀；截断音频结束与消息结束标记。
- [Realtime 固定处理器](https://github.com/OpenMOSS/MOSS-TTS/blob/934d6826b084c46a0d033402174d5f8ac4ed2519/moss_tts_realtime/mossttsrealtime/processing_mossttsrealtime.py)：make_user_prompt 配对文字／音频，12 token 延迟、音频 BOS / EOS，再进入新 assistant 轮次。相同前段音频另作 make_ensemble 的滚动音色参考。

Realtime 是前一轮上下文条件接续，不声明它拥有 Local 的同一 assistant 音频前缀模式。
每请求重建上下文，不跨执行保留 KV cache。Local decoder 消费前缀但不交付前缀 PCM；
Realtime 从新轮次重置 decoder。临时编码留在 owner 的内存，不写入音色或参考缓存。
首段仍使用原来的参考／默认音色；错误不静默退回独立生成。

CLI、JSON Lines v7、库直接合成和两种播放策略沿用现有 continuation 开关。
只保留完整成功前段，1–15 秒、2048 UTF-8 字节、合计 16 MiB；软换行／同音色连续
span 保留，硬边界／音色或风格变化／失败／seek／新执行清空；门禁重试复用同一快照。
本次未改变原文范围、实际播放检查点或播放队列背压规则。

## 可重复验证

无模型测试验证两种 Qwen Base 的导入音色能力隔离、VoxCPM2 两种权重的能力目录、
Local assistant codes 与无结束标记、Realtime 短文本／12 token／长文本的完整行矩阵。
Realtime 夹具用固定官方 make_user_prompt 的 AST 提取方法生成，使用合成 token 与 codes；
不需要下载权重或执行官方完整模型。重建命令：

```sh
python tools/tts/realtime_prompt_reference.py PINNED_OFFICIAL_SOURCE crates/moss-tts/tests/fixtures/realtime-previous-turn.json
```

真实模型探针新增 `qwen-base17`、`voxcpm`、`voxcpm-bf16`、`moss-local`、`moss-realtime`。
使用模型 manifest 固定 revision，不自动迁移目录；模型、文本、WAV 和报告留在忽略目录。
新模型的主观听感尚未验收，不能把之前 Nano 的试听结论用于这些适配器。

## 本轮结果与限制

本机 Apple M4 Pro、24 GiB，Metal release 探针，固定种子 42，`narrator`。
语料是两行平静叙述，源摘要 `ce39348273162948dfadbe3bc2eb9408ced73ffd0a4653177416fd297ea22117`。
使用明确指定的既有资源根／revision 目录；没有模型目录自动迁移。

| 模型／开关 | 生成耗时 ms | 交付音频 s | RTF | 首段／后段首 PCM ms | 进程峰值 footprint GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| VoxCPM2 Q8 on | 11176 | 9.50 | 1.176 | 1269 / 796 | 4.923 |
| VoxCPM2 Q8 off | 7967 | 8.10 | 0.984 | 176 / 184 | 4.988 |
| Realtime on | 11418 | 9.87 | 1.157 | 541 / 776 | 16.671 |
| Realtime off | 8348 | 8.64 | 0.966 | 517 / 427 | 16.669 |

这是短语料生成记录，不是隔离性能基准；回读任务曾并行运行，Metal 首次编译与缓存也会
影响延迟，不能由此断言接续造成全部耗时差异。RTF 使用流交付的音频时长，逐段 metrics
另记录未处理原始 PCM 的时长。接续组第二段 `continued=true`，关闭组均为 false。
两模型首段 WAV 在开／关组 SHA-256 完全相同，证实首段行为未改变。

VoxCPM2 on 的两段回读均 Passed；Realtime 两组均为首段 Suspect、第二段 Passed。
两个 ASR 都在首段识别出“月”而非“夜”，开关组首段音频完全相同，不能归因为接续。
第二段未发现前缀重复／漏读。所有这些结论仅适用于本次短语料；主观听感未验收。

Local 完成 manifest 校验，但本机两次初始化（第二次单独重试）均失败：
`kIOGPUCommandBufferCallbackErrorOutOfMemory`。失败发生在接续请求之前，原模型加载
路径未改；没有生成音频，不记录 Local 真模型接续通过，不添加静默设备替代或不相关
权重加载重构。需要在显存／内存更充裕的平台继续运行 `moss-local` 探针。
Qwen 1.7B Base 本轮未准备真实权重，VoxCPM BF16 未进行 CUDA 真模型验证。

无模型默认工作区 443 项、扩展 CPU 459 项、纯库 29 项测试通过；官方格式对照与
最终 worker 能力目录 7 项协议测试通过。格式、Clippy（workspace 与独立 Metal feature
组合）、rustdoc（warnings denied）、无模型 CLI／协议握手通过。首次整章暂存测试在
release 链接并发时超时；构建完成后的默认与扩展整套重跑均通过，未改变播放实现。
CUDA 编译仍按原生 CI 检查，未宣称本机已验证 CUDA。

对照文件：`target/continuation-all/{voxcpm,realtime}-{on,off}/comparison.wav`；
每目录含逐段 WAV、原文范围、metrics 和 readback corpus。on 目录还有回读结果；
Realtime off 回读结果也保留。Local 错误日志为 `local-on.log` / `local-isolated-on.log`。
