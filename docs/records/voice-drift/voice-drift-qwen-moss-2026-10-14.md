# Qwen 与 MOSS 接续策略对照（2026-10-14，第三轮）

## 状态与范围

用户在第二轮反馈 Qwen 音色稳定但语速逐渐加快，MOSS Realtime 音色变化且部分句子
维持首段慢速，随后授权进一步比较，并明确要求加入 Qwen 0.6B。
本轮新增八组真实模型音频，与前两轮同条件基线汇总为十三组。
未修改生产默认、CLI / 协议参数、文本分段算法或模型计算算法。

沿用同一《蛊真人》选段：1,032 字符、29 个固定分块、root seed 42，Scene 边界。
原文和派生文本均留在忽略目录；分块正文 hash 为
`d5823befc6e60fa5b932cd4eed87577e656de59058e8cce50c2969ae622bc1b9`。
此语料移除原始空白，以连续软换行分块，是压力测试，不能替代原始段落结构的整章验收。
前两轮方法和用户反馈见[前轮记录](voice-drift-experiment-2026-10-14.md)。

## 对照组

| 模型 | A：当前滚动接续 | B：替代策略 | C：每段原始参考 |
| --- | --- | --- | --- |
| Qwen3-TTS 1.7B Base | 默认 45 秒预算，复用第二轮 | 15 秒预算短链重锚 | 关闭接续 |
| Qwen3-TTS 0.6B Base | 默认 45 秒预算 | 15 秒预算短链重锚 | 关闭接续 |
| MOSS Realtime 1.7B | 默认 45 秒预算，复用第二轮 | 始终保留原始音色参考，同时滚动接续最近完整语音 | 关闭接续 |
| MOSS Nano Candle | 复用第一轮滚动基线 | 复用第一轮 15 秒预算短链 | 新增关闭接续 |

原 Nano 固定锚点对照额外保留为 D，不作为本轮主候选。
无空段落时，短链的 15 秒预算在约 30 秒以后、下一个分块才重置，
默认预算则在约 90 秒以后重置；不是每 15 / 45 秒精确重置。
所有组使用各模型原有默认参数，生成与播放均不做统一变速，无人工情绪指令。

Qwen 两种 Base 共用字节相同的原始旁白 WAV 及转写；选用 Base 是因为支持音频接续。
Realtime 使用第二轮已经适配好的同一完整参考内容（仅去掉开头 80 毫秒近静音）。
Nano 沿用 Weiguo，不能把其初始绝对音色与其他模型视为相同。
比较应先在同一模型内进行，再比较整体听感。

## 资源与实验实现

Qwen 0.6B Base 此前未在本机准备，本轮按生产 manifest 下载并逐文件核验大小和 SHA-256：
revision `5d83992436eae1d760afd27aff78a71d676296fc`，
`model.safetensors` 1,829,344,272 bytes，SHA-256
`180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6`。
与已有 1.7B Base 内容相同的 tokenizer 文件在摘要核验后复用。
网络单连接较慢，改用有界八连接 Range 下载；每段验证 HTTP 范围与长度，
组装后核验整个 pinned 文件摘要。模型和导入参考仅放第三轮独立目录。

Realtime 新策略使用显式 `experimental-moss-reference-anchor` feature，
探针旗标为 `--moss-reference-anchor`。Backend builder 仅允许 Realtime 与导入音色，
使用独立缓存身份；默认构造仍关闭。实验只改变适配器选择参考的规则：
静态参考继续进入 voice-timbre context，最近语音继续进入 previous-turn context。
原生 Realtime 已有这两个独立槽位，未改变权重、采样或 decoder。
格式依据[固定上游处理器](https://github.com/OpenMOSS/MOSS-TTS/blob/934d6826b084c46a0d033402174d5f8ac4ed2519/moss_tts_realtime/mossttsrealtime/processing_mossttsrealtime.py)。

Qwen / Nano 的新组使用第二轮归档探针；Realtime B 使用新 feature 构建。
这些组在同一模型内第一段 WAV 的 SHA-256 全部相同，确认初始参考和 seed 条件对齐。
共享参数与来源范围保存在各组 `command.json` / `summary.json` / `readback-corpus.json`。

## 生成结果与可作出的结论

| 模型与组 | 完整音频秒数 |
| --- | ---: |
| Qwen 1.7B A | 191.88 |
| Qwen 1.7B B | 218.48 |
| Qwen 1.7B C | 245.65 |
| Qwen 0.6B A | 202.25 |
| Qwen 0.6B B | 227.27 |
| Qwen 0.6B C | 247.90 |
| Realtime A | 278.57 |
| Realtime B | 259.48 |
| Realtime C | 283.26 |
| Nano A | 263.52 |
| Nano B | 261.87 |
| Nano C | 285.57 |
| Nano D，旧固定锚点 | 243.98 |

Qwen 短链相对当前续接总时长增加约 12–14%，独立生成增加约 23–28%。
这只证明生成节奏发生变化，不证明加速问题、段间连贯或情绪自然度已解决。
每组只有一个 seed 和一次运行，RTF 留在本地 JSON，不能据此评定小幅性能差异。

对前 10 段、中间 9 段、最后 10 段统计“原文可发音字符数 / 原始生成秒数”，
含停顿，不含 ASR 或语音活动检测。Qwen 1.7B 三组末段 / 首段平均比为约 0.97–0.99，
0.6B 为约 0.98–1.02。这一粗指标未表现出明显整篇单调加速；不同内容和周期重锚
可能掩盖句内 / 链内变化，不能据此否定用户试听反馈。

十三组均收到完整生成 End，29 段文字与来源范围一致，MP3 解码通过。
新增八组按每秒超过 10 字符的粗检查没有发现异常短段，仍未做逐字 ASR 验收。
旧 Nano A 第 18 段仍为 34 字符 / 1.28 秒；旧 Nano D 另外三段偏短：
第 4 段 45 字符 / 3.60 秒，第 20 段 37 字符 / 2.72 秒，第 22 段 35 字符 / 2.72 秒。
均标为疑似提前 EOS / 漏读，不以声音稳定或生成成功替代内容正确性。
本轮主候选的音色稳定、剧情快慢、句间连贯及内容正确性仍待用户试听。

## 产物与验证

产物位于忽略目录 `target/drift-experiment/round3/`：

- 十三组完整 MP3、相同第 14–22 段情绪片段及第 20–29 段尾部片段。
- 新增八组逐段 float32 / PCM16 WAV、完整 WAV、日志和统计；旧组复用前轮文件。
- `comparison-report.json`、准备 / 运行 / 比较脚本，以及仅含 MP3 和说明的试听 ZIP。
- Qwen 0.6B pinned 模型与其模型身份下的参考；没有修改已有模型或用户配置。

通过格式、diff 空白检查、CUDA feature 组合的 Clippy / rustdoc（warnings denied）、
8 项 MOSS Candle 无模型测试、未启用新实验 feature 的编译检查。
新增测试验证 Local 不得启用 Realtime 实验，以及无导入参考的默认 narrator 被显式拒绝。
普通测试未下载模型；本轮下载只由用户授权的真实模型实验准备脚本执行。
没有运行 ASR、声纹 embedding、跨 seed 或跨平台验证，没有推进实际播放检查点。
