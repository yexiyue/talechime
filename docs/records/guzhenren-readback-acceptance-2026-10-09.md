# 《蛊真人》章节回读验收（2026-10-09，已中断）

## 方法

基于主线 `7b9f3a3`，使用 TRNovel 本地 UTF-8 小说文件与既有章节目录，提取前五个完整章节，不包含卷前说明。章节边界分别为原文件字节 `685..10486`、`10486..19571`、`19571..26789`、`26789..36204`、`36204..42807`。合计 14,550 字符、42,122 字节；每章原始字节摘要在读取后再次验证，不修改小说内容。

原文件 SHA-256：`e9653ac316136efa982016ec4ef3397e3b0b6e576589429f81695cd4a0f45a51`。章节文本、报告、转写和音频仅保存在 Git 忽略的 `target/guzhenren-acceptance/`，不纳入仓库或发行包。没有修改 TRNovel 的小说、配置或阅读进度。

环境为 Apple M4 Pro、24 GiB、macOS 26.6.2、Rust 1.98.1，release 构建。TTS 使用 Qwen3-TTS 0.6B CustomVoice、Metal、默认福叔音色，原生参数保持默认，随机种子 42。TTS revision 为 `85e237c12c027371202489a0ec509ded67b5e4b5`。ASR 使用内置 Qwen3-ASR 0.6B 主识别与 SenseVoiceSmall INT8 复核，均 CPU；主识别有差异时才复核。模型清单与摘要以仓库 manifest 为准。

原计划覆盖五章：第一章执行 Off 基线，五章执行 ReportOnly，再对实际错误或代表性疑点定向 Gate 复测。实际完成第一章 Off，并完成第一章部分回读后，用户要求停止测试、先提交推送，由用户后续真人试听。因此第二至五章、Gate 和严格疑点门控均未运行。

通过公共 `Engine::synthesize_verified` 接口消费实际模型分段，逐段导出浮点 WAV。示例可选择 `off,report,gate,strict`，省略参数时为 `off,report,gate`；Gate 使用一次重试，strict 模式阻止疑点，其余门控默认交付疑点。仅选择 off 时不准备 ASR，便于单独生成试听材料。这些示例选项不表示本次已经实测了各模式。

首 PCM 指消费方收到首个音频项的时间，包含当次文本处理和校验，不包含模型准备。总耗时包含合成、ASR、报告收集和同步 WAV 导出；RTF 为此耗时除以实际交付 PCM 时长。回读证据绑定边界静音裁剪前的完整模型 PCM，导出 WAV 则是公共流实际交付的裁剪后 PCM，两者的音频摘要不能直接比较。门控片段不与完整章节吞吐直接比较。模式顺序固定，没有重复轮次或方差估计。

内存由外部进程每秒采样 RSS，并记录子进程峰值；RSS 不等于完整 Metal/unified-memory footprint。该路径是直接 PCM 消费，不打开声卡，也没有把生成完成当成实际播放完成。自动转写不能替代人工逐字试听；多音色、真实播放、AfterChapterReady 与恢复验收需分别标注证据。

## 复现

需先在忽略目录准备含 `id`、绝对 `file`、`sha256` 的 JSON 数组，以及固定模型资源；普通测试不会读取语料或下载权重。

```sh
cargo build --release --locked -p talechime --features asr,qwen-metal --example readback_corpus
target/release/examples/readback_corpus qwen 0.6b-customvoice metal \
  target/asr-spike/resources target/guzhenren-acceptance/corpus.json \
  target/guzhenren-acceptance/results report
```

输出目录每章、每模式分别保存 `summary.json`、`reports.jsonl` 与编号 WAV，顶层汇总在每组完成后写入。小说文本和报告中的原文不适合作为公开测试夹具。

## 结果

| 章节 | 字符数 | Off | ReportOnly | Gate |
| --- | ---: | --- | --- | --- |
| 第一章 | 3,387 | Completed，99 段 | 用户中断，69 份报告 | 未执行 |
| 第二章 | 3,139 | 未执行 | 未执行 | 未执行 |
| 第三章 | 2,498 | 未执行 | 未执行 | 未执行 |
| 第四章 | 3,245 | 未执行 | 未执行 | 未执行 |
| 第五章 | 2,281 | 未执行 | 未执行 | 未执行 |

第一章 Off 首 PCM 3.579 s，总生成与导出 1,433.382 s，实际交付音频 918.880 s（15 分 18.880 秒），RTF 1.560。最后交付范围结束于章节字节 9,796；余下 5 字节全为空白，没有据此漏掉正文。所有 99 个 WAV 已正常完成导出，独立解析确认浮点 PCM 有限且非全零，时长总和一致；这证明完整生成，不证明逐字读对或完整实际播放。ASR 模型也在 Off 测量前准备，内存不能视作仅 TTS 的占用。

第一章回读完成 69 份报告：34 Passed、35 Suspect、0 ConfirmedError、0 Unverified，识别证据错误为 0。最后完成的范围为章节字节 `6679..6815`，当次已运行 1,202.9 s；这不是整章耗时或通过率。人名同音字等替字差异被保留为疑点，不能仅凭识别转写认定实际发音错误。没有触发真实合成重试，固定种子对重试效果的影响仍未验证。

TTS 准备 12.498 s，ASR 准备 8.432 s，包含资源校验。整个已运行批次的子进程峰值 RSS 为 7.20 GiB，每秒采样最高 RSS 为 5.85 GiB；不包含完整 Metal footprint。终止码为 SIGTERM（-15），来自用户要求停止，而非观察到模型错误。续跑脚本和模型进程均停止，未把进程终止计作库内合作取消/close 验收。中断模式的最后一个 WAV 可能未写完头部，不作为完整试听素材。

供真人试听的完整第一章来自 Off 的 99 段裁剪后 PCM，按序无额外静音拼接为 PCM16 WAV：本地 `target/guzhenren-acceptance/chapter-0001-listen.wav`。原始分段、报告和日志一并保留在忽略目录。尚未人工试听；本轮也没有验收 TRNovel 播放接入、多音色、AfterChapterReady 或恢复，因此不能宣称五章验收通过。

提交前检查：新增示例 release 构建通过；最终 `talechime` 包启用 `asr,qwen-metal` 的 lib/tests/examples 共 45 项测试通过，Clippy `-D warnings`、rustdoc `-D warnings` 和格式检查通过。没有变更生产推理代码或模型参数。
