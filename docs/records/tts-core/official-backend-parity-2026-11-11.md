# 固定后端能力补齐验收（2026-11-11，进行中）

实施入口：[finish-official-backend-parity](../../../openspec/changes/finish-official-backend-parity/tasks.md)。
macOS Apple Silicon / 24 GiB，Candle 使用工作区固定 fork。代码未发行。
本记录只收录已复核证据；全变体矩阵仍未完成，不能声明全部官方能力补齐。

## IndexTTS front-v6

固定模型及前端来源见 [SOURCE](../../../crates/indextts/SOURCE.md)。日语 14 条、
两套平台中英各 20 条、西语 16 条独立官方前端 fixtures 通过；QwenEmotion
12 条输出转换及 4 条真实固定模型分类通过，提示词 token 和八维向量相同。
真实分类对照为 CPU F32，其他设备及逐 token/logit 等价没有由该结果替代。
资源完整准备及无下载离线复验通过。

Metal 完整公开文本矩阵 19 行中 18 行生成，韩语参数明确失败；固定模型仅支持
zh/en/ja/es/ar，该失败单独保留。三个日语样本包括无假名日期与假名读音标注，
有完整来源范围、执行文本/token、正常 API 完成及有效 PCM。模型原生 EOS
没有独立从 API 暴露；报告不把 End 改名为模型 EOS。

相同公开英文、相同参考及 seed 的控制样本：

| 控制 | 参数 | 音频长度 |
| --- | --- | --- |
| 快 | duration_factor=0.7 | 2.229 s |
| 慢 | duration_factor=1.3 | 4.145 s |
| 文本情绪 | use_emo_text=true，公开快乐描述 | 3.947 s |
| 显式向量 | happy=0.6、calm=0.2 | 3.866 s |
| 独立情绪参考 | 同一公开参考，alpha=0.7 | 3.193 s |

五项均有有效 PCM 和正常完成。时长结果与倍率一致；表达、相似度和读音待人工听验。
旧 Metal 情绪用例因 F64→F32 不支持而失败；修复标量创建后重新生成，旧失败未覆盖。
本机忽略目录：`.local/matrix-front-v6-verified/`、
`.local/matrix-index-controls-v6-fixed1/`，含机器报告及 `listen.html`。

## CosyVoice3 front-v4

25 条固定官方公开 WeText 路径完整前端 fixtures 通过，非专有 ttsfrd。
Metal ZeroShot 19/19 生成；CrossLingual 和 Instruct 各 4/4 生成，
覆盖公开混排、英文、日文和阿文探针。阿文不据此认定属于模型受支持语言，
尤其短音频不能证明文字读出；全部声音仍待人工听验。

真实测试首先捕获 CrossLingual 缺少官方目标前缀、Instruct 在 Metal
创建零字节可选提示 tensor 失败；按固定 example.py 加入目标前缀，
以有效存储的零长度视图表达空条件后重跑通过。显式 Metal 空条件测试通过。
前缀控制标记按官方跳过 TN；参考转写与跨语言正文的模式边界见 [指南](../../cosyvoice3.md)。
原始失败保存在 `.local/matrix-cosy-{cross,instruct}-v4/`，
后续声音在 `.local/matrix-cosy-{cross,instruct}-v4-fixed1/`。

固定旧官方数值 fixture 的 front-v4 CPU 复核仍失败：estimator max-abs
0.003486 > 0.003，tokens→PCM 0.16024 > 0.01；其他阶段通过。
不修改容差、历史结论或将文本前端通过当作整链声学数值等价。

## MOSS 与边界验证

Nano 固定官方 prompt fixture 通过；Realtime 使用实际固定 tokenizer 的
六条独立官方 processor/prefill 输入 fixture 通过，覆盖无/有参考及 3/12/16 token。
实际生成路径使用同一构造器；提前 EOS 不允许接受尚未消费完的文本。
Nano 应急分段不截断拉丁单词或数字表达式，超预算的不可分表达式明确失败。

Nano 真实 CPU release 生命周期测试通过：未消费即取消、首 PCM 后满队列取消、
非法音色之后均能复用同一 owner，固定 seed 的 222720 个 PCM 样本逐位一致。
debug 初次沿用 CUDA 的 60 s 测试预算超时，保留失败记录；CPU 预算独立设为
300 s，release 实测 5.32 s。此结果不是性能基准，也不修复用户确认的漏读和年份音色问题。

## Qwen 逐变体与参考语言

CustomVoice 0.6B 与 1.7B 各 19/19，Base 1.7B 19/19 的 Metal 公开矩阵已生成。
Base 0.6B 当前探针重跑为 18/19；Auto 条件下全角数字触及帧数上限，未产生完成音频。
中文参考的四项附加探针中，混读和英文生成成功，日文 Auto 与全角数字显式中文仍
触及帧数上限（2/4）；保留失败，不用中文参考推断年份或多语言已改善。
Base 1.7B 中文参考的混读、英文、日文三项生成成功。
CustomVoice 两个规模的 Eric、Dylan 显式中文方言探针共 4/4 生成；方言听感仍待确认。
VoiceDesign 资源仍在准备，不能推广上述结果。

本机目录：`.local/matrix-qwen-custom06-remaining-v1/`、
`.local/matrix-qwen-custom17-v2/`、`.local/matrix-qwen17-v1/`、
`.local/matrix-qwen-base06-v9/`、`.local/matrix-qwen-base06-zh-v9/`、
`.local/matrix-qwen-base17-zh-v1/`、`.local/matrix-qwen-dialect-v9/`。

MOSS Realtime 公开矩阵 19/19 已生成。Metal 两项真实生命周期门禁通过：
固定参考缓存复用、立即/满队列取消、实际帧数上限失败不发 End，复用后 51840 个 PCM
样本逐位一致；测试合计 146.14 s，不作为性能基准。CUDA 本轮未执行。
该结果不替代 Nano 的漏读和年份音色失败。

## VoiceDesign 输入复核

MOSS VoiceGenerator 最初四条官方 prompt 对照全部失败：多加 audio-start 行，且缺少
官方鲁棒性规范化。移除多余行并移植固定 normalizer 后，四条完整 token 行逐项一致；
61 条官方自带/公开矩阵/Unicode 清理用例通过。固定 Unicode 13 C 类别范围，
避免 Rust 新版 Unicode 将官方参考环境视为未分配的字符保留；实际规范化不依赖系统
Python。规范化不做数字日期语义展开，只属于
VoiceGenerator 所需前端，不影响 Nano 后置读法问题。真实权重仍在准备，不能把
prompt 对齐当作声音已通过。

## Kokoro 独立未知词后备

用户已批准独立 eSpeak NG 1.52.0 程序、数据、GPL 许可及对应源码随包分发；
Rust 不链接 eSpeak 库，本机动态链接列表和 Cargo 依赖树亦无该依赖。
固定 bundle 原生构建与离线摘要复验通过；普通词典/形态规则不启动子进程。
缺件、清单错误、失败退出、超时和取消有独立边界测试，未知词通过 stdin 加 LF 输入。
Kokoro 单元测试 24 通过、3 忽略；显式真实前端六项测试通过，含实际默认 bundle
发现、三个 OOV、普通版和 g2pW 混读、全角百分号拒绝。

普通版/g2pW 各 11/13 官方混读音素相同；剩余两类是带空格年份产品规则和未知字符
明确拒绝，不能报告 13/13 官方一致。完整 Metal 文本矩阵共 38 行：24 生成、8 命中
预先登记的拒绝、6 失败（emoji、读音标注、全角百分号，各变体一项）。全角百分号的
独立官方 fixture 本身含未知音素 `❓`；保留实际失败，无事后改标成功。
声音在 `.local/matrix-kokoro-espeak-v11/`，年份听感历史失败不被新生成覆盖。
公开推理 revision 已含固定 eSpeak 源码身份，v3 权重目录保持原资源身份。

VoxCPM Q8 的参考克隆、风格、两者组合和设计 API 各三条公开文本，共 12/12 生成，
位于 `.local/matrix-voxcpm-controls-v11/`；用户确认的年份读法问题继续保留。
原始 BF16 的公开适配器要求 CUDA，本机不能代替该实机门禁。

完整 macOS 标准配置（含 ASR、Metal）的 cargo-dist 归档和 shell 隔离安装已通过；
程序版本/模型目录/v7 协议、归档 SHA、安装前后 helper 及对应源码摘要和 CLI 探针均通过。
首次构建发现 ASR 门面遗漏共享 Downloads 参数，修复后重跑；早期关闭 ASR 的实验
不替代这次完整配置验证。Windows 和真实 Homebrew 安装仍未实机运行。

## 宿主复核

CastGlean 当前章节执行把来源范围的原文直接交给引擎；其文本语义常量能阻止
未知旧任务续跑，但不能识别同名模型资源/front-end revision 的改变。
已将该缺口登记到宿主 `backend-owned-speech-text` 规格及任务 1.4；
没有修改其并行业务实现、数据库或用户音频。宿主升级身份门禁仍未完成。


用户最新约束：不要求兼容旧格式，旧音频缓存可清理并重建；不为旧音频增加兼容路径。
该授权不把历史听验失败改为通过，本轮机器证据和公开样本保留供复核。

## 2026-11-12 本机续验（保留此前失败）

下列新结果补充前述快照，不将旧失败改写为通过。

- Qwen Base 0.6B 的英文固定参考 + 显式 `language=zh` 全角样本生成成功；
  Auto 全角失败、中文参考两条失败保持独立。VoiceDesign 17 条 Auto 生成，
  韩文 Auto 失败后显式 `ko` 生成成功，长段命中预先声明的 100 字设计参考限制。
  显式 zh/en/ja 控制另有 3/3 样本。
- OmniVoice 完整 19/19 生成；固定中英参考、合法设计标签、显式语言、速度及
  steps/CFG 控制合计 21/21。早期把自由描述当设计标签的三次失败仍保留，
  正确标签为 `male, middle-aged, low pitch`，详见 OmniVoice 指南。
- MOSS VoiceGenerator F16 在实际 Metal 权重上五次出现无效 logits，余项取消；
  F32 尝试在 codec 加载时出现本机内存压力并取消。当前适配器改用 BF16：
  19 项均有结果，17 段生成，两项命中事先登记的 60 字设计参考限制。
  产物位于 `.local/matrix-moss-design-bf16-full-v15/`；该结果不代表 CUDA 或听感通过。
- Qwen/OmniVoice/VoxCPM 全公开语料条件 fixtures 分别扩展为 220/140/120 例。
  输入 token、embedding 或参考特征是合成值，证明固定官方 AST 的条件结构，
  不证明 checkpoint tokenizer 或声学数值等价。IndexTTS 18 条支持语料、
  CosyVoice3 全 19 条的官方前端对照使用实际固定文本资源，另有 Index 韩文范围拒绝。
  MOSS Realtime 38 条无参考/有参考 prompt 和 VoiceGenerator 19 条 prompt 使用实际 tokenizer。

用户明确将原始 VoxCPM BF16 的 19 条 CUDA 实机项另行安排。
完整矩阵仍保留这 19 行未验证；见变更的 `deferred-cuda.md`，不恢复 Tailscale，
不拿 Metal 设备拒绝或合成结构 fixtures 充当 CUDA 实机通过。

## 2026-11-12 本机交付

最终主矩阵 `.local/matrix-local-official-v19/`：15 个变体 × 19 条公开输入，
285/285，无重复或缺行；248 段有效音频、18 条运行前声明的范围/符号拒绝，
19 条原始 VoxCPM BF16 因本机无 CUDA 保留 blocked，按用户要求延期。
`acceptance.json` 单列设备延期及数值/听验边界，`by-variant.json` 给出逐变体统计。
当前试听总入口 `.local/listening-official-parity/index.html` 包含主矩阵、模式/参考/控制矩阵
及三个代表性片段；音频保留原生成目录，聚合与注释重新验证原 WAV 摘要、实际模型、
原文 UTF-8 范围及 PCM 分段，来源报告和失败不覆盖。

官方证据收据 `.local/official-corpus-checks-v19/receipt.json` 关联全部 285 项：
57 条实际 tokenizer prompt、152 条合成条件结构、37 条实际前端、18 条相同音素、
14 条符号拒绝、1 条语言范围声明及 6 条产品执行策略差异。
合成条件及参考 codes 不证明 checkpoint 神经数值或音频等价。
Kokoro 空格连接差异已修复；通用 eSpeak CLI 的功能词重音与库不同，改用独立
`espeak_TextToPhonemes` 程序后 20 个功能词/OOV 全部一致。两变体各 9 条语料音素一致、
7 条明确拒绝、3 条策略差异（空格年份、换行、连续空格）；新的 38 行真音频矩阵
为 24 生成、14 声明拒绝，早期 v11 失败及 CLI 对照失败 fixture 保留。
Nano 全 19 条原文使用实际 SentencePiece 和官方 prompt 对照通过，但这不修复
混读漏词和年份音色；5 条既有用户听验失败保持 failed，其他 280 项 pending，
其中拒绝/延期项没有可供听验的音频，没有任何自动听验 passed。

验证：六个模型库普通单元测试合计 452 通过、50 忽略；后端 all-Metal 单元测试
168 通过、17 忽略；loqui fork 15 项通过；文本/辅助包工具 37 项、安装器模板 6 项通过。
Nano/Realtime/VoiceGenerator 实际 tokenizer 及完整 Kokoro 前端门禁另行显式运行。
fmt、模型库与 all-targets Clippy、含 ASR 的门面 Clippy、rustdoc 通过。
完整 macOS 标准 cargo-dist 归档及隔离 shell 安装使用最终独立 phonemizer 包通过，
含归档摘要、程序/模型目录/v7 协议及 CLI/官方音素接口探针；完整对应源码和 GPL 许可
随 helper 安装。早期仅 CLI 的 helper 保留，不复用其不同协议。
Windows、真实 Homebrew 安装及 CUDA 实机不在本机通过结论内，未发布任何版本。

共享文本语义身份已递增为 `official-backend-text-v2`，Kokoro 推理 revision 为
`01e7505-runtime-f32-v3-text-v3-espeak-1.52.0-4870adfa25b1a32b4361592f1be8a40337c58d6c`；
权重资源目录不变，不加旧格式迁移。宿主仍需完成其任务 1.4 的资源/算法身份冻结门禁，
本轮只更新规格并保留并行业务代码。
