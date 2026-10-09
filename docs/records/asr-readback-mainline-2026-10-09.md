# ASR 回读主线交付（2026-10-09）

## 选型决定与交付范围

首版内置 Qwen3-ASR 0.6B 主识别、SenseVoiceSmall INT8 跨家族复核，两者 CPU。0.6B/1.7B 在本轮 16 例 Transformers 归一化转写相同，较大模型没有显示足以抵消内存和延迟成本的收益；同家族大小变体也不能充当独立复核。SenseVoice 速度快、识别行为不同，因此采用此组；不把 ASR 模型名写入合成策略 API。权重来源、revision 与 SHA-256 分别固定在 `crates/talechime-backends/src/asr/qwen06.json`、`sensevoice.json`。

Whisper large-v3-turbo 已进行 CPU/MPS spike，保留候选：这组小样本未显示默认主筛查替代 Qwen 的收益，且贪心对照在静音/噪声出现幻觉。本实验没有完整 VAD/no-speech 门禁，不能将其当作 Whisper 官方默认用法的质量结论。FireRed Rust 前端等价仍待解决，暂不交付。候选细节见[首阶段 spike](asr-spike-2026-10-09.md)。

现已交付：独立 `Verifier::report`、显式 ASR 准备/复用、按次 off/report/gate、实际模型片段先校验后交付、保持原文/音色/风格的有界重试、Streaming 与 AfterChapterReady 共用门禁、证据缓存、取消/关闭与原生任务等待、CLI verify/开关、v7 计划可选校验字段和事件。标准包增加 `asr`，运行策略仍默认关闭。模型独立于 TTS 准备，调用中不隐式下载。当前 API 与限制见[使用指南](../readback.md)。

## 原生适配

Qwen Candle 源码来自固定 MIT revision（见 `crates/qwen3-asr/SOURCE.md`）；使用工作区 Candle 0.11.0，映射官方 HF 配置和权重名称，CPU BF16/F16 明确转 F32，周期 Hann、EOS 必须到达、encoder/decoder 层及 token 步进检查取消。移除了 upstream hub、流式缓存、设备 fallback、对齐与 unsafe Send 声明。不存在 ASR 时间戳对齐功能。

SenseVoice 使用工作区 ORT rc.13/api-21、kaldi-fbank 0.1.0 与 Rubato 重采样，不链接 sherpa C++。前端按模型 metadata 校验 LFR7/shift6、560 维 CMVN，16kHz/80 mel、Hamming、dither=0、snip_edges=true、高频边界0及 PCM ×32768，CTC 去重/blank/标签剥离。原生模型在各自 owner 内创建、使用、销毁；请求队列容量1，ORT RunOptions 可终止、Qwen 合作取消。全零音频直接空转写，防止生成式静音幻觉。

固定数值夹具用 kaldi-native-fbank 1.22.3 独立生成 640 个整数锯齿样本的两帧 80 维特征，Rust 逐项绝对误差小于 1e-3。它验证前端数值，不代表完整模型的所有运算等价。源码 MIT、Qwen 权重 Apache-2.0 模型卡、SenseVoice 独立 FunASR 权重许可分别保留；权重不进入 Git 或发行包。

## 确定性检查

普通测试不下载模型。门禁夹具覆盖：报告先于 PCM、失败尝试不交付、同片段/音色/风格重试、耗尽/ASR失败/超时、疑点默认交付与 strict 阻止、未准备与容量拒绝、取消/复用、AfterChapterReady 后段失败整章不播放、缓存绑定实际音频/音色/风格且失败不缓存。原文 UTF-8 差异映射、繁简、全半角、明确数字及歧义另有 core 测试；预处理回落整段时保留 normalized_range，避免不同位置漏同词被错误确认。

验证使用扩展 CPU workspace 测试、Clippy `-D warnings`、rustdoc `-D warnings`、纯库测试/ASR 独立 feature、格式检查与 Python 对照工具 Ruff/py_compile。cargo-dist 0.32.0 已按 dist-workspace.toml 重新生成工作流，generate --check 与 plan 通过；生成结果无需手工编辑。当前机器是 Apple M4 Pro /24GB，其他平台/GPU ASR 实机未验证，CI 编译不能替代这类验收。

## 实测方法与边界

真实探针使用 release 构建。同一份 16 例语料包括 11 个已有 TTS 原始片段和 5 个粗粒度人为变体，来源和限制沿用首阶段 spike。语料、模型、运行输出保留在忽略的 `target/asr-spike/`，不将已有原始片段当作人工逐字确认的正确音频。

先记录两个真实 ASR 的独立转写和耗时，再把记录注入 production Verifier 比较，避免计时额外跑一遍模型。生产默认只在主识别有差异时复核。RTF 是识别耗时/音频时长，取 11 个原始片段中位数，不含加载、TTS、报告收集、播放器；不是端到端吞吐或多人并发保证。未经优化的 debug 构建显著较慢，不用于性能选型。

原生与 Python CPU 对照按 NFKC/大小写/标点归一化比较：Qwen 15/16 相同，唯一差别是明确全零保护；SenseVoice 15/16 相同，银行“行长”例不同，未将差异猜测为某一已确认根因。SenseVoice 完整数值等价仍未建立，这个案例在保守规则下留为疑点。不能宣称所有原生输出与 Python 等价。

实测输出 `native-readback-release.json`：16/16 例原生转写完成。11 个原始片段中 6 个 Passed、5 个 Suspect、0 个 ConfirmedError。尾部截断、中间删除、整段重复、静音均 ConfirmedError；噪声为 Suspect（Qwen“嗯”、SenseVoice 空文本），默认疑点策略会交付且记录，strict 可阻止。4/5 人为变体得到确认，不能推断自然单字/漏句召回率或“原始正确音频误报率”。

release CPU 原始片段中位数 RTF：Qwen 0.2803，SenseVoice 0.0130。SenseVoice intra-op 4 线程；Qwen 使用 Candle CPU 默认线程池。这一轮加载/文件完整性验证 4.11s。Qwen 在真实请求 1 ms 超时后合作取消并 settled，用时 0.052s，随后同一 owner 完成全部 16 例；超时不是原生已经退出的保证，显式 settled/close 会等待它。

另用 `verified_synthesis` release 示例，真实 Qwen TTS 0.6B CPU 合成“你好，世界。”，通过内置主 ASR，随后直接流交付 34800 个 PCM 样本、1 份 Passed 报告，状态 Completed，Engine 关闭成功。没有打开播放器或创建检查点。真实样本这次没有发生合成重试；定向重试和整章失败禁播来自上述确定性测试，不当作真实 TTS 故障修复率。

独立 CLI `verify` 用同一尾部截断 PCM16 WAV 输出 ConfirmedError JSON，未准备 TTS、未播放。扩展 CPU workspace 共 420 项通过（无模型下载），纯库 17 项通过；格式、Clippy、rustdoc、Python 检查及 dist generate/check/plan 通过。
