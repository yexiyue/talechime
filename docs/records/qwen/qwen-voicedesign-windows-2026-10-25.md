# Qwen VoiceDesign：Windows 生产导出与官方基线

Windows11 / Ryzen7 9700X / RTX5070，CUDA13/cuDNN9.14；Candle fork
2266099d92265112ecc0d56ecee1761e3cc2ac00。外部证据根
`D:/workspace/talechime-benchmarks/windows-qwen-20261024`。没有生产推理算法改动。

## 生产行为与正确性

测试调用实际`qwen::design::reference`，不是narration Backend。该入口每次独立
线程加载模型、流式收集完整音频、检查生成完成及参考合法性、保存WAV，再释放模型。
固定文本“你好，欢迎收听今天的故事。”，描述“年轻女性，声音清晰自然，用平静
温柔的语气讲述。”，入口原有seed42、chunk20、MAX_FRAMES及默认采样保持不变。

两次独立调用均完成，每份导出53,760样本、24kHz mono、2.24秒；重新读取导出WAV
的PCM逐位一致，WAV SHA均为
`f9f9c4c5d47cf00538ba0b19895a3e85a45c1ecf116cf7d3381e252c0604595a`。
入口原本保存clamp/32767转换后的PCM16；此测试证明导出一致，不能反推导出前F32
每个位也一致。两次测试总9.31秒包含加载与校验，不作正式性能数字。

独立ignored测试要求已有模型与全新TALECHIME_QWEN_DESIGN_OUTPUT目录；不下载模型，
不覆盖既有输出。冻结exe SHA
`41b4cdffae2620466ca9e8eda7895dabf948155f1e8e6fc785a7dfcce576bfe4`。
复现使用design-production-test-receipt.json的exe，设置资源根与输出目录，并指定
real_cuda_voice_design_production_repeat --ignored --nocapture --test-threads=1。

## 官方对照条件

权重revision `5ecdb67327fd37bb2e042aab12ff7391903235d3`，八份生产manifest资产
均按大小/SHA复核，官方目录使用同卷硬链接共享相同字节，避免重复占用4.5GB磁盘。
未改生产manifest；补充同revision官方speech_tokenizer/preprocessor_config.json，
SHA `fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb`，
下载响应另存。两目录均在~/.talechime/resources/qwen下。

使用已有固定官方checkout和python-pread隔离环境，Torch2.8.0+cu128、
Transformers4.57.3、Safetensors0.8，SDPA。沿用已记录的F32 codec加载覆盖并实测
断言codec dtype；主模型请求BF16，checkpoint使用pread。调用实际
generate_voice_design，同文本/描述、seed42、top-k50/top-p0.9/temp0.9、
repetition1.05、subtalker不采样、max_new_tokens375、non_streaming_mode=False。
保留随机实现、跨框架采样及生成长度差异，不以固定seed要求跨框架位一致。

xtask串行3独立进程，每进程1预热+5热轮：18轮全成功，cleanup全通过，
所有18份WAV SHA/字节数复核通过，原始F32 PCM摘要全相同。适配器额外检查官方
talker最终token为自然EOS，不能以预算耗尽冒充完成。

| 官方进程 | 五热轮均值（秒） |
| --- | ---: |
| 0 | 3.404274 |
| 1 | 3.475010 |
| 2 | 3.242053 |

总均值3.373779秒，音频2.64秒，RTF约1.278。官方批量API首PCM等于整段耗时。
Rust生产入口当前只有独立加载/导出正确性检查，尚没有同边界正式计时矩阵；不能
用测试总时间除二与官方热轮排名，也不能忽略2.24/2.64秒音频差异。

辅助SenseVoice-int8分别识别两份Rust导出和官方首热轮，均为“你好欢迎收听今天的
故事”，标点外内容完整；这是单模型辅助回读，不是双ASR或人工试听，不证明年龄、
性别、温柔/平静描述满足程度。listening仍pending。

## 工程门禁与未完成项

26项普通测试通过、6忽略；release相同feature组合all-targets Clippy与rustdoc
-D warnings通过。完整日志design-production-{ordinary-tests,clippy,rustdoc}.log。
VoiceDesign取消/失败恢复、匹配生产调用边界的计时、官方/原生阶段差距、人工试听
仍未完成。原有Qwen narration性能结论不由此改变。

## 后续：生产调用计时与官方冷加载控制

新增显式performance-tools示例`qwen_design_reference`，由xtask perf reference
负责进程、输入SHA、预算和清理。它直接调用公开design::reference，不复制Rust
模型推理，逐轮计时覆盖加载、完整生成、PCM16导出及模型释放。文件交付为首PCM
边界，文件重新读取及摘要审计在计时外。原生产固定seed42、chunk20、375帧预算
和默认采样不变。18份WAV全部与上述生产集成测试WAV逐字节一致。

冻结程序SHA `5f611af380232f02a074a9dd1b0866ec5dca83a8beee02a706ca7d7289d8bd94`。
Native三个独立进程，各1预热+5次完整生产调用，均值分别5.085184、5.159278、
5.790037秒；平均5.344833秒，音频仍2.24秒。每轮重新加载模型，因此这里的“热轮”
指进程/文件缓存预热后的API调用，不是常驻模型合成。Job峰值约9.27–9.43GB。

独立官方cold适配器每轮调用同一官方加载器与generate_voice_design，BF16主模型、
F32 codec断言通过，同文本/描述/采样/预算。匹配Rust的clamp后乘32767、截断PCM16
导出，计时包含模型scope退出、gc.collect和cuda.empty_cache。两次尝试均首轮
完整生成，但同进程第二次加载发生1455页面文件不足；保留失败目录，不计为完整
基线。新增weakref及分配审计证明首轮后模型对象已释放，CUDA active仅8,519,680
字节；reserved仍681,574,400字节，不能将它等同泄漏或凭此解释后续提交峰值。

随后仅采用三个独立新进程各一次冷加载，均自然EOS且cleanup通过：7.131148、
6.747006、7.073078秒，均值6.983744秒，2.64秒音频，Job峰值约10.154GB。
**这不是与Native对称的3×(1+5)矩阵**。官方CUDA预初始化在计时外；Rust首次
预热可能包含上下文初始化。矩阵、输出长度、分配器/清理行为差异均保留，禁止
据此计算统一提速百分比或宣布VoiceDesign已达到官方性能。此前常驻官方热基线
3.373779秒仍单独有效，不能被本节冷加载数字替换。

独立复核21轮PCM16 WAV的文件SHA/大小、解码F32LE摘要全部通过。两个路径各自
所有轮次输出一致；两个代表导出另做单SenseVoice辅助回读，人工听感仍pending。
新增example的release all-targets Clippy、含examples rustdoc -D warnings通过。
复现：先编译并冻结example，再由同一冻结xtask依次运行
design-cold-native-baseline-job.json及design-cold-official-independent-job.json，
输出必须使用新目录；原失败smoke与生命周期v2任务保留，不能覆盖重跑。

## 后续：一次性调用图策略候选撤回

继续使用同一5f611af3冻结程序、相同文本/描述/资源，先按四种CP/decoder图组合
各1进程、1预热+2轮初筛。每轮仍完整加载、导出、释放；GPU计时期间无编译或
并行模型。两图开重复均值5.175991秒；CP关5.747806秒；decoder关5.199117秒；
两图关5.913506秒。这是单进程筛选，不计算正式收益或外推其他模式。

各进程首次调用比后续重复调用更快；初筛仅关闭decoder的首次3.298594秒看似优于
默认3.739951秒。针对实际一次性用途，另做八个独立新进程ABBAABBA交错，**各仅
一次调用、无预热**；A为两图开启，B仅关闭decoder。它是冷首次调用复核，不能
标成普通1+5热矩阵，也不与此前5.344833秒重复调用均值拼接。

| 首次调用 | 四进程均值（秒） | 单进程范围（秒） |
| --- | ---: | --- |
| A 默认图 | 3.191348 | 3.115288–3.362790 |
| B decoder关 | 3.202175 | 3.104295–3.314137 |

范围重叠，关闭decoder没有可信收益，初筛单点差异未重现。未修改生产图策略，
不新增运行时开关或模型特例；已有两个诊断开关仅在隔离进程里切换。时间序列
出现下降，不能在未测时将原因指定为缓存、GPU升频或分配器。

20轮均成功及cleanup通过，文件大小/SHA与解码后F32LE摘要均独立复算，全部与
此前默认生产PCM16一致。此处仅验证导出保持，未额外证明量化前codes/F32一致；
由于候选已撤回，不以导出保持替代生产数值验收。job、顺序、原始结果与统计位于
design-graph-screen-*、design-first-call-*和design-graph-policy-summary.json。
