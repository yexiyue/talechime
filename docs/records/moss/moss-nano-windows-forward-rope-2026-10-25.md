# MOSS Nano Windows：forward 内交错式 RoPE 表复用

保留Windows CUDA的forward内交错式RoPE表复用。短句/段落正式整链改善成立，
段落首PCM仍inconclusive。Windows11 /
Ryzen7 9700X / RTX5070，Candle0.11.0 fork2266099d、CUDA13。证据根
`D:/workspace/talechime-benchmarks/windows-nano-20261024`。

## 与旧原型、1.7B 的区别

Nano权重revision44502f80dbf9743528fa921cc544d662c685ebec，现有原生路径将官方
BF16 checkpoint转换F32计算，不改变此精度策略。全局GPT2为12层、12头、head_dim64，
local仅1层；两者使用interleaved RoPE、base10000，与1.7B的非交错/1e6参数不同。
不扩大预计算长度，不改变outer-product矩阵形状；此前未通过的宽表原型保持撤回。

只在Windows CUDA每次forward内共用positions/frequencies/cos/sin，Q/K及本次所有
层共用同一表。没有跨请求缓存或环境开关；CPU/Metal仍调用原始rope。global24份
表降为1份，local每次2份降为1份，不套用Realtime4层local的计数。

先独立原型，180组Q/K输出逐位相同：长度1/2/12/128/226/269、offset0/1/15/226/
4096、F32/F16/BF16。单进程局部12层decode批次均值约1.09–3.01ms→0.20–0.55ms，
226-token prefill约3.08–3.22ms→0.63–0.82ms；单层local约0.250–0.255ms→
0.140–0.160ms。均包含建表；4批次/臂，不能当作独立进程整链收益或运行阶段占比。

原型通过后才将前一案例的模型私有RotaryTables扩展为crate可见，并添加明确的
apply_interleaved入口。非交错apply的数学未变；没有抽成独立crate或泛化图框架。
共享助手新增第二个真实消费者后，旧消费者门禁也重新运行。

## 真实模型与状态

Nano模型+F32 codec，使用固定official-seed-aligned-run/process-0/input.json提示词，
三seed3560097011/42/7，分别42/29/32帧、322560/222720/245760个raw interleaved
PCM样本。原路径→共享表→原路径的codes及raw PCM位模式完全一致，generate正常
自然EOS返回；计数断言共享臂确实建表、原路径臂不建表。输入身份按上述提示词文件固定，
不能直接用不同seed或请求的样本数差额判断生产路径少生成。

共用同一表对象的Local/Realtime/VoiceDesign共9组真实输入重新通过，原有非交错
路径codes/PCM不变；两个布局的primitive共480个输出逐位相同。四项显式模型门禁
均通过。Nano后端未消费/实际满队列取消与拒绝未知音色后恢复，222720个PCM样本
逐位保持；此拒绝发生在推理前，不冒称验证了推理中失败恢复。

普通后端28通过1忽略，模型CPU12通过；模型/后端CUDA release all-targets Clippy、
rustdoc -D warnings通过。既有LNK4098警告保留，不隐藏、不改依赖/CRT。

## 同期生产探针

A `forward-rope-reference-build.json`：
`621bbca5ebc7c9858904ade16e84296498f4eedb2fb94b0eb8d4642c1646cef6`。
B `forward-rope-production-build.json`：
`82c7e09359fc9617255ce327eb2bc4d790b217d5505388b7c88be3902a639ecf`。
同feature moss/moss-nano-candle/moss-nano-candle-cuda/performance-tools；A只覆盖Nano
Transformer文件为候选前版本，共享助手及其他模型实现相同。复制源码后显式更新
mtime，确认两臂模型均重编译且SHA不同，再按摘要还原候选，避免上轮Cargo缓存陷阱。
127份运行时源码快照与参考覆盖文件保留；生成/统计/比较运行器为冻结dd22ce7e版本。

## 正式生产整链与保留决定

短句和段落各ABBAABBA、每进程1预热+5热轮，共16个串行独立进程。两种长度交错，
期间不编译或运行其他模型；原始请求、seed42、Weiguo、资源、观测与device sampling
配置保持。96轮success/cleanup通过，无删轮、重跑择优或改输出长度。表中范围是
4个独立进程热均值的min/max，不是置信区间。

| case | A合成秒，均值（范围） | B合成秒，均值（范围） | 结论 |
| --- | --- | --- | --- |
| 短句 | 0.576622（0.564519–0.596769） | 0.464448（0.437824–0.489670） | −19.45%，范围不重叠 |
| 段落 | 2.609205（2.515211–2.707795） | 2.229709（2.126266–2.425247） | −14.54%，范围不重叠 |

音频分别始终3.36/16.08秒，RTF0.171614→0.138229、0.162264→0.138664。
短句首PCM0.058890→0.046508秒，−21.02%；A范围0.054118–0.063366，B
0.041442–0.053436，不重叠。段落首PCM0.052153→0.052725秒（+1.10%），
A0.040880–0.064240、B0.043085–0.061527，重叠，inconclusive。
6项比较均覆盖完整、输出等价，5项pass、段落首PCM inconclusive。保留依据为
可信局部及两种长度的完整请求收益、数值/状态门禁和较小维护成本，不强制固定百分比。

### 数据与质量

576个资产SHA/字节数验证；96轮full F32按原生摘要域独立复算：采样率u32le+
声道u16le+interleaved F32LE。有限值和时长通过。原始source-ranges文件不改，
比较时排除consumer时间戳，所有UTF-8范围、文本SHA与chunk样本边界相同。
短句PCM摘要d2e6692e13ef20a1364e0547dc3703c0f9705b985274285e307ef759469fd6e0，
段落104f58edf218780c0b1a0baec192b632c266d2174df9e923bdfb2279af042f18。

固定SenseVoiceSmall-int8 revision2365baeacb507f821a0c8120fcee3d484dba7a07，CPU4线程，
对首个B进程的短句和完整段落辅助回读，均与输入文字一致（忽略标点）；识别器不
接收期望文本。不是双ASR或人工试听，听感仍pending。没有把ASR情感标签当音质验收。

analyze_forward_rope_abba.py逐轮审核并输出summary。aggregate只复制原始Attempt并
重编号process_index0..3，保留run/attempt、Source、条件，记录原results SHA；
不能将aggregate目录当作原始资产root。比较运行器固定为dd22ce7efc81753b96533899f94e60bb0df465c661500ec92a59f4ab0babcb75，
原始off条件未重标记，policy absolute1e-6/relative0.001、4进程及输出等价要求明确。
两个selected-first-B离线报告仅各代表首个B进程，完整统计以4进程每臂为准；不是
选择最快报告来代替ABBA结论。

### 准备与容量

加载、资源SHA验证、独立短合成warmup分别记录在
forward-rope-preparation-memory-summary.json。reference_prepare为null；warmup
不是单独参考编码成本。Job提交峰值短句A最大1,667,485,696/B1,668,665,344字节，
段落A最大1,707,524,096/B1,717,948,416字节，不是RSS或显存。500ms整卡显存采样
包含其他桌面进程，不能推成精确进程VRAM峰值或独占设备性能。

### 复现与边界

forward-rope-abba-schedule.json和run_forward_rope_abba.ps1保留确切16进程序列；
脚本拒绝覆盖已有输出，复测需新输出根。单case用冻结job和新目录：

```powershell
& $runner perf run --job D:/workspace/talechime-benchmarks/windows-nano-20261024/forward-rope-short-A-job.json --output NEW_A_OUTPUT
& $runner perf run --job D:/workspace/talechime-benchmarks/windows-nano-20261024/forward-rope-short-B-job.json --output NEW_B_OUTPUT
```

与前次mask/embedding等历史收益不相加；新同期A已包含这些既有优化。原生A/B也不
消除官方随机流、生成时长和听感差异，不宣称已完成全部跨框架质量验收。
