# MOSS Windows CUDA：forward 内复用 RoPE 表

保留Windows CUDA的forward内RoPE表复用。Realtime正式整链收益成立；Local仅有
局部收益和范围重叠的整链点估计，不宣称其稳定整链加速。
设备Windows11 / Ryzen7 9700X / RTX5070，Candle0.11.0 fork2266099d，CUDA13。
证据根 `D:/workspace/talechime-benchmarks/windows-moss17-20261024`。

## 重复工作与候选范围

Local/Realtime全局Transformer均28层、head_dim128、theta1e6；每层Q、K分别重新
构造positions/frequencies、相同形状的outer-product、cos/sin及dtype转换，一次
forward重建56份相同表。Realtime深度Transformer的4层也启用RoPE，每次深度
forward重建8份。Local的深度Transformer没有RoPE，不套用此项归因。

`transformer/rotary.rs`新增不可变表，仅在Windows CUDA、启用rotary的同一次
Transformer.forward内构造一次。每层由同一个config建立，长度/offset/维度/theta/
dtype/device相同；保留原有GEMM形状与旋转实现，没有扩大预计算长度、跨请求缓存、
精度变更或CUDA graph。表随本次forward释放，取消前先检查，offset/reset语义保持。
原`rope`函数不改，CPU/Metal及Nano/codec调用原路径。音色设计的共享Transformer
也共用这一实现，故已加入真实模型正确性门禁，不把它遗漏为“只是Local/Realtime”。

## 数值与状态门禁

独立原始`rope`为oracle。长度1/2/12/128/256、offset0/1/15/226/4096、维度64/128、
F32/F16/BF16、Q16头/K8头，共300个CUDA输出逐位一致。单进程局部计时每臂4个批次，
每批12次28层Q/K操作，计入每次forward建表，GPU只在批次边界同步；约3.4–4.3ms
降至0.43–0.52ms。这是局部筛选，不是独立进程整链统计或实际运行阶段占比。

真实Local/Realtime/VoiceDesign各两seed短句和一段落，共9组输入，均执行原路径→
共享表→原路径，完整codes与5帧分块解码的raw PCM位模式相同，生成正常完成。
测试计数断言共享臂确实建表、原路径臂不建表；TLS控制只编译进测试，不暴露生产
环境开关。VoiceDesign另用“年轻女性，声音清晰自然”描述及125帧生产预算；它的
完成为delay-drain，不冒称单一控制EOS轨迹。

生产默认候选的4项后端门禁通过：Local/Realtime未消费/满队列取消、真实帧预算
失败无End、异常后同owner恢复；冷/暖参考编码缓存、接续重复与部分取消、交错
克隆/无参考请求均保持各自PCM。普通后端29通过4忽略；模型CPU12通过；模型CUDA
两项显式RoPE门禁通过。模型及后端release all-targets Clippy、rustdoc -D warnings
通过。原有LNK4098警告保留，没有用链接器选项隐藏。没有修改依赖或强行升级CRT。

## 同期生产构建与Cargo缓存陷阱

候选B探针 `rope-production-build.json`，SHA
`7c6ec6af4a0be9ca2b36708e3f1e006cf082cbd66588ff9780b8251a8f7ad9d8`。
同期参考A探针 `rope-reference-rebuilt-build.json`，SHA
`8c2e6892977f9c197127ae85cbe9e8cd9b6615193160000f8af916e6d01c5ab3`。
两者同feature：moss/moss-candle/moss-candle-cuda/performance-tools。A在同一当前
工作区只覆盖Transformer文件为候选前的冻结实现，不引用更早整个工作区二进制作
单变量对照；新模块在A中没有mod声明，不进入A编译。构建完成后按SHA恢复候选文件。
128份源码冻结在rope-runtime-source，A的唯一覆盖文件另存，所有收据保留。

第一次尝试Copy-Item旧源文件保留了旧mtime，Cargo在0.38秒内复用了B产物，虽然
xtask记录了不同Source且构建前后源码稳定，二进制SHA却完全相同。该收据
`rope-reference-build.json`明确拒绝作为A，审计为rope-rejected-reference-build-audit.json。
更新本次覆盖文件mtime后，模型/后端确实重新编译1m06s，得到不同SHA的同期A。
还原候选时也更新mtime，避免后续Cargo复用A。没有删除整个target或改编译数学选项。

Source guard证明构建期间源码没有变，不能单独证明Cargo的缓存产物对应本次源码。
复制/恢复旧文件后核对编译记录、源码摘要及二进制身份；相同SHA的A/B不能当优化
对照。不能把一次错误标记的构建收据偷偷覆盖或纳入性能统计。

## 正式整链验证与保留决定

rope-abba-schedule.json预定16进程，Local/Realtime按模式各ABBAABBA、每进程1预热
加5热轮；两模式串行交错，计时期间不编译或运行其他模型。相同请求、种子、资源、
观测off及device sampling配置。96轮success/cleanup通过，未删轮或挑选后续快批次。
下面范围是4个独立进程热均值的min/max，不是置信区间。

| 模式 | A合成秒，均值（范围） | B合成秒，均值（范围） | 变化与结论 |
| --- | --- | --- | --- |
| Local1.7B | 2.817157（2.771779–2.853626） | 2.699051（2.657966–2.815901） | −4.19%，范围重叠，inconclusive |
| Realtime1.7B | 1.599175（1.535334–1.674984） | 1.205953（1.183256–1.233350） | −24.59%，范围不重叠 |

Local音频始终2.74秒，RTF1.028159→0.985055，范围同样重叠；Realtime始终2.40秒，
RTF0.666323→0.502481，范围不重叠。Local首PCM均值0.708876→0.676438秒，
A范围0.689210–0.731724、B0.649855–0.721020；Realtime0.272987→0.224514秒，
A0.224290–0.298664、B0.200340–0.244791。两个首PCM结论均inconclusive，不以
均值减少扩大成稳定首包收益。未对VoiceDesign做正式性能A/B，不外推该百分比。

全部480个资产SHA/字节数复核，96轮full F32的数值摘要独立重算：
sha256(u32le采样率+u16le声道+F32LE样本)，有限值、时长验证通过。原始source-ranges
文件保持，比较时仅排除consumer时间戳，全部chunk样本边界、UTF-8范围及文本SHA
相同。Local PCM为0f222678a68893c30c6ea968d3e69a113fd3f7ba6865c0e174ab5bb42687cda5；
Realtime为1f9328b4300ac1cfe5f7f8b695fedcf00f89a7ad07a698c5e69e459a242dba58。

最终两个Native短句都经冻结SenseVoiceSmall-int8 CPU4线程辅助回读，均为
“你好欢迎收听今天的故事”，与输入一致。ASR模型revision2365baeacb507f821a0c8120fcee3d484dba7a07；
识别器没有接收期望文本；不是双ASR或人工试听，听感仍pending。

### 原始数据与比较工具修复

analyze_rope_abba.py逐轮复算并验证来源/二进制；aggregate仅为统计复制并将4个原始
process_index0重编号0..3，run/attempt身份、条件和所有原始文件不改，provenance记录
原始results SHA。不能将aggregate所在目录当成真实资产root生成报告。

旧冻结比较器拒绝合法未开启观测的off标签，诊断probe原样保留为incomparable。
修复xtask比较器接受baseline/off各自的未插桩对照，但仍拒绝两臂标签不同及trace/
profile；不重标记原始JSON。两项回归测试后xtask单元58通过1忽略、all-targets
Clippy/rustdoc -D warnings通过。新比较工具SHA
`dd22ce7efc81753b96533899f94e60bb0df465c661500ec92a59f4ab0babcb75`，运行器仍为
上述ce6冻结工具；两者用途和版本分开固定。

6项完整比较中Realtime合成/RTF为pass，其余4项inconclusive，均输出等价、覆盖完整。
显式比较policy为absolute1e-6、relative0.001、expected_processes4，并非全项目保留
必须超过某个百分比。保留依据是可信局部收益、Realtime完整请求的明确收益、简单
的forward局部所有权，以及同一共享实现三个消费者的数值/状态回归。

### 准备成本与容量

加载范围Local A10.581–10.759秒/B10.625–10.710秒，Realtime A9.262–9.384秒/
B9.107–9.313秒；不把这点波动归因于RoPE。verify_seconds与各次短合成warmup另存
rope-preparation-memory-summary.json。reference_prepare仍为null，无参考请求；
reference_warmup是独立短合成，不冒充已隔离参考编码成本。

Job提交峰值Local A最大11,197,198,336/B11,197,296,640字节，Realtime A最大
9,747,030,016/B9,747,300,352字节；不是RSS或显存。采样系统最低可用commit为
640,307,200字节，容量余量小；沿用原始Native job预算，未改系统设置或终止用户
程序。采样整卡
显存峰值Local两臂均10,825,498,624、Realtime均9,380,560,896字节，包含其他桌面
进程且间隔500ms，不证明精确进程VRAM峰值不变。数据完整保留，不因内存条件紧
而删掉不利轮次。

### 复现与剩余边界

用原始job和新输出目录执行冻结运行器；按ABBAABBA顺序，每个step依次Local、
Realtime。run_rope_abba.ps1记录了本轮确切编排并拒绝覆盖已有输出，复测须指定新的
输出根。统计脚本、6个比较policy/结果、原始音频及source-ranges都在外部证据根。

```powershell
& $runner perf run --job D:/workspace/talechime-benchmarks/windows-moss17-20261024/rope-local-1.7b-A-job.json --output NEW_A_OUTPUT
& $runner perf run --job D:/workspace/talechime-benchmarks/windows-moss17-20261024/rope-local-1.7b-B-job.json --output NEW_B_OUTPUT
```

本项不改变官方精度/驻留差异。官方Local/Realtime仍只有分阶段诊断，不能由本项
原生A/B结论扩大成“达到或超过官方”；人工试听、正式多模式性能、风格等缺口仍单列。

收尾只修正Transformer模块头部“local均无位置编码”的旧注释；运行时实现仍与
冻结B一致，原编译源码快照保留。比较器修复是独立工具变更，没有改写任何测量文件。

两个rope-*-selected-first-B-report离线报告仅对应每模式首个B进程，明确不是全部
ABBA统计；完整结论以四进程每臂的summary/comparison为准，未挑选最快进程报告。
