# VoxCPM2 Windows 当前生产回归（2026-10-24，进行中）

设备为本机 Ryzen 7 9700X / RTX 5070 12GB / 驱动616.56，CUDA13。
本轮不沿用历史性能作为当前源码验收；保留既有图优化，先建立当前生产路径基线。

## 身份与准备

源码 HEAD 为42a7887e28f095684266eac43a68bd8fcc303d11及工作区修改；具体 dirty
摘要以本轮 xtask build receipt 为准。构建 features 为 performance-tools、voxcpm、
voxcpm-cuda，不合并其他后端或cuDNN功能。原始数据根目录：
`D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

Q8与Acoustic F16固定revision169f64d8b98bbaab1761e4ca3a83e6af653456cc；
官方BF16固定revision32279effe8c19989596f05d353d1447f51d9e915。4个主权重
的SHA全部按当前SOURCE.md复核通过，字节数及实际路径见model-audit.json。
模型保留在`~/.talechime/resources/voxcpm/models`，不复制进仓库。
官方源码在仓库外重新clone并detached checkout到
f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629；无浮动分支代替固定对照。

## 历史证据复审

旧target/tmp/official_voxcpm_bench.py在generate_streaming循环中累计全部chunk
时长，但循环后仅保存最后一个samples变量。因此历史RTF表是旧运行的时长
计数，旧WAV不能作为完整输出或内容通过的证据。本轮需使用新适配器拼接完整
chunk、独立校验音频长度和EOS/完成边界；不把旧脚本直接重新包装成合格基线。
该缺陷不自动证明旧时长计数错误，也不能从旧片段WAV推断完整质量。

C++开发基准已于2026-10-20退役，恢复位置为
[退役说明](../../legal/retired-voxcpm-native/README.md)，原始revision与许可证保留。
当前target/release/examples中的旧native exe不能当作当前源码新构建；需要恢复
固定源码的独立checkout及输入身份后才能补三方比较，不重新接入生产引擎。

## 当前状态

Q8参考与原始BF16默认音色基线、Q8图组合及C++参考短文本新对照已完成；Q8缓存身份缺口已通过新Job复测补齐。底层CUDA状态回归通过；完整Backend取消/恢复、长文本、官方Python和试听仍未完成。

## Q8短文本参考音色生产基线

冻结production-build.json对应exe SHA256为
`d2c5986564d9fad2e6758d8992cc8b88432c862bef9c08ec81233d83f10b3208`。
首次构建虽然编译通过，但期间文档修改触发source changed检查而拒绝发布；保持
工作区稳定后重试成功，失败产物未用于测量。当前Catalog独立构建并通过生产
manifest门禁，q8-prepared.json绑定资源身份。既存LNK4098仍有警告，未屏蔽。

输入为“你好，欢迎收听今天的故事。”，固定custom:acceptance参考音色；原音频
和voice.json均绑定SHA，seed Pinned42、steps10、CFG2、temperature1、默认
max_duration10.666666666666666秒（200帧）。使用原生流式完整Backend请求、
默认现有图设置；不沿用历史11段负载。1+1短测通过后，正式运行3进程各1+5。

| 进程 | 5热轮均值s | 首PCMs | RTF |
| --- | ---: | ---: | ---: |
| 0 | 0.65486518 | 0.11803670 | 0.27286049 |
| 1 | 0.60944706 | 0.12410526 | 0.25393628 |
| 2 | 0.63677936 | 0.12558998 | 0.26532473 |
| 总均值 | 0.63369720 | 0.12257731 | 0.26404050 |

全部18轮成功、各进程cleanup_verified=true；音频均2.40s，PCM摘要均为
`1aa1afcdf360ccb199f0c0ff83f1ea3bfeb484ff9f4ed469ce4a483f81f076fd`。
完整F32/PCM16 WAV、ranges与诊断保留于q8-baseline-run，Rust xtask生成的
离线报告为q8-baseline-report，逐轮及均值在q8-baseline-summary.json。
该基线只证明此例可运行且输出稳定，不证明codes、取消后复用或试听已验收，
也不是现有图相对eager的新收益验收；不与旧C++ RTF0.256直接比较。

## 既有图组合回归：当前生产路径 ABBAABBA

同一冻结exe、Q8短文本/参考/参数/seed，8进程按 eager,graph,graph,eager,
eager,graph,graph,eager 串行运行，各1预热+5热轮。eager明确设置
VOXCPM_CUDA_GRAPH=0和VOXCPM_DECODER_GRAPH=0，graph两者均未设置。
当前源码Q8默认启用CFM与decoder图；这里只验证该组合，不分别归因两项收益。
各Job冻结有效环境，未并行编译或推理；逐轮见graph-abba-summary.json与8个run，
对应8份Rust xtask离线report在同根目录。

| slot | arm | 热均值s | 首PCMs |
| --- | --- | ---: | ---: |
| 0 | eager | 1.22042744 | 0.23847506 |
| 1 | graph | 0.58993944 | 0.11481364 |
| 2 | graph | 0.60558466 | 0.11069220 |
| 3 | eager | 1.23960176 | 0.26398490 |
| 4 | eager | 1.26092430 | 0.22660676 |
| 5 | graph | 0.58909614 | 0.11115042 |
| 6 | graph | 0.63459266 | 0.12247802 |
| 7 | eager | 1.21926896 | 0.22318072 |

组合均值1.235055615→0.604803225s（−51.0303%）；eager进程均值范围
1.21926896–1.26092430s，graph为0.58909614–0.63459266s，不重叠。
首PCM总均值0.23806186→0.11478357s。48轮均成功、cleanup_verified=true，
音频全部2.40s、PCM摘要全部与上述生产基线一致。该结果支持在当前Windows
设备、Q8参考短文本负载保留已有图组合，不是新运行时代码的收益，也不与历史
优化百分比相加。当前baseline遥测没有逐图replay次数，因此源码默认配置和
冻结环境不能替代逐图执行计数，不能宣称每轮两张图均被独立观测到命中。

尚未新导出完整latent/codes、未新跑取消/失败后复用、BF16或长文本图回归，
不将本例PCM一致扩大为所有配置验收；官方/C++新对照及人工试听仍pending。

## 官方 Python 新适配器首次运行：初始化访问违例

`scripts/perf/voxcpm_cuda_reference.py`通过xtask reference协议运行固定官方源码，
准备一次官方prompt cache，热轮使用原官方_generate_with_prompt_cache流式接口，
不改模型算术。禁用denoiser、optimize/compile和badcase重试；steps10、CFG2，
使用与生产Pinned42首调用对应的7330414270409262323。显式max_len200并将
retry_badcase_ratio_threshold设100000，防止官方内部按文本长度缩小帧预算。
这是一项预算条件覆盖，需要标明；接口temperature为官方默认，Q8与原始BF16
仍是不同权重精度，不能据此视为严格同权重对照。

新适配器保留全部chunk拼接，核对累计样本数、WAV frames及报告时长；挂接
stop_head只保存最后一次tensor，生成结束后检查argmax为1，再允许成功round。
这些检查已实现但本次尚未运行到合成，不宣称完成真模型验证或边界正反例。
官方参考编码与生产缓存均排除热轮，参考准备计时单列，不当成完整参考前端性能。

独立python虚拟环境以.pth只读引用IndexTTS现有包，不安装或修改共享环境。
实际torch2.8.0+cu128、CPU capability AVX512；固定官方源码、脚本、权重、
参考音频/metadata、解释器、.pth和包METADATA均列入official-smoke-job.json。
失败产物位于official-smoke-run/process-0，cleanup_verified=true，终态为
failure/disconnected probe without terminal evidence，未产出prepared/round成功。

faulthandler显示Windows access violation发生于官方MiniCPMLongRoPE初始化的
_set_cos_sin_cache，model.py:96的emb.cos()，尚未打印加载主model.safetensors。
不是已证明的safetensors mmap崩溃，不能直接套用Qwen pread修复。用同环境/线程
数8、同配置单独创建官方MiniCPMLongRoPE成功，cos_cached为32768×128且
全部有限，日志rope-isolated.log。另一个8192×128 CPU sin/cos小测试也通过。
因此既不能宣称CPU RoPE普遍不可用，也不能确认完整构造崩溃的根因。

本次运行后C盘仅余约268MB，系统pagefile为C:/pagefile.sys、分配10519MB。
事后系统commit快照16.00GB/27.30GB、可用物理内存约9GB不是故障峰值，不能
据此确认或排除瞬时内存压力。resource-samples采到的rss峰值仅约5MB、private
约1MB，疑似虚拟环境启动器而非实际解释器；当前不作为模型内存证据，需要
检查Windows子进程采样归属。下一步应先恢复系统盘余量并核对真实进程树，
再做受控完整加载诊断；不通过改精度、缩上下文或删除模型规避失败。本轮无
有效官方热轮，不能给出新的官方性能排名或音质结论。

## Windows父启动器漏采：Job内存补充证据

已确认采样实现K32GetProcessMemoryInfo仅查询spawn PID，scope=process；虚拟
环境启动器可能再创建真实解释器，因此该RSS不能自动称为模型进程树内存。
旧文件不回填、不改名。xtask新增capture.json可选job_peak_memory字段，在
清理前查询本工具拥有的JobObjectExtendedLimitInformation.PeakJobMemoryUsed，
scope=contained_job、bytes、interval=null。这是系统持续跟踪的Job commit
高水位，包含所容纳的后代，不是RSS或显存，也不是各进程RSS峰值相加。
API依据[微软文档](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)。
查询失败保留unavailable/reason，不使原有清理跳过；Mac不增加伪造等价读数。

模型无关测试创建“启动器→分配64MB后代”，验证Job峰值>=64MB、清理后零
活动后代，以及退出后峰值仍保留。xtask 50单元测试和4集成测试全部通过，
Clippy --all-targets -D warnings与rustdoc通过；日志分别为本轮根目录下
xtask-job-memory-tests.log、xtask-job-memory-clippy.log、xtask-job-memory-rustdoc.log。
新工具副本及SHA保存在job-memory-tool.json，不以该测试替代VoxCPM加载峰值。

当前报告内存图仍按原有单进程采样绘制；新增Job高水位在capture.json单列，
尚未接入通用RSS图，不能混排行。此修改不补出上次崩溃的历史峰值，也不证明
内存压力就是VoxCPM访问违例根因；系统盘余量恢复后再用新工具做受控加载。

## 原始BF16默认音色生产基线与新Job内存实测

同一冻结production probe（d2c598…）支持原始BF16变体。新的Catalog资源校验
bf16-prepared.json通过；运行父工具为job-memory-tool.json固定SHA副本。
模型固定2b-bf16/revision32279eff…，主干BF16、codec F32；默认CFM图开启，
decoder图默认关闭，未强制打开或改变精度。短文本与Q8相同，但voice=narrator，
不使用参考，conditioning_files为空并重新冻结身份，不能与Q8参考音色当成公平
模型速度比较。steps10/CFG2/temperature1/Pinned42/默认200帧预算不变。

1+1 smoke成功，两轮PCM一致，随后3进程各1+5正式基线：

| 进程 | 5热轮均值s | 首PCMs | RTF | Job commit高水位bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.00301276 | 0.09531170 | 0.34232517 | 6881112064 |
| 1 | 0.99579264 | 0.09796666 | 0.33986097 | 6881361920 |
| 2 | 0.99515168 | 0.09564420 | 0.33964221 | 6881705984 |
| 总均值 | 0.99798569 | 0.09630752 | 0.34060945 | 不把各进程峰值相加 |

18轮全部success，三个进程cleanup_verified=true；音频均2.93s，PCM摘要均为
`b9d5174a564eaf1b50d4e5bdce26b081589c60b8e3349dcedff8848ca2904d37`。
详细数据bf16-baseline-summary.json、完整证据bf16-baseline-run，Rust xtask
生成bf16-baseline-report。Job高水位是contained_job的OS commit峰值，不是
RSS或显存；新读数首次用于本轮真实模型，不能反推旧官方加载失败的峰值。
运行stderr保留`skipping: unsupported storage type IntStorage`，未隐藏或将
成功合成扩大成全部原始checkpoint语义/数值验收。当前没有导出完整latent，
没有新增取消、参考、长文本、decoder图或官方BF16数值对照，试听pending。
本轮native运行后C盘余量基本未变；不因此认定官方Python崩溃必由内存导致。

## 当前CUDA图状态回归与decoder取消补测

当前crate的release CUDA测试构建通过，保留既存LNK4098警告。已有4项显式
本地实机测试串行通过，日志与原始冻结测试exe位于graph-tests目录：

- real_model_replay_noise_schedule_and_lru：改变噪声、步数/CFG、LRU驱逐，
  实际图输出与独立eager逐位一致；录图后取消不禁用图；注入同步拒绝后eager
  fallback及后续复用一致。并非任意异步CUDA错误可恢复的证明。
- views_capture_cleanup_and_fallback：非零偏移view、录图失败后退出capture、
  同步失败回退与取消检查。
- full_decoder_replay_preserves_state_and_reset：真实AudioVAE F16、3请求×8帧
  eager/图逐位对照与reset，无意外fallback。
- strided_output_copy_is_warmed_before_capture：非连续输出的复制预热及回放。

发现decoder测试未显式覆盖取消后新请求，遂仅扩展该ignored真模型测试：
执行前取消不消耗一帧；执行后取消返回错误且保留可复用图；双方请求reset后
连续8帧逐位相同。新测试通过（1.10s），普通10测试通过、11个显式跳过；
未下载模型。新的测试exe SHA为
`8fb38761ee02949f3c1fe236bffefb9534915378a505562e1164caa50dcfaddb`，
源码补丁/文件SHA/Cargo.lock SHA均在graph-tests；旧exe与日志不覆盖。

CUDA release lib/tests Clippy -D warnings、CUDA rustdoc、fmt及diff检查通过，
日志graph-tests-clippy.log、graph-tests-rustdoc.log。生产运行时未修改，不需要
将已有性能重命名为新优化。此处只证明CFM/decoder/recorder层，不替代完整
Registry/Backend的取消、有界流及失败后复用验收；尚未新导出完整latent/codes，
也不覆盖BF16 decoder图或异步设备上下文损坏恢复。

## 原生C++对照恢复准备

依退役说明在仓库外恢复干净detached checkout：
`D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/native-reference`，
HEAD固定30339e67c44a3e000f7d1259ec038bf7daff18e3，尚无新构建或计时。
旧voxcpm_native_benchmark在合成计时内逐段写WAV，边界不同于当前生产probe；
正式三方比较前需要独立适配器对齐热计时并冻结补丁/二进制，不能直接复用旧RTF。
该checkout不重新接入主项目生产引擎，不带模型，也未改旧源码或许可证。

## C++恢复构建与参考缓存身份补查（进行中）

固定30339e6 checkout已启动release构建voxcpm_native_benchmark，CUDA13、
CMAKE_CUDA_ARCHITECTURES=120、GGML_CUDA=ON、原动态CRT配置；Cargo与CMake
并发均限制为2，TEMP/TMP放D盘。日志native-build.log；CUDA模板仍在编译时
不运行推理。尚未发布新C++二进制收据，也没有新的C++性能结果。

仓库外native-reference-adapter.rs已准备并格式化，但尚未接入编译；计划采用
xtask reference协议，完整PCM收集、WAV写盘在热计时后、只接受Outcome::Complete。
旧桥接固定seed42，而生产Pinned42会派生7330414270409262323，且各实现RNG
不同；不能宣称seed流一致。旧默认steps10/CFG2/temperature1/max_frames200
已核对。旧参考load函数与当前相同，文件摘要差异仅为另一函数的crate别名。

另发现此前Q8参考Job只绑定WAV与voice.json，未绑定生产实际读取的
candle-reference-v1.json。旧PCM一致性观察保留，但参考输入身份链有缺口，
不能事后填入当前SHA冒充历史已冻结。当前缓存35帧/8960特征，SHA为
`ef4d6c1fa7d6ed63f936de01bf17349b2fb184c8591deaba984196a4e183f20c`，
身份及路径/字节数在reference-cache-audit.json。已生成新的q8-cache-bound-job.json，
包含缓存artifact并重算conditioning摘要；待构建结束后独立复测，尚未运行。
这不涉及修改生产缓存实现，也不删除或重写用户参考文件。后续三方排名使用
新冻结条件，不把旧基线静默升级为完整身份验收。

## 固定C++新构建与缓存身份完整的生产Q8对照

原始30339e6干净checkout的CUDA release构建完成（15m50s），未使用旧exe。
原始开发基准单独冻结native-original-build.json，但因旧计时包含WAV写盘，未用
其计时排名。新适配器源码保留于项目`scripts/perf/voxcpm_native_reference.rs`；
复制到旧checkout的examples/voxcpm_xtask_reference.rs，仅新增sha2开发依赖，
Cargo.lock diff只有已有sha2依赖边，无版本升级；C++实现/参数/许可证未改。
manifest/lock补丁native-adapter.patch，完整adapter源及SHA另存，不能仅靠该
tracked diff恢复untracked例子。构建收据native-adapter-build.json，exe SHA：
`405d501d44fb8ce822e63db4396b981f3d9274db2e4ad292817bfdcaab04bf7e`。
原有LNK4098仍保留。动态CUDA13 cudart/cublas/cublasLt的路径及SHA列入Job，
显式PATH以CUDA13 bin/x64优先；共1099个源码/权重/输入/库文件绑定SHA。

新适配器经xtask reference运行，完整PCM收集和正常Outcome::Complete才成功，
WAV写盘在热计时后，重新读WAV核对样本数；1+1短测后正式3进程各1+5。
生产Q8也用新q8-cache-bound-job.json正式3×(1+5)，包含磁盘参考缓存SHA并
核验运行前后不变。两组串行，不与编译并行；新生产基线补齐身份，不修改旧Job。

| 路径/进程 | 热均值s | 首PCMs | RTF |
| --- | ---: | ---: | ---: |
| Candle Q8 / 0 | 0.57823528 | 0.10986928 | 0.24093137 |
| Candle Q8 / 1 | 0.58903714 | 0.11203060 | 0.24543214 |
| Candle Q8 / 2 | 0.60242216 | 0.11273304 | 0.25100923 |
| Candle Q8总均值 | 0.58989819 | 0.11154431 | 0.24579091 |
| C++ / 0 | 0.73007050 | 0.23696730 | 0.28518379 |
| C++ / 1 | 0.74693476 | 0.25216848 | 0.29177139 |
| C++ / 2 | 0.74016546 | 0.24156742 | 0.28912713 |
| C++总均值 | 0.73905691 | 0.24356773 | 0.28869410 |

每组18轮成功、所有进程cleanup_verified=true。Candle全18轮PCM仍为1aa1afcd…
（完整摘要见前文及q8-cache-bound-summary.json），音频2.40s；C++全18轮为
`92ca88bd3eb100da17852586ff56d15536cb3499b4ae6149cee87e5c7214220d`，
音频2.56s。Candle本例RTF低约14.86%，这是当前短文本条件下的吞吐观察，
不是整后端或质量等价达标。共同GGUF/revision、输入WAV/转写、steps10/CFG2/
温度1/200帧预算相同，但C++固定seed42、生产派生seed不同，RNG/算子精度、
参考编码与输出长度不同；不能把少生成0.16s当成整链优化，也不作PCM跨实现
等价结论。官方Python仍未恢复，无法宣称三方全面达标，人工试听pending。

C++加载1.866–1.895s、参考读取/重采样/编码0.159–0.167s；生产加载
2.999–3.116s，reference_prepare_seconds=null，0.678–0.719s的可选reference
warmup包含短合成，不能拿来与C++纯参考准备比快慢。Job commit高水位：
C++8.385–8.422GB，生产5.818–5.861GB（十进制），不是RSS或显存，不混排。
准备详情在native-production-preparation-summary.json；原始结果分别为
native-baseline-run（external_reference_not_production）和q8-cache-bound-run。
后者生成Rust xtask离线q8-cache-bound-report；外部参考摘要由原始rounds计算，
没有改写为生产Attempt数组以绕过report类型检查。

适配器rustfmt、实机协议/完整输出/正常结束、Clippy -D warnings和rustdoc均已通过。
Clippy曾因独立Cargo OUT_DIR重新编译原生CUDA（13m41s），rustdoc随后2.98s完成；
日志native-adapter-clippy.log和native-adapter-rustdoc.log。主项目未重新接入retired native引擎。

### VoxCPM当前冻结Job恢复入口

下列命令仅复用本机已冻结二进制/资源/输入，输出必须为不存在的新目录；运行前
停止其他编译和推理，先确认所有Job SHA仍匹配。不要覆盖旧输出或绕过校验。

```powershell
$vox = 'D:/workspace/talechime-benchmarks/windows-voxcpm-20261024'
$owner = (Get-Content -LiteralPath "$vox/job-memory-tool.json" -Raw | ConvertFrom-Json).path
& $owner perf run --job "$vox/q8-cache-bound-job.json" --output "$vox/q8-recheck-NEW"
& $owner perf reference --job "$vox/native-baseline-job.json" --output "$vox/native-recheck-NEW"
```

C++适配器源码在[scripts/perf/voxcpm_native_reference.rs](../../../scripts/perf/voxcpm_native_reference.rs)，
仅用于固定退役checkout的开发对照。按本报告恢复旧commit，将该文件复制到
crates/voxcpm-sys/examples/voxcpm_xtask_reference.rs，新增sha2.workspace=true开发
依赖，并核对Cargo.lock只新增已有sha2依赖边；CUDA13/ARCH120下构建该example。
新构建必须生成新收据与Job，不能将新exe填入旧Job冒充相同二进制。

## 独立单ASR内容辅证与工程检查完成

固定SenseVoiceSmall INT8 revision2365baeacb507f821a0c8120fcee3d484dba7a07，
3文件共239549806字节，由Rust xtask resource-download根据项目原manifest准备，
大小/SHA均验证，含许可证，保存在~/.talechime/resources/asr/models/sensevoice-int8/
该revision下。资源计划和下载收据为sensevoice-resource-plan.json、
sensevoice-resource-download.json；不下载Qwen ASR主模型，不启用产品双模型校验。

新增显式本地开发example sensevoice_aux_readback（asr feature），Tokio LocalSet，
只接受1..16份有SHA的PCM16/F32 WAV，输入字节有界并在解码前验证SHA；只把音频
传给识别器，原文只用于报告。结果类型auxiliary_single_recognizer_readback，
dual_model_qualification=false、listening_acceptance=pending，不自动判断质量通过。
错误SHA实机负例拒绝且未发布输出。模型准备与显式实机验证分开，普通测试不下载模型。

抽取生产Q8、生产BF16、C++各进程0的第1热轮完整音频，3份独立转写均为
“你好欢迎收听今天的故事”，与原短句去标点后的内容一致。expected/转写/标签/
输入WAV SHA保留于aux-readback-results.json，未改写原音频或TTS结果。
该辅证仅覆盖这3份短句样本，不证明所有18轮、音色/音质或长期播放合格，不把
SenseVoice的HAPPY/NEUTRAL标签当作听感评分，不能替代双ASR或人工试听。

探针exe SHA为61ef8c153f4eab54cbd897504676693f3a91fb254c6f21d532a824d67561ff65，
收据aux-readback-build.json，源码/inputs/结果及下载收据摘要见aux-readback-evidence.json。
ORT为rc13静态onnxruntime链接，静态库SHA与exe绑定；import表含DirectML，
但SenseVoice Session未请求DML/CUDA，使用默认CPU配置，线程4；没有做逐节点EP
profiling，不能把链接DirectML称为DML实机验收。最初收据脚本误找独立ORT DLL，
未生成收据；后以实际静态链接信息补齐，原回读结果保留。导入检查日志为
aux-readback-link-dependencies.log。辅助探针release构建、Clippy -D warnings、
rustdoc、fmt/diff检查通过，日志aux-readback-build/clippy/rustdoc.log。

同一探针随后用于Qwen共享prompt贪心样本，验证方法复用，见对应Qwen报告。
官方Python加载仍未恢复，完整Backend取消/恢复、长文本和人工试听仍pending；
本节不扩大VoxCPM后端达标范围。

辅助回读本机恢复命令（输出必须不存在，模型仅本地验证，不自动下载）：

```powershell
$vox = 'D:/workspace/talechime-benchmarks/windows-voxcpm-20261024'
$aux = (Get-Content -LiteralPath "$vox/aux-readback-build.json" -Raw | ConvertFrom-Json).exe
$model = "$env:USERPROFILE/.talechime/resources/asr/models/sensevoice-int8/2365baeacb507f821a0c8120fcee3d484dba7a07"
& $aux $model "$vox/aux-readback-cases.json" "$vox/aux-readback-recheck-NEW.json"
```

重新构建使用cargo build --locked --release -p talechime-backends --no-default-features
--features asr --example sensevoice_aux_readback；新exe需生成新收据，不覆盖旧身份。

## 默认官方加载的提交内存复查：证据已定位资源压力

在无其他cargo/rustc/nvcc/Python推理进程的状态，用相同冻结官方脚本/权重/
环境重新运行1+1 Job，只把父工具换为带Job高水位的新xtask并设120s时限。
输入official-memory-audit-job.json，原始证据official-memory-audit-run，未覆盖
旧失败。实际在初始化阶段退出，未产生有效prepared/热轮；不是超时后补算的成功。

新capture.json：outcome=failure、cleanup_verified=true、Job commit峰值
11870855168 bytes（约11.87GB）。45次系统采样中，available commit从
10977366016 bytes降至23.728s时的9326592 bytes（约9.33MB）。同时旁路
PowerShell检查进程以-532462766报Out of memory，检查命令没有生成预期快照，
不能把失败命令当成磁盘/系统内存实测。摘要memory-summary.json保留该区别。

这次访问违例栈位于torch.nn.Linear初始化及官方ResidualLM构造，与前次RoPE
cos不同；系统提交余量近乎耗尽和旁路OOM是本次资源压力的直接证据。不能继续
把不同栈顶算子分别当成已定位的实现bug，也不能从此证明所有历史AV均为OOM。
固定源码在完整cls构造后才调用model.to(lm_dtype)；配置/打印BF16不能代表加载
全程没有较大的CPU临时参数。本次尚未独立量化每个构造阶段的dtype/峰值来源。

session句柄随后消失；重新核对capture及OS进程列表确认无残留Python，资源
已恢复，事后系统commit15.746GB/26.896GB、可用物理约5.6GB均不是峰值。
没有重启该Job、没有改模型精度/上下文或系统pagefile，也没有清理用户模型。
当前headroom不支持重复默认官方加载；后续先做按本案例实测预算的启动前检查，
或在内存余量改变/经过独立验证的低峰值加载路径下继续，不设全模型通用阈值。
该失败不产生新的官方速度排名，后端官方达标状态仍未证明。

## 可配置提交内存预检：默认官方加载不再盲目重试

xtask reference Job新增可选minimum_available_commit_bytes，省略的旧Job序列化
和执行保持原行为，0无效。Windows在每个参考进程前核对系统available commit，
保留process-N/commit-preflight.json；不足或查询不可用会在创建参考请求/进程前
拒绝。其他平台显式设置时拒绝并记录不支持，不增加Mac内存或优化结论。原来
单PID采样和Job高水位的口径保持不变，预检不修改精度、模型、采样或系统设置。

失败峰值11.87GB不是完整加载最大需求，后续仍要读入4.58GB主checkpoint等资源；
本轮给official-guarded-job.json暂设20,000,000,000字节可用commit审计预算。
这只是保守规划，不是官方最低内存，不是已证明足够，也不变成其他模型的默认值。
新冻结父工具身份commit-preflight-tool.json；实机official-guarded-run记录可用
11614326784 bytes、所需20000000000 bytes、status=blocked。没有创建request.json、
stdout.jsonl或results.json，OS核对无Python进程；这是准备阶段拒绝，不是新性能
结果或清理失败，不覆盖旧OOM证据。

模型无关测试覆盖旧字段省略、预算不足时无参考请求/输出、极小预算就绪。
xtask 51单元测试及4集成测试通过，Clippy --all-targets -D warnings与rustdoc通过，
fmt/diff通过；日志xtask-commit-preflight-tests/clippy/rustdoc.log。
预检是一次快照，不预留内存、不终止其他应用，也不保证加载中不再OOM。后续
只有资源余量变化或独立验证低峰值加载路径后再恢复官方对照；不放宽门槛凑成功。


## 后续 owner 验证与时长预算修正

[独立状态门禁记录](voxcpm-windows-owner-state-2026-10-24.md)补齐 Q8/BF16 的
unread/满队列取消、入口拒绝、实际部分输出后预算失败及同 owner PCM 恢复。
同时发现旧 max_duration 按输入 2560 样本误用输出 48 kHz 换算，预算放大三倍。
现改为每 patch 7680 输出样本、向下取整，默认 187 patch=29.92 秒，保持 30 秒上限。
旧报告的 200 patch 仍为原始测量配置；其“10.667 秒”只是旧参数标签，不是实际音频上限。
这是正确性修复，不追加提速百分比，新构建已各 3×(1+5) 复核：Q8 0.6432s、BF16 0.9613s；各18轮完整生产PCM
与旧基线一致，独立加载/参考预热/内存和报告门禁见上述状态记录的生产基线节。
不把跨批次差额解释为本修复提速。


## 官方加载控制的首次成功（非正式基线）

[独立调查](voxcpm-windows-official-loading-2026-10-24.md)用原官方类、meta主模型、
逐项权重物化与原顺序RoPE重建，精确校验577+312+9项，恢复1预热+1热轮的官方自然EOS。
主BF16/codecF32、8192上下文保持，明确偏离默认from_local及Safetensors加载依赖。
单热轮2.685s/首PCM0.162s/音频3.04s仅作可运行性证据；回读内容完整，试听pending。
Job峰值10.768GB且系统余量仅335MB，正式3×(1+5)在12GB可用commit预检下拒绝启动，
当前约10.63GB。默认加载的20GB guard不变，不能改称官方正式性能已达标。


## 10-25 段落与C++后续

[63字段落](voxcpm-windows-paragraph-2026-10-25.md)三路径各3×(1+5)：Rust Q8
2.894s/12.4s音频、BF16默认音色4.637s/15.0s，C++ Q8固定参考3.022s/12.8s；
各18轮PCM稳定，Rust默认/eager额外各两轮完整PCM与结束一致。Q8/C++的RTF
范围重叠，不能将短句速度结论外推为所有文本上稳定超过C++。辅助回读覆盖内容，
试听与框架间音色/质量等价pending；真正长篇/接续/style仍未覆盖。

后续[Windows K复制运行时验证](voxcpm-windows-key-packing-2026-10-25.md)在保持GEMM形状
及数值的前提下，取得Q8/BF16段落约5.26%/2.18%的完整请求收益；短句与首PCM仍
inconclusive。176个latent patch、完整PCM、owner状态和192轮生产结果分别验证，
不以局部改善或跨批次绝对数字替代官方同条件达标。

[官方加载控制版正式对照](voxcpm-windows-official-baseline-2026-10-25.md)随后完成3×(1+5)，
BF16无参考均值2.620s/3.04s音频，当前Native默认CFM图4进程为0.873s/宿主2.93s；
两边均19patch，真实BoundarySilence复现样本数差。CFM图单变量交错补验约33.08%，
保持48轮PCM。官方未编译/实验加载边界、原默认入口失败及跨框架质量未验收明确分开；
同步归因显示Native原始F32 VAE是后续热点，未直接扩大默认图覆盖。

后续[F32 decoder资格验证](voxcpm-windows-f32-decoder-graph-2026-10-25.md)补齐F32/F16
各33对codec状态、完整latent/PCM、公开owner及Qwen/VoxCPM六模型组合回归后，
Windows BF16默认启用decoder图。新旧冻结二进制ABBA的短句/段落改善21.33%/26.68%，
96轮生产PCM保持，首PCM亦有范围不重叠的改善。该比较的A已含CFM图与K复制，
不叠加历史百分比；其余模式和跨框架质量的未完成项保持。
