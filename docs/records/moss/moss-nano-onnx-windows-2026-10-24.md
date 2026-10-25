# MOSS Nano ONNX Windows provider 复核

local batch1特化虽减少执行节点且保持已测codes/PCM，但生产初筛未改善，现已撤回；
见[候选与撤回](moss-nano-onnx-local-batch1-2026-10-25.md)。

官方CUDA节点profiling与固定codec分块的输出控制见
[profiling后续](moss-nano-onnx-cuda-profile-2026-10-25.md)；trace不是正式Native计时。

后续主CLI显式ort-dynamic构建、暂存DLL实际加载与静音播放完成验证见
[Windows动态ORT主CLI](../platform/windows-ort-cli-dynamic-2026-10-25.md)；标准CPU
发行配置未改，完整CUDA分发仍未验收。

2026-10-24；Windows 11、Ryzen7 9700X、RTX5070 12GB（sm120）。
前半段为缓存分发复核；后续 DirectML 实测见文末，不能把 CUDA 分发限制推广为全部 GPU provider 不可用。
最新：完整新ORT分发已在专用动态加载构建中通过CUDA生产Registry请求、18轮基线及
取消/失败/换音色复用；默认静态分发未更换，官方同条件排名和试听仍未完成，详见末节。

## 当前生产路径

`moss/runtime.rs` 在 CUDA 路径使用 I/O Binding，将 present/cache 输出请求到 CUDA，
其余输出送 CPU。`devices.rs` 显式注册 CUDA 并 error_on_failure；该选项不等于禁止
ORT 将不支持的图节点分到 CPU。`ort-cuda` 必须显式编入；DirectML 仍是单独
`directml-probe` 开发入口，不是产品 Device，不改变 Auto 校准或持久化配置。
ORT Rust 依赖仍固定 rc.13、api-21，未升级依赖或替换 DLL。

## 当前 DLL 的实际架构

使用本机 CUDA13.0 的 cuobjdump --list-elf / --list-ptx 重新检查 AppData/Local/
ort.pyke.io/dfbin 中的两个 CUDA provider；不是从旧报告复制架构结论。
原始 inventory、ELF/PTX 输出和 SHA 在仓库外
`D:/workspace/talechime-benchmarks/windows-moss-onnx-20261024/`。

| 分发目录标识 | 字节 | provider SHA256 | ELF 架构 | PTX |
| --- | ---: | --- | --- | --- |
| efb686cb… | 62276096 | 4ec94428733d4f3bc3d03443d059e400293bda79691b8004ed15b4ed13eb643f | sm75/80/90a | 无 |
| 743380B9…/onnxruntime/lib | 180731904 | bfe2b37002764728a188a212d1ac72cdbf073e8b1b2c7f48ba1f82bd11382355 | sm60/61/70/75/80 | 无 |

PTX 日志一行是“No PTX file found”，不能将行数1误报为存在一个 PTX。
当前 target/release 的 CUDA provider 已不存在；debug 的链接也不能替代有效分发身份。
这次没有重跑生成；旧 sm120 `/Cast` 的 NoKernelImageForDevice 实机失败见
[rc.13 补记](../platform/ort-rc13-upgrade.md)。当前静态证据继续支持分发缺少
Blackwell 内核的判断，不把历史生成结果当作当前 main 的新实验。

## 下一步与资源约束

需要完整且匹配的 Runtime/provider 分发，再验证最小算子和生产完整请求；不能只替换
单个 provider DLL。官方新发行版本可以作为调查候选，但发行说明或架构标签本身不证明
本项目已链接、加载或成功生成。参考
[ORT rc.13](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.13)、
[Runtime1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)。
本轮 GitHub release API 查询受共享 IP rate limit 限制，未获得资产大小清单，未下载。

检查时 D 盘仅720220160 bytes 可用。已核验本轮 moss17 Cargo 缓存约2012560327
bytes，未发现音频/ONNX/Safetensors/结果数据，且外部冻结 catalog/probe 的 SHA 与
收据一致。限定生成物扩展名的清理命令被自动审批策略拒绝，未执行、未删除；不更换
实现绕过拒绝。模型、音频、证据与缓存均保留。当前没有 ONNX GPU 正式基线或提速
结论，CPU 旧成绩和 AMD iGPU DirectML 旧成绩不替代 RTX5070 验收。

## RTX5070 DirectML 实测与缓存比较

同机 Windows / Ryzen7 9700X / RTX5070、driver616.56；源码HEAD42a7887e及当前工作树，
ORT Rust rc.13/api21，实际静态 Runtime 报 rel-1.28.0/da9b5e3，DirectML DLL
1.15.4（SHA9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1）。
冻结 exe/源码/feature 身份见外部 windows-moss-onnx-20261024/directml-build.json。
未升级 ORT、未替换单个 CUDA provider、未修改产品 Device 或 Auto 配置。

恢复现有 directml-probe 的编译：Runtime::generate 新增的 prefix 参数显式传 None，
保持无接续语义。探针 WAV 改用64KiB BufWriter和create_new；Pcm::duration_ms已有
有限值/格式校验，显式finalize保留写入/flush错误。下面时间仍包含缓冲WAV导出与
背压，不是纯模型时间，也不和旧未缓冲探针成绩直接排名。

当前16个生产资源逐文件大小/SHA匹配嵌入清单，固定上游TTS
f52645cb467506d8e18e746ddd59482685b74e58、codec
ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae；完整URL/SHA见
directml-resource-verification.json。没有下载或修改模型。文本“你好，欢迎使用叙铃。”、
Weiguo、直接Runtime seed42（不是Candle Pinned42派生seed），关闭接续，音频3.28s。
不把其他Nano官方对照的seed/权重/dtype直接视为本ONNX导出条件已匹配。

DXGI显式选择index0、NVIDIA vendor4318/device12036、LUID00000000:0001482b、
D3D12=true、hardware=true。同机另有AMD核显及不同LUID的同名RTX条目，保存了全部
adapters.json，不能只按名字推断device索引。当前I/O Binding allocation device id为0，
本次只覆盖index0，不外推到其他适配器索引。

### 实际执行与测量所有权

开发探针使用当前原生Runtime/生成循环和容量1流，但不是完整Registry/Backend请求。
外部只读Python桥接调用冻结Rust探针，Rust xtask perf reference拥有进程组、时限、
输入校验、资源采样与清理；结果明确external_reference_not_production。
桥接在探针完成后转写其实际轮次，不作为逐轮实时事件延迟测量。first_pcm来自原生
探针接收首块，桥接沿用的segment_delivery_first_pcm标签不代表模型只能整段交付。
本例原生确实按块交付，不将标签用于跨后端流式能力排名。

独立开启ORT profile，保留完整原始trace（约340MB）；计时不混入下表。
实际provider节点事件次数：

| Session | DirectML | CPU |
| --- | ---: | ---: |
| codec decode step | 54137 | 21457 |
| TTS decode step | 117180 | 6552 |
| local sampled frame | 242004 | 22188 |
| prefill | 4348 | 400 |

这是整次诊断含预热/热轮/取消/恢复的trace事件计数，不是静态图节点数或GPU时间占比。
encode session在内置音色case没有provider事件，不能将空表说成编码失败或CPU执行。
证明DirectML实际参与生成，同时存在CPU fallback；error_on_failure不是全图GPU门禁。

### 基线和ABBAABBA

host模式3独立进程、各1+5热轮全部EOS与清理通过，进程热均值
1.39524558 / 1.38245090 / 1.41025658s，加载2.635–2.646s，首PCM约0.0976s。
设备cache模式先做1+1冒烟及取消后复用，随后两臂同冻结exe正式ABBAABBA，各1+5：

| 模式 | 四进程热均值 s | 进程均值范围 s | 首PCM均值 s |
| --- | ---: | --- | ---: |
| dml-host | 1.39334880 | 1.38984916–1.39895276 | 0.095885755 |
| dml-cache | 1.10222907 | 1.08424666–1.12713134 | 0.076143905 |

设备cache相对host约快20.9%，进程范围分离；范围不是置信区间。此为现有实验模式的
实机确认，不是本轮新实现的生产算法收益。48轮EOS、3.28s长度与全部PCM SHA
9fdf5a1be81504b8f99041f7623c874da572dd35e8729100da5b333014b5881b一致。
八次首块后丢弃接收端，取消10.42–17.56ms，随后完整请求EOS；所有恢复WAV与原轮
字节相同（SHAe39022182703d0b38ce6c667f9aad72c474c12360ccc995b6a34c27672dc3b7d）。
这覆盖首块后取消和复用，没有声称额外故障注入、切换音色、长文/接续或全模型验收。
Job commit峰值host约2.42GB、cache约2.19–2.22GB，是含后代提交高水位，不是RSS/VRAM。

复用冻结SenseVoiceSmall-int8回读相同PCM一次，识别“你好欢迎使用去灵”；专名与目标
“叙铃”不一致，不能宣称内容或音质通过。报告保留原文/实际转写，人工试听pending，
也不凭单识别器断言TTS一定读错。模型/音色/配置进一步条件对齐及官方ONNX参考待补。

### 恢复与未完成项

外部根 D:/workspace/talechime-benchmarks/windows-moss-onnx-20261024/ 保留冻结exe、
DirectML.dll、16资源收据、原始音频/trace、读回、逐轮结果和环境。
使用 windows-voxcpm-20261024/commit-preflight-tool.json 指定的冻结xtask：
`perf reference --job <directml-host-baseline-job.json> --output <全新目录>`；
比较使用directml-abba-0..7-job.json原顺序串行执行；profile另用directml-host-profile-job.json。
DirectML桥接脚本保存在各Job固定绝对路径，不能原地编辑后仍声称旧收据有效。
当前release探针原生组合构建、适配器选择测试、Clippy(lib/example，-D warnings)、
rustdoc(-D warnings)通过。仍缺GPU生产设备接入/完整Registry基线、官方同条件对照、
更多负载与试听。CUDA缺sm120分发的原限制仍在；DirectML实验成功不等于整个后端达标。

最后运行该feature的22项普通测试通过；其中命名real的旧测试无环境时提前返回，不当作
本轮真实模型覆盖。真实DirectML覆盖以上述xtask产物为准。相关skill和覆盖报告的本地链接检查通过。

## 官方 ONNX Python CUDA 对照（同日后续）

核对固定[TTS导出revision模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX/blob/f52645cb467506d8e18e746ddd59482685b74e58/README.md)：
它只提供导出，官方Python在现有干净checkout
`~/.talechime/benchmarks/windows/moss-nano-official`，HEAD
8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb，实际入口onnx_tts_runtime.py和ort_cpu_runtime.py。
使用原synthesize_single_chunk和原streaming codec，不复制Rust推理循环；没有顶层
WeText/文本分段前端，不把这个单块官方参考称作完整生产Registry请求。

新建外部python-cuda隔离env，不修改旧Torch环境或全局Python。pip安装报告固定
onnxruntime-gpu1.28.0、soundfile0.13.1、sentencepiece0.2.1及依赖wheel的来源/SHA；
pth只读复用原Nano与Index的依赖。实际ORT build为0368187f84，与Rust缓存的da9b5e3
不同。新CUDA provider SHA
b796fc6e2ff403977d25345ceb6321235f94a5ecfe7a3f92f1a22f177d7765d2，cuobjdump确认
存在sm120 PTX。它不推翻旧缓存缺内核的事实，而证明同1.28版本不同分发不能混同。
本轮没有把单个新provider塞进旧Rust静态Runtime；完整Rust分发接入仍待验证。

保留三项早期失败：缺soundfile、缺sentencepiece，以及官方相邻目录布局不满足；
均在有效轮次前失败，cleanup通过，没有把失败计作慢轮。直接给官方tts目录联接会
在Path.resolve后回到原canonical目录，../codec指向随之改变。最终将原16文件字节
复制到外部official-model-layout的官方目录布局，未编辑manifest/模型；正式Job固定
实际读取的复制文件SHA，生产资源与原件保持原状。

指定CUDA13 bin/x64 PATH后真实生成成功。官方preload仍警告Torch2.8cu128与CUDA13
wheel不匹配，且报告若干预加载失败；不能抹掉警告或只凭PATH断言实际加载版本。
适配器用GetModuleHandleW/GetModuleFileNameW实测：cublas13来自本机CUDA13，
cudnn64_9.dll却来自Torch/lib（9.10.2.21，SHA
9edbcdff73b0af070eb160b2ce66e59feca04aa017351d8eedcc5e8e149967d2）。
这是有真实成功证据的混合依赖环境，不作为推荐兼容栈；runtime.json保存实际路径/SHA。
各官方session都报告CUDA+CPU provider，但尚未单独收集官方CUDA节点trace，不声称全图GPU。

### 默认官方基线、随机流及prompt控制分开

Rust xtask调度official-cuda-baseline-job.json，三独立进程各1+5成功，cleanup全部通过：

| 进程 | 热合成均值 s | 首codec PCM均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 1.16943710 | 0.07331612 | 3.28 |
| 1 | 1.12844624 | 0.05911920 | 3.28 |
| 2 | 1.17929500 | 0.07041774 | 3.28 |

总均值1.15905945s、首0.06761769s；加载、实际session列表、Job内存及逐轮独立保存。
Job commit峰值6.158/5.632/5.897GB，不是RSS/VRAM。这里不含事后WAV写出，而DirectML
探针包含缓冲写出；官方首PCM在codec产生处，Rust在有界接收端，不能忽略交付边界差别。
默认numpy.default_rng42，375帧预算，固定sample_mode=fixed。18轮codes JSON SHA完全相同
d5cc931114e668f1c617eaeb7eb490eeda3a60af327e5dd867109231c038f65e，均41帧，
实际local_fixed返回should_continue=false。PCM SHA并非全部相同；官方codec按墙钟
动态调整分块，本轮未记录所有分块，不能仅凭波形变化归因RNG或CUDA不确定性。

单独LCG控制案例用与Rust相同的标量LCG公式替代rng对象，不改官方生成循环；
得到32帧/2.56s。又发现固定官方ONNX builder没有前置im_start，而Rust当前builder
遵循已验的PyTorch prompt fixture。第三个独立输入控制案例在原builder结果前仅添加
im_start行，仍使用LCG，得到34帧/2.72s（1+1）；没有匹配Rust DirectML的41帧。
新探针记录实际normalized text/token IDs，逐个对照官方SentencePiece：
[3985,10445,10364,2887,1809,12749,12887,10382]完全一致。不能把一行输入差异当成
全部差距原因，也不静默改生产prompt或把控制案例替代原官方基线。

真实max_new_frames=1负例触发“no observed graph stop before frame budget”，没有成功round
或结果，进程清理通过。门禁依据未裁剪图返回的真实should_continue，而非返回帧数
小于预算的猜测。所有控制case、失败和默认基线均为分离的Job与不可覆盖输出目录。

辅助SenseVoice分别回读官方默认及prompt控制音频，均为“你好欢迎使用续龄”；与目标
专名仍不一致。与DirectML的“去灵”一起保留，不宣称听感/专名质量通过或据ASR判定
某实现正确。人工试听pending，同条件排名仍未建立，不能因1.102与1.159s的点差称达标。

### 可恢复入口及工程状态

仓库scripts/perf/moss_onnx_cuda_reference.py与已测v3冻结源相同；默认使用官方随机流，
LCG及im_start仅为显式控制参数。原始v1/v2/v3、pip报告、cuobjdump、原始codes/PCM、
completion/audit、所有Jobs和capture均保留在外部Windows ONNX证据根。
`perf reference --job <official-cuda-baseline-job.json> --output <全新目录>`恢复默认基线；
另有official-cuda-lcg-control-job、prompt-control-job、eos-negative-job，不能混表。
当前新增探针metadata位于加载/计时前，另冻tokenized-smoke，PCM仍与旧探针逐位相同。
本轮probe构建、Clippy和rustdoc(-D warnings)通过；没有新增产品provider或运行时优化。
修正旧DirectML记录中的包名、默认资源根与已退役profile脚本命令，历史AMD证据未改写。

本轮显式--skip real_运行20项普通测试全部通过，2项环境门控实模测试筛除；实模证据
来自上述独立xtask Jobs。py_compile、fmt、diff检查及86个本地文档链接检查通过。

## Rust 完整新分发：动态加载生产路径验证

没有将新provider拼入旧静态Runtime。保持rc.13/api21及Cargo.lock，直接构建专用探针：

```powershell
cargo build --locked --release -p talechime-backends --no-default-features --features 'moss-nano-candle,ort-cuda,performance-tools,ort/load-dynamic' --example performance_probe -j 1
```

moss-nano-candle提供现有探针依赖的CPU Candle库，实际请求明确model=nano/device=cuda，
不是nano-candle推理。没有新增默认/产品feature或改变发布配置。实际ort compiler-artifact
包含load-dynamic，专用exe SHA
5d89bc8733828ec865e5d97a06dd49479c3d1ee22ed20e491c382bec9eb2106c。
dynamic-probe-build.json及原始Cargo JSONL固定有效features与二进制身份。
后续新增测试仅在cfg(test)编入，未改变此冻结探针的生产算法。

运行明确ORT_DYLIB_PATH到完整PyPI1.28.0/capi/onnxruntime.dll，PATH指定CUDA13
bin/x64及cuDNN9.14 CUDA13。独立模块审计读取实际子进程命令行和模块表，确认
onnxruntime.dll、providers_shared.dll、providers_cuda.dll全部来自同一分发；实际
cuBLAS13来自CUDA13，cuDNN为9.14.0.64，未加载官方Python环境里的Torch cuDNN9.10。
native-dynamic-loaded-modules.json及module-hashes.json保留PID/路径/版本/SHA。
该独立审计运行不混入正式计时。

### 实际Registry/Backend基线

复用冻结Rust xtask与薄native桥接，所有计时和音频来自原performance_probe的完整
Registry/Backend/SynthesisStream路径，fixture=false、Weiguo、无接续，同原中文文本。
桥接原样传事件，轮数/时限/输入校验/进程组/清理由xtask拥有；结构仍分类
external_reference_not_production，不将它改写成Native perf run的收据或report输入。
这里“生产路径”指被执行的API，非默认分发已修改或正式发行已验收。

native-dynamic-cuda-baseline-run：三独立进程各1+5，18轮正常结束/cleanup：

| 进程 | 热均值 s | 首PCM均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 0.67245522 | 0.07004652 | 2.8 |
| 1 | 0.70196578 | 0.07125324 | 2.8 |
| 2 | 0.70825116 | 0.07000584 | 2.8 |

均值0.69422405s、首PCM0.07043520s、RTF约0.24794。18轮PCM SHA均为
3cdb198c371fc7ae5dadf31ffb556f51bc63ddfd919ce6307a1630b15fec49a6。
模型加载3.071–3.134s，资源验证0.367–0.369s；reference_prepare未独立暴露，保持null。
Job commit峰值3.255/3.285/3.296GB，非RSS/VRAM。公开Pinned42派生到请求/attempt后
再给Runtime，不能与此前开发探针直接Some(42)混同。官方默认音频3.28s且随机流、prompt、
codec分块/cuDNN版本不同；不将0.694与1.159s点差当作同条件提速或达标证明。

### CUDA状态与负向验证

新增显式ignored测试real_cuda_queue_cancel_voice_switch_and_failure_reuse，要求本地
TALECHIME_NANO_ONNX_ROOT，不下载模型。LocalSet上加载实际CUDA backend、direct seed42：
五次完整Weiguo请求共用同一owner，各261120个F32位模式完全一致（2.72s，区别于公开
Pinned42生产case的2.8s）。覆盖立即丢弃接收端、收首PCM后观察容量1队列len=1再取消、
缺失音色导致InferenceFailure后复用、Weiguo/Adam/Weiguo；Adam不同且有效，回A一致。
“队列满已观察”不等于证明send正阻塞；missing-voice为真实验证失败，不是GPU内核
故障注入。所有完整请求必须有限PCM、正确48k双声道格式与真实End/Eos。
测试8.72s通过，冻结dynamic-state-build.json、cuda-tests.rs与原日志可恢复。

另将ORT_DYLIB_PATH显式设为不存在的文件：真实Native子进程加载失败，0成功轮次，
父进程cleanup通过；上游loader panic导致native非零退出（3221226505），没有CPU回退，
不能把它描述为友好的应用错误。首次普通测试漏带此变量时则拾取系统ORT1.17.1并
以BadVersion拒绝；带正确路径重跑27普通测试通过，4项real筛除，没有放宽版本门禁。
动态构建的运行库必须在首次ORT API查询前固定，不能依赖默认DLL搜索路径。

辅助回读本次Native音频为“你好欢迎使用续铃”，专名仍需试听，未宣称质量达标。
动态组合Clippy(lib/tests/example)及rustdoc -D warnings通过；普通测试不下载模型。

### 恢复和剩余工程

使用同冻结xtask `perf reference --job <native-dynamic-cuda-baseline-job.json> --output <全新目录>`。
Job固定ORT_DYLIB_PATH/RAYON及PATH，并断言后再启动Native；保留完整运行库目录、CUDA/
cuDNN依赖和16原生产资源。独立状态测试需同运行库变量、TALECHIME_NANO_ONNX_ROOT，
运行dynamic-state-build.json的exe及完整测试名 --ignored --nocapture。
当前xtask perf build只处理crate feature，不能仅放宽字符串过滤就声称验证了依赖
ort/load-dynamic；本次直接Cargo构建并检查ort实际artifact。Native perf run的通用
环境身份也尚未纳入ORT_DYLIB_PATH，所以使用显式环境断言的reference桥接，避免
Resume误复用不同DLL路径的结果。待补完整动态分发的正式构建/打包/原生收据支持，
默认旧静态分发限制仍保留；没有将专用探针成功等同于发行配置已修复。

## 输入与原始PCM边界对齐（同日继续）

原官方默认NumPy基线不覆盖。新增显式LCG控制，将seed改为生产实际截断值3560097011，
保留im_start输入控制，得到35帧/2.8s，与Native公开Pinned42案例长度一致。
official-cuda-matched-input-baseline-run三进程各1+5：

| 进程 | 热均值 s | 首codec PCM均值 s |
| --- | ---: | ---: |
| 0 | 0.99800188 | 0.05951980 |
| 1 | 1.02091044 | 0.05637448 |
| 2 | 0.98403264 | 0.05746086 |

18轮均35帧、实际should_continue=false、cleanup通过，codes JSON均为
69bb6c39f97d2d69e5ac86ac623fb17257185774921eecc59fd13927627a1dba。
这是“输入/随机流匹配控制”，不是默认官方配置或所有执行条件已匹配。
Native0.694s与上述约1.001s的点差不作为单项新优化收益或全部模式达标声明。

### 不同交付边界的波形不能直接归因模型

初始比较完整SynthesisStream WAV与官方原始PCM：同134400帧双声道，max abs0.05801252、
RMSE0.00293371、cosine0.99854264。先前将差距缩到codec的解释过早：生产producer
在发包前执行SilenceProcessor与LoudnessAligner，SynthesisStream不是未经处理的模型PCM。
该比较不是同一数值边界，也没有获得新的质量容差或更改生产处理。

依次做独立控制并保存原输出，避免把假设写成修复：

- 官方codec固定3帧：codes与自适应模式相同，raw PCM差距仍约RMSE0.00292811。
  固定3帧未解释主要差异，且改变首音频块大小，不能替代原官方流式能力结论。
- 干净codec进程不导入Torch，直接调用原CodecStreamingDecodeSession、实际cuDNN9.14，
  冻结同35帧codes，每块3帧。与带Torch的完整参考差约max0.000812/RMSE0.00003363；
  环境和入口一起改变，不能声称仅cuDNN版本导致这项差别。
- 同一原codec类以薄Session代理改用GPU状态I/O Binding，输出与普通Run逐位相同；
  再把graph优化All改为Layout，输出仍相同。这两项不是主要数值差距原因的证据。
- 直接查询C API默认CUDA options，并记录Python实际session options，use_tf32=1、
  fuse_conv_bias=0等一致；不因此推断每个kernel实际使用TF32。

临时Native诊断打印真正prompt、seed、codes，不改算法：实际seed为7330414270409262323，
截断值3560097011，两边226行prompt完全一致、35帧codes逐项完全一致。
捕获的是三次真实生成（参考准备暖请求及1+1轮），不能声称18轮Native codes均已导出。
诊断仍交付原SHA3cdb198c...的生产PCM，不能将诊断耗时混入正式基线。

随后在同一冻结诊断里只额外hash原始codec块。三次生成各12块、268800个F32样本，
逐块SHA全部与上述无Torch、cuDNN9.14、固定3帧的原官方codec输出一致；其拼接原始PCM
SHA为5321dbe4748b24ff23b2dd3768e2fc068a2524ef13db4f058e5e88adf73b9fb8。
因此本条件下原始模型音频已逐块一致，不能把处理后WAV的约0.058差距算成模型误差。
证据：native-official-code-comparison.json、native-official-raw-chunk-comparison.json，
原prompt/codes/块摘要日志和对应冻结exe/source均在外部证据根。

一次完整Python参考预加载cuDNN根DLL的控制仍得到原Torch环境输出；根DLL路径显示9.14
不足以证明所有组件/算子环境已对齐，未建立其因果解释。该预加载参数未保留在仓库
adapter，v6冻结实验仍留证；不将它称为成功的完整库匹配基线。
当前仓库adapter对应已实测v5：默认seed42/原官方随机流和自适应codec不变，仅可显式
选择seed、LCG、im_start与codec块预算控制；原默认基线及每个控制均分离。

两个临时Native日志钩子已彻底撤回，runtime.rs SHA恢复本轮前版本，没有残留开关。
新adapter py_compile、格式/diff检查通过；Rust运行时代码恢复原SHA，沿用上一节已过的
27普通测试、CUDA状态测试、Clippy/rustdoc证据。人工试听和专名回读问题仍pending。
下一步需用同一原始/处理后边界进行完整官方条件对照，并补正式动态构建/打包收据，
而不是放宽误差阈值或删除生产响度处理来获得容易通过的比较。

后续xtask已纳入显式ORT路径与同目录provider内容身份，并实测同路径换库拒绝Resume、
恢复原字节后复用成功；见[运行库身份回归](../platform/windows-ort-runtime-identity-2026-10-24.md)。
这关闭上文所述环境身份缺口，不改写旧工具/Job；依赖feature的正式build收据及打包仍待补。


后续标准 `perf build` 已支持并核实 ORT 依赖的 load-dynamic feature；正式 Native
3×(1+5) 全部通过，均值0.76722446s、音频2.8s、18轮PCM一致。构建/任务/报告、
计时边界及与早期桥接批次不可直接归因的限制见
[动态构建后续验证](../platform/windows-ort-runtime-identity-2026-10-24.md)。
动态分发打包、完整官方条件排名和试听仍待完成。


### 整套 cuDNN 控制与反向对齐（Windows RTX 5070，同日）

官方固定 checkout 的 onnx_tts_runtime.py 顶层导入 torch/torchaudio，后者会参与
DLL 加载。只预加载 cudnn64_9.dll 根库的旧控制不能证明各组件已对齐。本轮外部
v7适配器在导入官方源码前显式加载 cuDNN9.14 CUDA13 的8个 DLL，并计划在合成后
检查每个实际路径；官方源码未修改。结果在 Torch 导入阶段报 WinError127，指向
Torch 自带 cudnn_engines_runtime_compiled64_9.dll，0个模型轮次、cleanup=true。
这否定当前环境下此预加载方案的可用性；不是所有 Torch/cuDNN 组合的普遍结论。

反向控制保留正式 Native 探针二进制及完整生产请求，只把 Torch/lib 放在进程 PATH
前部；ORT整套库不变。标准 xtask 管理1进程、1预热+1热轮，模块监视确认 cuDNN
根库、ops、graph、heuristic、precompiled/runtime_compiled engines 都来自 Torch/lib
9.10.2.21。CUDA13 cuBLAS与CUDA12 cuBLASLt亦同时出现，故不是严格单因素 cuDNN
因果实验。两轮成功、cleanup=true、音频2.8s，PCM均70737f343d5659c02f222b0b334b652bcafd61d684626ca1c145915a6e386043。

同二进制9.14基线与此9.10控制的生产F32 WAV均134400×2、48kHz，最大绝对差
0.0009553581、RMSE0.00003822895、cosine0.9999987038。热轮0.8220615s、首PCM
0.088963s仅作诊断，单热轮且模块监视运行，不用于速度排名。未约定数值变更质量门槛，
不推广此库选择、不替换原输出基线、不声称试听或完整官方条件验收通过。

首次 Native 调度因 ORT_DYLIB_PATH 字符串斜杠与收据不同被拒；保留失败目录。
随后从 Job 读取原字符串，run2成功。工具当前有意固定字面路径，等价文件路径也不能
静默改写旧条件。没有将失败任务目录覆盖为成功。

外部证据：official-cuda-full-bundle-control-job.json及对应run的stderr/capture；
native-cudnn10-control-job.json、native-cudnn10-control-plan.json、
native-cudnn10-control-run2/results.json、native-cudnn10-control2-loaded-modules.json；
cudnn-bundle-control-comparison.json含逐组件SHA和误差。ORT自身bundle指纹不包含这些
外部库，不能把一致的ORT指纹误称为完整执行环境相同。下一步应在同一组件组合下
补模型原始PCM边界对照，或选择可独立固定的官方环境；现阶段不宣布排名达标。
