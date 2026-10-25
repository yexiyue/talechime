# Qwen Windows 生产路径与官方 CUDA 对照（进行中）

设备：Windows 11 / Ryzen7 9700X / RTX5070 12GB，驱动616.56。
证据根 `D:/workspace/talechime-benchmarks/windows-qwen-20261024`。
已完成0.6B与1.7B CustomVoice、0.6B Base双方基线及1.7B Base生产调查；VoiceDesign辅助路径与各模式完整质量/状态验收仍待补。

## 源码、构建和资源

生产构建基于 main `42a7887e28f095684266eac43a68bd8fcc303d11` 和收据记录的 dirty
worktree，不把旧 `e591c94` 探针冒充当前程序。`production-build.json` 冻结可执行文件：
SHA256 `3f5edbe481538f162fea196232a0ef52b325e6c55c26f2b47ed93c87f163051a`。
features 为 indextts、indextts-cuda-cudnn、performance-tools、qwen、qwen-cuda；
这个组合统一启用 Candle cuDNN，不能冒充无 cuDNN 的单 Qwen feature 构建。
借用已核验的 IndexTTS Cargo 缓存，无同时编译/推理。首次构建 LNK1181 找不到
cudnn.lib，补进程级 LIB 后成功；日志仍有 CUDA13 已知 LNK4098，未屏蔽警告或
改为静态 CRT，也未宣称该风险已经消除。失败日志和成功重试日志分别保留。

0.6B CustomVoice revision `85e237c12c027371202489a0ec509ded67b5e4b5`。
生产目录仍是 `~/.talechime/resources/qwen/`；固定 revision 目录有同身份副本。
8个文件通过生产清单，runtime 再独立核验，prepare 使用的旧 catalog 仅做只读
资源校验。`custom06-resources.json` 和新 probe 的 prepared 事件保留完整身份。
主模型1811626576 bytes，SHA `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`；
codec682293092 bytes，SHA `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`。

## 原生正式基线

文本“你好，欢迎收听今天的故事。”，Uncle_Fu，中文，无 style/参考/接续。
生产 SynthesisStream，Pinned42、每轮新状态（call0），内部 seed
7330414270409262323。语义 temperature0.9/top_k50/top_p0.9/penalty1.05，
声学贪心；max_duration30s/375帧未缩短，显式 chunk_frames20。
Talker/CodePredictor BF16，codec F32。默认 CP/decoder 图保持启用。

1+1短测成功后，3独立进程各1预热+5热轮串行执行。全部18轮成功、三个进程
cleanup_verified=true；各轮PCM与短测相同：
`f75dedfdba696767ae951415113c28302afffd2b5a3519f7563891aff520d337`。

| 进程 | 热合成均值 s | 首 PCM 均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 1.08541432 | 0.62626156 | 2.19 |
| 1 | 1.13363640 | 0.66507982 | 2.19 |
| 2 | 1.12218914 | 0.65247722 | 2.19 |

Rust xtask perf report 重新审计原始结果，mean-of-means：合成1.11374662s、
首PCM0.64793953s、RTF0.50856010。正式结果 custom06-baseline-run，离线报告
custom06-baseline-report；15/15热轮有效。这是当前版本绝对基线，没有新增运行时
优化，不从历史0.6B/1.7B报告直接计算此次增益。未完成人工试听、回读和实际播放验收。

## 官方环境与短测

官方仓库固定 `022e286b98fbec7e1e916cb940cdf532cd9f488e`，工作树干净。
独立 python 环境使用其要求的 Transformers4.57.3、Accelerate1.12.0；
Torch/Torchaudio2.8.0+cu128 从既有隔离环境只读共享。tokenizers0.22.1、
hub0.36.0；onnxruntime1.23.2 CPU 包仅满足被导入的25Hz模块依赖，本次12Hz
主模型与codec用 PyTorch CUDA，不把安装ORT当作它在执行GPU推理的证据。
未装 FlashAttention，显式 SDPA；不能代表 FlashAttention/编译版官方最佳速度。
SoX系统程序缺失的导入警告保留；本次无参考音频的调用未走 SoX 处理。

官方参考目录位于 `~/.talechime/resources/qwen/official-reference/`，权重在摘要校验
后以 NTFS 硬链接复用，额外 preprocessor_config.json 来自相同固定模型revision，
来源/响应头/SHA在 official-resource-preparation.json。没有编辑生产资源清单。
adapter 在加载阶段限定覆盖 tokenizer dtype=F32，避免官方BF16参数连带转换codec；
主模型数学实现、采样和解码源文件保持原样。此配置是匹配原生精度的官方路径，
区别于官方默认全BF16。43个输入文件连同冻结脚本、源码和元数据写入 xtask reference Job。

`custom06-official-smoke-run` 的1+1短测成功、cleanup_verified=true，主BF16/
codecF32断言通过。热轮5.690682s，音频4.24s，PCM摘要两轮一致：
`af80b99c3934bbdb364ecd9a1fa307cf74057102f0873165db8d060dcfb0c009`。
adapter 检查返回codes非空且少于375帧、PCM有限非零；没有独立抓取原始EOS token，
也没有完成内容完整性验证。官方API整段生成后解码，因此首PCM=完整耗时，不能
与原生20帧流式首PCM直接排名。4.24s与原生2.19s输出长度不同，固定seed不保证
跨框架随机轨迹相同；不能用两者耗时之比宣称公平提速。

## 官方正式基线完成

`custom06-official-baseline-run` 为3独立进程各1预热+5热轮，18轮全部成功，
三个进程 cleanup_verified=true，全部PCM摘要与官方短测一致。官方输出4.24s不变。

| 进程 | 热合成均值 s | 音频 s |
| --- | ---: | ---: |
| 0 | 5.39924098 | 4.24 |
| 1 | 5.35867868 | 4.24 |
| 2 | 5.47359708 | 4.24 |

mean-of-means5.41050558s，RTF1.27606264，整段首PCM与合成耗时相同。原始轮次、
输入摘要、环境、prepared与capture由Rust xtask reference保存并审计；表格从它的
results.json.processes用PowerShell复算，脚本汇总在official-baseline-summary.json。
当前perf report只接受原生Attempt数组，尝试读取reference结果对象被明确拒绝，未生成
可用官方HTML报告，也没有将reference改写成生产Attempt冒充同一口径。后续需补齐
Rust参考结果统计/报告入口。

本轮结论是双方基线完成，尚未达成同内容/同负载性能验收。输出长度接近两倍，RTF
也不能完全消除输入准备与解码工作量差异；不能把合成均值的比值当作公平速度增益。
下一步先核对文本token、prompt布局、实际生成codes/终止原因及内容，再拆分Talker、
CodePredictor和codec阶段成本。不要为了长度接近而缩短预算或更改采样规则。

## 文本入口与分词初查

本轮短句没有布局噪声，生产 normalize 不改变正文；中文启发式与显式zh一致。
官方包装 assistant role 后取正文切片；Rust直接编码正文并单独构造role，静态核对
二者均为role三token、think/think_bos/中文2055/think_eos、Uncle_Fu3010、pad/bos，
首文本token叠加codec_bos，剩余正文后追加tts_eos。官方non_streaming_mode=False
与原生该布局对应。这里只核对结构和ID，尚未比较设备上的prefill embedding/logits。

发现Rust `Split::new(PRETOKENIZE_REGEX, …)` 的参数实际转换为SplitPattern::String，
tokenizers0.22.2源码会regex::escape，不能因变量叫REGEX就假设执行正则。
但这不足以证明它造成当前输出长度差异：Python tokenizers0.22.1复现当前字面模式、
显式Regex模式，并与官方Transformers4.57.3/AutoTokenizer（fix_mistral_regex=True，
与官方包装入口一致）对照，当前短句及中英/日期金额/换行/字母数字混排共5例ID全部
相同。短句8个正文ID为108386,3837,100437,50009,49187,100644,104413,1773。
证据tokenizer-pattern-matrix.json保留版本和官方pre_tokenizer状态。
这不是实际Rust可执行分词回放，也不证明任意文本一致；未修改生产分词器，更没有
将这个疑点宣称为性能或准确性修复。后续若修正API用法，应先找到可区分夹具并运行
实际Rust对照，再单列兼容/准确性变化；当前先继续prefill、生成终止与内容核对。

## 官方终止与阶段诊断

新增仅在request.diagnostics=true时使用的参考适配器包装；未编辑官方源文件，
也未修改原生运行时。冻结脚本和Job分别为qwen_cuda_reference-diagnostic-frozen.py、
custom06-official-diagnostic-job.json；由xtask reference单独运行1+1，结果在
custom06-official-diagnostic-run。原正式Job继续引用此前独立冻结版本。

在官方Talker.generate与speech_tokenizer.decode边界同步并读取真实sequences。
热轮返回54个语义ID，最后一个2150（EOS），前53帧生成codec codes，未达到375帧
预算；输出4.24s。两轮PCM摘要均与正式官方基线相同，cleanup_verified=true。
至此该官方短句的实际EOS得到直接证据，仍不替代文字完整性或听感验收。

诊断热轮：完整5.21957060s；Talker.generate5.17159940s，codec.decode0.04422380s。
前者包含prefill、Talker逐帧、CodePredictor逐组、采样与框架控制，不能称为纯Talker
kernel时间，也没有分离CPU与GPU。此边界诊断支持优先调查自回归生成，当前codec
约44ms不足以解释秒级差距；不把诊断计时混进正式三进程均值。

静态核对双方min_new_tokens=2、EOS2150、末1024控制词区抑制（保留EOS）一致。
Rust使用SamplingContext内部PCG XSH RR 64/32，官方使用Torch随机流，跨框架同seed
不等于同随机数。下一实验应使用双方显式语义贪心控制来区分随机轨迹与模型数值/
prompt差异，并将其作为单独case；不改写当前随机采样生产基线。

## 贪心控制发现参考完成门禁漏洞

独立case保持同一正文/权重/精度/375帧预算/penalty，原生temperature=0，官方
do_sample=False，声学仍贪心。原生custom06-greedy-run触及帧上限，生产返回
incomplete，capture failure/cleanup_verified=true，没有可用热轮。

旧参考adapter的custom06-official-greedy-run错误地通过：返回375个语义token、
后处理codes374帧、29.92s音频，最后token1318而非EOS2150。旧条件只断言codes
少于375，不能证明正常结束。其原始success收据保留，但本记录明确将整个case
作废：不得用于排名、RTF或质量结论。该门禁错误属于参考adapter，不改变官方源文件。

修复为在原始Talker.generate结果上直接检查序列末尾EOS，失败时保存真实token数
和末尾ID。只在失败路径导出该工件，正常热轮不写诊断文件；新的检查包含一次
主机读取同步，旧正式基线不冒充新检查配置的性能结果。
custom06-official-greedy-eos-run已实机回归：375 token、last1318、ended_with_eos=false，
运行failure且cleanup_verified=true，未导出成功音频。Frozen adapter与Job独立命名，
不覆盖旧输入或失败证据。贪心控制因此也不形成可比较的正常完成负载；不能增加预算
或把触顶输出计为成功来强行比较。原随机采样的EOS门禁回归另记。

custom06-official-sampling-eos-run以新门禁运行原随机采样1+1，两轮通过真实EOS
检查，音频仍4.24s，PCM摘要与原正式官方基线逐位相同，cleanup_verified=true。
这同时验证了负例拦截与合法路径保留；不将这两轮诊断升格为新的三进程性能统计。
仓库外custom06-official-greedy-disposition.json绑定旧results.json摘要，明确将旧
false-success case排除出完成/性能结论，原始run内容不改写。

## 1.7B CustomVoice 生产基线与官方加载

复用同一冻结生产探针、相同短句/Uncle_Fu/语言/采样/375帧/20帧交付，模型切换为
1.7b-customvoice revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`。通过生产
资源门禁后运行1+1，再运行3×(1+5)，未重新编译或并行推理。主权重3833402552
bytes，SHA `38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137`。

| 进程 | 热合成均值 s | 首PCM均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 1.67608918 | 0.66453998 | 4.00 |
| 1 | 1.63474924 | 0.64977754 | 4.00 |
| 2 | 1.63841324 | 0.65530040 | 4.00 |

全部18轮成功且PCM与1+1短测相同：
`23dfbdabfbe03318d92f36b2f96eb6ebba16794333b97aca5ac2d7b63c9ae84b`。
三个capture cleanup_verified=true。原始custom17-baseline-run经Rust report审计，
mean-of-means合成1.64975055s、首PCM0.65653931s、RTF0.41243764。不是新优化增益，
也不是与输出长度不同的0.6B互相排名；试听、回读、style与其他音色仍pending。

官方沿用固定源码、Transformers4.57.3/Accelerate1.12.0/Torch2.8+cu128、SDPA，
主BF16/codecF32、修复后的真实EOS门禁。参考权重在清单摘要验证后硬链接，额外
preprocessor配置来自该模型同一固定revision。custom17-official-smoke-run在
Torch storage切片/Transformers meta权重加载时访问违例，无prepared或有效轮次，
cleanup_verified=true；不能单凭栈相似就认定与MOSS的1455完全同因。

另建python-pread环境仅加入Safetensors0.8.0，保留原环境不覆盖。新适配器在加载
范围内选择原生pread接口，沿用MOSS调查已通过的12张量CPU逐位IO夹具方法；
custom17-official-pread-smoke-run成功。实际runtime主BF16、codecF32、SDPA；
加载2.68439250s（单次预检，不当作加载性能基准）。真实序列43token，以2150结束，
code42帧、音频3.36s。热轮4.16996130s，PCM
`a1754ba72c829f23cba594b182c05782b2c0696779564d5a5fb21011aac96db5`，两轮一致。
这证明新环境/读取组合可运行，不单独证明原因在SF版本或pread；正式热轮关闭阶段
诊断，保留EOS检查。官方输出3.36s与原生4.00s不同，仍不宣称同内容性能达标。

官方custom17-official-baseline-run完成3×(1+5)，全部18轮通过真实EOS、PCM摘要
与预检一致，三个进程cleanup_verified=true。热合成逐进程均值4.12430012、
4.28134896、4.25679504s，mean-of-means4.22081471s，RTF1.25619485，音频3.36s。
首PCM为整段耗时。逐轮由Rust xtask reference审计，表格从其processes复算，保存
custom17-official-baseline-summary.json；没有将其对象格式冒充原生Attempt报告。

追加同环境读取对照custom17-official-sf08-mmap-run：同一python-pread解释器、
Safetensors0.8.0、冻结脚本与模型，仅改checkpoint_backend=mmap（关闭阶段诊断），
再次在Torch storage切片/Transformers meta加载处访问违例，无prepared/有效轮次，
cleanup_verified=true。这排除了“仅升级SF版本便足以解决本机加载”的解释；支持本机
该配置使用pread，不证明所有访问违例同因，不改系统页面文件，也不把失败加载用于
性能排名。预检诊断开关只影响加载完成后的推理阶段，不解释此次加载期差异。

本轮增加1.7B的可追溯双方基线和可运行官方加载方案，没有新增Rust运行时优化。
0.6B与1.7B的输出长度差异方向相反，更不能用一个固定音频长度校正系数排名。
后续分别补齐内容/数值控制、style及Base克隆变体，再以各自实际热点筛选候选。

## 1.7B CustomVoice 现有图优化的当前版本回归

同一冻结probe、权重、正文、请求与交付，仅切换两个进程环境变量：A将
QWEN_CP_GRAPH/QWEN_DECODE_GRAPH都设0，B均unset（当前默认启用）。条件环境
同步冻结进Job，不能通过改环境冒充原条件。1+1关闭图短测输出与默认PCM相同，
随后执行ABBAABBA八独立进程，每进程1预热+5热轮；测量期间无编译或并行推理。

| slot | 路径 | 热合成均值 s | 首PCM均值 s |
| --- | --- | ---: | ---: |
| 0 | eager | 3.15239958 | 1.24157492 |
| 1 | graph | 1.66372848 | 0.68264510 |
| 2 | graph | 1.72878792 | 0.70410292 |
| 3 | eager | 3.39276608 | 1.34021990 |
| 4 | eager | 3.26532376 | 1.30891386 |
| 5 | graph | 1.64488364 | 0.65626366 |
| 6 | graph | 1.65033754 | 0.66274694 |
| 7 | eager | 3.30103630 | 1.19969180 |

各臂进程均值再平均：eager3.27788143s，graph1.67193440s，当前4.00s输出负载
整链热合成减少48.9934%。进程均值范围分别3.1524–3.3928s与1.6449–1.7288s，
分离；范围不是置信区间。首PCM均值1.27260012→0.67643966s。保留全部slot，
相邻顺序平衡配对(0,1)/(3,2)/(4,5)/(7,6)均支持相同方向，不挑选低谷进程。

全部48轮含预热成功、8个cleanup_verified=true，PCM逐轮均与原生基线
23dfbdab…一致，音频均4.00s。没有独立导出codes，故不将PCM对照扩大为codes
或取消/状态复用的完整新验收。原始Job/run为custom17-graph-abba-{slot}-{arm}-*，
八份Rust perf report逐一审计并生成离线HTML/results.csv，原始每进程均值另存
custom17-graph-abba-summary.json。报告生成在全部推理结束后执行。

这是现有CP/decoder两类图在当前CustomVoice规格上的组合回归，不是本轮新增
运行时代码，也未拆分两者各自贡献。支持继续保留默认图；不能将该收益与历史图
百分比相加，或当作对官方同内容的速度比。cold load与首次录图成本仍单列在原始
prepared/预热事件中；此表是重复请求热路径，不泛化到首次启动、其他声线或克隆。

组件初筛另保持同一冻结程序与原始短句，分别1+1：仅关闭CP图
custom17-cp-off-run，热合成3.26939890s/首PCM0.98262140s；仅关闭decoder图
custom17-decoder-off-run，热合成2.26745790s/首PCM0.90283410s。四轮PCM均与
原生基线逐位相同，两进程成功清理。方向提示两类图均值得保留，但这是不平衡
单进程初筛，不分摊49%的组合收益，不把不同轮次差值当作单项正式贡献，也不据此
排除漂移。未新增运行时开关，沿用现有两个诊断环境变量。

## 1.7B Base 生产克隆基线

模型1.7b-base revision `fd4b254389122332181a7c3db7f27e918eec64e3`通过生产资源
校验。复用其自身已导入的custom:acceptance参考，不把CustomVoice或0.6B的prompt
移植过来。参考音频和voice.json先绑定摘要；生产1+1短测创建该模型的
prompt-encoder-rope-v2.json后，正式Job再绑定此缓存摘要。音频、转写和prompt均
留在用户资源目录，摘要工件写入base17-baseline-job.json，不提交入Git。

同一短句、中文、Pinned42、375帧/20帧交付、默认图；显式repetition_penalty=1.5，
对应生产克隆内部下限，不假称仍为CustomVoice的1.05。无接续；参考ICL与
x_vector_only路径不混用。正式请求warmup从磁盘加载已有prompt，热轮复用owner
内prompt；这不测首次音色编码、不同参考切换或跨请求取消恢复成本。

| 进程 | 热合成均值 s | 首PCM均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 1.68876516 | 0.97981822 | 3.07 |
| 1 | 1.69452082 | 0.98991436 | 3.07 |
| 2 | 1.68380920 | 1.00346426 | 3.07 |

全部18轮成功、三个cleanup_verified=true，PCM摘要
`ba986ac17849219842f2535ab9253658e643d38b3263c5ee2262d0e91b44aab9`。
Rust perf report审计后mean-of-means合成1.68903173s、首PCM0.99106561s、
RTF0.55017320；原始base17-baseline-run、离线base17-baseline-report和逐进程
base17-baseline-summary.json保留。不是相对历史数据的新提速，也未完成内容或试听。

官方克隆对照尚未运行，不能把CustomVoice官方结果代入。先明确第三个精度边界：
Rust try_load_speaker_encoder以F32 VarBuilder构建，官方extract_speaker_embedding
将mel显式转为self.dtype；因此主模型BF16配置可能连带改变speaker encoder的实际
输入/权重精度。后续需检查实际加载dtype与参考前端，不应先低精度加载再.float()
假装恢复原始F32。必须单列参考准备、prompt缓存与热合成，并匹配ICL、转写和实际
penalty后才能解释差距。当前新增的是生产Base基线，官方/0.6B Base/接续仍pending。

### 官方Base首个有效短测（精度差异明确保留）

base17-official-smoke-run使用同一Base固定revision、同一参考WAV和voice.json转写，
ICL、非x_vector_only、penalty1.5、相同正文与seed，主BF16/codecF32/SDPA/pread。
参考prompt在prepared前调用固定官方create_voice_clone_prompt一次，热轮只复用它；
没有每轮重新编码参考。输入文件、脚本与源码均冻结；没有修改官方数学实现。

实际加载speaker encoder为BF16，而Rust构建为F32。直接读取原始Safetensors
header显示speaker_encoder.*的76个tensor原本都是BF16，记录于
base17-speaker-checkpoint-dtypes.json；故不能笼统声称“官方丢失了原checkpoint的F32
权重”。这里的差别是计算精度，后续可以单列F32参考前端控制，不改生产结果。

1预热+1热轮均成功、cleanup_verified=true。官方参考codes86帧，speaker embedding
2048维；单次参考准备10.77530140s（包含首次框架/库准备，不是稳态均值），加载
2.87347390s，二者不计入热合成。热轮4.02208100s、音频3.36s，真实语义序列43token
以EOS2150结束、codes42帧。两轮PCM相同：
`ceacd0e3b0aca75a100a0e3fe7535824fe09a90648de1f01758fc3cda425eb6e`。
与原生3.07s音频和F32参考前端不同，不计算“公平提速”。这不是三进程官方正式
基线，也不证明参考embedding/codes数值一致；下一步先对齐或量化参考前端差异。

### 参考prompt数值审计

base17-official-prompt-audit-run使用独立冻结脚本，在参考准备后、热轮前导出官方
embedding/codes到仓库外reference-prompt.json；不导出到仓库，不把序列化计入
热合成。1+1成功、cleanup_verified=true，输出PCM两轮均与上述官方Base短测一致。
与已冻结原生prompt逐项比较，摘要在base17-reference-prompt-comparison.json：

- embedding均2048维，余弦0.9999973291、最大绝对差0.014942、RMSE0.000942625。
- codes均86×16，1296/1376个ID一致，69/86帧全部16组一致。
- 每codebook不一致数为0,0,0,1,3,2,5,3,5,7,11,5,9,9,9,11；前3组全部相同，
  但不能把后续残差组差异忽略为“参考codes一致”。

这些数值不是已约定的质量门槛，也不证明听感通过。参考codes由F32 speech
tokenizer编码，与随后独立提取的speaker embedding不是同一路计算，故不能把
全部codes差异归于speaker BF16/F32；单独提高speaker计算精度也不直接消除它们。
后续可向双方提供同一冻结prompt作为独立的生成器控制实验，明确跳过参考前端，
再比较生成路径；不能把共享prompt实验冒充官方完整参考编码链路的性能或质量。

### 共享prompt与贪心生成控制

新冻结适配器通过官方VoiceClonePromptItem接受原生缓存的embedding/codes和同一
转写，不调用官方参考编码器。base17-shared-prompt-job.json另绑定原生缓存SHA，
原正式/短测Job不覆盖。base17-shared-prompt-validation.json实测确认传入embedding
的F32位模式相同、codes逐项相同，官方对转写的正文token IDs与原生缓存也相同。
这是参考条件受控的生成器实验，不是端到端克隆性能结果。

原随机采样1+1成功、cleanup_verified=true，官方39帧、真实EOS2150、音频3.12s，
热轮4.01421320s，PCM两轮一致：
`13d4a7d6a73a1f5ca3a9abae3d73e04ae1fb19f8ba855805a60725edd6751435`。
比其原独立参考前端的3.36s接近原生3.07s，但长度接近不是生成等价证据。

进一步独立贪心控制：原生temperature0、官方do_sample=False，声学仍贪心，
penalty1.5与其余条件不变。base17-greedy-run原生1+1正常完成、两轮PCM一致，
音频3.06s，热轮2.51106840s；base17-shared-greedy-run官方1+1也通过严格EOS，
42帧、43语义token以2150结束，音频3.36s，热轮4.61563460s。两者各自成功清理。
这里无需重复CustomVoice的失败推断：Base该控制确实能正常结束，但输出仍不同。
两个单进程短测不形成正式速度排名。

共享参考且不使用随机采样仍有差异，排除了“仅参考前端或仅RNG”的充分解释，
不能据此断言某个框架不正确。后续需要同一输入下prefill/logits/逐步codes数值定位；
当前未导出原生生成codes，不能把输出时长差直接换算为某层错误。没有更改生产
算法、预算、精度或默认采样，也没有把该控制结果替代原生产随机基线。

### Prefill 捕获与磁盘阻塞

官方共享 prompt 贪心控制的诊断运行位于仓库外
`C:/Users/Administrator/.talechime/benchmarks/windows-qwen-prefill-20261024/official-prefill-run`，
冻结脚本与 Job 位于其父目录。仅诊断启用 `prefill_capture`，捕获首个多 token
forward；导出会同步设备并写文件，故本运行时间不作为性能基线。

实测输入 shape `[1,96,2048]`，三路 position IDs 均为 0..95，cache_position
为 0..95；导出前 9 位置 hidden 共18432值、末位置 hidden 2048值和 logits
3072值。1预热+1热轮成功、cleanup_verified=true，音频均3.36s，PCM均为
`91859960147b818b59f1dafb54b6c1ba2b8a672b084ac8b3065f4213492aff5d`，
与此前未启用此捕获的共享 prompt 贪心控制一致。此观察只验证本例最终输出，
不证明 hook 没有时间开销或跨框架中间结果一致。

当前源码已有显式本地模型的 ignored test `probe_voice_clone_prefill`。其本机
release CUDA/cuDNN 编译因 D 盘仅余约8MB报 os112，未生成可用的当前探针；
不使用旧测试二进制替代。原生将 prefix9 与 ICL87 分两次 prefill，官方一次96；
这只是待检验的数值差异假设，不是已定位根因。下一步在释放构建空间后编译该
测试，对照相同 prompt 的 prefix/末位置 hidden/logits，再决定是否需要逐层探针。

旧 MOSS 独立构建缓存约2.01GB的清理已完成内容检查和外存基准二进制 SHA
复核，但删除被工具自动审批以 `blocked by policy` 拒绝，未执行；模型、音频、
结果未删除。磁盘阻塞不等于该后端性能达标，也不改变既有基线。

### 恢复后的原生 prefill 数值对照

D 盘空间恢复后，当前源码的 ignored `probe_voice_clone_prefill` 已通过 release
CUDA/cuDNN 编译并实跑通过（1 test，4.18s；不是性能基线）。构建仍有既存
LNK4098 LIBCMT 冲突警告，未屏蔽也未声称解决。冻结 exe、SHA、原生张量及
对照摘要位于 `D:/workspace/talechime-benchmarks/windows-qwen-20261024/prefill-native`，
构建日志为其上级的 `prefill-build-resumed.log`。测试使用同一原生参考 prompt，
无下载；探针容量128帧仅用于 prefill，不作为生成预算控制或生产计时。

原生 prefix shape 为1×9×2048、ICL 后 offset96；官方输入1×96×2048。
对照官方上述冻结捕获，F32 导出值的统计为：

| 张量 | 元素数 / 完全相同数 | 最大绝对差 | RMSE | 余弦 |
| --- | --- | --- | --- | --- |
| prefix hidden | 18432 / 1589 | 1.75 | 0.05572946 | 0.99981189 |
| last hidden | 2048 / 178 | 0.25 | 0.03874620 | 0.99988254 |
| 首步 logits | 3072 / 465 | 0.21875 | 0.05197580 | 0.99993790 |

首步 logits 前5名 ID 顺序均为898,1431,1841,1826,476；原生值为
22.125,18.625,17.625,16.5,15.125，官方为22,18.5,17.5,16.5,15.0625。
因此首步未出现 argmax 翻转，但生成前已存在数值差异；不能把后续输出差异
全部归于随机采样或参考编码器，也不能仅凭高余弦判定质量通过。这些是诊断
数值而非已授权容差。分段 prefill / GEMM 形状仍只是候选解释，需要输入
embedding 或逐层对照才能分离组装差异、内核数值和分段计算影响。当前不更改
生产算法，不宣称首个生成 token 的完整声学 codes 或后续轨迹相同。

### 输入与合并 prefill 控制（诊断原型已撤回）

`D:/workspace/talechime-benchmarks/windows-qwen-20261024/prefill-inputs` 保存冻结
官方脚本/Job、成功运行、原生探针 exe/SHA、原生张量、comparison.json 与
native-diagnostic.patch。补丁仅在 cfg(test) 重建9位置前缀并与现有87位置ICL
拼接，使用现有 run_prefill_layers 一次计算96位置；不改变生产路径。该重建
不是从生产 forward 直接截获输入，故组装完全相同仍有此方法限制；后续严谨
归因应使用实际输入捕获或逐层探针，不能把本次结果称为唯一变量已完全隔离。
原生 ignored test 通过，官方1+1成功、cleanup_verified=true、两轮PCM与先前
共享prompt贪心控制相同。原型补丁已存于仓库外并从源码撤回，保留既有原生
探针，避免长期维护重复的prompt组装逻辑；官方可选输入捕获留在诊断适配器。

| 对照 | 元素数 / 完全相同 | 最大绝对差 | RMSE |
| --- | --- | --- | --- |
| 原生重建输入 vs 官方实际输入（全部） | 196608 / 162881 | 0.00390625 | 0.0001365994 |
| 同上前9位置 | 18432 / 11725 | 0.00390625 | 0.0001070235 |
| 同上ICL87位置 | 178176 / 151156 | 0.00390625 | 0.0001393011 |
| 原生合并 vs 原生分段：末位置hidden | 2048 / 147 | 0.375 | 0.04350514 |
| 原生合并 vs 原生分段：logits | 3072 / 843 | 0.25 | 0.05418570 |
| 原生合并 vs 官方：末位置hidden | 2048 / 136 | 0.3125 | 0.04554483 |
| 原生合并 vs 官方：logits | 3072 / 208 | 0.25 | 0.05403704 |

输入已经不逐位相同，简单合并也没有令末位置hidden RMSE低于原分段对官方的
0.03874620。不能从此推出分段不影响数值，但不支持“只需合并便对齐官方”。
本次不保留任何生产prefill变更、不设质量容差、不以诊断运行宣称速度收益。
后续定位应对比实际输入的text projection/codec overlay来源并定位首次语义或
声学选择分歧；数值小且首步排名相同不是全轨迹或音质验收。

### 独立单ASR辅助回读：共享prompt贪心样本

2026-10-24复用VoxCPM调查中新增的sensevoice_aux_readback开发探针，读取既有
base17-greedy-run原生与base17-shared-greedy-run官方的第1热轮完整F32 WAV。
不重新生成、不改音频；各输入WAV SHA冻结，识别器只接收音频，不接收原文或
参考转写。SenseVoiceSmall INT8固定revision2365bae…，按项目manifest校验，
默认CPU Session配置、线程4；这是单识别器内容辅证，不是双模型验证或听感验收。

两个样本均转写为“你好欢迎收听今天的故事”，与测试短句去标点后内容一致。
在这两份音频上未观察到短句缺字；不能将3.06s/3.36s长度差直接称为少读，也
不能据此判定prefill/logits误差无害、音色/音质相同或所有帧/codes一致。
原来的数值差异结论和人工试听pending保持不变，未改写旧性能基线。

原始cases、results和SHA证据位于仓库外
D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/aux-readback-qwen-*.json，
共同工具/模型身份见该目录aux-readback-build.json与aux-readback-evidence.json，
实现、负SHA用例与工程检查见[VoxCPM当前调查](../voxcpm/voxcpm-windows-production-2026-10-24.md)。


## 后续覆盖：Qwen 0.6B Base 克隆短句

同日Windows11 / 9700X / RTX5070，未修改Qwen运行时代码。当前工作树重新构建
performance_catalog/probe，收据current-catalog-build.json/current-production-build.json；
后者二进制SHA为5512dc0256a0a4dae0d36a0d067e3a7a1c11eebe4cf43ec208d449774604cafb。
features仍为indextts、indextts-cuda-cudnn、performance-tools、qwen、qwen-cuda，
保留原Qwen基线的cuDNN组合，不能标成无cuDNN单Qwen构建。catalog8m15s、probe1m32s，
两次Source guard通过；probe保留既存LNK4098警告。所有推理均在构建完成后串行执行。

### 身份与条件

模型revision5d83992436eae1d760afd27aff78a71d676296fc，8个文件通过当前生产manifest。
主权重1,829,344,272字节，SHA180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6；
codec682,293,092字节，SHA836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258。
完整资源收据base06-current-resources.json；实际请求再次独立校验同一身份。

参考沿用已有acceptance WAV（SHA45fc8a9410f8d9c212260160e4b814b17b03c0a9c74c179d84215e136cd582d7）
及准确转写，通过生产VoiceStore导入为本模型custom:acceptance06；旧CLI只负责准备
WAV/元数据，身份见base06-voice-preparer.json，不用于计时。未复用1.7B prompt cache。
首次原生合成生成0.6B自己的prompt-encoder-rope-v2.json，随后正式Job冻结WAV、
voice.json、cache三份身份。双方参考codes均86帧、speaker embedding1024元素；
不能将先前1.7B的ICL87位置误当本例参考codec帧数。

首次准备CLI误把模型叶目录当资源根，产生未用的嵌套音色记录；核对绝对目标、仅含
两份自产文件且WAV与正确副本一致后，用同一CLI删除该错误记录。正确音色保留，
原参考/模型不变，导入/清理日志均保留。官方0.6B独立布局在~/.talechime/resources/
qwen/official-reference/0.6b-base/REV下使用同卷硬链接，不改原权重；共享codec配置/
权重摘要一致后才复用preprocessor文件，base06-official-layout.json记录每项SHA。

文本“你好，欢迎收听今天的故事。”，中文、无style/接续，ICL参考每进程准备后复用。
原生Pinned42、call0，派生seed7330414270409262323；官方同一整数，但Torch与Rust
随机流不同。semantic温度0.9/top_k50/top_p0.9/penalty1.5、声学贪心，预算375帧；
官方适配器断言实际末token为EOS，不能以用完预算当成功。Talker/CP BF16、codec F32；
官方固定源码022e286b98fbec7e1e916cb940cdf532cd9f488e干净，Torch2.8.0+cu128、
Transformers4.57.3、Safetensors0.8.0原版pread（不使用MOSS的header IO实验轮子），
SDPA、未装FlashAttention。Rust CUDA13/cuDNN路径和Torch组件组合不同，不声称全库一致。

首次原生smoke进程实际成功、两轮PCM一致，但整体run因资源收据根字符串的正反斜杠写法
差异拒绝，不计作正式基线；旧结果不改写。新Job直接使用catalog的原资源根字符串，
保持既有校验，smoke-final和正式轮次通过。prepare_base06_native.py仅生成数据，
调度全由xtask完成；后续加exclusive-create，重跑旧stage会拒绝且原Job/plan SHA不变。

### 3独立进程各1预热+5热轮

| 路径 | 逐进程热均值 s | 组均值 s | 首PCM组均值 s | 音频 s | RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| 生产Native | 1.31025244 / 1.28348708 / 1.30340094 | 1.29904682 | 0.86074815 | 2.52 | 0.5155 |
| 官方batch | 3.24272538 / 3.26218776 / 3.28068548 | 3.26186621 | 3.26186621 | 2.56 | 1.2742 |

双方各18轮success，进程cleanup通过；原生PCM均14f92b5da20dc90d6441c474b1a1baaeab0a878c65f1fbf34d0dc9cfe93c6193，
官方均e19a532cc6919cc0e82b928387317cf168065450af34595c2defe4fc470db835。
官方public batch API返回整段，首PCM等于完整合成；不把non_streaming_mode=False
参数误称为已测原生音频流。原生是完整生产SynthesisStream，包含宿主后处理。
不将两者WAV直接逐位相减当模型数值误差，也不从相同seed推导codes相同。

原生每进程首个热轮1.587–1.621s，比后续1.194–1.266s明显慢，全部保留；原因尚未
定位，没有追加预热后删去旧慢轮或归因GPU频率。下一次性能调查应按请求序号及阶段
查该重复成本，当前统计继续使用全部15个热轮。

Native加载1.118–1.237s，reference_prepare为null，独立参考短合成warmup1.438–1.454s，
不能当已隔离参考编码；官方加载1.873–1.876s、参考准备1.226–1.231s。Job提交峰值
Native6,828,412,928–6,834,319,360字节，官方8,042,692,608–8,067,117,056字节，
为整个受控进程生命周期高水位，非RSS/显存或仅热轮峰值。逐轮/各阶段数据在
base06-summary.json，Native标准报告base06-baseline-report/index.html；官方保留
external_reference_not_production格式，不伪装成Native数组或复用其Source收据。

### 内容与结论

分别抽取一份热轮F32 WAV，复用冻结SenseVoiceSmall INT8/CPU4线程辅助探针。
两份均回读“你好欢迎收听今天的故事”，识别器不接收原文提示；输入WAV SHA和结果
在base06-aux-readback-cases/results.json，工具身份沿用VoxCPM的aux-readback-build.json。
这是单ASR辅证，不能证明音色、韵律或音质相同，人工试听pending。各自其余轮次PCM
相同，所以不重复跑同一识别器。跨框架参考数值、随机流/长度、GPU库和交付口径
仍有差异；本例观测Native耗时及RTF较低，不宣布完成严格克隆质量/性能等价验收。

本轮保留现有CP/decoder图，没有新增Qwen运行时优化；取消/满队列、跨音色、接续与
长文本状态门禁未在Base06补齐。当前cuDNN组合release all-targets Clippy -D warnings、
rustdoc -D warnings通过，日志base06-current-clippy.log/base06-current-rustdoc.log；
普通测试没有添加模型下载，也没有用未配置的可选测试冒充真模型验证。
VoiceDesign仍是单次参考生成辅助入口，需独立测其生产design路径，不能强塞普通
narration Backend任务。Base1.7的完整官方条件与状态验收仍保持原pending。

复现：先核对current-production-build.json、上述资源/三份音色收据和冻结环境，再以
xtask perf run/reference顺序执行base06-baseline-job.json、base06-official-baseline-job.json，
各使用全新输出目录。base06-evidence.json记录本轮构建、任务、结果和回读的SHA。


## 后续：Base06 / Base17 生产 owner 状态与预算失败门禁

Windows11 / 9700X / RTX5070，同一当前CUDA/cuDNN feature组合；新增
qwen/cuda_tests.rs，仅cfg(test)+qwen-cuda编译，两个用例默认ignored。没有Qwen运行时
算法/默认参数或依赖改动。使用既有Base06 custom:acceptance06、Base17 custom:acceptance，
TALECHIME_QWEN_STATE_RESOURCES明确指向资源根，普通测试不下载/加载这些模型。

各模型在独立Tokio LocalSet、固定CUDA owner上按以下次序验证，模型间串行释放：

1. 30秒预算、chunk_frames20、penalty1.5、Pinned42派生call0，收集完整原始PCM及End。
2. 发送后未消费即drop接收端；同owner下一完整请求与步骤1逐位一致。
3. 消费首PCM后等待capacity=1的队列实际len=1，再drop；下一请求逐位一致。
   这证明队列已满，不宣称观察到了正在阻塞的send。
4. 未知音色在Backend入口拒绝；下一请求一致。这不是GPU推理失败注入。
5. 将同一请求预算临时设为1秒，其他参数不变；经过实际生成返回部分PCM，随后报
   frame limit错误。错误之后继续recv直到通道关闭，确认没有追加PCM或正常End。
   恢复30秒预算，原owner下一完整请求仍逐位一致。

两个最终用例--test-threads=1通过（25.87s，仅正确性，不作为性能计时）：

| 模型 | 每次完整原始PCM F32位模式数 | 比较完整请求数 | 预算失败前部分样本 |
| --- | ---: | ---: | ---: |
| 0.6B Base | 63360 | 5 | 24960 |
| 1.7B Base | 76800 | 5 | 24960 |

全部收集到的PCM有限、mono24kHz。初版已通过，随后增加Err后关闭/禁止追加End的
检查并重编译重跑；最终日志base-state-cuda-tests-final.log，初版日志保留。
这关闭两种Base在此短句/参考/配置的未读取消、满队列取消、音色拒绝、实际预算失败
后复用缺口；跨音色A/B/A、接续、长文本和实际播放验收仍未完成。不把源预算失败
当成有效合成或加速样本，不把generation End当成实际播放完成。

本测试直接消费Backend原始PCM；Base06正式SynthesisStream WAV为60480样本，边界
包含宿主静音/响度处理，不能把两者样本数不同当成此次状态回退。各自同边界逐位
比较，未用后处理WAV替代原始owner状态验证，也没有设跨框架数值容差。

测试二进制冻结于base-state-test-build.json指定SHA目录；该收据保存HEAD、57份
相关Rust/CUDA源码和Cargo身份摘要。额外逐项核对两模型16个资源文件和6个音色/
缓存文件共22项均与原基线收据SHA相同，未改写参考。普通测试25通过、1 ignored、
3个real用例过滤；CUDA/cuDNN release all-targets Clippy -D warnings和rustdoc
-D warnings通过，日志base-state-ordinary-tests.log/base-state-clippy.log/base-state-rustdoc.log。
release测试仍有此前LNK4098警告，完整日志保留；fmt/diff及文档本地引用检查通过。

```powershell
$env:TALECHIME_QWEN_STATE_RESOURCES = 'C:/Users/Administrator/.talechime/resources'
# 配好CUDA13/cuDNN的PATH和LIB；先完成release构建，正式性能运行不与之并行。
cargo test --locked --release -p talechime-backends --no-default-features --features indextts-cuda-cudnn,qwen-cuda,performance-tools real_cuda_base -j1 -- --ignored --nocapture --test-threads=1
```


## 后续：四种 narration 变体状态矩阵与 CustomVoice 音色切换

同一Windows CUDA/cuDNN组合，将上述ignored Backend owner门禁用于CustomVoice06/17，
并同时重跑Base06/17。共用驱动显式区分参数：Base penalty1.5，CustomVoice为默认1.05；
其余为同短句、Pinned42派生call0、30秒预算、chunk20、无style/接续。测试仍只在
cfg(test)+qwen-cuda编译，无生产代码或默认图开关变更，不宣称记录了每次graph命中。

四个模型--test-threads=1串行通过（56.81s，仅正确性）；每个都包含未读取消、观察
实际capacity1队列填满后取消、入口未知音色拒绝、实际部分生成后的frame-budget失败，
以及错误后通道关闭、不能追加PCM/正常End，随后完整请求恢复逐位一致。

| 变体 | 正常A原始PCM F32位模式数 | 完整A比较次数 | 附加B |
| --- | ---: | ---: | --- |
| Base06 | 63360 | 5 | 未测跨参考音色 |
| Base17 | 76800 | 5 | 未测跨参考音色 |
| CustomVoice06 | 63360 | 6 | serena一次完整请求，与A不同 |
| CustomVoice17 | 96000 | 6 | serena一次完整请求，与A不同 |

CustomVoice的A为uncle_fu；在全部取消/失败用例后执行A→serena→A，B有限PCM且有End、
与A不相同，返回A又与最初A逐位一致。它证明此两种内置音色切换下的owner状态恢复，
不证明B音色辨识/内容或听感通过，也不是Base跨参考、Custom17 style或接续的替代。
四种模式的负向1秒预算均输出24960个部分样本后报frame limit并关闭，无正常End；
这些失败请求完全不计入性能基线。原始Backend PCM与宿主处理后的WAV仍分边界比较。

25项普通测试通过、1 ignored、5 real用例过滤；release all-targets Clippy -D warnings、
rustdoc -D warnings通过。编译仍保留LNK4098警告，未改为静态CRT或屏蔽。新冻结
all-variants-state-test-build.json记录独立exe/SHA、实际Cargo合并feature、57份相关
源码/manifest摘要，以及复核后的38个模型/参考/cache文件SHA；旧base-state测试收据
与其二进制不覆盖。摘要all-variants-state-summary.json、原始日志
all-variants-state-cuda-tests.log，工程日志all-variants-state-{ordinary-tests,clippy,rustdoc}.log。

复现使用此前资源根/CUDA配置，把测试过滤改为qwen::cuda_tests::real_cuda_，仍使用
--ignored --nocapture --test-threads=1。这些新门禁未重新计时或覆盖旧性能结果。
四个narration变体的此状态子集已验证；VoiceDesign辅助入口、Base跨参考/接续、
Custom17 style、长文本与人工试听继续pending。


参考前缀解码图的后续 ABBA、内存代价及状态验证见
[Windows 106 帧图候选](qwen-decoder-reference-graph-windows-2026-10-24.md)。

## 后续：CustomVoice 1.7B style 请求恢复（2026-10-25）

Windows11 / Ryzen7 9700X / RTX5070；仅扩展ignored CUDA owner测试，无生产算法
或精度改动。相同Pinned42短句及uncle_fu，style为“用温柔而平静的语气讲述。”。
带style的完整原始PCM为82,560个F32样本，与普通96,000样本请求不同；两者各自
完整End且有限24kHz mono。长度不同是不同输入条件的结果，不作为性能比较。

同owner两次完整style请求逐位一致；未消费style流即取消，随后普通请求恢复原PCM；
再消费style首PCM，观察capacity1队列确实len=1后取消，下一次完整style仍逐位
一致，最后普通请求再次恢复原PCM。该检查证明已测style输入的请求状态隔离，
不证明中文指令的语气符合度或人工听感，也未覆盖所有style、style推理失败及长文。

四种narration变体串行重跑通过（4 tests，60.79秒，仅正确性）。原有Base06/17、
Custom06/17的未读/满队列取消、未知音色、实际部分生成后的预算失败及恢复，以及
CustomVoice A/B/A音色切换均通过。普通原始PCM样本数仍为63,360/76,800/63,360/
96,000，负向预算仍为24,960部分样本后报错且无End。

冻结二进制SHA `ec41deb991e5de6701a73785fc39c9b804aa0544e6acbacd2c94ac2982a9a976`，
证据根沿用本报告外部windows-qwen-20261024目录，构建收据style-state-test-build.json、
源码副本style-state-cuda-tests-source.rs、原始style-state-cuda-tests.log。
恢复时使用该收据的exe，配好CUDA/cuDNN PATH与TALECHIME_QWEN_STATE_RESOURCES，
过滤qwen::cuda_tests::real_cuda_并指定--ignored --nocapture --test-threads=1。
这些状态测试未更新性能数字；VoiceDesign、Base跨参考/接续及人工试听仍pending。

本轮普通测试26通过、5忽略；相同feature组合release all-targets Clippy及rustdoc
均以-D warnings通过，fmt/diff检查通过。另复核38个既有模型/参考/cache文件SHA
不变，冻结54份相关源码/manifest，style-state-evidence.json共63个证据文件。
工程日志style-state-{ordinary-tests,clippy,rustdoc}.log分别保留，不复用旧门禁结果。

## 后续：Base06 / Base17 原始音频续接状态（2026-10-25）

同设备/权重/feature，仅新增ignored测试，不改生产代码。使用每个模型已完整生成
的原始PCM与原文“你好，欢迎收听今天的故事。”构造SpeechContext，不使用宿主
裁剪/响度处理后的WAV，更不将取消或预算失败的部分音频绑定完整原文。
续接文本“今天的故事开始了。”，沿用原固定seed、静态音色和参数。生产runtime
确实走context临时create_voice_clone_prompt分支，其参考编码不存入静态音色缓存。

Base06完整续接48,000个F32样本，Base17为44,160，均有限24kHz mono并有正常End。
每模型两次完整续接逐位一致；未消费续接流即取消后，静态参考原PCM恢复；消费
首PCM且实际capacity1队列len=1后取消，再次完整续接仍逐位一致，最后静态参考
再次恢复原PCM。本轮不比较不同文本或参考条件的性能，不声明跨音色身份、连续
多段漂移、上下文推理失败恢复或人工听感通过。

四种narration变体全部串行通过（73.58秒，仅正确性），同时回归原有取消、负向
预算及CustomVoice A/B/A与style检查。冻结exe SHA
`3e759133aa9e2f218dbd56b7ccbe3c25b633fb741ef6eb1d74b831003626199d`，
恢复使用continuation-state-test-build.json的exe及此前资源/DLL环境，过滤
qwen::cuda_tests::real_cuda_ --ignored --nocapture --test-threads=1。
原始日志continuation-state-cuda-tests.log；性能数字保持此前正式矩阵，VoiceDesign、
Base跨静态参考、长文/多段续接及人工试听仍pending。

本轮26项普通测试通过、5忽略；相同release feature组合all-targets Clippy与
rustdoc -D warnings通过，fmt/diff检查通过。38份资源/参考/cache身份重新核对
一致；54份源码与manifest已冻结，continuation-state-evidence.json记录63份证据。
工程日志continuation-state-{ordinary-tests,clippy,rustdoc}.log保留实际执行结果。

VoiceDesign实际生产导出与3×(1+5)官方基线现已验证，详见
[VoiceDesign Windows](qwen-voicedesign-windows-2026-10-25.md)；生产入口匹配计时与
取消状态仍待补齐，不将narration或普通测试耗时替代其性能矩阵。

## 后续：Base 两静态参考的 owner 切换（2026-10-25）

Windows11 / Ryzen7 9700X / RTX5070，固定Candle2266099d / CUDA13 / cuDNN9.14；
仅扩展ignored测试，无生产代码或默认采样/图策略改动。沿用原A参考，B为此前
VoiceDesign生成的完整同文本PCM16参考，SHA
`f9f9c4c5d47cf00538ba0b19895a3e85a45c1ecf116cf7d3381e252c0604595a`。
文件路径与预期SHA均由显式测试环境提供，不下载或覆盖数据。

各Base在加载owner前导入唯一custom:cuda-reference-PID-nanos记录，验证参考格式，
首次B请求确认不存在prompt-encoder-rope-v2.json，实际生产owner进行参考编码并
写cache。随后检查A恢复、B暖缓存重复、未读B取消后A恢复、消费B首PCM且实际
capacity1队列len=1后取消，再次完整B及A恢复。整个过程中缓存文件字节不变。

| 模型 | A原始F32样本 | B原始F32样本 |
| --- | ---: | ---: |
| Base06 | 63,360 | 59,520 |
| Base17 | 76,800 | 65,280 |

每个完整请求均有限24kHz mono、有正常End；B与A不同，各自重复与恢复逐位一致。
这是两个不同参考文件的缓存/owner切换验证，不据VoiceDesign描述断言年龄、性别
或跨说话人质量。测试只证明首次编码与同owner内存缓存复用，文件不变不能证明
新owner确实从磁盘cache加载。不同参考的输出长度也不计作性能差异。

临时记录仅在导入成功后持有Drop清理保护，正常路径显式remove并传播清理失败。
运行前后复核38份既有模型/参考/cache和B文件SHA均不变，临时目录均为零；未删
原始参考或用户音色。四种narration变体串行4 tests通过（90.79秒，仅正确性），
同时回归原未读/满队列取消、预算失败、单段续接、CustomVoice A/B/A及style。

冻结测试exe SHA `532004d0a6c0b63cb037d0aa624b2e355bfb8a63f44b76b0f89ab5b48e44e25f`，
收据cross-reference-state-build-receipt.json，原始cross-reference-state-real-tests.log，
运行前后资源/清理核对分别为cross-reference-state-before.json与-after.json。
复现设置原资源/DLL环境，另设置TALECHIME_QWEN_STATE_REFERENCE和
TALECHIME_QWEN_STATE_REFERENCE_SHA256，再运行收据exe，过滤
qwen::cuda_tests::real_cuda_ --skip real_cuda_voice_design_production_repeat
--ignored --nocapture --test-threads=1。VoiceDesign独立入口不混入此状态矩阵。

26项普通测试通过、6忽略，同feature组合release all-targets Clippy及rustdoc
-D warnings通过。当前关闭Base跨静态参考切换子集；新owner磁盘重载、多段漂移、
跨框架克隆质量、长文与人工试听仍pending，性能数字保持原正式矩阵。

随后已补齐新owner磁盘缓存读取：旧owner释放并join后，仅隐藏测试WAV副本，新
owner仍能从prompt文件恢复相同B并逐位恢复A；详见[磁盘缓存门禁](qwen-windows-prompt-cache-owner-2026-10-25.md)。
该记录也修正前轮外部扫描遗漏voice_目录前缀的问题，原始审计文件保持不变。

后续[三段接续状态验证](qwen-windows-continuation-chain-2026-10-25.md)补齐Base06/17逐段PCM重放及链中取消恢复；同进程Custom17初始化OOM保留为失败，独立复查通过，长链质量与试听仍pending。
