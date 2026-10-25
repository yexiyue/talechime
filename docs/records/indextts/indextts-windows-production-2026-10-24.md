# IndexTTS 2.5 Windows 生产路径调查（进行中）

设备：Ryzen 7 9700X、RTX 5070 12 GB、driver 616.56。main 为 42a7887，
准确 dirty source 身份见冻结 receipt；旧计算库结果不替代本次生产基线。
外部证据根 `D:/workspace/talechime-benchmarks/windows-indextts-20261024`。

当前状态：参考复用优化继续保留，官方GPT BF16 GPU路径已完成3×(1+5)基线及真实EOS
正反例；Rust当前F32热请求约1.524s，官方混合BF16/F32约1.462s。精度、长度、随机流
和输出契约未完全匹配，不宣称同条件达标。后文早期失败记录保留，最新证据见末节。

## 已完成准备

- 官方仓库 `D:/workspace/index-tts` HEAD 为 `d9e41aac89fd00b3d71497fddb287b7f24613712`，
  与 SOURCE.md 一致；存在未跟踪 checkpoints/config.yaml、reference/、runs/。
  正式对照仍须冻结实际配置和导入文件，不能仅靠 HEAD 证明全部身份。
- 官方 `.venv/Scripts/python.exe` 可导入 torch 2.8.0+cu128、transformers 4.52.1；
  indextts 来自该仓库。此次没有运行官方模型。
- 原转换目录 `D:/workspace/index-tts/checkpoints-converted` 的 12 文件、约 6.7 GB
  全部计算 SHA256，保存 converted-manifest.json。包括生产必需文件，但旧导出清单
  只有 tensor 计数，新文件摘要不能独立证明历史转换来源，后续结合 oracle 核验。
- C 盘仅约 200 MB：新增 `~/.talechime/resources/indextts/models/2.5/c39ce5ba-candle-v1`
  联接至原转换目录，不复制或删除原件；从该入口创建音色也会落在原目录。
  新 feature 的 `target/perf-builds/229e85a…` 联接至 D 盘 build-cache，临时文件亦放 D。
- `xtask perf build --probe performance_probe --features indextts-cuda-cudnn` 完成，
  CUDA 13.0/cuDNN 9.14，约 2m46s，仅既有 LNK4098。production-build.json 冻结 exe
  SHA256 `8961b91f8268a4e7fbe4710c8b0e46270292508408629cc002d914626af91061`；
  有效 features 为 indextts、indextts-cuda-cudnn、performance-tools。
  execution-info 启动通过，不代表模型 GPU 推理通过。
- 官方示例 voice_01.wav：48 kHz、单声道、117058 样本、2.438708 s，摘要
  `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`，
  reference-info.json 已保存。尚未生产导入，不声称转写已核验。

## 第一项待测假设：重复参考准备

生产 runtime.rs::generate 每请求读取 WAV、Reference::from_pcm、prepare_reference。
后者重算 w2v、mel、campplus、length regulator 并清空 reference_emovec。计算库
已有缓存因此不能在相邻生产请求间复用。正确的 2.5 官方入口 infer_v2_5.py
620–663 行按参考路径复用。初查误读了 2.0 的 infer_v2.py；在真正运行参考前已修正，
两者都存在缓存不代表可以混用入口或模型。
这是源码证据，还没有实际占比和收益，不能直接据此宣称优化成功。

下一步先用准确参考导入生产音色、冻结 Job，建 3×(1+5) GPU 基线，再诊断参考成本。
若确认缓存有价值，限定在模型 owner，以内容身份失效并验证同名替换、A/B/A、取消及
失败恢复。不能照搬官方仅路径键，不能把参考移出计时后冒充提速。保持有界流和取消。

## 转换数值核验与当前资源门禁

当前 perf::verify 没有 IndexTTS 分支，明确拒绝无生产清单/可信转换收据的测量；
冻结探针尚不能直接运行正式 Job。本次没有伪造 receipt 或把 fixture 当作真实资源。
另外 manifest 校验要求 canonical 文件仍位于 canonical resources_root 内，上一轮
home 下指向 D 盘的 model junction 会越界，不能用于该正式入口。后续应选择包含
真实文件的自定义资源根，保持路径边界检查；home 联接仅提供已有资源的逻辑入口。

为建立转换来源证据，从 Git e591c94 取出历史 exporter 到外部 frozen_exporter.py，
SHA256 `9a97af7776d2f6f432bfb4b7a1cdca8c32f33d7e4ebeccd215ac4038253ae740`。
使用官方 Python 环境运行外部 verify_conversion.py（SHA256
`754f6ee4b76025f5df17ca870227768fe2f70b3d88fcaa6da7f1c20f155005f4`），
只在 CPU 读取权重，不初始化模型、不下载、不重新写权重。

逐一核对 key 集合、shape、F32 dtype 和每个 int32 位模式：GPT 456、codec 243、
s2mel 284、CAMPPlus 937、BigVGAN 783、feats 4 个张量，全部一致；feats 包含
按原脚本计算的 sqrt(var)。BigVGAN config、tiktoken、w2v 权重/config 复制摘要一致；
OmegaConf resolve 后的官方 YAML 与转换 JSON 内容一致。日志和输入/输出 SHA 在
conversion-verification.log/json；这证明与本机原始 checkpoint 的转换对应关系，
并不代替正式生成数值/音频检查。原始 GPT 的 HF 下载元数据记录 revision
c39ce5ba981572cb187443877ff559dfb246ce63，ETag 与实际原始 SHA
43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85 相同。
其他输入也须保留来源核验，不能只凭张量对应就把新清单标为所有来源已验证。

生产音色导入、正式资源清单支持和完整请求基线仍待完成；本轮没有新增性能结论。

## 固定转换清单已接入；生产基线运行中

上述“待完成”为前一阶段状态。随后通过 Hugging Face 固定 revision 的 tree API
逐一核验 **13 个原始文件**：IndexTeam 8 个、CAMPPlus 1 个、BigVGAN 2 个、w2v 2 个。
大文件的实际 SHA256/size 与 LFS 元数据一致，小文件的 Git blob SHA1/size 一致并另记
本地 SHA256；四份 tree 响应、source-verification.log/json 和校验脚本保留在证据根。
这补齐了上一节只验证本机转换关系、未完整核实来源的缺口。

加入 `indextts/conversion-manifest.json`：固定来源 revision、原始摘要、历史 exporter
身份及 11 个转换输出摘要。仅开发 performance::verify 接入此清单，收据标记
`verified_conversion_manifest`，不把它当作可下载的上游模型。一般模型 prepare 的原有
非空检查未变；其他序列化版本不能自动冒充该转换。新增无模型测试校验清单覆盖完整
生产 FILES、无重复项及有效摘要。原来的损坏/缺失和 canonical 根检查继续共用。

选择明确自定义资源根：证据根 `app/resources` 联接到 `D:/workspace`，其中
`indextts/models/2.5/c39ce5ba-candle-v1` 联接原转换目录。canonical root 明确为
D:/workspace，真实文件在该根内；不放宽根边界、不复制 6.7 GB、不删除旧文件。
`~/.talechime/resources` 下的逻辑入口仍保留，正式 Job 使用上述自定义根。
此根覆盖整个 workspace，应只用于本次已知清单，不称为独立资源副本。

重新构建并冻结 verified-production-build.json：probe SHA256
`3ffbb67d712e2b91d918f1b178fd842ca13bf888642c585aba088473e200e2e3`；
catalog-build.json SHA256 `aa9049647e5dcf459f43553d94fd3eb73ed42f980eea413f02b2f788fb045655`。
通过 xtask prepare 只读校验全部 11 文件，resource-receipt.json 状态为
resources_verified、execution_verified=false。成功预检不是模型执行验收。

使用带 cli+indextts-cuda-cudnn 的当前 release talechime，通过生产 voices import
导入 `custom:acceptance`。参考复用 OmniVoice 已存储的 reference.wav 和原有转写，
不是上述历史 voice_01.wav；转写未重新回读确认，试听 pending。冻结 voice.json 和
reference.wav 摘要。baseline-job.json 冻结短中文文本、seed42、language=zh、F32、
CUDA/cuDNN、8 Rayon 线程和 500 ms 采样；基线按 3 进程×(1+5) 串行执行。
正式运行前全部构建已结束；后续官方对照必须使用同一新参考，不能复用旧示例计时。

首批 baseline-run 的 3×6 次合成及终止/清理成功，但父进程审计拒绝：prepare 使用
混合斜杠 root，手工冻结的请求使用反斜杠 root，resource_receipt JSON 字面身份不同。
不能把这批改成成功；原 run.json 含原始 Job，外部 baseline-job.json 已恢复为其快照。
修正后的 baseline-v2-job.json 直接沿用 resource_receipt.resources_root 原字符串，
另开 baseline-v2-run，不修改旧数据。首进程 preparation 摘要与新 Job 已核对一致。

首批诊断还显示 fixed seed 下 6 轮 PCM 摘要各不相同，音频时长约 2.585–2.592 s。
Options.seed 文档仅承诺采样 RNG；CFM::inference_with_cancel 另外使用设备 randn，
没有由该字段重置。CFM 噪声能解释 PCM 不固定，但没有证据把长度变化也归因给它。
后续保持输出候选必须用明确受控的随机输入/状态，单靠请求 seed 不足以证明逐位保持；
不得在性能候选中悄悄改变生产随机策略来获得更容易通过的 PCM 验收。

### 正式 baseline-v2 结果

3×(1+5) 全部经过父进程资源身份、参数身份、终止及音频工件审计，3 进程均
outcome=success、cleanup_verified=true。原始结果在 baseline-v2-run/results.json，
逐进程统计 baseline-v2-summary.json；失败的 baseline-run 不参与以下成绩。

| 进程 | 5 热轮均值 s | 首 PCM 均值 s | 音频范围 s |
| --- | --- | --- | --- |
| 0 | 4.09486070 | 4.09479852 | 2.585034–2.592154 |
| 1 | 4.28335108 | 4.28328752 | 2.585034–2.592154 |
| 2 | 4.13506696 | 4.13500396 | 2.585034–2.592154 |

mean-of-means 约 **4.171093 s**，整段交付；不是原生流式首块。每进程的 6 轮 PCM
均不相同，不能将本表当作逐位可复现的 fixture。请求含参考读取/重采样/重编码，
没有单独短请求参考预热。资源校验约 3.12 s、模型加载约 2.32–2.34 s，均单列于
preparation；reference_prepare_seconds 为 null，参考成本仍在合成中而非遗漏。
未单独测量前，不把两种成本相减声称参考编码耗时。音质试听/内容回读仍 pending。

本轮新增内容是资源来源与校验支持、正式生产基线，没有运行时提速。下一步在同一
参考和匹配参数上准备官方 F32 GPU 对照，并通过受控随机输入评估参考缓存候选。

### 本轮工程门禁

当前源码 CUDA/cuDNN+performance-tools release 后端库测试 **9 passed / 0 ignored**，
包括固定转换清单覆盖和既有损坏/缺失保留；all-targets Clippy `-D warnings`、private
rustdoc（含 examples）`-D warnings`、fmt 和 diff 空白检查通过。原生 probe、catalog、
带 cli 的 talechime release 构建通过，仅既有 LNK4098。日志为证据根下
backend-tests.log、backend-clippy.log、backend-rustdoc.log。
额外用当前 catalog 对 home 下跨根 junction 做负向实机检查，明确返回
`model file escapes resources root`，outside-root-negative.log 保存，未放宽边界。
普通测试未下载或加载真模型；上述 3×(1+5) 另为真实 GPU 验证，不混计普通单测。

## 官方 2.5 对照适配器与未成功的短测

新增 `scripts/perf/indextts_cuda_reference.py`，实际导入 pinned checkout 的
`indextts.infer_v2_5.IndexTTS2`，不使用 2.0 同名类。官方 stdout 进度重定向到 stderr，
stdout 仅保留 xtask 事件。HF/Transformers 离线，全部辅助模型已存在，不执行下载。
冻结 adapter 副本、实际上游 Python、配置/权重、文本/音色，共 230 项输入；用
xtask perf reference 的原生串行进程归属、超时、音频工件审计，不冒充 production。

参数与 Rust 默认匹配：F32 起步，25 diffusion steps、cfg0.7、duration1、normalization
开启、中文、top_p0.8、top_k30、temperature0.8、3 beams、repetition_penalty10、
max_mel_tokens1500、segment120、interval200ms。关闭 DeepSpeed、accel、compile、
BigVGAN custom CUDA 和 QwenEmotion；后者未用于本任务。随机 seed 采用当前 core
Pinned42 的首调用派生值 7330414270409262323。跨框架 RNG 不同，不能据同数值宣称
token/长度相同；CFM seed 覆盖差异另列。官方普通 infer 返回 int16，适配器在计时外
按 32767 转 F32 写证据；Rust 交付未量化 F32，输出契约有差异。

先跑 1 进程×(1+1) 短测，没有直接扩大到 3×(1+5)。如下均失败，**没有官方热轮成绩**：

| 目录 | 模式与定位 | 结果 |
| --- | --- | --- |
| official-smoke-run | 原 F32 | 加载阶段进程断开，无 terminal；Windows APPCRASH 记录 torch_cpu.dll / c0000005，不能仅凭此判为 OOM |
| official-smoke-diagnostic-run | F32 + faulthandler | 模型加载成功 13.38 s；参考 W2v attention mask 处报 CUDA out of memory，无 round |
| official-bf16-smoke-run | 官方 GPT BF16，其他阶段 F32 | 载入后 allocated 5.309 GB / reserved 5.400 GB；BigVGAN 附近报异步 CUDA unknown error；仅诊断，且此最早 BF16 Job 替换 adapter 输入时漏掉了原第 1 项 codec 输入，不能作为完整来源收据 |
| official-bf16-cleanpath-smoke-run | 修正完整输入；仅子进程 PATH 优先 torch/lib 并排除外部 CUDA/cuDNN | 相同 BigVGAN 附近异步 unknown error，清理 PATH 没有证明解决问题 |
| official-bf16-blocking-diagnostic-run | 正确完整输入 + CUDA_LAUNCH_BLOCKING=1 | 在 W2v from_pretrained 的 meta state load / torch storage slice 报 Windows access violation；不能把该次错误归到未执行的 BigVGAN |

原失败目录、capture.json、stderr、resource samples 均保留，cleanup_verified=true；
不修改失败结果，不将 prepared 当作成功合成。正式 BF16 Job 为 official-bf16-job.json，
按旧 adapter 路径匹配替换一项，完整输入保持，仍没有正式运行成绩。
BF16 与 Rust F32 精度不同，即使成功也只能作为明确标记的另一组参考，不静默替代。
关闭异步的运行仅诊断，不作性能计时。没有修改官方源码、驱动或系统设置。

额外独立进程仅加载同一 W2v F32 模型到 CUDA，通过，耗时 2.386685 s、allocated
2321979392 bytes，isolated-w2v-load.log 保存。这不能证明完整模型内存足够，也不能
证明所有访问冲突都是内存原因。本机物理内存约 16 GB；应继续独立核验加载峰值、
参考编码容量和 BigVGAN 调用。Python venv launcher 的根进程 RSS 不代表全部子进程
模型内存，不能用几 MB 的根进程采样声称官方内存很小；整卡显存也不等于模型独占。

adapter 的 Python AST 检查、git diff 空白检查通过；本轮未改变 Rust 运行时算术。
IndexTTS 官方对照仍未达成，因此不报告 Rust 比官方快/慢，也不声称该后端达标。

## Windows CUDA owner 参考复用

新增私有 indextts/reference_cache.rs，在现有模型 owner 内只保存最近一次成功准备的
PCM 摘要（32 字节身份和少量状态），不增加 GPU 权重/特征副本，不新增全局或持久
缓存。仅 Windows CUDA 启用；其他设备/平台继续每请求准备，不引用本轮收益。
每请求仍验证 VoiceStore、读取 WAV 并执行既有验证和 22.05 kHz 重采样，然后对实际
将送入 Reference::from_pcm 的同一不可变 PCM 快照计算 SHA256。没有另读文件取键，
不按路径/mtime 命中，保留同名替换语义。准备失败前清空旧身份，成功后才发布新身份。
缓存命中同时保留原先每次准备会清空的 emovec；不改模型算术、采样或生产 RNG。
因此收益覆盖重复参考编码、参考衍生特征和 emovec 的复用，不能全算成单个 W2v 算子。

### 受控随机状态的真模型验证

原生产 CFM 不由 Options.seed 重置，因此正式多轮 PCM 自然不同。没有为性能比较
改变这个行为。独立 ignored 测试使用测试拥有的同一 CUDA device，在每次生成前
重建 curand 状态（20261024），同时保持 host sampling seed 不变；真实调用同一
runtime::generate，配合容量为 2 的原始 PCM/End 通道。以下完整 PCM 位模式一致：
uncached A、首次 A、命中 A、同路径替换 B 与 uncached B、恢复 A、再次命中 A、
预先关闭接收端后的 A、无效 WAV 恢复后的 A、生成中取消后的 A。取消要求实际返回
模型的 cancelled 错误，不把因接收端关闭而直接跳过当成中途取消。该测试未额外
捕获 GPT codes，不宣称新增了逐步 codes oracle；它验证完整 PCM 和这些状态路径。
实际 Model::prepare_reference 的设备 OOM 没有故意注入，普通回调失败测试覆盖身份
失效；无效 WAV 是进入准备前的真实验证失败，两者分开报告。

冻结最终测试 exe SHA256
`7745d92868ee8baf73b76b9febb740e85b4d87545f69152f47f2e41872e0b013`，
日志 cache-real-pcm-final-0.log。诊断首请求约 4.0–4.6 s、命中约 1.44–1.57 s；
这不是独立进程吞吐验收。普通测试 11 passed / 1 ignored；该 ignored 真模型测试
已单独明确通过。CUDA release all-targets Clippy 和 private rustdoc（含 examples）
均 `-D warnings` 通过，日志 cache-clippy.log/cache-rustdoc.log。此前 cache-real-pcm-0.log
是未扩展取消场景的早期测试，不代替最终测试。

### 完整请求验收身份

基线为 verified-production-build.json，exe `3ffbb67d…`，保持原每请求准备；
候选为 cache-production-build.json，exe SHA256
`59a4b355a9eb3316c165f39dd9d4f03d9f34f237bd404f0e1b846fad442bace1`。
相同 features、请求、语料、音色/资源摘要、线程和 500 ms 采样；各版本源码身份
由各自 build/plan 保存，不覆盖旧 binary。脚本 run-cache-abba.ps1 仅串行调用
Rust xtask perf run，按 base/cache/cache/base/base/cache/cache/base 八独立进程，
每进程 1 预热+5 热轮。每个 subjob 的 process_index 保留原 0，不改原始收据索引；
跨 Job 的次序另存 slot。统计先算每进程均值，不混池或挑最快轮。
此负载是同一音色连续请求：预热填充 owner 的参考状态，热轮测复用；不代表冷请求、
交替不同参考或所有文本长度的收益。正式计时不与编译或其他推理并行。

### ABBAABBA 完成：保留同参考热请求收益

| slot | 路径 | 5 热轮均值 s | 首 PCM 均值 s |
| --- | --- | --- | --- |
| 0 | base | 4.21396740 | 4.21389866 |
| 1 | cache | 1.52596316 | 1.52589244 |
| 2 | cache | 1.55012686 | 1.55006636 |
| 3 | base | 4.18079048 | 4.18072522 |
| 4 | base | 4.18544502 | 4.18538234 |
| 5 | cache | 1.47886182 | 1.47881472 |
| 6 | cache | 1.49446106 | 1.49438394 |
| 7 | base | 4.09461856 | 4.09455940 |

两臂各 4 进程、20 热轮；48 轮（含预热）全成功，8 进程 cleanup_verified=true。
mean-of-means **4.168705365→1.512353225 s（−63.72%）**，进程均值范围
base 4.094619–4.213967 s、cache 1.478862–1.550127 s，分离。首 PCM
4.168641405→1.512289365 s（−63.72%），仍为整段交付。两臂每个进程音频均在
2.585034–2.592154 s，不通过短生成换提速；不把这些范围称作统计置信区间。
原始每轮仍有 CFM 随机差异，**不是 48 轮 PCM 相同**；输出保持证据来自前述独立
受控随机状态的完整 PCM/状态对照，未改变正式生产 RNG。结合显著热请求收益和
单 owner 有界状态的维护成本，保留该 Windows CUDA 优化。

原始八 run 的结果不改写，cache-abba-summary.json 记录 slot/各进程均值，
cache-abba-group-summary.json 记录分组。没有重编号原始单进程 subjob 来伪装单个
多进程收据，也没有将单测诊断计时混入本表。run-cache-abba.ps1 已保存完整复现入口；
须先核对硬件、请求、输入/二进制摘要，并使用全新输出目录。

cache-abba-telemetry-summary.json 汇总 346 个 500 ms 样本，包含启动、校验、加载、
预热、热轮及收尾，不是热轮专属：整卡显存峰值 7225737216 bytes、SM 时钟
367–2947 MHz、功耗 29.51–188.72 W、温度 40–60°C；进程采样 RSS 峰值
3812642816 bytes、private bytes 峰值 7324749824 bytes。整卡含桌面驻留，采样不等于
精确 allocator 峰值；本表不单独声称内存下降或完全独占 GPU。

最终 fmt、diff 空白检查通过；测试、Clippy、rustdoc 门禁见上文。此结果不等于
IndexTTS 全部场景达到官方：官方 GPU 尚无成功热轮、质量回读/试听 pending，其他
参考长度与交替音色负载仍待扩展。未发布 PR、issue 评论或发行。

## 官方GPU恢复、当前基线及差距定位

本节更新早期“没有官方有效热轮”的状态，不改动任何旧失败目录。
先运行独立官方BigVGAN F32：同一固定源码/权重、use_cuda_kernel=False、合成mel
[1,80,128]，CUDA_LAUNCH_BLOCKING=1，1预热+1热轮成功、清理通过。
`bigvgan-diagnostic-run` 的load1.43651s，GPU allocated450975744 bytes，
Job commit峰值3403898880 bytes；热forward约0.07196s。
这是合成mel组件测试，不是语音内容或IndexTTS基线，只证明本机该组件可单独执行。
不能把旧异步reciprocal栈直接认作该算子固有故障。

### 独立环境与加载改动

创建外部python-pread环境，不改官方checkout或原venv。保持Torch2.8.0+cu128、
Transformers4.52.1，通过只读pth复用原Index环境；仅将Safetensors0.8.0包与元数据
联接到已验证的Qwen python-pread环境，覆盖原Index环境0.5.2。
新环境实际import路径与safe_open签名均核实支持backend参数，不从其他venv推断版本。
权重、配置、参考与语料全部沿用原固定输入；Job增加新adapter、pth、SF源码/pyd/
元数据与Torch/Transformers元数据摘要。依赖的Qwen SF包是恢复资产，不能当普通
构建缓存删除。原模型及生产资源清单不修改、不绕过。

adapter仅在模型构造期间包装transformers.modeling_utils.safe_open，显式传pread，
记录实际打开的w2v-bert-2.0/model.safetensors（本机4次调用）。模型算术不改。
先以同步报错模式跑1+1成功，再关闭CUDA_LAUNCH_BLOCKING跑异步1+1成功，才扩大正式矩阵。
本机没有Index的“同SF0.8版本mmap vs pread”单因素对照，因此只称隔离环境+SF0.8/pread
组合恢复，不能单独归因SF升级、pread或某个错误栈，更不能外推修复其他模型。
启动预算10000000000 bytes是案例规划值，preflight仅快照，不是预留或充足性证明。

### 官方正式基线（GPT BF16，其余F32）

`official-pread-bf16-baseline-job.json` / `official-pread-bf16-baseline-run`，
3独立进程各1预热+5热轮，固定官方revision/配置/参考/seed与采样条件均沿用前述Job。
use_bf16=True只使GPT为BF16，semantic codec/CFM/BigVGAN仍F32；不开DeepSpeed、
accel、compile、自定义BigVGAN CUDA、QwenEmotion，不用同步报错模式作正式计时。
参考仍由官方在首次warmup中准备并缓存，reference_prepare_seconds保持null。

| 进程 | 热合成均值 s | 音频 s | 加载 s | Job commit峰值 bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.51389286 | 2.75156463 | 12.60883790 | 13439090688 |
| 1 | 1.42414928 | 2.75156463 | 12.47535210 | 13439176704 |
| 2 | 1.44919776 | 2.75156463 | 12.55429660 | 13407465472 |

组均值1.46241330s，RTF0.53148426，整段交付首PCM等于完整耗时。
18轮均success、三个进程cleanup通过；所有PCM一致：
`5cb9310d725103862692e8de9adac10ef0bcc9c43a98506fe30a45e52aa3c761`。
模型加载后的torch GPU allocated为5308669952 bytes；Job高水位约13.41–13.44GB
包含后代commit，不能称RSS或显存，也不能与启动前系统余量的另一时点作静态相加。
原始结果、runtime、capture、逐轮细节及汇总保留，不拿prepared当生成成功。

### 真实EOS及负向门禁

包装gpt.inference_speech原方法，不改返回值和采样；保留原始codes末尾标量。
18轮原始长度70、实际最后token8193，再进入有效round；不是凭后处理长度判断结束。
`official-pread-eos-negative-run` 将预算设为1，并启用仅诊断的解码前检查，真实返回
长度1、末尾992（非8193），因此在CFM/vocoder前明确拒绝，错误为
`no confirmed GPT EOS before decode`。parent失败、cleanup通过、无成功round或results。
这个提前检查模式没有用于正式计时；正式adapter在完整infer后核验同一EOS条件。
默认官方模型只警告触顶而继续，参考适配器不能将其后处理音频计为有效完成。

### 新导出策略下的Rust当前基线

沿用现有参考缓存实现，重新冻结CUDA/cuDNN+performance-tools探针，SHA256
`f7b4d3ef32454d5a3df3f976d8464a3f14d64c185c7ddd28be6f500ffc4a17c4`。
本次只更新测量探针WAV缓冲，不把导出改变算成模型优化；旧参考缓存ABBA证据原样保留。
plan.source来自实际不可变build，而非后写文档的工作区；条件/参考文件仍严格校验。
`buffered-cache-job.json` / `buffered-cache-run`，3×(1+5)：

| 进程 | 热合成均值 s | 首PCM均值 s | 音频范围 s | 平均RTF |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.50569990 | 1.50563816 | 2.585034–2.592154 | 0.58117323 |
| 1 | 1.55437390 | 1.55432428 | 2.585034–2.592154 | 0.59997387 |
| 2 | 1.51314766 | 1.51307654 | 2.585034–2.592154 | 0.58405696 |

当前组均值1.52440715s、平均RTF0.58840135；18轮全部成功且清理通过，仍为F32、
整段交付。CFM生产随机行为未改，不能宣称这些正式轮次PCM全部相同。
官方GPT BF16当前略快、音频更长；不能用本组差额宣称同精度已达标或把长度归一化
当作消除全部差异。旧栈全F32的异步CUDA OOM与下述新环境同步失败分别保留，
不能把旧错误直接推广为当前显存容量结论，也没有静默用BF16替代F32。

### 内容与阶段归因

复用冻结SenseVoice辅助回读工具及原固定ASR资源，原生缓存热轮与官方正式热轮各取
完整WAV，均识别为“你好欢迎收听今天的故事”。见official-native-readback-results.json。
仅是单识别器、两条完整PCM的内容辅助证据；不证明音色/韵律、全部随机轮次或试听通过。
官方int16返回后转换F32，与Rust原生F32交付契约的差异继续保留。

原官方打印的分段perf_counter没有在CFM/BigVGAN边界同步，不能直接与Rust同步stage
相减定位GPU差距。另开official-synchronized-stages-run（1+5），包装原GPT入口、
CFM.inference、BigVGAN.forward，在边界同步CUDA；六轮PCM与正式官方基线完全相同。

Native复用原二进制，经native_probe_diagnostic_adapter.py桥接到xtask reference所有的
进程组，传递原Native请求和轮数，要求INDEXTTS_PROFILE=1、RAYON_NUM_THREADS=8，
原样转发Native事件。Rust控制超时、采样、终止与工件校验；桥接无模型实现。
该结果分类external_reference_not_production，仅用于诊断，不能混入正式生产基线。
变量与请求在Job中冻结并检查；负向环境设为0时，在写native-request或启动Native子进程
之前拒绝，cleanup通过。正向native-stage-profile-run得到6份真实模型profile。

| 同步诊断阶段 | Rust F32热均值 s | 官方混合BF16/F32热均值 s |
| --- | ---: | ---: |
| GPT / 官方inference_speech入口 | 0.57741056 | 0.83455904 |
| CFM | 0.73630072 | 0.53449514 |
| BigVGAN | 0.37132362 | 0.05546826 |

这些是不同粒度的host同步诊断，不是严格匹配kernel对照：Rust含内层计时，官方只包
三个外层；dtype、生成长度、随机流与内部算法/布局不同。不要将所有嵌套key相加，
也不要把诊断耗时替换正式基线。官方实际记录cudnn_allow_tf32=True、matmul_allow_tf32=False、
float32_matmul_precision=highest；存储F32不等于已经证明两边所有卷积/归约计算精度一致。
本轮没有切换任何TF32选项，也不从标志推断每个实际kernel。

Native内部热点进一步为：FIR up0.08136934s、Snake0.07672034s、FIR down0.04008198s，
BigVGAN res-conv等细项保留原始JSON；warpers0.01019012s、sampling0.00055782s。
优先调查vocoder执行复用/常量及CFM estimator，不能照搬Nano多通道采样的阶段比例，
也不能只凭调用次数断言分派受限。任何改变归约或计算精度的方案仍先量误差、另议门槛。

本轮没有新增Rust模型算术改动；新Python adapter的py_compile、真实正/负向与进程
所有权检查通过。当前Index CUDA/cuDNN原生组合编译4m05s通过，仅原有LNK4098；
之前参考缓存实现的测试/Clippy/rustdoc继续以原冻结证据为准。新Native报告在
buffered-cache-report/index.html；外部reference结构不伪装成perf report支持的Native数组。
本后端尚未宣称整体达标，剩余精度对齐、长文本/交替参考与人工试听等待推进。

### 同环境全F32的最新有限诊断

为避免把旧栈异步错误当作永远无法运行，在上述可运行SF0.8/pread环境中另做一次
同长参考、use_bf16=False、CUDA_LAUNCH_BLOCKING=1的1+1诊断，启用解码前EOS检查。
`official-pread-f32-diagnostic-run`：模型确实载入12.599944s，所有记录阶段均F32，
加载后GPU allocated6952131584、reserved7082082304 bytes；随后参考CAMPPlus的
BatchNorm报CUDA unknown error，尚未生成GPT codes，无有效round，cleanup通过。
Job commit高水位13804675072 bytes。这次没有重现旧W2v加载AV或旧的异步OOM位置，
但仍未得到全F32对照；不能仅凭栈顶断言BatchNorm固有bug、显存容量不足或已经修复。
不连续重启同一模式，也不将BF16结果写入F32表。

2026-10-25后续：同参考、权重及官方处理参数的CAMPPlus F32独立CUDA控制在两个
新进程中成功，937张量与checkpoint逐位一致，特征/embedding跨进程一致；它不是
全模型F32恢复，也未证明旧unknown error由内存引起。新xtask非音频诊断保留完整
进程/资产证据，不伪造TTS轮次，见[CAMPPlus隔离记录](indextts-windows-campplus-isolation-2026-10-25.md)。

当前Native三进程Job commit峰值分别7221911552、7223861248、7325671424 bytes。
与官方数据同为含后代的Job高水位口径，但精度、框架/加载策略不同；仅作各模式资源
实测，不把dtype混合的差额当作算法内存优化百分比。
恢复入口使用外部windows-voxcpm-20261024/commit-preflight-tool.json中记录的冻结xtask。
`perf reference --job <official-pread-bf16-baseline-job.json> --output <全新目录>`重跑官方；
`perf run --job <buffered-cache-job.json> --output <全新目录>`重跑Native。
`native-stage-profile-job.json`是有环境断言的独立诊断；
`official-synchronized-stages-job.json`是官方同步诊断，均不混入上述正式基线。
保留python-pread环境及其只读SF0.8目标、冻结adapter、旧失败目录和所有模型/音色原件。
### Snake 不可变常量复用初筛

在同一 RTX5070/Candle fork2266099d/CUDA13/cuDNN9.14 上新增显式 ignored
`bigvgan::tests::cuda_snake_constant_reuse_diagnostic`，不下载模型、不修改生产算术。
测试原 Activation1d 与预先计算 exp(alpha)、recip(exp(beta)+1e-9) 的同顺序路径，
F32 合成参数和输入三形状 (1,512,172)、(1,128,2756)、(1,32,22048) 输出逐位一致。
一进程、1预热+5热轮、每臂40次，轮间交替次序，计时边界同步且不含CPU读回。
局部均值下降36.69%、26.61%、12.36%；最大形状有单轮反向，尚非稳定整链结论。
原始日志、冻结测试exe/源码SHA及统计保存在外部 Windows Index 证据目录的
snake-constant-diagnostic.log / snake-constant-diagnostic-summary.json。
这些不是采集的真实中间激活，不证明真实权重覆盖、完整PCM/EOS或取消复用通过。
下一步以真实权重和实际阶段输入验证，再决定是否采用加载期常量；不据本初筛修改
所有设备的生产路径。release CUDA/cuDNN测试构建通过（原有LNK4098），相关普通
测试105通过、10忽略、12筛除，真CUDA初筛单独通过。

本轮 CUDA/cuDNN release Clippy（lib/tests，-D warnings）、rustdoc（-D warnings）、cargo fmt 和 git diff --check 均通过。

### 真实 BigVGAN 常量复用：暂不采用

后续加载同一已验证转换权重（bigvgan.safetensors SHA054e00342ad121da0b93f1fd2ecb0fcf99f7da6b5a0c03969d6b602e2c08e517），
在测试构建中临时缓存各激活的exp与reciprocal；生产构建没有候选路径。
三独立进程，各按8→128→224→8帧切换，1+5轮，每轮交替原/候选，边界同步；
mel为合成平滑输入且narrow后非连续，不冒充真实CFM中间值或完整生产请求。
全部144次完整vocoder输出逐位匹配相应原路径基准，样本数2048/32768/57344。
128帧每进程均值收益4.58%、4.07%、0.81%；224帧2.14%、1.89%、-1.47%；
短输入抖动且部分反向。尚无足够稳定的完整阶段收益，不据先前合成Snake的12–37%
推断生产收益。本次不采用，不因固定整链10%门槛拒绝，而因真实阶段证据不足。
外部snake-real-vocoder-build.json、source.rs、summary.json和三个log保留可恢复
二进制/SHA/逐轮证据；临时cfg(test)字段及切换分支已撤回，源SHA恢复上一轮已通过
测试/Clippy/rustdoc的版本，fmt/diff检查通过。下一步优先实际FIR执行或CFM热点。

### DiT 真实权重分阶段诊断

新增 INDEXTTS_PROFILE 门控 dit.input / rope_tables / transformer / wavenet / final
同步范围，关闭时沿用已有 profiler 的空路径；没有更改模型算术。dit.wavenet 包含
long-skip、conv1、第二时间嵌入、WN 和残差投影，不等于纯 WN 网络时间；Transformer
包含块循环与末尾norm。各范围互不嵌套，但边界同步会改变派发，不能替代生产耗时。
显式 ignored real_cuda_stage_diagnostic 读取当前 config.json 和已校验 s2mel 权重
SHA91a502f117d7e38f00652cb32b901e71cc43a60b00e4c9a651329bfce15117e2，真实CUDA/cuDNN，
CFG样式B=2，长度64/256，各1预热+5热轮。输入是合成常量，各轮时间值按round/25变化，
输出有限且形状正确；没有宣称内容/PCM对照、完整25步solver或Registry请求验收。

| 长度 | input ms | RoPE ms | Transformer ms | WaveNet范围 ms | final ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.40232 | 0.06856 | 9.37512 | 6.59412 | 0.63584 |
| 256 | 0.32476 | 0.05346 | 9.88846 | 6.75484 | 0.69016 |

仅单进程初筛，64早期热轮有明显抖动；不把这些数乘25当作实测完整CFM。
尽管RoPE每步重建明显，当前只占很小成本，降低其优先级；下一候选优先Transformer
布局/派发及WaveNet的重复padding-index上传，必要时考虑请求拥有的图回放。
不从阶段时间单独断言dispatch-bound，也不将跨Nano结论迁移为本模型事实。
外部证据 dit-stage-build.json/source.rs/diagnostic.log/summary.json 提供冻结exe与源码、
权重身份和原始逐轮；可运行冻结exe real_cuda_stage_diagnostic --ignored --nocapture，
显式设置INDEXTTS_PROFILE=1及INDEXTTS_CONVERTED_DIR到上述本地转换目录。
本轮普通测试105通过、10忽略、13筛除；CUDA真实诊断单独通过。未新增性能候选路径。

### WaveNet 调用内 padding index 复用初筛撤回

同RTX5070与固定s2mel权重，临时forward_with_padding诊断按原层顺序执行完整WN。
原臂逐层reflect_pad1d，候选每次forward构造/上传一个相同反射index并复用八层，
其构造/上传计入候选耗时。没有跨请求缓存、算法/精度变更。合成输入B=2、512通道，
长度64→256→512→64，各1+5、轮间交替两臂、边界同步，48次完整输出均与各形状原臂
基准逐位一致。仅单进程诊断，不是完整CFM、PCM/EOS或生产请求验证。
64复测近似持平，512候选均值反向，收益不稳定；不扩大正式ABBA，不采用。
冻结wavenet-padding-build.json、candidate.rs、baseline.rs、diagnostic.log、summary.json
保留二进制/SHA/逐轮证据。临时helper与测试已整体撤回，wavenet.rs逐字节恢复原版；
不留下低收益运行时切换或为该候选建立共享框架。当前继续保留上一轮DiT诊断spans，
而不把明显重复上传自动提升成优化成果。下一轮推进ONNX实际RTX5070 provider覆盖。

### FIR细分后续

[三进程组件诊断](indextts-windows-fir-stages-2026-10-25.md)确认padding与224帧相位拼接/缩放值得局部布局A/B；72份profile输出与关闭控制一致，尚无生产候选或新的整链收益。同步嵌套范围、合成mel和首版跨通道乱序局限已分别记录。

[FIR布局候选撤回](indextts-windows-fir-layout-withdrawal-2026-10-25.md)：边界广播cat局部快，但三进程真实完整vocoder长形状反向；相位交错候选无稳定局部收益，均未采用。144输出逐位一致，候选源码与测试开关已整体撤回，保留冻结收据和raw证据。
