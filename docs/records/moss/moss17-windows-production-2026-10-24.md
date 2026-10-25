# MOSS Local / Realtime Windows 生产基线（进行中）

范围为当前生产 ID local-1.7b、realtime-1.7b，分别建立基线，不沿用 Nano 数字。
设备 Ryzen 7 9700X / RTX 5070 12 GB / Windows / 16 GB RAM，driver616.56；
main42a7887 加本轮 Windows 工作树，Candle0.11 fork2266099d，CUDA13.0。
本轮没有修改 MOSS 运行时算术、公开协议或模型资源。

证据根 `D:/workspace/talechime-benchmarks/windows-moss17-20261024`；模型仍位于
`~/.talechime/resources/moss/models/`。生产清单全部匹配：Local 7 文件、Realtime
4 文件、共享 codec 4 文件（local-resource-check.json）。通过当前 catalog 的
xtask prepare 再分别校验，各有独立资源收据，不将手工摘要代替正式预检。

| 资源 | 固定 revision |
| --- | --- |
| OpenMOSS-Team/MOSS-TTS-Local-Transformer | 12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e |
| OpenMOSS-Team/MOSS-TTS-Realtime | 75682787d8e2fcc73faca37ba2931453ca9c4022 |
| OpenMOSS-Team/MOSS-Audio-Tokenizer | 3cd226ba2947efa357ef453bcad111b6eafba782 |

构建使用 moss-candle-cuda、performance-tools；实际 features 为 moss、moss-candle、
moss-candle-cuda、performance-tools。新 target/perf-builds/80b54b83… 联接到 D 盘
专用 build-cache，不删除旧缓存。两个 release 收据均冻结 exe，构建后再推理：
catalog-build.json 的 SHA256
`3edbc8882ab5e8618a01c9858aa0988b895b7ee90d46de13573150cc113db34b`；
production-build.json 的 SHA256
`9a333dd33024bdd6c463e09fc18c183aace8b6aff4fcd4b2d3ea08c4557cd06e`。
仅既有 LNK4098；catalog/probe 都通过真实 CLI 执行。未以普通测试下载模型。

生产适配器 GPU-only；CUDA 主模型 BF16、codec F16，每 5 帧交付，24 kHz 单声道。
Local 使用 32 codebooks，Realtime16；不减少 codebooks 或生成预算换提速。
冻结文本“你好，欢迎收听今天的故事。”、Pinned42、内置 narrator、无 continuation、
无 style、各模型参数默认值（30s 段预算，Local instruction/language 为空）。
这是无参考的随机音色场景，不能代表克隆/参考准备或执行接续。路径原字符串沿用
正式资源收据，避免重现 IndexTTS 的斜杠身份错误。

## 短测先确认实际执行

每模型独立 1 进程×(1预热+1热轮)，全部通过父进程资源/参数/工件审计并清理。

| 模型 | 预热 / 热轮合成 s | 预热 / 热轮首 PCM s | 音频 s |
| --- | --- | --- | --- |
| Local | 3.097622 / 3.276434 | 0.702911 / 0.852562 | 2.74 |
| Realtime | 1.507418 / 1.908577 | 0.226799 / 0.413775 | 2.4 |

各模型两轮 PCM 摘要分别一致：Local `0f222678a68893c30c6ea968d3e69a113fd3f7ba6865c0e174ab5bb42687cda5`；
Realtime `1f9328b4300ac1cfe5f7f8b695fedcf00f89a7ad07a698c5e69e459a242dba58`。
短测不是正式多进程统计，也不证明试听/文字完整性。没有并行编译或其他推理。
正式三进程各 1+5 分别串行测量，原始路径为各模型 baseline-run，完成后单列结果。

## 官方来源已准备，Python 运行仍待建立

官方 checkout 为证据根下 official/，HEAD
`934d6826b084c46a0d033402174d5f8ac4ed2519`，工作树干净。源码准备结束后才开始
正式测量，没有在正式热轮中 clone。首次 Git schannel 握手失败；命令级指定 OpenSSL
后 fetch/checkout 成功，未关闭证书校验、未改系统 TLS 或全局 Git 设置。

注意版本识别：本次固定的是旧 MOSS-TTS-Local-Transformer，而不是 README 的新版
Local-Transformer-v1.5（4B）。旧 config.model_type/architectures 仍叫 moss_tts_delay /
MossTTSDelayModel，固定源码 moss_tts_local/modeling_moss_tts.py 内也用该旧类名，但
确实创建 Local Transformer。仅看 model_type 就切去 moss_tts_delay 是错误的；需要
结合 repo/revision、完整配置、权重键和实际文件。此处仅源码核对，没有宣称官方导入
或模型推理通过。

官方 pyproject 的 torch-runtime 推荐 torch/torchaudio2.9.1+cu128、transformers5.0.0，
现有其他后端的2.8环境不自动视为匹配。尚无官方同条件性能排名、人工试听或回读验收。

## 三进程生产基线完成

每个模型分别 3 独立进程，各 1 预热+5 热轮。6 进程全部 outcome=success、
cleanup_verified=true，36 轮含预热全部有效；每模型 18 轮 PCM 摘要与上面各自短测
一致，模型间不比较 PCM。未缩短默认生成预算，没有将长度上限当成自然完成。

| 模型 | 进程 | 热轮合成均值 s | 首 PCM 均值 s | 音频 s |
| --- | --- | --- | --- | --- |
| Local | 0 | 3.16128596 | 0.79263010 | 2.74 |
| Local | 1 | 3.20548232 | 0.80838312 | 2.74 |
| Local | 2 | 3.15231572 | 0.75832794 | 2.74 |
| Realtime | 0 | 1.66234372 | 0.29348338 | 2.4 |
| Realtime | 1 | 1.65359812 | 0.26113076 | 2.4 |
| Realtime | 2 | 1.54253494 | 0.27680118 | 2.4 |

Rust xtask perf report 对各原始 run 重新审计并生成 results.csv 和离线 index.html：

| 模型 | mean-of-means s | 首 PCM mean-of-means s | RTF |
| --- | --- | --- | --- |
| Local | 3.17302800 | 0.78644705 | 1.15803942 |
| Realtime | 1.61949226 | 0.27713844 | 0.67478844 |

结果在 local-1.7b-baseline-run、realtime-1.7b-baseline-run；对应 *-baseline-report
含已校验音频工件和 Rust 统计，baseline-summary.json 另列进程数据。报告生成在全部
推理结束后执行；未并行推理/编译，也未将短测或加载计时混入热轮。Local 完整吞吐
在此短句负载上慢于实时，Realtime 快于实时；这不是两种模型在不同输出长度下的
实现等价排名，不推出所有文本/音色都如此。

资源校验单列 Local 6.706–6.740 s、Realtime6.015–6.074 s；加载单列 Local
10.669–10.811 s、Realtime8.858–9.285 s（baseline-preparation-summary.json）。
无参考音色，因此没有用短合成预热参考编码，也没有声称覆盖克隆/接续成本。

baseline-telemetry-summary.json 汇总 500 ms 采样；覆盖启动、校验、加载、预热、
热轮和退出，不是热轮独占或 allocator 精确峰值。Local202 样本，整卡显存最大
10737418240 bytes，进程 RSS 最大7398240256 bytes，private bytes最大11193761792；
Realtime140 样本，整卡显存最大9292480512，RSS最大7436185600，private最大9747087360。
整卡包含桌面驻留；不从这些跨阶段采样推断某个算子的独占显存。
Local SM180–2932MHz、10.60–120.28W、44–59°C；Realtime270–2947MHz、
23.26–101.90W、39–48°C。没有锁频或更改电源计划。

本轮只新增基线证据，没有运行时提速候选；fmt/diff 检查通过，相关原生 release
catalog/probe 构建通过，普通测试没有下载/加载模型。未因文档/测量重复运行无关
整仓测试。数值稳定不等于听感、内容完整或播放完成验收；试听/回读仍 pending。
下一步分别匹配官方 BF16/F16 的实际入口和采样参数，再做阶段归因；不能直接把
OmniVoice 的 F32 decoder kernel 用到本例 F16 codec，亦不能以降低 RVQ 层数提速。

## 官方 Local Windows 加载调查（未进入热轮）

证据根仍为 `D:/workspace/talechime-benchmarks/windows-moss17-20261024`。
`python/` 隔离环境使用 Transformers5.0.0，借用 IndexTTS 已安装的
Torch/Torchaudio2.8.0+cu128；这与官方推荐2.9.1不一致，不作为正式同条件排名。
主模型 BF16、codec F16、32 RVQ 和官方采样参数保持不变。官方 codec 源码固定为
`8c50ac4c5d7287d2ed6ea20a08c90ca439887d23`。

- `local-official-compat-smoke-run`：异步加载时进程访问违例，无 prepared/有效热轮。
- `local-official-sync-load-smoke-run`：`HF_DEACTIVATE_ASYNC_LOAD=1` 确认生效，
  主线程权重读取仍访问违例。不能把禁用线程视为问题已解决。
- `local-official-numpy-io-smoke-run`：外部 NumPy IO 原型在加载阶段得到 Windows
  1455（页面文件太小）；原型夹具另有 Ellipsis 切片兼容失败，已停止扩展，未进入生产。
- `local-official-pread-smoke-run`：另建 `python-pread/`，仅将 Safetensors 升到0.8.0，
  使用其原生 `backend="pread"`，通过 scoped patch 限于 Transformers checkpoint
  读取；原环境与此前冻结脚本不覆盖。`pread-io-test.log` 记录12个 CPU tensor 的
  mmap/pread 全量及切片逐位一致，包括 BF16/F32 特殊值。主模型556项加载完成，
  随后 codec 的 safe_open 仍报1455。未进入 prepared/推理，没有可用合成计时。

以上失败 capture 均记录 cleanup_verified=true；失败加载不能当成性能结果。
上游已有 Windows mmap 提交额度问题的报告，Safetensors0.8.0也提供 pread，但本机
结果只证明该配置加载 codec 时提交资源不足，不证明此前每次访问违例都是同一原因。
源码及发行说明：
[Transformers Windows mmap 调查](https://github.com/huggingface/transformers/issues/48285)、
[Safetensors0.8.0](https://github.com/safetensors/safetensors/releases/tag/v0.8.0)。
未修改系统页面文件、电源设置、官方模型数学实现或减少 codebook。
下一步应先取得系统 commit used/limit 及阶段资源证据，再选择有资源预算的官方环境；
当前不继续对同一资源条件反复重试，也不宣称达到或超过官方。
进程退出后 CIM 快照：CommittedBytes=14380666880、CommitLimit=26896412672、
AvailableMBytes=8566；这是退出后的系统值，不是加载失败瞬间峰值，不能据此声称
失败时仍有相同余量。


## 后续：共用 CUDA mask 省略的完整请求回归

同日同一Windows11 / 9700X / RTX5070、CUDA13、Candle0.11 fork2266099d。
Nano已验证的严格全可见单query mask省略位于共用attend，因此对1.7B单独重建与回归。
未改变CPU/Metal路径、dtype、采样、codec层数、原文本/seed或无参考narrator条件。
候选机制、540组dtype/位置oracle和Nano codes/raw PCM/EOS证据见
[Nano验证记录](moss-windows-performance-2026-10-24.md)的CUDA单query节。

标准perf build使用moss-candle-cuda，4m20s完成、Source guard通过；unmasked-build.json
保留真实dirty源码身份和独立二进制，SHA256为
6940c934d2751c1b2e84f8cf1cdf509f649507cdc51fd3c1ce465f1521d1df48。
已观察LNK4098链接警告，完整构建日志保留。随后两个模式串行运行，各3独立进程、
1预热+5热轮，关闭阶段观测，期间没有编译/其他推理：

| 模式 | 逐进程热均值 s | 组均值 s | 首PCM组均值 s | 音频 s |
| --- | --- | ---: | ---: | ---: |
| Local1.7B | 2.79860856 / 2.82626390 / 2.77343894 | 2.79943713 | 0.69592135 | 2.74 |
| Realtime1.7B | 1.39204034 / 1.41440022 / 1.42897476 | 1.41180511 | 0.23248823 | 2.40 |

36轮success、cleanup通过，各模式PCM均与本记录早期基线完全相同：Local
0f222678a68893c30c6ea968d3e69a113fd3f7ba6865c0e174ab5bb42687cda5；Realtime
1f9328b4300ac1cfe5f7f8b695fedcf00f89a7ad07a698c5e69e459a242dba58。
这是当前绝对基线和完整请求输出回归；没有两模式的同配置ABBA对照，且早期观测/
探针导出策略不同，不把相对历史3.173/1.619s的差额全部归因mask省略。
未单独捕获1.7B codes/EOS，不把PCM摘要相同改称记录到这些模型内部证据；也没有
把Nano取消/满队列门禁冒充1.7B的完整状态验证。克隆、接续、取消和内容/试听仍待补。

资源校验Local7.049–7.346s、Realtime6.307–6.360s；加载Local10.694–10.848s、
Realtime9.195–9.326s，逐进程见unmasked-preparation-summary.json。reference_prepare
仍为null，无参考音色；reference_warmup为独立短合成，不是已隔离参考编码成本。
本轮observation=off，不扩称采集了新的RSS/VRAM峰值；早期跨阶段遥测保持原口径。

moss-candle-cuda,performance-tools release all-targets Clippy -D warnings及rustdoc
-D warnings通过，日志unmasked-clippy.log/unmasked-rustdoc.log。共用模型源码的
普通测试和CUDA dtype oracle已在上述Nano记录验证，本轮未为无新代码重复运行。
外部根保存prepare_unmasked_jobs.py、两个*-unmasked-job/plan.json、Native数组结果、
unmasked-summary.json（全部逐轮）、两个*-unmasked-report/index.html及证据SHA目录。

复现时按Job保留资源、源码/二进制及环境身份，用冻结xtask顺序执行：

```powershell
# 先将CUDA13和匹配cuDNN DLL目录置于PATH。
xtask.exe perf run --job D:/workspace/talechime-benchmarks/windows-moss17-20261024/local-1.7b-unmasked-job.json --output NEW_LOCAL_OUTPUT
xtask.exe perf run --job D:/workspace/talechime-benchmarks/windows-moss17-20261024/realtime-1.7b-unmasked-job.json --output NEW_REALTIME_OUTPUT
```

另复核官方pread失败：适配器with patch实际同时包住主模型和codec加载，不能只凭
报错中的safe_open调用行就推定codec漏出作用域或仍用mmap。本轮未重试该官方加载，
没有新官方合成结果。下一步需记录实际safe_open参数/backend及失败瞬间提交内存，
不能用“扩展已有作用域”重复已做的实验，也不能据本次Rust变快宣布超过官方。


## 官方加载后续：pread元数据仍有整分片瞬时提交

独立IO诊断、隔离Safetensors0.8.0+windows.headerio1轮子及两个完整加载控制见
[Windows元数据提交记录](../platform/windows-safetensors-header-commit-2026-10-24.md)。
原pread返回后的常驻提交较低，却有接近5GB分片大小的瞬时峰值；补丁把该峰值降至
约10MB，12类合成及真实小切片字节对照通过。主模型和codec本就在同一patch作用域。
新完整请求推进到codec分配器OOM；跳过codec预分配后推进约81%再MemoryError，
都无有效官方热轮且cleanup通过。仍未恢复官方基线，不能宣布达标或将失败计时排名。
Local分片556张量实际全BF16、约30.61亿参数，不能以1.7B标签/文件大小推断F32回退。

2026-10-25补充：旧官方适配器的 codec F16 并非官方 processor 默认精度。现已恢复
F32，并完成两个精度的独立 codec 加载/固定 codes 解码控制。官方默认主模型与 codec
参数合计13.219GB，超过本机12.792GB独立显存，加载前容量门禁已实测拒绝。仍无完整
官方TTS基线；详见[官方 codec 合约](moss17-windows-official-codec-contract-2026-10-25.md)。

## 2026-10-25：Local / Realtime 1.7B 所有者状态门禁

新增 `moss/candle/cuda_tests.rs` 两项显式 ignored 的 Windows CUDA 真模型测试，
普通测试不加载或下载模型。没有修改模型、codec 算术或生产运行时。测试程序由
`cargo test -p talechime-backends --release --no-default-features --features moss-candle-cuda --lib --no-run --message-format=json`
实际 artifact 定位并复制冻结，身份见外部根 `owner-state-build.json`，SHA为
`986a3b8ee8f3e28a6e2105916ad62e97e113d8c6717ec6833a657748043f7564`。

两项在同一模型所有者上分别验证：正常完整请求；未消费接收流即取消；收到首PCM后
观察 capacity-one 队列实际已满再取消；用 max_duration=1.0 触发真实模型推理
`frame limit reached before EOS`，且错误请求不发送 End。每个异常后再次执行原请求，
完整 PCM F32 位模式均与首次相同，所有正常请求均有非空有限 PCM 和 End。
测试文本“你好，欢迎使用叙铃。”，seed 为 SeedPolicy::Pinned(42).resolve(0,0)，
narrator、无参考/接续、默认生成参数。Local 为65,280个24kHz单声道样本，Realtime
为48,000个；串行运行各20.14s、15.52s，这是含加载与多请求的测试耗时，不是合成
性能数据。不能与前文生产基线按样本长度或速度直接排名。

日志 `owner-state-local.log` / `owner-state-realtime.log` 及退出收据均为成功。
普通测试29通过、2忽略；同feature release all-targets Clippy -D warnings、
rustdoc -D warnings通过，日志分别为owner-state-clippy.log / owner-state-rustdoc.log。此门禁证明这两个具体无参考模式的完整输出及错误/取消后
复用；没有捕获内部逐帧 codes，不将 End 观察扩大成独立 codes/EOS 轨迹采集，
也未覆盖克隆、风格、接续或人工试听。既有 Nano/ONNX 测试不再作为1.7B门禁替代。

复现先构建，再设置 `TALECHIME_MOSS17_CUDA_ROOT` 为资源根并配置CUDA/cuDNN DLL，
串行执行冻结测试程序：

```powershell
& $testExe moss::candle::cuda_tests::real_cuda_local_owner_cancel_failure_and_reuse --exact --ignored --nocapture --test-threads=1
& $testExe moss::candle::cuda_tests::real_cuda_realtime_owner_cancel_failure_and_reuse --exact --ignored --nocapture --test-threads=1
```

## 2026-10-25：参考缓存与接续状态交错验证

同一 Windows CUDA 生产适配器再增加两项显式 ignored 真模型测试，未改变运行时。
冻结测试程序 SHA `b122fd5403b379cea23285e12c04772ef8c03870108cde7d432a8f92e8645e74`，
外部根收据为 `reference-state-build.json`。每种模式先自行合成上述完整 narrator 语音，
写临时24kHz单声道F32 WAV，再用唯一 `custom:cuda-gate-*` ID通过 VoiceStore 导入。
不使用用户私人音频、不覆盖已有音色；测试结束明确移除自身音色并确认目录消失，
异常退出另有 guard 尝试清理。测试后目录扫描也无残留测试音色。

重新加载适配器以读取新增能力，依次检查：首次参考编码并生成克隆、命中同一参考
缓存再次生成、无参考生成、接续重复生成、接续首PCM后取消再生成、切回克隆、
再切回无参考。各模式内同类完整 PCM F32 位模式完全一致，所有成功请求都有 End；
参考缓存内容在冷编码后的重复请求中逐字节不变。

| 模式 | 克隆完整PCM样本 | 接续完整PCM样本 | 初始/恢复narrator样本 | 参考缓存字节 |
| --- | ---: | ---: | ---: | ---: |
| Local1.7B | 55,680 | 36,480 | 65,280 | 4,496 |
| Realtime1.7B | 44,160 | 38,400 | 48,000 | 1,788 |

接续上下文为完整初始语音与其原文，新文本“今天的故事开始了。”，种子与前述状态
门禁相同。Local 39.91s、Realtime29.45s为含两次加载和多请求的测试总耗时，
不作冷热合成性能结论。日志 `reference-state-local.log` / `reference-state-realtime.log`
及退出收据均成功；普通29通过、4忽略；同feature release all-targets Clippy与
rustdoc -D warnings通过，格式和diff检查通过。此处补充前节未覆盖的克隆/接续状态，
不将数值恢复扩称为音色相似度、内容或人工听感验收。风格和独立codes轨迹仍待补。

复现构建及环境同前节，用新冻结测试程序分别运行：

```powershell
& $testExe moss::candle::cuda_tests::real_cuda_local_reference_and_continuation_reuse --exact --ignored --nocapture --test-threads=1
& $testExe moss::candle::cuda_tests::real_cuda_realtime_reference_and_continuation_reuse --exact --ignored --nocapture --test-threads=1
```

另对 Realtime checkpoint 进行独立头计数：BF16参数4,663,881,728字节，加官方默认
F32 codec 合计11,762,147,328字节，低于独立显存12,792,037,376字节。参数下界以内
不代表含context/激活一定能运行；官方 Realtime 完整加载与合成仍未验证。不能把
Local13.219GB超限结论外推给Realtime，后续应分别评估。

后续已完成Realtime官方BF16主模型/F32 codec两次分阶段CUDA驻留诊断，得到相同
51帧codes/4.08秒PCM及完整辅助ASR回读；含加载/换载约12.97/13.61秒，不是常驻
热基线，不能与原生热轮排名。实际403主模型参数张量均CUDA BF16，两个原始请求
和容量证据见[Realtime官方分阶段诊断](moss-realtime-windows-official-staged-2026-10-25.md)。

Local随后也完成默认精度分阶段GPU诊断：补齐固定权重revision的官方chat template后，
两次原始40×33序列及3.12秒PCM完全一致，实际末控制为EOS、辅助ASR完整；在codec
加载前后对照了完整提示词token/mask。官方processor还会解码EOS行的非pad音频，
与当前Rust在控制EOS后返回的行为不同。常驻显存超限和无官方热基线结论不变，
详见[Local分阶段及EOS审计](moss-local-windows-official-staged-2026-10-25.md)。

## 2026-10-25：Windows CUDA forward 内 RoPE 表复用保留

仅同一次forward共用相同形状的表，不扩大预计算或跨请求缓存。新同期ABBAABBA
各8进程1+5：Realtime1.599175→1.205953秒，−24.59%，进程均值范围不重叠；
Local2.817157→2.699051秒，−4.19%但范围重叠，整链inconclusive。两个首PCM也
范围重叠，不宣称稳定改善。96轮PCM、UTF-8范围/chunk样本边界保持，480资产复核；
最终两种Native短句辅助ASR完整。Local/Realtime/VoiceDesign共9组真实codes/raw PCM
门禁及4项owner状态门禁通过。此前绝对基线保留为历史批次，不按差额归因本优化。

同期构建曾抓到Copy-Item保留旧mtime导致Cargo错复用二进制，已拒绝错误参考并
重编译；旧比较器把off误判为诊断，修复与回归后保留全部原始条件。完整原始数据、
默认范围、容量/准备成本与复现见[RoPE复用记录](moss-windows-forward-rope-reuse-2026-10-25.md)。
