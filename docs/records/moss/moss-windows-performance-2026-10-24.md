# MOSS Nano Candle Windows 性能调查（进行中）

本轮服务于 issue #5；仅 Windows，Mac 由另一台设备推进。已完成短句/多句生产与
官方GPU基线、CUDA状态回归，保留经过局部归因和逐位验证的生成嵌入复用。
稳定整链收益、人工试听、超长与接续仍有未验收范围，不宣称全后端等价质量达标。

## 早期基线源码与环境

- 源码 `f9e1d4f322da9f2c0b17a0bea09e2fea6ec9ce5b`，探针构建时工作区干净。
- Windows 系统身份由 xtask doctor 保存；Ryzen 7 9700X、16 逻辑线程、RTX 5070
  12 GB，驱动 616.56。桌面 GPU 客户端仍驻留，未锁频、未调整电源。
- Rust 独立 release `performance_probe` 与 `performance_catalog`，features
  `moss-nano-candle-cuda,performance-tools`（有效展开见构建收据），F32、Rayon 8。
- TTS revision `44502f80dbf9743528fa921cc544d662c685ebec`；codec revision
  `6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`，均通过生产清单校验。
- release 链接仍报告既有 LNK4098 LIBCMT 冲突警告，未通过修改 CRT 或忽略库掩盖。

## 早期短句生产路径绝对基线

文本“你好，欢迎使用叙铃。”，内置 Weiguo，seed 42，continuation=false，默认生产
采样，原生流式交付；每进程 1 预热 + 5 热轮，共 3 独立进程。计时从生产消费请求
开始至显式 Completed；加载、写音频不计入热合成，不能理解为 kernel 时间或播放完成。

| 进程 | 热合成均值 s | 音频 s | RTF | 首 PCM 均值 s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.768231 | 3.36 | 0.228640 | 0.069425 |
| 1 | 0.742044 | 3.36 | 0.220846 | 0.069748 |
| 2 | 0.704879 | 3.36 | 0.209785 | 0.068147 |

三个进程均 success、cleanup_verified=true，各 5 个热轮；各进程内部热轮 PCM 摘要
一致。均值约 0.738384 s，RTF 0.219757；存在进程间漂移，不能用此单臂批次推断优化
收益。尚未人工试听、长文本、接续或官方对照，不从自然完成推断逐字朗读完整。

原始 Job、Plan、构建/准备收据、全部音频和结果位于
`~/.talechime/benchmarks/windows-nano-20261024-*`；报告为同前缀 `report/index.html`。
恢复和报告使用现有 Rust xtask `perf resume/report`，原始 run 目录不覆盖。

## 官方对照准备

官方 CLI 仓库 OpenMOSS/MOSS-TTS-Nano 固定到
`8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb`。实际模型 Python 从上述 TTS/codec 各自
revision 获取，不能用 CLI revision 代替实际导入代码身份。官方目录与生产资源隔离，
权重/配置只读硬链接复用，禁止原地修改链接文件。

独立 Python 环境位于 `~/.talechime/benchmarks/windows/moss-nano-python`，仅通过
显式 `.pth` 只读复用既有 index-tts 环境的 CUDA Torch，安装独立 transformers 4.57.1；
已验证 torch 2.8.0+cu128、CUDA available。Torch 与官方 requirements 的 2.7.0 不同，
正式对照需披露，并确认模型导入、实际 attention 和采样路径可运行。尚未运行官方合成。

## 下一步与 skill 回流

1. 对齐官方输入 codes/参考、采样、F32、分段与流式边界，完成离线导入预检和短句验证。
2. 用 xtask 调度官方基准进程并记录有效参数、来源、同步、EOS、输出与逐轮耗时；不
   复活已退役的 Python 编排器。补长段和接续覆盖。
3. 测量逐层 RoPE/mask 构建、逐码本主机往返及 codec 成本；源码观察目前仅是假设。
4. 每个候选独立输出对照与平衡 A/B，通过门槛才保留。

本轮已修订 skill 的官方模型代码身份、catalog/probe 收据区分、历史 Job 字段检查与
target 联接清理边界，并修正“必须提取独立图辅助 crate”的过时指导。结构校验通过；
尚需在后续模型实际验证这些经验，不将文档更新当作全部 skill 目标完成。

## 追加：官方可运行性与第一个候选

官方固定模型已通过离线 F32 CUDA 合成 smoke：32 帧、2.56 秒、双声道 48 kHz，
PCM 有限且非空。单次冷请求生成 1.974 s、含整段解码 2.154 s；这是环境验证，
不是预热后的性能基线，更不能与 Rust 3.36 秒的流式音频负载相减宣称加速。
采样显式对齐 Rust（text temperature=1/top-k=2/top-p=1，audio .8/25/.95，
repetition=1.2）。跨框架随机轨迹不同，未人工试听。原始输入、codes、PCM 和结果位于
`~/.talechime/benchmarks/windows/nano-official-smoke-sdpa-propagated/`。

准备时两次失败均保留目录：codec 不接受通用 SDPA 配置，使用其 eager 入口；TTS
即使 from_pretrained 传入 SDPA，子解码器仍继承 FlashAttention 设置，需要沿用官方
infer.py 的 `_set_attention_implementation("sdpa")` 显式传播。不改官方模型数学。
当前 smoke 直接运行 Python，后续正式批次仍须接入 xtask 的进程生命周期与采集。

首个 Rust 候选省去单查询、无窗口且全部 key 可见时的全零因果 mask。绝对 key offset
和未来 key 的回归测试通过；12 个模型库夹具测试、CPU Clippy/rustdoc 通过。
候选 3×5 热轮均成功，均值 0.703157 s，三个进程均值范围 0.672333–0.722232 s；
全部热轮 PCM 摘要与原始基线相同。与原基线区间重叠，约 4.8% 的顺序批次差异
不构成收益验收，候选仍处于调查阶段，尚未正式保留。

平衡四槽初筛在执行前被 `probe binary changed` 拦住：旧 xtask 收据引用同 feature
共用的 Cargo example，候选重编覆盖了 baseline。没有修改旧收据，也没有将失败
计作已跑 A/B。现修复 xtask 构建后按 SHA 原子保存二进制，增加覆盖重建/复用/篡改
回归；3 个 build 测试、xtask Clippy/rustdoc 通过，真实 CUDA 探针副本发布成功。
此前 run 的原始记录仍有效，旧 target 收据已不能恢复执行；需重新构建保存 baseline
再完成平衡初筛，未达门槛则撤回模型候选。

后续状态修正：已按 SHA 保存 baseline/candidate 并完成四槽平衡初筛，原始结果在
`windows-nano-mask-balanced-{slot}-{arm}-run`。每槽各 1 预热+2 热轮，PCM 摘要一致，
两臂同时出现明显时序漂移，尚无可信整链收益结论。为重建 baseline，当前工作树已移除
mask 快路径，仅保留绝对 offset 的回归测试；候选源码另存忽略的本地证据目录。

用户随后明确先做 OmniVoice，Nano 暂后移；当前 main 已同步到 `42a7887`。用户同时
取消默认整链 10% 门槛，上文门槛描述仅记录当轮决策，不用于后续自动撤回。若恢复
Nano 调查，按局部真实收益、整链影响、输出与维护成本重新判断，不复用旧二进制冒充
合并后的性能结果。

## 当前共同版本重新建立生产基线

源码 HEAD `42a7887e28f095684266eac43a68bd8fcc303d11`，包含未提交 Windows 改动，
具体 tracked/untracked 身份见新构建收据。工作区摘要
`f6cea6871e3d34a06d6f0c19f1eecfd29233d4c8c5b850ea410b962343cc8236`；
冻结 probe SHA256 `336f2ebac0623cb48473779b95e1aaffe5a0a63bc5b33ee8f5d7e04ca79095cb`。
同一版本 catalog 已通过六个生产资源文件校验；原生 CUDA/F32、Rayon 8，测量期间无并行编译或推理。
构建 target 已恢复 D 盘实体目录；不能将旧 C 盘副本的可变 example 作为新证据。

新证据根目录 `D:/workspace/talechime-benchmarks/windows-nano-20261024`：
`production-build.json`、`catalog-build.json`、`catalog.json`、`plan.json`、
`acceptance-job.json`、`acceptance-run/results.json`、`acceptance-summary.json`、
`acceptance-report/index.html`。固定工具见 VoxCPM 外部证据目录的 `commit-preflight-tool.json`。
恢复命令为该工具 `perf resume --job <acceptance-job.json> --output <acceptance-run>`；
报告命令为 `perf report --run <acceptance-run> --output <新的报告目录>`，不要覆盖既有证据。

文本“你好，欢迎使用叙铃。”，Weiguo、seed 42、continuation=false、默认采样，
3 独立进程各 1 预热+5 热轮，完整生产 Registry/Backend 流式路径：

| 进程 | 热合成均值 s | 首 PCM 均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 0.68719102 | 0.07009396 | 3.36 |
| 1 | 0.67880422 | 0.06919116 | 3.36 |
| 2 | 0.68398146 | 0.06591050 | 3.36 |

组均值 0.68332557 s，RTF 0.20337071，首 PCM 0.06839854 s。
18 轮均 success，三个进程均 cleanup_verified；18 轮 PCM SHA256 一致：
`d2e6692e13ef20a1364e0547dc3703c0f9705b985274285e307ef759469fd6e0`。
这次重建基线不是 ABBA 候选验收，不能把相对早期批次的耗时变化归因于某项优化。
额外“你好。”用例另存 `native-job.json` / `native-run` / `greeting-summary.json`，
输出仅 0.48 s，不能混入上述句子的统计或与旧 3.36 s 输出比较。

官方对照当前仍未完成正式测量。源码复查确认官方 generation output 仅携带 audio codes，
没有最终 EOS 字段；`finished_mask` 只在已产生的 frame 事件发布，最终 EOS 分支会直接 break，
因此不能用最后一个 frame 的 finished_mask 或 trimmed 长度替代真实最终 EOS 检查。
后续适配器须观察实际 text-token 终止选择且不修改推理决策。两边均为 48 kHz 双声道，
音频时长按 samples/(sample_rate*channels)；官方完整 decode 的首 PCM 与生产分批 codec
流式首 PCM 应分别标记。人工试听、内容回读、长文本和完整状态回归仍 pending。
## 独立官方 Python 正式对照与实际 seed（补充取代“尚未正式测量”状态）

官方 CLI checkout `8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb`，真正模型 Python
分别来自上述固定 TTS/codec revision 的 assets；二者都固定摘要，不能把 CLI revision
当作模型 Python revision。`production-official-assets-audit.json` 重新核对六个生产
权重/配置/tokenizer 文件与官方 assets 字节相同。生产编译内嵌的 manifest 与官方
使用的磁盘 manifest SHA256 相同，见 `conditioning-audit.json`。

新增 `scripts/perf/moss_nano_cuda_reference.py` 由 Rust xtask reference 独立调度，
运行时模型源码不修改；只包装原 assistant-token sampler，保留实际返回 tensor，
终止后检查 token=7，逐步不额外同步。保留官方有限值检查与默认 KV 路径。
TTS SDPA 显式传播给子模块；codec 不支持 SDPA，使用 eager。两边 TTS/codec 都 F32。
Torch 2.8.0+cu128、Transformers 4.57.1；官方 requirements 固定 Torch/torchaudio 2.7.0，
因此 Torch 版本不匹配，当前结果明确属于此可运行环境，不声称官方原依赖全复现。
官方自己的虚拟环境通过只读 .pth 复用 IndexTTS Torch；本次 Job 固定自身 .pth、
模型源码/权重、环境 METADATA、Torch init/version、语料和适配器摘要，离线运行，
HF remote-code 缓存在本次 D 盘隔离目录。

生产 `Pinned(42).resolve(call=0,attempt=0)` 得到 7330414270409262323，Nano 的
`generate_nonempty` 再截取为 u32=3560097011。官方首批直接 seed42 的结果保留在
`official-baseline-run`，均值 1.41170449 s、32 帧/2.56 s；属于 seed 不匹配的独立
case，不能替代对齐后的主对照。修正 Job 为 `official-seed-aligned-v2-job.json`，
输出 `official-seed-aligned-run`，3 独立进程各 1 预热+5 热轮：

| 进程 | 完整热合成均值 s | 生成 s | codec+回传 s | 音频 s | 加载 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.80530984 | 1.79323096 | 0.01207888 | 3.28 | 0.55577030 |
| 1 | 1.92811614 | 1.91399048 | 0.01412566 | 3.28 | 0.54306880 |
| 2 | 1.83587262 | 1.82437784 | 0.01149478 | 3.28 | 0.56444780 |

组均值 1.85643287 s，RTF 0.56598563；生成阶段均值 1.84386643 s，占约 99.32%。
这是 generate 包含 global/local、采样和 Python 控制的同步 host 墙钟，不能称作
单一网络或 GPU kernel 时间。官方当前适配器整段生成、整段解码，首 PCM=完整耗时，
不是官方流式能力不存在。生产首 PCM 68.4 ms 是每三帧 codec 解码的原生流式边界。
参考构造 2.47–2.64 ms（不含模型加载）；生产 reference_prepare 未独立暴露，保持 null，
不能拿生产 warmup 冒充参考准备。生产首个进程加载 0.3301801 s，实际更多进程记录在 preparation.json。
官方 Windows Job commit 峰值 4.350–4.362 GB，生产首个进程 1.625 GB；不是 RSS/显存，
可由各 capture.json 重算，不能把这个端点当作全矩阵峰值。

18 轮官方均 success、cleanup_verified，41 code frames，实际末尾选择 EOS7，PCM统一为
`94c75ef2f1499d567b1f10048f2d51ae1d7ba5266b19940fed083bcd711638f0`。
生产为42帧/3.36 s，LCG 与 Torch RNG 不同，即便 seed 数值相同也不能要求轨迹一致。
在本机现有条件下生产完整热请求耗时与 RTF 更低，但环境、输出和质量条件仍有差异，
不把该观察升级为全部模式或完全等价质量下的达标。

`official-eos-negative-job.json` 限制1帧，真实模型未选择EOS，适配器断言
`no confirmed model EOS`，父进程返回失败且 cleanup_verified=true，无成功round、
无 results.json。负向证据保留，不覆盖失败；它验证“短于预算”不能自动认作成功。
常规375帧正式用例没有使用该限制。两版 adapter 的差别仅是显式可选帧上限；正式值仍375。

完整 WAV 采用 frames×channels 交错 F32，时长按单声道帧数/48k；旧 smoke 的
pcm.f32 是2×frames平面布局。将旧文件正确转成交错后的 SHA256 恰为首批seed42的
`ae33ebf61f36e8f5ab06c29cbaad5aa149923a33b4cf85ef31fe73cec97827f8`，
因此旧/新原始PCM摘要差异在此案例是布局差异，不是模型数值漂移。

### 辅助回读与尚未通过的质量门槛

复用已冻结 SenseVoiceSmall-int8 辅助工具和固定资源，没有再编译或下载。
`aux-readback-aligned-results.json`：生产回读“你好欢迎使用叙铃”，官方对齐seed回读
“你好欢迎使用续兵”；首批seed42官方回读“你好欢迎使用续名”。单个识别器对专名
可能出错，这既不是官方漏字的证明，也不能当作两边内容等价通过。
人工试听、音色/韵律判定、长文本与接续仍 pending。该辅助路径未启用产品双模型校验。

## RoPE 表同次 forward 复用候选：实测后撤回

原始 Nano 每层 Q、K 独立重建同一位置 RoPE 表。原型只把表计算移到同一次
Transformer::forward 的层循环外，复用 cos/sin，保持原形状/精度和旋转算术，
不引入跨请求缓存。12层 global 可从24次表构建缩成1次，1层 local 从2次缩成1次；
这些是源码操作次数，不是已量到的阶段耗时占比。
候选完整 patch、源码/二进制身份保存在 `rope-candidate.patch` 与
`rope-candidate-build.json`，初筛 `rope-screen-run` 均值0.66730884 s、PCM与基线相同。
正式 ABBAABBA 每槽1预热+5热轮，结果 `rope-abba-[0..7]-[eager|candidate]`：

| 槽 | 臂 | 热合成均值 s |
| --- | --- | ---: |
| 0 | eager | 0.71179918 |
| 1 | candidate | 0.63842090 |
| 2 | candidate | 0.67628970 |
| 3 | eager | 0.63311062 |
| 4 | eager | 0.72478042 |
| 5 | candidate | 0.61834936 |
| 6 | candidate | 0.61253708 |
| 7 | eager | 0.70903626 |

Eager均值0.69468162 s，区间0.63311062–0.72478042；候选均值0.63639926 s，
区间0.61253708–0.67628970，表观改善约8.39%。48轮PCM与3.36s长度全部一致、
所有进程正常清理，但两臂区间重叠，前四槽约2.25%改善、后四槽约14.15%，
时序漂移仍大；没有独立阶段计时支持局部收益归因。撤回运行时代码，外部冻结产物
继续可恢复，不因未到10%撤回。`git diff --exit-code -- crates/moss-tts/src/nano/transformer.rs`
确认该文件回到实验前。下一步先测生产 global/local、RoPE/采样/codec 的成本占比，
再决定是否重试；不要把源码重复次数当作瓶颈已证实，或搬回已撤回的 mask 候选。

本轮只保留官方适配器、证据和文档。Python py_compile、cargo fmt --check、
git diff --check通过；生产与候选都完成独立 CUDA release 构建，LNK4098警告仍保留。
本轮无保留的 Rust 运行时代码，不将上述真模型测试代称为全后端取消/有界流回归。
## 多句段落验证（当前冻结生产版本）

新增 `paragraph.txt` 是自拟64字元、192 UTF-8字节的完整段落，非用户小说：
“清晨的阳光穿过窗帘洒在桌上，小明整理好书包，准备出门去学校。他走过安静的小路，
看见路边的花朵正在开放，心里充满了对新一天的期待。”（原文件为单行，无换行。）
本轮沿用已冻结 probe 与官方带 EOS guard 的适配器，没有同时编译或推理其他模型。

`paragraph-native-job.json` / `paragraph-native-run`，3进程各1预热+5热轮：

| 进程 | 完整热合成均值 s | 首PCM s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 3.28001822 | 0.07596540 | 16.08 |
| 1 | 3.36307920 | 0.07771160 | 16.08 |
| 2 | 3.42857486 | 0.14590892 | 16.08 |

组均值3.35722409 s，RTF0.20878259。18轮PCM均为
`104f58edf218780c0b1a0baec192b632c266d2174df9e923bdfb2279af042f18`，
均success且所有进程清理通过。生产自动拆成两段，字节范围[0,90)、[90,192)，
`paragraph-source-output-audit.json` 对18轮逐段重算原文摘要，确认相邻无缺口且覆盖到192。
这证明调度范围完整，不单独证明语音逐字准确。continuation=false，第二段是新参考请求。

`paragraph-official-job.json` / `paragraph-official-run` 同样3×(1+5)：

| 进程 | 完整热合成均值 s | 生成 s | codec+回传 s | 音频 s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7.59529780 | 7.54893926 | 0.04635854 | 15.20 |
| 1 | 8.17848162 | 8.13122604 | 0.04725558 | 15.20 |
| 2 | 8.02546622 | 7.97896124 | 0.04650498 | 15.20 |

组均值7.93308188 s，RTF0.52191328；18轮均成功，190帧、实际EOS7，PCM统一为
`754ee798d54659420747dd6cd8d0aa5d15d34a6b2413b0b4390be668b9a89387`。
官方Job commit峰值4.858–4.907 GB。官方适配器一次完整提示，不复制生产两段调度，
仅对齐首个请求有效seed；第二个生产请求会使用不同call seed，RNG算法也不同。
因此该结果是完整段落、各自实际调度路径的比较，不能归因为同工作量的内核加速。
官方首PCM仍按整段交付。Torch版本偏差等短句记录的条件继续适用。

复用同一冻结SenseVoice辅助工具，`paragraph-readback-results.json` 中两份完整WAV均回读为：
“清晨的阳光穿过窗帘洒在桌上小明整理好书包准备出门去学校他走过安静的小路看见路边的花朵正在开放心里充满了对新一天的期待”。
当前未发现漏字；这是单识别器、每臂一条固定PCM的辅助证据，情绪标签不作质量分。
它不替代人工试听、音色/韵律判断、双模型验证或超长篇章/接续验收。
原始输入、Job、完整音频、逐轮结果、summary与生产HTML报告均在本轮D盘证据根目录。
### 真实 CUDA 后端取消、队列与复用回归

新增 `moss::nano::tests::real_cuda_cancel_full_queue_and_reuse_preserves_pcm`，
仅CUDA feature编译、默认ignore，显式 `TALECHIME_NANO_CUDA_ROOT` 读取本地资源，不下载。
在Tokio LocalSet上只加载一个生产NanoBackend，直接采样seed42（此处不经过Pinned策略），
完整请求收集48k双声道有限PCM并要求End。随后取消未消费流、接收首PCM后实际观测
capacity=1且len=1的满音频队列再取消、提交被拒的无效音色，每步后复用同一后端
重新收集完整请求；四次完整结果222720个F32样本逐位相同。
满队列观测证明容量已占满，不声称已捕获owner正阻塞在某次send的瞬间。
无效音色是请求前校验失败，不代称任意模型推理错误的注入恢复。

真实GPU测试通过，1项/2.37s。普通Nano测试5项通过，新增真实测试默认忽略；
普通命令显式排除旧的环境触发CPU真模型EOS测试，没有进行CPU完整模型复测。
冻结二进制、源码/Cargo.lock摘要与恢复入口见外部 `cuda-state-build.json`，
原始构建输出 `cuda-state-build.jsonl`、`cuda-state-build.log`，测试日志
`cuda-state-test.log`、`nano-ordinary-tests.log`。复现：设置上述资源目录与CUDA DLL PATH，
运行收据中exe `real_cuda_cancel_full_queue_and_reuse_preserves_pcm --ignored --nocapture --test-threads=1`。
测试仅验证此提示/音色/设备路径；未覆盖所有推理故障、接续、播放完成或长时间压力。
本次新增CUDA状态测试的收尾门禁通过：release Clippy lib/tests -D warnings（1m44s）、
release rustdoc -D warnings（3.85s）、cargo fmt检查与diff空白检查；skill本地引用70条均可解析。
Clippy/rustdoc原始日志为 cuda-state-clippy.log、cuda-state-rustdoc.log。

## 阶段归因与生成嵌入复用（当前候选）

新增仅 `cfg(all(test,feature="cuda"))` 编译的 `nano/cuda_profile.rs`。
诊断显式提供本地模型、官方对齐seed用例导出的prompt rows、新输出路径，默认ignore，
无下载。只加载一份Nano和codec，先调用原generate并按3帧解码，再同步重演相同生成
顺序；每轮完整比较codes和F32 PCM bits，要求自然EOS。该诊断不经过Registry、
SynthesisStream、发送队列或播放，不把它的时间替换生产请求基线。
所有阶段前后同步CUDA，host计时包括分派/同步和该阶段执行，不是GPU kernel profiler。

基线 `stage-build.json` / `stage-profile.json` / `stage-summary.json`，1进程1预热+5热轮；
固定有效seed3560097011，42帧、322560个双声道PCM样本，6轮codes/PCM逐位一致。
未插桩模型+codec均值0.58383042 s，同步重演均值0.60245520 s，均不是生产请求时长。

| 基线阶段 | 次数/请求 | 热轮同步host均值 s |
| --- | ---: | ---: |
| local audio Transformer | 672 | 0.20854968 |
| global Transformer（含prefill与decode） | 43 | 0.16470708 |
| 3帧流式codec | 14 | 0.08442620 |
| 生成后通用row嵌入重建 | 42 | 0.05385702 |
| audio sampling（包括回传与CPU处理） | 672 | 0.03922078 |
| audio embedding（采样后供下一local步使用） | 672 | 0.01465536 |

其他阶段及逐轮值保留原始JSON。不能由audio_sample总量直接断言排序或传输占比，
也不能从上述同步host占比推算真实GPU kernel占比。

### 实现和逐位门禁

原实现采样每个音频通道后已经做一次embedding，最后又通过通用 `embeds([row])`
重新上传17个ID、查相同embedding并执行mask/where/multiply。新实现复用原有slot9
和16个audio embedding，从slot开始按channel0..15顺序逐项相加，作为下一步global输入。
只优化完整生成行；prompt/参考/接续初始化仍使用原通用mask路径。生成code一定落在
0–1023的合法码本范围，16个通道均有效；不能推广成允许忽略prompt的1024 mask。
没有把有序加法改成stack+sum，没有改变dtype、采样/RNG、权重、EOS、codec分块，
不增加跨请求缓存或运行时开关。累加Tensor是本帧局部变量。

冻结生产候选 `embedding-candidate-build.json`，probe SHA256
`b62b684e7c2e4201b628ba79dc482487d26d9af08eadb3b57fffdf23bf6eaada`，
features与原基线一致。主实现patch为 `embedding-candidate.patch`；最终实现与诊断源
另存 `embedding-final-nano.rs` / `embedding-final-cuda-profile.rs`。
诊断基线/候选源的外部重建快照分别为stage-baseline-nano.rs、stage-baseline-cuda-profile.rs、
stage-candidate-cuda-profile.rs，均逐字节核对了冻结构建收据中的原始SHA256；
这是可恢复的历史源，不用当前文件冒充旧构建输入。

候选 `stage-candidate-profile.json` 同样1+5，同步重演codes/PCM与generate逐位一致。
原row_embed每请求0.05385702 s，改为672次顺序row_accumulate共0.00555720 s，
该局部工作观察到约89.68%时间减少；两臂各一个进程，不称跨进程置信区间。
新row_accumulate每步分别同步，比原整个row_embed一次同步粒度更细，原始边界明确。
候选未插桩模型+codec0.52030544 s，同步诊断0.57769610 s，仅作本地归因参考。

补充独立CUDA码本门禁 `real_cuda_generated_embedding_reuse_matches_generic_rows`：
1026组行覆盖16个码本各自全部1024条目，外加全0/全1023边界行，复用结果与保留的
原通用embeds逐位相同、全部有限；1项/2.55s通过。它验证生成行的数学等价，不覆盖
含mask或不同prompt结构。完整真实CUDA owner取消/满队列/无效音色复用测试在候选
重新执行通过（2.57s，四次完整结果222720个样本逐位相同）。
候选模型普通fixture测试8项、后端普通Nano测试5项通过，不下载、不隐式跑CPU完整模型。
原始日志和冻结资格二进制见embedding-codebook-test.log、embedding-model-ordinary-tests.log、
embedding-backend-ordinary-tests.log、embedding-state-test.log、embedding-qualification-builds.json、
embedding-final-model-build.json。

### 正式生产ABBAABBA与收益边界

每槽独立进程，各1预热+5热轮，A为原不可变probe，B为嵌入复用probe。
不并行编译/推理，不剔除慢轮，原始证据 `embedding-abba-[0..7]-[eager|candidate]`，
汇总 `embedding-abba-summary.json`：

| 槽 | 臂 | 热合成均值 s | 首PCM均值 s |
| --- | --- | ---: | ---: |
| 0 | A | 0.60722146 | 0.04465844 |
| 1 | B | 0.60833022 | 0.04261720 |
| 2 | B | 0.60484538 | 0.06136400 |
| 3 | A | 0.66986470 | 0.06243762 |
| 4 | A | 0.65070080 | 0.08169430 |
| 5 | B | 0.55737312 | 0.06419624 |
| 6 | B | 0.70639766 | 0.07817536 |
| 7 | A | 0.68177650 | 0.06426060 |

A均值0.652390865 s、进程范围0.60722146–0.68177650；B均值0.619236595 s、
范围0.55737312–0.70639766，整链点估计约5.08%改善，但范围重叠且B槽6更慢，
不能宣称已证明稳定整链收益。48轮全部成功、进程清理通过、音频均3.36s，
PCM均与原基线d2e6692e…相同。首PCM组均值也只记录原值，不归因小差异。
本候选的保留依据是独立局部阶段证据和等价验证，而不是单次整链均值或10%硬门槛；
最终长文本回归与保留状态在本节收尾记录。

Clippy模型lib/tests（38.11s）、后端lib/tests（2.27s），均release且-D warnings；
rustdoc模型（1.85s）、后端（2.82s），均-D warnings，通过。完整日志以embedding-为前缀。
本轮未运行Metal或CPU完整模型；公共Rust有序复用实现没有该平台的实机性能结论。

### 长文本收尾与最终保留项

候选 `embedding-paragraph-job.json` / `embedding-paragraph-run` 再跑3独立进程各1+5：

| 进程 | 完整热合成均值 s | 首PCM均值 s | 音频 s |
| --- | ---: | ---: | ---: |
| 0 | 2.60354376 | 0.15234462 | 16.08 |
| 1 | 2.91363488 | 0.11200420 | 16.08 |
| 2 | 2.88857530 | 0.07432012 | 16.08 |

当前组均值2.80191798 s、RTF0.17424863，首PCM0.11288965 s。
这不是长文本ABBA，不能把相对较早3.35722409 s批次的变化全部归因于候选。
18轮全部success、cleanup通过、PCM与原104f58ed…完全相同；再次逐段重算源摘要，
两段完整覆盖192字节。因此复用旧同PCM的辅助回读证据，不重复跑同一识别器。
新报告 `embedding-paragraph-report/index.html`，逐进程统计 `embedding-paragraph-summary.json`。

较早段落首PCM均值0.09986197 s，当前均值反而更高；原始逐轮范围分别
0.0495066–0.2240492 s与0.0430440–0.2103049 s，重叠明显，不能宣称首包改善。
复核owner_first_pcm与consumer首包观测发现大部分差异已发生在owner就绪前，
不能直接归因队列交付；没有设备profiler证据，不指认GPU频率或OS调度为已证明原因。
保留所有慢轮，不为得到更好首包数值删除样本。

最终保留生成嵌入复用这个局部优化：约89.68%的对应阶段时间减少，输入嵌入1026组
逐位等价、短句48轮与长句18轮完整PCM一致、取消/有界队列复用通过、普通测试及
Clippy/rustdoc通过。整链短句5.08%仅作有较大漂移的点估计，不标记为稳定收益；
首包回退也未被平衡长句试验证实或排除。该结论不依赖整链10%门槛。
此前RoPE原型继续保持撤回状态，两者没有合并。

后续优先检查local/global的真实设备成本，以及audio_sample中CPU排序与回传各自占比。
另注意当前生产探针用WavWriter::new(File)逐样本导出音频；导出在热计时外，
但长文本轮间存在较长未计时区间。本轮未改导出，A/B都使用同一导出实现；
尚未独立测量导出成本或证明它造成频率漂移。若后续缓冲写入，应单独验字节/错误
处理并重建两臂，不能拿改了导出的probe与旧probe直接归因模型收益。

## 缓冲导出后的当前基线

采用不可变probe `20d2ec2d28d710c9414eb1c8b77d6e449e823d89b8775d6923113405cfb89394`，
只变导出策略、保留已验嵌入复用；不能将其与旧直写探针的耗时变化当成新模型收益。
`buffered-native-plan.json`沿用实际build.source，Job仍验证同一生产资源和参考条件。
短句与64字元段落各3×(1+5)，全部成功且清理通过，分别保持原d2e6692e…与104f58ed…PCM。
短句逐进程热均值0.54013536、0.55233110、0.70914602s，组均值0.60053749s，
首PCM组均值0.05887285s；仍有明显进程间波动，不能称缓冲已经消除GPU漂移。
段落逐进程2.70276898、2.78672332、2.61332172s，组均值2.70093801s，
输出16.08s、首PCM组均值0.05907557s。原始buffered-[short|paragraph]-run、汇总
buffered-native-summary.json与同名前缀报告均保留；旧ABBA不回填新收据。

### 后续撤回：Nano CPU 采样局部排序

同一 Windows RTX5070/9700X 上，检查同步阶段归因后调查 CPU audio_sample。
原采样每次全排序1024个分数，top_k=25；候选以 select_nth_unstable_by 取得阈值，
保留阈值所有并列项，再按分数及原token ID排序，恢复稳定并列顺序。权重计算、
逐token求和/逆CDF扫描及LCG均不变，不改变采样策略。

独立 release Rust 排序微基准：256组1024分数（含重复分数），每臂20000次、5轮，
全排序0.286577–0.289748s，候选0.030418–0.031637s。它只证明合成分数上的排序局部
收益，不涵盖GPU读取、全部采样或真实模型。完整原实现作为独立测试oracle，128组
分数×5种top_k×32次连续抽样，共20480次；含正负零、重复值、历史重复惩罚及交替
nucleus设置，token和RNG状态均一致。测试源码和日志保存在仓库外。

随后通过标准 perf build 冻结候选，CUDA release构建4m19s成功、Source guard通过。
保留原构建出现的LNK4098 LIBCMT冲突warning，不把成功构建称为无警告链接。
基线为 buffered-production-build.json（同一嵌入复用和缓冲WAV实现），候选为
sampling-selection-build.json；只有Nano采样排序是本次模型候选。其他已有工作不
据这次对照重新验收。正式运行关闭阶段观测，串行ABBAABBA，各1预热+5热轮：

| 顺序 | 臂 | 热轮均值 s |
| --- | --- | ---: |
| 0 | A 原实现 | 0.58573486 |
| 1 | B 候选 | 0.58099992 |
| 2 | B | 0.61782344 |
| 3 | A | 0.55778156 |
| 4 | A | 0.55061338 |
| 5 | B | 0.56272206 |
| 6 | B | 0.58335810 |
| 7 | A | 0.58516288 |

A组均值0.56982317s、进程范围0.55061338–0.58573486；B组0.58622588s、范围
0.56272206–0.61782344。点估计慢2.88%，范围重叠，未证明回退或改善的稳定因果。
48轮success、cleanup通过，音频均3.36s，PCM SHA全部
 d2e6692e13ef20a1364e0547dc3703c0f9705b985274285e307ef759469fd6e0。
本次未另采真实codes，不把PCM一致改称已记录codes一致；CPU采样oracle另提供
严格的抽样序列证据。既有取消/状态门禁没有在本轮重跑。

结论：撤回运行时候选及其测试副本，sampling.rs恢复为实验前完全相同字节。
微基准是合成排序输入，没有真实采样局部收益证据，且完整请求没有改善，不为此
增加维护路径。该结论不是整链10%硬门槛；有真实局部收益的其他候选仍可独立评估。
下一候选优先从占比更大的global/local Transformer GPU路径提出具体假设。

外部根保存 sampling-rank-screen.rs/exe、sampling-selection-candidate.rs、
sampling-before-selection.rs、sampling-selection-oracle-tests.log、8组job/plan/run、
sampling-selection-abba-summary.json（含全部热轮）、构建日志/二进制收据及撤回后
测试/Clippy/rustdoc日志。prepare_sampling_selection_abba.py保留任务生成方式。


撤回后首次检查因Copy-Item保留旧mtime，Cargo复用了含候选测试的产物；该批
sampling-selection-restored-*.log不作最终源码验证。发现测试名不符后更新源文件mtime，
重新编译，最终12个普通测试通过（真模型可选测试无环境时提前返回，不算实机门禁），
Clippy/rustdoc通过，最终日志为 sampling-selection-restored-*-final.log。
最终sampling.rs与实验前SHA一致，且git diff为空。复制旧源码后的“Finished”不能
单独证明已按恢复内容重编译，需检查实际测试列表或新编译证据。


### CUDA 单 query 的全可见因果 mask 候选

设备/依赖沿用本记录的 Windows11 / 9700X / RTX5070、CUDA13、Candle0.11 fork
2266099d。shared attend 每次构建CPU mask并上传、创建负无穷张量、执行where。
候选仅在CUDA、qlen=1、klen>0、无context窗口、query_offset-key_offset=klen-1时
跳过必然全false的mask。checked_sub防止位置下溢；CPU/Metal、prefill、未来key、
窗口及其他位置关系继续原路径。没有改变QK缩放、softmax或V matmul及缓存状态。

独立旧实现CUDA oracle覆盖540组F32/F16/BF16、klen=1/2/17/226/269、qlen=1/3、
key起点0/7、不同query位置和None/1/8窗口，输出逐位一致。首轮“全部结果有限”
断言错误地覆盖了原实现本就全mask的空窗口；保留失败日志，修正为无窗口输出有限、
所有组合仍逐位对照，不改变运行时空窗口行为。

test-only线程局部RAII参考作用域可强制执行原始完整mask；不进入产品构建，也没有
生产环境开关。真Nano、同prompt、原始LCG seeds3560097011/42/7，分别42/29/32帧，
完整codes及322560/222720/245760个原始PCM F32位均相同，均观察自然EOS。
最终按--test-threads=1串行执行，日志unmasked-cuda-model-serial.log，测试二进制
冻结于unmasked-model-test-build.json。此正确性运行不混入正式计时。

标准perf build的unmasked-build.json冻结候选；基线buffered-production-build.json
已有相同生成嵌入复用和缓冲WAV。两臂均observation=off，串行ABBAABBA，各1+5：

| 顺序 | 臂 | 热轮均值 s |
| --- | --- | ---: |
| 0 | A 原mask | 0.56479132 |
| 1 | B 候选 | 0.52433492 |
| 2 | B | 0.53052750 |
| 3 | A | 0.55720478 |
| 4 | A | 0.55673796 |
| 5 | B | 0.52988584 |
| 6 | B | 0.52880738 |
| 7 | A | 0.56625310 |

A均值0.56124679s、进程范围0.55673796–0.56625310；B均值0.52838891s、范围
0.52433492–0.53052750，约降低5.8544%，范围不重叠。48轮success、cleanup通过，
音频均3.36s、PCM全部d2e6692e13ef20a1364e0547dc3703c0f9705b985274285e307ef759469fd6e0。
这不是与官方Python的同条件排名，也不把此收益与嵌入历史百分比直接相加。
性能收据冻结于新增test-only oracle之前；生产实现相同，Source身份仍保留各自实际快照。

本函数也被MOSS1.7B CUDA调用；primitive等价门禁覆盖其支持dtype，但本轮不宣称
1.7B完整请求的性能或状态已复验。Mac路径没有启用这项快路，未执行Mac测量。
owner取消/满队列、长文本及最终工程检查在本节后续补记。


候选后续验证：生产owner立即取消、容量1队列实际填满后取消、无效音色失败后复用
均通过，同一owner多次完整222720个PCM位保持一致。首次把模型目录误当资源根，
初始化即失败，未产生有效轮次；改用~/.talechime/resources后同一冻结测试二进制通过，
保留两份日志unmasked-cuda-owner.log/unmasked-cuda-owner-final.log及
unmasked-owner-test-build.json，不修改运行时路径解析。

64字元/192字节段落再跑3独立进程各1+5，逐进程热均值2.53541026/2.52643114/
2.53258114s，组均值2.53147418s，音频全部16.08s。18轮success、cleanup通过，
PCM均104f58edf218780c0b1a0baec192b632c266d2174df9e923bdfb2279af042f18，与先前
相同生产段落完全相同，既有分段覆盖/辅助回读证据继续适用；没有重复宣称新的试听。
这不是长文本ABBA，不把相对历史批次2.8019s的全部变化归因本候选。统计和标准
Native报告为unmasked-paragraph-summary.json、unmasked-paragraph-report/index.html；
短句单个B进程报告为unmasked-short-report/index.html，不能把它当完整ABBA汇总。
完整短句8进程统计在unmasked-abba-summary.json。

保留这项无状态CUDA mask省略，依据短句ABBA非重叠范围约5.85%收益、540形状数值
oracle、三个seed的完整codes/raw PCM/EOS、生产取消复用及段落PCM一致；不存在整链
强制10%条件。历史RoPE及CPU排序候选继续撤回。本次仍不解决官方完整条件排名、
超长/接续和人工试听，MOSS1.7B完整请求的新组合回归另列待补。


最终工程门禁：普通模型测试12项通过（可选真模型测试无环境提前返回，不计实机验收）；
release模型cuda all-targets及backends moss-nano-candle-cuda,performance-tools all-targets
Clippy -D warnings通过，CUDA模型rustdoc -D warnings通过；fmt/diff检查通过。
日志unmasked-cpu-tests.log、unmasked-cuda-clippy.log、unmasked-backends-clippy.log、
unmasked-cuda-rustdoc.log；原始release链接LNK4098警告仍保留在构建/测试日志。
unmasked-evidence.json固定本轮收据、逐进程结果、原始源码oracle及工程日志SHA。

## 2026-10-25：forward内RoPE复用保留

新方案不扩大预计算表，不改变GEMM形状；Nano按自身F32/interleaved/base10000、
12层global/1层local重新验证。正式ABBAABBA短句0.576622→0.464448秒（−19.45%），
段落2.609205→2.229709秒（−14.54%），两者进程范围不重叠；短句首PCM−21.02%，
段落首PCM仍inconclusive。96轮PCM、576资产、源范围/chunk边界保持，三seed
codes/raw PCM/EOS及旧1.7B/VoiceDesign消费者回归通过。先前宽表原型继续撤回，
不能将此新方案与旧失败或各历史百分比混算。源码、同期A/B、状态、ASR与容量
见[forward交错RoPE记录](moss-nano-windows-forward-rope-2026-10-25.md)。
