# OmniVoice Windows CUDA 调查（进行中）

当前结论：保留 CUDA confidence gather、GQA、请求内 forward 图和保序 decoder kernel。
既有图回放的整链验收见对应阶段；新增 decoder 的局部收益和完整 codes/PCM 已验证，
增量整链均值观察少 3.46%，但进程范围重叠，不宣称整链改善已证明。参考音色的官方
复测仍有明显波动，尚未严格证明整个后端达标；质量试听及其他负载仍待验证。以下按实验阶段
保留旧结论和后续修正，不将早期状态视为当前状态。

## 范围与当前基线

按用户 2026-10-24 的最新指令，Windows 性能目标调整为优先 OmniVoice，基于已合入
Mac 优化的 main `42a7887` 开展；Nano 调查和原始数据保留，暂时后移。用户同时取消
整体链路默认 10% 门槛：可信局部收益可结合维护成本保留，整链结果独立报告，输出、
质量和生命周期门禁不变。先前 goal 中的顺序和 10% 描述以本次明确指令为准。

main 已快速前进到该提交。恢复本地 Windows 修改时，两份 skill 文件出现文本冲突，
已合并保留上游 OmniVoice CPU/Metal 经验、新收益准则及本地资源/二进制保存经验。
恢复时未修改上游 OmniVoice 模型实现；后续 CUDA 专用修改如下。Mac 的阶段结果不作为 Windows CUDA 的收益。

本机为 Windows / Ryzen 7 9700X / RTX 5070 12 GB，驱动 616.56，Rayon 请求 8 线程。
未锁频、未改变电源计划，桌面 GPU 程序仍驻留。模型 revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4` 的生产资源校验通过。独立 release catalog
和 probe 使用 `omnivoice-cuda,performance-tools`，完整身份见构建收据。

## 准备与 smoke

官方仓库 `k2-fsa/OmniVoice` 固定到与 Mac 调查相同的
`08be0b4ccbac3e13e374e86fbfead4b4cac343e2`，实际导入路径已核验。
隔离环境使用 torch 2.8.0+cu128、transformers 5.3.0，Torch 通过明确的只读 `.pth`
复用既有 CUDA 安装，不修改相邻项目环境。官方 import 已通过，正式合成对照待执行。

Rust production Registry/Backend 短句“你好，欢迎使用叙铃。”，默认 narrator、seed 42、
continuation=false、32 步、CFG=2、zh、speed=1 的 CUDA smoke 已成功。此路径按整段
交付，不是原生实时流式；人工试听和逐字内容验收 pending。

初次 smoke 模型执行成功但父级拒绝提交成功：Job 的资源根 `C:/...` 与 prepare 收据
的 `C:\...` 拼写不同，造成 JSON 身份摘要不一致。失败目录完整保留；采用准备收据
原路径的新 Job 后重跑成功。不得修改旧结果或摘要绕过校验。

收据、输入、原始音频及结果在 `~/.talechime/benchmarks/windows-omni-20261024-*`。
`smoke-canonical-run` 是有效接线验证；正式短句 `short-run` 使用 3 个独立进程、各
1 预热+5 热轮，最终结论必须核对全部轮次后再填写。构建的源码摘要还包括本地 Windows
工具和 skill 修改，不把提交号单独当作完整源码身份。

短句正式批次已完成：三个进程均 success、cleanup_verified=true，各 5 热轮。
进程均值为 0.759231 / 0.697154 / 0.661862 s，音频均 1.63 s；mean-of-means
0.706082 s，RTF 0.433179。首 PCM 与整段完成时间接近，符合 segment delivery。
每个进程内热轮摘要一致。进程间耗时存在明显漂移，单臂绝对基线不能证明候选收益。
报告为同前缀 `short-report/index.html`；尚未与官方配对，不宣称优于官方。

## 待调查方向

当前 CUDA 与 Metal/Accelerate 的分支不同：CUDA attention 仍手写 QK/softmax/PV、
重复 K/V，并使用低精度 RMSNorm 的既有差分路径；不能直接启用 Mac 特化并假定数值
保持。greedy confidence 的 argmax gather 特化目前覆盖 CPU/Metal，CUDA 仍有独立
归约。先用实际输入和阶段归因排序，再独立验证候选，不凭源码观察宣称瓶颈或收益。

后续交付仍包括当前主线短长语料生产基线、匹配条件的官方 CUDA 对照、算子/阶段
候选、完整 codes/PCM 与取消复用检查、组合整链结果及 Windows 经验回流 skill。

## 官方 CUDA 首轮绝对基线

Rust xtask 新增显式 reference 适配器运行入口，官方 Python 仅推理，复用 native owner
清理、超时、事件/音频审计。Job 固定官方源码、权重、语料及适配器摘要；环境和实际
attention/dtype 单独记录。3 个进程各 1 预热 + 5 热轮，均完成并确认清理。

| 进程 | 合成均值 s | 音频 s | RTF |
| --- | ---: | ---: | ---: |
| 0 | 0.815297 | 2.12 | 0.384574 |
| 1 | 0.842764 | 2.12 | 0.397530 |
| 2 | 0.823402 | 2.12 | 0.388397 |

mean-of-means 0.827154 s、RTF 0.390167；实际执行为官方 SDPA、LLM CUDA F16、
decoder CUDA F32，与 Rust 精度分层一致。每轮重置官方 seed=42，Rust 仍按生产
固定 seed 路径。源码/依赖/参数详见 `windows-omni-20261024-official-job.json` 及
`official-run-v2/process-*/runtime.json`。首次工具接线因重复发布 executor 已写的
capture 被拒绝，保留 `official-run/process-0`，修正后完整重跑为 v2，未补成功标记。

Rust 0.706082 s / 1.63 s 与官方 0.827154 s / 2.12 s 的负载不同：绝对耗时和 RTF
指向不同排序，不能据此宣布追平或超过官方。两边是先后批次，未做框架间交错；
尚无人工内容/听感验收。后续须核对时长估计、文本/seed 派生，补长文本与同输入
算子归因，不能靠强行缩短生成来追平。所有音频保留。

首个 CUDA 候选将 greedy confidence 的第二次词表 max 改为复用 argmax gather。
GPU 免权重位级测试覆盖 1024/1025/4097 词表、两 batch、重复最大值、mask 与 CFG
0/2，逐项 token 和 confidence bits 相同；这不是完整模型正确性或收益验收。

### confidence 局部探针与短句筛选

独立 release 测试可执行文件已保存到 `~/.talechime/benchmarks/`，三进程串行运行。
合成 F32 `[1,8,T,1025]`，比较 argmax+I64 cast+max 与 argmax+gather+I64 cast，
各自预热，5 组 ABBA，每槽 20 调用、槽前后 CUDA synchronize。输出日志
`windows-omni-confidence-local-{0,1,2}.log`。不是完整 scoring，也不含真实模型输入。

| T | 进程 | 原边界 μs | gather μs | 耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| 41 | 0 | 45.148 | 29.246 | 35.2% |
| 41 | 1 | 48.614 | 28.386 | 41.6% |
| 41 | 2 | 69.880 | 57.176 | 18.2% |
| 130 | 0 | 119.815 | 65.062 | 45.7% |
| 130 | 1 | 120.835 | 64.017 | 47.0% |
| 130 | 2 | 118.236 | 66.059 | 44.1% |

短句四个独立进程 ABBA，各 1 预热 + 5 热轮：base 0.731655、gather 0.742213、
gather 0.648213、base 0.657562 s，均为 1.63 s 音频。两臂进程均值分别约
0.694608 / 0.695213 s，未证明整链改善；后半两臂同时更快，存在漂移。
所有热轮 PCM 摘要均为 `481dcfedaaea22629e08b1374e9ed252b58230411f9bbeece816134cd064ca44`。
原始四槽在 `windows-omni-gather-screen-{slot}-{arm}-run`。候选仍需完整 codes、真实
输入及长文本证据，不因局部探针而宣布优化已验收，也不因整链小于 10% 自动撤回。

Rust reference 工具 48 个 xtask 单测及 native process 的两个集成测试通过；默认
取消 CLI 用例未设置独立 build receipt 时仅走跳过路径，不记为实际控制台信号验收。
xtask Clippy 和拒绝警告的 private rustdoc 通过。CUDA 发布探针与库测试均构建成功，
链接保留既有 LNK4098 警告；尚需最终组合门禁和本轮新增路径的进一步验证。

### 长文本基线与完整 codes 诊断

固定长文本保存在 `windows-omni-20261024-long.txt`，身份由 Job 冻结。生产 baseline
3×(1 预热+5 热轮) 全部成功，音频 16.4 s，三个进程合成均值 2.645781 / 2.761613 /
2.688211 s，mean-of-means 2.698535 s、RTF 0.164545；各进程热轮摘要一致。
原始数据在 `windows-omni-20261024-long-run`。尚无官方对应长文本对照或人工试听。

另用 baseline/candidate 各 1 个进程、0 正式预热+1 长文本请求做诊断，显式设置
`OMNIVOICE_CODES_DUMP`，并在 Job observation_config 记录 dump 路径，故不作为吞吐
测量。此 dump 包含探针独立的短参考预热及三个生产文本段，四条记录的整数数量分别
384/736/688/1776，按 little-endian u32 count + i64 codes 完整解析，无尾部缺失。
两臂整个 dump SHA 均为 `5fbfa8e88e86c180210f6d2234cf2f0fcf934382aac8bb2517cb4401253b19e4`，
长文本完整 PCM 摘要也一致。原始目录/文件为 `windows-omni-codes-{base,gather}-*`。
诊断变量在每轮后清除，没有混入正式吞吐；生成成功不等于人工内容验收通过。

冻结的 CUDA release 库测试可执行文件运行普通测试：41 passed / 9 ignored；没有将
ignored 的真实 CPU/Metal 回放算作通过。新增 CUDA greedy 位级用例已另行显式运行，
合成局部性能用例另行串行运行。CUDA 库/测试 Clippy 拒绝警告通过。候选尚需真实
log_probs 的局部消费者成本和更多端到端证据；下一轮重点仍是阶段归因与官方负载对齐。

## 真实输入 confidence 验收与后续热点

新增 `cfg(test,cuda)` 私有捕获/回放，无生产导出开关。显式本地模型和新目录下运行
`stage0_model::cuda_attribution::capture_real_scoring_inputs`，按当前 core 的
Pinned(42).resolve(0,0) 派生模型 seed `7330414270409262323`。保存第 0/15/31 步
真实 c/u logits、masked log_probs、原布局；实际均为连续 F32 `[1,8,48,1025]`。
原始资产在 `windows-omni-real-scoring-inputs`，捕获不用于计时。

冻结 release replay 测试三进程串行执行，每步各 5 组 ABBA、每槽 20 调用，复现原
argmax→I64→max 与候选 argmax→gather→I64 的真实边界与操作次序，槽前后同步。
每步 token 和 confidence bits 均一致。完整逐槽数据为 `windows-omni-real-replay-*.log`。

| 进程 | 步 | 原边界 μs | gather μs | 耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 60.479 | 49.125 | 18.8% |
| 0 | 15 | 59.739 | 45.886 | 23.2% |
| 0 | 31 | 60.864 | 48.586 | 20.2% |
| 1 | 0 | 48.408 | 29.411 | 39.2% |
| 1 | 15 | 48.033 | 29.285 | 39.0% |
| 1 | 31 | 48.427 | 29.498 | 39.1% |
| 2 | 0 | 48.217 | 29.556 | 38.7% |
| 2 | 15 | 48.039 | 29.255 | 39.1% |
| 2 | 31 | 49.972 | 29.732 | 40.5% |

这是局部边界成本，不能乘层数或直接外推整链百分比；进程间存在漂移，未锁频。
结合真实输入局部改善、短句 ABBA 完整 PCM 一致、长文本完整 codes/PCM 一致及仅一条
设备分支的维护成本，按用户更新后的规则保留 CUDA gather。随机采样路径不变。
普通冻结 release 测试 41 passed / 11 ignored，新增 capture/replay 显式运行通过，
CUDA lib/tests/example Clippy 与拒绝警告的 private rustdoc 通过。

同步诊断另设 `OMNIVOICE_SYNC_PROBE=1`、`OMNIVOICE_STAGE_TIME=1`，清除后才恢复
吞吐。剔除独立短参考预热后，三个生产段 stage0_generate / forward_sync 墙钟为
634.37/589.31、802.53/749.02、1151.20/1041.26 ms。仪器模式下 forward 约占
stage0_generate 的 90%–93%，不是 kernel 独占时间。原始日志在
`windows-omni-sync-profile-run`。Nsight Systems 未找到；不冒称已做 kernel 级归因。

## 官方负载对齐：分段与后处理

长文本默认产品比较中，官方全文一次调用的三个进程均值为 1.052148 / 1.046464 /
1.038608 s，音频 16.24 s；Rust 核心实际按 UTF-8 范围 `[0,57)`、`[57,111)`、
`[111,249)` 发出三个独立请求。每个请求均执行 32 步，因此不能将全部差距算作
相同计算量下的框架开销。该产品默认比较仍保留，不更改 Rust 分段来换取表面速度。

新增官方同分段请求，按 Rust range 和 text_sha256 取原文；每段使用 core 当前
派生 seed `7330414270409262323/15690801507913080125/16434375112636111858`。
三进程各 1 预热+5 热轮：合成均值 2.209873 / 2.037580 / 2.124424 s，音频 16.6 s，
首段 PCM 均值 0.766036 / 0.654210 / 0.743144 s。mean-of-means 2.123959 s；
相较 Rust 原基线 2.698535 s，观察上 Rust 耗时高约 27.1%。非交错配对实验，
尚不作为严格框架排名，但已说明剩余差距不能靠 confidence 这一个小改动解决。
原始结果在 `windows-omni-official-matched-segments-run`，Job 冻结分段与全部 seeds。

独立官方长度诊断确认三段目标帧数为 92/86/222，与 Rust codes 数量一致；原始
解码样本分别 88320/82560/213120，后处理后 93120/87360/217920。诊断 wrapper
仅记录参数和 shape，不改变生成；不是吞吐运行。短句双方目标均为 48 帧；固定
Rust codes 的 CUDA raw decode 实测 46080 samples，官方短句 raw 同为 46080。
最终时长受不同生成波形及静音后处理影响，不能把 1.63/2.12 s 直接当作帧数比例。
同一 Rust raw PCM 交给官方 remove_silence 得到 36720 samples；后处理算法的严格
同输入对照仍需单列，未为追平计时改动生产处理或放宽输出门禁。

旧版单段参考适配器副本保存在本地 `windows/omnivoice_cuda_reference-before-segments.py`；
新 Job 重新冻结扩展适配器的摘要，不篡改旧 Job。新适配器支持显式 texts/seeds，逐段
串行推理并记录真正首段 PCM，最后拼接写盘不计合成；这与 Rust consumer 的汇总拷贝
边界略有不同，报告保留区别。未测试人工内容或听感，未宣称后端达到最终目标。

## CUDA GQA：保留原始 KV 的分组乘法

候选只作用于 CUDA F16、head_dim=128、Q heads 为 KV heads 的两倍、seq=8..512，
并检查 Q/K 连续性及各输入维度。将 Q 的 group 维折入 query 行，再与原始 K/V 相乘，
恢复原 attention 输出形状；省去两份 repeat-KV 的物化。CPU/Metal、其他 dtype、短于
8 或长于 512 的序列仍回退原路径。此处不是 fused SDPA，也未修改 softmax 或精度。

测试专用捕获第 0/15/31 步、第 0/13/27 层。真实 Q 为 F16 `[2,16,S,128]`，KV 为
8 heads；短句 S=63，长文本首段 S=117、末段 S=258。回放重建原 V 的 transpose view，
核对 stride，避免把意外预先 contiguous 的输入算作收益。局部边界包含 context 输出
transpose/contiguous/reshape，不含 o_proj；每个进程 5 组 ABBA，每槽 10 次，前后同步。
首末段各三个进程，九个真实层/步输入均 bit-exact；首段边界减少 34.2%–34.6%，末段
28.1%–28.8%。这是局部结论，不能加到 confidence 的百分比上。末段捕获作为独立请求
使用 call0 seed；完整长文本 codes/PCM 另覆盖实际生产的 call2。

合成位级检查覆盖 S=8/31/63/117/128/256/258/512、无 mask/末列 padding mask；
F32 与 S=7/513 的回退验证通过。计数式真实模型取消测试验证取消错误，再清除 probe、重设
seed 并复用 pipeline，完整 PCM 与未取消基线位级一致；进入模型步骤的证据见后续更正。
普通 release 库测试 41 passed /
15 ignored；这些 GPU ignored 用例另行显式串行运行。CUDA all-targets Clippy 和包含
examples 的 private rustdoc 拒绝警告通过。

完整长文本诊断与原 baseline 的四条 codes dump SHA 均为
`5fbfa8e88e86c180210f6d2234cf2f0fcf934382aac8bb2517cb4401253b19e4`，完整 PCM 一致。
原始数据为本地 `windows-omni-gqa-codes-*`。正式吞吐不启用捕获或 dump。

增量基线两臂均含已保留的 confidence gather，分别使用冻结的 gather build 和
`windows-omni-gqa-validated-build.json` 收据。首组采用 ABBAABBA，8 个独立进程，
每进程 1 预热+5 热轮，未启用连续 GPU 采样；每槽热轮合成均值如下。

| 槽 | arm | 合成秒 |
| ---: | --- | ---: |
| 0 | base | 2.57425418 |
| 1 | GQA | 2.48639640 |
| 2 | GQA | 2.48765002 |
| 3 | base | 2.80429802 |
| 4 | base | 2.61093298 |
| 5 | GQA | 2.44057402 |
| 6 | GQA | 2.50519558 |
| 7 | base | 2.70019472 |

base/GQA mean-of-means 为 2.672419975/2.479954005 s，耗时减少 7.20%，进程均值
范围不重叠；音频均为 16.4 s，RTF 约 0.16295→0.15122。首段 PCM 延迟分布重叠，
不宣称已证明首包改善。所有 48 轮完整 PCM 摘要均为
`ac88a7aa02dc6d2e5ba0ac283af9a7f5405237fb55ca14aa4dd6af783449f815`。
原始八槽为 `windows-omni-gqa-abba-{slot}-{arm}-run`，汇总保留每槽来源，不篡改各自
process_index=0 以通过多进程比较器。第二组另以相同顺序启用 GPU 采样，单独报告。

按更新后的 skill 保留此 CUDA 专用改动，无需达到整链 10%。与先前官方同分段
2.123959 s 相比，当前 2.479954 s 仍观察上慢约 16.8%；跨实验、非交错官方比较，
不能视为严格排名。质量试听、官方同输入后处理差异及剩余 forward 成本仍待解决。

### 连续 GPU 遥测复验

第二组仍为 ABBAABBA、每槽 1+5，两臂均启用 500 ms GPU 采样，其余使用相同冻结
输入和二进制。八槽均值依次为 base 2.73674072、GQA 2.53896338、GQA 2.42085672、
base 2.63264234、base 2.60986928、GQA 2.49714882、GQA 2.53899112、base 2.64706304 s。
两臂 mean-of-means 为 2.656578845/2.498990010 s，耗时减少 5.93%，进程均值范围
仍不重叠。48 轮 PCM 与首组同摘要，全部 success 且 cleanup_verified=true。
首包两臂仍有重叠，不提升为已证明的首包收益。两组独立报告，不合并成更好看的数字。

每槽采得 61–64 个样本，范围包含启动、加载、预热、热轮及收尾；不是热轮独占指标。
整卡时钟 202–2940 MHz、功耗 24.17–169.40 W、温度 40–60°C，整卡显存峰值
2921332736 bytes（约 2.72 GiB）。GPU 上存在桌面程序，WDDM 的进程显存读数不可用；
不能把整卡峰值当作本模型独占显存，也不能用全进程样本均值证明热轮锁频或能耗下降。
未锁频/改电源策略，采样工具本身也有成本。两臂在同样观测条件下的复验支持保留该优化。
原始数据为 `windows-omni-gqa-telemetry-{slot}-{arm}-run`，每个 attempt 的
`resource-samples.jsonl` 保留逐点值与 scope；汇总为
`windows-omni-gqa-telemetry-summary.json`。最终 fmt 和 diff 空白检查通过。

## stage0 forward CUDA Graph 可行性（候选，尚未验收）

固定形状的 32 步去噪没有跨步 KV 历史，候选边界为 prepare_embed_inputs、整个
backbone 和 audio_heads，输出完整 logits；采样、host RNG、tokens 更新及取消留在图外。
先以测试专用 hook 在真实请求中逐步对照，不改变生产消费路径。每个文本三个独立
进程，所有 32 步完整 logits 位级一致，包含 seq=63/117/258，合计 288 步。
局部计时在第 0/15/31 步各 3 组 ABBA、每槽 3 调用；双方同步，graph 包含固定输入
D2D 与连续输出复制，不包含 CPU bits 读回。原始日志为
`windows-omni-forward-graph-{short,first,last}-{0..2}.log`，冻结测试文件为
`windows-omni-forward-graph-tests.exe`。

| seq | 三进程 eager 均值 ms | 三进程 replay 均值 ms | 额外预热+录制+实例化+同步 ms |
| ---: | --- | --- | --- |
| 63 | 16.668 / 13.510 / 12.988 | 7.521 / 7.178 / 7.348 | 26.623 / 25.065 / 25.107 |
| 117 | 16.579 / 16.322 / 16.523 | 11.867 / 11.973 / 12.368 | 28.565 / 24.941 / 38.611 |
| 258 | 29.435 / 29.353 / 29.489 | 26.196 / 26.318 / 26.218 | 46.072 / 34.366 / 33.359 |

这些是局部可行性，不是整链收益；各步骤仍混有主机/GPU 环境漂移。录制额外执行的
预热仅用于实验；生产候选复用第 0 步的真实 eager 结果，录制不执行，之后第 1–31 步
才回放，全部录制成本计入请求。候选为 request-owned，静态与可变张量均走固定输入
缓冲；借用 Stage0Model 保证捕获权重活到图之后，不缓存跨请求历史。同步诊断和 debug
捕获走 eager。只选择 CUDA F16 dense、CFG batch=2、8 codebooks、seq=8..512、至少
两步的已验证范围，其他条件回退。recorder 暂留消费者私有模块，使用 workspace 既有
cudarc 0.19.10，不更改 Candle、链接模式或 CRT。

首次生产候选测试完整 A/B/A PCM 一致，但取消后的录图次数断言失败（实际 4，期望 5）：
固定第六次取消检查发生在模型开始前，不能证明图后取消。失败日志保留在
`windows-omni-production-graph-parity.log`。测试改为观察该请求第 0 步录图成功再取消，
重测前不将取消恢复记为通过。前文 GQA 的计数式取消用例只证明请求拒绝与后续复用，
“首步后”的归因不成立，以本段更正。候选尚需完整 codes/PCM、整链对照与组合门禁。

生产候选重测（`windows-omni-production-graph-tests-v2.exe`）：A/B/A 完整 PCM 对照、
录图后取消再复用通过，并断言 5 个 graph 实际录制成功；失败录制退出捕获态、错误
shape 拒绝后再次正确 replay，以及非连续输出复制预热的两个 recorder 用例显式通过。
对应日志为 `windows-omni-production-graph-{parity,recorder}-v2.log`，不覆盖首版失败证据。

生产 Registry/Backend 长文本诊断 `windows-omni-forward-graph-codes-run` 成功，四条
codes dump 与原基线同 SHA `5fbfa8e88e86c180210f6d2234cf2f0fcf934382aac8bb2517cb4401253b19e4`，
完整 PCM 同 SHA `ac88a7aa02dc6d2e5ba0ac283af9a7f5405237fb55ca14aa4dd6af783449f815`，
音频 16.4 s。此轮开启 dump，不用 1.920253 s 诊断值宣布正式收益。正式 build 为
`windows-omni-forward-graph-build.json`；后续测试改动只修正测试触发点，生产算法未变。

### 生产图回放正式对照：保留

两臂 baseline 均已含 GQA/confidence，增量仅接入请求内 forward 图。ABBAABBA、8 个
独立进程，每进程 1 预热+5 热轮，500 ms GPU 采样两臂同时开启，录制成本计入每次
请求。未锁频、未改变电源策略、未并行编译/推理。

| 槽 | arm | 合成秒 | 首段 PCM 秒 |
| ---: | --- | ---: | ---: |
| 0 | base | 2.51252710 | 0.69379522 |
| 1 | graph | 1.90629728 | 0.46553234 |
| 2 | graph | 1.91480088 | 0.46810766 |
| 3 | base | 2.57986968 | 0.79175996 |
| 4 | base | 2.48549062 | 0.74300908 |
| 5 | graph | 1.91035852 | 0.47578982 |
| 6 | graph | 1.91104534 | 0.47032466 |
| 7 | base | 2.48321854 | 0.73450784 |

合成 mean-of-means 2.515276485→1.910625505 s，耗时减少 **24.04%**；首段 PCM
0.740768025→0.469938620 s，减少 **36.56%**。两项进程均值范围均不重叠。音频均为
16.4 s，48 轮 PCM 全部同 SHA `ac88a7aa02dc6d2e5ba0ac283af9a7f5405237fb55ca14aa4dd6af783449f815`，
全部成功且进程清理通过。首 PCM 是 segment delivery，仍不等于实际播放完成。
原始目录 `windows-omni-forward-graph-abba-{slot}-{arm}-run`，汇总
`windows-omni-forward-graph-abba-summary.json`；所有逐轮和二进制身份保留。

遥测范围含加载/预热/热轮/收尾：整卡时钟 210–2932 MHz、功耗 22.60–185.15 W、温度
41–62°C；base/graph 整卡显存峰值分别 2921332736/2959081472 bytes，增加约 36 MiB。
这些是整卡数据，不是模型独占；WDDM 的进程显存仍不可用。全进程 RSS 峰值随进程
变化，不能从此声称请求工作集下降。图按文本段销毁，未增加跨请求图缓存。

结合真实逐步 logits、完整 codes/PCM、录图后取消复用、失败 capture 收尾、布局
复制门禁和正式整链改善，保留该候选。局部/GQA/graph 百分比不相加；整体相对 main
的收益需要直接共同基线另算，不能将本组误称为对 main 的 24%。官方对照另行重测。

### 官方同分段复测与当前边界

不修改原 Job/输入，使用 `windows-omni-official-matched-segments-job.json` 在新目录
`windows-omni-official-matched-after-graph-run` 重跑官方三个进程、各 1+5，进程均值为
2.074190 / 2.133999 / 2.098129 s，mean-of-means **2.102106 s**，音频均 16.6 s；
首段 PCM 均值 0.733486 / 0.701811 / 0.737812 s。所有进程清理通过，热轮官方 PCM
同 SHA `bd599c3605c5b895d2e642c60522435ac08493e1488ea49af6f96a8ccd6d6d7d`。
实际仍为 torch 2.8.0+cu128、transformers 5.3.0、SDPA、LLM F16 / decoder F32，
固定官方 checkout 无修改，适配器 SHA `057dd1555e80995af1ea2ba691065dd44e6f53b1bd74307c9cf585e669b36500`。
分段、每段 seeds、目标帧数和计时边界仍遵循前述对齐，不把 Rust 较短音频算成少帧。

当前 Rust 1.910626 s 相较本次官方 2.102106 s，观察上耗时少 **9.11%**，两组进程
均值范围不重叠。这里 Rust 来自正式 A/B 中的四个 graph 进程，官方另跑三个进程，
不是跨框架交错 A/B，也不是相同随机轨迹的算子排名。不能据此宣布所有语言、参考
音色、长度、步数和流式模式完成。官方全文单请求与 Rust 三段默认产品比较仍分别
保留；没有删除原先暴露差距的结果，也没有放宽输出、修改分段或后处理来获胜。

工程门禁：冻结 CUDA release 库普通测试 41 passed / 19 ignored（真实 GPU 用例另列），
CUDA 和默认 CPU 的 all-targets Clippy 拒绝警告通过；CUDA private rustdoc（含 examples）
拒绝警告通过；生产 performance_probe release 编译通过，fmt/diff 检查通过。构建继续
存在既有 LNK4098 警告，未改 CRT 或链接选项。原型/首版失败/修正版三个测试可执行
文件的摘要保存在 `windows-omni-graph-test-binaries.json`。人工试听仍 pending。

默认 CPU 库测试补跑完成：40 passed / 7 ignored，耗时 80.10 s；最后的既有 CPU
conv7 位级用例通过。普通测试没有下载模型。它与 CUDA release 的 41/19 计数来自
不同 feature，不合并成一个不存在的测试总数。

## 短句与参考音色补覆盖

本节仍使用相同冻结 base（GQA/confidence）与 graph build，两组各为 ABBAABBA、
8 个独立进程、每进程 1 预热+5 热轮，两臂启用 500 ms GPU 采样；没有并行编译/推理。

| 槽/arm | 默认音色短句秒 | 参考音色短句秒 |
| --- | ---: | ---: |
| 0 base | 0.62559340 | 0.94518586 |
| 1 graph | 0.30272192 | 0.73393798 |
| 2 graph | 0.30410230 | 0.73733756 |
| 3 base | 0.63790996 | 0.93749526 |
| 4 base | 0.60049188 | 0.95241516 |
| 5 graph | 0.29726102 | 0.73811516 |
| 6 graph | 0.29768830 | 0.73350066 |
| 7 base | 0.66865932 | 0.95512828 |

默认音色合成 mean-of-means **0.633163640→0.300443385 s（-52.55%）**，首 PCM
0.633115555→0.300394315 s。参考音色 **0.947556140→0.735722840 s（-22.36%）**，
首 PCM 0.947507795→0.735665420 s。两组各自的进程均值范围均不重叠，不能把 52%
外推到带参考前缀的计算量。此处均为整段首 PCM，与完成时间接近，不代表原生流式。

默认音色 48 轮 PCM 仍同 SHA `481dcfedaaea22629e08b1374e9ed252b58230411f9bbeece816134cd064ca44`，
音频 1.63 s；参考音色另 48 轮 PCM 均为
`01043855ad3be24ee83dcb35fb4c13f8ab9a10f2814980b6f639ce61091bda1b`，音频 2.0 s。
两组全部清理通过。参考音色独立 dump 诊断的两臂 codes SHA 同为
`ee1e8e496d397ebba6a36b745e2e3e39f65b1e376207f1d896cfd070f1ead880`；不混用诊断计时。
原始数据分别为 `windows-omni-graph-short-abba-*`、`windows-omni-clone-abba-*`、
`windows-omni-clone-codes-{base,graph}-*`，每组 summary 保留逐槽来源。

参考复用现有 VoiceStore 的 `custom:acceptance`，没有覆盖 WAV 或转写。首次独立
`windows-omni-clone-prepare-run` 新建生产 `prompt.json`，正式测量前同时冻结 WAV、
voice.json、prompt.json 的内容摘要；两臂消费同一缓存。参考 tokens `[8,129]`，
原始 ref_rms 约 0.0828476。转写来自现有记录，未新做人声身份、ASR 或人工试听认证。
不提交该参考录音或模型。该准备轮 load=0.296846 s、verify=1.553617 s、reference
warmup=2.224056 s；Rust 模型懒加载与完整短合成混在 warmup 中，不能称为纯参考编码。

官方参考适配器支持可选 reference_audio/reference_text，显式给出转写，禁用隐式 ASR
下载；create_voice_clone_prompt(preprocess_prompt=True) 在准备阶段执行一次，合成计时
复用该 prompt。三进程准备出的参考 tokens 均 `[8,129]`，与 Rust 帧数对齐；双方编码
值/预处理数值不要求逐位一致。官方热合成均值为 0.720994 / 0.720318 / 0.677546 s，
mean-of-means **0.706285980 s**、音频 2.12 s，PCM 摘要稳定为
`3c93ee946f90f74eedd59771bae7586fbadc1defe15ff0faebd59d708c0ecaa0`。当前 Rust 参考案例
观察上仍慢 **4.17%**，尚未达标；不能只报告无参考长文本已快过官方的结果。
官方各进程 model load 与 reference_prepare 分开记录在 runtime.json；例如进程 0
为 1.884799/0.398549 s，不与 Rust 0.296846 s 的适配器 load 直接排名。

官方原始目录 `windows-omni-official-clone-run`，Job 固定生产相同 WAV/转写、core 派生
seed=7330414270409262323、源码/模型摘要。Rust 每轮仍走生产缓存 JSON 读取与输入准备，
官方用内存 prompt；这是默认缓存边界差异，未人为删掉 Rust 开销。跨框架另批测量，
不是交错排名。后续重点是参考前缀下的剩余 forward/head/采样成本及后处理差异。

适配器扩展前已保存旧源码 `windows/omnivoice_cuda_reference-text-only-frozen.py`，
并新建 `windows-omni-official-matched-text-frozen-job.json` 保留旧长文本恢复入口。
旧 Job、旧结果未篡改；旧 Job 原工作区脚本已变化时，应使用新冻结 Job，不能改摘要
伪装原 Job 重跑。clone 的新 Job 直接引用独立冻结脚本，后续工作区编辑不影响复测。

磁盘检查发现 target/debug/incremental 有约 14 GiB 可再生 Rust 缓存。已核对 target
junction 指向，但删除被自动审批以 blocked by policy 拦截，未执行删除；模型、结果和
冻结二进制均保留。本轮性能测量期间没有清理构建目录。

### 派生 seed 短句复测与适配器回归

早期官方短句 seed=42 并非 core 实际交付的 seed，不用该旧值给当前短句排名。新增
冻结 Job `windows-omni-official-short-resolved-seed-job.json`，明确实际模型 seed
7330414270409262323；三个进程各 1+5 的均值为 0.717578 / 0.710650 / 0.780540 s，
mean-of-means **0.736256093 s**，音频均 1.69 s，官方 PCM 稳定 SHA
`10e7c63a4912e1f964ddc5587ca9b8af2c317106fd2ad4ebbb48e449662f55fb`。
当前 Rust 0.300443385 s，绝对耗时观察上少 59.19%；仍是另批跨框架测量，不当作严格
统计显著性证明。双方使用自己的完整后处理，最终时长不同。默认短句 Rust RTF
0.18432（此前增量基线 0.38844），官方 RTF 0.43565；不把不同分母隐藏在单一排名中。

独立 `windows-omni-official-clone-length-run` 诊断确认 clone 目标 48 帧、参考 129 帧、
raw 46080 samples、官方后处理 50880 samples；Rust 两条 codes dump 记录均 384 个
整数，即每条 8×48。Rust clone RTF 0.36786（增量基线 0.47378），官方 0.33315；
绝对耗时慢 4.17% 不等于 RTF 差距，也不将时长差异当成主干帧数差异。

扩展适配器的新冻结副本在无 reference 字段时做了 1 进程、0 预热+1 正式 smoke，
`windows-omni-extended-adapter-default-run` 的官方完整 PCM 与旧冻结适配器上述短句
SHA 一致。该 smoke 只验默认模式兼容，不补入正式 3×(1+5) 的吞吐统计。
本轮未修改 Rust 推理源码；Python AST、真实 clone/default 回归、fmt/diff 检查通过。
后续仍需缩小参考前缀下的剩余 forward/head 与采样成本，不以默认短句大收益替代它。

## 参考前缀剩余差距：target-only head 初筛

仅 cfg(test) 捕获实际每步 final_hidden，比较完整 audio_heads 后截取目标与先打包目标
hidden 再投影。默认 seq=63、参考 seq=212，目标均 48；各三个独立进程，全部 32 步
被消费 logits 位级一致。原始 corrected 记录为 `windows-omni-target-head-{default,clone}-{0..2}.log`，
冻结测试为 `windows-omni-target-head-corrected-tests.exe`。

成本边界必须包含实际两路 target 的 F16→F32 cast：原路径只做视图切片再 cast，并
不拼接 logits。初版探针多算了 baseline 的输出 cat，日志 scope 为
head_and_target_packing_eager，不采用其计时判断收益；其 bits 检查仍有效。修正版
scope=head_target_packing_and_scoring_casts_eager，每个真实步骤 0/15/31 各 3 组
ABBA、每槽 10 次、前后 CUDA 同步。

| 场景/进程 | 完整 head+消费 μs | target-only+消费 μs |
| --- | ---: | ---: |
| 默认 0 | 69.159 | 74.008 |
| 默认 1 | 71.668 | 73.816 |
| 默认 2 | 69.069 | 73.727 |
| 参考 0 | 148.555 | 85.013 |
| 参考 1 | 152.106 | 81.086 |
| 参考 2 | 145.771 | 78.376 |

默认短句局部退化，参考前缀局部改善，但尚未证明当前 graph 整链收益。若落地还需
目标布局分支、改变输出 shape、GPU GEMM 形状切换与回退验收；暂不为此增加生产条件
分支，不是因未过 10% 而拒绝。探针保留为后续归因工具，未改变模型生产数学。
下一候选是在保持全部 head 数学不变的前提下，将 greedy scoring 接入已有 forward 图，
避免交付完整 logits；整数 tokens 和 F32 confidence 必须分开保留 dtype，不能借
浮点打包绕过 recorder 的输出类型限制。该候选尚在独立测试阶段。

## 联合 scoring 图原型：局部改善，但暂不扩大生产接口

单独原型把原始完整 head、两路 F32 cast、greedy scoring 录为一个图，返回 I64 tokens
和 F32 confidence 两个输出，未用浮点编码整数。与现有 forward-only 图+外部 scoring
比较，所有真实 32 步 tokens/confidence bits 一致；默认/参考各 3 进程。每个步骤
0/15/31 各 3 组 ABBA、每槽 3 调用，边界同步，录制/额外预热不计入局部 replay。

| 场景/进程 | 原边界 ms | 联合图 ms |
| --- | ---: | ---: |
| 默认 0 | 7.741980 | 7.640646 |
| 默认 1 | 7.606152 | 7.497224 |
| 默认 2 | 7.711411 | 7.611996 |
| 参考 0 | 21.622209 | 21.354704 |
| 参考 1 | 21.786237 | 21.564509 |
| 参考 2 | 21.595794 | 21.381754 |

约 1% 的边界改善，不直接外推整链；需要增加混合 dtype 多输出、greedy/随机分支、
debug logits 交付与逐步消费路径的维护。结合收益量级和接口成本，暂不接入生产，
不是按整链 10% 拒绝。多输出 recorder 的整数/浮点回放、输出数量不符时的 capture
收尾等测试通过，原型源码/摘要归档在本地 `windows/omni-joined-graph-source`，冻结
测试 `windows-omni-joined-graph-probe-tests.exe`，日志 `windows-omni-joined-{default,clone}-{0..2}.log`。
已撤回生产 recorder 的 Vec 输出扩展与联合 hook，保留先前已验收的单输出 forward 图。
仅新增 head 的 cfg(test) 诊断入口，普通测试不运行真实模型；当前原型不成为运行时开关。

## 固定 codes 的 decoder 对照

进一步定位不能仅凭 CUDA 利用率或原先未录图时的阶段占比。使用相同
`windows-omni-fixed-short-codes.json`（8×48）对照 raw decoder，排除不同生成轨迹和
静音后处理；双方输出均 46080 samples / 24 kHz。Rust 固定 release stage1_probe
三进程，各 1 预热+5 热轮，均值 21.24244 / 19.70048 / 19.60560 ms。原始目录/日志
为 `windows-omni-fixed-decoder-current-{0..2}`，模型和源码沿用已冻结 stage1 探针。

官方独立 adapter 经 xtask perf reference 调度，固定模型/source/codes，F32 decoder
实际类来自 transformers 5.3.0 的 higgs_audio_v2_tokenizer。首版 GPU tokens 已预上传，
三进程均值 5.94008 / 7.45788 / 6.54838 ms，不能将 native 的 CPU codes→PCM 与该
GPU tokens→PCM 边界混为一谈。新增 host-input adapter 将 tokens.to(cuda) 纳入每轮
计时；没有改旧 Job/源码摘要。首个 host-input 运行期间另有 CPU 单测，保留为诊断，
不将它作为干净串行相位对照；重新单独运行 `windows-omni-official-fixed-decoder-host-serial-run`。
此处是 codec 阶段调查，不是新一轮完整生产请求验收，不代替人工试听或后处理对齐。

head frozen release 普通测试 41 passed / 20 ignored，新增 ignored head 已分别显式
运行；当前源码 CUDA all-targets Clippy 拒绝警告通过。普通单测与正式 GPU测量分开。

匹配 host-input 端点后的干净串行官方三进程均值为 **7.83940 / 10.28178 / 7.86190 ms**，
mean-of-means 8.66103 ms；Rust 20.18284 ms，观察上约 2.33 倍。保留原 GPU-input
首版数据，但不把跨批差值直接称为 H2D 的独占成本（未锁频且短阶段有漂移）。输入
codes 相同，输出均为 46080 samples，官方热轮 raw PCM SHA
`87c5ff79257d25af8c0864e790805331b4386edb6e87cc749c3b78244f4db8c7`。Rust raw PCM SHA
`12206570fe323939eebec5d6efa2a405dafe5cee05ea90506d80eae767752add`，二者非逐位一致，
单样本 max_abs=0.000925138、MAE=0.000054368、cosine=0.999999868，仅描述当前
固定输入，不构成新质量容差或试听通过。接下来优先测试 decoder GPU 执行边界，
保持 Rust 自身原始 PCM 不变；不先修改 F32/TF32/卷积算法来追赶数字。

本轮生产 recorder 已恢复先前单输出实现；CUDA/default CPU all-targets Clippy 与
CUDA private rustdoc（含 examples）拒绝警告、fmt/diff 检查通过。冻结 head release
普通用例 41/20 与真实 head 测试分开记录，未把归档原型测试当作新生产路径验收。

## CUDA decoder 图可行性与阶段归因（未接入生产）

仅 cfg(test) 使用实际 48/92/222 帧 codes，录制 Stage1Model.decode_tensor，图外完成
PCM 读回。模型/权重在图之后销毁；每个形状回放原 codes、逐元素+1模1024的有效
扰动 codes、原 codes（同形状 A/B/A），三个进程全部 raw PCM bits 一致。每个 case
5 组 ABBA，每槽 5 次，CUDA 同步，边界为 device codes→CPU raw PCM。预热与计时均
在 h2d 参数 cache guard 内，故此 eager 列是诊断模式，不冒充生产 host-input 基线。

| 帧数/进程 | eager ms | replay ms | warmup+record+instantiate+sync ms |
| --- | ---: | ---: | ---: |
| 48 / 0 | 18.91927 | 17.48390 | 19.5298 |
| 48 / 1 | 19.20986 | 17.68763 | 20.0145 |
| 48 / 2 | 19.17745 | 17.72095 | 19.8487 |
| 92 / 0 | 34.82062 | 33.39518 | 36.5530 |
| 92 / 1 | 35.13844 | 33.78103 | 35.5330 |
| 92 / 2 | 34.68382 | 33.70569 | 35.4482 |
| 222 / 0 | 83.32542 | 81.19452 | 151.8836 |
| 222 / 1 | 82.72190 | 81.20666 | 142.1751 |
| 222 / 2 | 83.16612 | 81.95787 | 140.8542 |

原始输入来自既有 baseline codes dump 的记录 0/1/3（8×48/92/222），不生成新轨迹
来替换基线。数据为 `windows-omni-decoder-graph-codes`、`windows-omni-decoder-graph-{0..2}.log`，
汇总 `windows-omni-decoder-graph-summary.json`，冻结测试 `windows-omni-decoder-graph-tests.exe`。
总计三个进程×三个形状×三次 raw PCM 比较均一致；没有冒称跨形状 LRU 或生产取消已验收。

录制可行，但回放仍明显慢于官方固定 codes 的 decoder，主要差距不能归因成 host
launch。decoder 每段通常只调用一次，不能套用 stage0 32 步的摊销假设；若接入图，
需验证真实形状复用、内存上限、缓存淘汰、首调用及取消，不直接加入跨请求缓存。
表中 combined recording 包含 warmup，不能将整个值冒称为 capture 的额外开销。
下一步用同步算子边界查卷积/反卷积/残差块，计时不相加为 kernel 独占占比。

原型最初通过 path 再加载 recorder 源文件触发 Clippy duplicate_mod；已改为 cfg(test)
的 crate 内类型别名复用原 recorder，生产 API 和模块归属未变，不用 allow 压警告，也
没有在第二个生产消费者尚未验收时提取公共辅助 crate。新增归因测试将逐边界验证重建
raw PCM 与正常 decode_tensor bits 一致，可按需把代表性真实卷积输入保存到仓库外。

### 48 帧同步阶段归因与 CUDA 路由

新增归因测试对 trace、逐块、逐 residual 重建的 raw waveform 均与正常 decode_tensor
位级比较通过。无图、无 h2d cache guard 的三个独立进程，各边界 3 组×5 调用、前后
同步。以下仅列对应反卷积边界，不把与 parent block 的重叠计时相加为 kernel 百分比。

| 反卷积 | 进程0 ms | 进程1 ms | 进程2 ms |
| --- | ---: | ---: | ---: |
| block 0 | 2.99837 | 3.00068 | 3.01009 |
| block 1 | 2.51559 | 2.34919 | 2.36741 |
| block 2 | 2.05774 | 2.09519 | 1.96197 |
| block 3 | 0.54093 | 0.57607 | 0.54925 |
| block 4 | 0.64508 | 0.64643 | 0.64695 |

原始日志 `windows-omni-decoder-stages-48-{0..2}.log`，汇总
`windows-omni-decoder-deconv-stage-summary.json`，冻结测试
`windows-omni-decoder-profile-tests.exe`。未启用可选 activation export；不能声称已
归档该阶段的输入布局。算子边界主机同步改变排队行为，尚不是 Nsight kernel profile。

核对实际 Candle fork `2266099d`：cuda_backend/mod.rs 的 conv_transpose1d 只有在
kernel contiguous、dilation=1、padding=0、output_padding=0 时走 GEMM+Col2Im1D；
否则调用 ConvTranspose1D(params) 的直接实现，candle-kernels/src/conv.cu 明确为 naive
conv_transpose1d。OmniVoice decoder 配置 padding=stride.div_ceil(2)、output_padding=
stride%2，因 padding>0 不满足该快路径。图回放保持此 kernel，不能消掉它的计算成本。

因此下一候选应围绕这些反卷积，而不是先加入跨请求图缓存。改变 padding 再裁剪虽
数学等价，会把逐 input-channel/tap 的归约换成 GEMM/Col2Im 的分组归约，不能默认
F32 位级不变；必须先做真实输入及完整 PCM 误差/位级检查，不通过修改精度掩盖差距。
本轮没有改变卷积、精度或生产 decoder。当前 CUDA all-targets Clippy 已通过；最后
的工程门禁按最终测试源码补齐，ignored 真模型测试不混进默认测试通过计数。

源码进一步确认归约顺序：该 naive CUDA kernel 的 k_x/tap 是外层循环，input channel
是内层循环，F32 使用 float accumulator。后续保持输出的候选必须以实际 tap→channel
顺序为基准；不能从 Conv1d 的 lowering 或历史 CPU 路径推断这里的顺序。优先考虑
保留该 K 顺序的按 stride phase 打包/GEMM 实验，并先逐 primitive 位级对照；若
产生数值差异则仅记录，不直接替换生产。普通无模型 release 测试最终为 41 passed /
22 ignored；CUDA/默认 CPU Clippy、private rustdoc（含 examples）、fmt/diff 均通过。

## 保序 CUDA 反卷积：候选验收与保留

### 不保留的 GEMM 与 relayout 实验

按 tap→channel 打包的 phase GEMM 虽保持逻辑 K 顺序，cuBLAS 内部执行仍改变舍入：
48 帧首轮五层 bit 差数分别 187541/463479/891963/0/0，前三层 max_abs 分别
0.0001106262/0.0000457764/0.0000152588。最重三层局部约 3.001→0.652、
2.365→0.394、2.065→0.572 ms，但测试以非零位差失败，未接入生产，不放宽容差。
原始 `windows-omni-phase-deconv-48-0.log` 与冻结测试保留。随后只重排输入/权重、
继续调用同一个原 CUDA kernel，五层 15 个布局比较均 bit-exact，但含 packing 的
成本变慢；例如 block0 input/weight/both 为 4.162/4.519/4.527 ms，原边界约
3.075–3.344 ms。`windows-omni-deconv-layout-48-0.log` 因 GEMM 的保留门禁仍失败，
不能把其中布局 bits 通过误报成整个测试通过。这些路径均仅在 cfg(test) 原型中。

### 最终机制与边界

最终不换归约算法：保留原始连续 NCL 输入、连续权重、tap 外/channel 内的每次 F32
FMA，特化整数寻址、已知几何和循环展开，跳过必然无贡献的 tap。NVRTC 原型在
48/92/222 帧、各三个进程、五层真实输入全部 bits 一致；此后改为构建时 nvcc 编译
compute_80 PTX 并嵌入 crate，生产不在运行时编译。NVRTC 仍只用于 cfg(test) 探索。
生产 PTX 由 Candle 的 device/stream/module owner 加载，沿用当前动态 CRT 和 cudarc
0.19.10；没有新增原生依赖或公共 CUDA 辅助 crate。

选择范围为 CUDA F32 dense、batch=1、contiguous 输入/权重（允许逻辑起始 offset），
groups=1、dilation=1、kernel=2×stride、padding=ceil(stride/2)、output_padding=stride%2，
精确几何 `(Cin,Cout,stride)` 为 `(1024,512,8)/(512,256,5)/(256,128,4)/(128,64,2)/(64,32,3)`。
检查 i32 索引上界，设备 capability major≥8；其他设备、dtype、布局和几何回退原路径。
CPU/Metal 不运行新的 CUDA build step。源码/许可见 crate SOURCE.md 及 Candle 许可副本。
`cargo package --list` 已确认 build.rs、.cu、Rust helper 和许可文件都在源包列表内。

### 正确性与局部成本

构建时 PTX 的合成 GPU 门禁覆盖五个几何×长度 1/2/3/7/48/49，包含正负零、次正规数、
非零 storage offset，以及 dtype/非连续布局的回退；全部与原 kernel bits 一致。
实际模型在 48/92/222 帧各做原/扰动/原 codes，全量 raw PCM 一致。冻结测试为
`windows-omni-ordered-production-tests-v2.exe`，SHA
`22a14ec40c54752b5d36658ef3286dc7d9bee36ec75a4805027509e0cdf1bcdd`；新增 build.rs
改变了 Cargo test artifact 哈希，实际读取新产物，未误用同目录的旧测试 exe。

生产 PTX 的三进程同步边界复验如下，原边界见上一节；不把局部数字相加为整链收益。

| 反卷积 | 进程0 ms | 进程1 ms | 进程2 ms |
| --- | ---: | ---: | ---: |
| block0 | 0.428107 | 0.487940 | 0.444833 |
| block1 | 0.526640 | 0.595413 | 0.543373 |
| block2 | 0.533847 | 0.604160 | 0.551793 |
| block3 | 0.291927 | 0.321907 | 0.295773 |
| block4 | 0.235440 | 0.267880 | 0.246000 |

完整生产长文本诊断 codes dump 仍同 SHA
`5fbfa8e88e86c180210f6d2234cf2f0fcf934382aac8bb2517cb4401253b19e4`，16.4 s 最终 PCM
同 SHA `ac88a7aa02dc6d2e5ba0ac283af9a7f5405237fb55ca14aa4dd6af783449f815`；取消后
复用通过。该次 dump 运行不计作吞吐。生产 receipt 为
`windows-omni-ordered-deconv-build.json`，新 exe SHA
`b86c10d396946c8600689d8561b922d878b2aa60b31488995d94f171ab5417b4`，增量 baseline
为已含 stage0 图的 `windows-omni-forward-graph-build.json`（SHA b76b62c7…）。

### ordered decoder 增量完整请求对照

ABBAABBA、8 独立进程、各 1 预热+5 热轮；两臂 stage0 图均启用，区别只有 decoder。
两臂均 500 ms 遥测；生产请求同参考音色/同 seed/同配置、输入文件与 prompt cache 摘要。
本轮新结果在 `D:/workspace/talechime-benchmarks/windows-omni-ordered-20261024`，模型和
旧结果仍在 ~/.talechime；C 盘空间较低，只把新结果写 D 盘，未删除旧证据或模型。

| 槽 | arm | 合成秒 |
| ---: | --- | ---: |
| 0 | base | 0.81711550 |
| 1 | ordered | 0.77361006 |
| 2 | ordered | 0.79315720 |
| 3 | base | 0.83494616 |
| 4 | base | 0.77847468 |
| 5 | ordered | 0.78579952 |
| 6 | ordered | 0.77919464 |
| 7 | base | 0.81339978 |

mean-of-means **0.810984030→0.782940355 s**，观察少 **3.46%**；base 范围
0.778475–0.834946、ordered 0.773610–0.793157 s，范围重叠，整链提速未证明。
不能用更早时段的 0.735723 s 替代本组 baseline，或把 primitive 百分比相加解释本组。
所有 48 轮最终 PCM 均为 `01043855ad3be24ee83dcb35fb4c13f8ab9a10f2814980b6f639ce61091bda1b`，
音频 2.0 s、全部 success 且 cleanup_verified；首 PCM 仍为整段交付。

保留依据为已复验的明显 primitive 改善、完整输出/取消门禁和有限的五个几何，没有
为整链百分比新设例外开关。尚不把该模型视为所有模式达到官方目标。原始 `clone-{slot}-{arm}-run`、
`clone-summary.json`、`clone-telemetry-summary.json` 保留逐槽与每点来源，原 process_index
不改写。whole-device 样本覆盖启动至收尾：时钟 630–2932 MHz、功耗 30.88–140.62 W、
温度 43–63°C、整卡显存峰值 2538602496 bytes；并非热轮独占时钟或本模型独占显存。
桌面 GPU 程序驻留，未锁频/改电源策略。不得仅凭这些全程样本认定漂移由某个原因造成。

官方同 Job 在新目录 `official-clone-repeat-run` 复测 3×(1+5)，进程均值
0.805669/0.694585/0.775964 s，mean-of-means 0.758739287 s，音频 2.12 s、同官方
PCM 摘要 `3c93ee946f90f74eedd59771bae7586fbadc1defe15ff0faebd59d708c0ecaa0`。当前
Rust 均值仍观察上慢约 3.19%，但两组范围重叠且跨框架非交错；本轮不能严格确认
排名。保留该波动及前轮差距，不挑选对自己有利的一次官方值宣称达标。

### 原生组合与发布边界检查

实际生产 probe 的 PE import 表与旧 probe 一致，均只有既有 cuBLAS/curand 和系统/
动态 CRT DLL；无新增 NVRTC import。`pe-imports.jsonl` 保留结果。CUDA 构建用已有
nvcc，CPU/Metal feature 不调用 CUDA 编译；未更改 Candle/cudarc/CRT 版本或库加载模式。
源码包列表包含新增 .cu/build.rs/许可，未发布包、PR 或 issue 评论。

最终 CUDA release 普通库测试 42 passed / 26 ignored；边界、实际 raw PCM、取消复用
各自显式运行，未把 ignored 算作普通通过。当前 CUDA/default CPU all-targets Clippy、
CUDA private rustdoc（含 examples）拒绝警告通过，生产 release probe 构建通过；格式
及空白检查 `cargo fmt --all -- --check`、`git diff --check` 均通过。没有为本次候选添加持久历史/无界缓存或修改实际播放完成语义。

### 后处理 CPU 成本初筛：暂不改静音检测

检查当前生产 `decode_final_tensor` 后，用既有真实 decoder 输出（46080 个 F32 样本，
24 kHz）直接调用当前源码的静音移除、clone RMS 恢复和 fade/pad。该局部 CPU 初筛
不加载模型、不并行 GPU 测量；使用外部 std-only Rust 包装器 `rustc --edition 2024 -O`
直接引用 `postprocess.rs`，避免为单个筛选重新链接模型，不能冒充 xtask 完整请求验收。
包装器及原始日志位于上述 D 盘结果目录 `postprocess_probe.rs`、
`postprocess-baseline-{0,1,2}.log`。每进程 6 组、每组 100 调用；首组视为预热，
后 5 组约 **1.10–1.21 ms/调用**，输出长度均 41520。RMS 参数为 0.0828476；
这是固定原始 decoder 样本的成本实验，不是当前参考音色完整请求的阶段归因。

静音检测确实重复扫描重叠窗口，但当前测得约 1 ms，不能解释整链约数十 ms 的
跨框架均值差距及更大的进程波动。暂不为此引入前缀和或滚动归约；这些方案还需
另证阈值边界行为与长音频收益。没有修改生产后处理，也没有把局部测量上升为所有
音频长度的结论。继续优先检查模型阶段与不同负载，而非盲目优化可见的重复循环。

身份 SHA256：原始 WAV `f10f3e1475e0983ad1335acd948052cc12a7d9c67ddacc79ea531a4f9ccf7e45`；
当前 postprocess 源码 `13decd5da8862ee306c6ac56dae1e8bcd3a5d28ec7b9cf9faa9216900f011ac8`；
包装器 `0c1df6ad9ec51d51794e8aa3cdd54e6c5d868228fa0fb5dfae7cb7c553e07d3a`；
exe `827a390442c07b0fa9c0db1514aa4968086d87b83f2ed4c6a35ce538af46dab6`。

### 保序 decoder 的短/长完整请求补测

继续使用冻结 production build `b86c10d…`，通过 xtask 串行执行短/长文本，各 3 个
独立进程、每进程 1 预热+5 热轮。Job 继承此前 graph 请求，短文本 request JSON
序列化核对未变；更新 build/source 及 decoder_path 身份。目录为上述 D 盘根下
`short-ordered-current-run`、`long-ordered-current-run`；逐轮数据和摘要保留于
`text-ordered-current-summary.json`。本批为新绝对基线，不把历史批次当作同步 A/B。

| 负载 | 各进程热轮均值 s | 各进程首 PCM 均值 s | 音频时长 s |
| --- | --- | --- | --- |
| short | 0.324294 / 0.321320 / 0.310682 | 0.324237 / 0.321275 / 0.310638 | 1.63 |
| long | 2.192795 / 2.166720 / 2.165845 | 0.564583 / 0.499236 / 0.480270 | 16.4 |

短/长各 18 轮的 PCM 摘要全部与之前 graph 基线一致，分别为 `481dcfed…`、
`ac88a7aa…`；6 进程全部 success、cleanup_verified。这里只证明当前实测请求的
输出保持与绝对成本，不能凭 PCM 一致证明没有性能退化。尤其长文本比早期 graph
批次更慢，因此补测相同时段原 decoder，不能用“局部更快”跳过整链差异调查。

紧接本批运行原 decoder 冻结 build `b76b62c…`，相同 long 请求、3×(1+5)，
目录 `long-original-current-run`，逐进程摘要 `long-original-current-summary.json`。
其均值为 **2.162202 / 2.262639 / 2.239236 s**；首 PCM 为
0.495299 / 0.570264 / 0.484602 s。18 轮 PCM 同样保持 `ac88a7aa…`，全部成功且清理
通过。新/旧 mean-of-means 约 2.175120 / 2.221359 s，进程范围重叠；本批按版本
分组串行而非交错 ABBA，不能据此证明整链提速，也不能证明绝无退化。旧版本在当前
时段也明显高于历史 1.91 s 批次，历史差额不是新增 decoder 退化的充分证据。
本轮补齐不同负载的完整输出保持和当前绝对基线，仍保留局部优化的原验收边界。
