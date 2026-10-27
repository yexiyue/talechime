# CUDA / cuDNN / CUDA Graph

一次性生成也不能先验关闭图：Qwen VoiceDesign初筛decoder图关闭的首次调用似乎
更快，但独立新进程ABBAABBA复核范围重叠，候选撤回；CP关闭的重复调用也更慢。
先区分首次进程调用、每次重新加载和常驻模型热轮，记录图录制及释放边界；不能
凭单点推断录图成本或归因时间漂移。见[VoiceDesign图策略复核](../../../docs/records/qwen/qwen-voicedesign-windows-2026-10-25.md)。

## ONNX provider 的架构门禁

EP 可注册不代表目标 GPU 的算子可执行。先核对实际加载分发和 DLL 摘要，再检查
cuobjdump ELF/PTX；没有兼容 cubin 或 PTX 时，停止性能矩阵，先解决完整分发的
架构覆盖。PTX 命令输出可能只有“No PTX file found”，不能靠非空输出判断存在。
`error_on_failure()` 是注册失败策略，不是逐节点禁 CPU fallback；成功生成后仍需
单独 profiling 验证执行分区。保持 Runtime/provider 配套，不能只换单 DLL。
本机 [Nano ONNX 复核](../../../docs/records/moss/moss-nano-onnx-windows-2026-10-24.md)
记录 sm120 与缓存分发不匹配；不要外推其他设备，也不要把静态检查改名为实机推理。

探针成功后再核查主程序的feature入口、实际动态库与完整请求。Windows主CLI补显式
ort-dynamic后，实际加载暂存ORT1.28三库并完成静音播放、checkpoint到原文末尾；
这不代表CUDA依赖已完整打包或干净机器验收。仅ONNX构建的catalog也可能以空model
选择默认模型，不应为复用多模型探针参数而改模型身份。见
[主CLI动态部署验证](../../../docs/records/platform/windows-ort-cli-dynamic-2026-10-25.md)。

官方profile只能给Native提出候选：再在实际Runtime、相同I/O Binding及完整库组合
下复核。Nano Native的四轮原始PCM在有/无profile间一致，local仍是Node跨度主项；
实际图又保留符号batch与形状节点，才形成batch1 session特化的待验证候选，见
[Native profiling](../../../docs/records/moss/moss-nano-onnx-native-profile-2026-10-25.md)。
原始ONNX节点数、优化后执行节点数、Node跨度和整链计时分别记录，不互相替代。

batch1候选的后续实测已撤回：local执行节点减少31.56%、CPU节点归零，多个seed/
段落codes与原始PCM保持，但无profiling生产初筛两输入均未改善。ORT还融合出
Gemm，不能把“只固定形状”视作没有数值/执行路径影响，也不能把少节点当提速。
先检查图变化与完整输出，再用实际生产边界计时；无收益就移除特化和专用分支，
保留冻结原型。见[batch1撤回证据](../../../docs/records/moss/moss-nano-onnx-local-batch1-2026-10-25.md)。

## 算子候选

确认当前 Candle、cudarc、Toolkit、cuDNN 和设备架构的组合。编译、设备可创建、库能
加载与目标算子能执行是不同证据。检查 grouped conv 是否逐组分派、FIR 是否逐 tap
重复遍历、KV 是否反复复制、固定位置/mask 是否重建，再选择小实验。

IndexTTS 曾参考 Candle depthwise PR 做局部实现，保留其几何与 workload 选择条件；
不能只抄数学公式而丢掉 fallback、bias、padding 或选择门槛。FIR 可把 (B,C,L) 变为
(B*C,1,L) 复用同一个核，但 kernel 方向、相位、裁剪和大形状都要对照。

cuDNN auto 在小输入通过，不保证长音频容差。算法选择影响精度与性能，先做固定输入
算法探针，再明确局部选择；不因误差变化就断言由 TF32 导致。
NVIDIA_TF32_OVERRIDE=0 可用于单独的严格 F32 诊断，记录为不同运行配置，不能把其
验证结果冒充默认配置验收，或悄悄将全局精度修改混入性能优化。

## 图边界与所有权

优先固定形状、无主机读回的子图；固定循环可展开，跨帧动态 KV 需要独立设计。
先核对当前 API；下列经验在 Candle 0.11.0 / cudarc 0.19.10 下验证过，不承诺其他版本
接口或 allocator 行为相同。

首选目标是 **dispatch-bound 区域**：每帧数百个极小 kernel 的子图，其成本主要是
主机分派而非算力，一次 launch 即可整体消除。判断方法是在无插桩构建里按 API 边界
统计主机墙钟与每帧 kernel 数，而不是看 GPU 利用率——被 CPU 饿着的 kernel 队列
利用率同样低。已验证实例：Qwen CodePredictor 每帧约 34 ms 主机墙钟（2-token
prefill + 14 步），整帧成图后端到端完整合成 -38.0%、流式 -33.3%，输出逐字节一致。
反之，帧内大算子主导的区域成图只省少量 launch，不要先做。

录制前沿实际调用链检查以下前提，而非只看入口 shape：

| 检查 | 决策依据 |
| --- | --- |
| 主机控制流 | 循环次数、offset、分支是否对该特化固定；设备值驱动的 Rust 分支或读回不能直接留在图内 |
| 实际缓存类型 | 若设计依赖固定 KV 地址，确认分配没有静默回退其他实现；不能只看到 new_kv_caches 就假定是预分配。已实测：Qwen 的 `Concat` 回退（K/V 由 `cat` 增长、capture 内分配且被缓存保留）会让驱动在 `end_capture`/实例化就以 `CUDA_ERROR_INVALID_VALUE` 拒绝录制；预分配变体正常。结论是“图在 capture 之后读写的一切都必须预先分配”，不是“所有 Concat 图都不可能” |
| 主机与设备状态 | reset/current_len 等 Rust 更新只在录制时执行，replay 不会重新运行；用不变式判定：**每次回放是否重写图内稍后读取的每个字节**。全量重写则无需图内重置（Qwen CP 每帧重写全部 16 个 KV 位置，容量 17 只是上限）；跨帧卷积历史不被重写，必须改为图内更新的持久状态，否则只能终止请求 |
| 缓冲接口 | 查 recorder 的具体 dtype 分支和逻辑字节长度，包括 token 的 U32/I64；浮点图可用不代表整数输出可复用。整数输出需要 recorder 的 storage-slice 支持，不要用“转浮点再转回”绕过接口限制——那是数值-changing 绕路 |
| 特化身份 | shape/dtype/device 之外，捕获的分支、模型资源、缓存实现等变化是否要求重录；优先用 owner 隔离，避免庞大通用缓存键 |

固定步数循环可以展开成不同长度的固定节点序列，不要求每个节点具有相同 shape。
反过来，固定容量也不代表有效 KV 长度固定。若完整子图每次覆写其全部有效前缀，
可另行证明无需重放主机 reset；不要机械地把所有主机字段改为设备标量。

- 固定输入、输出、状态缓冲在图外分配。recorder 持有录制流，使复制与 launch 同流。
  连续 view 仍可有 storage offset，按 start_offset 和逻辑字节数复制并校验设备/布局。
- Candle h2d 参数缓存在 enable_cuda_graph_htod_cache guard 生效期间填充，预热也
  要在 guard 内。首次 capture 只录制不执行，返回真正算过的 warmup，不交付未执行输出。
- 预热须覆盖 recorder 添加的操作。若模型返回非连续 view，eager 调用可能从未执行
  输出物化，而 recorder 在 capture 内首次调用 `contiguous`，会缺少布局参数缓存。
  在 capture 外用真实 warmup 输出先跑相同的输出复制；只多跑一次模型前向未必有用。
  CosyVoice3 DiT 的转置输出曾触发该问题，共享 recorder 修复后真实权重回放逐位一致。
- 捕获的临时 alloc/free 应配对。Vox 将图内结果复制进图外输出、在 end_capture 前释放
  临时结果，解决重复 launch 问题；不先假设必须私有 mempool 或修改 Candle allocator。
  CUDA AUTO_FREE_ON_LAUNCH 处理下一 launch 前遗留分配，不是在上一 launch 末尾。
- 配平的对偶：**capture 内分配、capture 后仍被主机状态持有的张量同样破坏配平**。
  症状可能是实例化即 `CUDA_ERROR_INVALID_VALUE`（Qwen Concat KV），也可能是
  capture 成功但只能 launch 一次（Vox 早期输出缓冲）；失败 capture 后必须验证流
  已退出捕获态（RAII），否则后续录制静默损坏。
- 图持有 schedule、权重、流及其他被捕获地址的资源；steps 切换和 LRU 淘汰不能释放
  其他有效图仍引用的张量。销毁顺序必须保证这些地址直到图销毁仍有效。
- scratch 输出在下一 replay 会被覆盖，消费或复制后再回放。状态快照与恢复使用同一
  状态枚举；有状态 warmup、capture、replay 只推进一次，请求开始明确 reset。
- 取消和 host profiling 同步留在图外合适的边界，验证取消后可复用及公共延迟语义。
  有状态decoder分别检查执行前取消不消耗帧、执行后取消可能已推进状态、下一请求reset后
  连续输出与独立eager一致；不把底层图测试当成完整Backend取消验收。实机案例见
  [VoxCPM Windows](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)。
  profiling eager 与 graph 的计时不能混为一组。
- RAII 收尾失败 capture。launch 失败若无法证明状态未推进，终止当前请求，不能重算
  同帧并重复推进。上下文被异步非法访问破坏时，不承诺静默恢复 eager。
- 失败策略按状态性分档，不要一刀切：
  | 区域状态 | launch 失败 |
  | --- | --- |
  | 每帧自覆写、无跨帧持久状态（本帧 CP） | 警告 + 本 owner 永久停用图 + 本帧 eager 重算 |
  | 跨帧持久状态、无法证明未推进（AudioVAE decoder） | 终止当前请求；下一请求显式 reset 后 eager |
- `profiling`/插桩 feature 下停用图：逐步 span 会把 replay 压成一次 launch，
  trace 失真。正式基准用无插桩 release 构建，两组分开记录。

## 归属范围与共享边界

- 带请求状态的图归每次生成或每个 session，避免跨请求共享 KV/卷积状态。
  无内部历史状态、完全由输入历史决定的子图可以归模型并跨请求复用，但必须
  串行覆盖输入、launch、复制输出；图自身固定持有全部参数和常量，且在权重前
  销毁。不能把这种例外推广到有状态 AudioVAE。
  录制成本包含真实预热、主机录制和实例化，不能按“两遍 eager”估计；分别报告
  首次录制与多请求稳态，并限制缓存的形状数、形状大小和常量字节数。
- 共享机制需第二个真实消费者及各自的性能证据，不强制提取独立 crate。历史上
  Talechime 曾提取图辅助包，当前已退役，recorder 位于 Qwen/VoxCPM 各自私有模块；
  修改共同来源代码时检查另一份。模型缓存键、状态保活、取消与失败策略仍各自维护。
  移动或提取用消费者原有测试（最好有免权重真 GPU 测试）复验。
- 设备机制依赖 Candle 的 `cuda` feature 来获得 cudarc 的库加载模式，而不是
  自己挑 `static-linking`/`dynamic-loading`；并导出小 helper（如 `is_capturing`），
  在实际有跨模块使用需求时减少消费者对底层 driver 的耦合。

最小测试阶梯（Qwen CP 帧图按此顺序全部通过）：

1. CPU 单测：session 包装器与直接调用的 eager 输出逐位一致（无 GPU、无模型）；
2. 真模型真 GPU 测试：多组输入状态 replay 与 eager 严格相同；跑完若干帧后回放
   早期帧仍一致（验证全量重写不变式）；故意用错误宽度输入触发 launch 失败，
   断言图停用且后续帧仍与 eager 一致；
3. 端到端产物：codes、原始 PCM、WAV 与改动前 eager 基线逐字节一致；
4. 正式 ABBA（记录是否锁频、丢弃预热、报 mean-of-means 与区间；未经授权不改系统
   策略）。整链区间重叠时不宣称整链收益，可信局部证据独立判断，不设整链百分比门槛；
5. 提取共享机制后复跑消费者的免权重真 GPU 测试 + workspace 编译/Clippy/fmt，
   并记录哪些真模型测试因缺权重未复验。

另有：特化 A→B→A、LRU 淘汰、非零偏移 view、录制失败清理、取消、跨请求 reset
按适用性补充。忽略的真 GPU 测试要用 `--test-threads=1`：同一进程内多次
`Device::new_cuda` 共享同一条流，并发 `begin_capture` 会互相破坏（并行失败、串行
通过的现象要先怀疑流共享，而不是被测代码）。缺少硬件时不声称通过。

## 不把布局变化视作免费

分组 GQA、去 K transpose 的 contiguous、填充静态 KV 虽数学等价，可能更慢或改变
GEMM 形状和舍入。量化模型下少量隐藏态误差也可能改变 EOS；先做独立候选，不一次
叠加多种改动。Qwen 的固定 15 步 CodePredictor 图已实现并通过（见案例索引），
但这只证明该模型该输入；同一循环在别的模型上仍需重新核对缓存变体、状态与输出
dtype。

图接口与释放语义核对 [NVIDIA CUDA Graph Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)，
并读目标版本的 Candle/cudarc 源码，特别是分配、event 和 storage 生命周期。

## 复用已计算的下一步输入

自回归循环重建下一步输入前，沿当前步数据流核对是否已经算过相同embedding等组件。
优先复用当前步不可变Tensor，保持原dtype、布局与逐项加法顺序；不要为了省几行改为
stack+sum，也不要自然扩展成跨请求缓存。仅当分支保证所有通道有效，才能省掉通用
prompt路径的mask；直接用原通用实现作逐位oracle，再测完整codes/PCM和取消后复用。

Windows RTX5070 的[MOSS Nano案例](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)
通过同步阶段重演和codes/PCM对照，定位生成row重建约53.9ms；复用16个已算embedding，
按原顺序相加约5.6ms。1026组行覆盖每个码本全部1024条目并逐位匹配通用路径。
短句ABBA点估计改善5.08%但进程范围重叠；长句与状态验证通过后按局部收益保留，
不把阶段比例当成整链提速，也不机械应用整链10%。这是带设备与版本的实例，
下一模型仍先确认数据已经计算、mask成立条件与所有权，不能照搬成通用embedding捷径。

## CUDA 单 query 的因果 mask 省略

先证明所有 key 均可见，再考虑省略全false mask。qlen=1本身不够：需要核对绝对
query/key起点、缓存末端及滑动窗口。Nano实机候选只接受无窗口、query恰在末key位置，
其余保留原mask；位置运算检查下溢，不把prefill、未来key或窗口路径一并放开。
保持QK缩放、softmax和V matmul不变，使用原始masked实现作数值oracle；再验证完整
codes、PCM、自然EOS、owner取消及满队列后复用。测试参考切换只存在cfg(test)，
不为验收保留生产环境开关。

Windows RTX5070 / Candle0.11 fork2266099d验证540组F32/F16/BF16逐位一致，三seed
真模型codes/raw PCM一致；短句ABBAABBA 0.56125→0.52839s（约5.85%，进程范围不
重叠），48轮PCM相同，段落18轮PCM保持。该省略只启用CUDA，没有Mac性能结论。
详见[MOSS Windows记录](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)
的CUDA单query节。新模型仍需验证自身位置/窗口条件，不直接复用百分比或质量结论。


该方法随后用于MOSS Local/Realtime1.7B的共用函数回归：分别冻结moss-candle-cuda
探针并各跑3×(1+5)，36轮完整生产PCM与旧基线一致。当前2.799/1.412s仅作新绝对
基线；缺少这两模式的ABBA和完整状态门禁，不能转用Nano的收益、取消或EOS结论。
见[1.7B后续回归](../../../docs/records/moss/moss17-windows-production-2026-10-24.md)。
跨消费者验证按完整请求分别执行，primitive等价只是起点。


Qwen Base06/Base17随后将状态门禁落实到生产Backend owner：立即丢弃接收端、观察
实际满队列再取消，恢复请求逐位比较；入口音色拒绝与owner中真实frame-budget失败
分别验证，不能把前者称为推理失败。预算负向应消费部分PCM及错误后继续读到关闭，
排除错误后追加正常End，再用同owner恢复默认预算验证PCM。保持负向预算与性能
条件分开。Windows同CUDA/cuDNN组合两种Base各5次完整原始PCM一致，见
[Qwen状态门禁](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)。
这些是Backend生成边界，不能替代生产宿主后处理、跨固定音色或实际播放验收。


Qwen随后把同一owner门禁用于四种narration变体并实际重跑：共用测试驱动时仍显式
保持Base的1.5 penalty与CustomVoice默认1.05，不能用一个“共同默认”掩盖模式差异。
CustomVoice06/17增加A/B/A（uncle_fu/serena），B不同且返回A逐位一致；这不覆盖Base
跨参考或 style。四变体CUDA实测、源/二进制与38项资源身份见上述Qwen记录的
四变体状态矩阵节；计时证据继续使用原独立基线，不把正确性测试耗时当性能收益。


## 图形状覆盖与第二次访问录制

一次请求预热不等于每种实际形状已录制。先核对 cache miss、第二次 capture、后续 replay
的策略，把参考前缀、左上下文和生成块共同计入输入形状；不能只看新生成块长。
首热轮承担录制时保留该轮，单列冷启动/参考准备/预热成本，不通过删轮或临时加预热改善数字。
profiling feature 可能主动关闭图，先查源码；同路径 coarse diagnostics 仅作阶段归因，
正式比较仍冻结无诊断二进制与配置。

扩图范围须同时限定缓存项数、验证边界内外与输出持有期，并重新测内存和启动代价。
逐帧 codes 捕获会增加同步，只用于独立正确性门禁，不能混入正式性能计时。
Windows RTX5070 的 Qwen Base06/17 将 64 扩至 128，覆盖 86+20=106 帧参考首块，
ABBAABBA 完整耗时下降约 13.9%/13.0%，但 Job commit 增加约 1.2–1.4 GB。
此结果不构成其他 GPU 或长参考的可用性保证，见
[带版本的候选记录](../../../docs/records/qwen/qwen-decoder-reference-graph-windows-2026-10-24.md)。
资源采集状态 unavailable 与可用但空列表分开报告；整卡显存不能当逐进程分配器峰值。


Qwen 的 owner 门禁随后用于 VoxCPM Q8/BF16，发现“一秒预算失败”刺激实际仍返回
自然 EOS。不要先换长句或放宽负向断言：核对输入/输出采样率、latent 与生成 patch
的单位，再检查实际 PCM 时长。该模型四 latent 在输入端为 2560/16k 秒，在输出端
为 7680/48k 秒，误用 2560/48k 将预算放大三倍；round 还可能跨越硬上限。
修正预算须单列行为变化，不能记作性能提速；先测部分 PCM→Err→关闭→复用，再测
参考/默认/参考的完整请求恢复。两种精度分别验证，不共享数值结论，详见
[VoxCPM Windows 状态与预算证据](../../../docs/records/voxcpm/voxcpm-windows-owner-state-2026-10-24.md)。


## 单query GQA与复制顺序：VoxCPM后续实践

OmniVoice的多query ratio2经验在VoxCPM ratio8单query上先做CUDA数值门禁：改变
GEMM行数后F16/BF16也出现位差异，候选拒绝，不能从数学等价或别的模型成功直接移植。
另一条仅将K展开/转置合成一次复制的候选保持GEMM形状，192组布局/输出相同，真实
Model输入的三进程局部回放也保持context位模式。长度1原来已免转置复制，保留旧stride
路径；不要对没有复制收益的singleton强改dispatch。Q8局部点值改善17%–32%，但这不是
模型提速。BF16局部进程区间仍有重叠，不能统一宣称全部输入显著改善。
证据及后续门禁见
[VoxCPM单query调查](../../../docs/records/voxcpm/voxcpm-windows-attention-candidates-2026-10-25.md)。

后续按生产段界补81组context后才接运行时，Windows CUDA限定单query、16/2头、
宽128和KV长度2..512，其他形状回到旧路径。176个真实latent patch/PCM逐位门禁、
Backend取消/失败复用与192轮生产输出通过；两种精度段落完整耗时约改善5.26%/2.18%，
短句和首PCM仍inconclusive。保留有证据的复制收益，不从局部百分比推断所有模式
的加速，也不通过改变GEMM形状来扩大覆盖。见
[运行时证据](../../../docs/records/voxcpm/voxcpm-windows-key-packing-2026-10-25.md)。

后续官方对照实际重查profile是否改变路径：VoxCPM measure_modules=true会跳过CFM
及decoder图launch，缓存active仍可为true。先做默认/同步eager的raw PCM门禁，再把
eager阶段数值用于归因，不能标为默认图耗时。官方同步归因以CFM为主，而Native原始
F32 VAE约0.19–0.22秒、官方约0.09秒，给出下一候选；同二进制CFM开关ABBA证明
既有图贡献约33.08%，但不能把各跨框架阶段差额相加解释完整差距。完整条件见
[官方与Native阶段记录](../../../docs/records/voxcpm/voxcpm-windows-official-baseline-2026-10-25.md)。

上述热点随后落实为F32 AudioVAE图验证：先用原始F32权重做多帧/多请求、取消和重置，
断言active及逐位输出，再做完整模型latent/PCM、Backend与组合回归。F16既有通过不能
替代F32；共用测试辅助函数后也重跑旧F16门禁。Windows BF16默认启用后，短句/段落
分别改善约21.33%/26.68%，96轮PCM保持，其他平台/原始主dtype没有扩大默认范围。

默认策略改变后重查环境开关的依赖链：VOXCPM_CUDA_GRAPH=0也会使decoder工厂关闭。
旧BF16 decoder-off时期的CFM单变量实验，不能换新默认后继续沿用同一归因；要隔离
CFM须固定两臂decoder=0。源码、各层门禁与新旧二进制ABBA见
[F32 decoder图资格记录](../../../docs/records/voxcpm/voxcpm-windows-f32-decoder-graph-2026-10-25.md)。

RoPE先查重复构造的范围，再决定缓存粒度。MOSS同一forward的每层Q/K用相同
length/offset/dim/theta/dtype/device，可仅共用一次表，保留原来的矩阵形状；与扩展
整张预计算表、改变GEMM形状的早期失败原型不同。检查实际load的rotary开关，不能
凭“local transformer”或旧注释推定无位置编码，Realtime深度层也有RoPE。
300组CUDA逐位oracle、三个真实消费者codes/PCM和owner状态门禁后，Realtime正式
整链耗时约−24.59%；Local仅点估计−4.19%且范围重叠，两个首PCM仍inconclusive。
表归当前forward且无跨请求状态，不为此预建全模型持久缓存。见
[MOSS forward表复用](../../../docs/records/moss/moss-windows-forward-rope-reuse-2026-10-25.md)。

下一消费者Nano重新核对为F32/interleaved/base10000、12层global和1层local，不能
直接套1.7B的旋转布局或4层local成本。先180组逐位oracle和局部计时，再扩展原私有
表对象的明确interleaved入口；三seed真实codes/PCM/EOS通过，并回归旧三个消费者。
Nano短句/段落正式ABBA耗时约−19.45%/−14.54%，短句首PCM−21.02%，段落首PCM
仍inconclusive。这是保留原GEMM形状的forward共享，不复活此前宽表预计算失败方案。
见[第二消费者实证](../../../docs/records/moss/moss-nano-windows-forward-rope-2026-10-25.md)。
