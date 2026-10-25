# OmniVoice CUDA：官方对照与候选边界

Windows RTX 5070、Candle 0.11.0 fork `2266099d`、OmniVoice revision `c5fdb5cc` 的
2026-10-24 调查，见[完整记录](../../../docs/records/omnivoice/omnivoice-windows-cuda-2026-10-24.md)。
这是进行中的调查，不把合成张量、短句或另一台 Mac 的结论外推为整模型验收。

后续[Windows合并验收](../../../docs/records/omnivoice/omnivoice-windows-merged-acceptance-2026-10-25.md)
补齐短句、长文和参考音色的54轮输出回退及图/decoder门禁；Clippy初次磁盘失败与
空间恢复后的Clippy/rustdoc通过分别保留，不能用推理通过或历史组合替代缺失门禁。

## 从共同源码重建设备路径

Mac 合入的优化有明确设备边界：Metal 融合 attention 和 Accelerate CPU GQA 不代表
CUDA 自动启用它们。逐项核对 `is_metal/is_cpu/is_cuda`、dtype、contiguous 和真实
消费者；本次 CUDA 仍走手写 attention 与 repeat-KV。默认主干 F16、decoder F32。

官方使用固定仓库 `08be0b4c`、torch 2.8.0+cu128、transformers 5.3.0，实际路由 SDPA。
`from_pretrained(dtype=...)` 不足以证明全模型同精度：官方独立加载 audio_tokenizer，
本次确认仍为 F32。分别记录主干/codec 的 dtype、设备、源码导入路径和 attention。

## 比较负载而不只比较 RTF

相同短文本和 seed 下，Rust 音频 1.63 s，官方 2.12 s；Rust 绝对耗时更短但 RTF
更大。默认时长估计、seed 派生、生成轨迹和静音后处理要分别核对，不能把语言运行时
当作原因。产品比较与相同张量的算子比较分开；不为获胜而修改生产时长或质量。

官方推理可用 `xtask perf reference` 的显式适配器运行，复用 native 进程清理、超时、
有界日志与音频资产审计。prepared、每轮和 terminal 缺项不能计成功；固定输入/源码
摘要，失败用新目录。executor 已负责写 capture，不要由上层重复发布同名文件。

## confidence 候选（未完成整链验收）

greedy 分支已取得 argmax，可 gather 对应 log_probs，省去第二次词表 max。
随机 class_temperature>0 仍使用分布最大 confidence，不能替换为抽中 token 的概率。
GPU 位级夹具覆盖实际 1025 及 1024/4097 词表、重复最大值、mask、CFG 与多 batch。

合成 F32 边界在三进程 ABBA 小组中观察到局部改善；短句四槽 A/B 的 PCM 一致，
整链均值几乎不变且有漂移。长文本诊断另验证完整 codes/PCM 一致，dump 中含独立
短参考预热，不要误当作第一段正文；必须按实际记录大小解析和标明边界。诊断变量
在正式吞吐前清除，dump 运行不用于计时结论。这里的局部输入不是实际 log_probs，
正式保留前还需真实输入或更强消费者成本证据。不得把合成局部百分比称为模型提速。

后续真实捕获补齐了证据：短句实际张量为 `[1,8,48,1025]`，三步、三进程局部改善
18.8%–40.5%，token/confidence bits 一致；结合完整 codes/PCM 回归及改动成本，
CUDA gather 已保留，整链提速仍未证明。真实 T=48 不能从裁剪后音频 1.63 s 推导；
不要用最终 WAV 秒数倒推模型 kernel shape。

## 分段和派生 seed 是实际负载的一部分

Rust 宿主的长文本被拆成三个 32 步请求，官方默认一次处理全文；先比较默认产品行为，
再按 Rust 的 immutable UTF-8 ranges 与文本摘要冻结官方同分段请求。对齐后仍有差距，
但不能把原先的三次模型执行与一次模型执行直接归因给算子或语言。

Pinned(42) 是宿主策略，实际模型 seed 来自 call/attempt 派生，不一定为 42。
从当前实现导出并冻结每段实际 seed；不要把 DefaultHasher 某次结果当作跨编译器的
通用常量。相同派生 seed 仍不保证 Torch/Candle 随机序列或波形一致。

分别记录目标帧、raw decode 样本和后处理样本。本次短句双方同为 48 帧/46080 raw
样本，最终秒数仍不同。静音裁剪和边缘 padding 会改变 RTF 分母；不要因此修改生产
裁剪或强制时长来“达标”。算子对照重放同一真实输入，产品对照保留各自完整后处理。

## CUDA GQA：布局、回退和消费者边界

CUDA F16 可将 Q 的 GQA group 折入 query 行，与原始 KV 相乘，避免 repeat-KV。
不要直接套用 CPU Accelerate 的转置视图条件：CUDA GEMM 的 K 侧需按实际 kernel
要求 contiguous，V 也要计入布局转换。真实捕获回放必须重建原始 V transpose view
并核对 strides，不能让两个实现消费不同的预处理输入。成本边界包含恢复输出 context
的 transpose/contiguous/reshape；注明是否包含 o_proj。

本次生产 helper 限定 head_dim=128、Q/KV head ratio=2、seq=8..512 和 F16，检查
shape/布局，其他输入回退。实际首/末段 seq=117/258，三个步骤×三个层、各三进程
回放局部耗时减少约 34%/28%，输出 bit-exact；完整生产分段 codes/PCM 另行一致。
合成边界、mask、dtype/seq 回退以及取消后复用也要覆盖，不能靠九个实际输入证明
所有 shape。独立捕获末段时的 seed 与生产 call2 不同，必须披露并保留整链回归。

增量 baseline 已含 confidence gather；ABBAABBA 的 8 个进程（各 1+5）长文本耗时
2.67242→2.47995 s，减少 7.2%，不叠加局部百分比。首包分布重叠，单独保持未证明。
保留这类可信收益不需要整链 10%，但仍需给出维护范围、回退、完整输出与环境证据。
连续 GPU 采样有自身开销，应两臂同时开启，并与未采样组分开报告；whole_device
功耗/时钟/显存不能冒称模型独占资源。官方差距仍需继续缩小，不能据此宣布后端达标。

同顺序的第二组开启两臂 GPU 采样，耗时 2.65658→2.49899 s（减少 5.93%），PCM 仍一致。
两组收益分别报告为 7.2% 和 5.9%，不择优丢弃复验。500 ms 样本含加载和收尾，不能
将全程均值称作热轮时钟；WDDM 进程显存缺失时保留 unavailable，不填整卡值代替。

## 取消测试必须证明进入了目标区域

按第 N 次 cancellation probe 调用取消，只能定位检查次数，前端多一次检查就可能
在模型开始前结束。本次首版图测试的 PCM 对照通过，但录图计数为 4 而非 5，揭示
取消请求根本没有执行图。改为观察该请求的第 0 步已录图，再在步边界取消，并断言
取消请求及恢复请求都真实录图。不要把“收到取消错误+下一请求成功”直接等同于
“图执行后可恢复”。测试 hook 限于 cfg(test)，避免为验证加入生产运行时开关。

## 固定去噪 forward 图先按请求持有

本次 OmniVoice 无跨步 KV 的 32 步 forward 可直接复用 Qwen 私有 recorder 的配平
分配、输出复制预热和 RAII 原则。先测试全量 logits（3 种 seq、各 3 进程、32 步
全部位级一致），再接生产。缓存边界不必从模型级 LRU 开始：第 0 步真实 eager 同时
预热，录制不执行，因此交付 eager 结果，第 1–31 步回放；将录制成本计入每个请求。
测试中另算的额外 warmup 成本不能直接当作生产录制开销。

图借用模型以约束权重生命周期，持有固定输入/输出/流；每次 launch 全量覆盖 ids、
audio mask、attention mask、cached text embeddings，scratch logits 必须在下一次
launch 前由同流消费者处理。按请求销毁避免跨请求共享状态与无界形状图缓存。当前
Candle h2d 参数缓存按线程保活；模型及图需在推理 owner 上使用并在线程退出前销毁。

真实 A/B/A 请求 PCM、录图后取消再请求、失败 capture 清理、错误形状拒绝后再 replay
和非连续输出复制分别验证，不用单个 bits 检查替代生命周期门禁。CPU/Metal/量化/
未覆盖形状与同步诊断保持原路径；实验结果仍按设备和版本限定。

OmniVoice 生产全请求 ABBAABBA 证据补齐：各 1+5、两臂遥测，增量图回放耗时
2.51528→1.91063 s（-24.04%），首段 PCM 0.74077→0.46994 s（-36.56%），进程均值
范围不重叠；完整 codes 和 48 轮 PCM 一致。该 24% 的 baseline 已包含 GQA/confidence，
不能与前述局部或 GQA 百分比相加，也不能改称 main 的直接收益。只保留已验证的
CUDA F16 dense 范围；官方质量与其他模式仍独立验证。

官方同分段新复测为 2.10211 s，当前 Rust 1.91063 s，观察上少 9.11%。必须附限定：
Rust 四进程与官方另三进程、非交错跨框架对照；相同帧数不保证同随机轨迹，后处理
和音频时长仍不同。单案例达到/超过官方是扩大负载验证的起点，不等于整个后端验收。

## 带参考前缀的收益必须单测

默认短句图回放正式 ABBAABBA 为 0.63316→0.30044 s（-52.55%），相同短句加 129 帧
参考后为 0.94756→0.73572 s（-22.36%），两组完整 codes/PCM 各自一致。参考前缀会
增加每步实际 forward 计算量，不能只用默认短句的 dispatch 收益预测 clone 模式。
官方 clone 0.70629 s，当前仍观察慢 4.17%，虽然默认音色案例已快过官方；保留剩余
差距再定位，不把一个案例获胜改称整个后端达标。

先独立准备生产参考，再冻结 reference.wav、voice.json 和生成的 prompt.json，两臂
用同一缓存。官方显式转写、预处理并预建 prompt，冻结源文件与新适配器；双方 `[8,129]`
参考与 48 目标帧对齐，仍不要求跨框架 codes 相同。不要把 Rust adapter load（此例
0.297 s）与官方真实模型 load（1.885 s）直接排名：Rust 懒加载发生在完整 reference
warmup（2.224 s）中，不能冒称纯参考编码。正式热轮和首次准备单列。

修改官方适配器前保存旧源码及内容 SHA，并创建新的恢复 Job 指向冻结副本；不要改旧
Job 摘要使其继续“通过”。新增 clone 功能还应以同 seed 实测无参考默认模式与旧适配器
完整 PCM 一致；语法检查不足以证明默认模式兼容。本轮实际复验通过。

Pinned seed 的派生对每个案例都要核对，不能只给长文本修正，短句却继续用字面 42。
本次官方短句从旧 seed 的 2.12 s 音频变为派生 seed 的 1.69 s，说明后处理分母也会
受轨迹影响；当前帧数相同仍需分别报告绝对耗时、音频时长和 RTF。

## head 裁剪先检查真实消费者

先投影完整序列再切目标，与先打包目标 hidden 再投影，只在数学上等价；GEMM 行数
改变仍需逐位测。本次默认 S=63、参考 S=212、T=48，各 3 进程×32 步消费 logits
一致，但不能据此推广任意 shape。计时要匹配实际原路径：若消费者仅切片再 F32 cast，
不能额外给 baseline 加 cat。修正后默认 head 边界略慢（约 69–72→74 μs），参考约
146–152→78–85 μs；因此先不加入生产的 shape/布局分支，转查 forward 图输出与
scoring 的边界。这是维护成本和局部证据的判断，不是恢复整链 10% 门槛。

## 重新归因录图后的剩余差距

forward 图已启用后，不要继续用 eager 同步 profile 的旧占比选择候选。OmniVoice
追加 greedy scoring 的多输出图保持全部 tokens/confidence bits，但边界仅约 1%
改善，需新增多输出、随机分支和 debug 消费接口；本轮先撤回运行时扩展，归档原型，
再对照同 codes 的 decoder。小收益可保留，但仍应根据真实接口成本做判断。

固定 codes/帧数能把 decoder 与随机生成/静音裁剪分开。先问 tokens 在 CPU 还是 GPU：
Rust 公共 raw decode 接口包含 CPU codes 校验/H2D/PCM 读回，官方若预上传 tokens，
不能直接当作同边界对照；另测 host→raw PCM 并保留前版结果。记录真正的 decoder
类与 dtype，F32 并不自动证明双方算法/TF32 设置相同，不先用精度设置解释差距。
局部官方 adapter 也经 xtask 约束进程和审计，不伪装为生产 generation；编码与后处理
另列，阶段提速不能加到整链百分比上。

OmniVoice fixed 8×48 codes 的匹配 host-input raw decode：Rust 三进程均值约 20.18 ms，
官方约 8.66 ms，均输出 46080 samples。它比联合 scoring 图的约 1% 局部改善更值得
继续归因；仍不能把这段比例外推整模型。跨框架 raw PCM 非逐位一致（该样本
max_abs≈0.000925、cosine≈0.999999868），只记观察值，不据此放宽 native 输出门禁。
后续 graph/复制候选仍须保持 Rust PCM bits、重复调用和取消后的资源归属。

## decoder 图可行不等于该建跨请求缓存

OmniVoice F32 CUDA decoder 在实际 48/92/222 帧、各 3 进程、原/扰动/原 codes 中
raw PCM 位级一致；48 帧 replay 仍约 17.5–17.7 ms，诊断 eager 约 18.9–19.2 ms，
明显未消除固定 codes 下的官方差距。算子图不改变图内的慢卷积，不能把“图能录”
当作 dispatcher 是主因。先重新归因图内计算，再判断缓存的接口与内存成本。

每段只调用一次的 decoder 与 32 步去噪不同：若需要跨请求摊销，必须实测形状命中/
淘汰、owner 生命周期、首调用和内存上限。warmup+capture 的合计时间不是 capture
独占开销；h2d cache guard 下的 eager 也不是默认 production eager。保留这些测量边界。
同形状 A/B/A 不代替跨形状缓存 A/B/A。消费者测试可在 cfg(test) 内借用私有 recorder
类型，不用 path 将同文件编译成两个模块（Clippy duplicate_mod），也不提前提取公共 crate。

## 反卷积快路径的几何条件

Candle 0.11 fork 2266099d 的 CUDA conv_transpose1d 需 contiguous weight、dilation=1、
padding=0、output_padding=0 才走 GEMM+Col2Im；OmniVoice 的 padding>0 使其走
直接 naive kernel。48 帧前三个 deconv 同步边界各约 3.0/2.35–2.52/1.96–2.10 ms，
三进程重建 PCM 一致；回放图不会换掉该慢 kernel。先检查实际分支谓词，再解释
“CUDA 已启用却仍慢”。这些是固定 fork 的事实，不泛化到其他 Candle/cuDNN 版本。

用无 padding 反卷积再 crop 可以换算法，但也可能改变 F32 归约顺序：原来的
channel/tap 累加与 GEMM 后 Col2Im 合并并不保证相同舍入。数学等价仍先实测 bits，
不能自动通过更宽容差。归因时 parent block、residual、primitive 的计时有重叠，
不要相加伪装独占占比；对 trace/逐层重建做 raw PCM 对照以确认测的是生产算术。

实际 conv_transpose1d naive kernel 是 tap 外层、input-channel 内层的 float 累加。
保序 GEMM 原型的 K packing 必须遵守该顺序；普通 Conv1d 的 channel/tap 布局不能
直接照搬。先读固定 revision 的 kernel，再设计 packing，并以每个 primitive 的
实际 bits 为准，不凭数学式宣布保持输出。

## GEMM 保逻辑 K 顺序仍可能变 bits；可优化整数控制而保持 FMA

后处理也先量绝对成本。Windows 固定真实 1.92 s 原始 PCM 上，当前静音检测、RMS
恢复和 fade/pad 合计约 1.10–1.21 ms；看到重叠窗口扫描并不足以优先重写归约。
该局部 std-only 初筛没有替代生产 xtask，也不能外推长音频；详见同日记录的
“后处理 CPU 成本初筛”。先判断它能否解释剩余差距，再承担阈值边界验证成本。

实际 CUDA deconv 的 phase GEMM 原型按 tap→channel 打包，前三个大通道层仍有非零
位差（max_abs≈1.11e-4），尽管更快，不接入生产；小通道层零位差不能证明所有层。
未主动开启 reduced precision，也不能把任何误差都归因成 TF32。只改输入/权重布局
继续跑原 kernel 可保持 bits，但含物化成本后反而更慢，不能将 relayout 当成免费。

最终用已知五个几何特化整数寻址/循环，保留原 NCL、连续权重、tap→channel 顺序及
每次 __fmaf_rn；模型原始路径与新路径在 48/92/222 帧、合成边界/偏移/次正规数中
逐位一致。最重 deconv 从约 3.0 ms 到 0.43–0.49 ms，完整 codes/最终 PCM 和取消复用
通过。相比另起归约算法，先检查能否消除动态除法、64 位索引与无贡献循环，同时
仍须实测每一位，不能仅靠“相同算式”宣布相同。

NVRTC 可用于私有原型，但生产改为构建时 nvcc 编译并嵌入 PTX：CPU/Metal build 不
调用 CUDA 编译，运行时不新增编译器要求；限定 compute capability≥8，未验证类型/
布局/几何回退。检查 source package 的 .cu/build.rs/许可文件和最终 PE import 表，
不能把开发机安装了 Toolkit 等同于发行用户有 NVRTC。Candle 版本/CRT/加载模式保持
统一，派生 kernel 保留固定上游来源与许可。新增 build.rs 会改变 Cargo artifact hash；
用 Cargo 实际 artifact 路径和 SHA 冻结测试 exe，不能硬编码旧哈希后误复制旧 binary。

完整请求 ABBAABBA 此轮 0.81098→0.78294 s（观察 -3.46%），进程范围重叠，故只
确认强局部收益，整链改善未证明；按当前规则结合实现成本保留，不强制整体过 10%。
新的官方均值也波动，不用旧时段更低或更高的值替换当轮 baseline。现有输出门禁、
公式和 phase 表都见对应 Windows 记录；原始新结果在 D 盘仓库外，模型仍留 ~/.talechime。

补测中 long 新版约 2.166–2.193 s，看似比历史新版 1.91 s 慢；随后同条件旧版也达
2.162–2.263 s。因此跨时段变慢不能直接归因代码回归。先补同期旧产物，仍应说明
按版本分组测量的时间偏差；范围重叠时，不反过来宣称新版整链更快或绝无退化。
输出相同证明正确性，不能替代性能对照。短/长各 3×(1+5) 的证据见记录补测节。
