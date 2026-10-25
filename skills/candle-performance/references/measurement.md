# 测量与准确性

连续接续的状态门禁应从同一初始完整PCM重放整链，每段使用实际上一段的PCM与
对应文本，按生产call索引派生seed；只重复第一段上下文无法验证接续演进。
在链中间取消并重试，再核对后续段和静态参考恢复。逐位重放通过不等于内容、
长链漂移或克隆质量通过。Windows Qwen Base双尺寸的三段实例见
[接续链证据](../../../docs/records/qwen/qwen-windows-continuation-chain-2026-10-25.md)。
该轮同进程最后一个模型加载OOM、退出后独立进程复查通过：两种结果应分别保留，
不能用新进程成功覆盖原组失败，也不能未经归因就认定泄漏。绝对基线优先独立
进程，跨模型回收另立诊断，不将两种目的混成同一通过声明。

## 定义可比实验

记录代码 revision 与 dirty diff、权重/输入摘要、设备/驱动/运行时、模型与 codec 的
实际 dtype、量化方式、线程环境和 Cargo features。生成任务还要记录实际生效的采样
策略、seed、上下文、帧上限和终止原因，不能只记录被实现覆盖前的请求参数。

设备在计时边界同步；加载、参考准备和导出是否计入必须明确。将 profiling 与无插桩
基准分开。host span 包含 API、等待和同步，不能直接称作 GPU kernel 耗时；阻塞的
to_vec/to_cpu 可能是在等此前整条队列，而不是传输本身慢。用已完成的固定输入测
孤立回传，再用同步阶段或设备 profiler 定位生产队列。

从其他平台复用探针时，核对其设备分支和同步实现：支持 Metal 的探针不一定支持 CUDA，
仅对 Metal 调用 synchronize 会漏掉 CUDA；组件预热返回 tensor 后也要清空异步队列。
记录 Rayon 和 ORT/BLAS 各自实际线程设置，不能把前端 `--threads` 当作所有计算线程数。
诊断 canary 的张量缓存也归对应 runtime/device；不能复用首个设备的全局张量、
却同步另一个设备来计时。归因探针沿用实际生产入口；原路径控制组应单独标明。
缓存存在不代表生产命中，沿 owner 请求链核对写入和清空。固定 seed 也不代表全部
随机流已固定；输出验收应明确控制实际随机输入，保持生产随机语义。
文件摘要只标识文件，转换来源仍需固定上游输入并验证变换。Windows Job 路径的
字面身份、资源联接的 canonical 边界、同名官方模型入口和 launcher 内存范围均须
核对；详见 [IndexTTS CUDA 经验](indextts-cuda.md) 及其带版本证据。
类名、model_type 还可能保留旧命名；MOSS 固定旧 Local 的 Delay 类名实际包含 Local
Transformer，须结合配置/权重键和实际源码目录判断，不能只凭名字切换到另一模型。
见 [MOSS 双基线](../../../docs/records/moss/moss17-windows-production-2026-10-24.md)。
本机生成的参考特征可用于实现内 A/B，但不作为独立官方数值 oracle；标明 fixture 来源。

核对探针实际保留几份模型。为阶段归因另载一份权重，可能改变低内存设备的工作集、
分页和缓存状态；吞吐基线应另测单实例，并记录峰值工作集与归因模式。移除探针的
额外副本是测量修正，不能计为生产推理优化。GPU/BF16配置不代表加载全程都是低精度；
检查constructor与最终dtype迁移的顺序。系统available commit近零时，旁路监控进程也
可能OOM；保存native sampler与Job高水位，先确认回收，再按实测预算检查启动条件，
不要凭栈顶cos/Linear就归因算子bug或连续重启加载。阈值按案例，不硬编码通用值。
当前xtask reference支持每Job可选minimum_available_commit_bytes（Windows）；
不足时保留commit-preflight.json且不启动参考模型。省略不强制检查，查询失败不能
当作余量足够；快照不预留内存，失败峰值也只是完整加载需求下界。
[VoxCPM官方加载复查](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)
记录了11.87GB Job峰值、系统仅余9.33MB提交余量和旁路OOM。
CosyVoice3 的 Windows 16 GB 主机暴露了
这项风险；不能从 Mac 24 GB 上可运行的探针推断其他机器没有内存干扰。

初筛可用短语料和少量进程。正式比较可按 ABBA 或重复 ABBA 交替独立进程，每进程
单列预热和多个热轮，避免固定先 A 后 B 的温度/频率漂移。先算每进程均值，再给
两组均值、范围和样本数；范围不是置信区间，不将不重叠直接视作统计显著性证明。
不同任务长度时并列输出长度和绝对耗时；RTF/每 token 成本并不能完全消除工作量差异。

Talechime Vox 的历史实验采用八进程、1 预热+5 热轮和当轮的 10% 收益门槛。
当前项目不设默认的整链改善百分比：局部显著收益可结合成本与回退情况保留，组合后
重新测量整链。历史轮次的门槛与样本预算不自动适用于后续任务；长期运行前先遵循
用户的预算与验收要求。

## 同机官方对照

与官方实现（PyTorch/原生）同机对比是独立的实验类型，先固定口径再谈快慢：

- **同口径四要素**：同语料、同参考音色（含转写）、同模式（流式对流式、非流式
  对非流式；不能成对的单列）、固定官方 revision 与运行时版本。计时边界逐项
  对齐（参考特征提取、导出写盘是否计入，两侧各在哪一侧）；对不齐的差异方向
  要写明对谁不利。跨框架输出音频长度不同时只用各自 RTF，不比绝对耗时。
- **设备差异 ≠ 语言/框架差异**：官方源码常硬编码 `cuda if available else cpu`
  （CosyVoice3 无 MPS 路径；IndexTTS2 原生支持 MPS——逐家核对，不要假设）。
  我们 Metal vs 官方 CPU 只能叫设备差异；GPU 对 GPU（Metal vs MPS）才支持
  实现层结论。
- **parity 与组件差距可以并存**：端到端区间重叠（设备优势掩盖实现劣势）与
  受控组件级慢 4–6× 同时成立时，两个维度分开报告，分别给判定。
- **判定口径**：parity = 同口径 RTF 双方区间重叠或均值差 ≤10%；超过 = 更快
  且区间不重叠；均值小幅领先但区间重叠不得报成「超过」。
- **多臂交错**：Rust/官方/中间候选三路以上时用 ABCCBA 及独立进程（OmniVoice
  2026-10-23 审计的协议），沿用每进程预热与热轮、逐轮数据全部保留。
- **先排除官方侧假象再下结论**：官方首轮巨慢（IndexTTS2 MPS 首热轮 117 s）
  用稳态轮并披露全部轮次；官方流式内部 hop/缓存状态跨请求保持时，注明
  「新鲜模型态」与「热轮」两个数字。
- **官方输入契约陷阱**：跑不通先怀疑输入格式再怀疑性能——CosyVoice3 zero-shot
  转写必须自带 `<|endofprompt|>` 后缀，缺失时 LLM 线程断言退出、主生成器在
  轮询睡眠里假死（0% CPU + time.sleep 栈，像挂死其实是崩溃后的空转）。
  对照脚本对官方 API 的每个前置条件做显式验证，假死先采样栈定位。

## 官方实现与资源身份

自回归参考API返回完整音频时，可在单独诊断轮包装原方法，读取真实生成序列末尾
EOS与code帧数，并在生成/codec边界同步。验证诊断输出与正式PCM一致，仍将诊断
计时单列。方法叫Talker.generate可能内含CodePredictor、采样和主机控制，不应把
整个方法耗时标成某个网络或GPU kernel。若两框架RNG不同，用显式贪心的独立case
区分随机轨迹影响；不能把贪心结果替代用户原有随机采样验收。

不要以“后处理codes少于预算”推断EOS：包装器可能删除末尾token，触顶375个
语义token也可能只返回374帧。完成门禁必须检查未裁剪生成序列或可信终止原因。
把触顶案例用于负向回归，确认父进程确实拒绝；发现历史false success时保留原始
收据，另记作废原因，不能继续用于排名。Qwen Windows贪心控制已实机验证此陷阱。

发现入口或API用法可疑时，先做能区分假设的实际输入对照。变量名叫正则不代表
下游按正则解释，但源码层面的疑点也不自动解释输出差异；记录原路径、候选与官方
token ID。代理复现和真实目标二进制分开标注。没有观察到差异的夹具只能排除该
输入上的解释，不能宣布通用等价或把改动算作优化收益。

框架的顶层 dtype 可能连带传给内部 codec/tokenizer；核对各子模型实际参数 dtype，
不能只看主模型参数。若要对齐原始 checkpoint 精度，在首次加载该子模型时选择对应
dtype；先转成低精度再 `.float()` 不能恢复原值。官方所需的额外处理器配置应从相同
固定 revision 准备并记录身份，不靠手写猜测。只覆盖加载选项的参考仍需标注区别于
官方默认配置，不能顺便更改采样或数学实现。见 [Qwen Windows 案例](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)。

同时区分checkpoint存储dtype与实际计算dtype：Qwen Base speaker权重本来BF16，
原生F32计算与官方BF16计算不一致，并不说明官方加载丢失了原始F32权重。先读固定
文件header和运行时参数，再解释误差来源；参考准备时间与缓存热轮必须分开记录。
若生产路径读取磁盘参考embedding/latent缓存，WAV和metadata摘要不足以冻结实际条件；
先完成准备，再把缓存作为conditioning artifact绑定并验证运行前后摘要。发现旧Job漏绑时，
生成新Job复测，不将当前缓存SHA补写成历史事实。
[VoxCPM Windows缓存身份补查](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)记录了该缺口与恢复入口。

参考prompt有并行子路径时分别验收：embedding接近不代表离散codec codes相同，
总一致率也可能掩盖不同codebook的差异。先比较形状、各组ID及embedding误差，
再定位负责的前端；不要把独立编码器的差异归于speaker dtype。共享冻结prompt
可以隔离生成器，但须标注跳过的参考准备，不能代替完整链路对照。

共享prompt控制还要验证实际传入的F32位模式、离散codes和参考转写token IDs；
形状相同不等于输入相同。随后用独立贪心case排除RNG的充分解释，仍有差异时转向
prefill/logits等中间数值，不因音频长度碰巧接近就宣布生成等价或精度达标。

固定实际导入的官方源码，而不只固定外层 CLI 仓库。Hugging Face 自定义模型的
`auto_map` 可能额外加载模型仓库 Python；TTS 与 codec 还可能来自不同 revision。
在准备阶段取得这些代码并记录摘要，避免首次正式计时隐式联网；离线预检先验证导入。
权重可只读复用；若用硬链接构建独立官方目录，禁止原地修改链接的配置或权重。

CLI 的帮助文字、参数默认值与传入模型的有效参数可能不一致，沿实际调用链确认采样、
精度、attention、分段和流式策略，再显式冻结对照条件。跨框架 seed 不代表同一轨迹，
需保留长度、EOS 和质量证据，不能单凭 RTF 声称实现等价或获得同工作量提速。

Talechime 当前生产测量使用 Rust `xtask perf`，接口见
[性能工具说明](../../../docs/performance-harness.md)。模型准备需要 `performance_catalog`
的构建收据，测量使用 `performance_probe` 的独立收据；两者不得互换。历史 Job 结构
可能已过时，按当前类型核对字段，不修改旧原始证据来伪装成新版本产物。

确认 A/B 收据指向的是独立保存的二进制，而非 Cargo 会覆盖的 example 路径。只把
features 用作构建目录键，不能隔离同 feature 下不同源码；下一次构建就可能破坏
baseline。Talechime 已改为在收据目录按 SHA 保存副本；旧收据仍需校验，失败后重建。

清理磁盘前先解析 `target` 的真实路径及目录联接，并检查是否混放模型、音频、报告和
必要的复现二进制。仅删除已授权的可再生缓存；整目录 `cargo clean` 并不适用于混放
证据的 target。释放磁盘不等于解决运行时 RAM/VRAM 不足。


## 退化调查

先重复不改代码的基线。分别观测频率、功率、温度、外部 GPU 进程、懒加载、冷/热
状态和计时污染，每项独立进程；不要同时改代码和系统状态。
系统调优在用户授权范围内进行，检查命令返回值与实际生效状态，并恢复原设置。
不硬编码某台机器的电源 GUID、SM 频率或线程数；不支持锁频时保留限制说明。
环境变量在进程创建前设置。关联不足则记“未定位”，不臆断 allocator 或驱动缺陷。

## 输出门禁

| 改动 | 对照重点 |
| --- | --- |
| 图/复制/缓存 | 同输入 eager、不同输入连续调用、形状/steps/cfg 切换、LRU 淘汰、别名与资源生命周期 |
| KV/attention | 非零 offset、prefill/单步、GQA 头映射、因果/滑窗 mask、跨请求清空、真实量化权重 |
| 卷积/布局/BLAS | 独立参考；大形状、短输入、多 batch、bias/padding/dilation/groups、实际 dtype/设备 |
| 排序/采样 | 同分、累计恰好等于阈值、原词表 ID、空历史、多行、实际词表大小、RNG 推进 |
| 流式状态 | 首块参考、短尾块、样本覆盖、块边界、失败/取消后重用、下一请求 reset |

小张量容差通过不代表真实生成相同；低精度/量化和迭代模型可能放大误差。
先定位第一个分歧：条件输入、隐藏态/logits、token、latent、PCM，避免只看最终波形。
审查 eager 对照的实际入口：默认开启图的 session 不能同时充当 eager oracle。
比较固定 KV 长度的图时，eager 也须使用相同有效长度；缓存容量相同不足以证明
工作量相同。缓存准备可以在计时外，但不能把不断增长的 eager KV 与固定长度回放
相减后称为同工作量收益。
顺序 F32 累加与 GEMM cumsum 不保证相同；不要以数学等价代替累加顺序证据。

保持输出的 TTS 改动比较 codes、原始 PCM、样本数、EOS 和相同 writer 下的 WAV hash。
数值变化另报误差、信号统计与质量验收；SNR/ASR 回读不能替代听感或内容验收。
单ASR可作内容辅证：冻结实际WAV摘要，只把音频送入识别器，原文仅用于结果对照；
保留模型/revision、原文/转写/标签，标明未做双识别器或人工试听。相同转写不证明
音色、韵律、发音细节或数值等价，不将情绪标签当质量评分。
[VoxCPM与后续Qwen实用验证](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)
使用了显式本地的sensevoice_aux_readback，模型准备与实机读取分开，普通测试不下载模型。
帧上限不等于自然 EOS，生成完成也不等于播放完成。其他模型对应比较 token/张量和
其公共完成/取消契约。普通测试不下载模型；缺少实机证据明确列出。

## 持续负载离群段（先于退化调查）

长进程在持续 GPU 负载下可能出现**可复现的暂态离群段**，且**在负载不中断时自行恢复**。
它不是单调热衰减，恢复这一点是与热衰减最可靠的区别。

M4 Pro / VoxCPM 的实测形态（1 预热 + 12 热轮，每轮约 60 s）：

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTF | 1.002 | 0.999 | 1.002 | **1.139** | **1.244** | 1.081 | 1.041 | 1.003 | 1.008 | 1.008 | 1.006 | 1.003 | 1.007 |

约 3–4 分钟起、峰值 +24%、持续 3–4 轮、之后完全回到基线并保持 6 轮。

**处置顺序**：先复测一次长时程确认可复现，再逐项排除，逐项只动一个变量：

| 候选 | 观测量 | 结论口径 |
| --- | --- | --- |
| 进程内累积 | 2 s 采样 RSS | 无单调增长且离群结束后不回落，可排除 |
| 系统内存压力 | `memory_pressure`、`vm_stat` 的 throttled pages | 空闲充足可排除 |
| 第二个推理进程 | 全程 `ps` | 只有一个可排除 |
| 频率/功耗/温度 | `powermetrics`（**需 root**）、平台热日志 | 未采集就写「未定位」 |
| 其他 GPU 客户端 | 常驻 GUI/同步进程 | 未授权关闭就写「未排除」 |

**协议影响**：一旦存在离群段，「1 预热 + 5 热轮」这种单进程长跑不可用于 A/B；
历史基线若还使用了不同参考音频、不同每轮音频时长或不同版本，应把全部条件差异与
离群段风险一起列出，不能直接比较单个数。改为每臂一个短进程、臂间交错、
进程间冷却（历史 `tools/tts/voxcpm_perf_macos.py` 即此实现）。该轮采用
「≥10% 且两组区间不重叠」，这是历史判据；当前按局部和整链的可信收益、成本及
回退情况决定是否保留，不沿用默认百分比门槛。macOS 没有 `nvidia-smi -lgc` 等价物，
**不要伪造锁频步骤**；把「无法锁频」写进限制，而不是假装已锁。

离群段幅度超过任何优化候选时，先定位它：在解释清楚之前，
本机任何小于该幅度的「提速」都可能只是漂移。

## 筛选工具必须先自证没有顺序偏差

Windows 官方参考在加载阶段崩溃时，同时检查系统 commit used/limit、进程 private
bytes 与 RSS；空闲物理内存、空闲显存和磁盘空间不能替代提交额度。记录失败发生于
导入、打开 checkpoint、搬运权重还是推理。1455 是提交资源不足的直接证据，但不能
反推所有访问违例的原因。禁用异步加载后仍失败，就停止将线程作为已确认根因。
若固定库版本支持无 mmap 的官方读取接口，可先用小夹具验证 dtype、特殊值与切片
逐位一致，再在单独环境做一次加载预检；记录 IO/依赖偏离，保留旧环境和冻结收据。
不要为了强行加载而扩展未验证的自制权重解析器或降低模型精度。
实例见 [MOSS Windows 官方加载调查](../../../docs/records/moss/moss17-windows-production-2026-10-24.md)。

改变库版本并切换读取后端同时成功时，还不能单独归因于读取方式；条件允许时在
同一新环境补默认读取的加载对照。Qwen1.7B Windows上SF0.8 mmap仍访问违例，
pread则完成正式推理，验证了该方法的第二个模型用途；这是本机条件事实，不是
任意加载崩溃的通用修复。参考[Qwen生产对照](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)。

用 A/B 工具筛候选之前，先证明工具本身不会制造假信号。本机实测到两个真实缺陷：

| 缺陷 | 现象 | 后果 |
| --- | --- | --- |
| 筛选序列不平衡 | 2 进程序列 `(baseline, candidate)` 让第二臂系统性吃亏：首 PCM 421 → 535 ms（+27%） | 把候选 +6.9% 的真实收益显示为 **-5.5%**，会错误撤回正确改动 |
| 整轮累积漂移 | 8 进程 ABBA 的后段两个臂**同时**变慢（baseline 末位 +7.4%，candidate 中段 +6.7%），短冷却恢复不了 | 只看均值会把漂移算进收益或损失 |

**规则**：

- 任何筛选序列都必须**顺序平衡**：`(b,c,c,b)` 起步；两进程的不平衡序列只能
  当粗筛，且不得作为撤回依据。
- 报告**按 slot 配对**的差值，而不只报两臂均值；保留全部 slot；子集只能作为补充描述，不代替正式验收。
- 判断漂移：若某臂在靠后的 slot 单调变差而另一臂不变，是候选问题；
 若**两臂同时**变差，是环境漂移，应延长冷却或分批测量。
- 在自检通过前，不要用该工具的"无收益"结论撤回候选。

对齐参考 prompt 且关闭随机采样后仍有输出差异时，从生成前的 prefill 开始
比较：绑定相同权重、输入和 dtype，导出位置、prefix/末位置 hidden 与首步
logits，报告误差和 top-k 排名。先区分“已有数值差异”与“已经改变选择”：
高余弦不证明质量通过，首步 argmax 相同也不证明后续 codes 相同。分段与合并
prefill 可能改变 GEMM 形状，未做受控实验不能把误差直接归因于分段；诊断
hook 的主机同步和导出不计入正式性能基线。先核对实际输入 embedding；重建输入
的诊断助手可能复制组装逻辑，必须声明此限制，完成实验后保存补丁并撤回重复代码。
合并 prefill 若未缩小误差，不据此更改生产路径或宣称根因已排除。Windows Qwen1.7B 的首步实例见
[原生/官方 prefill 对照](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)。

审计流式参考脚本时，同时核对“循环内累计”与“循环外保存”：完整WAV应包含
全部chunk，其样本数应等于累计有效样本数，并与报告音频时长一致。只保存末个
chunk不能支持完整输出/内容验收，但不据此自动否定独立累计的耗时；分别说明
哪些证据失效，重建新适配器后再测。实例见
[VoxCPM2旧官方脚本复核](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)。

通过虚拟环境或启动器运行官方进程时，先确认采样PID是否仍是实际推理进程。
单个root PID的RSS可能只含启动器；不要直接称为模型峰值。Windows现有xtask
在capture.json另记contained_job的OS commit高水位，包含后代但不等于RSS或
显存；旧样本不能补回，通用RSS图也不自动合并该口径。用显式分配内存的后代
夹具验证归属和退出后保留，再用于模型加载诊断。证据见
[VoxCPM Windows](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)。

生成器只在产帧时发布 finished_mask 时，最终 EOS 可能发生在下一个不产帧的分支；
最后一个 frame 的 mask 不能证明自然结束。先检查最终返回类型与 break 路径，
若没有 EOS 字段，在适配器中记录实际终止 token 的选择，保持原推理决策和计时边界。
也不要把 trimmed frames < cap 作为唯一 EOS 证明。MOSS Nano 固定官方源码存在此形态；
当前生产基线已重建，正式官方验证仍待完成，见
[MOSS Nano Windows 当前基线与 EOS 调查](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)。
沿调用链核对有效seed：会话Pinned策略可能先哈希call/attempt，模型再截为u32，
请求里的42不一定是采样器得到的42。官方对照先匹配最终数值，再单列RNG算法差异；
保留修正前独立case，不把更短输出用于主排名。多声道原始PCM也须注明平面或交错布局，
先统一布局再比较摘要，否则只是重排就会被误判数值漂移。
[MOSS Nano Windows](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)
实际验证了两项：有效seed3560097011；旧平面PCM转交错后与新seed42参考摘要完全相同。
该案例也撤回了8.39%表观收益的RoPE复用：前后半程差异大且缺独立阶段证据。
撤回依据是归因不足；若未来测到可靠局部收益，应按当前项目策略重新评估，不能机械卡整链10%。
取消回归应观察状态而非只设置取消标志：在真实owner复用测试中先完成固定请求，再取消
未消费流，之后实际观察有界接收队列capacity/len已满再丢弃流，重跑同一请求逐位比PCM。
把“队列已满”与“捕获到send正阻塞”区分；把无效输入被拒与推理内错误恢复区分，
每条结论对应实际触发路径。真实测试默认ignore、显式本地资源且禁止隐式下载。
[MOSS Nano Windows](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)
在CUDA owner上验证四份完整PCM相同，并补充64字元段落的源范围及辅助内容回读；
多段生产调度对比单提示官方时，单列call seed/分段策略，避免把调度差异当作算子收益。
热计时外的工作也应审计：若完整轮次墙钟远大于合成时间，检查序列化、文件写入与
摘要边界，不能先把差额归到GPU。未计时导出仍会改变轮间设备空闲时间；修复它后
统一两臂实现并冻结新探针，不能把导出变化同时混进模型候选的性能归因。
[Windows WAV导出案例](../../../docs/records/platform/windows-artifact-export-2026-10-24.md)
测到逐样本File直写约2.9s/格式、64KiB缓冲约4ms，完整字节匹配原工件。
这里只验证导出收益，未证明GPU频率因果。缓冲写必须确认库的finalize显式flush并
返回错误；用底层write/flush失败注入验证，不能依赖Drop悄悄完成导出。
Windows provider验证要分开能力枚举、实际节点执行与生产入口：RTX5070的CUDA分发
缺内核不妨碍另一DirectML分发实测成功。固定DXGI index和LUID，检查profile中的实际
DML/CPU节点；error_on_failure不禁止CPU fallback。当前Runtime开发探针不冒充Registry，
缓存比较需同时核对PCM、EOS、取消后复用及WAV导出计时。见
[MOSS ONNX Windows实测](../../../docs/records/moss/moss-nano-onnx-windows-2026-10-24.md)。

ONNX后续实践：同1.28版本的PyPI GPU与Rust缓存分发有不同build commit/架构，前者实际
包含sm120 PTX并完成官方18轮，不能只按包版本判断。预加载警告、PATH与实际模块路径
也可能不同，保存GetModuleFileNameW与DLL SHA；不要只替换一个provider假装完整分发。
官方codes相同而PCM变化时先查自适应codec分块；相同seed还需核对RNG流和prompt行，
独立控制实验不能覆盖原默认基线。证据仍见MOSS ONNX Windows记录的官方Python节。

动态ORT后续已在真正Registry/Backend路径运行：显式load-dynamic构建、ORT_DYLIB_PATH
固定完整新分发，另用真实进程模块表验证主库/shared/provider同源，27普通测试与
CUDA取消/失败/换音色复用通过。见同一ONNX记录末节。依赖feature必须查依赖crate的
compiler-artifact，不只查顶层features；普通测试中的cache identity也可能先调用ORT，
因此环境要在首次API前设置。若原生harness身份未冻结动态DLL路径，先用明确环境断言
和输入SHA的隔离桥接，不把不同运行库的结果当可Resume；默认搜索拾取旧DLL的失败须保留。

下一轮ONNX实践发现完整SynthesisStream带静音/响度处理，与官方raw codec直接相减会
制造约0.058的模型误差假象。先确认实际seed、完整prompt行和codes，再在模型原始边界
采集PCM；本次226行prompt/35帧codes及同库原始codec块均一致。不能静默关闭生产处理
来美化对照。根DLL同版本/路径也不证明全部组件一致，单项排除实验不能直接当因果结论。
证据见MOSS ONNX记录的输入与原始PCM边界对齐节，临时日志钩子已撤回。

动态ORT的下一轮工程实践已补Native环境身份：显式绝对ORT_DYLIB_PATH及同目录provider
内容指纹进入条件，调度和Resume复用前重算。同路径换库拒绝旧结果，恢复原字节可复用；
未选择动态库的旧静态身份不变。这不涵盖外部CUDA/cuDNN或实际模块搜索，仍须原有审计。
[Windows身份回归](../../../docs/records/platform/windows-ort-runtime-identity-2026-10-24.md)
记录实际CLI负向/恢复实验；不能把模型无关夹具当GPU质量或动态ABI验证。

下一次正式 Native 构建已实际使用上述方法：依赖限定 feature 要从对应库的 Cargo
compiler-artifact 验证，不能只记录请求参数。Windows xtask 的 ort/load-dynamic
收据通过独立构建、夹具和18轮真实 CUDA 请求验证，见同一身份记录的后续节。
桥接与正式 Native 的耗时不同，不能将工具入口切换解释为模型优化收益。


进一步的Windows ONNX实践表明：预加载同版本根DLL不足以固定插件/engine组件；
整套较新cuDNN预加载又可能在Torch导入时缺少入口而失败。先做独立进程加载预检，
合成后审计实际组件路径及SHA，失败时停在该组合，不绕过导入或伪造成功轮次。
反向让Rust使用官方现有cuDNN可运行，但同二进制PCM出现约0.000955最大差；PATH
还可能同时引入其他版本cuBLAS，不能仅凭根版本归因。细节见MOSS ONNX记录的整套
cuDNN控制节；外部库不在ORT自身bundle指纹内，须另存环境证据，不能复用旧质量验收。


后续Nano CUDA实践撤回了CPU top-k局部排序候选：合成1024分数排序微基准约快9倍，
20480次完整抽样与旧实现逐token/RNG一致，但真实ABBAABBA完整请求0.570→0.586s，
范围重叠；未证实真实采样阶段收益。不能把合成热点的比例直接换算成端到端收益，
也不能用正确性通过替代性能证据。保留阈值所有并列项、恢复稳定token次序是该类
候选的必要等价条件；实现仍撤回，避免为未证实收益维护额外路径。设备、逐进程
结果和冻结证据见[MOSS Windows记录](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)
的“后续撤回：Nano CPU采样局部排序”。这不是恢复整链10%通用门槛。

同轮撤回还发现Windows复制备份保留mtime，Cargo可复用较新的候选产物。恢复源码后
除核对内容SHA，还要确认重编译及实际测试列表；必要时更新时间戳重建。不要把旧
产物的测试成功记为恢复源码的门禁。本例发现多出的候选测试后重编译，最终12测通过。


Windows官方加载IO实践：Safetensors0.8.0即使选pread，也先为元数据建立整文件COW
映射，打开返回后的常驻内存低可能掩盖整分片提交峰值。分别测瞬时高水位和常驻，
跨文件归因用独立进程；切片可能保活映射，先清引用，且峰值计数不会随close清零。
核对实际backend参数/模块路径，不能从报错调用行推断patch漏出作用域。局部只读
header控制已在两个真实分片降低峰值，但官方整模型仍在codec分配/物化失败；IO
改善不等于已恢复推理或通过质量验收。先读checkpoint头核实dtype/参数总量，不从
模型标签或文件大小猜测精度回退。证据、依赖偏离及失败判据见
[Windows Safetensors提交调查](../../../docs/records/platform/windows-safetensors-header-commit-2026-10-24.md)。


后续Qwen0.6B Base实践：同一参考WAV可以经生产接口分别导入，但不能跨模型复制
prompt cache；先让新模型准备，再冻结自己的WAV/metadata/cache三项身份。catalog
资源根与请求使用同一字符串，否则即使文件相同也可能触发收据摘要拒绝；修正新Job，
不弱化校验或把已拒绝结果改成成功。数据生成脚本也应拒绝覆盖冻结Job。
参考codec帧数与含额外slot的ICL位置数须分别命名，本例两边实际参考都是86帧。
三进程的第一个热轮重复偏慢应保留并按请求序号追查，不能悄悄增预热后删样本。
Native数组和external_reference对象按各自capture/round审核，不能假定结构或Source
可互换。双方36轮及单ASR辅证、边界/随机流偏离见
[Qwen Windows Base06记录](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)。


Windows GPU 进程采集也需要独立验证编码：真实进程路径可能不是 UTF-8，不能让
名称解码失败掩盖整份 PID/显存列表，也不能把失败列表当空列表。保留原始字节的
显式转义与编码说明优于猜 code page；结构和数值字段仍严格拒绝损坏输入。
修复只作用于新采样，不追溯证明旧基准没有竞争；WDDM 未提供的逐进程显存仍为 null，
compute-apps 列表也不能代表所有图形负载。实现与实机证据见
[Windows GPU 进程编码调查](../../../docs/records/platform/windows-gpu-process-encoding-2026-10-24.md)。


上述编码修复随后在 VoxCPM 正式报告发现接口遗漏：名称编码说明属于进程明细，
不能放进 available 数字 Reading 的失败 reason。collector 单测和实机返回成功后，
仍要验证实际 frame→Reading::validate→report 全链。保留失败 run，修数字映射并
新建运行，不改旧日志、不放宽校验；详细证据见上述 Windows GPU 编码记录的后续节。


VoxCPM 官方加载后续实际复用 MOSS 的 IO 方法：先以 meta 骨架与 checkpoint 头逐项
核对键、shape、dtype、参数字节量，再决定是否能绕过未使用的 F32 随机初始化。
meta 权重可按原 dtype 逐项物化，但 checkpoint 没有的非持久 buffer 必须按原构造/
转换顺序重建；逐项验证加载值仍不能替代完整默认路径的输出对照。保持 forward、
上下文、随机与输出模式，加载控制及依赖偏离单列，不把额外校验的冷加载速度拿来排名。
这里验证577主张量+312VAE状态+9RoPE buffer并恢复两轮自然EOS，但系统commit仅余
335MB，正式3×(1+5)提高预检预算后拒绝启动，不能将smoke扩称正式达标。见
[VoxCPM官方加载控制](../../../docs/records/voxcpm/voxcpm-windows-official-loading-2026-10-24.md)。

同次实践发现虚拟环境 python.exe 可能只是 launcher，实际解释器在子进程：父工具
单PID RSS几MB不是模型RSS。先核对进程树与scope，Job commit高水位覆盖子树但不是
RSS；缺少子树RSS时明确未知，不把不同scope内存放进相同排行。成功运行后也要检查
最低系统余量，启动前快照预算不预留内存，更不是已经验证足够的内存需求。


VoxCPM 后续 meta VAE 组件对照实际检验了首次调用陷阱：单进程 A首次/A第二次/B
可能连 A 自身都不逐位重复。先保留失败和误差，再用独立进程按调用序号比较；该例
13对组件结果及整模型两请求的同序号PCM精确相同，没有放宽容差或删除首轮。不能
从此猜测具体库机制，更不能将warmup丢掉后称全部输出恒定。冷内存快照还会漏掉
瞬时峰值：阶段结束1.38GB余量时，父采样实际捕获过0.408GB；保留启动预算，不因
局部常驻改善就宣称完整运行安全。版本、失败与成功门禁见上述VoxCPM官方加载记录的
v3后续节。该阶段仍是实验对照，未把单轮Job峰值差宣称稳定收益。


VoxCPM 后续将固定短句扩至63字段落，三条路径各3×(1+5)，并用默认/eager完整
请求分别复核PCM与结束。比较不能只看wall区间：Q8 2.894s/12.4s音频、C++
3.022s/12.8s，wall范围虽不重叠，RTF范围却重叠；没有稳定吞吐胜出的证据。
当前后端自己的18轮摘要一致不等于跨实现/跨开关等价，须另做输出对照；不同精度和
默认/参考音色也不能放进同条件排名。新case见
[VoxCPM Windows段落记录](../../../docs/records/voxcpm/voxcpm-windows-paragraph-2026-10-25.md)。

同例显式关闭图时，Job声明的环境不是父进程环境赋值；Native前置核对会拒绝不匹配，
须在调用工具的Shell设置相同值并用新输出目录，不修改校验或把拒绝算作模型失败。
单识别器文字、同音字及情绪/背景标签作为辅助证据分别保留，不能替代人工音质结论。


上述VoxCPM段落后续还发现段界遗漏：即使源文只有63字，也不能推断生产只发一次
模型请求。逐份核对raw ranges、文本SHA、实际call/seed；本例Native先14字后49字，
C++和组件回放却一次全文。RTF不能消除prefill次数差，直接Model的真实张量也不等于
生产宿主相同负载。先补正旧报告、保留产品行为结果，再按冻结段界补模型对照，不将
raw长度差猜成后处理或性能回退。见上述单query调查的边界补正与完整证据。

补测时先在CPU校验源摘要、UTF-8范围、每段摘要和实际seed，再加载模型；同一owner
按真实顺序跑完全部段，最后才回放捕获张量，避免局部预热插入段间改变状态。
VoxCPM两种精度六进程81组context逐位一致，但这仅关闭了输入边界缺口，不能替代
候选整模型PCM与宿主验收。较短段到不了预定的晚期ordinal，应记录实际捕获数，
不能用不同段的张量补齐或把未访问点算通过。结果见上述单query调查的分段补测节。

同一owner比较旧/新路径时，先检查是否可能复用旧实现录制的图；逐位门禁可明确用
eager，再以各自冻结的默认图进程补生产比较。组件同步/捕获不能混入正式计时。
脏工作区除了源码摘要，还应保存当轮改动源或diff，避免下一轮改动后只剩不可复原的hash。
输出摘要也要按工具实际域复算：格式前缀、布局与裸WAV data块的hash不是同一对象。

VoxCPM后续4case交错验证中，两种段落约5.26%/2.18%改善且范围不重叠，短句却
inconclusive。单独用同一旧二进制对照GPU采样on/off，均值约0.606/0.608秒、范围重叠，
未支持采样干扰假设。保留负向归因结果，不因短句波动便删轮、关采样或挑选新批次；
局部、长请求、短请求与首结果分别下结论。完整数据、真实patch/PCM/状态门禁见
[VoxCPM复制运行时验证](../../../docs/records/voxcpm/voxcpm-windows-key-packing-2026-10-25.md)。

资源阻塞是当时条件，不是永久结论：VoxCPM用户释放内存后，保留12GB门禁重新预检，
控制版官方3×(1+5)完成，最低系统余量约1.70GB。同时保留默认加载失败与控制版偏离，
不把“已能正式测推理”扩大为默认入口恢复；解释优化开关还须核对实际依赖，当前无
Triton的官方optimize分支会警告退出，不能把optimize=True文字当成已编译执行。

同次对照从真实raw/宿主输出两层验证工作量：两边19patch/145920样本，Rust实际
BoundarySilence复现140640样本，不能按2.93与3.04秒猜测少生成。解码PCM比较还发现
WAV的PEAK时间戳导致文件SHA不同，数据却逐位相同；分别保留资产摘要和数值摘要，
不要改写证据文件使hash“相等”。既有ASR可复用的前提也是PCM精确相同，并保留原文件
身份及识别结果链接，不升级为试听通过。见
[VoxCPM官方正式对照](../../../docs/records/voxcpm/voxcpm-windows-official-baseline-2026-10-25.md)。

合并后回退验收也要区分纯输出和混合计时JSON：ranges文件可能同时包含分包样本区间、
source_range与consumer时间戳。保留整文件SHA，再分别比较稳定语义和时序，不要求
计时字段相等。旧资产须由results中的attempt_id定位，目录可能留有非对应同名文件；
枚举实际文本分段/资产，不能硬编码单段。Nano新旧36轮的实例见
[Windows合并后验收](../../../docs/records/moss/moss-nano-windows-merged-acceptance-2026-10-25.md)。

后续VoxCPM的baseline还包含带时序的raw_owner_events，不能只排除consumer时间戳
后比较整个JSON；明确列出稳定字段、完整区间覆盖与分包边界，并检查采集flush和丢弃。
该方法在Q8/BF16四case72轮、432资产上复用，同时另验真实图active/EOS和owner恢复，
见[VoxCPM合并后验收](../../../docs/records/voxcpm/voxcpm-windows-merged-acceptance-2026-10-25.md)。

该资产/时序分离方法也用于[Qwen 0.6B合并后验收](../../../docs/records/qwen/qwen-windows-merged-acceptance-2026-10-25.md)。
覆盖范围须按实际模型分支记录：0.6B owner通过不能覆盖仅Custom17执行的style测试，
内存余量不足而未启动的1.7B保持待验；共用测试入口或二进制不表示所有分支都执行。

转换摘要不匹配时，把融合计算与序列化分开干预。CosyVoice3后续发现参考要求SF0.8，
而新增Torch2.8环境实际为0.5.2；CPU重转换及仅用SF0.8重序列化均仍得到原摘要，
246张量位模式一致，版本切换没有解决门禁。保留这种否定结果，不能见版本差就宣布
根因，更不能只据大小/键数或另一版本导出成功修改生产摘要。缺少canonical文件时
明确缺口并继续其他模型，详见
[CosyVoice3版本控制实验](../../../docs/records/cosyvoice3/cosyvoice3-windows-production-2026-10-24.md)。

官方精度要追到 processor 实际加载和推理上下文，不能只看主模型 dtype。
MOSS Local 官方主模型 BF16、codec 默认 F32；旧参考强制 codec F16，加载成功后
仍会因 quantizer 输出 Float / Half 卷积不匹配而失败。F16+autocast 虽能执行，
固定 codes 的7680样本全部改变，不可默认为原官方基线或无损加载修复。权重逐位
一致只证明加载，不证明混合精度执行相同。先做 codec-alone 的加载、固定输入推理，
再恢复完整链路；参数下界已超过独立显存时明确方案限制，不能暗中降精度、依赖
共享显存或把 Job 提交峰值称作显存。见
[MOSS 官方 codec 合约](../../../docs/records/moss/moss17-windows-official-codec-contract-2026-10-25.md)。

共用底层模型不等于共用所有者验证：MOSS Nano 的取消通过，不能覆盖独立 Local/
Realtime 适配器。后续为两种1.7B模式分别补上未消费流取消、实际满队列取消和真实
帧预算推理失败，异常后都用同一所有者恢复出完整逐位相同 PCM；失败不得发送 End。
拒绝未知音色发生在推理前，不能替代推理过程中失败的恢复验证。测试的总运行耗时
不当作性能热轮，宿主 End 也不冒充内部 codes 轨迹，见
[1.7B状态门禁](../../../docs/records/moss/moss17-windows-production-2026-10-24.md)。

参考与接续还需交错测试：自行生成完整参考并用其真实文本导入唯一临时音色，检查
首次编码/命中缓存的完整PCM与缓存字节，再交错无参考、接续、接续取消、克隆与
无参考请求。MOSS两种1.7B模式均通过这种状态恢复验证；它不证明音色或接续听感。
只清理本次创建且身份匹配的音色，不覆盖用户数据。容量分析也按实际变体计数：
Local默认参数超显存，而Realtime参数下界尚能容纳，不能仅据共同的1.7B标签把
一种模式的失败当作另一种模式的阻塞。证据仍见上述1.7B记录。

默认精度无法同时常驻时，可把分阶段GPU驻留作为明确隔离的诊断：先生成codes，
卸载主权重，再加载默认codec解码。记录卸载后的实际allocated、每阶段时间、所有
加载/换载成本和提交余量，不将阶段时间相加冒充常驻完整请求热轮。MOSS Realtime
两次这样得到相同codes/PCM并完整ASR回读，证明兼容路径可产生语音；仍没有常驻热
基线。保留原始失败和非默认驻留方式，不让一份“可运行”诊断替换原验收范围。见
[Realtime分阶段诊断](../../../docs/records/moss/moss-realtime-windows-official-staged-2026-10-25.md)。

原生资源清单不等于官方Python完整资源：Local权重能加载，仍因chat template缺失
在生成前失败。核对相同权重revision树，从官方取原始模板并固定SHA；不手写近似
模板或关闭apply_chat_template绕过。无参考请求延迟codec加载时，分别在codec未加载
和真实加载后构造官方processor，比较完整input_ids/attention_mask，才能排除提示词
改变。Local这样恢复了两个相同原始序列/PCM及完整ASR回读的默认精度GPU诊断。

EOS与音频帧必须共同核对：此版本Local原始40行含首pad行及最后EOS行，最后行仍有
非pad音频codes，官方processor实际解码39行；Rust在控制EOS后即返回。仅据EOS前
帧数对齐会误判工作量，不能删官方末行或加随机尾帧来“匹配”。把这种版本限定
差异、原始控制/音频通道、processor裁剪和最终PCM分开记录；诊断成功仍不等于热
性能或听感达标。见
[Local分阶段与EOS审计](../../../docs/records/moss/moss-local-windows-official-staged-2026-10-25.md)。

构建前后Source一致仍不能证明Cargo缓存产物对应它：复制旧源码保留mtime，可能使
Cargo返回fresh并复用另一候选二进制。MOSS同期参考初次即出现不同Source、相同
二进制SHA，已拒绝该收据；更新实际覆盖文件mtime、确认相关crate重编译后才得到
有效A，并按摘要恢复候选。保留失败收据，不把相同二进制A/B当优化实验；不要为
此盲目清空全部target。见
[RoPE构建身份审计](../../../docs/records/moss/moss-windows-forward-rope-reuse-2026-10-25.md)。

比较器拒绝标签时先核查实际观测语义，不改写原始结果来过门禁。此轮两臂均为合法
未插桩off，旧比较器却只接受baseline。修复后增加off正例、不同标签不兼容及同为
trace/profile仍拒绝的反例，再用新冻结工具复算相同原始数据；测量运行器版本不变。
工具误判、修复前结果及完整条件都保留，不能把诊断模式简单重标成baseline。

WAV数值审计也须识别WAVE_FORMAT_EXTENSIBLE的浮点子类型，不能只接受format tag 3。
优先用已有音频reader读取；原生WAV资产摘要、PCM数值摘要及格式元数据分别验证，
不要因头格式差异改写原始音频。此次RoPE的96轮另按原生摘要域独立复算通过。

不产生音频的组件应使用明确的诊断记录，而非伪造静音WAV或TTS耗时。xtask的
reference --diagnostic复用进程/预算/身份与资产审计，观察事件承载summary和数值
资产，禁止round，普通reference也不能把诊断当成功音频覆盖。权重审计同时包含
buffer及整数计数器；比较字节前可reshape临时副本，不能因0维Long无法直接view
Byte便跳过buffer或修改模型dtype。IndexTTS隔离937张量后CAMPPlus可运行，但这
不证明整模型共驻留资源足够或定位了旧unknown error，详见
[非音频诊断实例](../../../docs/records/indextts/indextts-windows-campplus-isolation-2026-10-25.md)。

提示/风格条件也属于请求状态。对共享owner分别固定普通与带条件请求，检查每组
重复输出、带条件取消后恢复普通请求、满队列取消后恢复带条件请求，避免只测单一
输入掩盖条件残留。不同条件生成不同长度不构成提速；位一致的状态恢复也不证明
语气指令或听感达标。见[Qwen style恢复](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)。

续接测试须从已完整生成的原始PCM与对应原文构造上下文，再分别验证临时参考与
静态缓存分支的恢复；不要用宿主处理后WAV或部分取消音频配完整原文。Qwen Base
两种尺寸实际走临时参考编码后，重复/取消恢复通过，但不由此宣称多段音色漂移或
完整链路性能通过。见同一Qwen记录的Base续接状态节。

辅助生成入口也须沿用实际产品边界：VoiceDesign每次加载模型并导出PCM16，不能
把其两次集成测试总耗时均分后，与官方常驻热轮排名；导出PCM一致也不证明裁剪/
量化前F32位一致。先验证自然完成、实际文件与条件，再建立匹配计时。见
[VoiceDesign实例](../../../docs/records/qwen/qwen-voicedesign-windows-2026-10-25.md)。

“每次加载”与“每次新进程”也不是同一矩阵。官方VoiceDesign首次scope结束后
weakref已失效且CUDA active仅8.5MB，但第二次加载仍1455；不要仅凭reserved非零
诊断泄漏，也不要把三个独立冷样本重标成1+5热轮。分别记录对象存活、active/
reserved、系统commit和Job峰值；失败保留，冷/热及清理边界差异写在数字旁。

插桩可能改变依赖墙钟的控制流。MOSS官方ONNX codec按音频lead在1/2/4/8帧间
选择分块，profiling时相同prompt/codes并不保证PCM摘要相同。先记录实际块计划，
另立固定3帧的有/无profile控制，确认prompt/codes/停止/分块/原始PCM一致，再用
trace提出热点假设；固定策略不替换官方默认或变成优化结果。ORT Node跨度也不
等于GPU独占kernel耗时，见[CUDA profiling控制](../../../docs/records/moss/moss-nano-onnx-cuda-profile-2026-10-25.md)。

静态参考缓存应检查同owner A/B/A与取消后的分支恢复。可用已确认完整文本的固定
生成参考，按SHA导入唯一临时ID，先确认cache缺失，再检查首次编码、暖请求与
清理；不覆盖原音色。缓存文件不变只证明未重写，不能冒充新owner的磁盘重载。
要证明重载，释放并join旧owner、创建新owner，可仅让测试拥有的WAV副本暂时
不可用，再验证cache读取与完整输出恢复。清理按存储API的精确路径断言，不按
裸ID猜目录名。两个参考生成不同PCM也不证明说话人质量或提速，见
[Qwen磁盘缓存门禁](../../../docs/records/qwen/qwen-windows-prompt-cache-owner-2026-10-25.md)。
