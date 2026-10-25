# IndexTTS 2.5 Windows CUDA 经验

后续[FIR布局候选](../../../docs/records/indextts/indextts-windows-fir-layout-withdrawal-2026-10-25.md)
已拒绝：广播边界三块cat虽在局部快约36–58%，真实权重完整vocoder的128/224帧却在
三进程中变慢约4–8%/13–16%，144输出仍逐位一致。相位stack/transpose候选亦无
稳定局部收益。不要从局部复制收益直接接生产或增加持久缓存；保留真实stride、
在完整阶段无profiler复查，再决定是否投入owner/整链门禁。候选及测试开关已撤回，
不把组件初筛扩大成生产性能或根因结论。

最新[BigVGAN FIR细分诊断](../../../docs/records/indextts/indextts-windows-fir-stages-2026-10-25.md)
把下一步从小常量缓存转向布局/复制：三个独立进程中，224帧相位拼接/缩放同步范围
约37–39ms，padding亦有成本；这不是生产收益或带宽根因证明。先保留真实stride、
逐位输出及完整阶段门禁再做候选，不能从cat输入数推断kernel数。72份profiling输出
对关闭profiling控制逐位一致。诊断标签与阶段数据使用同一有序通道并明确round；
跨stdout/stderr的合并顺序不可靠，先要求配对完整再统计，不靠相邻行猜测归属。

证据：[Windows 生产调查](../../../docs/records/indextts/indextts-windows-production-2026-10-24.md)。
设备 Ryzen 7 9700X / RTX 5070 12 GB / 16 GB RAM，Candle 0.11 fork 2266099d、
cuDNN 9.14、官方 torch 2.8.0+cu128 / transformers 4.52.1。以下不是其他模型的收益保证。

## 从生产请求找缓存失效

计算库已有参考及 emovec 缓存，但宿主每次请求调用 prepare_reference，使缓存失效。
先量 Registry/Backend 完整请求，不能直接套独立计算库的历史成绩。检查缓存的写入、
命中与清空链路，不以“模型中有 cache 字段”证明宿主已经复用。

在持有模型的 owner 中只保存一个条件身份，实际特征仍归模型；无需另建全局音色缓存。
本例身份为实际读出、验证和重采样后的 F32 PCM 位模式 SHA256，只保存 32 字节。
继续读取音色元数据和 WAV，避免按路径或 mtime 命中同名替换。对实际用于编码的
同一不可变快照取摘要，不能先 hash 文件、再另读文件编码而形成替换竞态。

仅在准备成功后发布身份；开始 miss 准备前清除旧身份，失败不能声称旧状态仍可用。
相同内容不同音色可复用的前提是准备确实仅依赖音频；本例转写/语言不进入该准备。
不要推广到会读转写、模型参数、情绪或设备的模型，届时这些都须进入身份和所有权。
本次只启用 Windows CUDA，其他平台/设备保持原路径，未测量不宣称有相同收益。

## 两种 RNG 的输出验收

Options.seed 控制 host StdRng 采样，CFM 的设备 randn 是另一流。固定请求 seed 的
多轮生产 PCM 仍不同，不能强行要求它们在独立进程中自然逐位一致，也不能悄改生产
随机策略以获得容易通过的“优化”。在显式真模型测试中重置测试拥有的 CUDA RNG，
对照同一采样 seed 下的 uncached、首次准备、命中、A/B/A 与取消后复用。
本例这些完整 PCM 逐位一致；普通测试另验证失效、失败与 owner 隔离。
真实失败准备若无法安全注入，明确普通回调失败测试与真实无效 WAV 的覆盖差别。

正式同参考热请求八进程 ABBAABBA、每进程 1+5：4.168705→1.512353 s，约 −63.72%，
两臂进程范围分离且音频时长范围相同。保留；不外推到冷请求/交替参考，不把每轮自然
随机的 PCM 宣称成全部相同。正确性来自独立受控随机输入对照，完整证据见记录。

## 来源清单与官方入口

新算本地 SHA 只标识当前文件；历史 tensor 计数不证明转换来源。先匹配固定 revision
的 LFS SHA / Git blob，再逐 key、shape、dtype、位模式验证 checkpoint 转换，派生
统计和配置另核对。本例据此建立 verified_conversion_manifest，不让任意本地清单
冒充上游来源。不因资源放在联接后就放宽 canonical 根检查，使用明确自定义资源根。
冻结 Job 时沿用收据路径原字符串，斜杠混写也可能改变 JSON 摘要；失败另开新 run。

infer_v2 与 infer_v2_5 都叫 IndexTTS2，实际是不同模型入口。核对导入文件而不是类名。
2.5 原版使用路径缓存，不能把其键直接照搬到允许原地替换参考的宿主。

## 参考失败不转成性能结论

早期官方 F32 在参考 W2v 编码报异步 CUDA OOM；BF16 仅改变 GPT，codec/CFM/BigVGAN 仍 F32，
且另出现异步 unknown error。清理 PATH 未证明修复；同步诊断又捕获加载访问冲突。
prepared 不等于生成成功；诊断不能混入热轮或把末尾算子当作根因。单独 W2v 可加载
也不能证明完整实例容量充足。BF16 与 Rust F32 条件不同，不可静默替换为“公平对照”。
Windows venv launcher 根进程的几 MB RSS 不能冒充实际 Python 模型内存，整卡显存
也不等于该模型独占。失败参考如实留证，并继续推进可独立验证的优化。

## 已恢复的官方模式与未恢复模式分开记录

随后将Qwen/MOSS已验证的读取方法用于Index：独立env保留Torch2.8/Transformers4.52.1，
实际导入Safetensors0.8、构造期显式pread并记录w2v文件实际打开调用，官方GPT BF16
其余F32恢复3×(1+5)，18轮真实EOS与PCM成功，内容辅助回读一致。
先查实际python、包__file__及safe_open签名，普通env和python-pread并存时不能凭
另一个目录的metadata推断本次API。Index没有同SF0.8版本mmap控制组，组合成功不能
拆成“pread单独修好了BigVGAN”。原栈失败仍保留；新全F32只通过加载，后在CAMPPlus
BatchNorm/CUDA unknown失败，不能把某个成功精度扩成整个后端通过。

官方触顶可能仅警告后继续解码；参考适配器另看未裁剪GPT输出的实际末尾token8193。
真实1-token负例得到992并在解码前拒绝，父进程失败/清理通过、无成功结果。
BF16热1.462s与Native F32热1.524s只作显式不同条件的模式对照；长度、RNG、int16返回
契约和实际TF32许可标志分开，不以F32存储声明内部计算已匹配，不宣称同条件排名。

## 归因后重新排热点

原官方阶段打印未同步，不能和Rust同步span直接相减。另包原方法边界同步，核验与
正式PCM相同，再把阶段时间作为诊断；嵌套span不能全部求和替代整链基线。
Index本次主要看CFM/vocoder，不能照搬Nano每帧16通道的采样占比。
现有env门控Native profiler可经native_probe_diagnostic_adapter接到xtask reference，
Rust仍拥有进程组/轮数/时限/采集，桥接只翻译请求并原样转发事件，不实现推理。
required_environment必须实测检查：负例在Native子进程前拒绝；结果明确作为诊断，
不能伪装成正式production baseline。详见本页首链的最新同步归因与未完成项。
Snake 常量复用初筛见同一记录最新节：先验证原顺序 exp/recip 预计算的位模式，
再量不同形状。合成激活的局部收益只决定是否继续真实权重实验，不代替完整PCM、
请求状态或整链验收；本次生产路径未改动。

真实BigVGAN跟进改变了结论：三进程长形状出现反向，常量复用暂不采用，临时钩子
撤回；保留外部冻结实验。详见同记录的真实vocoder节。先确认合成算子收益能在真实
权重完整阶段重现，再付出生产状态/整链验收成本；不能将算子百分比直接外推。

DiT跟进验证了优先级：重复RoPE仅约0.05–0.07ms，Transformer/WaveNet范围合计约16ms。
见同记录最新DiT节的设备、权重和范围定义。重复构造易发现但不等于高收益；在缓存或
图复用前先按真实权重归因，说明合成输入、同步扰动及范围内含操作，避免阶段名误导。

WaveNet索引复用跟进也未显示稳定完整阶段收益，冻结实验后撤回，见同记录最新节。
缓存上传成本应放在候选计时内；合成输入须覆盖形状切换。小缓存假设连续不成立时，
及时调整热点或后端优先级，不持续堆积容易实现的局部候选。

F32整模型在CAMPPlus BatchNorm报unknown error，不等于组件固有不支持CUDA。
后续固定同源码/权重/5.2秒参考，保留加载默认重采样、GPU fbank及F32/eval，在两个
独立进程中937张量逐位核对、embedding重复通过。只支持隔离条件可运行，不能
证明旧失败根因或整模型资源足够；下一步转向完整加载/共驻留峰值与首次参考环境。
无音频组件用xtask reference --diagnostic保存数值资产，不生成假的TTS轮次。见
[CAMPPlus隔离](../../../docs/records/indextts/indextts-windows-campplus-isolation-2026-10-25.md)。

同一记录的GPT mmap控制进一步表明：checkpoint backing改变前先核对实际模型state、
非持久buffer和别名。本例456个state与48个非持久buffer通过，但完整F32仍在更早的
semantic attention失败，不能把加载通过、错误位置变化或两次失败的不同提交峰值
当作修复或稳定收益。启动预算应由完整工作负载实测峰值加余量决定；组件低预算
不能沿用到完整模型，旧10GB门禁在此环境不足以保证执行。

meta/assign须处理非持久buffer，不能只验证checkpoint键。本例独立普通构造留下
48个attention mask/标量buffer摘要，重建后与456个state一起核对，未识别meta资源
直接拒绝。完整F32因此进入GPT生成，但仍明确CUDA OOM；加载状态一致、加载完成、
自然EOS及完整PCM分别是不同门禁，不能用前两项代替后两项。

合并“另一平台优化”仍要审计共用fallback：本例main的DiT RoPE更改也作用于Windows
CUDA；BigVGAN组件逐位通过，完整PCM仍变化。恢复Windows CUDA原乘加链后，18轮
与合并前对应轮次逐位一致。生产CFM随机流逐轮演进，比较相同process/round，不能
固定取首轮当全批oracle。新增参数还须做真实正例：命名情绪首次调用即发现标量
Tensor乘向量未广播，普通参数解析测试不能覆盖此错误。见
[合并验收与回退修复](../../../docs/records/indextts/indextts-windows-merged-acceptance-2026-10-25.md)。
