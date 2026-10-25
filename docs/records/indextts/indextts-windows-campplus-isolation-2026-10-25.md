# IndexTTS 全 F32 对照：CAMPPlus 隔离验证

固定参考音频、官方源码与权重下，CAMPPlus独立F32 CUDA执行成功；全模型F32
仍没有有效热轮，未宣称旧错误根因或完整路径已修复。本轮不改Rust推理、精度或
BatchNorm数学。Windows11 / Ryzen7 9700X / RTX5070，外部证据根
`D:/workspace/talechime-benchmarks/windows-indextts-20261024`。

## 为什么不直接重启全模型

前次official-pread-f32-diagnostic-run确实加载完成，GPU allocated6,952,131,584字节，
随后参考CAMPPlus BatchNorm报CUDA unknown error，无有效GPT codes或合成轮次。
Job提交峰值13,804,675,072字节。当前系统可用commit约11.70GB，不假定原样重试可
恢复，不把BF16结果冒充F32。先验证同输入组件，区分组件本身与整模型共驻留条件。

## 身份与执行

官方checkout HEAD `d9e41aac89fd00b3d71497fddb287b7f24613712`，已跟踪源码未改，
checkpoint/config及reference/runs为既有未跟踪资源；实际源文件身份以job逐文件SHA
为准。Torch/Torchaudio2.8.0+cu128，沿用原python-pread隔离环境与offline配置，
CUDA_LAUNCH_BLOCKING=1。CAMPPlus参数全部CUDA F32、eval；cudnn.allow_tf32=True，
matmul.allow_tf32=False。未改变这些选项，存储F32不扩大为每个卷积kernel精度证明。

CAMPPlus checkpoint SHA
`3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8`；
参考WAV SHA `8d1162cd924576eaf21a95b87000b0d515df39deea0c760069a2ca5909dfbc71`，
与旧失败任务相同，不缩短输入。直接使用官方_load_and_cut_audio方法的librosa默认
22,050Hz加载及15秒上限，再原样Torchaudio CPU重采样到16kHz，83,200样本，5.2秒。
loader rate不是未经重采样的文件头采样率。GPU Kaldi fbank80维、dither0、均值归一，
输入(1,518,80)，与原infer参考处理的调用参数相同，不先跑semantic/CFM等其他模型。

937个参数及buffer与checkpoint的CPU位模式逐一相同，包括整数计数器。首次诊断
在校验0维Long buffer的view(Byte)时失败，尚未做CAMPPlus forward；该失败不是旧
BatchNorm错误复现。仅将校验临时副本reshape(-1)后读字节，未改模型张量或类型，
保留首次失败目录campplus-isolated-f32-run。

v2、v3两个独立新进程均完成1次预热+2次组件调用，输出(1,192) embedding，有限值，
三次逐位相同；跨两个进程的features.npy及embedding.npy完整字节也相同。
这些为组件诊断内部重复，不是正式TTS的1+5热轮矩阵，不输出RTF/首PCM。
初次预热0.3699/0.1679秒，后两次0.0468/0.0222及0.0482/0.0462秒，仅保留观测，
不据此排名或归因抖动。

两进程Job提交峰值2,814,005,248 / 2,813,644,800字节，cleanup通过。最终v3启动
门禁4GB，预检可用11,695,681,536字节。与全模型13.8GB不是同一工作负载，不能
计算“内存优化百分比”。旧全模型10GB启动门禁未降低；隔离组件有自己的预算。

## xtask 非音频诊断模式

新增`perf reference --diagnostic`，仍使用已有reference job、输入/可执行文件SHA、
Windows Job containment、超时/取消、commit预检及清理审计。协议warmup_rounds=0、
hot_rounds=1表示每进程一次诊断调用；内部重复次数由组件自己记录。复用现有
observation事件的diagnostic_result载荷，包含结构化summary及diagnostic_*资产，
核对SHA/字节数；不扩展事件kind或产品JSON Lines版本。

该模式禁止round事件，要求唯一诊断结果后有success terminal；普通reference仍
要求真实音频轮次与完整覆盖。根kind为external_diagnostic_not_tts_performance，
没有rounds/RTF，listening_acceptance为not_applicable。不能伪造静音WAV凑成功轮次。
单元测试包括组件资产正例、篡改资产、普通测量拒绝无音频诊断、诊断拒绝round及
terminal-only、原始EventSequence验证。60通过1忽略，Clippy all-targets及rustdoc
-D warnings通过；旧正常音频参考测试保持通过。

最终工具SHA `08cafd5a641a321ad0179cf3d7da13a63e3f022f19c9f3e25926d288713252c1`。
v2工具及其原始输出保留，v3仅改诊断试听状态为不适用、loader-rate字段名，输入
和模型数学不变。原始成功/失败、工具版本、代码冻结与资产核对见
campplus-isolated-f32-audit.json及各run目录。复现使用新输出目录：

```powershell
& $xtask perf reference --diagnostic --job D:/workspace/talechime-benchmarks/windows-indextts-20261024/campplus-isolated-f32-v3-job.json --output NEW_COMPONENT_OUTPUT
```

## 下一步边界

### GPT mmap 加载控制的实际结果

进一步`gpt-meta-loading-diagnostic-run`保留独立普通构造审计为oracle，在meta设备
构造同一个UnifiedVoice，mmap checkpoint后assign state。仅重建源码明确的24层
attention causal bool mask及masked_bias，共48个非持久buffer，每项与普通构造的
shape/dtype/位摘要匹配；未处理的meta buffer直接失败。456个state的shape/dtype/
位摘要与普通构造完整审计文件一致，无missing/unexpected及参数对象别名变化。
CPU组件Job峰值5,476,372,480字节，cleanup通过，非正式内存性能矩阵。

`official-f32-gpt-meta-control-run`采用该加载控制，完整加载6.4672秒，模型GPU
allocated仍6,952,131,584字节，实际恢复48个非持久buffer。首次请求通过参考处理，
进入GPT beam search，在logits top-k报明确CUDA out of memory。Job峰值
12,253,589,504字节，cleanup通过。没有GPT自然EOS或PCM，不构成有效TTS基线；
单次加载时间不作稳定提速声明。错误栈infer_v2_5.py:767明确晚于原636/645参考
步骤，但不能证明此前unknown error的根因，亦不证明完整F32现已能驻留生成。
工具仍为冻结的08cafd5a版本，输入、adapter及原始运行证据单独留存。

后续`gpt-mmap-loading-diagnostic-run`通过独立CPU进程检查：官方UnifiedVoice普通
构造后，用`torch.load(mmap=True, weights_only=True)`读checkpoint，再沿用
`load_state_dict(strict=False)`复制。456个state张量全部逐位相同，无missing或
unexpected键；48个非持久attention buffer在加载前后摘要相同，参数对象别名集合
为空且未改变。保留普通初始化，没有meta构造或assign加载，不改变推理数学。
此证据只验证CPU加载状态；未将其扩大为完整GPU输出一致或加载性能收益。

随后`official-f32-gpt-mmap-control-run`仅对GPT的torch.load启用mmap，其余官方
构造、pread semantic checkpoint、F32及输入不变。GPU加载后allocated仍为
6,952,131,584字节，完整加载12.3945秒。首次参考在Wav2Vec2-BERT attention的
matmul/缩放处报CUDA unknown error，源码infer_v2_5.py:636，早于CAMPPlus所在645行。
没有成功轮次、GPT codes或PCM；与旧CAMPPlus错误位置不同，不代表跨过该阶段。
Job提交峰值12,275,892,224字节，cleanup通过，逐条复算采样系统剩余commit最低
25,989,120字节；最后一个样本为509,235,200字节。两个失败运行的峰值
不能充当稳定内存收益对照，更不能据此确定unknown error根因。

此次仍使用旧10GB启动门禁，实测表明其不能保证完整F32运行所需空间。后续相同
驻留模式至少以已观察到的13.805GB Job峰值加余量规划预算，或先验证更低峰值
加载方案；当前条件不再原样重试。不降低精度或缩短参考来冒充完整F32对照。

补充仅使用 ZIP central directory 的 checkpoint 存储清单：`gpt.pth` 有456段
storage，总计3,259,397,840字节；codec为607,000,716字节，s2mel为414,806,848字节。
清单不反序列化pickle、不创建模型、不运行GPU，见外部
`checkpoint-storage-inventory.json`。官方`load_checkpoint`先完整`torch.load`
到CPU，再`load_state_dict`复制进已经构造的模型，返回后才将GPT迁移GPU。
这是下一轮检查加载峰值的具体入口，但storage总量不是模型驻留或系统commit峰值，
不能直接相加推断OOM。mmap/逐张量加载若作为控制，须先验证参数、buffer、别名和
缺失键的初始化语义，再测实际峰值；不直接修改官方源码或生产精度。

原始成功/失败目录、最终工具及源码快照已归档为外部
`campplus-isolation-final-evidence.json`（109个文件）。该清单固定的是上述组件
实验的交付状态；此处后补的存储量分析不回写已冻结源码快照。

隔离成功不支持“这个输入/模块在任何F32 CUDA条件都不能工作”的判断，但不能证明
旧整模型失败来自内存。完整驻留还多了模型、算子顺序、分配器状态与动态库环境。
下一步应量化加载峰值、首次参考前后的系统commit/GPU分配及实际模块身份，必要时
先验证更低峰值加载控制的权重和非持久buffer，再恢复完整官方F32基线。不要将
隔离embedding成本与混合精度热轮相加，或据此宣布Native达到/超过官方。
