# Windows 后端优化清单（2026-10-24，进行中）

范围为 [issue #5](https://github.com/yexiyue/talechime/issues/5) 的 Windows RTX 5070 路径。
本机 Ryzen 7 9700X / RTX 5070 12 GB，Mac 由另一台电脑负责。历史记录只作为起点，
不等于当前 main 的新验收。按用户最新指令，OmniVoice 调整为首位；不再使用 issue
旧文及原 goal 中的整链强制 10% 门槛。模型以 GPU 为主，CPU 仅保留有价值的阶段调查。

| 后端/变体 | 本轮状态 | 可追溯证据与下一步 |
| --- | --- | --- |
| OmniVoice 0.6B CUDA | 进行中；保序 decoder 已保留局部收益，参考音色尚未严格达标 | [本轮报告](../omnivoice/omnivoice-windows-cuda-2026-10-24.md)：既有图验收保留；新增最重 deconv 约 3.0→0.43–0.49 ms，完整 codes/PCM 一致。增量整链观察 -3.46% 但范围重叠；官方复测亦有波动。补后处理与其他负载，试听 pending。 |
| MOSS Nano Candle | 保留embedding复用、CUDA全可见mask省略及forward内交错RoPE共享；质量/接续负载待验收 | [最新RoPE同期对照](../moss/moss-nano-windows-forward-rope-2026-10-25.md)：短句0.577→0.464s（−19.45%）、段落2.609→2.230s（−14.54%），范围不重叠；短句首PCM−21.02%，段落首PCM仍inconclusive。三seed codes/raw PCM/EOS、旧三个消费者及取消/满队列复用通过，96轮PCM与576资产/源范围/chunk边界保持，短句/段落ASR完整。[此前记录](../moss/moss-windows-performance-2026-10-24.md)的embedding局部53.9→5.6ms和mask−5.85%为独立历史实验，不相加；宽表RoPE与CPU排序仍撤回。官方工作量/RNG与听感差异未关闭。 |
| IndexTTS 2.5 | 参考复用保留；官方BF16 GPU正式基线恢复，同条件达标未证明 | [当前调查](../indextts/indextts-windows-production-2026-10-24.md)：原缓存ABBA−63.72%；当前Rust F32约1.524s/RTF0.588，官方GPT BF16其余F32约1.462s/0.531，精度/长度/输出契约不同。SF0.8+pread后18轮官方成功、EOS正负验证、两边回读完整；新环境全F32在CAMPPlus BatchNorm报CUDA unknown，未排名。同步归因下一步看CFM/vocoder。 新增[官方CAMPPlus F32隔离](../indextts/indextts-windows-campplus-isolation-2026-10-25.md)双进程成功，937参数/buffer及特征/embedding一致；仍无全模型F32热基线，未将隔离成功或内存差额当作根因证明。 |
| CosyVoice3 0.5B | 当前生产资源复核仅 HiFT 不匹配，未运行生产基线 | [10-24 复核](../cosyvoice3/cosyvoice3-windows-production-2026-10-24.md)：6/7 文件匹配；三种 CPU dispatch 重转换均与本机 246 张量逐位一致，仍未复现生产 SHA。保留门禁，待 canonical 对照。 |
| MOSS Local / Realtime 1.7B | 保留Windows CUDA RoPE表复用；原生状态门禁通过，官方常驻热基线未恢复 | [RoPE同期对照](../moss/moss-windows-forward-rope-reuse-2026-10-25.md)：Realtime1.599→1.206s（−24.59%，范围不重叠）；Local2.817→2.699s（−4.19%，范围重叠）；两个首PCM仍inconclusive。96轮PCM、UTF-8/chunk边界及480资产验证，Local/Realtime/VoiceDesign真实codes回归、取消/失败/克隆/接续owner门禁通过。官方默认F32 codec已修正：[Local](../moss/moss-local-windows-official-staged-2026-10-25.md)/[Realtime](../moss/moss-realtime-windows-official-staged-2026-10-25.md)分阶段GPU均输出稳定语音及完整ASR，含换载而非热基线；Local默认参数仍超独立显存。风格、多模式计时、推荐栈及人工试听待补。 |
| MOSS Nano ONNX | 完整新ORT专用构建中CUDA生产路径通过；默认分发未更换 | [provider实测](../moss/moss-nano-onnx-windows-2026-10-24.md)：动态Native Registry 3×(1+5)均值0.694s/音频2.8s，18轮PCM一致，取消/失败/换音色复用通过；实际DLL模块身份已核对。官方CUDA1.159s/3.28s不可直接排名，DirectML实验亦保留；标准Native动态收据已通过，新增3×(1+5)均值0.767s、18轮PCM一致（不同批次不作提速比较）；[主CLI动态入口](windows-ort-cli-dynamic-2026-10-25.md)已补齐并以暂存ORT1.28三库完成真实静音播放/checkpoint；标准发行不变，正式分发打包、同条件对照及试听待补。 |
| Qwen 主要已接入变体 | CustomVoice与0.6B Base双方基线完成，1.7B默认图组合收益已复验；质量/完整条件待验证 | [本轮报告](../qwen/qwen-windows-production-2026-10-24.md)：0.6B/1.7B原生1.114/1.650s、官方SDPA5.411/4.221s，输出长度不同不直接排名。1.7B ABBAABBA关闭图3.278→默认图1.672s，约−49%，48轮PCM一致。官方pread对照及EOS门禁正反例通过；新增0.6B Base Native1.299s/音频2.52s、官方3.262s/2.56s，各18轮稳定、抽样回读完整，四个narration变体CUDA owner取消/满队列/拒绝/预算失败后逐位复用通过，CustomVoice06/17还通过uncle_fu→serena→uncle_fu切换；Custom17 style及Base06/17单段接续取消后逐位恢复已补齐；[VoiceDesign](../qwen/qwen-voicedesign-windows-2026-10-25.md)生产导出/18轮调用计时及官方热基线完成，冷对照矩阵不对称，未宣称达标。Base跨静态参考A/B/A、取消恢复及[新owner磁盘prompt重载](../qwen/qwen-windows-prompt-cache-owner-2026-10-25.md)已补齐；[三段接续逐位重放/中段取消恢复](../qwen/qwen-windows-continuation-chain-2026-10-25.md)已补齐；同进程末尾Custom17加载OOM，独立进程复查通过，未改写整组失败。长链漂移、克隆质量等价与试听待补。 |
| VoxCPM2 | Q8/BF16生产及owner状态、官方加载控制版无参考对照、Windows BF16 F32 decoder图通过；多模式质量待补 | [本轮报告](../voxcpm/voxcpm-windows-production-2026-10-24.md)：参考缓存、CFM图及K复制保留；[F32 decoder增量](../voxcpm/voxcpm-windows-f32-decoder-graph-2026-10-25.md)短句0.880→0.692s、段落4.430→3.248s（−21.33%/−26.68%），96轮PCM保持，Qwen四变体/Vox两变体组合状态通过。[官方控制版](../voxcpm/voxcpm-windows-official-baseline-2026-10-25.md)无参考3×(1+5)为2.620s/RTF0.862，明确加载控制及未编译条件。默认官方初始化OOM、C++不同RNG/段界、长篇/接续/style与跨框架听感仍待补。 |

## 复测入口与资产

原始音频和结果在仓库外，以各后端报告的绝对路径为准；本轮主要位于
`D:/workspace/talechime-benchmarks`，部分诊断位于 `~/.talechime/benchmarks`。模型位于
`~/.talechime/resources`。本次 OmniVoice 已冻结 build receipt 与对应 SHA 路径的 exe，
不能用 Cargo 的可变产物覆盖 A/B。以下命令读取已有冻结 Job，输出必须是全新目录；
首次复现需先核对记录中的机器、资源与输入路径，不能把本机收据拷到其他设备冒充验收。

```powershell
$bench = "$env:USERPROFILE/.talechime/benchmarks"
cargo xtask perf run --job "$bench/windows-omni-forward-graph-abba-1-graph-job.json" --output "$bench/omni-recheck-NEW"
cargo xtask perf reference --job "$bench/windows-omni-official-matched-text-frozen-job.json" --output "$bench/omni-official-recheck-NEW"
```

先编译再串行测量，不与推理并行编译。恢复旧运行前校验其二进制、源码、权重和输入
身份；代码变化时新建 receipt/Job/输出目录。2026-10-24 已将 `target` 从 C 盘
迁回项目 D 盘真实目录，110185 个普通文件 SHA-256 与内部 reparse 链接校验通过。
C 盘旧副本删除被工具策略拒绝，仍保留；迁移日志及校验摘要位于
`D:/workspace/talechime-target-restore.log` 与`D:/workspace/talechime-target-restore-verification.json`。
目录内存在历史模型和结果，不能整体删除或盲目 cargo clean；只清理已核验可再生目录。

优化 skill 持续迭代于 [candle-performance](../../../skills/candle-performance/SKILL.md)，
本轮已将 Qwen 图录制经验用于 OmniVoice，并补充真实布局回放、输出复制、录图后取消
及官方分段/seed 对齐的实证。此清单保持未完成项，不将本轮单案例收益替代全部后端目标。


## Qwen 参考前缀图后续验证

[Base06/Base17 参考前缀图](../qwen/qwen-decoder-reference-graph-windows-2026-10-24.md)
完成 ABBAABBA：完整请求耗时下降 13.91%/13.00%，各 48 次 PCM 保持；独立真实
codes/raw PCM/自然 EOS 对照及四变体 owner 子集通过。代价为约 1.2–1.4 GB
额外 Job commit，1.7B 系统 commit 余量曾低至 342 MB；人工试听与其他模式仍 pending。
当前竞争 GPU 进程采集编码修复单列于
[Windows 采集调查](windows-gpu-process-encoding-2026-10-24.md)，不能追溯修复旧轮次的未知竞争情况。


VoxCPM [owner 状态补验](../voxcpm/voxcpm-windows-owner-state-2026-10-24.md)在 Q8/BF16
分别验证完整 PCM 恢复，Q8 另有参考→默认→参考切换；也定位并修复 max_duration 把
16 kHz 输入 patch 样本数误用为 48 kHz 输出单位的问题。现 1 秒预算为 6 patch=0.96 秒，
默认 187 patch=29.92 秒，向下取整遵守 30 秒限制。该行为修复不计作性能收益；
历史 200 patch 测量保持原配置，当前构建已各完成 3×(1+5)：Q8 0.6432s/首PCM0.1235s，BF16
0.9613s/0.0944s，各18轮PCM与旧生产基线一致；新父工具采集和离线report门禁通过。
批次差额不归因预算修复，官方/质量条件的未完成项保持。


VoxCPM 官方 [实验加载控制](../voxcpm/voxcpm-windows-official-loading-2026-10-24.md)
恢复首个完整流式请求及一个热轮，原类/权重和9个非持久buffer精确核验，ASR内容完整。
其早期加载入口与Safetensors偏离默认，曾内存仅余335MB、正式预检不足。随后用户释放
内存，[v3正式无参考基线](../voxcpm/voxcpm-windows-official-baseline-2026-10-25.md)三进程均
通过原12GB门禁，最低余量约1.70GB；只恢复控制版推理对照，默认官方加载、试听及质量
等价状态不变。


10-25 [VoxCPM段落验证](../voxcpm/voxcpm-windows-paragraph-2026-10-25.md)新增Q8/BF16/C++
各18轮完整请求及两种Rust模型的默认/eager逐位对照；同后端输出稳定。Q8和C++
按音频时长归一后的进程RTF范围重叠，不扩大为通用胜出声明。63字段落不替代长篇、
接续或style；官方内存与人工质量门禁保持未完成。


VoxCPM [单query候选调查](../voxcpm/voxcpm-windows-attention-candidates-2026-10-25.md)
拒绝改变GEMM行数的GQA原型，并补正63字段落的内部两次请求与C++一次全文差异。
随后[K复制运行时验证](../voxcpm/voxcpm-windows-key-packing-2026-10-25.md)完成192组算子、
176个真实latent patch/PCM、状态复用及4case各ABBAABBA：Q8/BF16段落改善约5.26%/2.18%，
范围不重叠；两种短句及首PCM收益inconclusive，192轮生产PCM一致。单独的采样on/off
对照未支持用nvidia-smi解释短句波动。官方/质量/长篇模式的未完成项保持。

## 10-25 main合并

已合入main至1746d72，保留两端优化；Candle统一至884b359。原生组合构建、274普通测试、FIR旧新24份输出及IndexTTS缓存真模型回归通过，详见[合并验证](windows-main-merge-2026-10-25.md)。此前性能数字仍属于各自冻结版本，不重命名为合并后性能。
