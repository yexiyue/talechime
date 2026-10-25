# VoxCPM Windows：官方加载控制版正式基线与阶段差距

本文Native数据固定为a7722f54…的CFM-only阶段；后续
[F32 decoder图默认值验证](voxcpm-windows-f32-decoder-graph-2026-10-25.md)另建二进制及测量，
不追溯改写以下原始对照。

设备Windows / Ryzen 7 9700X / RTX 5070 12GB，驱动616.56。官方checkout固定
f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629且无修改，权重revision
32279effe8c19989596f05d353d1447f51d9e915，主BF16/AudioVAE F32。
证据根 `D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

## 恢复的范围

此前[加载调查](voxcpm-windows-official-loading-2026-10-24.md)的v3外部控制版，本次完成
3独立进程、各1预热+5热轮。依然使用meta主模型及VAE、逐项assignment和原顺序RoPE
重建、Safetensors0.8+windows.headerio1/pread；正式基线未修改官方forward。
每进程再次验证577主张量、312个VAE状态和9个非持久buffer精确相同，max_length8192
及KV上限不变。没有把原默认from_local加载改称成功，也未替换仓库内仍对应v2的helper。

用户释放内存后，原12GB系统可用commit门禁在三个进程前分别观察到
12.014/12.016/12.008GB，均通过。Job峰值10.206–10.227GB，整个采样期间系统commit
余量最低1.695GB。没有降低12GB控制版或20GB默认加载预算，没有修改pagefile。
预检仍只是快照；Job峰值是子树commit，不是RSS，venv launcher的几MB RSS不代表模型。

环境为Python3.11.13、Torch2.8.0+cu128、Transformers4.52.1，8个Torch线程，interop1。
实际optimize=False。另调用固定官方类的optimize方法检查依赖分支：当前无Triton，
在最小cuda owner上捕获“torch.compile disabled - triton is not installed”，返回对象
及属性未变。这只证明依赖拒绝分支，不是optimize=True完整模型或已编译路径的实测。

## 正式无参考短句

文本/输入SHA沿用short.txt，seed7330414270409262323，steps10、cfg2、frame_limit187，
retry关闭；无参考narrator，非克隆。每个请求检查stop_head=1、19个流式块、145920个
48kHz单声道样本（3.04秒），不是碰到frame limit当成功。

| 官方进程 | 热合成均值秒 | 热首PCM均值秒 | 加载秒 | 预热秒 / 预热首PCM秒 |
| --- | --- | --- | --- | --- |
| 0 | 2.686830 | 0.131956 | 8.423791 | 4.356669 / 1.677749 |
| 1 | 2.635582 | 0.208076 | 4.617146 | 3.480174 / 0.933008 |
| 2 | 2.537324 | 0.153463 | 4.472579 | 3.558691 / 0.809060 |

进程均值再平均为2.619912秒，首PCM0.164498秒，RTF0.861813。
加载时间含逐项回读核验，不能当默认加载速度；保留所有预热成本及波动。
18轮WAV文件大小/SHA、解码F32 PCM及自然EOS重新校验，3个预热PCM相同、15个热轮
PCM相同；预热与热轮仍存在此前记录的微小位差，原因未定位，不删除首轮。

当前Native冻结探针a7722f54…在相同无参考BF16模式的对照数据如下。两条路径分批运行，
不是跨框架ABBA；独立RNG及浮点实现、交付边界不同，未证明跨框架音色/质量等价。

| 路径 | 进程/热轮 | 完整秒 | 音频秒 | RTF | 首PCM秒 |
| --- | --- | --- | --- | --- | --- |
| 官方未编译推理、v3加载控制 | 3 / 各5 | 2.619912 | 3.04 raw | 0.861813 | 0.164498 raw流式 |
| Native当前生产默认CFM图 | 4 / 各5 | 0.873199 | 2.93宿主输出 | 0.298020 | 0.094202 consumer |
| Native同二进制关闭CFM图 | 4 / 各5 | 1.304798 | 2.93宿主输出 | 0.445324 | 0.142845 consumer |

此case的Native完整耗时/RTF低于跑通的官方控制版，不能扩大为官方默认加载、已编译
路径、其他文本/参考模式或质量已经达标。Native的first PCM经过宿主流，不与官方raw
首块视作完全相同边界。Native模型原始输出同样145920样本/19patch；独立测试把这些
样本送入真实core::audio::BoundarySilence，四次均得到140640样本，恰为生产2.93秒。
因此长度差来自已验证的宿主边界处理，没有通过少生成patch取得该时间数值。

## 阶段归因和单变量检验

官方另跑1预热+1热轮的同步模块诊断，热PCM与正式热轮逐位一致。选中模块独占时间
合计2.630秒，外层诊断2.649秒；这不是新的正式基线。Native新ignored测试先运行两次
默认图，再运行两次measure_modules=true，四次raw PCM逐位一致并自然EOS。
该选项明确禁用图launch；缓存仍显示active，不能用“图存在”推断本次在replay。

| 同步归因 | 官方热轮秒 / 调用数 | Native两次eager范围秒 | 可比边界 |
| --- | --- | --- | --- |
| CFM | 1.580173 / 19 | 0.755018–0.897380 | 各19patch；非生产图时间 |
| base decode | 0.572518 / 18 | 0.166272–0.180378 | 单步主干 |
| residual decode | 0.151681 / 18 | 0.038887–0.041348 | 单步残差 |
| local encoder | 0.182555 / 20 | 0.069449–0.077726 | 官方含prefill及最终帧；Native此计时不含prefill且EOS后不再编码 |
| 流式VAE decoder | 0.093063 / 19 | 0.193397–0.219735 | 各F32新latent块；下一优化假设 |

同步诊断改变执行调度，不能把这些阶段差额相加宣称正式差距已全部解释，也不能把
Native eager CFM时间冒充默认图成本。它表明官方CFM是主要成本，Native F32 VAE相对
仍偏慢；后续优先验证原始F32 VAE的图回放，而非直接扩大默认开关。

以当前生产二进制做CFM开关ABBAABBA，各进程1+5，48轮PCM与既有b9d5174a…摘要
完全一致且所有产物重新校验。关闭图的四进程均值为1.236840/1.219799/1.559026/1.203526秒；
默认为0.878344/0.877654/0.890498/0.846298秒，范围不重叠，平均改善33.08%。
decoder默认在两臂都关闭，仅CFM环境值0/未设置不同；保留这个实际条件差异，按明确
单变量实验汇总。本轮是对既有图收益的补验，没有新增生产运行时优化。

## 摘要、质量和复现

官方输出SHA域是裸F32LE样本，Native输出SHA还前缀u32le采样率及u16le声道数，不能
直接相等比较。官方WAV包含PEAK时间戳：正式热轮1/2的data完全相同，PEAK除时间戳
外也相同，文件SHA仍不同。文件SHA用于资产身份，解码PCM用于数值；不改写原WAV。
新官方PCM与既有ASR回读的v2预热/热轮逐位相同，Native48轮也与已回读的Native PCM
相同，复用已有“你好欢迎收听今天的故事”的辅助识别证据。没有新增人工试听或宣称
跨框架音质通过，quality/listening仍pending。

正式结果及摘要：official-meta-stream-v3-baseline-recovered-run、
official-meta-stream-v3-baseline-recovered-summary.json；诊断及阶段计数：
official-meta-stream-v3-stage-diagnostic-run、voxcpm_official_stage_timing.py；
Native图对照：bf16-cfm-control-{schedule,summary}.json；
Native诊断：bf16-module-attribution-boundary.{json,log}及-build.json；
ASR/PEAK/48轮产物复核：official-baseline-readback-reuse.json。

```powershell
$bench = 'D:/workspace/talechime-benchmarks/windows-voxcpm-20261024'
$tool = (Get-Content "$bench/budget-recheck-xtask-r2-tool.json" -Raw | ConvertFrom-Json).path
$env:PATH = 'D:/dev-tools/cuda/v13.0/bin/x64;' + $env:PATH
& $tool perf reference --job "$bench/official-meta-stream-v3-baseline-job.json" --output "$bench/official-recheck-NEW"
# 使用冻结ignored测试exe，先设置TALECHIME_VOX_PROFILE_MODEL/TEXT/OUTPUT；OUTPUT须为新文件。
cargo test --locked --release -p talechime-voxcpm --features cuda real_bf16_module_attribution -j1 -- --ignored --nocapture --test-threads=1
```

新增代码仅为Windows CUDA ignored诊断，不进入生产。普通CUDA测试10 passed/15 ignored，
真模型诊断通过，release all-targets Clippy、rustdoc与格式/diff检查通过，LNK4098既有警告保留。
默认官方加载、compiled路径、长篇/接续/style/参考音色质量与完整多模式验收仍未完成。
