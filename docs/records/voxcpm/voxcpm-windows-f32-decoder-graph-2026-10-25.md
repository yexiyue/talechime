# VoxCPM Windows BF16：F32 AudioVAE 图回放

Windows / Ryzen 7 9700X / RTX 5070 12GB，驱动616.56，CUDA13/SM120，Candle0.11
fork2266099d。源码基点42a7887，含本轮工作区修改。证据根
`D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

## 结论和启用范围

保留原始F32 AudioVAE decoder图：Windows BF16原始模型默认启用。既有有状态回放实现
未改算术，仅扩大经过验证的默认使用范围；Q8的F16图策略不变，其他平台和原始主模型
dtype仍沿用此前策略。可以用既有VOXCPM_DECODER_GRAPH=0关闭decoder图。

方向来自[官方阶段对照](voxcpm-windows-official-baseline-2026-10-25.md)：Native同步eager
F32 decoder约0.19–0.22秒，官方约0.09秒。没有将measure_modules=true下的eager计时
当默认图耗时，也没有直接把F16已通过的结论外推到F32。

## 数值和生命周期门禁

1. 原始audiovae.pth/F32真实codec：3次请求各8帧、请求重置、取消前不消费帧、回放后
   取消及重置恢复，共33对完整解码输出逐位一致；明确断言图active，排除静默fallback。
   同一辅助函数的F16旧权重路径另做33对回归并通过，F32输入直接由F32夹具生成。
2. 同一BF16完整模型按eager/eager/graph/graph/eager运行，CFM图保持开启。每请求
   19个latent patch、145920个raw PCM样本及分块长度与eager基线逐位一致，自然EOS；
   graph阶段断言decoder active，返回eager后恢复一致。该测试使用Model内部捕获边界，
   不能替代公开Backend所有权验收。
3. 候选默认值的新Backend测试二进制通过Q8/BF16 unread取消、满队列取消、入口拒绝、
   预算失败后关闭且不追加End、同owner复用；Q8还通过参考→默认→参考。BF16恢复PCM
   145920样本，Q8为130560，失败前均46080样本。
4. Qwen/VoxCPM CUDA组合实际链接后，29项普通测试通过；四个Qwen变体及两个VoxCPM
   变体的6项真实模型owner测试全部通过，包含取消及预算失败后的PCM恢复；
   Qwen CustomVoice及Vox Q8另覆盖音色A/B/A切换。
   这是组合正确性回归，不是Qwen性能复测。

组件、完整模型、公开owner以及组合测试分别保存日志和冻结exe，不用单一窄测试
替代全部门禁。原始权重revision32279effe8c19989596f05d353d1447f51d9e915，主模型BF16、
codec F32保持，不改变采样、预算、内容或结束条件。

## 默认值的正式ABBAABBA

A冻结探针a7722f5426c03ecca0f048c07d39ea5c44d12362befb1825b45d7de4c602d1f4，
已含CFM图及K复制；B为232b9ceef76f2ad0517fcaaa53f21b1d4db7906629859aa58af0a13c0abad63a。
features均performance-tools/voxcpm/voxcpm-cuda。两臂环境中的图开关都未设置，唯一生产
实现变化为Windows BF16默认启用decoder图，使用新旧二进制比较，未重命名环境条件。
每case8独立进程、每臂4个，各1预热+5热轮，串行且不与编译/其他模型推理重叠。

| 无参考narrator | A完整秒 | B完整秒 | 改善 | A/B进程均值范围秒 | 音频秒 | A→B RTF |
| --- | --- | --- | --- | --- | --- | --- |
| 短句 | 0.879830 | 0.692183 | 21.33% | 0.869252–0.893775 / 0.635857–0.742175 | 2.93 | 0.300283→0.236240 |
| 63字段落 | 4.430081 | 3.248120 | 26.68% | 4.391009–4.461871 / 3.133556–3.291276 | 15.0 | 0.295339→0.216541 |

| 首PCM | A均值ms / 进程范围 | B均值ms / 进程范围 |
| --- | --- | --- |
| 短句 | 94.768 / 92.271–96.675 | 73.261 / 67.617–78.028 |
| 段落 | 42.262 / 40.713–44.190 | 33.061 / 31.781–34.040 |

完整耗时、RTF及首PCM的进程均值范围均不重叠；范围不是置信区间。96轮（含预热）
PCM、长度与生产段界相同，576项资产大小/文件SHA重新验证。短句摘要b9d5174a…，
段落049f97a0…；域为u32le采样率+u16le声道数+交错F32LE样本。段落仍按冻结范围
[0,42)/[42,189)及各自seed执行，没有变为一次全文或更早停止。

B Job commit峰值最高约6.8955GB，A约6.8866GB；正式测量采样的系统commit余量最低
约5.05GB。Job高水位不是RSS；500ms设备采样及竞争进程明细保留，不声称完全独占GPU。
加载A约2.55–2.76秒，B约2.58–2.76秒；另记录引擎短句预热A约0.735–0.842秒、
B约0.590–0.696秒。字段reference_warmup_seconds在此无参考模式并非纯参考编码。
不删除预热成本，也不把本轮增量与历史CFM/K复制百分比相加。

## 图开关的归因边界

VOXCPM_CUDA_GRAPH=0会经共享enabled判断同时关闭CFM及decoder，不是只关CFM。
此前BF16 decoder默认关闭时的33.08%对照确实只改变CFM；新默认值下，若要重新隔离
CFM，须将两臂decoder显式固定为0。不要只复用旧Job标题，就宣称干预仍是单模块。
本轮比较两臂开关均未设置，Source/二进制收据记录了默认策略变化。

## 工程、复现和剩余范围

普通CUDA模型测试10 passed/17 ignored，无CUDA普通测试10 passed/5 ignored；
Backend15 passed/2 ignored；CUDA release all-targets
Clippy及rustdoc分别检查模型/适配器，格式和diff门禁通过。组合实际Candle core/nn/
transformers features为cuda/default（core另有cudarc），源码统一同一fork，无新增原生依赖
或CRT设置。既有LNK4098保留。

f32-decoder-production-build.json为生产构建收据；f32-decoder-final-{model,owner}-build.json
冻结测试，完整模型数据在f32-decoder-final-model-gate.json；组件和owner日志分开。
f32-decoder-source/保存改动时模型包源码、Cargo文件；f32-decoder-abba-schedule.json
索引16个原始进程，f32-decoder-production-summary.json保留逐轮/进程统计与聚合重编号映射。
6份xtask比较（两case各完整耗时/RTF/首PCM）要求4进程且输出一致；两份candidate-report
仅是每case一个B进程的离线视图。聚合副本不覆盖原始results，产物仍属于原进程目录。

```powershell
$bench = 'D:/workspace/talechime-benchmarks/windows-voxcpm-20261024'
$tool = (Get-Content "$bench/budget-recheck-xtask-r2-tool.json" -Raw | ConvertFrom-Json).path
$env:PATH = 'D:/dev-tools/cuda/v13.0/bin/x64;' + $env:PATH
& $tool perf run --job "$bench/f32-decoder-short-bf16-B-job.json" --output "$bench/f32-decoder-recheck-NEW"
# 组件门禁另设VOXCPM_TEST_ORIGINAL_MODELS；普通测试不加载模型。
cargo test --locked --release -p talechime-voxcpm --features cuda original_f32_decoder_replay_preserves_state_and_reset -j1 -- --ignored --nocapture --test-threads=1
```

未新增官方对照运行；此前官方加载控制版和C++各自条件差异仍保留。此改动保持既有PCM，
未新增人工试听或宣称跨框架音质通过。BF16参考/接续/style及更长负载未新增完整性能
矩阵；默认官方加载仍受阻，Windows总目标保持进行中。
