# VoxCPM Windows CUDA：合并 K 复制的运行时验证

Windows 11 / Ryzen 7 9700X / RTX 5070 12GB，驱动616.56，CUDA13，Candle0.11
fork2266099d。源码基点42a7887，含当前Windows工作区修改；Mac/Linux未优化或实测。
证据根 `D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

## 改动与保留理由

沿用[候选调查](voxcpm-windows-attention-candidates-2026-10-25.md)的复制方案：
旧路径先展开K再转置物化，新路径先转置视图再展开，合成一次复制。GEMM维度、
dtype、F32 scores/缩放、softmax及V布局不变，不采用已拒绝的group折入query方案。
运行时仅在Windows CUDA、单query、16/2头、宽128、KV长度2..512启用；singleton、
prefill及其他设备/形状使用参考路径。没有新增公开参数或运行时环境开关。

保留此小范围改动：真实输入局部收益与两种精度的段落完整请求收益均有证据，
没有数值或状态取舍。不要求整链强制改善10%；短句及首PCM的收益未证实，见下表。
这也不是对所有请求的性能非劣保证，短句波动仍是后续需要保留的调查边界。

## 输出与状态

- 192组F32/F16/BF16、batch1/2、连续/视图K/V、长度1..512的布局与context门禁，
  同时逐位比较真正运行时入口与旧路径；packed-runtime-oracle.json及冻结测试exe。
- 按冻结生产UTF-8段界[0,42)/[42,189)和各自SeedPolicy结果，在同一模型先跑完旧路径
  两段，再跑候选两段。Q8为19/62个patch、145920/476160个raw样本；BF16为23/72个
  patch、176640/552960个raw样本。176个latent patch、完整PCM、分块边界及自然EOS
  全部逐位相同；这类连续生成用latent patch验收，不虚构离散token codes。
- 完整模型逐位门禁显式关闭生成图和decoder图，避免同owner比较复用先前录制路径。
  正式生产计时则两臂分别使用默认图设置和独立冻结进程。
- 新候选Backend owner的Q8/BF16测试均通过：unread及满队列取消、非法音色、部分PCM
  后预算错误再关闭且不追加End、同owner恢复；Q8另有参考→默认→参考。
  恢复请求分别保持130560/145920个raw PCM位模式，失败前各46080样本。

packed-runtime-full-{q8,bf16}.json、packed-runtime-owner-state.log分别保存模型与
宿主门禁。两者是不同边界，不把raw长度直接当生产导出长度或播放完成。

## 完整生产请求：ABBAABBA

A为预算修复后的冻结探针89716b79…，B为本候选a7722f5426c03ecca0f048c07d39ea5c44d12362befb1825b45d7de4c602d1f4。
features均为performance-tools/voxcpm/voxcpm-cuda。父工具ce6f4d64…；每case8独立进程，
每臂4进程，每进程1预热+5热轮，串行且不与编译重叠。参考、文本、seed、187patch预算、
dtype及图环境保持一致；Q8用固定acceptance参考cache，BF16用narrator。

| case | 音频秒 | A完整秒 | B完整秒 | 均值改善 | A/B进程均值范围 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| Q8段落 | 12.4 | 2.885413 | 2.733702 | 5.26% | 2.847740–2.908669 / 2.687950–2.810948 | 范围不重叠 |
| BF16段落 | 15.0 | 4.499902 | 4.402010 | 2.18% | 4.448064–4.544576 / 4.369685–4.429738 | 范围不重叠 |
| Q8短句 | 2.4 | 0.617910 | 0.630870 | -2.10% | 0.542871–0.694839 / 0.605080–0.658266 | inconclusive |
| BF16短句 | 2.93 | 0.903309 | 0.885830 | 1.94% | 0.878524–0.927585 / 0.880319–0.898466 | inconclusive |

范围不是置信区间。Q8段落RTF为0.232695→0.220460，BF16为0.299993→0.293467；
因每case音频长度相同，其RTF结论与完整耗时一致。首PCM均值分别为Q8段落
0.120128→0.109425、BF16段落0.042445→0.044783、Q8短句0.123864→0.117199、
BF16短句0.095647→0.093667秒，四项进程范围均有重叠，不宣称首PCM改善。

192轮（含预热）完整输出、段界及1152个产物大小/文件SHA已重新读取校验。
输出摘要的实际域为u32le采样率+u16le声道数+交错F32LE样本，不是WAV文件或裸data块：
Q8短句1aa1afcd…、BF16短句b9d5174a…、Q8段落e0e5edd0…、BF16段落049f97a0…，
均与各自旧生产基线相同。未重新做人耳试听；原质量状态仍pending。

B的Job commit最高Q8约5.870GB、BF16约6.886GB，与A接近；整个生产测量期间采样的
最低系统commit余量约5.02GB。采样500ms，Job峰值是commit而非RSS；整卡显存与
GPU时钟、竞争进程列表另保留，不当逐进程分配器峰值或独占GPU证明。

## 短句波动与采样假设

Q8短句A自身逐热轮约0.527–0.806秒，不能将候选均值慢约13ms直接定为因果回退。
另用同一A二进制、同一请求，ABBAABBA隔离父工具GPU采样on/off，OS采样和native
观测保持：四进程均值为on0.605891、off0.607795秒，范围分别0.587237–0.634856与
0.578647–0.629008，48轮PCM相同。该对照不支持“nvidia-smi导致当前差额”的解释；
不删除原轮次、切换采样条件或挑选最快批次来重命名收益。波动原因尚未定位。

## 证据、复现与未完成项

packed-runtime-production-build.json为xtask构建收据；packed-runtime-tests-build.json及
packed-runtime-owner-build.json冻结两类测试exe。六份改动源文件另复制在
packed-runtime-source/，摘要索引为packed-runtime-source-snapshot.json。
生产计划、Job、顺序及原始进程目录由packed-runtime-abba-schedule.json索引。
packed-runtime-production-summary.json含每进程均值、内存和聚合映射；aggregate仅在副本
重编号process_index，原始results.json不改，产物路径仍归各自原始进程目录。

12份xtask compare覆盖完整耗时/RTF/首PCM，显式使用4进程、要求输出相同、相对0.1%
及绝对1e-6的比较阈值；pass不自动等于改善。四份candidate-report是各case一个B进程的
离线视图，不能代替8进程统计。原始baseline模式与采样对照分开汇总。

```powershell
$bench = 'D:/workspace/talechime-benchmarks/windows-voxcpm-20261024'
$tool = (Get-Content "$bench/budget-recheck-xtask-r2-tool.json" -Raw | ConvertFrom-Json).path
$env:PATH = 'D:/dev-tools/cuda/v13.0/bin/x64;D:/dev-tools/cudnn/v9.14-cuda13/bin;' + $env:PATH
# 新输出目录；默认图环境须与冻结Job一致，不覆盖旧结果。
& $tool perf run --job "$bench/packed-runtime-paragraph-q8-B-job.json" --output "$bench/recheck-packed-q8-run"
```

模型普通测试10 passed/14 ignored；Backend普通测试15 passed/2 ignored，真模型两项
状态测试通过。CUDA release all-targets Clippy及rustdoc检查模型和适配器通过，
CPU无CUDA普通测试10 passed/5 ignored，格式与diff检查通过；日志与最终证据清单
一并保存。既有LNK4098保留，未改变动态CRT。

[官方Python加载](voxcpm-windows-official-loading-2026-10-24.md)的默认入口OOM保持；随后
[加载控制版正式基线](voxcpm-windows-official-baseline-2026-10-25.md)已完成，不能等同默认加载。
[C++对照](voxcpm-windows-paragraph-2026-10-25.md)仍是一次全文、不同RNG
及输出长度，不用本次数字宣称同条件超过官方。真正长篇、接续、style、人工试听和
跨框架质量等价未由本轮关闭。
