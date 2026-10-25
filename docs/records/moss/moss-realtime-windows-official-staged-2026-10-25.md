# MOSS Realtime：Windows 官方分阶段 CUDA 诊断

本轮恢复了固定官方代码的完整 codes→PCM 执行与辅助内容回读，但不是模型同时常驻
GPU 的热性能基线，也没有原生流式性能结论。没有修改 Rust 模型或生产加载策略。
设备 Windows 11 / Ryzen 7 9700X / RTX 5070，证据根
`D:/workspace/talechime-benchmarks/windows-moss17-20261024`。

## 身份与控制条件

- 官方 TTS 源码 `934d6826b084c46a0d033402174d5f8ac4ed2519`，checkout干净。
- 官方 codec 源码 `8c50ac4c5d7287d2ed6ea20a08c90ca439887d23`，checkout干净。
- Realtime 权重 `75682787d8e2fcc73faca37ba2931453ca9c4022`，codec 权重
  `3cd226ba2947efa357ef453bcad111b6eafba782`；源码与权重 revision 分别固定。
- Torch2.8.0+cu128、Transformers5.0.0、Safetensors0.8.0+windows.headerio1，
  非官方推荐 Torch2.9.1+cu128。原始模型类型告警保留在stderr，不宣称推荐栈验收。
- 官方 `MossTTSRealtimeInference.generate`，SDPA、BF16主模型；官方 codec
  `decode(..., chunk_duration=8)`，默认 F32、无autocast；两段都在CUDA执行。
- 与官方示例加载方式的偏离：direct CUDA device_map、pread元数据IO控制；主模型
  生成后卸载，再加载codec。未调整算法、精度、上下文、采样或权重以腾挪容量。

输入为short.txt“你好，欢迎收听今天的故事。”，seed7330414270409262323；无参考，
temperature0.8/top_p0.6/top_k30/repetition_penalty1.1/window50，batch1。
max_length375对应本任务请求预算，非示例5000；不把达到上限的截断算成功。
固定官方循环仅在首通道EOS时提前退出，并在返回前裁掉EOS；本次51帧小于375，
是沿该源码合约得到EOS证据，没有添加影响采样的逐token同步hook。尚未单独记录
EOS token轨迹，也不把这项判据扩展到其他官方版本或batch。

## 执行与实际输出

冻结Rust xtask `ce6f4d64ceb23a8488c358ae1935f3c2eb24f4169f60e5cba4971db850997f02`
依次运行realtime-official-staged-job.json和realtime-official-staged-v2-job.json，
各1进程、0预热+1请求。v2仅增加主模型实际dtype/device/参数计数断言，没有改数学。
其403参数张量全部CUDA BF16，4,663,881,728字节；codec参数断言全部CUDA F32。
原始脚本分别冻结为moss_realtime_staged_cuda_diagnostic.py与*_v2.py，没有覆盖前次
输入身份。两次均terminal success、cleanup通过，资产SHA和raw PCM摘要独立复核。

| 阶段（秒） | 初次诊断 | 增加参数断言的诊断 |
| --- | ---: | ---: |
| 主模型加载 | 3.10149 | 2.96178 |
| 官方codes生成 | 4.74503 | 5.51543 |
| 主模型卸载 | 0.09699 | 0.09884 |
| F32 codec加载 | 4.84536 | 4.85253 |
| 官方codec解码 | 0.17996 | 0.18101 |
| 含加载/换载总wall | 12.97188 | 13.61272 |

这两个单请求都含首次CUDA执行成本，没有热轮分布；阶段变化不作提速或回退结论。
总wall写入原始合成/首PCM字段用于保留实际整段边界，报告不得将其重命名为热性能，
也不得把阶段时间直接相加伪造常驻模型完整请求时间。没有缓存命中或参考准备数据。

两次codes.npy完整字节相同（51×16、int64、范围1..1023），raw F32LE PCM逐字节相同，
均97,920样本、24kHz单声道、4.08秒；PCM SHA为
`8b0ecb73b73f3b62e2e7bfd54a9a477cbb2a91e9710ff963da859ddd3642df30`。
此摘要不含Rust生产摘要的sample_rate/channels前缀，不能跨摘要域直接比较。
两次相同不意味与原生RNG/codes相同；既有原生2.40秒音频也不等工作量，不能排名。

SenseVoiceSmall-int8 revision2365baeacb507f821a0c8120fcee3d484dba7a07，CPU4线程
辅助回读结果“你好欢迎收听今天的故事”。冻结识别程序SHA61ef8c153f4eab54cbd897504676693f3a91fb254c6f21d532a824d67561ff65；
期望文本仅用于结果对照，没有传入recognizer。完整PCM相同允许复用此回读证据，
但不是双ASR验收或人工试听，听感pending。

## 容量与结论边界

启动前系统可用commit约11.95GB，保留10GB单job门禁。两次Job提交峰值分别
10,608,345,088 / 10,641,985,536字节，采样系统最低余量1,264,222,208 /
1,239,080,960字节。这是Job commit，不是RSS或显存。初次采样whole-device显存峰值
8,358,199,296字节，含桌面其他进程，不冒充本进程独占用量。

主模型卸载后Torch allocated仍为8,519,680字节；小于明确64MiB诊断门禁，证明没有
保留4.66GB主权重，不声称整个CUDA上下文清空。两次实际codec加载后allocated均
7,106,816,000字节。未修改系统页面文件、结束用户程序或触碰模型文件。

分阶段驻留使默认精度的官方语音可运行，提供了后续对齐的真实codes/音频与加载归因。
它不满足“常驻完整请求3×(1+5)”要求，不能据此宣布官方热基线恢复或原生超过官方。
后续仍需足够资源下的常驻对照、官方流式入口/compile实际可用性、匹配参考及采样
工作量的对照和听感验证。Local的超独立显存下界结论也保持独立。

复现使用冻结工具、上述job与新的输出目录：

```powershell
& $xtask perf reference --job D:/workspace/talechime-benchmarks/windows-moss17-20261024/realtime-official-staged-v2-job.json --output NEW_OUTPUT
```

Python AST、真实双次执行、资产及PCM复核通过；本轮只加诊断脚本与文档，未改Rust
推理或模型依赖，未为此重复整套模型编译门禁。详见*-audit.json、*-repeat-audit.json、
readback案例/结果、两次原始run和冻结源码清单。
