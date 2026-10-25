# MOSS Local：默认精度的官方分阶段 CUDA 诊断

恢复了固定官方 Local 的完整生成、EOS和processor解码执行，尚不是常驻完整请求
热基线，也没有原生流式验收。本轮只增加参考诊断脚本与官方配套资源，没有改Rust
推理、生产资源清单、精度或播放行为。设备Windows11 / Ryzen7 9700X / RTX5070；
外部根 `D:/workspace/talechime-benchmarks/windows-moss17-20261024`。

## 条件与新发现

TTS源码 `934d6826b084c46a0d033402174d5f8ac4ed2519`，codec源码
`8c50ac4c5d7287d2ed6ea20a08c90ca439887d23`；Local权重
`12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e`，codec权重
`3cd226ba2947efa357ef453bcad111b6eafba782`。Torch2.8.0+cu128、Transformers5.0.0、
Safetensors0.8.0+windows.headerio1，非官方推荐Torch2.9.1+cu128。
主模型556参数张量实际全CUDA BF16，6,121,212,928字节；codec实际全CUDA F32，
无autocast。使用direct CUDA device_map / pread IO控制，不称为默认加载入口。

初次分阶段请求成功加载主模型，但tokenizer缺少chat template，在生成前失败；
原始 `local-official-staged-run` 保留，cleanup通过，Job提交峰值9,195,765,760字节。
本地生产资源清单只覆盖Rust所需文件，不能推断已拥有官方Python完整配套资源。
经固定revision树核查，官方提供[chat_template.jinja](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer/blob/12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e/chat_template.jinja)。
下载到 `~/.talechime/resources/moss/official/local-1.7b/<revision>/`，没有修改生产目录。
模板352字节，SHA `894e3ace884e212ae34e1f46063686e7ebcc8b2fa0153af4579dfa5685f63184`；
processor_config.json也保留来源和SHA，但未将其作为额外推理配置覆盖。下载证据
local-official-template-download.json与revision-tree.json，原件不覆盖。

诊断显式赋予tokenizer此原始模板，不编造模板或关闭apply_chat_template。
`moss_local_cuda_reference.py`也支持明确的chat_template资源路径；其常驻显存容量
拒绝策略保持不变，旧失败job与冻结脚本不改写。

## 两个成功控制

Rust xtask `ce6f4d64ceb23a8488c358ae1935f3c2eb24f4169f60e5cba4971db850997f02`
分别运行local-official-staged-v2-job.json、*-v3-job.json，各0预热+1请求。
脚本v1/v2/v3各有不可覆盖冻结副本。输入short.txt“你好，欢迎收听今天的故事。”，
seed7330414270409262323；无参考，max_new_tokens375、n_vq32，text温度1.5/
top_k50/top_p1.0/penalty1.0，audio温度1.0/top_k50/top_p0.95/penalty1.1。
这个375为本任务预算，非官方默认100000，不将触顶当作成功。

官方processor允许无参考请求先不装audio_tokenizer。先完成生成并保存CPU原始序列，
卸载主模型，再加载F32 codec并调用原始processor.decode。v3额外用真实codec重新
构造官方processor，重算同一输入并断言input_ids/attention_mask与加载codec前完全
相同。延迟加载没有改变该无参考请求提示词；不外推到需要参考编码的请求。

| 阶段（秒） | v2 | v3（含提示词等价检查） |
| --- | ---: | ---: |
| 主模型加载 | 3.57203 | 3.62764 |
| 官方生成 | 7.86317 | 7.59865 |
| 主模型卸载 | 0.11384 | 0.10616 |
| F32 codec加载 | 5.01614 | 4.90386 |
| processor解码 | 0.18077 | 0.20148 |
| 含加载/换载总wall | 16.74986 | 16.44150 |

v3解码阶段还包括提示词等价检查，不是纯codec计时。两个进程均success/cleanup通过；
单请求包含首次CUDA执行与所有换载，没有正式热轮分布，不解释成稳态提速或回退。
总wall保留在原始合成/整段首PCM字段，不得重命名为常驻热轮或用于和原生排名。

## 真实序列、EOS与音频

两次codes.npy完整字节相同，保存原始40×33整数序列（控制通道+32 RVQ），不是
裁剪后的纯音频codes。start_length0，首控制151652、最后控制151653等于配置EOS。
首行音频全pad1024，末行音频非pad，值域75..989。官方processor按所有音频通道
是否全pad分段，而非按控制EOS删行，实际解码39行，包括最后EOS行的音频codes。
39×1920=74,880样本、24kHz单声道、3.12秒。原始序列、EOS-row-audit.json和PCM
共同验证了工作量，不能只数控制EOS前的帧推断“多生成/少生成”。

当前Rust Local在检测控制EOS后返回，不再为该行生成/发送音频codes。这是已定位的
跨实现行为差异，不能在跨框架RNG不同的情况下仅靠时长反推其影响；本轮未据此
更改Rust EOS语义或认定官方/原生哪一方应被修正。后续需要受控同codes解码与听感
证据，不能为了帧数对齐删官方末行或加原生随机尾帧。

两个控制的raw F32LE PCM逐字节相同，SHA
`3db4ccfa9f77b0ea77d13d0b6d1992b8f22e14c5f1218b582e2ba51b5c49b589`；WAV资产SHA
另存，且此数值摘要不含原生sample_rate/channels前缀。辅助SenseVoiceSmall-int8
revision2365baeacb507f821a0c8120fcee3d484dba7a07、CPU4线程，识别为
“你好欢迎收听今天的故事”。冻结识别程序与前述Realtime记录相同，期望文本没有
输入recognizer；不是双ASR或人工试听验收，听感pending。

## 容量与保留边界

两次Job commit峰值10,611,175,424 / 10,648,109,056字节；采样系统最低可用commit
1,264,193,536 / 1,210,855,424字节，保留10GB启动门禁，没有改页面文件或终止
用户程序。主模型卸载后Torch allocated8,520,192字节，低于64MiB诊断门禁，不能
据此宣称整个CUDA上下文清空。Job commit、Torch allocated和whole-device显存分开。

分阶段GPU驻留提供了默认BF16主模型/F32 codec的真实音频与生成证据，但参数合计
13,219,478,528字节仍超过独立显存12,792,037,376字节。原先的常驻容量限制保留；
没有常驻3×(1+5)官方热基线，不能宣布原生达到或超过官方。大显存对照、推荐栈、
匹配参考/接续/风格、官方流式与人工试听仍待补。

复现用冻结xtask与新的输出目录：

```powershell
& $xtask perf reference --job D:/workspace/talechime-benchmarks/windows-moss17-20261024/local-official-staged-v3-job.json --output NEW_OUTPUT
```

Python AST、两个真实成功请求、原始序列/PCM/资产摘要复核通过，首次失败保留。
本轮未修改Rust推理或依赖，没有为纯参考脚本重跑整套模型编译。下载、原始run、
readback、EOS审计、逐阶段数据与不可变源码清单保存在外部根，模型/音频不提交仓库。
