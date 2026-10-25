# VoxCPM Windows 段落负载与 C++ 对照

测量跨 2026-10-24 至 10-25，设备 Windows 11 / Ryzen 7 9700X / RTX 5070，
driver 616.56、CUDA 13。没有新的模型算术修改；在现有预算修复后补充真实负载，
不是新优化的 A/B 收益验收。短句基线与状态门禁见
[owner 记录](voxcpm-windows-owner-state-2026-10-24.md)，官方 Python 限制见
[加载实验](voxcpm-windows-official-loading-2026-10-24.md)。

## 配置与身份

63 字自写段落为：

> 夜色渐深，山风轻轻吹过松林。远处的小屋亮着一盏灯，旅人沿着石阶慢慢走来。他停在门前，听见屋里传出熟悉的笑声，心里终于安定下来。

UTF-8 文件含末尾换行，SHA256
`7daa53436749deaa6e31a8d9340b66801c40c2671600e456277f5b6c4e665893`。
文档是一段，生产宿主实际拆成两次模型请求；不是长篇章节、接续、style 或超长参考验收。
证据根 `D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

Rust 生产 Registry/Backend/SynthesisStream，release 探针 SHA256
`89716b79559558d7c08366a5ff4cbf3d0bf4670df0a0b54a19397bbcecb87ced`，
收据 budget-recheck-production-build.json；源码基点
`42a7887e28f095684266eac43a68bd8fcc303d11`，实际脏源码身份在收据中。
feature 为 performance-tools、voxcpm、voxcpm-cuda，observation=baseline。
Q8 使用原 custom:acceptance WAV/metadata/cache，BF16 为 narrator；权重/音色身份沿用
owner 记录。steps10/cfg2/temperature1，Pinned42 对两次内部调用分别派生 seed（见补正节），
默认上限187 patch=29.92秒。正式测量前已完成构建，没有并行编译或推理。

C++ 用冻结的 retired voxcpm-sys / llama.cpp-omni 对照，源码基点
`30339e67c44a3e000f7d1259ec038bf7daff18e3`，adapter SHA256
`405d501d44fb8ce822e63db4396b981f3d9274db2e4ad292817bfdcaab04bf7e`。
与 Rust Q8 使用相同两份 GGUF 权重及参考 WAV/文字，参考编码由各自实现准备。
C++ 固定 seed42、max_frames200，steps10/cfg2/temperature1；不能把 seed 整数差异
或框架随机流称为一致。回调还有30秒样本上限；本例音频远低于上限。

父工具为冻结 xtask，SHA256
`ce6f4d64ceb23a8488c358ae1935f3c2eb24f4169f60e5cba4971db850997f02`。
三条路径均各3独立进程，各1预热+5热轮，串行执行，共54次完整请求。
Native结果为数组，C++为external_reference_not_production对象，分别核对每个capture
的success/cleanup_verified及所有round，不合并伪装为同一种结果结构。

## 完整请求结果

| 路径 | 热合成均值 s | 三个进程均值 s | 首 PCM s | 音频 s | RTF |
| --- | ---: | --- | ---: | ---: | ---: |
| Rust Q8 固定参考 | 2.893720 | 2.924200 / 2.861101 / 2.895857 | 0.120119 | 12.4 | 0.233364 |
| Rust BF16 narrator | 4.637295 | 4.494236 / 4.693787 / 4.723862 | 0.043959 | 15.0 | 0.309153 |
| C++ Q8 固定参考 | 3.022464 | 3.035271 / 3.031072 / 3.001050 | 0.383949 | 12.8 | 0.236130 |

同一路径18轮PCM摘要分别唯一：

| 路径 | PCM SHA256 |
| --- | --- |
| Rust Q8 | e0e5edd09ba6737b8e91901fda81c4b64a7e363233e93e654634df997d6843ab |
| Rust BF16 | 049f97a086d1293b6a8788adbac815f36d3ea8b1847ef24042f12846d6f4c2a8 |
| C++ | 1700e08eaa09172af2d6e281857088fb6174fa9ba667d09930b919e5686b8ec1 |

Native 完成路径只在模型 Eos 且已有 PCM 时发送 End；预算错误不完成段落。
C++ adapter 要求 Outcome::Complete；桥接只在final回调时返回0，runtime 的is_final
来自 stop_on_predictor、min_steps 和 should_stop，未达到 stop 则桥接返回frame limit。
因此本例不是预算截断被当成成功。源码审计身份另存证据摘要。

Q8 完整wall均值较C++小约4.3%，但输出也短3.1%；RTF点值仅约1.2%差异。
Q8逐进程RTF范围约0.2307–0.2358，C++约0.2345–0.2371，**存在重叠**。
这些范围不是置信区间；不宣称稳定吞吐优势，不将短句结果外推为所有文本都超过C++。
首PCM差异还包含Native首段14字与C++全文63字的不同prefill，以及上述随机、参考编码
和宿主后处理偏差；不能据此归因后端算子更快。BF16默认音色/精度不同，
不能把它与固定参考Q8的耗时直接作同条件优劣排名。官方Python没有新增段落基线。

Q8加载3.042–3.361s、参考短合成预热0.783–1.004s；BF16加载2.588–2.685s，
短合成预热0.766–0.865s。Native纯reference_prepare仍为null，不能拿短合成与
C++纯参考准备0.158–0.165s直接相减。C++加载1.889–1.968s。
Job peak commit Q8 5.856–5.861GB、BF16约6.886GB、C++8.984–9.104GB。
其他500ms进程/整卡/系统采样在摘要中保留，各scope不混称独占模型显存。

## 关闭图的完整请求正确性

同一个生产二进制，各模型另跑1进程、1预热+1热轮，显式设置
VOXCPM_CUDA_GRAPH=0、VOXCPM_DECODER_GRAPH=0；两次完整请求的生产PCM与长度
均与默认路径18轮相同，正常完成。这里比较的是默认路径与显式eager配置，不将某个
未启用的图宣称为已录制，也不把这组小样本当正式图性能ABBA。
此边界是宿主处理后的PCM，不扩大为raw latent逐位证明；短句owner状态门禁独立保留。

最初只改Job声明而没有设置父工具环境，Q8在启动模型前被性能环境校验拒绝，保留
paragraph-q8-eager-correctness-run.log；没有弱化校验。修正Shell环境后新建v2-run，
BF16随后同样显式设环境。配置声明与实际环境都要匹配。

## 回读与复现

固定SenseVoiceSmall-int8单识别器回读：三份抽样覆盖整段；BF16将“他”写成同音
“她”。识别器的HAPPY/NEUTRAL/BGM标签不是人工听感结论，音色、听感、双模型质量
验收均pending，结果在paragraph-budget-readback.json。相同摘要只代表各自输出可重复，
不表示跨实现音频或质量相同。

原始Job/plan/run为paragraph-budget-{q8,bf16,cpp}-*，汇总
paragraph-budget-summary-v2.json（C++加载/参考从prepared事件提取）；
默认/eager摘要paragraph-eager-correctness-summary.json。Rust两份离线report已成功，
各自results.csv与表中均值复核一致，manifest仍标listening pending。

```powershell
$tool = (Get-Content D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/budget-recheck-xtask-r2-tool.json | ConvertFrom-Json).path
& $tool perf run --job D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/paragraph-budget-q8-job.json --output D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/paragraph-q8-NEW
& $tool perf reference --job D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/paragraph-budget-cpp-job.json --output D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/paragraph-cpp-NEW
```

BF16替换对应Native Job；eager复现还须在当前Shell显式设置两个图开关为0，输出目录
必须新建。所有模型/原始音频/结果保留仓库外；本次没有Rust源码变更，不重新声称
新的编译门禁。长篇、跨段接续、多音色、style、人工试听及官方正式矩阵仍有未完成项。


## 后续补正：内部段界必须从原始ranges读取

前版记录漏列了内部拆分。后续注意力调查逐一验证Q8/BF16共36份ranges与源字节SHA，
确认每个生产完整请求都是两个内部调用：[0,42)、[42,189)。当前rustc1.98.1下对应
Pinned42实际seed为7330414270409262323、15690801507913080125，冻结在
paragraph-production-boundaries.json。C++只调用一次全文并固定seed42。
因此本表保留的是默认产品行为对照，并非对齐段界后的模型性能基线；RTF也不能消除
两次prefill/缓存与一次prefill的工作量差别。各自18轮成功与默认/eager对照仍有效，
不同框架同条件排名仍未证明。详见
[单query候选调查](voxcpm-windows-attention-candidates-2026-10-25.md)的边界节。
