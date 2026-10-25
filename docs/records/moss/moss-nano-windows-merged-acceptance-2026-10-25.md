# MOSS Nano Candle 合并后验收（Windows，2026-10-25）

目标为合并版本的回退检查，不新增性能候选。Windows / Ryzen7 9700X / RTX5070，
CUDA13，驱动616.56；main为`1746d7232ab79367f1a8b9411a7f8e917a0a4fe5`加Windows本地改动，
Candle统一`884b359895a16c8fbd3647789b1f7c7974894708`。

使用冻结xtask（SHA `08cafd5a641a321ad0179cf3d7da13a63e3f022f19c9f3e25926d288713252c1`）
的`perf build`生成新release收据，源码guard通过；新performance_probe SHA为
`bd043ec38bfc8b1a3868655ca5d57dded57b1aa5ea3a47c6d83f7314c153029e`。
features为moss、moss-nano-candle、moss-nano-candle-cuda、performance-tools，F32，Rayon8。
构建8m10s，仍有既有LNK4098 CRT冲突警告，未添加屏蔽选项，不视为已解决的交付风险。

## 当前生产请求结果

沿用[原RoPE验收](moss-nano-windows-forward-rope-2026-10-25.md)B臂的固定短句/段落语料、
Weiguo音色、Pinned42、无continuation、原参数、资源和采集预算；新建计划与Job，
不覆盖旧数据。原生Registry/SynthesisStream，每case三独立进程、各1预热+5热轮，
不与编译或其他推理并行。段落包含两个文本分段，不能按单段计资产或验证范围。

| case | 三个进程热均值s | 均值s | 首PCM均值ms | 音频s | RTF均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 短句 | 0.42434 / 0.42635 / 0.53239 | 0.46103 | 48.69 | 3.36 | 0.13721 |
| 段落 | 2.17839 / 2.24366 / 2.12775 | 2.18327 | 46.29 | 16.08 | 0.13578 |

Windows Job commit高水位：短句1.553–1.554 GiB、段落1.598–1.599 GiB；
不是RSS或独占显存，保留原生采集工件。六进程全部success、exit0且cleanup_verified。

历史B臂短句均值0.46445s、进程范围0.43782–0.48967s；段落2.22971s、
范围2.12627–2.42525s。新旧范围重叠，当前未显示持续性性能回退；
短句第三进程较慢如实保留，不剔除。不同日期和进程矩阵不是同期A/B，
不能宣称新版本提速，也不能从此排除所有负载的回退。

## 输出与证据审计

36轮完整输出摘要与相应历史B臂一致；180个完整/分段F32及PCM16 WAV资产的
SHA和长度也与对应历史输出一致。另36个ranges JSON按自身原始SHA/长度核验，
除chunks外的元数据与旧控制一致；分包样本起止和source_range全部一致，覆盖连续完整，
时间戳单独保留，不要求旧新相等。共216个本轮资产核验通过。

最初汇总器误要求计时JSON的整文件SHA恒定，并假设只有一个分段、固定资产数，
因此拒绝汇总；v1–v3脚本保留。最终v4枚举真实分段和资产，严格使用results里的
attempt_id定位旧文件，不能以目录首个同名文件代替对应attempt。早期目录扫描
看到的分包差异并非对应旧控制，精确定位后36轮边界全部一致，原始文件未改写。

此轮不读取内部codes/EOS，也未新增ASR或人工试听；成功完成由生产流与父采集审计确认。
当前两种固定负载的数值/分段/性能回退检查通过，不扩大为长文接续或听感验收。
普通测试/Clippy/rustdoc的已验证合并组合范围见[main合并记录](../platform/windows-main-merge-2026-10-25.md)，
本轮只新增Nano release测量探针构建与生产请求实测，不冒充又跑过整工作区门禁。

## 复现

外部根`D:/workspace/talechime-benchmarks/windows-acceptance-20261025`保存
`nano-build.json`、构建日志、两个case的plan/job/run、`nano-acceptance-summary.json`、
`nano-acceptance-memory.json`及v4离线汇总器。旧参考仍在windows-nano-20261024。

```powershell
# 配置同一CUDA DLL路径，先校验tool与build收据中的exe SHA。
$r = 'D:/workspace/talechime-benchmarks/windows-acceptance-20261025'
$tool = Get-Content 'D:/workspace/talechime-benchmarks/windows-indextts-20261024/component-diagnostic-final-tool.json' -Raw | ConvertFrom-Json
& $tool.path perf run --job "$r/nano-short-job.json" --output "$r/nano-short-replay-NEW"
& $tool.path perf run --job "$r/nano-paragraph-job.json" --output "$r/nano-paragraph-replay-NEW"
```

复现输出必须全新；机器、环境、资源改变时按当前工具生成新Job，不修改旧结果。
