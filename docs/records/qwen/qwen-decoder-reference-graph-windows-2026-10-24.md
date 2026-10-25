# Qwen Base 参考前缀解码图：Windows 候选

Windows 11 / Ryzen 7 9700X / RTX 5070，源码基点
`42a7887e28f095684266eac43a68bd8fcc303d11`，Candle 0.11 fork 2266099d，CUDA 13 / cuDNN 9.14。
官方对照、权重与音色身份见 [Qwen 生产记录](qwen-windows-production-2026-10-24.md)。

## 原因及改动

Base 参考 codes 为 86 帧，首个生成块 20 帧，decoder 实际输入为 106 帧。
原 MAX_FRAMES=64 导致首块一直 eager。缓存首次访问形状走 eager、第二次才录制，
因此一次预热后首个热轮仍可能承担录制成本；这些慢轮全部保留。
候选仅扩至 128，MAX_ENTRIES 仍为 4；129 帧继续 eager，没有减少左上下文或改变采样。
106/128 的 miss、capture、replay、保留输出与 129 回退的逐位 oracle 已通过。
修改属于共享 CUDA 路径，仅本 Windows 设备实测，不能外推 Linux 或其他 GPU；Metal 不变。

## 完整请求 ABBAABBA

证据根 `D:/workspace/talechime-benchmarks/windows-qwen-20261024`。
每变体 8 个串行进程，各 1 预热 + 5 热轮，关闭 owner diagnostics，父进程每 500 ms 采样资源。
A=`current-production-build.json`，二进制 SHA256
`5512dc0256a0a4dae0d36a0d067e3a7a1c11eebe4cf43ec208d449774604cafb`；
B=`large-decoder-graph-build.json`，SHA256
`92803f3fa106dfe8768444b8c547d1b526accefbb1e4671fd49de9deaa0daebe`。
实际 feature 合并包含 indextts-cuda-cudnn、qwen-cuda、performance-tools。

| 变体 | A 均值 / 进程范围 s | B 均值 / 进程范围 s | 耗时下降 | 首 PCM A → B s |
| --- | --- | --- | --- | --- |
| Base06 | 1.409783 / 1.399438–1.419356 | 1.213668 / 1.200481–1.229604 | 13.91% | 0.940472 → 0.742718 |
| Base17 | 1.757305 / 1.739094–1.770413 | 1.528780 / 1.505793–1.553236 | 13.00% | 1.025996 → 0.810768 |

范围是四个进程均值 min/max，不是置信区间。两变体各 48 次完整 PCM 一致，音频为
2.52 / 3.07 s。Base06 PCM SHA256=`14f92b5da20dc90d6441c474b1a1baaeab0a878c65f1fbf34d0dc9cfe93c6193`；
Base17=`ba986ac17849219842f2535ab9253658e643d38b3263c5ee2262d0e91b44aab9`。
逐轮、加载及参考预热见 `large-decoder-abba-summary.json`，原始 jobs/plans/runs 为
`{base06|base17}-large-decoder-{0..7}-{A|B}-*`。不能与旧批次绝对基线直接相减或累加历史收益。

## 内存与限制

十进制 GB：Base06 Job peak commit A 6.795–6.831、B 8.162–8.163；
Base17 A 8.901–8.939、B 10.101–10.141。预热均值分别 1.587→1.705 s、1.874→2.022 s。
整卡显存采样最高值 Base06 6.707→7.927 GB、Base17 8.933→9.943 GB，包含桌面，
不能称为模型分配器或逐进程峰值。Base17 B 系统可用 commit 最低仅 342,065,152 bytes。
内存代价约 1.2–1.4 GB，其他配置可用性尚未确认。RSS/private/系统 commit 独立记录在
`large-decoder-telemetry-summary.json`，不混用。

nvidia-smi 竞争进程列表因 UTF-8 解码失败而 unavailable；空列表不能证明无竞争。
已控制本任务编译与推理串行，外部竞争检测仍有此限制。

## 正确性与剩余工作

四种 Backend owner 在候选上重新通过取消、观察满队列后取消、入口拒绝、部分输出后
frame-budget 失败及复用；CustomVoice 包含 A/B/A。日志 `large-decoder-owner-state.log`，
收据 `large-decoder-owner-state-build.json`。观察满队列不等于证明 send 正在阻塞。
test-only codes 采集增加主机同步，不计入性能：Base06 在生产 375 帧预算下，33 帧
codes、63360 个 raw PCM F32 位模式与自然 EOS 在 eager/miss/capture/replay 全相同，
见 `large-decoder-base06-codes-375.log`。Base17 同预算 40 帧 codes、76800 个 raw PCM
位模式及自然 EOS 亦全部相同，见 `large-decoder-base17-codes-375.log`。221 项普通测试通过、12 ignored；release CUDA all-targets Clippy -D warnings、
rustdoc -D warnings、cargo fmt 与 diff --check 通过。LNK4098 既有警告仍保留，未改 CRT。
最终 test-only 二进制与四份源码摘要见 `large-decoder-final-codes-build.json`；
独立工程日志为 `large-decoder-final-{ordinary,clippy,rustdoc}.log`。
人工试听、Base 跨参考/接续、长文本、style 与 VoiceDesign 仍 pending。


复现正确性使用本机已固定资源与 CUDA/cuDNN DLL PATH，设置 QWEN_CP_TEST_MODEL 为
对应 Base 模型 revision 目录，QWEN_CP_TEST_PROMPT 为同 revision 音色的
prompt-encoder-rope-v2.json，然后执行：

```powershell
cargo test --locked --release -p talechime-qwen3-tts --features cuda,candle-core/cudnn decoder_graph_large_clone_preserves_codes_pcm_and_eos -j1 -- --ignored --nocapture --test-threads=1
```

这是独立正确性用例，不替代 xtask perf 的正式计时。正式 ABBA 的每份 job/plan 均冻结
资源、构建收据和输出目录；恢复时使用原 job/plan 交给 xtask，避免把当前脏工作树的新构建
冒充旧 A/B 二进制。完整 evidence 摘要在仓库外 large-decoder-final-evidence.json。
