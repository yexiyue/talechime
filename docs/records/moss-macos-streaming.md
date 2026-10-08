# MOSS 1.7B macOS 流式输出与静音错误复测（2026-10-08）

> 同步自 TRNovel [c5d7daa](https://github.com/yexiyue/TRNovel/commit/c5d7daa60b435293d77f1ab886e22fdd56a214c0)。以下为上游 macOS 实机记录；产物路径属于原检出，crate 路径和命令已映射到 Talechime，未在此次同步中重做实机试听。

## 当前结论

MOSS Realtime 1.7B 的计算和适配器支持逐块流式输出。当前 Candle 0.11 Metal 加载路径原本在未显式完成设备工作时发送 ready，真实推理出现提前 EOS 或无效 token；截图所示的 `backend produced only silence` 不能当作不支持流式的证据。
在模型加载与 codec 加载后分别执行 Metal `Device::synchronize()`，本机 Realtime 连续三轮生成非静音音频、正常 End，首块早于整段结束约六秒。该改动是实际修复证据；尚未定位更底层的具体 Metal 内核/驻留行为，不笼统宣称已证明上游缺陷原因。

Local 1.7B 当前 Candle 0.11 / 24GB Mac 路径仍无法通过：同步后初始化明确报告 GPU `Insufficient Memory`。因此没有有效 Local 性能数字，也未宣布 Local 支持情况通过实机验收。

## 环境与测试条件

- M4 Pro，12 CPU 核，24GB 统一内存；macOS 26.6.2 (25G83)。代码基线 `cd8d710`，Cargo release / locked，Candle 0.11.0，模型与 codec F16。
- 默认根 `~/.novel-tts`。Local revision `12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e`，Realtime `75682787d8e2fcc73faca37ba2931453ca9c4022`，共享 codec `3cd226ba2947efa357ef453bcad111b6eafba782`。三套全部资源按 manifest 大小/SHA-256 复核通过，无重新下载。
- 原文：“山风吹过松林，星光照亮归途。两个人沿着小路慢慢往前走。”。`narrator`、无参考，单段，默认采样/seed 42；没有修改温度、帧预算或音频校验。
- 生产 `CandleBackend`，生成线程每 5 个完整 codec 帧解码一次。codec hop 1920、24kHz，即 80ms/帧；常规块 400ms，末块可短于此值。
- 临时探针在消费端记录每块到达时间；预热一次、热测两次，复用同一个模型和 codec。临时 example 已恢复，源码快照保存在产物目录。

## 失败与旧版本对照

| 路径 | 实际结果 | 性能资格 |
| --- | --- | --- |
| 0.11 Realtime，原加载路径，单模型隔离 | `Realtime EOS before all text was consumed`，没有 PCM | 失败，不报告 RTF |
| 0.11 Local，原加载路径，单模型隔离 | `unexpected MOSS text/control token 127695`，没有 PCM | 失败，不报告 RTF |
| 保留的 0.9.2 Realtime，完全相同输入/权重 | 非静音 PCM 6.96s，正常 End；首 PCM 1523ms，生成 11180ms | 单次冷生成，仅功能对照，非热性能比较 |
| 0.11 Realtime，加载同步后 | 三轮非静音 PCM 7.12s，均正常 End | 下表热测 |
| 0.11 Local，加载同步后 | 初始化 GPU OOM | 未验收 |

旧二进制的嵌入源码路径确认为 candle-core-0.9.2；用户运行的 debug worker 为 candle-core-0.11.0。MOSS 计算/适配器源码在 `8d13984..cd8d710` 没有改动。相同输入旧版成功、当前原路径失败、加入加载同步成功，说明版本升级后的加载执行路径需要这项修复；不把音频差异一概解释为浮点舍入。

## 修复后的 Realtime 流式测量

| 轮次 | 首 PCM ms | 生成 ms | 音频 s | RTF | PCM 块 | 平均块间隔 ms | 最大块间隔 ms | 累积 3s 音频时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 预热 | 420 | 6305 | 7.12 | .885534 | 18 | 346.2 | 370 | 2795 |
| 热 1 | 395 | 6343 | 7.12 | .890871 | 18 | 349.9 | 384 | 2788 |
| 热 2 | 400 | 6335 | 7.12 | .889747 | 18 | 349.1 | 372 | 2791 |

热平均 RTF .890309，首 PCM 397.5ms。18 块中通常每块 400ms，末块 320ms；远早于整段 End 就交付音频，证明实际流式。加载 22679ms 与首 PCM 分开统计。
全部三份修复 WAV 逐轮非静音、有限值；peak .670898、RMS .107667，逐 10ms 窗口通过播放器使用的 -45dB peak / -55dB RMS 静音判断。旧版对照 WAV peak .616211、RMS .102933。非静音不等于人工听感或逐字覆盖验收。

进程 `/usr/bin/time -l` 修复 Realtime peak memory footprint 16.72GiB、峰值 RSS 9.27GiB，包含加载瞬时峰值；原路径分别 20.52GiB / 9.29GiB。不是独占 GPU 显存，也不作内存泄漏结论。Local 同步后 OOM 保留原始错误，不能沿用旧版报告中可运行的结论。

## 为什么可能感觉没有流式

播放器默认先缓冲 3 秒音频，然后起播；速度提高时目标按 speed 增加。首 PCM 与实际起声是不同指标。此次探针大约 2.79 秒才累计 3 秒音频，即使首块 0.4 秒就到达，播放器也不会立即起播。
在实际播放结束前出现 Buffering 时，目标每次增加 2 秒，最高 10 秒；生成速度慢于播放速度时会反复等缓冲。源码见 `crates/talechime-core/src/session/buffering.rs`。实际播放器复验结果见下方最终补记。

## 内存并发样本排除

首轮测试时，用户打开的阅读器 debug worker 仍保持约 10GiB `owned unmapped (graphics)`，采样显示 MOSS 推理线程正在 `blocking_recv`，没有生成。另起模型造成两套 GPU 权重同时驻留，测试进程阻塞在 Metal residency commit、整机交换空间耗尽。此轮没有 PCM，标为 `valid_for_performance=false`，不用于性能或升级缺陷判断。
随后停止测试进程及闲置 debug TTS worker，保留阅读器；所有有效对照均单模型串行进行。没有修改用户持久配置、检查点或选择。

## 修复与复现

生产修复仅在 `crates/talechime-backends/src/moss/candle/runtime.rs`：Metal 模型加载后同步一次，codec 加载后再同步一次，之后才发送 ready。CUDA 行为没有改变。同步错误走现有 Initialize 错误路径，Local 的 GPU OOM 不再被延迟为无效 token/音频错误。

原始材料 `target/tts-integration/moss-mac-streaming-20261008/`：`streaming-summary.json`、`audio-check.json`、所有设备/版本 `.jsonl`、`.stderr.log`，三轮 WAV，原/临时 example 源码、线程采样和内存统计。`stream-probe-load-sync` 是实际修复行为的 release 计时二进制；源快照可在临时 example 中重建。

```bash
MOSS_MODEL=realtime-1.7b METAL_BENCH_RUNS=3 target/tts-integration/moss-mac-streaming-20261008/stream-probe-load-sync moss "$HOME/.novel-tts" /tmp/moss-realtime.wav metal @target/tts-integration/moss-mac-streaming-20261008/short.txt narrator
```

质量检查及实际协议播放器结果在最终补记记录。仅修改 Metal 加载顺序，不以本次短句证明 30 分钟稳定性、完整语料覆盖、其他音色或人工音质验收。

## 最终实际播放器与质量检查

重建的 release worker 使用独立配置、音量 0、1x、对齐关闭，默认 3 秒预缓冲。两次完整请求都正常 Completed，零 underrun，中间在首次非零缓冲后发送 stop，得到 Cancelled，之后的完整请求仍成功；正常 Shutdown，进程 exit 0。未触发 `backend produced only silence`。

| 请求 | 首次 Playing ms | 总耗时 ms | underrun | 结果 |
| --- | ---: | ---: | ---: | --- |
| 0 | 3328 | 10527 | 0 | completed |
| 1 | 取消前未起播 | 1033 | 0 | cancelled |
| 2 | 2862 | 10096 | 0 | completed |

原始日志 `playback-realtime-1.7b/events.jsonl` 同时保存协议事件和接收时间；`result.json` 保留每次 buffer_status。停止/复用的证据是实际 worker 和真实播放时钟，音量为零，并未改变用户配置。短请求不替代长正文、30 分钟播放或人工听感。

`cargo test --locked --all-features --workspace --lib --tests --examples`：541 passed / 0 failed / 6 ignored；all-targets/all-features Clippy `-D warnings`、fmt、rustdoc（私有项、examples，warnings denied）、diff whitespace 检查均通过。debug/release worker 重建通过，feature `coreml,metal,voxcpm-metal,omnivoice-metal,moss-candle-metal` 加默认后端/对齐。阅读器进程保留；需要重启当前阅读器后再测试新 worker，避免继续使用已关闭的旧子进程连接。

使用 `simplify` 复核后保留两个明确加载边界的同步，没有引入辅助抽象、调整推理参数或添加新依赖。
