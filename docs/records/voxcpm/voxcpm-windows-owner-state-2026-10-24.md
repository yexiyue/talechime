# VoxCPM Windows Backend owner 状态门禁

Windows 11 / Ryzen 7 9700X / RTX 5070；源码基点
`42a7887e28f095684266eac43a68bd8fcc303d11`，CUDA 13，Candle 0.11 fork 2266099d。
这是独立正确性验证，不重计或替代 [性能与官方对照](voxcpm-windows-production-2026-10-24.md)。

`crates/talechime-backends/src/voxcpm/cuda_tests.rs` 使用生产 VoxBackend、显式 CUDA，
每项在 current-thread Tokio + LocalSet 内加载一次模型，后续请求复用同一个 owner。
仅 cfg(test, voxcpm-cuda) 编译，ignored 用例不下载权重。使用 48 kHz 单声道、有限 PCM、
成功 End 与 F32 位模式作原始 Backend 边界比较；宿主后处理和实际播放不在本用例范围。

## 配置与用例

两种权重：Q8 revision `169f64d8b98bbaab1761e4ca3a83e6af653456cc`，
原始 BF16 revision `32279effe8c19989596f05d353d1447f51d9e915`。
Q8 使用已导入 custom:acceptance，另切换 narrator；BF16 使用 narrator。
两者保持 steps=10、cfg=2、temperature=1；预算换算错误修正后默认 max_frames=187
（29.92 秒），不再把历史 200 patch 当成 10.667 秒。
文本“你好，欢迎收听今天的故事。”；Pinned(42).resolve(0,0) 的实际 seed 为
7330414270409262323，不能与直接使用整数 42 的历史底层探针混称相同随机配置。

顺序为正常 A、unread 取消后 A、收到首 PCM 且观察 capacity=1 的队列已满后取消再 A、
无效音色入口拒绝后 A、有效一秒预算的实际 frame-limit 失败后 A。
失败请求要求已有部分 PCM、无成功 End，错误后继续 recv 必须关闭。
Q8 附加 A→narrator→A：B 要完整且不同，返回 A 与最初 A 逐位相同。
这不是两种参考音色之间的切换，也不证明默认音色的主观身份或音质。
观察满队列不是证明某个 send 当时正阻塞；unread 用例也不承诺取消发生于特定 kernel。

资源清单逐项重新核对两份生产 manifest 的 size/SHA256，以及 Q8 音色文件，9 项匹配。
证据目录 `D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`，清单
`backend-owner-resources.json`，运行日志 `backend-owner-state.log`。
两项真实模型测试已通过（18.67 s，不是性能基准）。BF16 正常 A 为 145920 个 raw PCM
F32 位模式，5 次完整 A 一致；Q8 A 为 130560，6 次完整 A 一致，附加 narrator B
完整且不同。两者负向均输出 46080 个样本（0.96 秒）后报 frame limit，无 End，
错误后通道关闭，再正常请求恢复。日志 `backend-owner-corrected-budget.log`。
15 项普通测试通过、2 ignored，日志 `backend-owner-ordinary.log`；release CUDA all-targets Clippy -D warnings、
rustdoc -D warnings、格式及 diff 检查通过，分别见 backend-owner-{clippy,rustdoc}.log。
既有 LNK4098 警告保留，没有修改动态 CRT 设置。

## 测试实际发现：输出时长预算用了输入采样单位

初轮及增加详细断言后的两轮均失败，保留 `backend-owner-state.log`、
`backend-owner-initial-diagnostic.log`、`backend-owner-detailed-failure.log`。
一秒参数下两模型仍分别完整输出 145920 / 130560 样本并自然 EOS；不能把这类
负向刺激失效当测试通过，也不能仅换长句或放宽断言掩盖单位错误。
原适配器按 2560/48000 换算，而 2560 是四个 latent 的 **16 kHz 输入**样本量；
实际 decoder 六级 strides 8×6×5×2×2×2=1920，每 patch 四 latent，输出为
7680/48000=0.16 秒。每秒应为 6.25 patch，而非 18.75。

params.rs 改用正确输出单位并向下取整，1/12/30 秒对应 6/75/187 patch。
向上 round 会令 30 秒变为 188 patch=30.08 秒；默认旧 200 patch 实际可达 32 秒，
现默认 187=29.92 秒，catalog 默认也保持合法。新增单元测试同时检查这些固定计数、
实际样本时长不超过预算、下一 patch 会超限。模型算术、随机流和 EOS 判定未修改。
这是预算正确性修复，不计作少生成带来的性能收益；旧默认预算的性能身份仍是历史值。
本次短句两模型的正常长度与修复前相同，但没有跨构建保存旧 raw PCM，因此不宣称
已证明修复前后逐位相同；逐位验收范围是修复后同 owner 的状态复用。

最终模型/Backend 的 29 份源码/manifest 摘要与测试二进制收据已存 `backend-owner-state-build.json`；
修复前失败二进制已另存 SHA256
`bb10604b1278bcfdc7db9ba0ab93a1ed4a0fdc145c8fd1e80ee7426e9fb3c557`。

## 复现

先设置本机 CUDA DLL PATH、CUDA_PATH 与 CUDA_COMPUTE_CAP=120，并设置
TALECHIME_VOX_STATE_RESOURCES 为已校验资源根；不从普通测试下载模型。

```powershell
cargo test --locked --release -p talechime-backends --no-default-features --features voxcpm-cuda voxcpm::cuda_tests::real_cuda_ -j1 -- --ignored --nocapture --test-threads=1
```

该构建只启用 voxcpm-cuda，不等于 cuDNN 合并组合的验收。长文本、接续、style、
双参考实验、人工试听及官方加载的内存阻塞仍须分别处理。


## 修复后的生产 Registry/SynthesisStream 基线

使用 Rust xtask 串行执行，两变体各 3 独立进程、各 1 预热 + 5 热轮，observation=baseline。
Q8 保持原 custom:acceptance WAV/metadata/cache，BF16 保持 narrator；相同短句、Pinned42、
默认 steps10/cfg2/temperature1，默认预算现 29.92 秒。资源根字符串沿用原 Job，
未改变参考缓存或校验。生产探针 release SHA256
`89716b79559558d7c08366a5ff4cbf3d0bf4670df0a0b54a19397bbcecb87ced`，
收据 budget-recheck-production-build.json；feature 为 performance-tools、voxcpm、voxcpm-cuda。
构建期间源码 guard 通过，编译与推理串行，既有 LNK4098 警告保留。

正式采用第二轮父工具（见下文报告问题），工具 SHA256
`ce6f4d64ceb23a8488c358ae1935f3c2eb24f4169f60e5cba4971db850997f02`，
收据 budget-recheck-xtask-r2-tool.json。模型二进制在两轮间没有重建；父工具与模型收据
分开保存，不把新采集器源码冒充模型构建源码。

| 模式 | 完整热合成均值 s | 三个进程均值 s | 首 PCM s | 音频 s | RTF |
| --- | ---: | --- | ---: | ---: | ---: |
| Q8 固定参考 | 0.643189 | 0.622398 / 0.638834 / 0.668335 | 0.123504 | 2.40 | 0.267995 |
| BF16 narrator | 0.961286 | 0.954705 / 0.959353 / 0.969800 | 0.094368 | 2.93 | 0.328084 |

表中进程均值按数值排序，逐进程原顺序及全部热轮在 budget-recheck-r2-summary.json。
均值/RTF/首 PCM 与 Rust report/results.csv 复核一致；范围不是置信区间。
Q8 加载 3.152–3.751 s、参考短合成预热 0.776–0.883 s；BF16 加载 2.667–2.886 s、
短合成预热 0.837–0.971 s。reference_prepare_seconds 均为 null，短合成预热不是纯参考编码。
这些冷/准备数据不并入热合成均值。

每个变体 18 轮均与历史生产 PCM 摘要相同：Q8
`1aa1afcdf360ccb199f0c0ff83f1ea3bfeb484ff9f4ed469ce4a483f81f076fd`，BF16
`b9d5174a564eaf1b50d4e5bdce26b081589c60b8e3349dcedff8848ca2904d37`。
这是生产宿主处理后的完整输出边界；不把它扩大成跨构建 raw latents/codes 的逐位证明。
相同输出及长度表明此短句未借预算修复少生成。历史 Q8 0.590 s、本轮 0.643 s；
历史 BF16 0.998 s、本轮 0.961 s，批次/预算/工具不同，不归因单因素或宣称增量提速。
官方 Python 仍受加载内存阻塞，C++ 条件/长度偏差与试听 pending 不因这轮改变。

Job peak commit：Q8 5.818–5.854 GB、BF16 6.881–6.881 GB（十进制）。
500 ms 全生命周期整卡显存最高分别 5.825 / 6.859 GB，包含桌面；进程 RSS 最高
约 1.046 / 5.199 GB。系统可用 commit 最低约 4.504 / 3.562 GB，均不是独占 GPU
分配器峰值。GPU 查询每帧列出 20–21 个进程，包含桌面及被测子进程，不能解释为
20 个竞争推理进程或证明它们没有 GPU 活动；未终止用户应用。

首轮 budget-recheck-{q8,bf16}-run 虽推理成功，但离线报告因 available 数字 count
带名称编码 reason 而拒绝，部分 report assets 保留。修复父工具的数值映射后原 Job
重跑到 budget-recheck-{q8,bf16}-r2-run，输出 budget-recheck-{q8,bf16}-r2-report；
两份 report 成功，index.html/manifest/results.csv/音频资产已生成，试听仍为 pending。
完整故障与修复见 [Windows 采集编码记录](../platform/windows-gpu-process-encoding-2026-10-24.md)。
没有修改旧 run、弱化 Reading 校验或覆盖失败目录。

复现使用预算复核目录中冻结的工具 path 和 Job，输出目录须新建：

```powershell
$tool = (Get-Content D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/budget-recheck-xtask-r2-tool.json | ConvertFrom-Json).path
& $tool perf run --job D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/budget-recheck-q8-job.json --output D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/budget-recheck-q8-NEW
& $tool perf report --run D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/budget-recheck-q8-NEW --output D:/workspace/talechime-benchmarks/windows-voxcpm-20261024/budget-recheck-q8-NEW-report
```

BF16 替换 Job 与输出名中的 q8 为 bf16；不把其他设备直接复用本机身份作为验收。
