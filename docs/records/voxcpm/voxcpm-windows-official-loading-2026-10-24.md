# VoxCPM 官方 Python：加载控制恢复首个完整请求

以下保存10-24至10-25早期的加载调查。用户释放内存后的
[正式3×(1+5)与阶段对照](voxcpm-windows-official-baseline-2026-10-25.md)已完成；
下文“正式受阻/未启动”指当时阶段，原默认from_local仍未恢复。

Windows 11 / Ryzen 7 9700X / RTX 5070，官方源码 revision
`f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`，checkout 无改动。
权重为 openbmb/VoxCPM2 revision `32279effe8c19989596f05d353d1447f51d9e915`，
模型/音色 SHA 沿用 [生产记录](voxcpm-windows-production-2026-10-24.md)。
本轮恢复的是 **实验性加载控制下的官方推理代码**；原默认 from_local 加载仍未成功。
没有修改产品模型、系统 pagefile 或原 20 GB 默认加载预检预算。

## 结构核对

原 from_local 顺序为：构造 AudioVAE、读 VAE checkpoint、构造完整 F32 主模型、
转 BF16（VAE 随后恢复 F32）、读完整主 checkpoint、copy state、最后转 CUDA。
meta 骨架核对 577 个主张量、312 个 VAE 状态项，共 889 项，键和形状无缺失或多余。
主 checkpoint 的 2,290,004,544 个元素全部 BF16，约 4.580 GB；默认 F32 参数构造
单此部分约 9.160 GB，这是几何估算，不是新的峰值实测。

骨架发现 9 个不在 checkpoint 中的 RoPE buffer：三个 MiniCPMLongRoPE 各有
inv_freq、cos_cached、sin_cached。仅 assign 权重不能补齐这些 buffer。
max_length 保持 8192，base/residual KV 分别为
[2,28,1,2,8192,128] / [2,8,1,2,8192,128] BF16，没有缩短上下文换内存。
结构证据 official-meta-audit.{json,log}，脚本 official_meta_audit.py，均在仓库外
`D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

## 加载控制及精确检查

实验先打开 pread reader，再构造真实 F32 VAE 与 meta 主模型。577 个 BF16 主权重
逐项送 CUDA，并逐项回读与 checkpoint 精确比较；没有将全部主权重同时物化在 CPU。
三个 RoPE 用原类、原 config 在 CPU F32 重建，再按官方顺序转 BF16、转 CUDA，
9 项 buffer 精确比较。VAE 仍用原类/原 F32 checkpoint 的标准 load_state_dict，
312 项状态（含整数 buffer）转 CUDA 后逐项与 checkpoint 精确比较。
最终拒绝残留 meta 参数/buffer，检查主参数 BF16、codec F32，并 eval；未修改 forward。
冷加载含这些额外检查，不能与官方默认冷启动速度直接排名。

首个 smoke 因原隔离环境的 safe_open 不支持 backend 参数，在读取大权重前失败；
Job peak commit 1.492 GB，cleanup_verified=true。保留 official-meta-stream-smoke-run。
第二版显式复制已经验证的 Safetensors 0.8.0+windows.headerio1 模块、metadata 与许可证
到独立 official-loading-runtime-v2，不安装或替换原环境。22 项新增输入/来源被固定；
fork 仅改变 Windows pread header 的只读映射，见
[此前 IO 验证](../platform/windows-safetensors-header-commit-2026-10-24.md)。
Torch 2.8.0+cu128、Transformers 4.52.1 保持原参考栈。Safetensors 偏离官方环境须单列。

## 实际成功与限制

Rust xtask reference 使用冻结父工具
`ce6f4d64ceb23a8488c358ae1935f3c2eb24f4169f60e5cba4971db850997f02`，
Job official-meta-stream-v2-smoke-job.json，结果 official-meta-stream-v2-smoke-run。
1 独立进程，1 预热 + 1 热轮，显式 CUDA，narrator 无参考，seed
7330414270409262323，steps10、cfg2、max_len187、retry=false、optimize=false。
prepared 后两个完整流式请求均确认 stop_head=1，自然 EOS，19 块 / 145920 样本 / 3.04 s，
无截断，退出成功、cleanup_verified=true。

| 阶段 | 完整合成 s | 首 PCM s |
| --- | ---: | ---: |
| 预热 | 4.645067 | 1.775505 |
| 单个热轮 | 2.684686 | 0.161681 |

加载含校验为 5.473381 s；无参考。冻结旧 adapter 的 reference_prepare_seconds
约 12 微秒只是空分支计时，不是参考编码；仓库 adapter 对无参考模式明确记录 0。
预热与热轮 PCM 不逐位相同：同长度 max_abs=0.000249267、RMSE=0.00000693277，
26784 个样本改变。原因尚未定位，不假定 seed 保证逐位确定，也不建立任意容差。
这是官方参考数值观察，不是产品数值优化验收。所有权重/buffer 校验只证明加载状态，
不能替代与未能运行的默认 from_local 完整生成逐位对照。

Job peak commit 10,768,101,376 bytes，系统 available commit 最低 335,183,872 bytes。
整卡显存采样最高 7,289,700,352 bytes；500 ms 读数不保证捕捉瞬时峰值。
单 PID RSS 仅数 MB，因为虚拟环境 python.exe launcher 下还有实际解释器子进程，
不能用它宣称 Python RSS 极低；Job 高水位覆盖子树，是 commit 而非 RSS。
另用相同解释器做无模型进程审计，launcher PID 23060、实际解释器 PID 18172、
后者 ppid=23060，确认该环境存在重定向子进程，见 official-launcher-audit.json。
该成功不证明 9 GB 预检预算足够，也不与原失败峰值作“完整加载降低百分比”。

基于成功运行的余量风险，新正式 3×(1+5) Job 暂设 12 GB 可用 commit 规划预算。
official-meta-stream-v2-baseline-run 的预检实测 10,628,276,224 bytes，低于 12 GB，
在创建参考请求/进程前拒绝。没有启动新推理，没有伪造正式基线；12 GB 也不是已证明
足够的最低需求。原默认加载 guard 仍为 20 GB。

SenseVoiceSmall-int8 固定 revision 的单识别器回读，官方两份音频及原生 BF16 一份
均为“你好欢迎收听今天的故事”，结果 official-meta-stream-readback.json；
它是辅助内容证据，人工试听、音色/质量等价及双模型验收继续 pending。
原生 BF16 当前 3×(1+5) 为 0.961 s/2.93 s 音频，官方这里只有一个热轮且为 3.04 s、
没有宿主后处理，不作正式胜负排名或把差值归因某个算子。

## 复用与下一步

仓库 scripts/perf/voxcpm_lowpeak_loading.py 的 AST 与成功的冻结 helper 完全相同；
voxcpm_cuda_reference.py 新增可选 loading_control=meta_stream、loading_runtime、
use_reference=false、frame_limit，默认仍执行原 VoxCPM 路径。官方进程重复/计时继续
归 xtask，helper 只负责加载；正式复现冻结 helper/adapter/runtime 和输入 SHA 后再运行。
成功证据使用仓库外冻结 v2 adapter；仓库新增配置入口尚未重跑完整模型，不能冒充该 SHA。

下一步先独立减少加载/校验或 Python 子树的峰值，或在资源余量改变后恢复正式 Job；
不重复盲目默认加载、不删掉内容/上下文，也不把这两轮 smoke 扩称官方已达标。


## 后续：meta VAE 与首次调用对照（v3 实验，未替换仓库 helper）

此前只将主模型置于 meta，VAE 仍完整 CPU 构造。旧 weight_norm 的派生 weight
属性及构造时 autograd 引用可能保留临时存储，这是待验证的内存来源，不能仅凭源码
估算其实际峰值占比。新 v3 将 VAE 也置于 meta，严格赋入原 F32 checkpoint
（load_state_dict(assign=True)），再 CUDA 物化；推理类和 hook 保持官方实现。

先做独立 VAE 组件 oracle。第一次单进程 A首次→A第二次→B首次，原路径自身也非
逐位重复：前四块 max_abs 7.81e-6–1.24e-5；测试失败，保留
`official-vae-meta-oracle.{json,log}`，没有据此放宽容差。
随后两条路径分别在新解释器进程启动，保持相同 seed、输入、调用序号；各两次完整
六块流式解码加一次编码，13 对输出全部逐位相同，312 项状态的 dtype、shape、stride
也一致。原失败体现首次/后续调用条件不相同，尚未定位内部机制；不能直接把它归因
加载候选或宣称 PyTorch 任意调用均确定。新证据
`official_vae_meta_fresh_oracle.py`、`official-vae-{original,meta}-fresh.npz`、
`official-vae-fresh-comparison.json`。这些真实权重组件测试不计作语音性能基准。

v3 再经 Rust xtask reference 跑 1 预热+1 热轮，19 块/145920 样本/3.04 秒及自然
EOS 均保持，889 项模型状态和 9 项 RoPE 校验通过。两个请求的 PCM 分别与 v2
**同序号**请求完全相同，仍保持 v2 中预热/热轮自身不相同的事实：

| 序号 | v2 / v3 共用 PCM SHA256 |
| --- | --- |
| 预热 | aff78d7305fe843c2b29fb9675f41fa6b92093cfb89a8a32def4b2302d105e55 |
| 热轮 | 8b0601c55c458574c6fdd0a9b334e8b28ba364b2c81e29270cef149ce17ffd7b |

v3 仅作为加载实验：单热轮 2.629661 秒、首 PCM 0.160456 秒；不与旧单热轮相减
宣称优化提速。Job commit 高水位 v2 10.768 GB、v3 10.225 GB，各只有一个进程，
不是稳定改善百分比。父工具500ms采样的系统可用commit起始分别10.5272/10.5275GB，
最低分别0.3352/0.4076GB。全模型复测仍缺安全余量，不能因为Job点值较低就降低
12GB正式预检预算；v3 baseline Job 已冻结，未再次启动同类受阻运行。

v3 在实际解释器内用 Win32 API 采集冷阶段快照（不把 launcher 当实际模型进程），
`loading-memory.jsonl` 的关键值如下，GB 均为十进制：

| 阶段 | 进程 RSS GB | 进程 private GB | 系统可用 commit GB | Torch CUDA allocated / reserved GB |
| --- | ---: | ---: | ---: | --- |
| meta 构造后 | 0.808 | 2.254 | 8.249 | 0.302 / 0.302 |
| 主权重校验后 | 1.145 | 7.204 | 3.262 | 4.882 / 4.972 |
| VAE 校验/回收后 | 1.842 | 8.382 | 2.078 | 5.310 / 5.373 |
| 第一轮完成后 | 2.977 | 10.222 | 1.383 | 5.502 / 5.945 |
| 第二轮完成后 | 2.978 | 10.224 | 1.391 | 5.502 / 5.945 |

这些是阶段快照，不是每阶段瞬时峰值；表中1.383GB并不能覆盖采样实际观察到的
0.408GB最低余量。Torch allocated/reserved 也不是整卡显存或系统 commit。
实验的加载后1.5GB余量检查通过，但不足以证明后续推理会有足够余量；它不是内存
预留或运行时硬限制。因此保留12GB父预检，下一步不继续盲目同条件重跑。

v3 源码/Job/结果保留为 `voxcpm_lowpeak_loading_v3.py`、
`voxcpm-meta-stream-reference-v3.py`、`official-meta-stream-v3-{smoke,baseline}-job.json`、
`official-meta-stream-v3-smoke-run`；v2→v3 patch 与摘要在
`official-v2-to-v3-loading.patch`、`official-v3-loading-summary.json`。
原仓库 helper 仍对应此前 v2 的已验加载控制，不把外部 v3 实验冒充该源码身份。
