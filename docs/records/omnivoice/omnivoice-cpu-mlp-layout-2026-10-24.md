# OmniVoice CPU 真实 MLP 布局筛选（2026-10-24）

以 `3e0623ed39c95bbc4acfa3bd2d0a40b528ef7c48` 为生产基线，真实输入的三种
gate/up 连续化方案均变慢，全部拒绝接入生产。保留私有 ignored 探针及
[三进程原始测量与哈希](omnivoice-cpu-mlp-layout-2026-10-24.json)，不增加运行时开关。
这次排除的是显式连续化，不能据此认为 CPU 或激活算子已到性能上限。

## 假设与实际路径

macOS CPU F32 已启用 Accelerate。固定 Candle revision
`2266099d92265112ecc0d56ecee1761e3cc2ac00` 的 SiLU 使用 Accelerate 向量函数；
非连续输入的连续行块会分别调用该函数，不是完全退化为标量 exp。
合并 gate/up 投影输出 `[2,157,6144]`，两块 `[2,157,3072]` 共享行距 6144 的存储。
假设是连续化可以减少 SiLU 的逐行调用或乘法处理非连续 up 的成本；
额外复制可能抵消收益，因此计时必须包含复制和下游。

实验只改变 gate/up 是否调用 `contiguous()`，保持 `gate.silu() * up` 的算术与顺序，
再使用同一个 down projection。没有引入新内核、近似 exp、精度或采样变化。
此前合并投影、grouped GQA、K/V 缓存和 decoder pointwise 路径保持现状。

## 输入与协议

- M4 Pro，8P+4E / 24 GB，macOS 26.6.2；release CPU F32，Accelerate。
- 复用真实捕获的 step 0/15/31 × layer 0/13/27，共 9 个 case，B2/S157/H1024/FFN3072。
- 从归档的 `mlp_input` 使用实际层权重重新计算 merged gate/up，重建生产非连续视图；
  不将 safetensors 中已经连续化的 gate/up 当作原始布局。
- 逐层加载一份 MLP，三个独立进程串行运行，Rayon/VecLib/OMP 均请求 8 线程。
  这不是对 Accelerate 内部实际线程数或 AMX 派发的直接观测。
- 每个 case、scope、variant 预热一次；5 个 group 中每个候选分别按原/候选/候选/原
  交错，每 slot 测 5 次调用。轮换候选在 group 中的位置。三个进程采用相同 case 次序。
- activation 边界预先计算 merged projection，计时包含 narrow、所需复制、SiLU 和乘法；
  MLP+残差边界另外包含 merged projection、down projection 与残差相加。
  两个范围重叠，不能相加或外推为整链耗时。

生产基线二进制复用上一轮 `probe-candidate`，SHA256
`1ffd7b89679329dc20b11be3813cf4d43cb6329b7f7481f59edf100b7de25f18`，本轮未重跑整链。
探针源码、探针二进制、Cargo.lock、9 个输入归档和原权重/输入 manifest 的哈希见配套 JSON。
原始日志、receipt 及冻结探针位于 `target/perf/omni-cpu-mlp-real/`。
模型张量、codes 和音频不提交。

## 结果

以下每格为该进程 9 个 case 的等权均值；每个候选使用自身 ABBA 的原路径对照。
正数表示候选更慢。完整逐 slot 数据全部保留，没有删除离群值。

| 候选 | activation 原→候选 / ms，进程 1 / 2 / 3 | MLP+残差 原→候选 / ms，进程 1 / 2 / 3 | 完整消费者变慢 |
| --- | --- | --- | --- |
| gate 连续 | .542→.621 / .559→.624 / .551→.622 | 3.164→3.246 / 3.194→3.270 / 3.158→3.243 | 2.59% / 2.35% / 2.70% |
| up 连续 | .534→.609 / .552→.624 / .545→.612 | 3.161→3.230 / 3.196→3.260 / 3.170→3.239 | 2.19% / 2.02% / 2.17% |
| 两者连续 | .538→.694 / .550→.701 / .546→.690 | 3.151→3.311 / 3.187→3.337 / 3.161→3.312 | 5.08% / 4.70% / 4.79% |

三个独立进程方向一致；显式复制的净成本超过连续布局带来的收益。
本轮没有单独拆解 vForce 内部成本或复制带宽，不能将所有差距都归因于某一个内部指令。
此前真实输入上 Rust/PyTorch 的激活差距仍存在，**没有被这三种布局方案消除**。
不将旧官方耗时与本轮未配对数字拼成新的排名，也不以此确认整链提升。

## 输出与验收范围

每个 case 的原始重放先与归档的完整 MLP 输出逐位相等，再对四个 variant 各执行两次，
检查完整激活、完整 MLP、完整 MLP+残差的所有 F32 位。所有检查通过；
逐位比较在计时外，热轮中不做比较或张量导出。有限输入覆盖不是所有浮点值的证明。
三个进程均成功退出，编译活动采样均为 0；未采集温度、频率或带宽，不能断言系统绝对空闲。

本轮只新增 `cfg(test)` 私有 ignored 诊断代码，生产源码、配置和输出路径没有修改。
因此不重复运行短/长整链和 decoder 验收，也不将上一轮完整 codes/WAV 一致性冒充本轮新测。
普通测试不加载或下载模型。未在其他 Apple 芯片、Linux/AMD、CUDA 实机验证。

格式检查、CPU/Metal `--all-targets` Clippy（`-D warnings`）、Metal rustdoc
（`-D warnings`）通过。CPU release 普通测试 37 passed / 6 ignored，Metal release
50 passed / 6 ignored；新增 ignored 探针另外在三个独立 CPU 进程中全部通过。

后续优先定位 CPU 7-tap 卷积的 im2col 与输出物化，继续用真实激活和完整消费者筛选。
MLP 若要减少 SiLU 中间写入或改变底层算子，需要独立实验；本轮没有启用先前失败的融合候选。
仅本地记录，不推送、合并或发布。
