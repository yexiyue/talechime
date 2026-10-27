# Kokoro CPU：继续优化与同期官方对照（2026-10-27）

状态：本机实现、数值及性能验证完成，未发布。本轮起点是[第一轮 CPU 优化](kokoro-cpu-optimization-2026-10-27.md)后的版本；下表“本轮前”不是最初约 0.21 RTF 的实现。

## 最终保留的改动

- CPU Snake 借用连续输入，在原存储读锁存活期间使用 F32 切片；复用 scaled 缓冲作为输出。保留 Candle 同长度的向量 `Sin::f32_vec`、square、divide、add 的原有舍入；非连续或其他设备走原 Tensor 链。
- macOS CPU affine linear/conv 对连续输入、权重和 bias 借用存储，非连续布局保留物化。同步 BLAS 完成后释放读锁，SGEMM 几何、beta=1 和 bias 预填充不变，不增加跨请求缓存。普通卷积宽窗口仍需展开。
- CPU AdaIN 在拥有的 Vec 上完成归一化与 style affine，去除 Tensor 中转及广播数组。保留 learned affine 后每个 style multiply/add 的舍入，不合并系数。
- macOS CPU、groups=1、带 padding 的转置卷积按独立输出行调度，复用 Candle 的 `VecOps::vec_dot` 和每个输出的 tap 累加顺序。padding=0、分组、GPU 保留原后端路径。

代码：[算子](../../../crates/kokoro/src/ops.rs)、[macOS CPU](../../../crates/kokoro/src/blas.rs)、[解码器调用](../../../crates/kokoro/src/decoder.rs)。没有改变依赖 revision、dtype、模型/词表、采样或音频处理策略。

## 归因与筛选

第一轮之后重新 profile，声学总耗时约 4.83 s：普通卷积 2.13 s、Snake 0.73 s、转置卷积 0.48 s、InstanceNorm 0.42 s、BiLSTM 0.30 s；AdaIN 含 InstanceNorm 为 0.59 s，这两个嵌套值不能相加。LeakyReLU 仅 0.015 s，没有为它增加快路径。

首版 Snake 复制输入，再融合后半段，长文约 5.17 s，相对当时 5.23 s 仅小幅改善；随后改为借用输入并复用输出缓冲。借用版与转置卷积组合约 4.85 s；再加入 linear 借用与 AdaIN 为 4.72 s。以上少量筛选轮不用于最终排名，组合收益以以下正式交错测量为准。

## 条件和协议

Apple M4 Pro（12 CPU / 16 GPU）/ 24 GiB / macOS 26.6.2；Rust release/LTO；Candle 0.11 固定 fork `884b359895a16c8fbd3647789b1f7c7974894708`，CPU F32 Accelerate。没有证明具体 AMX 硬件分派。

官方固定 Kokoro 0.9.4 源码 `dfb907a02bba8152ca444717ca5d78747ccb4bec`、Misaki `fba1236595f2d2bf21d414ba6e57d25256afada3`、g2pW `b9328ce8129c4717a70f16a2688542a0197541dd`；PyTorch 2.7.1、transformers 4.51.3、ORT 1.28。Kokoro 官方没有 g2pW 开关，表中“官方组合”为本地适配器将作者 g2pW 结果在变调/儿化前注入官方 Misaki，再调用未经修改的官方 KModel。不是官方发布的集成产品。

双方使用 v1.1-zh HF revision `01e7505bd6a7a2ac4975463114c3a7650a9f7218`、zf_001、speed=1、每段 seed=7，固定同一 180 音素预算分段。21 字短句 1 段，252 字多音字原稿 6 段；开启 g2pW，核对所有音素、整数时长和样本数一致。跨框架随机波形不要求逐位相同；原生优化前后要求相同。

`VECLIB_MAXIMUM_THREADS=4`、`RAYON_NUM_THREADS=4`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`；PyTorch intra=4/inter=1，g2pW ORT CPU 2 线程。原生 ORT 与官方采用同作者权重/窗口/batch，未换模型。

组件边界包括前端、声学和取得完整主机 PCM，排除加载、预先冻结分段和写盘。官方返回 NumPy PCM，原生将分段拼入主机 Vec 后停表；原生额外拼接也计入。每进程每语料 1 预热 + 5 热轮，先冻结独立二进制，再按“本轮前→新版→官方→官方→新版→本轮前”（ABCCBA）串行运行。

公共 Engine 另按“本轮前→新版→新版→本轮前”（ABBA）测量，每进程 1 预热 + 5 热轮，消费至显式 Completed。计入生产分段、边界静音整理、响度对齐和通道交付，排除 prepare、导出和播放。请求 seed=7 由生产者按段派生，分别与其自己的基线比较。

正式计时没有与本任务编译或其他本任务推理并行。未锁频；记录后台进程与 thermal 快照。每组两个独立进程，先算各进程热轮均值再平均；范围为两个进程均值的范围，不是置信区间。

## 同期组件端到端对照

| 语料 | 版本 | 音频秒数 | 合成秒数 | RTF | 进程均值 RTF 范围 |
| --- | --- | ---: | ---: | ---: | --- |
| 短句 | 本轮前 | 5.175 | 0.5198 | 0.10045 | 0.09999–0.10091 |
| 短句 | 新版 | 5.175 | 0.4630 | 0.08947 | 0.08859–0.09035 |
| 短句 | 官方组合 | 5.175 | 0.4192 | 0.08101 | 0.08041–0.08160 |
| 多音字原稿 | 本轮前 | 53.500 | 5.4139 | 0.10119 | 0.09886–0.10353 |
| 多音字原稿 | 新版 | 53.500 | 4.7264 | 0.08834 | 0.08801–0.08868 |
| 多音字原稿 | 官方组合 | 53.500 | 4.3347 | 0.08102 | 0.07824–0.08380 |

短句：本轮耗时减少 10.9%；新版相对官方组合耗时差为 +10.4%。

长文：本轮耗时减少 12.7%；新版相对官方组合耗时差为 +9.0%。

本例按 skill 的“均值差不超过 10% 或进程均值范围重叠”判定：长文达到 parity；短句差 10.4%，区间也不重叠，尚未达到这一门槛。没有超过官方的证据，也不把有限语料结果推广到所有输入。

长文声学均值仍为原生 4.1835 s / 官方 3.7555 s，前端则为 0.5423 / 0.5779 s。剩余差距主要在声学，后续需重新定位已优化后的卷积/归一化成本，不能再将整链差距笼统归因于 g2pW 或 Rust。

## 公共 Engine 的实际交付

| 版本 | 音频秒数 | 合成秒数 | RTF | 首 PCM 秒数 | 进程均值 RTF 范围 |
| --- | ---: | ---: | ---: | ---: | --- |
| 本轮前 | 48.810 | 5.3719 | 0.11006 | 1.0761 | 0.10993–0.11018 |
| 新版 | 48.810 | 4.6972 | 0.09623 | 0.9723 | 0.09567–0.09680 |

公共 Engine 本轮耗时减少 12.6%，约 10.39 倍实时。Engine 与组件输出长度及边界不同，不能把此 RTF 直接与官方组件 RTF 排名。

## 数值与行为门禁

- 正式原生序列 48 份完整 F32 PCM（包含预热）与本轮前逐字节相同；所有官方/原生正式轮的音素、整数时长和样本数一致。
- 五组独立官方夹具覆盖 zf_001 标准/0.8/1.2 speed、zm_010 短/长句；CPU/Metal 共十次均通过原有阶段与 PCM SNR>=40 dB、duration_equal 门禁。十份完整 PCM 同时与第一轮最终产物逐字节一致。

| 官方夹具 | CPU 音频 SNR / dB | Metal 音频 SNR / dB |
| --- | ---: | ---: |
| export-v1 | 43.112 | 43.111 |
| case-female-slow | 42.740 | 42.740 |
| case-female-fast | 40.987 | 40.978 |
| case-male-short | 45.025 | 45.053 |
| case-male-long | 41.953 | 41.950 |

- 公共 Engine 两模型完整输出分别与原试听逐字节相同：g2pW 48.810 s、原版 48.960 s。CPU/Metal 同 seed 重复、首 PCM 前取消、取消后复用与 close 均通过。
- 默认 CPU 与 g2pW+Metal 的模型库各 19 测试通过（2 真模型测试默认忽略）；后端 51、facade 8 测试通过。opt-in 两项真模型通过：整模型 SNR 43.1115 dB、固定特征 decoder 77.0372 dB，原门槛分别 40/60 dB。
- `cargo fmt --all --check`、三包全 targets Clippy `-D warnings`、严格私有 rustdoc 全部通过。单元覆盖原 Tensor 链/物化 GEMM/Candle dot 路径逐位对照、非连续与有偏移视图、不同 batch/风格广播、转置 stride/padding/output_padding 与行内归约。
- 未测 Windows/Linux/CUDA 实机性能；Metal 本轮只做完整数值/PCM/生命周期回归。有限语料的逐位一致不证明所有潜在输入的数学等价，也不等于实际播放验收。

## 证据与技能反馈

本地原始轮次、探针源码、冻结二进制、SHA manifest、完整 PCM、数值夹具、构建及测试日志在 `target/kokoro-parity-2026-10-27/`。`run_formal.py`、`summarize.py` 和 `write_report.py` 可复算；正式数据为 `model-[0-5]-*.json` / `engine-[0-3]-*.json`，筛选数据不混入。模型、音频与临时探针均未加入 Git。

已将热点迁移、读锁借用、向量 sin、按输出行调度及“融合引入额外复制”的经验补入 [Candle Accelerate/AMX 参考](../../../skills/candle-performance/references/accelerate-amx.md)。性能数字仅适用于上述硬件、版本、线程和语料。
