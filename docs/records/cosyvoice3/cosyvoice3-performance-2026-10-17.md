# CosyVoice3 Rust(Candle) 与官方 PyTorch 本机性能比较

日期：2026-10-17。结论与原始数据可复现：逐轮 JSON/CSV、日志、音频均在
git 忽略的 `runs/cv3-bench/`；复现命令见文末。本轮只做测量与基准工具，
未改推理算法、未放宽数值门禁、未提交。严格非流式整链波形等价门禁
（max ≈ 0.162）仍是已知失败，性能结论不改变它。

## 直接回答：谁更快、慢多少、什么不可比

1. **CPU 对 CPU（同模型、同 f32、同 8 线程、同输入 fixture）：Rust 全面更慢。**
   固定 token/mel 的受控组件对比（唯一同负载组）：
   Flow/DiT Rust 慢 **4.5×**（16 token）至 **5.6×**（184 token），
   HiFT 慢 **2.5×** 至 **4.7×**。端到端非流式 RTF：Python 1.09–3.38 vs
   Rust 4.33–11.34（Rust 采样轨迹多生成约 4%–94% 的 token，放大了差距；
   见"token 不对称"）。
2. **Rust Metal 是当前 Rust 最快路径，但只与 PyTorch CPU 大致相当：**
   非流式 RTF 1.62–2.81（Python CPU 为 1.09–3.38）；组件级 Metal Flow
   1.50s/5.77s vs Python CPU 1.51s/4.05s（短文本持平，长文本 Metal 反而
   慢 1.4×）。
3. **Metal/MPS 不可比**：官方源码把设备硬编码为 `cuda if available else cpu`
   （`cosyvoice/cli/model.py:36`），无 MPS 路径。把官方模块强制搬到 MPS
   的额外实验在官方代码内部崩溃（`torch.cat(): mps:0 and cpu`，
   `runs/cv3-bench/mps-probe.json`）。因此 **CPU↔Metal 差异只能叫设备差异，
   不能叫语言加速**。
4. **冷启动**：文件缓存热时 Rust 模型就绪 1.2–3.2s（CPU）/1.3–3.2s（Metal），
   Python 就绪 135–161s（import ≈ 8s + torch 权重加载 ≈ 127s）。Rust 冷启动
   快约两个数量级，但首次磁盘冷读（外置盘页入）双方都要 100–160s。
5. **流式**：Rust Metal 首包 2.1–3.0s（从建立会话起算的纯推理首包 2.5s/
   long），Python 官方（新鲜模型状态）3.9–6.4s；Rust CPU 首包 8.8–10.6s。
   所有实现 RTF>1，连续播放必然断供（见缺料分析）。
6. **主要耗时来源（实测，非源码推断）**：Flow/DiT 占 Rust 端到端的
   **76%–90%**（long 样例 32.9s/43.1s），是第一优化目标；HiFT 与 LLM 解码
   各占个位数百分比到 18%。

## 环境与版本

| 项 | 值 |
| --- | --- |
| 机器 | Apple M4 Pro，12 核 CPU（8 性能 + 4 效能），24 GB 统一内存 |
| 系统 | macOS 26.6.2 (25G83) |
| Rust | candle 0.11.0（crates.io，与 candle-metal-kernels 0.11.0 统一），ORT rc.13 |
| Python | 官方 CosyVoice 固定 `074ca6dc9e80a2f424f1f74b48bddd3fea531cc`，torch 2.7.1，numpy 2.2.6，onnxruntime 1.24.4 |
| 仓库 HEAD | `a765e6a`（基准二进制与本轮最终源码仅差 clippy/格式清理，见"方法局限"） |
| 权重 | Fun-CosyVoice3-0.5B-2512，SHA-256 与转换流程沿用 [本机对照记录](cosyvoice3-local-parity-2026-10-17.md)；转换产物 `runs/cv3-converted`，官方原始 `runs/cv3-model` |
| 线程 | 可比组双方统一 8（Rust `RAYON_NUM_THREADS=8` + ORT 前端 8 线程；Python `torch.set_num_threads(8)`）；默认线程敏感性单独补测 |
| 精度/模式 | 双方 f32、Flow 10 Euler 步、CFG 0.7、seed 42、非流式 speed=1.0 |

输入四档（文本/参考音频与双方完全一致，来自既有官方 fixture）：
short（快出发！，官方 16 token）、long（……做沙拉，再来一壶热茶，184 token）、
xlong（春天的清晨……太阳终于升起来了，442 token）、longprompt（夜色渐深……，
较长参考音频 `cross_lingual_prompt.wav`，117 token）。
注意 seed 42 跨实现采样轨迹不同（官方文档已声明无跨框架位级对齐），
双方实际生成 token 数见各表——这是端到端比较的固有噪声，受控组件对比不受影响。

## 方法与计时边界

- **Rust**：`crates/cosyvoice3/examples/bench.rs`，围绕公开库调用计时；
  Metal 计时结束前显式 `Device::synchronize()`（GPU 异步命令缓冲必须收敛）。
  阶段划分：参考特征提取（ORT 前端，含重采样/token/说话人向量/mel）→
  LLM 预填充（`start_incremental`）→ LLM 解码（逐 token）→ Flow
  （`flow.inference` finalize）→ HiFT（`hift.inference` finalize）。
  端到端总时另行对常驻 `CosyVoice3::synthesize` 单独计时（生产路径）。
  流式对 `start_stream` + `next_pcm` 逐块计时（拉取式，消费策略为立即收
  无人工 sleep）；"纯推理首包"= 会话建立（含预填充）后首块到达。
- **Python**：`tools/tts/cosyvoice3_bench.py`，零 hook、计时区内零导出：
  前端 `frontend_zero_shot`；LLM 用官方公开生成器逐 `next()` 计时
  （首个 next ≈ 预填充+首 token）；Flow/HiFT 直接调用官方
  `flow.inference`/`hift.inference`；端到端总时走官方 `model.tts`
  （LLM 线程与 token2wav 并发，属官方生产路径）。官方流式循环里的
  `time.sleep(0.1)` 轮询按原样测量。
- **官方 hop 状态泄漏（重要）**：官方 `CosyVoice3Model` 把 `token_hop_len`
  存在共享模型对象上并跨请求泄漏（Rust 移植时已按会话隔离并记录该差异）。
  带 warmup 的 Python 流式轮会继承 hop=100，首块膨胀到 104,640 样本
  （4.36s 音频）、首包 5.9s。为公平比较首包，另做了 3 次无 warmup 新鲜
  进程补测：首块 32,640 样本——与 Rust 首块尺寸完全一致——首包 3.90–3.98s。
  两组成数据都在 `runs/cv3-bench/stream-python-cpu-long*`。
- **执行顺序**：runner 按 case 交替 Rust/Python 降低热漂移；每模式先 1 轮
  不计时 warmup，再测 5 轮（个别 3 轮），报告逐轮原始值与中位数/最小/最大
  （样本量小，不给 P95）。错误与截断单独统计，不进均值。
- **方法局限（如实记录）**：
  1. 冷启动组无法控制磁盘冷缓存（purge 需 sudo）：矩阵首个作业专门预热页
     缓存，之后每组第 0 次仍偶发 100–120s 级页入（外置卷），在表中单列。
  2. RSS 时序采样首版采到了 `/usr/bin/time` 包装进程（序列无效，已修复
     runner）；峰值不受影响，以 `/usr/bin/time -l` 与 Python
     `ru_maxrss` 双源为准（两者互差 <0.5%）。
  3. Rust warm 模式同时驻留两份模型对象（生产 `CosyVoice3` + 分阶段部件，
     mmap 权重共享页），其峰值不代表生产单管线；生产代表值取冷启动组。
  4. 基准二进制构建于 clippy/格式清理前，差异仅为去 `mut`、`#[allow]`
     与换行，不影响代码生成路径。
  5. Python 阶段和是顺序复现（官方 tts 中 LLM 线程与 Flow 并发），故阶段
     相加 ≠ 官方端到端；两者分别报告。

## 一、冷启动（3 次独立进程；文件缓存热）

就绪 = 进程启动→模型可服务；首次请求 = 参考特征提取 + 完整合成（short 输入）。

| 配置 | 就绪 rep0/1/2 (s) | 首次请求 (s) | 说明 |
| --- | --- | --- | --- |
| Rust CPU | 105.3† / 3.2 / 1.2 | 9.77 / 9.75 / 9.80 | †外置盘页入一次 |
| Rust Metal | 117.0† / 3.2 / 1.3 | 2.74 / 2.62 / 2.67 | 同上 |
| Python CPU | 161.3 / 135.0 / 135.3 | 3.07 / 2.33 / 2.35 | import ≈ 8.1s + 权重加载 ≈ 126.9s（rep1 拆分） |

新进程 ≠ 磁盘冷缓存：除标注 † 外全部为热缓存。Python 每次进程都要完整
`torch.load` 三个 .pt；Rust mmap safetensors 热缓存就绪 1–3s。

## 二、热请求（模型驻留；1 轮 warmup 后 5 轮，中位数 [最小–最大]）

### 非流式端到端（生产路径总时，参考特征已备好）

| case | 实现 | tokens | 音频 (s) | 总时 (s) | RTF |
| --- | --- | --- | --- | --- | --- |
| short | Python CPU | 16 | 0.64 | 2.16 [2.13–2.17] | 3.38 |
| short | Rust CPU | 31 | 1.24 | 9.70 [9.63–9.73] | 7.82 |
| short | Rust Metal | 31 | 1.24 | 2.46 [2.42–2.51] | 1.98 |
| long | Python CPU | 184 | 7.16 | 8.39 [8.28–8.52] | 1.17 |
| long | Rust CPU | 241 | 9.64 | 43.10 [40.41–44.01] | 4.47 |
| long | Rust Metal | 241 | 9.64 | 11.86 [11.78–11.87] | 1.23 |
| xlong | Python CPU | 442 | 17.36 | 18.90 [18.81–19.80] | 1.09 |
| xlong | Rust CPU | 460 | 18.40 | 79.60 [79.53–85.35] | 4.33 |
| xlong | Rust Metal | 460 | 18.40 | 29.73 [26.48–30.62] | 1.62 |
| longprompt | Python CPU | 117 | 4.68 | 9.78 [9.74–9.80] | 2.09 |
| longprompt | Rust CPU | 131 | 5.24 | 59.41 [54.48–59.52] | 11.34 |
| longprompt | Rust Metal | 131 | 5.24 | 14.68 [14.67–14.69] | 2.80 |

token 不对称：Rust 轨迹在同 seed 下生成更多 token（long 241 vs 184，
short 31 vs 16），RTF 差距里含这部分工作量差。longprompt 双方都用较长参考
（prompt 更长→Flow 序列更长），RTF 双方都变差，Rust 更敏感。

### 分阶段（同一轮内的顺序复现；秒，5 轮中位数）

| case | 实现 | LLM 预填充 | LLM 解码 | Flow | HiFT |
| --- | --- | --- | --- | --- | --- |
| long | Python CPU | 0.106 | 3.79 | 4.06 | 0.32 |
| long | Rust CPU | 0.329 | 7.93 | 32.94 | 2.21 |
| long | Rust Metal | 0.051 | 3.10 | 7.75 | 0.89 |
| xlong | Python CPU | 0.123 | 9.65 | 8.46 | 0.77 |
| xlong | Rust CPU | 0.338 | 14.54 | 61.09 | 3.62 |
| xlong | Rust Metal | 0.064 | 6.96 | 20.09 | 1.87 |

逐 token 解码成本：Python ≈ 20.6ms/tok，Rust CPU ≈ 32.9ms/tok（慢 1.6×），
Rust Metal ≈ 12.9ms/tok（比 Python CPU 快 1.6×——小 batch 逐 token 推理
是 Metal 少数占优项）。Flow/DiT 是绝对主导：Rust CPU 端到端的 76%–90%。

### 参考特征每次重新提取（每请求；3 轮中位数）

| case | Rust CPU | Rust Metal | Python CPU |
| --- | --- | --- | --- |
| short（zero_shot_prompt） | 0.115s | —（前端固定 CPU） | 0.082s |
| longprompt（较长参考） | 0.536s | 0.506s | 0.295s |

Rust 前端（ORT 8 线程）比 Python（onnxruntime 默认线程）慢 1.4–1.8×；
该阶段占比很小（longprompt 也 <1s）。Talechime 生产后端前端固定 1 线程，
未在本表（生产配置差异已在 `docs/cosyvoice3.md` 记录）。

## 三、受控组件对比（同 token/prompt/mel/noise fixture；唯一严格同负载组）

5 轮中位数。加速比只在组内计算（Python 耗时 / Rust 耗时）。

| case | 组件 | Rust CPU | Python CPU | Rust/Py | Rust Metal |
| --- | --- | --- | --- | --- | --- |
| short（16 tok） | Flow/DiT | 6.77s | 1.51s | **4.5×慢** | 1.50s |
| short | HiFT | 0.141s | 0.057s | **2.5×慢** | 0.056s |
| long（184 tok） | Flow/DiT | 22.87s | 4.05s | **5.6×慢** | 5.77s |
| long | HiFT | 1.53s | 0.326s | **4.7×慢** | 0.68s |

长序列上 HiFT（卷积为主）Rust CPU 差距扩大（4.7×），Flow（GEMM 为主）
稳定 4.5–5.6×。Rust Metal Flow 短序列追平 Python CPU、长序列反慢 1.4×；
Metal HiFT 2× 慢于 Python CPU。

## 四、流式（模型驻留；1 轮 warmup 后 5 轮；首包 = 请求→首块可交付 PCM）

Python 主表数值为泄漏 hop=100 状态下的官方真实行为（含 warmup 后的共享
模型），另列新鲜状态（无 warmup、新进程 ×3）的公平首包。

| case | 实现 | 首包 (s) | 块数 | 总时 (s) | 音频 (s) | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| short | Python CPU | 2.05 | 1 | 2.05 | 0.64 | 3.20 |
| short | Rust CPU | 8.79 | 1 | 8.79 | 1.00 | 8.79 |
| short | Rust Metal | 2.08 | 1 | 2.08 | 1.00 | 2.08 |
| long | Python CPU（hop 泄漏） | 5.90 | 2 | 9.85 | 7.16 | 1.38 |
| long | Python CPU（新鲜 hop）×3 | 3.90–3.98 | 3 | 11.7–12.0 | 7.16 | 1.63–1.68 |
| long | Rust CPU | 10.36 | 4 | 82.02 | 8.96 | 9.15 |
| long | Rust Metal | 2.49 | 4 | 20.73 | 8.96 | 2.31 |
| xlong | Python CPU | 6.37 | 5 | 36.58 | 17.36 | 2.11 |
| xlong | Rust CPU | 10.56 | 6 | 206.26 | 18.40 | 12.31 |
| xlong | Rust Metal | 3.01 | 6 | 69.03 | 18.40 | 4.12 |

- **纯推理首包**（特征已备好、会话建立后）：Rust Metal long 2.49s、
  xlong 3.01s；Rust CPU long 10.36s。Rust 首块 32,640 样本（hop 25+pad）；
  官方新鲜状态首块同为 32,640 样本，首包 3.9s——**首包延迟上 Rust Metal
  优于官方 CPU**（1.5–1.6×），尽管总 RTF 劣势仍在。
- **块生成时间**（Rust，`next_pcm` 调用净时长）：随 hop 增长而增长
  （官方 O(n²) 累积重算所致，语义如此，不优化）：rust-cpu long 四块
  9.9/14.5/27.2/29.0s。这也是官方行为的复现，不是 Rust 退步。
- **缺料分析**（连续播放：块 k 必须在已播完前 k−1 块音频前到达）：
  所有实现 RTF>1，**连续播放必然断供**。xlong 最坏余量：Rust CPU
  −182.4s（第 6 块迟到 182s）、Python CPU −18.2s、Rust Metal −42.5s。
  结论：当前两实现都不能裸奔流式听书，需要"先缓冲整段/多段再播"或把
  RTF 压到 1 以下。平均 RTF 不能替代该逐块分析。
- **Talechime Engine 完整后端路径**（真实导入音色、含每请求参考特征
  重新提取、请求/PCM 队列容量 1、无播放器无 sleep）：

| 设备 | prepare (s) | 首块 (s) | 总时 (s) | RTF | 状态 |
| --- | --- | --- | --- | --- | --- |
| Metal | 131.1† / 后续 CPU 对照 28.9 | 2.69 | 19.9 | 2.55 | 5/5 Completed |
| CPU | 28.9 | 10.63 | 79.0 | 10.13 | 5/5 Completed |

  †Engine-Metal 首次运行遇磁盘页入；Engine-CPU 后跑（缓存热）。
  collect-all（整段收集）Metal 21.9s / RTF 2.69。Engine 路径与计算库
  路径的差值（首块 2.69 vs 2.49）即音色参考重提取 + 会话/队列开销，
  约 0.2s。

## 五、资源

峰值 RSS 双源一致（`/usr/bin/time -l` 与 Python `ru_maxrss` 互差 <0.5%；
macOS 上 ru_maxrss 单位为字节）。

| 场景 | Rust CPU | Rust Metal | Python CPU |
| --- | --- | --- | --- |
| 生产单管线（冷启动组，模型+前端+一次请求） | 5.17 GB | 5.48 GB | 7.87 GB |
| Talechime Engine（含运行时） | 6.18 GB | 6.31 GB | — |
| warm 基准模式（双模型驻留，非生产） | ≤9.48 GB | ≤9.78 GB | 7.87 GB |

GPU 内存：**不可得**——candle 0.11 未暴露 Metal 分配器查询，
`powermetrics` 需要 sudo；如实标注为未测量。模型驻留与请求峰值的进一步
拆分受工具限制（ps 时序采样首版失效，见方法局限），未重跑矩阵：
峰值结论不受影响，"驻留 vs 请求增量"留作后续。

## 六、敏感性与额外实验（单列，不入主表）

- **默认线程 vs 8 线程**（long warm，3 轮）：Rust 默认（rayon 12 线程）
  Flow 33.79s vs 8 线程 32.94s（+2.6%）；Python 默认（torch 自选）4.05s
  vs 8 线程 4.06s。线程数在 8–12 区间不敏感，主表用 8 不构成偏置。
- **MPS 强制探测**：官方模块 `.to('mps')` 后在官方代码内崩溃
  （`torch.cat(): all input tensors must be on the same device`），
  证据 `runs/cv3-bench/mps-probe.json`。不作为对比行。
- 未做（明确）：torch.compile、量化、降精度、减采样步数、CUDA 编译。

## 七、有效性统计

48 个作业 0 失败；201 行逐轮 CSV 无 NaN；全部轮次 PCM 非空、有限、
24 kHz 单声道。Rust 侧所有轮正常 EOS（`hit_limit`=0，final 块齐）；
Engine 5+5 轮全部 `Completed`。Python 官方 API 不暴露 EOS/终态与 token
上限标志（其生成器无此信息），流式轮的"完成"记为未知而非失败——这是
接口不对称，不是异常。采样 token 数不对称已逐表列明。

## 八、与历史 RTF 数字的关系（计时边界差异）

历史"Metal 非流式 RTF 1.37"用预提取特征且文本不同；"stream_demo 流式
RTF 2.19"计时始于会话建立之后（不含加载/特征/预填充），且当时 hop 状态
与文本不同。三者计时边界互不相同，不可直接换算；本轮同边界重测为准
（Metal 非流式 1.62–2.81、流式 2.08–4.12，文本更长）。

## 九、下一步优化优先级（仅排序与验证方法，本轮不实施）

1. **Flow/DiT CPU GEMM**（占端到端 76%–90%，对 Python 差 4.5–5.6×）：
   首选实验是启用 candle 的 `accelerate` feature（macOS Accelerate 框架
   GEMM）重跑本组件基准，验证能否收敛到 torch CPU 水平；其次核对
   candle-metal-kernels 的 DiT 注意力融合度。
   （已实施，见 §10 与 §11：Rust CPU 路径追平并略超官方，实验特性
   `cosyvoice3/accelerate` 默认关闭，转正需试听 gate。）
2. **HiFT 卷积**（差 2.5–4.7×）：对比 candle conv1d 与 torch conv 的
   im2col/直接卷积路径，长序列差距更大提示布局/复用问题。
3. **Metal Flow 长序列反超问题**：短序列 Metal≈torch CPU 但长序列慢
   1.4×，指向大序列下的内核启动/带宽瓶颈，可抓单个 Euler 步做
   Metal-vs-CPU 剖析。
4. **流式连续性**：RTF>1 时缺料不可避免；产品路径应先落地"预缓冲整段
   后播"（现有 AfterChapterReady 暂存即为此设计），再谈实时。
5. **冷启动**：Rust 已具数量级优势；Python 侧 126s 权重加载是官方
   `torch.load` 行为，非本仓库可控。

## 范围外发现（未修，留证）

`default_home_migration_keeps_hello_lightweight_and_legacy_voices_visible`
（`crates/talechime/tests/protocol.rs`）在今日基准收尾阶段持续超过其 5 秒
门限：带遗留 `.novel-tts` 数据的首次协议运行阻塞 4.4–11s 且 CPU≈0
（纯 HOME 首跑 0.05s、同 HOME 第二次 0.01–0.04s、仅 `.novel` 配置无数据
0.05s），debug/release、默认与 `cosyvoice3-metal` feature 组合均复现，
且在不含本轮任何改动的默认 feature 构建上同样失败——与基准改动无关
（本轮对二进制源码零修改，唯一被跟踪改动是 talechime 的 dev-dep）。
现象是阻塞 I/O 等待（锁/fsync），今日长时间大流量 I/O 后出现，与
[后端接入记录](cosyvoice3-backend-2026-10-17.md)当日“并行真模型运行时
超过 5 秒、事后复验通过”的观察同源。留待单独排查，不在本轮修改。

## 复现

```sh
# 工具（本轮新增）
#   crates/cosyvoice3/examples/bench.rs            Rust 计算库基准
#   crates/talechime/examples/cosyvoice3_engine_bench.rs  Engine 完整路径
#   tools/tts/cosyvoice3_bench.py                  官方 PyTorch 基准
#   tools/tts/cosyvoice3_bench_matrix.py           矩阵编排（含 RSS 采样）
#   tools/tts/cosyvoice3_bench_summarize.py        汇总（summary.json/rounds.csv）

cargo build --release -p cosyvoice3 --features ort,metal --example bench
cargo build --release -p talechime --features cosyvoice3-metal \
  --example cosyvoice3_engine_bench --bin talechime
# 音色导入（Engine 基准前置，一次性）
target/release/talechime --backend cosyvoice3 --model 0.5b \
  --model-dir runs/cv3-backend-resources voices import bench --name bench \
  --text '希望你以后能够做的比我还好呦。' \
  runs/reference/CosyVoice/asset/zero_shot_prompt.wav

# 全矩阵（≈3h；首作业预热文件缓存；逐作业 JSON+time 日志进 runs/cv3-bench/）
python3 tools/tts/cosyvoice3_bench_matrix.py --group all
python3 tools/tts/cosyvoice3_bench_summarize.py runs/cv3-bench

# 单点示例（与 Python 同边界）
RAYON_NUM_THREADS=8 target/release/examples/bench runs/cv3-converted \
  runs/cv3-case-long metal warm --rounds 5 --threads 8 --label manual \
  --reextract --wav-out runs/cv3-bench/manual.wav --out runs/cv3-bench/manual.json
PYTHONPATH=runs/reference/CosyVoice:runs/reference/matcha \
  runs/cv3-venv/bin/python tools/tts/cosyvoice3_bench.py \
  --model_dir runs/cv3-model --case_dir runs/cv3-case-long --mode warm \
  --rounds 5 --threads 8 --label manual --out runs/cv3-bench/manual-py.json
# 官方新鲜 hop 首包（无 warmup）×3
PYTHONPATH=runs/reference/CosyVoice:runs/reference/matcha \
  runs/cv3-venv/bin/python tools/tts/cosyvoice3_bench.py \
  --model_dir runs/cv3-model --case_dir runs/cv3-case-long --mode stream \
  --rounds 1 --no_warmup --threads 8 --label fresh-hop \
  --out runs/cv3-bench/stream-python-cpu-long-freshhop-0.json
```

原始数据：`runs/cv3-bench/*.json`（49+3 个逐作业文件，含每轮原始值、
RSS 峰值、stderr 标记）、`rounds.csv`（201 行）、`summary.json`、
`*.time.txt`（/usr/bin/time 全量输出）、矩阵日志 `runs/cv3-bench-matrix.log`；
代表性音频 `runs/cv3-bench/audio-rust-metal-xlong-stream.wav`、
`audio-python-xlong-warm.wav`。数值等价门禁状态与权重 SHA-256 见
[本机对照记录](cosyvoice3-local-parity-2026-10-17.md)。

## 十、同日后续：借鉴 IndexTTS 的热路径位级等价优化与 ABBA 复测

对照 [IndexTTS 推理性能排查与优化](../indextts/indextts-candle-performance-2026-10-17.md)（`b98e17b`）已验证的优化模式，在不修改 native 依赖（`Cargo.lock` 不变）、不放宽任何数值门禁的前提下，对 `crates/cosyvoice3` 实施以下 **100% 位级等价（bit-exact）** 热路径优化：

### 1. 已保留的位级等价优化

| # | 位置 | 改动 | 原理 | 语义与数值影响 |
| --- | --- | --- | --- | --- |
| O1 | `flow/embeddings.rs` / `flow/dit.rs` / `llm.rs` / `vocoder/source_module.rs` | `LinT2d`：将 DiT（22 层 `to_q/k/v/out`、FFN `linear1/2`、`input_embed.proj`、`proj_out`）、LLM（24 层 `q/k/v/o_proj`、`gate/up/down_proj`、`llm_decoder`）与 `SourceModuleHnNSF.l_linear` 的 3D×2D 线性投影在加载期预转置并物化为连续权重 `weight_t`；`TimestepEmbedding` 的 `inv_freqs` 表同样加载期预置 | 消除每次调用的 `.t()?.contiguous()?` 权重重排拷贝与内核派发（DiT 单次求解 10 步 × ~156 处投影）；inv_freqs 免去每次重建频率向量 | 无（与原 `Linear` 逐位一致） |
| O2 | `flow/flow_matching.rs` / `flow/dit.rs` | `DiT::prepare_solve` + `forward_prepared`：将 Euler 循环内步间不变的 CFG 条件拼接 `[cond_t, mu_t, spks_t]`、RoPE `(cos, sin)` 表、时间表 `t_span` 标量提取外提至 10 步循环前准备一次 | 避免 10 个 Euler 步重复构造零张量、拼接条件与重算三角函数表；求解结束即释放，无跨请求状态 | 无 |
| O3 | `flow/dit.rs` | 注意力掩码预构建为加性偏置（`keep_mask_to_additive`）+ 单次 `broadcast_add`；非流式且 `mask_all_ones` 时直接跳过掩码核 | 原实现在 10 步 × 22 层重复对 `[2, 16, T, T]` 执行非连续广播的 3 操作数 `where_cond_strided`；流式模式在 `prepare_solve` 中一次性转为 `[1, 1, T, T]` 加性掩码（`0.0` / `-inf`），每层仅一次 `broadcast_add` | 无（单测与 `stream_probe` 验证 softmax 输出逐位一致） |
| O4 | `flow/embeddings.rs` / `flow/dit.rs` | `AdaLayerNormZero::forward_act` / `AdaLayerNormZeroFinal::forward_act`：每步 `t_emb` 的 `silu` 激活在块循环外只算一次，并以 `[B, 1, 6*D]` `narrow` 切片替代 `chunk(6)` | 消除每步 22 次重复 `silu(&t_emb)` 与多余 `.unsqueeze(1)` 包装 | 无 |
| O5 | `llm.rs` | KV cache 按注意力布局保存展开后的 `k_t: [B, H, D, T]` 与 `v: [B, H, T, D]`；解码步（`t == 1`）以零拷贝 `reshape` 代替 `.transpose(2, 3)?.contiguous()?` | 避免每步对完整历史 key 重复转置与 `repeat_kv` 扩展 | 无 |
| O6 | `llm.rs` | 加载期在目标设备预置 `sos_emb`、`task_emb` 与 `eos_mask`；`sample` 单步仅做一次 `softmax_last_dim(&logits)?.to_vec1()?` 同步回读；`nucleus_sample` 用 `select_nth_unstable_by` 部分排序 `top_k` | 消除 `ignore_eos` 步的“GPU→CPU 回读改值→CPU→GPU 上传→GPU softmax→第二次 GPU→CPU 回读”双同步阻塞，以及 RAS 回退时的重复 softmax | 无（单测覆盖 `ignore_eos` 与 `top_k` 边界） |
| O7 | `activations.rs` / `vocoder/*` | `Snake` 预整形 `[1, C, 1]` 并预计算 `inv_alpha`；`SineGen2` 预置 `harmonics`/`rate` 并消除 `uv` 的 9 通道冗余广播；最近邻整数倍上采样改用 `unsqueeze + broadcast_as + reshape` 替代每次构造索引张量 `index_select`；`CausalConv1d` 用 `pad_with_zeros` 替代手工构造零张量拼接；`HiFTStft` 以 4 路步进窄视图拼接替代逐帧 `narrow + stack`，`HiFTiStft` 以设备侧 `index_select` 镜像谱替代 CPU 三重循环拷贝，并缓存 `win_sq` | 减少 HiFT 声码器与 STFT/iSTFT 每次推理的张量分配、CPU↔GPU 往返和逐帧派发 | 无 |

**已验证并撤回（无明确收益，如实记录）**：
- `rotate_half` 改用 `narrow` + `Tensor::cat`（不 `squeeze`）：在 Metal 上因 `cat_contiguous` 触发 `d2 = 1` 的 2D `copy2d` 核，实测 `comp_long_metal` 从 3.06 s 退化至 3.42 s，已撤回。
- `PreparedRotary` 预置 `swap_idx` + `sin_signed` 并用 `index_select` 替代 `rotate_half`：实测 Metal `comp_long_metal` 3.58 s vs 原版 3.59 s（处于同一热状态噪声带），撤回以保持结构简洁。

### 2. ABBA 性能对照（同机、串行、固定 seed 42、`long` 样例）

保存优化前基线二进制 `runs/cv3-perf-opt/before-bench`（SHA-256 `e339f684…`）与最终优化二进制 `runs/cv3-perf-opt/final-bench`（SHA-256 `78d5f688…`），按 **A1 → F1 → F2 → A2** 顺序串行运行（每次间隔 3 s，包含完整预热轮）。汇总数据保存在 `runs/cv3-perf-opt/abba_summary.json`：

| 模式 / 设备 | 指标 | 优化前均值 (A1/A2) | 优化后均值 (F1/F2) | 变化 |
| --- | --- | ---: | ---: | ---: |
| **受控组件（Metal，184 tok）** | Flow/DiT | 5.960 s | **3.642 s** | **−38.9%**（已快于官方 PyTorch CPU 的 4.05 s） |
| **受控组件（Metal，184 tok）** | HiFT | 0.696 s | **0.628 s** | **−9.7%** |
| **受控组件（CPU，184 tok）** | Flow/DiT | 24.890 s | **16.238 s** | **−34.8%** |
| **受控组件（CPU，184 tok）** | HiFT | 1.578 s | **1.544 s** | **−2.2%** |
| **非流式端到端（Metal warm，241 tok / 9.64 s 音频）** | LLM 解码 | 3.752 s | **2.978 s** | **−20.6%**（12.4 ms/tok） |
| **非流式端到端（Metal warm，241 tok / 9.64 s 音频）** | Flow/DiT | 10.307 s | **5.133 s** | **−50.2%** |
| **非流式端到端（Metal warm，241 tok / 9.64 s 音频）** | HiFT | 1.078 s | **0.898 s** | **−16.7%** |
| **非流式端到端（Metal warm，241 tok / 9.64 s 音频）** | 生产 `synthesize` 总时 | 15.092 s | **9.025 s** | **−40.2%** |
| **非流式端到端（Metal warm，241 tok / 9.64 s 音频）** | **端到端 RTF** | 1.566 | **0.936** | **−40.2%（突破实时线 RTF < 1.0；冷热态最佳单轮 7.17 s / RTF 0.744）** |
| **非流式端到端（CPU warm，241 tok / 9.64 s 音频）** | LLM 解码 | 7.875 s | **6.745 s** | **−14.3%** |
| **非流式端到端（CPU warm，241 tok / 9.64 s 音频）** | Flow/DiT | 30.887 s | **18.334 s** | **−40.6%** |
| **非流式端到端（CPU warm，241 tok / 9.64 s 音频）** | 生产 `synthesize` 总时 | 41.701 s | **28.016 s** | **−32.8%**（RTF 4.326 → **2.906**） |
| **流式（Metal stream，8.96 s 音频，4 块）** | 纯推理首包延迟 | 2.859 s | **2.271 s** | **−20.6%** |
| **流式（Metal stream，8.96 s 音频，4 块）** | 整段流式总耗时 | 24.468 s | **16.516 s** | **−32.5%**（RTF 2.731 → **1.843**） |

### 3. 音频一致性与门禁验证

- **WAV 逐字节一致**：`warm_metal`（SHA-256 `1a7af74e427849cf…`）、`stream_metal`（SHA-256 `54b3e425ab947f5f…`）、`warm_cpu`（SHA-256 `34be135584f2ec03…`）优化前后生成的 WAV 文件 **3/3 逐样本、逐字节 100% 相同**。
- **组件与真模型数值门禁**：`compare` 18/18 全部 PASS（误差精确到小数点后 8 位不变）；`stream_probe`（Metal）逐块样本数与误差（`mel_rmse` 1.394e-4 / 2.689e-4）完全不变；`compare_model`（Metal）`real_first_logits`（1.62e-5）、`real_time_embed`（2.26e-6）、`real_input_embed`（8.58e-5）、`real_estimator`（2.81e-3）、`real_flow_mel`（1.86e-3）、`real_hift_pcm`（3.33e-4）全部 PASS，仅已知 f32 混沌项 `real_tokens_to_pcm`（1.62e-1）保持原状。
- **工程检查**：`cargo test --locked -p cosyvoice3 --features ort,metal --lib --tests --examples`（36 项单测 + 5 项示例测试）通过；`cargo clippy --locked --all-targets -- -D warnings`、`cargo fmt --all -- --check`、`RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items -p cosyvoice3 --features ort,metal --examples` 全部零告警通过。

## 十一、Accelerate（macOS AMX BLAS）CPU 路径

按 §9-1 的建议验证 macOS Accelerate 路径，并于 2026-10-18 由用户决定
**在 macOS 上默认启用**：`crates/cosyvoice3` 的
`[target.'cfg(target_os = "macos")'.dependencies]` 直接为 `candle-core`
声明 `features = ["accelerate"]`（无开关、不经 tts-candle-platform——
平台 crate 是可选共享依赖，在那里默认启用会连带改变 qwen/voxcpm 等
其他适配器的 macOS 数值；且 cargo feature 无法按平台门控，走
`default` 特性会破坏 Linux/Windows 构建）。
它把 Candle CPU 后端的 matmul 从纯 Rust `gemm` crate 换成系统
Accelerate 框架（`cblas_sgemm`/`cblas_dgemm`，驱动 M4 Pro 的 AMX 矩阵单元）。
`Cargo.lock` 仅新增 `accelerate-src 0.3.2`（7 行），Candle 版本未动；
仅影响 CPU 后端，Metal 路径与 Linux/Windows 构建不受影响。
加速二进制存 `runs/cv3-perf-opt/accel-bench`（SHA-256 `400b157a…`）。

### 性能（同机背靠背 A/B，long 样例，RAYON_NUM_THREADS=8，中位数）

| 路径 | 指标 | gemm crate | Accelerate | 变化 |
| --- | --- | ---: | ---: | ---: |
| 受控组件 CPU（184 tok） | Flow/DiT | 13.9 s | **4.8 s** | **−65%（2.9×）** |
| 受控组件 CPU（368 帧） | HiFT | 1.50 s | **1.03 s** | **−31%（1.45×）** |
| 端到端 warm CPU（241 tok） | LLM 解码 | 6.7 s | **3.2 s** | **−52%** |
| 端到端 warm CPU（241 tok） | Flow | 18.3 s | **5.9 s** | **−68%** |
| 端到端 warm CPU（241 tok） | 端到端总时 / RTF | 30.4 s / 3.15 | **10.5 s / 1.089** | **−65%** |

对照官方 PyTorch CPU（long：184 tok / 8.39 s / RTF 1.17，同为
Accelerate BLAS 后端）：按每 token 成本计，Rust CPU + Accelerate
**43.6 ms/tok vs 官方 45.6 ms/tok，首次纯 CPU 反超官方**（约 −4%）；
受控组件 Flow/DiT 4.8 s 仍略慢于官方 4.05 s（1.19×，剩余为算子级
im2col/逐元素开销，非 GEMM 差距）。

线程敏感度：Accelerate 默认线程策略最优；`VECLIB_MAXIMUM_THREADS=8`
使 Flow 退化至 5.0–5.6 s，`=1` 退化至 6.4–7.3 s——不要限线程。
另注意：若不设 `RAYON_NUM_THREADS`（rayon 取 12）同时叠加 BLAS 自身
线程，会出现明显争用（一次实测 Flow 反而变慢 2.6×）；生产接入时
应固定 rayon 线程数。

### 数值（非位级一致，性质同 IndexTTS `--decode-on-cpu` 实验）

- 组件门禁 `compare`：18/18 PASS；误差仅微移（`dit` 1.14e-6 → 1.24e-6；
  `f0` F64 dgemm 仍精确为 0）。
- 真实权重 `compare_model` CPU：通过/失败集合与 gemm 路径**完全相同**
  （仅两个已知 f32 混沌项 `real_estimator` 3.49e-3、`real_tokens_to_pcm`
  1.60e-1 保持 FAIL，且后者反而小于 gemm 路径的 3.78e-1）；
  `real_hift_pcm` 从 3.23e-5 改善到 1.29e-5。
- 采样轨迹：seed 42 下 token 数 241 不变（logits 差异未翻转任何
  nucleus 采样边界）。
- 波形差异（同 token、gemm vs Accelerate 输出）：max_abs 0.826 /
  RMSE 0.061 / 相对峰值 0.83——与官方对照中已归因的 F0 相位漂移放大
  机制同源（mel 级 ~3e-3 求和序差 × 10 步 Euler × F0 相位），不是
  实现缺陷。

### 结论与边界

- **macOS 默认启用（2026-10-18 用户拍板）**；非位级一致的数值差异
  （上文已量化并归因）随该决定一并接受，主观试听验收仍未执行——
  音频在 `runs/cv3-perf-opt/warm_cpu_accelbench.wav`（对照
  `warm_cpu_gemmbench.wav`）。§10 的位级等价 ABBA 与 WAV 逐字节一致
  结论属于 `gemm` 路径（仍是 Linux/Windows 的默认）；macOS 默认构建
  的复跑数值以本节为准。
- 附注：O5 的 KV cache 布局以内存换拷贝（缓存按 query 头数展开，本模型
  14 头 GQA 使缓存体积为 KV 头数的 7 倍——与优化前一致，变化只是免去
  每步重排）。
- 36 项单测与 `clippy -D warnings` 在默认（含 Accelerate）与
  `--features ort,metal` 组合下全部通过。

