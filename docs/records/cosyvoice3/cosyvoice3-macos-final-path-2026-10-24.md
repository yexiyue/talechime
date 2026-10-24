# CosyVoice3 macOS 最终路径统一协议复测（2026-10-24）

issue #5 / Mac goal 第 6 步：对当前默认路径（Metal 与 CPU Accelerate）按统一协议
补正式测量。**只测量，无优化、无默认改动**。本记录取代此前 bench 边界数字作为
「Mac 最终路径」的当前结论来源；与 2026-10-17 优化记录的数字**不可相减**（边界不同，
见 §四）。

## 一、资源导入（本轮前置工作）

本机 `runs/cv3-converted/`（2026-10-17 转换产物）与 crate 生产清单
`crates/talechime-backends/src/cosyvoice3/resources.json` 逐一对照：7 个文件
尺寸与 SHA-256 全部一致。经 xtask `resource-inventory`（7/7 ready）+
`resource-import`（身份锁、no-clobber 分阶段发布）导入
`~/.talechime/resources/cosyvoice3/models/0.5b/2512-candle-f64-v1/`；两个本地
音色（review/bench，用户自建数据，不在生产清单内）手工复制到
`voices/cosyvoice3/0.5b/2512-candle-f64-v1/`。`prepare` 复核
`resources_verified`。收据：`target/perf/cv3b-{inventory,import,prepared}.json`。

## 二、设备与执行身份

| 项 | 值 |
| --- | --- |
| 设备 / 系统 | Apple M4 Pro（12 逻辑核，16 GPU 核，24 GiB），macOS 26.6.2（25G83） |
| 源码 | `42a7887`（除 goal 文档外工作区干净） |
| 工具 | 探针 features `cosyvoice3,cosyvoice3-metal,performance-tools`；数值路径 Metal `build:candle-metal;host-cpu-accelerate`、CPU `build:candle-accelerate`；精度全 F32（runtime 固定 `DType::F32`） |
| 线程 | RAYON_NUM_THREADS=12 |
| 协议 | 3 进程 × 1 预热 + 5 热测；短句/长语料与 OmniVoice/MOSS/Qwen 基线相同（sha `f29b49d8…` / `2ac7b201…`）；seed 42；音色 `custom:bench`（reference.wav+voice.json conditioning 收据冻结）；`native_streaming_first_pcm`；计时 `production_synthesis_stream_consumer_start_to_completed/1` |

## 三、结果

| 案例 | 均值 s | 音频 s | RTF | 进程均值区间 | 首 PCM |
| --- | ---: | ---: | ---: | --- | ---: |
| Metal 短句 | 10.038 | 7.100 | **1.414** | [10.028, 10.045] | 2.386 s |
| CPU（Accelerate）短句 | 13.638 | 7.100 | **1.921** | [13.599, 13.687] | 3.033 s |
| Metal 长语料 | 124.096 | 61.670 | **2.012** | [123.911, 124.376] | 2.429 s |

全部 attempt terminal success、preflight 通过、cleanup_verified；各案例内
逐轮输出 sha 完全一致（固定 seed+固定音色确定性成立）。

**结论：当前默认流式最终路径在 M4 Pro 上短句 RTF 1.41、长文 2.01，均高于实时线。**
CPU Accelerate 1.92 同样超线（但显著好于 2026-10-17 非流式 CPU 的 2.906——
不可相减，方向参考）。按 goal §三 政策，CPU 不再投入优化。

## 四、与 2026-10-17 优化记录的口径差异（不可比，逐项列明）

10-17 的「Metal 非流式 RTF 1.566→0.936（长样例 241 tok / 9.64 s 音频）」成立
于：**非流式** `synthesize` 单次调用边界、bench 自备 fixture（固定 prompt，
非生产 zero-shot）、长样例摊薄。本轮统一协议是**生产流式路径**：
`production_synthesis_stream_consumer_start_to_completed`、真实分段交付、
zero-shot 克隆音色（每段重编码参考 prompt，长文 6 段 ≈ 6×2.4 s 仅 prompt 一项）、
短句不摊薄。两条口径都真实，差异方向（流式+分段+零样本开销把 RTF 推回实时线
之上）本身是本轮复测的发现。汇总表登记：**复测完成，已测量无改动**；
若后续要把 cv3 产品化，优化目标已明确：流式分段开销、每段 prompt 复用、
长序列 flow 成本。

## 五、原始数据

`target/perf/cv3b-*`：导入/计划/job 收据、三案例 `results.json`（权威逐轮）、
attempts 音频与 preparation/diagnostics。
