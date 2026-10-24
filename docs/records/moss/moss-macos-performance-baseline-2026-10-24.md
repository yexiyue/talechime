# MOSS macOS M4 Pro 性能基线（2026-10-24）

issue #5 Mac 侧第一批的正式测量：MOSS 全部产品可用档位在当前主线建立真实模型
性能基线。**本记录只测量，没有做任何优化，也没有改变任何默认配置**；结论是
Nano Candle 与 Realtime 1.7B 在 M4 Pro 上都快于实时，按 issue 口径「测量后没有
值得保留的收益，也算完成该轮调查」。

## 一、设备、源码与执行身份

| 项 | 值 |
| --- | --- |
| 设备 | Apple M4 Pro，12 逻辑核（8P+4E），16 GPU 核，24 GiB 统一内存 |
| 系统 | macOS 26.6.2（25G83），aarch64 |
| 源码 | 仓库 `42a7887`（工作区干净，唯一未跟踪文件为本 goal 计划文档） |
| 工具 | xtask perf harness；探针 `performance_probe` / `performance_catalog`，features `moss,moss-nano-candle,moss-nano-candle-metal,moss-candle,moss-candle-metal,performance-tools`（收据 `target/perf/mossb-{catalog,probe}-build.json`） |
| 数值路径 | CPU `build:candle-accelerate;ort-if-selected`（当前主线 Candle 已默认启用 Accelerate）；Metal `build:candle-metal;host-cpu-accelerate`；operator usage 未测 |
| 精度 | nano-candle F32（BF16 checkpoint 加载时转 F32，`crates/moss-tts/src/nano.rs:37`）；nano ONNX F32（ORT CPU）；local/realtime 1.7B Metal 模型 F16 + codec F16（`moss/candle/runtime.rs` `model_dtype`） |
| 线程 | RAYON_NUM_THREADS=12（runner 显式覆盖）；VECLIB_MAXIMUM_THREADS 未设置 |
| 资源 | `~/.talechime/resources/moss/`，四套模型逐一经 `prepare --backend moss --model <id>` 生产校验（收据 `target/perf/mossb-prepared-*.json`） |

计划 `target/perf/mossb-plan.json`（macos，3 进程 × 1 预热 + 5 热测）。
语料与 OmniVoice 2026-10-22 基线完全相同：短句
`target/perf/omni-input/corpus-short.txt`（82 字节「山风…」句，sha
`f29b49d8…`）；接续案例同语料 `continuation=true`。seed 42；音色 nano 系
Weiguo（内置固定音色）、1.7B narrator（目录内唯一档位）。计时边界
`production_synthesis_stream_consumer_start_to_completed/1`；MOSS 全档
`native_streaming=true`，首 PCM 语义为 `native_streaming_first_pcm`。

## 二、测量结果

统计为进程均值的均值；区间为三进程热测均值的范围。RTF = 合成耗时 ÷ 音频时长。

| 案例 | 均值 s | 音频 s | RTF | 进程均值区间 | 首 PCM | 输出确定性 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| nano-candle Metal 短句 | 1.417 | 6.800 | **0.208** | [1.415, 1.419] | ≈90 ms | 18 轮同 sha |
| nano-candle CPU 短句 | 2.096 | 6.800 | **0.308** | [2.087, 2.101] | ≈131 ms | 18 轮同 sha |
| nano-candle Metal 接续 | 1.408 | 6.800 | **0.207** | [1.403, 1.411] | ≈88 ms | 18 轮同 sha |
| nano-candle CPU 接续 | 1.408 | 6.800 | **0.207** | [1.399, 1.412] | ≈88 ms | 18 轮同 sha |
| nano（ONNX）CPU 短句 | 1.402 | 8.080 | **0.174** | [1.400, 1.404] | ≈93 ms | 18 轮同 sha |
| realtime-1.7B Metal 短句 | 7.171 | 8.400 | **0.854** | [7.161, 7.186] | ≈394 ms | 18 轮同 sha |

所有成功 attempt 均 `cleanup_verified`、terminal success、preflight 身份校验
通过；逐轮输出 sha 在每个案例内完全一致（固定 seed + 固定音色，含 1.7B narrator
——实测 seed 42 下音色确定，不因轮次漂移）。

观察：

- **Nano Candle 无需优化**：Metal RTF 0.21、CPU 0.31 都快于实时一个量级；
  接续不劣化（CPU 接续反而更快，1.408 vs 2.096，与上下文复用一致）。
- **Nano ONNX（ORT CPU）与 Nano Candle Metal 大致同档**（RTF 0.174 vs
  0.208），但两者音频时长不同（8.08 vs 6.80 s，不同模型不同输出），只能单列
  对照，不构成同输出提速结论。
- **Realtime 1.7B RTF 0.854**：低于实时但余量不大（+17% 负载即越过 1）；
  本轮未做热点剖析，若后续产品化压力增大可再立项。
- Metal 相对 CPU 的差距远小于其他后端（nano 系 1.48×，对比如 OmniVoice 4.7×），
  与 Nano 主干为小模型、CPU 亦有 Accelerate 有关。

## 三、阻塞与不适用项（如实登记，不算已测量）

| 项 | 状态 | 证据 |
| --- | --- | --- |
| local-1.7B Metal | **阻塞：GPU Insufficient Memory** | 模型初始化即失败：`Metal error … kIOGPUCommandBufferCallbackErrorOutOfMemory`，exit 1，3 进程 0 轮（`target/perf/mossb-local17b-metal-short-v2/`）。与 2026-10-08 记录一致：24 GiB 统一内存装不下 F16 模型 + codec。产品未暴露 CPU 档（plan 中 `not_applicable`），按 CPU/资源政策不强行 |
| nano-candle 长语料（Metal/CPU） | **阻塞：切分缝隙不一致** | 预热轮即失败 `consumer ranges differ from production segmentation`（`performance_support/probe.rs:376`）：探针用 `backend.next_segment()` 重放生产切分，与实际 `stream()` 交付的 range 不一致。短句单段不受影响。这是 MOSS nano 两条公开 seam 的真实分歧信号（`next_segment` vs `stream` 内部切分），影响产品行为与测量归因，应单独立项排查；本基线不放宽该校验 |

## 四、与历史记录的关系

- 2026-10-08（TRNovel 同步）Realtime 1.7B 有流式修复记录，无正式 RTF 协议数据；
  本轮是当前主线、生产请求边界下的首份正式基线。
- 2026-10-20 的 nano CPU/Metal 仅是接线 smoke（1 请求）；本轮 3×6 协议取代之，
  资源收据 `mac-tool-nano-prepared` 的 resource_sha256（`ca041c97…`）与本次
  重新 prepare 的 digest 完全一致，权重未变。
- 汇总表登记：Nano Candle「已测量无改动」；Nano ONNX「已测量无改动（单列）」；
  Realtime 1.7B「已测量无改动」；Local 1.7B「阻塞」；长语料「阻塞（切分缝隙）」。

## 五、工具使用备忘（供后续 Mac 基线复用）

1. cpu provider 的 `conditions.hardware` 不带 `; Metal=…` 后缀、`driver`
   填 `not_applicable`；metal provider 才追加 Metal 身份与 OS 串（沿
   `environment.rs` 实测格式）。
2. `conditions.parameters` 必须包含 `prepared` 事件展开的**全部**有效参数：
   有默认生成参数的模型（1.7B 的 `generation.max_duration`、local 另有
   `instruction`/`language`）要一并冻结，否则 preflight 判负。
3. baseline 观察模式下 `diagnostics.trace_complete=false` 是设计行为
   （仅 trace 模式为 true），不影响 attempt 成败。
4. digest 序列化规则（compact + 键排序 + 原生 UTF-8）已对照 2026-10-20 smoke
   冻结值验证一致；job 生成脚本 `target/perf/mossb-make-jobs.py` 可作模板。

## 六、原始数据

`target/perf/mossb-*`：plan/build/prepared/job JSON、各案例
`results.json/results.json`（权威逐轮）、attempts/*/round-*.wav、
preparation/diagnostics/capture。local-1.7B 的两次失败运行
（`-short` 参数补齐前、`-short-v2` OOM）均保留。
