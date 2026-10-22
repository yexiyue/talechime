# OmniVoice 0.6B 性能基线（2026-10-22，Apple M4 Pro）

首次在当前主线上为 OmniVoice 0.6B 建立真实模型性能基线。本记录只报告已测数据、
已定位的量级关系和未完成项；**没有做性能优化，也没有改变任何默认配置**。
人工试听与 ASR 回读未执行，仍为 pending。

## 一、设备、源码与执行身份

| 项 | 值 |
| --- | --- |
| 设备 | Apple M4 Pro，12 逻辑核（8P+4E），16 GPU 核，24 GiB 统一内存 |
| 系统 | macOS 26.6.2（25G83），aarch64 |
| 源码 | 仓库 `f9e1d4f`（工作区干净，`dirty_sha256=e3b0c442…`） |
| 探针 | `performance_probe`，features `omnivoice,omnivoice-metal,performance-tools` |
| 数值路径 | CPU `build:candle-rust;ort-if-selected;operator-usage-unmeasured`；Metal `build:candle-metal;host-cpu-rust;operator-usage-unmeasured` |
| 精度 | CPU 全 F32；Metal stage0 F16、audio/stage1 F32（`DTypeSpec::Auto` 的设备规则） |
| 线程 | `RAYON_NUM_THREADS=12`（runner 显式覆盖）；`VECLIB_MAXIMUM_THREADS` 未设置 |
| 模型资源 | `omnivoice/models/0.6b/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/`，10 个生产清单文件按 SHA-256 校验通过 |
| 音色 | `narrator`（模型内建），无参考音频、无 conditioning 文件 |
| 参数 | 默认值：`num_step=32`、`guidance_scale=2.0`、`language=zh`、`speed=1.0`；`seed=42`；`continuation=false` |

资源校验（10 个文件 SHA-256）耗时 6.3–6.6 s，模型加载 0.16–0.26 s，参考预热
（一次短文本整次合成）Metal 3.79 s / CPU 12.74 s；三者都不计入轮次计时。

## 二、协议与语料

- 计划 `target/perf/omni-plan.json`（macos / mac-m4-pro），正式案例为
  **3 个独立进程 × 每进程 1 次预热 + 5 次热测**，预热不计入统计。
- 短语料：`山风吹过松林，星光照亮归途。两个人沿着小路慢慢往前走。`
- 长语料：`crates/voxcpm/tests/fixtures/acceptance-corpus.txt`（781 字节，固定验收语料，原字节未改）。
- 计时边界：`production_synthesis_stream_consumer_start_to_completed/1`，导出写盘在计时外。
- OmniVoice `native_streaming=false`，因此首 PCM 语义为 `segment_delivery_first_pcm`
  （整段交付延迟），不当作原生流式首块。

## 三、测量结果

RTF = 合成耗时 ÷ 实际输出音频时长。统计为进程均值的均值，区间为各进程热测均值范围。

| 案例 | 设备 | 协议 | 合成均值 s | 音频 s | RTF | 进程均值区间 | 首 PCM 均值 s | OS 峰值 footprint |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 短句 | Metal | 3×6 | 5.999 | 5.400 | **1.1110** | 1.1048–1.1158 | 5.999 | 4.07 GiB |
| 短句 | CPU | 3×6 | 27.886 | 5.320 | **5.2418** | 5.0892–5.4655 | 27.886 | 3.20 GiB |
| 长文 | Metal | 3×6 | 59.979 | 50.990 | **1.1763** | 1.1402–1.2250 | 2.385 | 4.12 GiB |
| 长文 | CPU | 1×4（筛选） | 299.85 | 51.380 | **5.8359** | — | 11.51 | ≈3.4 GiB |

长文 CPU 行是筛选协议（1 进程 × 1 预热 + 3 热测），不是正式 3×6 协议，单独标注。

其他观察：

- Metal 相对 CPU 约 **4.7×**（短句 RTF 1.111 vs 5.242），长文同量级（1.176 vs 5.836）。
- 长文 Metal 的首 PCM 是 **2.39 s**（首个语义段），短句是 **6.00 s**（整段），
  两者都是 `segment_delivery_first_pcm`。
- 短句 Metal 与 CPU 输出时长不同（5.320 s vs 5.400 s），属于不同数值路径的正常差异，
  不作等价结论。
- 统一内存峰值不是独占显存；Metal allocation 采集当前为 unavailable，Instruments 为 deferred。

## 四、有效语音检查（非试听替代）

- 每个案例 18 轮（长文 CPU 4 轮）PCM 摘要**逐轮完全一致**，说明固定 seed 下输出确定。
- 输出 24 kHz 单声道 f32：短句 Metal peak 0.500 / RMS 0.109；短句 CPU peak 0.500 /
  RMS 0.073；长文 Metal peak 0.608 / RMS 0.114。
- 100 ms 帧 RMS 的 p10 为 0.0006–0.0225、p90 为 0.110–0.180，过零率 0.088–0.109：
  包络有静音段与语音段起伏，符合语音，不符合 2026-10-07 记录的噪音特征。
- 以上只是有限性、非空、包络与确定性检查，**不等于逐字覆盖、听感或音色验收**；
  人工试听与 ASR 回读仍 pending。

## 五、量级定位：时间几乎都在扩散步循环里

用 `num_step` 做诊断扫描（Metal、短句、筛选协议 1 进程 × 1 预热 + 3 热测）。
步数会改变输出（PCM 摘要与默认不同），因此这**不是加速验收**，只用于定位量级。

| `num_step` | 合成均值 s | RTF |
| ---: | ---: | ---: |
| 32（默认） | 5.999 | 1.1110 |
| 16 | 3.224 | 0.5971 |
| 8 | 1.881 | 0.3484 |

线性拟合：**每步 ≈ 0.168–0.174 s，请求内固定开销 ≈ 0.45–0.54 s**，
即 32 步里约 **92–93%** 的合成时间落在 step 循环内。
要在保持 32 步的前提下把短句 RTF 压到 1.0（合成 5.40 s = 音频 5.40 s），
每步需要降到约 0.155 s，即**每步成本约 -11%**——刚好越过 10% 的保留门槛。
长文是另一种结构（步数相同但每步序列更长），不能直接套用这个比例。

代码侧的对应结构（读码结论，**未经 profiling 验证**）：

- `crates/omnivoice/src/stage0_model.rs` 的 step 循环每步做一次**全序列 forward**
  （文本 embedding 已缓存，但音频位置按 any-order 扩散逐轮变化，没有可复用的 KV 缓存），
  因此成本与步数线性相关。
- 每步还要把打平后的分数（8 codebook × 目标长度 ≈ 1040 列）做一次**全量 argsort**
  以挑出本步要解 mask 的位置；`argsort_descending` 在 Metal 且列数 > 1024 时会
  **回退到 CPU 排序并把张量拷回主机**。这构成每步一次强制主机往返，是候选热点。
- `crates/omnivoice` 未提供细粒度 span，现有 `trace` 观察只能给到 request/segment 级，
  无法区分“forward”与“排序往返”各占多少。

以上三条都是**假设**，需要先独立 profiling（Instruments 或补 stage0 细粒度 span）定位，
再做无插桩 release A/B（≥10%、两组区间不重叠、输出等价），不得直接按假设改实现。

## 六、与历史记录的关系

- `docs/records/platform/metal-tts-efficiency.md`（2026-10-07）有 Metal RTF 1.049，
  计时边界是适配器内部计时且音频经裁剪；本轮是生产请求起止边界，两者口径不同，
  **不能相减**。可以确认的是：当前主线在生产请求边界下 RTF ≈ 1.11，仍高于 1。
- 2026-10-07 记录的 Metal 排序越界修复仍在代码中（`argsort_descending` 的 CPU 回退），
  本轮未单独测量它在每步成本中的占比。

## 七、测量工具使用中发现的问题（不是模型问题）

1. `job.conditions` 必须带 `delivery_boundary`；OmniVoice 填 `segment_delivery_first_pcm`。
2. macOS 上 metal provider 的 `conditions.hardware` 要追加 `; Metal=<身份 JSON>`，
   `driver` 填 OS 串；cpu provider 则是 `driver=not_applicable`。两者不能互相套用。
3. `request.max_pcm_bytes` 用默认 4 MiB 时长文直接失败（`PCM retention budget exceeded`）：
   51 s 音频的 f32 就接近 4.9 MB。本轮长文改用 64 MiB。
4. `timeout_seconds` 给小了（900 s）会在长 CPU 轮次中被 runner 杀掉，表现为
   `cleanup unverified; refuse next measurement`，**不是模型崩溃**。排查先看
   `capture.json` 的 errors 与 job 超时。
5. 测量必须在前台会话里跑；后台 `nohup` 起的 runner 会随会话结束被回收，
   留下只有 `prepared` 的空 attempt。

## 八、未完成项与下一步

未完成：人工试听、ASR 回读、Instruments profile、Metal allocation 采集、CPU 长文正式 3×6 协议、
与官方实现的同条件对照（仓库内没有 OmniVoice 官方基线）。

建议顺序：

1. 给 stage0 step 循环补细粒度 span（或用 Instruments）拆分“forward / 排序往返 / 更新”，
   确认 0.17 s/步 的构成。
2. 若排序往返占比可观，评估避免全量 argsort 的等价实现（例如按分数阈值做 device 侧筛选），
   以“输出等价 + ABBA ≥10% 且区间不重叠”为验收门槛。
3. 若 forward 本身占主导，评估 Metal 端 F16 attention/GEMM 的形状相关优化空间
   （同形状、等价门槛需单独定）。
4. 步数本身是质量参数，不作为优化项；降低 `num_step` 可以换实时，但那是产品取舍，
   需明确音质与验收口径。

原始数据与报告在忽略目录 `target/perf/omni-*`：`results.json`（权威逐轮）、
`attempts/*/round-*.wav`（F32/PCM16）、`preparation.json`、`resource-samples.jsonl`，
以及 `target/perf/omni-*-report/` 的离线报告。
