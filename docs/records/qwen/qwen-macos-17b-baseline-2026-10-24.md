# Qwen3-TTS 1.7B macOS M4 Pro 基线（2026-10-24）

issue #5 / Mac goal 第 5 步的首批补测：0.6B Base Metal 优化验收（2026-10-18）之后的
**模型变体覆盖**。本轮只测量，无优化、无默认配置改动。Base 与 CustomVoice
分开记录，0.6B 结论不外推。

## 一、设备与执行身份

| 项 | 值 |
| --- | --- |
| 设备 / 系统 | Apple M4 Pro（12 逻辑核，16 GPU 核，24 GiB），macOS 26.6.2（25G83） |
| 源码 | `42a7887`（工作区干净，除 goal 计划文档未跟踪） |
| 工具 | xtask perf；探针 features `qwen,qwen-metal,performance-tools`（`target/perf/qwenb-*-build.json`） |
| 数值路径 | Metal `build:candle-metal;host-cpu-accelerate`；精度 BF16（Metal） |
| 线程 | RAYON_NUM_THREADS=12 |
| 资源 | `~/.talechime/resources/qwen/models/{1.7b-base,1.7b-customvoice}`（revision `fd4b25…` / `0c0e30…`），`prepare` 生产校验通过；1.7b-base 音色 `custom:aibaoliang` 的 reference.wav + voice.json 以 conditioning 收据冻结进 job |
| 协议 | 3 进程 × 1 预热 + 5 热测；语料同 OmniVoice/MOSS 基线短句（sha `f29b49d8…`）；seed 42；计时边界 `production_synthesis_stream_consumer_start_to_completed/1`；`native_streaming_first_pcm` |

## 二、结果

| 案例 | 均值 s | 音频 s | RTF | 首 PCM | 确定性 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1.7B Base（ICL 克隆 aibaoliang）Metal | 8.493 | 8.560 | **0.992** | 2.030 s | 18 轮同 sha |
| 1.7B CustomVoice 预置 uncle_fu Metal | 6.302 | 8.320 | **0.758** | 0.478 s | 18 轮同 sha |
| 1.7B CustomVoice uncle_fu + style Metal | 6.549 | 8.720 | **0.751** | 0.481 s | 18 轮同 sha |

全部 attempt terminal success、preflight 通过、cleanup_verified；进程均值区间
均 < ±0.5%（8.487–8.499 / 6.302–6.303 / 同量级）。

观察：

- **Base 1.7B Metal RTF 0.992，贴着实时线**（余量 <1%）。相对 0.6B 的已优化
  Metal 路径明显更重：模型翻倍 + 每次请求的 ICL 克隆 prompt（首次生成
  `prompt-encoder-rope-v2.json` 缓存后热轮仍带克隆 prompt 前缀）。首 PCM
  2.03 s 也主要是克隆 prompt 编码 + 首帧。**未做热点剖析**；若产品化需要
  余量，候选方向是 Metal 端 CP 图（CUDA 已有，Metal 无）与克隆 prompt 复用
  边界，属后续立项，不在本轮。
- **CustomVoice 明显快于 Base**（0.75–0.76 vs 0.99）：预置 speaker 无克隆
  前缀。style instruct 只增加少量 token（0.751 vs 0.758，同档）。
- 汇总表登记：三案例均「已测量无改动」。

## 三、补充：0.6B CustomVoice（2026-10-24 同日补齐）

资源侧发现 0.6B CustomVoice 使用旧版平铺布局（`~/.talechime/resources/qwen/`
根目录，`resources.json` 清单）：本地 7/8 文件 SHA 已吻合，仅缺 `config.json`
（旧下载留有 `config.corrupt` 痕迹）；按 pinned URL
（`Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c`）下载单文件、SHA 复验一致后经
harness `resource-import` 就位，`prepare` 复核 `resources_verified`。

| 案例 | 均值 s | 音频 s | RTF | 首 PCM | 确定性 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.6B CustomVoice 预置 uncle_fu Metal | 11.774 | 14.090 | **0.836** | 0.821 s | 18 轮同 sha |

注意：0.6B CustomVoice 的输出音频（14.09 s）明显长于 1.7B（8.32 s），两者
绝对耗时不可比，RTF 各自成立。**至此 goal 第 5 步的变体矩阵在 Metal 侧补齐**
（1.7B Base / 1.7B CV 预置 / 1.7B CV 风格 / 0.6B CV）。

## 四、CPU 对照与未完成

- Qwen 1.7B Base CPU（Accelerate）对照一轮：按 goal §三 执行（结果见
  `target/perf/qwenb-17bbase-cpu-short/`，登记于 goal 文档）。
- 1.7B VoiceDesign 档未测（goal 未列必须项）。

## 五、原始数据

`target/perf/qwenb-*`：plan/build/prepared/job、各案例 `results.json`（权威逐轮）、
attempts 音频与 preparation/diagnostics；job 生成脚本
`target/perf/qwenb-make-jobs.py`（含 `custom:` 音色 conditioning 收据的冻结方法）。
