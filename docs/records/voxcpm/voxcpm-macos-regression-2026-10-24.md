# VoxCPM2 macOS 回归与 CPU 基线（2026-10-24，M4 Pro）

issue #5 / Mac goal 第 7 步（最后一项）：当前共同源码版本（`42a7887`）的
Metal 回归核对 + 一直缺失的 CPU 正式基线。统一协议 3×6（3 进程 × 1 预热 +
5 热测），生产请求边界，与本周其余 Mac 基线同协议同语料（短句 sha
`f29b49d8…`，seed 42，音色 narrator）。只测量，无优化、无默认改动。

## 一、执行身份

探针 features `voxcpm,voxcpm-metal,performance-tools`（收据
`target/perf/voxb-*-build.json`）；数值路径 Metal
`build:candle-metal;host-cpu-accelerate` / CPU `build:candle-accelerate`
（当前主线 Candle 默认启用 Accelerate）；精度 Metal `q8-weights;f16-acoustic`、
CPU `q8-weights;f32-acoustic`；RAYON_NUM_THREADS=12；资源
`~/.talechime/resources/voxcpm/`（`prepare` 生产校验 `resources_verified`，
收据 `target/perf/voxb-prepared-model.json`）；`native_streaming_first_pcm`。

## 二、结果

| 案例 | 均值 s | 音频 s | RTF | 进程均值区间 | 首 PCM | 确定性 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 2b-q8_0 Metal 短句 | 4.566 | 5.840 | **0.782** | [4.564, 4.568] | 0.271 s | 18 轮同 sha |
| 2b-q8_0 CPU（Accelerate）短句 | 10.280 | 5.840 | **1.760** | [10.252, 10.318] | 0.641 s | 18 轮同 sha |

全部 attempt terminal success、preflight 通过、cleanup_verified。

## 三、回归判定

- **Metal 无回归，且优于历史**：2026-10-08（TRNovel 同步、旧码）同短句
  Candle Metal RTF 0.922；10-18/19 tile 优化（-6.9% 授权例外）后推算约 0.86。
  当前主线统一协议测得 **0.782**——方向与幅度都优于历史值。三方口径差异
  （旧码 adapter 内计时 vs 本轮生产请求边界、Accelerate 主机路径默认启用）
  使数字不可相减，但「无回归」结论成立。
- **CPU 基线首次正式建立：RTF 1.760**。与 2026-10-08 旧码 CPU 3.689 相比
  大幅改善——主要来自当前主线 Candle 默认启用 Accelerate（BLAS 路径），
  与该记录「未额外开启 Accelerate feature」的条件差异是方向性解释，不是
  相减关系。仍高于实时线，按 goal §三 登记为不优化（Metal 是产品路径）。
- 汇总表登记：Metal「回归通过」；CPU「已测量无改动（基线建立）」。

## 四、原始数据

`target/perf/voxb-*`：build/plan/prepared/job 收据、两案例 `results.json`
（权威逐轮）、attempts WAV 与 preparation/diagnostics。
