# Qwen CUDA CodePredictor 帧图（2026-10-18）

本文的性能数字针对 0.6B Base；后续 1.7B Base 的独立正确性与性能结果见
[1.7B 验收](qwen17-cuda-acceptance-2026-10-18.md)，不将本表收益直接套到 1.7B。

## 结论

CodePredictor 每帧的完整自回归循环（2-token prefill + 14 步单 token 前向/argmax）
按 session 录制一次 CUDA graph 后回放，正式 ABBA 八进程：完整合成
3.0840 s → 1.9113 s（**-38.0%**），流式 3.6529 s → 2.4353 s（**-33.3%**），
两组区间不重叠；八进程 × 三轮 × 五类产物共 60 份 codes/PCM/WAV 与 eager 基线
逐字节一致。分支 `codex/qwen-cp-graph`，commit `59a7d7b`（实现）与 `8ab5962`
（共享 recorder 提取）。

| 指标（59 帧 / 4.72 s 音频） | eager | 帧图 | 改善 | 区间是否重叠 |
| --- | ---: | ---: | ---: | --- |
| 完整合成耗时 | 3.0840 s | 1.9113 s | **-38.0%** | 否（[3.0382, 3.1214] vs [1.8921, 1.9401]） |
| 完整合成 RTF | 0.653 | 0.405 | -38.0% | — |
| 流式合成耗时 | 3.6529 s | 2.4353 s | **-33.3%** | 否（[3.6030, 3.7024] vs [2.4194, 2.4533]） |
| 流式合成 RTF | 0.774 | 0.516 | -33.3% | — |
| 首 PCM | 0.7688 s | 0.5663 s | -26.3% | 否（[0.7546, 0.7773] vs [0.5567, 0.5743]） |

即每帧节省约 19.9 ms（完整路径 (3.0840-1.9113)/59），与“CodePredictor 是主机
分派主导、kernel 时间保留”的解释一致；这不是分解证明，分解仍需独立 profile。

源码固定点 `94b4054`（合并 VoxCPM 与 Qwen 主线）。承接
[Qwen CUDA 调研](qwen-cuda-performance-2026-10-18.md) 与
[candle-performance 试用](../platform/candle-performance-skill-trial-2026-10-18.md)
的候选结论，本轮把该候选实现并测量。

## 基线与边界

- Windows / RTX 5070 12 GB / CUDA 13.0；Candle 0.11.0 fork revision
  `bafb03db81c3eae48e63ff28d6b30f5a6576ba38`，cudarc 0.19.10。
- 模型：本地 Qwen3-TTS 0.6B Base（`target/drift-experiment/round3/qwen-base06`），
  权重 SHA-256 与 `models/0.6b-base.json` 固定清单一致，revision
  `5d83992436eae1d760afd27aff78a71d676296fc`。1.7B 目录本轮仍未验收。
- 输入与 [Qwen CUDA 调研](qwen-cuda-performance-2026-10-18.md) 完全一致：ICL
  voice_narrator_ref prompt、文本“山风吹过松林，星光照亮归途。”、seed 42、
  chunk_frames 10、semantic temperature 0.9/top-k 50/top-p 0.9/repetition 1.05、
  acoustic argmax、min 2 / max 2048、EOS 2150。
- 计时：`qwen_timing_probe` 无 profiling release 构建，入口/出口设备同步，
  导出在计时外；每进程 1 轮预热 + 2 轮热测，丢弃预热。
- 测量：`nvidia-smi -lgc 2700` 锁频，288 个采样中 SM 2685–2692 MHz（均值
  2692）、功耗 41.8–111.8 W（均值 73 W）、温度 48–63 °C；期间不编译、不并行推理。
- **偏差如实记录**：本机「高性能」电源 GUID 无效（`powercfg /setactive` 返回
  “参数无效”，脚本未检查退出码的历史结论已撤销），实际电源计划保持「平衡」，
  A/B 两组在同一计划下交替，组间可比但不声称高性能计划；桌面 GPU 进程驻留，
  非完全独占；未复测官方 PyTorch。

## 实现

### 每代 session 拥有图与 KV（SOLID 单一职责）

`models/code_predictor/session.rs` 的 `CodePredictorSession` 持有一个
`CodePredictor` 引用、CP KV caches 和（CUDA）帧图。完整合成在
`generate_codes` 每次调用建一个；流式在 `StreamingSession` 持有一个。图的
捕获地址因此永远不跨请求共享，也不放在全局 `Qwen3TTS` 上。两处调用点从
`generate_acoustic_codes(hidden, semantic, &mut caches)` 变为
`session.generate(hidden, semantic)`，CPU/Metal 走同一条 eager 分支。

结构体字段顺序即生命周期契约：`graph` 声明在 `caches` 之前，先销毁图再释放
其捕获的 KV 缓冲。

### 图边界

- 捕获区域只有 CP 的 15 个声学码：prefill(2) → lm_head[0] argmax → 14 ×
  (embedding → layer → norm → lm_head → argmax → slice_assign)。
- 图外：talker 单步、语义采样与惩罚、EOS 判定、residual VQ 求和、codec 解码。
- 输入经持久缓冲 d2d 覆盖；`[15] U32` 输出在 capture 内拷入录制前分配的常驻
  缓冲后释放 capture 分配，图即配平（沿用 VoxCPM 已验证的规则）。
- Standard RoPE 表在加载时预建；offset 2..15 是主机常量，在录制中展开；无
  host 读回、无取消轮询。

### 为什么回放不需要重置 `current_len`

`PreAllocKVCache.current_len` 是 Rust 侧元数据，回放不会重新执行 `reset` 与
`update`。本帧安全的原因是**不变式**：每次回放都把 prefill 的 2 个位置和 14 步
的 14 个位置全部重写到 0..16，而图内读取的 narrow 视图长度（2,3,…,16）在录制
时固定。旧帧的 KV 值在读取前必被同帧写入覆盖。容量 17 只是上限；有效长度终点
是 16。

这个不变式不是所有图都成立：跨帧卷积历史（VoxCPM AudioVAE）不被每帧重写，
必须用图内更新的持久状态缓冲，否则只能终止请求。因此失败策略按状态性分档：

| 区域状态 | launch 失败策略 |
| --- | --- |
| 每帧自覆写、无跨帧持久状态（本帧 CP） | tracing 警告 + 本 session 永久停用图 + 本帧 eager 重算 |
| 跨帧持久状态、无法证明未推进（Vox AudioVAE） | 终止当前请求，绝不重复算该帧；下一请求重置后 eager |

### 环境开关

CUDA 默认开启；`QWEN_CP_GRAPH=0` 关闭；`profiling` feature 下强制 eager，
否则 replay 会把逐步 span 压成一次 launch、使 trace 失真。

## 验证

### 单元（CPU，无需模型）

`session_matches_direct_eager_across_frames`：小配置 CP（hidden 32 / 2 层 /
4 组）逐帧比较 session 与直接 `generate_acoustic_codes` 的 U32 codes，
证明 session 的缓存重置语义与直接调用一致（`cargo test -p
talechime-qwen3-tts --lib`，220 passed）。

### 真模型 CUDA（`#[ignore]`，需 `QWEN_CP_TEST_MODEL`）

`cuda_code_predictor_replay_matches_eager`，串行运行：

1. 8 组不同 hidden/semantic 状态，replay 与 eager codes 逐位相同；
2. 跑完 8 帧后回放第 0 帧，结果仍与首次相同（KV 全量重写不变式）；
3. 故意用错误宽度输入触发 launch 失败 → 图停用，后续帧仍与 eager 一致。

结果：1 passed（RTX 5070，3.1 s）。

### Concat 变体 capture（关闭试用报告的遗留问题）

`new_kv_caches` 在预分配失败时回退 `Concat`，其 K/V 由 `cat` 增长：最终那对张量
**在 capture 内分配、并在 capture 结束时仍被缓存持有**，即图节点之外存在主机侧
保留的 capture 分配。用真模型直接录制该变体（`cuda_concat_cache_capture_matches_eager`）：

- `Replay::record` 在 `end_capture`/实例化阶段被驱动拒绝：
  `CUDA_ERROR_INVALID_VALUE`；recorder 的 RAII 收尾生效，事后流不在捕获态
  （`is_capturing` 为 false），设备可继续使用；
- 即 Concat 变体**无法完成图录制**，不是“录成但只能回放一次”。这与 VoxCPM 记录的
  未配对分配节点是同一机制（那里表现为延迟到 launch 报错），差别只在驱动把校验
  放在实例化还是回放。

生产路径因此依赖既有设计：`CodePredictorSession` 的图在 `record` 报错时只发警告
并对该 session 永久走 eager（预分配失败 → 回退 → 图不可用 → 正确降级），而正常
CUDA 分配成功时走 PreAlloc，缓存在 capture 之前分配。测试钉住两个半边：拒绝必须
是 INVALID_VALUE 且流已恢复；若将来驱动接受该图，测试改走逐位对照路径。

### 端到端产物

ABBA 八进程每个三轮，`codes-{0,1,2}.json`、`full-*.f32le`、`stream-*.f32le`、
`full-*.wav`、`stream-*.wav` 共 15 类 × 8 = 120 份；跨 A/B 变体每一类只有
一个 SHA-256（脚本报告 15 checked / 0 mismatched，即每轮全部 8 进程一致）。
与 `final/`（改动前 eager 基线）逐字节一致：codes、完整与流式原始 PCM、两份 WAV
的 SHA-256 全部相同；均为 59 帧、113280 samples、EOS、有限值。

### 性能（ABBA 八进程，顺序：eager graph graph eager eager graph graph eager）

每进程 1 预热 + 2 热轮，取进程内热轮均值：

| 指标 | eager mean-of-means / 区间 | graph mean-of-means / 区间 | 改善 |
| --- | ---: | ---: | ---: |
| 完整合成 | 3.0840 [3.0382, 3.1214] | 1.9113 [1.8921, 1.9401] | 38.02% |
| 流式合成 | 3.6529 [3.6030, 3.7024] | 2.4353 [2.4194, 2.4533] | 33.33% |

RTF（4.72 s 音频 = 59 帧 × 1920 / 24000）：完整 0.6535 → 0.4049；流式
0.7739 → 0.5160。门槛为 ≥10% 且区间不重叠，通过。

### 共享 recorder 提取后的复验

`8ab5962` 把 recorder 从 `voxcpm::cuda_graph::replay` 移到
`crates/candle-cuda-graph`（整数输出支持、`is_capturing()`、依赖 Candle 的
`cuda` feature 以保留 cudarc 库加载决策）。复验：

- `cargo test -p voxcpm --features cuda --lib` 10 passed / 10 ignored；
  免权重的 `views_capture_cleanup_and_fallback` 1 passed（真实 GPU 上覆盖
  view 复制、录制失败清理与 fallback）；
- Qwen CUDA lib 220 passed / 4 ignored，真模型图测试 1 passed；
- 提取后重跑一次三进程探针：与 `final/` 可比的 10 份产物仍逐字节一致，
  full 1.808 s / stream 2.440 s，落在 ABBA graph 区间内；
- workspace `cargo check`、三 crate clippy（默认与 CUDA feature）、fmt 全过。

注意：ignored 的真 GPU 测试必须 `--test-threads=1` 串行。同一进程内两次
`Device::new_cuda(0)` 共享同一条 candle 流，并发 `begin_capture` 会互相破坏
（并行时两个图测试同时失败，串行均通过）。

**未复验项**：本机没有 VoxCPM GGUF 权重，`real_model_replay_noise_schedule_and_lru`
与 `decoder_tests` 三个真模型测试未运行。对 VoxCPM 的风险限于“浮点 dtype 分支
代码逐字未变 + 编译与免权重 CUDA 测试通过”，需要在有权重的机器上补跑。

## 复现

```powershell
cargo build --locked --release -p talechime-qwen3-tts --features cuda `
  --example qwen_timing_probe -j 1
$env:QWEN_CP_GRAPH = '0'   # eager 基线；设为 '1' 或不设置即 graph
target\release\examples\qwen_timing_probe.exe $MODEL $PROMPT `
  '山风吹过松林，星光照亮归途。' cuda $OUT 2 10
```

ABBA 驱动与逐进程 JSON 在忽略目录
`target/tmp/voxcpm-opt/qwen-cuda/{run-abba.ps1,analyze-abba.ps1,graph-abba/}`。

## 对 skill 的反馈

本轮验证并补强了试用报告里列出的每个预检项，详见
`skills/candle-performance/references/cuda.md`：主机侧簿记不回放、输出 dtype
必须被 recorder 支持、失败策略按状态性分档、第二个真实使用者出现后才提取共享
recorder，以及“每帧数百小 kernel 的 dispatch-bound 区域是首选目标”这一实测
结论（CP 34 ms/帧的主机墙钟被一次 launch 取代）。
