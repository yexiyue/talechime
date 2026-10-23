# macOS CPU Accelerate 默认配置与线程实验（2026-10-23）

本轮按用户要求，将 macOS Candle CPU 的 Accelerate 接线补齐到所有模型库。
Qwen3-TTS、CosyVoice3 已有该默认；新增 OmniVoice、MOSS、VoxCPM、IndexTTS、Qwen3-ASR。
仅在 macOS target 激活，不更换 Candle revision、模型、采样或公共 API。
VoxCPM 的历史 `accelerate-experiment` feature 保留，但不再提供 macOS 纯 Rust GEMM 对照。

标准全模型 macOS 发行配置此前已通过 Qwen/CosyVoice 合并开启 Accelerate，本轮不是
该发行配置的新提速。ORT CPU provider 独立于 Candle feature；量化等路径也不保证
调用 Accelerate。启用该 framework 不等于已证明 AMX 执行。

Accelerate 是 Apple 平台 framework，Windows/Linux 的 AMD CPU 不能使用。
AMD 的 SIMD、第三方 BLAS 或 ORT CPU 优化属于其他实现，本轮未做 AMD 实机测量。
平台机制参考 [Apple Accelerate](https://developer.apple.com/accelerate/)。

## 基线与协议

源码基线 `11cff7b`，本轮生产改动仅上述 target 依赖；另有测试夹具与说明更新。
M4 Pro / 12 核 CPU（8P+4E）/ 24 GiB / macOS 26.6.2。
Candle 固定 `2266099d92265112ecc0d56ecee1761e3cc2ac00`，CPU F32。
沿用 [CPU 完整链路记录](omnivoice-cpu-full-chain-2026-10-23.md) 的短文本、
0.6b 权重、narrator、seed 42、32 步与默认采样参数。
输入 SHA256 `f29b49d81cce2947186d9b6f6d5fc719a2aaee26e7993c14e394db0afb313657`。

独立默认 release 构建，不再附加 `candle-core/accelerate`：

```sh
cargo build --release --locked -p talechime-backends --no-default-features \
  --features omnivoice,performance-tools --example performance_probe
```

冻结二进制 SHA256 `038f3ba59fca322c3994c926a30583dcb77b838ae3be1fb7627dc645944d14b5`；
`otool -L` 确认链接 Accelerate.framework。
源码中针对 CPU F16 错误行为的测试修改不进入此二进制的运行时。

所有构建/测试完成后串行推理；开始前等待外部编译结束，不终止其他任务。
三种策略按 ABCCBA，每组两个独立进程，每进程 1 预热 + 3 热轮：

| 组 | RAYON_NUM_THREADS | VECLIB_MAXIMUM_THREADS | OMP_NUM_THREADS |
| --- | ---: | --- | ---: |
| A | 8 | 8 | 8 |
| B | 1 | 8 | 8 |
| C | 8 | 未设置，交由系统策略 | 8 |

这些是环境配置，未逐算子确认实际 BLAS 线程数。计时包括完整合成与 PCM 消费，
不包括资源验证、加载、单独参考预热和 WAV 导出；正式计时关闭阶段探针与 codes dump。
未锁频、非独占机器；每 0.5 s 按 basename 记录 Cargo/编译器活动。

## 验证范围

七个模型库分别检查 macOS、Windows MSVC、Linux GNU 的依赖树，共 21 组：
仅 macOS 的 Candle core 启用 Accelerate。跨平台只核验依赖图，不宣称完成交叉编译。
CPU 多模型组合与 Metal 多模型组合均通过 all-targets Clippy（warnings denied）。
五个受影响模型库通过 173 项 CPU 单元测试（22 项显式忽略），OmniVoice 通过
34 项 Metal 单元测试；五库 rustdoc warnings denied 与格式检查通过。普通测试不下载模型。

一个旧 OmniVoice 夹具要求 CPU F16 卷积成功；Accelerate 的 F16 matmul 明确不支持。
调整后在该配置验证包装与原路径返回同一错误，其余路径仍比较完整数值。
正常 OmniVoice CPU 使用 F32，不新增静默类型转换或算术 fallback。

其他模型未在本轮完成真实权重音频验收，不能把普通测试通过写成跨 BLAS 逐位一致。
切换 BLAS 可能改变浮点结果；此前 OmniVoice 纯 Rust 与 Accelerate 已观察到微小 PCM
变化，详见上一轮记录。本轮默认与此前显式 Accelerate 的结果另行比对。

原始配置、脚本、逐轮 JSONL、WAV、哈希与编译活动在忽略目录
`target/perf/omni-cpu-threads/`；不提交模型、音频或正文。

## 下一步的实现边界

线程配置实验只验证调度假设，不改模型算术，也不把环境变量固化成全模型策略。
后续仍优先分解 stage0 的投影/MLP、注意力与布局成本。上一轮官方 profiler 实际
执行 CPU Flash Attention，而 Rust CPU 是 repeat_kv → QK → mask/softmax → PV。
这支持做对应局部实验，不足以把完整差距归因于 attention。

固定 Candle 的 `attention::cpu_flash::flash_attn` 并不是可直接开启的等价 feature：
实际实现对 batch > 1 + 显式 mask 报错，而本例 CFG 为 batch 2 且有显式 mask。
若后续试验该实现，需要拆 batch 并计入复制/拼接成本，比较完整注意力输出与最终
codes/PCM；在线 softmax/SIMD 累加也可能改变浮点结果。本轮没有在生产接入它。

## 逐进程原始结果与判断

| 顺序 / 组 | 三次热测（s） | 进程均值（s） | 编译活动观测次数 |
| --- | --- | ---: | ---: |
| 1 / A | 8.175853958 / 8.081281416 / 8.104121041 | 8.120418805 | 0 |
| 2 / B | 9.905678167 / 9.780775583 / 9.792738458 | 9.826397403 | 0 |
| 3 / C | 8.071294417 / 8.207202042 / 8.076532958 | 8.118343139 | 0 |
| 4 / C | 8.330156209 / 8.208092167 / 8.202747000 | 8.246998459 | 0 |
| 5 / B | 10.070082459 / 10.105347958 / 9.783378000 | 9.986269472 | 25 |
| 6 / A | 9.501741625 / 8.616863042 / 8.639978750 | 8.919527806 | 71 |

B5 / A6 受外部编译干扰，保留原始活动记录，不将各组混合均值当作干净的 A/B
提速结论。A1、B2、C3、C4 没有编译活动观测；其余系统干扰未排除。
C3/C4 均值分别为 8.118 / 8.247 s，与 A1 的 8.120 s 未显示可信收益。
B2 为 9.826 s，没有支持 Rayon=1 更快的证据；B5 不能用来精确量化回退幅度。
本轮不改应用全局线程策略，不把不同日期的历史 8.390 s 与本轮 8.120 s 相减称为优化。
尚未完成完整的无编译干扰配对复测；这些线程配置不外推到长文、其他模型或 AMD。

六进程共 18 个热轮与 6 个预热轮，整段/逐段 F32 和 PCM16 共 96 个 WAV，均与上一轮
显式 Accelerate 的对应完整文件逐字节相同。每次正文为 127680 样本 / 24 kHz / 5.32 s；
完整 F32 WAV SHA256 为
`2de9f6fb2688b651a33a36470cace6acf3c1e8126d17feb5094c18691d1a9b5c`。
随后另跑仅用于正确性的默认配置请求，完整 codes dump 与上一轮显式 Accelerate
逐字节相同（参考预热 384 codes、正文 1040 codes）；不将该诊断的耗时计入正式结果。
其余输出哈希和逐文件对照在 `audio-equality.json`，此处不重复罗列音频。
本轮未复测长文/参考音色，未做新增试听或 ASR 质量验收。

