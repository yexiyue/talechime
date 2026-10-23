# OmniVoice CPU 完整链路检查（2026-10-23）

CPU 与 Metal 是两套需要分别测量的执行路径。本轮当前配置下，Rust Accelerate CPU
完整短文本合成均值 8.390 s，官方 CPU 6.103 s；Rust 耗时约高 37.47%。两边均为
F32，当前剩余成本主要在 stage0 forward，不是排序或音频导出。

**独立 OmniVoice 构建未默认开启 Accelerate，不等于 Mac 全模型发行配置没有开启。**
发行配置通过 Qwen/CosyVoice 的 macOS 依赖统一启用 Candle Accelerate；不能把
独立构建的二十多秒当成当前安装包成绩，也不能把实验显式开启 Accelerate 的收益
写成现有发行包可以直接获得的新收益。本轮未检查用户已安装程序的二进制。

没有证明 Metal 已接近硬件极限。最近几项局部优化的整链收益小，只能说明那些具体
候选的收益；CPU 当前 F32 与历史 Metal F16 也不是相同精度的设备 A/B。

## 版本、设备与计时

- 源码基线 `c5155fe`，分支 `codex/omnivoice-hotspot`；正式二进制先构建冻结。
- M4 Pro / 12 CPU 核（8P+4E）/ 24 GiB / macOS 26.6.2。
- Candle `2266099d92265112ecc0d56ecee1761e3cc2ac00`。
- 本机 0.6b 权重 `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`，沿用生产清单校验。
- 官方 `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`，torch 2.8.0 / transformers 5.3.0；
  官方模型源码无本地修改，checkout 的 uv.lock 有既有差异。CPU 计时脚本是本地包装器。
- Rust CPU stage0 / decoder 均 F32；官方运行时确认 LLM、tokenizer 均 CPU / float32，
  attention 配置 sdpa，torch CPU threads=8。
- `RAYON_NUM_THREADS=8`、`VECLIB_MAXIMUM_THREADS=8`、`OMP_NUM_THREADS=8`；它们是线程配置，
  未逐算子核验 BLAS 实际线程数，也不据此声称使用了 AMX。

相同短文本 SHA256 `f29b49d81cce2947186d9b6f6d5fc719a2aaee26e7993c14e394db0afb313657`，
text-only / narrator，无参考音频、无 conditioning，seed 42、32 步、guidance 2、
position_temperature 5、class_temperature 0、t_shift 0.1、layer_penalty_factor 5。
Rust 按请求设置 seed；官方包装器在进程开始设置一次 seed。跨框架 RNG 和浮点路径
不同，不视为相同张量输入或音质等价实验。

三个路径按 ABCCBA 串行，各两个独立进程，每进程 1 预热 + 3 热轮：
A 默认 Rust GEMM；B Accelerate；C 官方 CPU。
release 正式计时不启用阶段同步或 codes dump。计时含文本前端、stage0、解码、后处理和
消费完整 PCM；生产资源校验、加载、探针的单独短句预热及 WAV 导出不计入热合成。

## 完整链路结果

| 路径 | 两进程热均值（s） | mean-of-means（s） | 平均 RTF | 输出音频 |
| --- | --- | ---: | ---: | --- |
| A：独立默认 CPU | 25.350 / 28.911 | 27.130，仅作受干扰观察 | 5.100，仅观察 | 5.32 s |
| B：CPU + Accelerate | 8.517 / 8.262 | **8.390** | **1.577** | 5.32 s |
| C：官方 CPU | 6.087 / 6.119 | **6.103** | **1.130** | 5.40 s |

A1 / A6 分别有 4 / 94 次外部构建活动观测；A6 热轮为 30.837 / 31.598 / 24.296 s，
波动大。原始数据全部保留，不将 A/B 的名义比例当作无干扰、正式提速结论。
CPU 默认原路径明显值得调查，但本轮不报告一个精确的“开启后提升 X%”。

B 两进程、C 两进程均无 cargo/rustc/clang/ld/clippy-driver 等活动观测；
按进程 basename 每 0.5 s 监控，另包含 cargo-clippy。B 相对 C 耗时高 37.47%，
RTF 高约 39.5%，差异不是只由 0.08 s 的音频时长差造成的。这是本例宏观生成配置的
观察，不是逐张量相同负载或所有长度的排名。没有锁频或独占系统，样本数每侧两个进程，
未给出置信区间；未终止其他任务。`pmset -g therm` 没有可用的 CPU 功耗状态，
不能据此排除全部调度/温度因素。

峰值进程 RSS（含准备与热测的采样峰值，非独占内存）：A 约 2.52–2.59 GiB，
B 约 2.41–2.80 GiB，C 约 2.88–2.89 GiB。加载和参考预热单独保存在 prepared 事件，
不将热测数据命名为冷启动。

## feature 与实际 BLAS 路径

独立 A 构建：

```sh
cargo build --release --locked -p talechime-backends --no-default-features \
  --features omnivoice,performance-tools --example performance_probe
```

独立 B 构建：

```sh
cargo build --release --locked -p talechime-backends --no-default-features \
  --features omnivoice,performance-tools,candle-core/accelerate --example performance_probe
```

两个二进制分别冻结为 `probe-default` / `probe-accelerate`。execution-info 分别报告
CPU math 为 candle-rust / candle-accelerate；`otool -L` 确认仅 B 链接 Accelerate.framework，
两者都没有 Metal 链接，不是静默走 GPU。F32 CPU matmul 的 Accelerate 分支调用 sgemm。
Rust B 和官方 libtorch_cpu.dylib 的动态符号都包括 `sgemm_` / `dgemm_`，没有证据支持
“官方只因换了另一套 BLAS API 就快”；也不能从符号判断具体 AMX kernel。

发行源码配置 `dist-workspace.toml` 包含 qwen-metal / cosyvoice3-metal / omnivoice-metal。
Qwen 和 CosyVoice 的 macOS 依赖显式启用 `candle-core/accelerate`，Cargo 合并到共享
Candle。按发行 feature 集合执行反向 `cargo tree -e features -i candle-core`，已确认
这两条启用链，证据 `release-candle-features.txt`。因此发行配置的 CPU 路径应以 B
作为本轮更相关的比较入口；这仍不能替代对某个已安装二进制的实际核验。

## 阶段诊断与官方算子

正式计时结束后单独运行阶段诊断。默认 A 只开 STAGE_TIME；B 另开 SYNC_PROBE，
CPU 执行同步，不需要 GPU 完成屏障。B 用只增加 CPU canary 跳过判断的新诊断二进制。
诊断进程分别有 23 / 6 次外部 Cargo/Clippy 活动观测，**仅作成本定位，不当作新吞吐成绩**。

正文 target_len=130、cond_len=157、32 步，B 诊断结果：

| 阶段 | 耗时 |
| --- | ---: |
| stage0 forward | 244.959 ms/步，约 7.839 s |
| logits 转型 | F32→F32，无新增转换耗时 |
| predict / scoring | 4.880 ms/步，约 0.156 s |
| 更新（含排序） | 0.055 ms/步 |
| 排序 | 0.015 ms/步 |
| 写回 | 0.012 ms/步 |
| stage0 全程 | 8.004 s |
| decoder / stage1 | 0.472 s |

forward 包括音频 embedding、28 层骨干和音频输出头，不等于已单独测得的 LLM 层时间。
这次诊断完整合成为 8.477 s；阶段值不外推成正式吞吐或未来优化收益，不重复相加
有父子关系的阶段。A 的粗诊断为 stage0 26.604 s、stage1 0.590 s。

官方另跑 CPU profiler，实际观察到 896 次
`aten::_scaled_dot_product_flash_attention_for_cpu`，即 28 层 × 32 步，
Q/K/V 为 `[2,16,157,128]`；LLM 输入 `[2,157,1024]`，mask `[2,1,157,157]`。
这里确认的是实际 CPU Flash Attention 算子执行，不只是配置名为 sdpa。
Rust CPU 当前仍执行 repeat_kv、QK 矩阵、mask、softmax、PV 的手写路径。
官方 profiler 的 mm 自耗时居前，但没有用其独立算子时间倒推出 Rust 的投影占比，
也没有把官方 attention 时间与 Rust 整个 forward 相减归因。

## 输出与修复

两条 Rust CPU 路径各自六个热轮 F32 WAV 摘要稳定。另开 codes dump 对照，
探针短句预热的 384 个 codes 和正文 1040 个 codes 全部一致；本例更换 BLAS 未改变最终 codes。
同样本数 127680 / 24 kHz 的完整 F32 WAV：343 个样本位模式不同，最大绝对差
`1.831e-5`、RMSE `9.482e-7`、SNR 97.77 dB。不是逐字节相同，也不将数值指标
当作听感、ASR 或所有文本的准确性验收。本轮没有自动改变默认 CPU math。

CPU 默认 F32 WAV SHA256：
`0bfc4c0cb1011211eca51f15fb9bfcf433e427bcbd7d598349281195d5a599df`；
Accelerate：`2de9f6fb2688b651a33a36470cace6acf3c1e8126d17feb5094c18691d1a9b5c`。

唯一运行时代码修改是诊断工具：F16 GPU canary 原先也会在 CPU SYNC_PROBE 中构造，
而 CPU Accelerate 明确不支持 F16 matmul。现在 CPU 直接跳过该 GPU canary，
保留常规 CPU 阶段诊断；正常合成的算术和配置不变。新诊断产物的 WAV 与正式 B 一致。

## 后续优先级与交付

1. 先细分真实 CPU stage0 的投影/MLP、attention 和布局开销，并测 Rayon / BLAS 线程策略，
   不从 feature、理论 FLOPs 或官方 profile 猜各项收益。现有连续 F32 RMSNorm 已是融合
   实现，Linear 也会对连续输入合并 batch/sequence 行，不将它们误列为尚未实现的优化。
2. 单独试 CPU attention 路由。固定 Candle 有 CPU flash API，但当前双行 CFG + 显式 mask
   不能直接走其 B>1 入口；需要验证分行或其他实现，且 online softmax 会改变浮点路径，
   必须独立做数值/输出验收，不是打开一个遗漏的 feature。
3. 再测 CPU decoder 的转置卷积/布局；不能直接搬用仅对 Metal 验收的快路径。
   0.472 s 有局部优化价值，但它不是当前约 8 秒完整合成的主体。

本轮只覆盖默认旁白短文本完整链路；长文本、多段接续、参考音频克隆、量化模型及
线程矩阵尚未重新验收。未改依赖、模型资源、采样参数或正常推理路由，未 push、合并或发布。

原始数据在 `target/perf/omni-cpu-chain/`：`comparison.json` / `summary.json`、
逐进程 jsonl/err、`profile-summary.json`、`official-profile.json`、
`cpu-output-comparison.json`、codes dump、执行信息/链接/feature 树与构建日志。

CPU lib 23 tests、Metal lib 34 tests、Metal + Accelerate all-targets Clippy
`-D warnings`、Metal library/examples rustdoc `-D warnings`、格式与 diff 检查通过。
普通测试未下载模型。
