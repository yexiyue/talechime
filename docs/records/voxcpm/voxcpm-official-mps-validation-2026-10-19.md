# VoxCPM 官方 Python/MPS 与原始权重验证（2026-10-19）

结论：**没有发现可迁移到生产 Candle Metal 路径的已验证性能机会。**

- 官方固定源码在 MPS 上将配置的 BF16 自动改为 F32，故没有可用的“官方 BF16
  MPS”基线；它不是与 Candle 原始 Safetensors + F16 的同 dtype 对照。
- 官方 `VoxCPMModel.optimize()` 明确只编译 CUDA 路径；MPS 调用后回退 eager，故没有
  有效的 eager/optimize A/B，也没有 Python 迁移或运行时开关。
- Candle 生产 Q8 + F16 与官方原始 Safetensors + F16（AudioVAE F32）的正式八进程
  ABBA 中，原始权重候选平均 RTF **慢 0.60%**，区间重叠且逐对方向不一致。

生产默认、公开 API、GGUF 和模型权重均未改变。本记录只覆盖一次短进程、串行的
macOS Metal 实验；不宣称 AMX、音质或长时间无断流能力。

## 1. 固定条件与边界

| 项目 | 值 |
| --- | --- |
| 主机 | Apple M4 Pro（Mac16,8，24 GB 统一内存），macOS 26.6.2 (25G83) |
| 工作区 | `cd1089b30b73682bf0c5f72c995cc6d4c4cb9411`，测量前已是 dirty；保留原有未提交改动和 `docs/learning/` |
| Candle | fork `2266099d92265112ecc0d56ecee1761e3cc2ac00`，Metal 路径 |
| 官方源码 | OpenBMB `f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`，独立 checkout 且 clean |
| 官方模型 | `openbmb/VoxCPM2` revision `32279effe8c19989596f05d353d1447f51d9e915` |
| 主权重 | `model.safetensors`，4,580,080,592 B，SHA-256 `f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d` |
| AudioVAE | `audiovae.pth`，376,951,122 B，SHA-256 `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1` |
| Python | `uv` 创建的 Python 3.12.14 venv；PyTorch 2.14.1；MPS available |
| 输入 | `acceptance-corpus.txt` SHA-256 `2ac7b2013eafc8ac8f3439df46bedbd811f7fc105b35037c4049c710256d29a6`，生产 `preprocess_text(_, 180)` 后 11 段 |
| 参考 | `voice_dongqing/reference.wav` SHA-256 `2d07a2135e526b52c8ecd6ed8383d8f8e833d75c092c28a16f60f23c9f0c756b`；转写 SHA-256 `ab782e561594694eeb5d77acaeadc346822185ca3f8999c5abf12a7362ef072d` |
| 生成设置 | seed 42、CFM 10 步、CFG 2.0、温度 1.0、帧上限 200、48 kHz float32 WAV |

官方权重保存于 revision-scoped 用户资源目录，下载器通过 PATH 中的 `curl` 断点续传并
校验 Hugging Face 元数据的 SHA-256；`source-manifest.json` 与 Python 环境、WAV、trace
工件均留在 Git 忽略的 `target/` 或用户资源目录，未提交。

## 2. 探针与测量方法

新增的开发工具不导入 Talechime 的模型实现：

- `tools/tts/voxcpm_official_mps_probe.py` 仅经官方 `VoxCPM.from_pretrained` / `generate`
  公共入口运行固定源码。加载前后显式 `torch.mps.synchronize()`；计时涵盖每段公共 API
  内部的参考准备和生成，修正后的探针在生成前后同步，逐段累加 API 时间并在计时外写 WAV。历史数值包含 WAV 写入，不能重命名为修正后的新基准。
- Candle probe 写出生产分段的 `segments.json`，官方探针复用它，避免把分段规则差异计入。
- `tools/tts/voxcpm_perf_common.py` 集中 ABBA 序列、统计、配对和门槛；Candle 与官方 Python
  包装器复用该逻辑。screen 为四个对称槽位 `B,C,C,B`，正式为八槽
  `B,C,C,B,B,C,C,B`；每进程 1 预热 + 1 热轮，槽间空闲 20 s，串行且无编译。

正式 acceptance gate 是平均 RTF 至少降低 10%，候选最大值低于基线最小值，并保留全部
slot 与逐槽方向。RTF 是同步后的生成耗时 / 此次实际 PCM 时长，不包含加载。

## 3. 官方 Python MPS：可用，但不是同 dtype 对照

官方 eager 的 11 段均生成有效、有限的 48 kHz PCM（60.96 s），公共非流式 API 不公开
首 PCM，所以字段明确为“不暴露”，没有把完整生成时间冒充首 PCM。历史无审计 hook 的 1 预热
+ 1 热轮探针中，加载为 12,995.216 ms，热轮生成 67,245.728 ms，RTF 为 **1.103112**。

官方配置初始为 BF16，但固定源码的 `pick_runtime_dtype` 在 MPS 上将 BF16/FP16 改为 F32；
实测模型参数和 buffer 都在 `mps:0`，浮点参数为 `torch.float32`。另做一轮带执行 hook 的
设备审计（无预热，**不纳入吞吐**）确认以下真实生成边界的张量入参都是 `mps:0`：

- BaseLM：prefill 与 `forward_step`
- ResidualLM：prefill 与 `forward_step`
- `feat_encoder`、`feat_decoder.estimator` 与 `audio_vae.decode`

官方 eager 输出 60.96 s，Candle Q8 输出 58.24 s，Candle 原始 F16 输出 58.88 s；同时
dtype、权重转换和公共 API 的每段参考准备边界均不同。因此未把这些 RTF 并列归因成
Rust/Python 运行时差异，也没有对这个 dtype 混杂组合运行 acceptance A/B。旧官方探针还同时提供隔离 identity 与 continuation，并启用 badcase 重试，存在条件语义和实际 seed 的额外混杂；旧值只保留为历史观察。修正后的探针只传 transcribed continuation，关闭重试并记录实际 seed。

### optimize 检查

`optimize=True` 仍成功加载模型，但固定源码的 `VoxCPMModel.optimize()` 在设备不是 CUDA 时
抛出 `VoxCPMModel can only be optimized on CUDA device`，随后捕获异常并保持 eager。探针确认
没有 compiled component，`optimize_effective=false`。这条路径不可用，不执行无效的
eager/optimize screen 或八槽 ABBA。

## 4. Candle Q8 对原始 Safetensors + F16

两臂使用同一预构建 Candle probe。基线为生产 GGUF Q8，候选为官方 Safetensors 的
backbone F16；Q8 Metal 的 AudioVAE 为 F16，原始权重的 AudioVAE 为 F32（以当前 loader 为准）。这个实验只回答“权重/精度组合是否
能带来端到端收益”，不回答跨框架运行时优劣。

### 四槽 screen

| 组 | 均值 RTF | 范围 | 槽位配对收益 |
| --- | ---: | ---: | --- |
| Q8 baseline | 1.056188 | [1.029526, 1.082850] | +3.97%, -1.81% |
| 原始 F16 candidate | 1.044040 | [1.039870, 1.048209] | |

表面均值仅快 1.15%，配对方向已翻转，故不作为撤回候选的依据；按协议继续完整 ABBA。

### 正式八槽 ABBA

| slot | arm | RTF | 首 PCM ms | 进程墙钟 s |
| ---: | --- | ---: | ---: | ---: |
| 0 | Q8 baseline | 1.032236 | 429 | 129.9 |
| 1 | 原始 F16 candidate | 1.116210 | 403 | 138.4 |
| 2 | 原始 F16 candidate | 1.134779 | 499 | 145.3 |
| 3 | Q8 baseline | 1.074708 | 463 | 136.8 |
| 4 | Q8 baseline | 1.084349 | 469 | 137.0 |
| 5 | 原始 F16 candidate | 0.988757 | 389 | 138.6 |
| 6 | 原始 F16 candidate | 1.036634 | 440 | 131.0 |
| 7 | Q8 baseline | 1.059785 | 416 | 125.3 |

| 组 | n | 平均 RTF | 范围 | 相对基线 |
| --- | ---: | ---: | ---: | ---: |
| Q8 baseline | 4 | 1.062770 | [1.032236, 1.084349] | - |
| 原始 F16 candidate | 4 | 1.069095 | [0.988757, 1.134779] | **-0.60%** |

按相邻平衡槽位的候选收益依次是 **-8.14%、-5.59%、+8.82%、+2.18%**。候选均值更慢、
范围大幅重叠、逐槽方向不一致，`performance_gate=false`；因此结论是未确认收益，不能断言候选稳定更慢；不改变
Q8 + F16 生产默认，也不继续据此改 Candle。

## 5. 输出、数值与状态门禁

- 所有八个 Candle 进程的 probe 回归均为 `passed`：11/11 自然 EOS，取消、取消后复用、
  空文本、非法采样及 `max_frames` 截断检查均通过；测得输出均为非空、48 kHz、有限 PCM。
- Q8 与原始 F16 的代表性热轮（slot 0/1）均有 11 个 WAV，但 0/11 文件逐字节相同。
  6 段样本数不同，故不伪造逐样本误差；另外 5 段样本数相同的 SNR 为 0.1351–8.1876 dB，
  max abs 为 0.11835–0.17473，mean abs 为 0.002276–0.008578，RMSE 为
  0.007267–0.016215。数值变化只报告，不认定音质可接受。
- 官方 Python 的 11 段 WAV 同样均非空、48 kHz、有限；终止仅能观测为非流式
  `generate()` 正常返回，不能替代 Candle 的逐段 EOS 契约。

## 6. Profiling 限制

本机只有 Command Line Tools，`xcodebuild` 不可用；`xcrun xctrace list templates` 没有
提供可用的 `Metal System Trace` 模板。因此没有伪造 trace，也不从 MPS 注册、
`torch.compile` 调用或设备 audit 推断 kernel 加速、同步热点或 CPU/GPU 空档。要进行该项
分析需先安装包含对应 Instruments 模板的完整 Xcode，再独立采样，不把 trace 时间计入吞吐。

## 7. 审查修正与验证

2026-10-19 交付前复核发现并修正以下问题：

- 共享模块统一序列、冷却、落盘和统计；拒绝不完整序列、非有限/非正 RTF 和非法参数。共同后半段超过 5% 的变慢标为漂移警告，不能诊断根因，也不允许验收通过；首 PCM 可观测时要求不回归超过 5%。输出质量仍是独立门禁。
- Python 测试原先直接运行时没有执行任何测试（缺少 `unittest.main()`）；已补入口及参数、计时、参考模式与 PCM 校验的回归测试。此前的“Python 单测通过”不能作为旧代码测试覆盖的证据。
- 官方探针的 `output_checks=passed` 仅表示 PCM 校验，EOS 明确不暴露；设备 hook 使用独立 `--audit-devices`，不计入吞吐。默认禁止 MPS CPU fallback，检查固定源码、模型 manifest、48 kHz，并保存依赖版本快照。快照不是可移植的完整锁文件。
- 当前 Q8 与原始 F16 八槽结果的多模块精度边界如实保留；没有重写旧结果。原始权重 F16 与官方 eager 的完整四槽/八槽观察尚未执行，不能认定原计划三组比较全部完成。
- 完整 Xcode/Instruments、官方自然 EOS、听感与长时播放验收仍缺失。

### 修正后的真实模型复测

工件 `target/voxcpm-official-mps/official-eager-reviewed/`：固定官方源码 clean，
147 个依赖版本保存于 `requirements-snapshot.txt`。仅 transcribed continuation、seed 42、
关闭 badcase 重试，11 段在预热和热轮都输出 62.08 s，PCM 检查通过。加载 14,884.560 ms；
预热生成 76,068.082 ms、RTF 1.225323；热轮生成 93,452.354 ms、RTF 1.505354。
本轮包含每段官方参考准备、排除 WAV 导出；dtype 仍为 F32。前后轮波动较大，且只有一个进程，
**只验证修正入口可运行，不与旧数据做速度比较或宣称性能收益**。自然 EOS 和首 PCM 均不暴露。

修正后另以第一段作独立 `--audit-devices --warmup-rounds 0` 验证，
`official-reviewed-audit/summary.json` 确认七个生成张量边界均为 `mps:0`，仅用于设备审计。

验证：18 项 Python 工具单测通过；Rust 默认库测试 10 通过/5 忽略，全 feature 11 通过/6 忽略；
格式、VoxCPM 全 feature Clippy/rustdoc、Metal probe 编译/Clippy 和 platform 全 feature 构建通过。
同时验证 macOS 默认 Metal 不开启 Accelerate、实验才开启；Linux/Windows 实验 feature 不启用
Accelerate（依赖解析检查，不冒充两平台原生编译）。历史八槽共 176 个 WAV 均经离线检查为
非空、有限、单声道 48 kHz。

### 审查结论

规范轴原始发现 3 项：Accelerate 跨平台依赖泄漏、新依赖未集中继承、来源声明旧 pin；均已修正。
计划轴原始发现 6 项：参考模式/重试、AudioVAE dtype、tile 门槛例外、计时写盘、流水线停止归因、
`regressions=passed` 的过度宣称。前后检查已修复工具/记录；tile 约 6.9% 的例外已获得用户明确授权。
流水线和跨框架完整四槽/八槽仍列为未完成实验，不以库测试或单次探针替代。

## 8. 可复现入口与未验证范围

```bash
# 先由生产 Candle probe 写出 segments.json；Candle ABBA 复用同一 binary。
python3 tools/tts/voxcpm_perf_macos.py --screen \
  --model "$q8_model" --candidate-model "$official_model" \
  --reference "$reference" --transcript "$transcript" \
  --baseline target/release/examples/voxcpm_candle_probe \
  --candidate target/release/examples/voxcpm_candle_probe \
  --baseline-precision q8 --candidate-precision f16 --output "$out"

target/voxcpm-official-mps/venv/bin/python tools/tts/voxcpm_official_mps_probe.py \
  --source target/voxcpm-official-mps/source --models "$official_model" \
  --segments "$segments" --reference "$reference" --transcript "$transcript" \
  --mode eager --output "$out"

# 仅用于设备路径审计；不要将其结果用作吞吐基准。
# 在相同命令后追加： --audit-devices --warmup-rounds 0
```

没有验证官方 MPS 的低精度 override、CUDA、其他 Apple Silicon/批次/模型、听音或 ASR
质量，也没有证明长时播放。官方 eager 更快或更慢都不会自动授权生产路径迁移；本轮也没有
证据支持这种迁移。
