# IndexTTS 2.5 Candle 推理性能排查与优化

日期：2026-10-17。范围：`crates/indextts` 计算路径（独立 crate，未接生产后端）。关联：[移植验收记录](indextts-candle-acceptance-2026-10-17.md)、[官方对照复核](indextts-official-community-review-2026-10-17.md)。本文为同日后续轮次：先建可复现基线，再用插桩定位瓶颈，仅保留经测量的改动；数值容差不放宽。

> 同日审查后的计时修复与进一步优化见 §9。§1/2/5 中误把含同步 API 墙钟当作纯 CPU 提交时间的解释已更正；此前端到端结果保留为历史数据，不能与 §9 不同机器状态下的新运行直接相减。

## 1. 环境与方法

Windows、RTX 5070 12 GB、CUDA 13.0、cuDNN 9.14、Candle 0.11.0、Rust release；官方 Python torch 2.8.0+cu128 / transformers 4.52.1（revision、权重、语料、参考音频与对照记录完全一致）。语料 `tools/tts/acceptance-corpus.txt` 6 段 × 2 轮 = 12 次，seed 基值 7，逐段 seed = 7 + round×1000 + seg。**所有 GPU 进程串行运行**，无并行编译/推理。

新增 env 开关的分阶段插桩（`INDEXTTS_PROFILE=1`，默认关闭，环境值通过 `OnceLock` 读取一次）：

- 模型层：`emovec / gpt / codec / regulator / cfm / bigvgan`，CUDA 计时在阶段入口与出口 `Device::synchronize`，归因正确（剖析运行专做，不计入基线）。
- 生成循环：`gen.prompt_forward / gen.forward / gen.head / gen.warpers / gen.sampling / gen.reorder`；GPT2 层内 CPU 侧 `g2.ln_qkv / g2.split / g2.kvcache / g2.attn / g2.proj_mlp`（无独立设备同步，只记录 API 墙钟，不据此推断排队深度）。
- CFM 内：`cfm.pack / cfm.estimator / cfm.combine / cfm.integrate`。
- 同日后续复核已移除 `*.submit` / `cfm.estimator_submit`：旧计时包含嵌套同步或阻塞回传，不能解释为纯 CPU 提交时间。无设备的内层 scope 只表示 host API 调用墙钟。
- 输出经 `serde_json` 序列化为 `{"profile":{"stage":[seconds,calls]}}`；每次 generate 的 RAII owner 在成功和失败时都清空报告，避免污染后续请求。旧 `*profile*.log` 中的手工拼接格式不是合法 JSON，仅保留为历史产物。

测量局限（如实声明）：无 kernel 级 profiler（本机无 nsight/compute；未虚构瓶颈占比）；单段剖析进程之间存在约 ±10% 的 CPU 频率态漂移（未改动代码的 `g2.proj_mlp` 等在两组进程间漂移 8–16%），因此阶段级对比只用于定性，量化以 12 段语料整轮 A/B 为准。

## 2. 瓶颈结论

基线（优化前代码，本轮实测）：12/12 成功，总音频 98.562 s，生成 69.398 s，**RTF 0.704**（与既有记录 0.700 一致，跨日可复现）。

单段剖析（seg3 文本，6.19 s 音频 / 4.09 s 生成 / 157 步）分阶段：

| 阶段 | 耗时 | 占比 | 备注 |
| --- | --- | --- | --- |
| GPT 循环 | 1.537 s | 37% | 每步 7.5 ms；层内 CPU 提交 6.5 ms |
| CFM（estimator） | 0.932（0.916）s | 22% | 25 步 × 36.6 ms |
| emovec | 0.611 s | 15% | 每次生成都重算 |
| BigVGAN | 0.627 s | 15% | 单次 |
| codec 解码 | 0.467 s | 11% | 单次 |
| regulator | 0.004 s | <1% | — |

**计时归因更正**：旧 `cfm.estimator_submit` 跨过了 estimator 出口同步和 combine；`gpt.submit` 包含逐 token 同步；`bigvgan.submit` 包含阻塞回传。因此这些计时接近含同步墙钟不能证明 CPU 派发受限，也不能据此排除带宽或计算瓶颈。上表只作为插桩下的阶段墙钟，不作为 CPU 利用率或 GPU 空闲比例。端到端语料的加速和 WAV 一致性证据仍独立成立。

**未确认项**：算子派发、内存分配和同步都是候选开销；没有 kernel / allocator 级剖析，不宣称 Candle/驱动没有缓存分配器，也不据此解释算子成本差异。

**实测的反常点**：codec 解码在 **CPU 上 0.094 s vs CUDA 上 0.467–0.510 s**（同一段、同权重）——该阶段张量小，cuDNN 分组卷积反而慢约 5 倍。这是 §4 实验开关的依据。emovec 同样 CPU 更快（0.135 vs 0.30），但它位于采样之前，设备切换会改变 GPT 条件进而改变采样轨迹，不动。

## 3. 已实施优化（默认路径，全部位级等价）

语义影响一栏为"无"的项均在同 seed 语料 A/B 中 **12/12 WAV 逐样本一致**（见 §6 验证）。

| # | 位置 | 改动 | 原理 | 语义影响 | 维护成本 |
| --- | --- | --- | --- | --- | --- |
| O1 | `model.rs` / `gpt.rs` | emovec 计算移出段循环；单参考时走 `emovec_from_reference`（conformer 只跑一次） | 同一有限参考向量的差为零，原实现把同一参考喂两次 conformer | 默认语料 WAV 一致；情绪随机锚点在改动前后都在段循环外选择，没有因此改变其 RNG 消耗顺序 | 极低 |
| O2 | `gpt2.rs` | causal+padding 合成单个加性掩码（`s==1` 时 causal 全零直接跳过；`padding_additive` 用单 `affine` 核） | 每层 2 次 `broadcast_add` → 1 次；`m·(−MIN)+MIN` 与 `(1−m)·MIN` 逐值相等（含 0/−0 与 MIN+MIN→−inf 的组合） | 无 | 低 |
| O3 | `gpt2.rs` | `split_qkv_heads`：5D 视图一次 gather 代替 narrow→contiguous→reshape→transpose→contiguous 两次拷贝 | q/k/v 各省 1 次拷贝与派发 | 无 | 低 |
| O4 | `generation.rs` | beam 恒等置换时跳过 `reorder_beams` | `index_select` 恒等映射是纯拷贝 | 无 | 极低 |
| O5 | `generation.rs` | warper/采样 finite-only：top_k 用 `select_nth_unstable`（同一顺序统计量）；top_p 与不放回采样只遍历有限项（top-k/top-p 后仅 ≤数十个） | `(-inf−max).exp() == 0` 精确成立，0 权重对分母、商的和、累计 CDF 与 `last_positive` 回退全部是精确加 0，逐位复现全行扫描；采样器从每次抽取 2×3×8194 次超越函数降到幸存候选数 | 无（单测覆盖集中分布/确定性/候选不足） | 中（等价性注释已写明，附回归测试） |
| O6 | `dit.rs` | `LinT2d`：DiT 全部 3D×2D 投影在加载期转置一次 | 原每次调用 `.t().contiguous()`，25 步 × ~90 投影反复拷贝权重并各烧一次派发；缓存操作数值位相同 | 无 | 低 |
| O7 | `length_regulator.rs` | out_proj 权重加载期预转置 | 同 O6，每次调用省一次转置拷贝 | 无 | 极低 |

**效果（12 段语料整轮，同 seed）**：RTF 0.704 → 0.613（**−13%**），12/12 成功，总音频 98.5624 s 完全一致，12 个 WAV 逐样本一致。分阶段复测：emovec 0.611→0.313、warper 0.113→0.024、采样 0.068→0.0016。

**已撤回（无收益，如实记录）**：DiT RoPE 表按长度缓存 + `apply_rope` 公式重构（`x·cos + rot(x)·sin`，位级等价论证成立并有单测）+ qkv 窄视图化——语料整轮 0.622–0.625，不优于批次一的 0.613；虽然单测与 WAV 一致性均通过，收益低于进程间漂移，按"无明确收益即撤回"回退，避免携带无价值的复杂度。

## 4. 实验开关（默认关闭，单独验证）

`Options::decode_on_cpu`（example 旗标 `--decode-on-cpu`）：GPU 模型设备上把**语义 codec 解码**路由到惰性加载的 CPU codec 副本（codes 下行 ~KB 级、特征上行 ~3 MB，均计入实测）。依据即 §2 的实测反差（CPU 0.094 s vs CUDA 0.47–0.51 s/段）。

- **效果**：RTF 0.616–0.623 → **0.564–0.566**（较优化前基线累计 −20%），12/12 成功。
- **独立验证**：12 段音频时长与 GPU 路径**完全一致**（codes 在采样之后，帧数与设备无关）；RMS 一致到 4 位小数、过零率一致到 3 位；波形逐样本差异最大 ~13% 满刻度（单点），来源是 s_infer 的后端舍入差经 CFM 25 步积分放大——**不是位级一致，默认关闭，采用前须用户试听**。
- 维护成本：中（一份 CPU codec 惰性加载 + 两处设备拷贝；首次调用多 ~0.3 s 加载）。

## 5. 与官方 Python 的位置

串行同日语料（官方 12/12、生成 46.70 s、RTF 0.482；既往另一日 0.450，官方自身也有 ~7% 日间波动）：

| 版本 | 成功 | 生成总时 | RTF | 相对官方 |
| --- | --- | --- | --- | --- |
| 官方 Python CUDA | 12/12 | 46.70 s | 0.482 | 1.00× |
| Rust 优化前 | 12/12 | 69.40 s | 0.704 | 慢 ~55% |
| Rust 默认路径（本轮） | 12/12 | 60.46–61.59 s | 0.613–0.623 | 慢 ~28% |
| Rust + `--decode-on-cpu`（实验） | 12/12 | 55.76 s | 0.564–0.566 | 慢 ~18% |

剩余差距需要进一步测量。官方与 Rust 的 logits / 采样路径及张量组织不同；现有 scope 不能证明 CPU 派发是主导瓶颈，也不能证明不升级 Candle 或不写自定义核就已到优化上限。

## 6. 验证记录（全部通过）

- 94 无模型单测（含新增采样/warper 等价性与边界回归）；`cargo clippy -p indextts --all-targets -- -D warnings` 零告警；全仓 `cargo fmt --all --check`；rustdoc `-D warnings --document-private-items --examples`；workspace `cargo check` 通过。CUDA release 构建仅既有 LNK4098 警告（保留，未改 CRT）。
- 真权重 oracle：**16/16 通过，0 ignored，22.93 s**（INDEXTTS_CONVERTED_DIR/TIKTOKEN/ORACLE_JSON/CAMPPLUS_LAYERS/EMO_LAYERS/REGULATOR_LAYERS 全设置）。关键数值与既往一致：campplus.stage 5.3e-6、emovec 5.96e-7、regulator 3.8e-6、w2v.hidden_17 1.56e-4、首步 logits 1.383e-5。容差未放宽。
- 语料：默认路径最终二进制两轮 12/12（0.616 / 0.623），**12 个 WAV 与优化前基线逐样本一致**；日期句（固定规范化文本、seed 9、无文本正则化）6.908 s 正常，RTF 0.61。实验路径两轮 12/12（0.5657 / 0.5643），时长与默认路径逐段相等。
- 当时的状态检查：LinT2d 为加载期只读；emovec 为调用内常量；批次二的 rope 缓存已撤回。后续参考派生缓存及其替换失效验证见 §9。实验路径的 CPU codec 为无状态 `&self` 解码器；剖析清理的失败路径缺口已由 RAII owner 修复。
- 显存：0.5 s 间隔采样，corpus 单轮峰值 **8439 MiB**（含运行前桌面占用 1280 MiB；采样法，非分配器精确峰值；实验路径会额外常驻一份 CPU codec 权重在内存而非显存）。

## 7. 复现命令

Git Bash，仓库根；产物均在 `target/indextts-perf/`（Git 忽略）：

```sh
# 构建（LNK4098 为既有警告）
MSYS2_ARG_CONV_EXCL="*" CUDNN_LIB_PATH='D:\dev-tools\cudnn\v9.14-cuda13\lib\x64' \
LIB='D:\dev-tools\cudnn\v9.14-cuda13\lib\x64' \
cargo build --release --locked -p indextts --features cuda-cudnn --example generate --example corpus

# 语料（默认路径 / 实验路径）
PATH="/d/dev-tools/cudnn/v9.14-cuda13/bin:$PATH" ./target/release/examples/corpus \
    --dir /d/workspace/index-tts/checkpoints-converted \
    --reference /d/workspace/index-tts/examples/voice_01.wav \
    --corpus tools/tts/acceptance-corpus.txt \
    --out-dir target/indextts-perf/<run> --rounds 2 --seed 7 --device cuda [--decode-on-cpu]

# 分阶段剖析（stderr 的 {"profile":...} JSON，含 CUDA 同步边界）
PATH="/d/dev-tools/cudnn/v9.14-cuda13/bin:$PATH" INDEXTTS_PROFILE=1 \
./target/release/examples/generate --dir ... --reference ... --text "行长走过银行门口，看见远处重重的山影。他说这项工作很重要。" \
    --lang zh --out target/indextts-perf/prof.wav --seed 9 --device cuda

# 官方对照 / oracle 环境变量：见 indextts-official-community-review-2026-10-17.md §复现
```

## 8. 剩余风险与未做

- 阶段归因受 ±10% 进程间 CPU 频率态漂移影响；本文所有量化结论以 12 段整轮 A/B 为准，单段剖析仅定性。
- `--decode-on-cpu` 为实验：非位级一致（§4），默认关闭；是否转正需用户试听 gate。
- 未测量/未做：kernel 级 profile、Metal、CUDA 图、自定义融合核、emovec 设备路由、GPT 循环 KV cache 预分配。此前“Candle 无张量原地写 API / 预分配必然净损失”的判断撤回；仓库 Qwen 已有原地缓存实现，但其与 beam 重排的取舍仍需单独验证。
- 用户试听 gate 仍未执行；两路径音频在 `target/indextts-perf/final-rust-a/` 与 `target/indextts-perf/final-rust-cpu-decode/`，官方在 `target/indextts-perf/python-cuda/`。

## 9. 审查后的修复与第二轮优化

### 诊断修复

- 删除手工 JSON 拼接，使用 `serde_json` 序列化阶段名到 `[seconds, calls]`，真实日志已用标准 JSON parser 检查。
- 移除所有误导性的 submit 指标。保留的带设备 scope 是入口/出口同步墙钟；无设备 scope 是 API 调用墙钟，可能包含阻塞，二者都不能直接给出 CPU/GPU 利用率。嵌套同步会扰动执行，剖析与性能对照分开运行。
- `OnceLock` 替代两项独立原子初始化状态。每次 `generate` 有一个 RAII 报告 owner，进入时清空旧数据、退出时排空，包括推理错误路径；新增合法 JSON / 次数累计、失败后清理回归测试。

### 保留的默认路径优化

| 改动 | 生存期与语义 |
| --- | --- |
| 缓存参考情绪向量 | 同一已准备参考的首次合成才计算，后续请求复用。成功 `prepare_reference` 后失效；独立情绪参考合并仍走原路径。没有把首次计算移到基准计时外，也没有持久化或全局缓存 |
| GPT key 缓存按 attention 布局保存 | key 存为 `(B,H,D,T)`，只转置新增 key，再沿 T 追加；value 保持 `(B,H,T,D)`。避免每步重排完整历史 key，beam 重排仍沿 batch 维执行 |
| CFM 固定条件只准备一次 | guidance 的 prompt / mu / style 拼接和 DiT 条件线性投影移到 Euler 循环外；求解结束即释放，不跨参考、长度或请求复用。步数、时间序列、算术顺序、seed 和采样参数不变 |

普通测试新增了 beam 重排后增量注意力与整段注意力比较，以及不同长度 / timestep / drop_style 下缓存投影与新投影的比较。真实权重测试增加参考向量复用、替换参考后失效、替换结果与新计算一致的检查。

### 测量方法与产物

当前机器、原语料 6 段 × 2 轮、原参考、seed 7、默认全 CUDA 解码、F32、25 步。使用保存的优化前后独立二进制，按 **after-a → before-b → before-c → after-b** 顺序串行运行，每个进程都重新加载模型并准备参考。运行期间不并行执行测试、编译或另一推理。先前 `before-a` 与准备检查存在重叠，只作预热，不纳入正式前后均值。

证据目录：`target/indextts-followup/`。每个运行目录有 `index.json`、12 个 WAV，以及记录进程总墙钟与二进制 SHA-256 的 `process.json`；`before-src/` 保留本次开始时的相关源码快照。`profile-before.log` 是修复诊断后、进一步优化前的合法剖析日志。

验证：98 项普通单测和两个 example 测试目标通过；17 项真实权重检查通过（0 ignored，32.13 s，包括新增参考替换检查）。Clippy `--all-targets -- -D warnings`、rustdoc `-D warnings` 通过；CUDA 两个 release 探针构建成功，仍保留既有 LNK4098 警告。

### 结果

| 进程 | 成功 / 尝试 | 音频总时长 | 合成总时长 | RTF | 进程总墙钟 |
| --- | --- | --- | --- | --- | --- |
| after-a | 12/12 | 98.5624 s | 51.3746 s | 0.52124 | 57.6699 s |
| before-b | 12/12 | 98.5624 s | 56.5569 s | 0.57382 | 62.1805 s |
| before-c | 12/12 | 98.5624 s | 56.4514 s | 0.57275 | 61.7603 s |
| after-b | 12/12 | 98.5624 s | 51.7450 s | 0.52500 | 57.0079 s |

两次均值：**RTF 0.57328 → 0.52312，合成调用耗时下降 8.75%**；进程总墙钟 **61.9704 → 57.3389 s，下降 7.47%**（含加载、准备和写入）。四个运行均 12/12 成功，优化后两组共 **24/24 WAV 文件与 before-b 逐字节一致**；before-c 也与 before-b 一致。原始汇总 `comparison.json` 记录所有均值、文件一致性、二进制 SHA-256 与语料 SHA-256。本次未重新跑官方 Python，不能把旧官方的不同运行直接当作当前机器状态下的速度上限。

修正后的独立剖析（不计入上表）：同 seed 9 的首句，GPT 1.4345 → 1.3303 s，CFM 0.8384 → 0.8315 s；后者变化很小，不单独宣称显著收益。同一参考的第二次调用，emovec 从首调用 0.3106 s 降为约 2.9 µs；这是派生值复用，首次计算仍在合成计时内。第二次剖析使用 seed 10，不能用它的其他阶段时长与 seed 9 做严格比较。三项优化的独立贡献没有逐项消融，收益以组合的未插桩 A/B 为准；不根据这些阶段墙钟推断 GPU 利用率。

`profile-before.log` 的 1 条报告和 `profile-after.log` 的 2 条报告均能直接 `json.loads`；汇总在 `profile-comparison.json`，没有遗留 submit 指标。新增缓存只绑定当前已准备参考，不属于资源哈希绑定的持久缓存；没有扩大当前支持语言、设备验收或人工听感结论。

本轮未重新采样显存峰值；§6 的显存值属于上一轮运行。

复现新增代码的基准（PowerShell，先按 §7 构建两个 CUDA examples）：

```powershell
$env:PATH = 'D:\dev-tools\cudnn\v9.14-cuda13\bin;' + $env:PATH
$env:INDEXTTS_PROFILE = '0'
target/release/examples/corpus.exe --dir D:/workspace/index-tts/checkpoints-converted --reference D:/workspace/index-tts/examples/voice_01.wav --corpus tools/tts/acceptance-corpus.txt --out-dir target/indextts-followup/reproduce --rounds 2 --seed 7 --device cuda

# 单独的逐调用剖析：同参考重复两段，检查首次计算与缓存命中。
$env:INDEXTTS_PROFILE = '1'
target/release/examples/corpus.exe --dir D:/workspace/index-tts/checkpoints-converted --reference D:/workspace/index-tts/examples/voice_01.wav --corpus target/indextts-followup/profile-corpus.txt --out-dir target/indextts-followup/reproduce-profile --rounds 1 --seed 9 --device cuda

# 全部真实夹具变量仍按官方对照记录设置，再运行这 17 项检查。
cargo test --release --locked -p indextts --lib -- --ignored --test-threads=1
```

## 10. 第三轮：codec 深度卷积与 BigVGAN FIR（当日续）

同日第三轮，以 §9 之后的工作树为 before（该轮基线二进制已另存 `target/indextts-depthwise/before-corpus.exe`，sha256 `278fb619…`）。目标按证据排序：semantic codec 深度卷积、BigVGAN FIR、CFM（视证据）。全部测量为串行、同机、同权重/参考/语料/seed；ABBA 顺序（before-after-after-before）抵消进程间 CPU 频率态漂移。

### 上游核实

- candle issue [#3389](https://github.com/huggingface/candle/issues/3389)（open）：`conv1d` 在 `groups>1` 时逐 group 调用单组卷积再 cat，深度卷积产生 O(groups) 微型内核发射；RTX 5090 上报每层 33 ms。
- PR [#3531](https://github.com/huggingface/candle/pull/3531)（**仍 open、未合并**）：深度卷积 fast path，条件 `c_in_k==1 && c_out==c_in && groups==c_in && stride==1 && dilation==1 && is_cuda()` 加防回退门槛（`3·groups·2 ≥ (2·taps−1)·3` 且 `group_work ≤ 2²⁰`），实现为 pad + 逐 tap narrow·broadcast_mul 累加；测试用 block-diagonal 稠密卷积交叉验证，注明数值仅求和序等价。**该轮未采用其 `is_cuda` 限定**：CPU 只验证了数值，未提供 CPU/Metal 性能证据；后续审查已恢复 CUDA 限制（§11）。未升级/替换 Candle、未改本机 registry。

### 已实施（该轮保留；后续审查见 §11）

1. **codec 深度卷积 fast path**（`codec.rs` `DepthwiseConv1d` + `depthwise_conv_or_stock`）：按 PR #3531 思路本地实现，PR 的形状门槛与工作量门槛全部保留（载入期查 margin，调用期查 group_work；不满足即回退原 `Conv1d` 分组路径）。bias 以 `(1,C,1)` 广播，padding/bias/dtype/设备/边界语义与原路径一致。效果：单段 codec 阶段 **0.473 s → 0.010–0.021 s**。数值：固定输入对照 7 组形状（含实际几何 (1,384,688)/(1,384,1551) 与边界长度），对 stock 路径 max_abs ≤ 4.8e-7、无 NaN/Inf、形状一致（`codec::tests::depthwise_fast_path_matches_stock_conv`，逐案打印误差）。
2. **BigVGAN FIR 批量卷积化**（`bigvgan.rs` `FirFilter`）：FIR 滤波 tap 对所有通道相同，`(B,C,L)` 可免费 view 成 `(B·C,1,L)`，用共享 `(1,1,K)` 核的单次 conv1d 替换逐 tap 的 narrow+affine+add 链（~3K 次全张量遍历→1–2 个卷积核）。`fir_downsample` = stride-2 conv；`fir_upsample` 保留奇偶相位分解但相位用批量 conv（`fir_conv_sum` 为反转核的 batched conv + 切片，语义即 `np.convolve` 切片）。滤波器、replicate padding、裁剪窗与输出长度不变。效果：bigvgan 阶段 ~1.09 s → ~0.91 s（fir_down 0.331→0.195、fir_up 0.312→0.266，含会话漂移的单段剖析）。数值：新实现对逐 tap 参照（保留在测试中）max_abs ≤ 4.8e-7（4 组形状含 (2,128,641)），torch `UpSample1d/DownSample1d` 模块级对照（1e-5 容差）继续通过。
3. 剖析插桩补充（`bv.*` 作用域，env 开关，默认关闭；关闭时仍有开关检查）。

### 已撤回（本轮，如实记录）

- **BigVGAN ResblockGroup 合并执行**：把每级 3 条并行 resblock 的逐元素链按通道拼接批量跑（卷积保持逐块）。位级等价论证成立、测试通过，但未测得稳定收益；额外拼接/复制可能影响结果，现有测量不能证明带宽是主导瓶颈：单段剖析 1.094 s（合并）vs 回退后同漂移归一 ~0.98 s，收益在漂移带内；按"无明确收益即撤回"回退，保留顺序 `AmpBlock` 路径。此前的 "submit≈sync" 判读被阶段末阻塞读干扰；没有 kernel 级指标，不能据此确定 CPU 派发或内存带宽的占比。
- **fir_upsample 直接 conv_transpose1d 形式**：代数等价（相位分解即其拆式），但 candle 的 transposed-conv 核在 `(B·C,1,L)` 批形状上走慢路径（单段 0.40 s vs 相位形式 0.25 s），撤回并留注释。

### CFM（按计划暂缓）

CFM 常量外提/条件投影已在 §9 完成；本轮剖析 cfm.estimator 0.86–0.91 s，剩余为 DiT 本体算子（尚未做 kernel 级归因），无新的结构简单且证据明确的改动，未动。

### 测量（ABBA，6 段 × 2 轮 = 每进程 12 次，seed 7）

| 运行 | 二进制 | RTF | 首调 | 除首调 RTF | 成功 |
| --- | --- | --- | --- | --- | --- |
| before-A | before-corpus.exe | 0.5606 | 2.057 s | 0.5510 | 12/12 |
| after-B1 | 本轮 | 0.4734 | 1.521 s | 0.4676 | 12/12 |
| after-B2 | 本轮 | 0.4648 | 1.486 s | 0.4592 | 12/12 |
| before-A2 | before-corpus.exe | 0.5357 | 1.981 s | 0.5264 | 12/12 |

before 均值 0.5481 / after 均值 0.4691 → **生成耗时 −14.4%**。最终二进制追加 run-B3：RTF 0.4409（除首调 0.4358），同会话紧随其后的官方 Python：RTF 0.4753（首调 3.595 s，除首调 0.4477）——两端进入同一噪声带（官方当日早间另测 0.4414）。合计三轮：0.704 → 0.469（ABBA 均值）。

### 音频与数值验证

- 输出音频与改动前（`target/indextts-current-comparison/rust-cuda`）**逐段时长完全相等**（codes 在两处算子改动之前生成，帧数不受影响）；总时长 98.5624 s。RMS 相对差 ≤1%、过零率 3 位小数一致、无 NaN/异常静音、峰值 < 32767 无新增削波。波形不再逐字节相同：单样本差至 ~7k/32767，来源为深度卷积与 FIR 求和序变化经 codec→CFM→BigVGAN 的传播（两类算子的固定输入偏差均 ≤ 4.8e-7）。这些检查只证明长度及部分信号统计一致，不能证明波形内容逐字完整或听感等价；内容完整性与主观听感尚未验收。
- 真权重 oracle **17/17 通过（32.0 s，六个夹具 env 全设）**：codec.decoded 对 Python max_abs **6.378e-6**（改前 6.91e-6，容差 5e-4 未放宽）、codes 0/61 错配不变、其余各项与 §6 一致。
- 101 无模型单测（新增深度卷积固定输入对照、FIR 参照对照、 ineligible 配置回退）；Clippy `-D warnings` 零告警；全仓 fmt；rustdoc `-D warnings`；CUDA release 重建仅既有 LNK4098。日期句（固定规范化文本 seed 9）6.908 s 正常。
- 状态检查：`DepthwiseConv1d`/`FirFilter` 为加载期只读结构；无跨请求缓冲复用；连续不同长度输入（语料 6 段 × 4 次进程级重复）无污染。

### 复现（本轮）

```sh
# 构建（LNK4098 为既有警告）
MSYS2_ARG_CONV_EXCL="*" CUDNN_LIB_PATH='D:\dev-tools\cudnn\v9.14-cuda13\lib\x64' \
LIB='D:\dev-tools\cudnn\v9.14-cuda13\lib\x64' \
cargo build --release --locked -p indextts --features cuda-cudnn --example corpus --example generate

export PATH="/d/dev-tools/cudnn/v9.14-cuda13/bin:$PATH"
COMMON="--dir /d/workspace/index-tts/checkpoints-converted --reference /d/workspace/index-tts/examples/voice_01.wav --corpus tools/tts/acceptance-corpus.txt --rounds 2 --seed 7 --device cuda"
./target/indextts-depthwise/before-corpus.exe $COMMON --out-dir target/indextts-depthwise/run-a-before
./target/release/examples/corpus.exe $COMMON --out-dir target/indextts-depthwise/run-b1-after
# … ABBA 同构；剖析加 INDEXTTS_PROFILE=1（单段 stderr JSON）
```

结果目录：`target/indextts-depthwise/`（run-*、prof-*.log/json、before-corpus.exe、python-cuda/、date-check.*）。

## 11. 第三轮优化的审查与简化

### Standards 审查

- 本地 depthwise 实现去掉上游 CUDA 限制，CPU 正确性不证明 CPU/Metal 性能；恢复仅 CUDA 快速路径，其他设备走 stock 卷积。
- 将 stock/fast 两套字段、两个 guard 函数、外部 fallback helper 收进一个 owning wrapper，统一从 `Conv1d.config()` 取配置，去掉重复传参和输出长度重复计算。
- `FirFilter` 改为单一所有权，移除未使用的 `Arc`/`Clone` 与生产代码不需要的原始 taps；滤波器仍在加载期构造。
- 为改写的 Candle 算法补齐固定来源 revision 与 Apache-2.0 许可证。

### Spec 审查

- 原新增算子测试与 17 项真实权重 oracle 都在 CPU 上执行，不能作为 CUDA/cuDNN 数值证据。新增显式忽略的 CUDA 固定输入测试，分别对照 depthwise stock 卷积与 FIR 逐 tap 参照，普通测试不需要 GPU 或模型。
- 修正 §10 超出证据的表述：同 codes、同长度不证明逐字完整；现有阶段墙钟不能证明 BigVGAN 带宽受限。

### Git 依赖评估

核实官方 PR #3531 head 为 `70bf01e7a56c8cc8d42649e78564d83930ab2cf7`，其根 `Cargo.toml` 声明 **0.10.2**；当前项目统一使用 **0.11.0**。官方 main 检查到 `12bb5d1f6350f4ea9827d9086f213feec1f0af6a`，`conv.rs` 仍无该 fast path。直接把依赖 URL 改成官方 Git main 不会获得修复；指向 PR head 则需回退全工作区 Candle 家族及重新验证其他模型和 native 构建。

此外，现有 `.github/scripts/publish-libraries.py` 发布 Candle 消费库到 crates.io；Git-only 依赖不能原样发布，Git+version 在打包时回到 registry 版本，根 `[patch]` 也不会传递给下游。只在应用源码构建采用固定 Git 提交是可行方案，但目前没有包含此修复的兼容 0.11 官方提交。此次不降级、不引入 fork，不修改 Cargo.toml/Cargo.lock；待兼容修复进入上游后再统一切换并删去 wrapper。

新增 `(1,32,22048)` 的真实音频量级 FIR CUDA 对照暴露了原实现的缺口：auto 算法 upsample max_abs **1.063e-3**，超过原定 **1e-5**；小形状均在约 4.8e-7 内，故原 CPU/小形状证据未发现它。独立固定输入算法探针中，auto/ImplicitPrecompGemm 约 2.326e-4，ImplicitGemm/Gemm 约 5.960e-8。最终 FIR 明确选择 `ImplicitGemm`，保持容差，不改全局 CUDA 设置；没有确认 auto 所选具体 kernel/math type，不能确定归因为 TF32。复现探针与日志为 `fir-algo.rs` / `fir-algo.log`。

新增完整 codec 的 CUDA 对 CPU oracle 检查：默认 TF32 环境有元素超出原 `atol=5e-4 / rtol=5e-3`；以 `NVIDIA_TF32_OVERRIDE=0` 在进程启动前禁用 TF32 后，max_abs **9.537e-6**，原容差通过。此环境仅用于明确的严格 F32 验证，生产代码与后续性能对照不设置它；默认 CUDA 的跨设备 oracle 限制仍存在，不能把严格 F32 的结果当作默认精度的验证。

新增算子检查：默认 CUDA/cuDNN 下 depthwise max_abs **4.768e-7**；修复后的 FIR 大形状 max_abs **7.153e-7**，保持原 **1e-5** 容差。最终严格 F32 重跑三项 CUDA 检查均通过。

回归：101 项普通测试通过；17 项真实权重检查通过（release 测试二进制，39.02 s，全部夹具变量已设置）；CPU oracle codec max_abs **6.914e-6**。CPU Clippy、rustdoc `-D warnings`、fmt、CUDA release examples 构建通过，链接仍有既有 LNK4098；无依赖/CRT 改动。

CUDA Clippy `--features cuda-cudnn --all-targets -- -D warnings` 亦通过。审查/运行产物：`target/indextts-review/`；`local-corpus.exe` 和两个 `local-*.rs` 是本轮开始时的基线快照。

### 最终代码的端到端对照

全部检查结束后，保存最终二进制 `reviewed-corpus.exe`，按 `local → reviewed` 串行运行各 6 段 × 2 轮，seed 7、同模型/参考、默认 CUDA/cuDNN、`INDEXTTS_PROFILE=0`，**未设置 NVIDIA_TF32_OVERRIDE**。这只是单组 A/B，未做 ABBA，不将百分比解释为稳定收益，也没有重新跑官方 Python。

| 版本 | 成功 | 音频总时长 | 生成总时长 | RTF |
| --- | --- | --- | --- | --- |
| 本次审查前的优化实现 | 12/12 | 98.5624 s | 46.6451 s | 0.47325 |
| 简化并修复 FIR 算法选择后 | 12/12 | 98.5624 s | 43.6272 s | 0.44264 |

本组生成耗时减少 **6.47%**。12 个输出逐段长度一致，相对审查前优化实现的 RMS 变化不超过 **1.0%**，非逐样本一致；不代表内容完整性或主观音质已验收。二进制 SHA-256、输出比较见 `target/indextts-review/comparison.json`；每轮目录 `process.json` 记录墙钟和开关。复现使用 §9 命令，将可执行文件替换为上述两个快照、输出目录分别设为新的忽略目录即可。

明确保留的限制：默认 CUDA 整 codec 对 CPU oracle 的严格数值容差未通过（严格 F32 环境通过）；Metal 未实机验证；人工听感与 ASR 内容回读未执行。本轮未升级任何 native 依赖，未提交/推送代码。

## 12. 最终代码与官方 Python 的当前对照

§11 全部修复和简化完成后，用当前工作树重新构建 CUDA release examples（`corpus.exe` SHA-256 `2a2392eb…55fb`），与官方 Python 按 **Python A → Rust B → Rust B2 → Python A2** 串行交替对照。两端使用同一语料、参考音频、seed 7、`cuda`、2 轮（6 段 × 2 = 12 次）、`INDEXTTS_PROFILE=0`；Rust 使用转换后 checkpoint，Python 使用官方 checkpoint。未设置 `NVIDIA_TF32_OVERRIDE`，四次运行之间没有并行构建或推理。运行时 GPU 为 RTX 5070 / driver 616.56；结束后空闲状态约 41 °C、25.9 W、615 MHz SM clock。

计时口径与 corpus runner 一致：`generate_secs` 从每次 `infer()` 调用开始，到 WAV 写出结束；`load_secs` 单独记录，不计入 RTF。“稳态”口径排除每个进程的第一条请求，用于移除首次 CUDA kernel/缓存初始化影响。

| 运行 | 模型加载 | 成功 | 音频总时长 | 生成总耗时 | RTF | 首条耗时 | 除首条 RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python A | 18.121 s | 12/12 | 96.8905 s | 44.0878 s | 0.45503 | 4.6669 s | 0.41573 |
| Rust B | 4.571 s | 12/12 | 98.5624 s | 41.6817 s | 0.42290 | 1.3676 s | 0.41763 |
| Rust B2 | 2.114 s | 12/12 | 98.5624 s | 41.3808 s | 0.41984 | 1.3152 s | 0.41506 |
| Python A2 | 14.672 s | 12/12 | 96.8905 s | 44.2309 s | 0.45650 | 3.4628 s | 0.42994 |

按两次运行取均值：官方 Python 整体 RTF **0.45577**，Rust **0.42137**，Rust 低 **7.55%**；排除每个进程首条请求后，Python 稳态 RTF **0.42283**，Rust **0.41634**，Rust 低 **1.53%**。因此当前主要可感知优势来自首个推理请求（Python 3.46–4.67 s vs Rust 1.32–1.37 s）和模型加载（Python 14.7–18.1 s vs Rust 2.1–4.6 s）；在长会话稳态下，两端已经处于同一性能带，Rust 只有小幅领先。

解释边界：两端 RNG 不 transferable，同 seed 下 Python 与 Rust 生成的音频长度不同（96.8905 s vs 98.5624 s），RTF 已按各自生成时长归一，但长度分布仍可能带来少量偏差；ABBA 只有一次完整重复，1.53% 的稳态差距不应外推为稳定收益。当前结论也没有人工听感或 ASR 内容回读验收。官方 Python 输出的乱码文本提示来自 Windows 控制台编码显示，`index.json` 本身按 UTF-8 写入。

结果与复现：

```powershell
$env:PATH = 'D:\dev-tools\cudnn\v9.14-cuda13\bin;' + $env:PATH
$env:INDEXTTS_PROFILE = '0'
D:/workspace/index-tts/.venv/Scripts/python.exe tools/tts/indextts_corpus_py.py `
  --checkpoints D:/workspace/index-tts/checkpoints `
  --reference D:/workspace/index-tts/examples/voice_01.wav `
  --corpus tools/tts/acceptance-corpus.txt `
  --out-dir target/indextts-latest-official-comparison/python-a `
  --rounds 2 --seed 7 --device cuda

target/release/examples/corpus.exe `
  --dir D:/workspace/index-tts/checkpoints-converted `
  --reference D:/workspace/index-tts/examples/voice_01.wav `
  --corpus tools/tts/acceptance-corpus.txt `
  --out-dir target/indextts-latest-official-comparison/rust-b `
  --rounds 2 --seed 7 --device cuda

# 按 Python A → Rust B → Rust B2 → Python A2 继续替换输出目录完成 ABBA。
```

完整输出与机器可读汇总在 `target/indextts-latest-official-comparison/`（四个 `index.json` 与 `comparison.json`）。本轮只做性能重测与记录，没有改变行为代码。
