# OmniVoice Metal 原生 GQA 增量优化（2026-10-23）

本轮只改变 stage0 注意力的 K/V 路由：Metal 直接传入 8 个 KV 头，
CPU 和其他设备仍沿用 repeat_kv + manual attention。不新增依赖或运行时实验开关。
基线为 `44f0f12`，已包含前轮融合 SDPA/RMSNorm 和 decoder 转置卷积优化。

## 原因与范围

固定 Candle fork `2266099d92265112ecc0d56ecee1761e3cc2ac00` 的
`candle-nn/src/ops.rs` 明确要求 GQA 不预先平铺 K/V。
`candle-metal-kernels/src/kernels/sdpa.rs::call_sdpa_full` 计算
`gqa_factor = q_shape[1] / k_shape[1]`；Metal kernel 用
`query_head / gqa_factor` 映射到 KV 头，并读取原始 stride。
旧实现先通过 `repeat_kv` 的 cat 物化 K/V，每层每步重复一次。
移除它减少复制、分配和分派，不改变注意力公式或精度。

旧热点记录把 `q_head == k_head` 误读为头数相等；它实际检查 head dimension。
该记录已更正。真实短输入的 Q 为 `[2,16,157,128]`，K/V 为
`[2,8,157,128]`；CFG 拼接 batch，不是把 sequence 翻倍。
287 仅是额外合成长度，不能再称为该短语料的实际长度。

另核实本地官方运行环境：transformers 5.3.0 的
`integrations/sdpa_attention.py` 在 `attention_mask != None` 时也先 repeat_kv。
OmniVoice 的短语料批处理确实传入 mask。因此不能将本优化描述为照搬官方原生 GQA；
它利用的是固定 Candle Metal 内核已有能力，官方 MPS 与 Candle Metal 的路由不同。

## 验证方法

- M4 Pro，macOS 26.6.2；stage0 Metal F16、decoder F32。
- 本地固定 0.6b 权重 revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`。
- 单元测试覆盖长度 1/33/157/287，有无加性 mask，V 保持投影后 transpose view；
  对照直接 GQA 与重复 K/V，要求完整输出精确相等。
- `gqa_probe` 无需权重，包含 KV repetition、SDPA 和输出分配；mask 为 expand view，
  V 为 transpose view。预热后成组计时，仅组边界同步，按 ABBAABBA 顺序报告。
- 整链使用独立 release 二进制，32 步、seed 42、narrator、相同短文本和线程配置；
  每进程 1 预热 + 3 热轮，ABBA，测量时诊断关闭且不并行编译/推理。
- 输出验收另跑短/长语料，比较完整 codes dump、所有 F32 和 PCM16 WAV，
  此时开启 codes dump，不将该耗时当作正式吞吐。

## 下一批候选

1. 贪心预测当前分别做 argmax 和 max；可实验用 argmax 索引 gather confidence，
   减少一次全词表归约。随机采样仍须取全局 max，不能改成采样 token 的分数。
2. cond/uncond 的 F16 logits 切片后分别 strided cast F32；比较 contiguous 后转换、
   全 batch 一次转换，计时须覆盖后续 scoring，避免只把成本搬到下游。
3. decoder 尚有与官方的局部差距，继续按固定 codes 分离卷积、激活和调度成本。

不优先把 CFG 多次 log_softmax 代数合并：浮点归约和舍入可能改变离散 codes。
不把随机张量精确相等解释为所有模型/设备/采样配置都已验收；不从数值指标推断听感。

## 初筛与干扰记录

先用前轮冻结二进制 `probe-final-metal` 做 ABBA，A 两进程均值为
3.76349 / 3.87350 s，B 为 3.10989 / 3.16806 s，组均值
3.81849 → 3.13898 s（-17.80%）。12 个热轮 F32 WAV hash 全部相同。
进程采样未观察到编译器活动，但 A 的历史构建含其他后端 feature，
不将它作为本轮最终配对结论。

随后两侧均用 `--no-default-features --features omnivoice-metal,performance-tools`
独立构建，再做 ABBA：

| 顺序 | 路径 | 热轮耗时（s） | 进程均值（s） | 编译活动 |
| --- | --- | --- | --- | --- |
| 1 | repeat K/V | 3.87071 / 3.86485 / 3.73946 | 3.82501 | 未观察到 |
| 2 | native GQA | 3.09577 / 3.07540 / 3.05458 | 3.07525 | 观测到外部 Cargo，保留但标记受干扰 |
| 3 | native GQA | 3.20264 / 3.19713 / 3.19885 | 3.19954 | 未观察到 |
| 4 | repeat K/V | 3.88979 / 3.87584 / 3.83195 | 3.86586 | 未观察到 |

包含受干扰进程的组均值 3.84543 → 3.13739 s（-18.41%），仅作初筛。
不终止其他任务的进程，不把受干扰样本改名为静默基准。
排除第 2 进程后，A 两进程均值仍为 3.84543 s，B 单进程为 3.19954 s
（-16.80%，RTF 0.7121 → 0.5925）。这是样本数不平衡的无编译活动子集，
不是完整无干扰 ABBA，也不提供置信区间。
之后尝试启动反序 BAAB 确认，但启动检查发现其他任务持续编译，未开始测量；
本轮不为凑齐样本而在并行编译下补测，不声称已完成这组确认。
所有热轮生成的 F32 WAV SHA256 都是
`2eadc756eb4482fe320efe678ef8a67312ce79dcfa6d6f4dbf41ed4642be324e`。

## 完整输出对照

同 feature 的原路径/候选各跑一轮，无显式请求预热，但均保留 backend 自身的参考准备。
codes dump 包含参考准备与正文生成的全部记录；所有同名文件直接比较 SHA256。

| 输入 | 音频长度 | codes dump SHA256（A=B） | WAV 对照 |
| --- | --- | --- | --- |
| 短文本 | 5.4 s | `d1f037724f6a674505e3e0f52c5ae23d3c341d95cefb6c7ad74d7a7015d829c3` | 4/4 逐字节相同 |
| 长文本 | 51.12 s | `69dd8d8476dcb38aef411a588461245ae39026dc9159c4d6af63d5bdbb42b29c` | 24/24 逐字节相同 |

WAV 覆盖整段与各切分段，分别导出 F32 和 PCM16；不是仅比较最终 PCM16。
普通测试不下载模型，真模型验证只读取本机既有资源。

独立 release 二进制 SHA256：

- 原路径：`78c6e10b01d26764cfd346d3b53da5d067754d5f5236a422e3a79dd9401e0e5b`。
- 候选：`f0b24630a0cbbe49fbfea8cddc04af62288c29adedf968f143356ea8bacad47b`。

本轮没有重跑官方整链，不将之前官方的耗时直接与本轮新结果相减得出领先幅度。

## 工程验证

| 检查 | 结果 |
| --- | --- |
| Metal crate 测试 | 29 passed，新增 GQA 测试按 F32 bit pattern 精确比较 |
| 无默认 feature 的 CPU crate 测试 | 22 passed |
| Metal + Accelerate，Clippy all-targets，`-D warnings` | 通过 |
| Metal rustdoc，library + examples，`-D warnings` | 通过 |
| release `gqa_probe` 编译 | 通过 |

没有改变取消、会话所有权、采样配置或资源清单；没有新增运行时开关。
没有 push、合并、发布或部署。

## 复现

```sh
cargo test --locked -p talechime-omnivoice --features metal native_gqa_matches_repeated_kv_exactly
cargo build --release --locked -p talechime-omnivoice --features metal --example gqa_probe
target/release/examples/gqa_probe
cargo build --release --locked -p talechime-backends --no-default-features --features omnivoice-metal,performance-tools --example performance_probe
```

原始数据位于忽略目录 `target/perf/omni-followup/`；模型和输出音频不提交。
