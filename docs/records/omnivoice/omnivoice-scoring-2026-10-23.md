# OmniVoice Metal 贪心 confidence 与 logits 转型实验（2026-10-23）

基线 `a7c02ab`，已包含原生 GQA、融合 SDPA/RMSNorm 和转置卷积优化。
继续按算子、阶段、整链分别验证，不设默认整链改善百分比。

## 两个假设

1. 默认贪心预测先 argmax，再 max，重复扫描同一份词表。
   用 argmax 的 U32 索引 gather confidence，只读取选中位置，然后将 token 转 I64。
2. 当前 cond/uncond logits 是切片后分别 F16→F32；strided 转型可能较贵。
   比较先 contiguous 再转型、整批转型后再切片，并将后续 scoring 纳入计时。

设备为 M4 Pro / macOS 26.6.2；stage0 F16，scoring F32。
固定 Candle revision `2266099d92265112ecc0d56ecee1761e3cc2ac00`。
没有修改依赖或内核。

## 最小探针

`scoring_probe` 不读取模型，使用固定短输入形状：

- 选择阶段 `[1,8,130,vocab]`，vocab 为 1024/1025/4097。
- 转型输入 `[2,8,157,1025]`，cond 切片 27..157，uncond 0..130。
- 完整 scoring 包括 guidance=2 的三次 log_softmax、mask token 排除和最终选择。
- 先比较完整 token 与 confidence 的 F32 位模式，再预热，每组 100 次、组边界同步。
  两种选择方案采用 ABBA；三种转型方案采用 ABCCBA。

首次筛选启动前存在外部编译；实际探针运行期间每 250 ms 采样，未观察到编译器活动。
原始数据 `target/perf/omni-followup/scoring-screen.txt`：

| 部分 | 原路径（微秒/次） | 候选（微秒/次） | 初筛判断 |
| --- | --- | --- | --- |
| 1025 词表选择 | 65.826 / 60.600 | gather 38.032 / 40.624 | 继续验证 |
| 完整 scoring | 2601.630 / 2632.465 | gather 2575.802 / 2602.543 | 阶段收益小于选择算子收益 |
| cast + scoring | 3405.085 / 3355.321 | contiguous 后转型 3406.389 / 3410.044 | 不保留 |
| cast + scoring | 同上 | 整批转型后切片 3554.673 / 3506.094 | 不保留 |

1024 词表第一组原路径出现 153.256 µs、最后一组 76.405 µs 的启动漂移；
不删除该数据，也不拿该异常高值扩大收益。实际模型使用 1025 词表。
转型实验只拒绝这两个具体候选，不推断所有布局优化都无效。

## 三进程局部复测与保留结论

先完成编译和工程检查，再连续运行三次独立 release 探针。启动检查与运行期间
每 250 ms 采样均未观察到编译器活动；线程环境 Rayon/VecLib/OMP 各设为 8。
每进程内部的顺序与初筛相同；下表每格是该进程两组各 100 次的均值，单位 µs。

| 比较部分 | 原路径进程 1/2/3 | 候选进程 1/2/3 | 组均值变化 |
| --- | --- | --- | --- |
| 1025 词表选择 | 60.2165 / 62.8335 / 57.6340 | gather 34.8545 / 37.9275 / 37.0970 | 60.2280 → 36.6263，-39.19% |
| 完整 scoring | 2592.4895 / 2605.2030 / 2565.5365 | gather 2553.3365 / 2553.1740 / 2530.9245 | 2587.7430 → 2545.8117，-1.62% |
| cast + scoring | 3327.5175 / 3276.9240 / 3303.2150 | contiguous 后转型 3378.7455 / 3342.8885 / 3415.9350 | 3302.5522 → 3379.1897，慢 2.32% |
| cast + scoring | 同上 | 整批转型后切片 3448.1440 / 3428.4870 / 3458.1620 | 3302.5522 → 3444.9310，慢 4.31% |

保留 Metal 贪心 gather：局部收益可重复、真实输出一致，生产改动仅一个私有分支。
完整 scoring 探针的改善约 1.6%，不是整链改善。其他两个转型候选仅留在离线探针
用于复现拒绝结果，生产路径仍为原来的切片后分别转型。

原始数据 `scoring-local-{1,2,3}.txt`、`scoring-local-activity.json`、
`scoring-local-summary.json`；启动整链 ABBA 时检查到外部编译，未开始运行。
本轮没有可信整链提速结论，不把微秒差乘以步数当作真实整链测量。

## 实现与正确性边界

只在 Metal、贪心分支用 gather 替代第二次 max。
CPU 路径不变；随机采样继续取完整分布的最大 confidence，不能读取采样 token 分数。
返回 token 仍是 I64，confidence 仍是 F32。

新增测试覆盖 1024/1025/4097 词表、重复最大值、不同 batch/codebook 行、被屏蔽的
最高原始 logit、guidance 0/2；与原 max 路径比较全部 token 与 confidence 位模式。
范围是有效有限 logits 经过现有 log_softmax/mask 的正常推理；不新增无效 NaN 输入契约。

保留依据必须是局部实测节约、输出一致和很小的维护成本；不能把选择算子的百分比
套到整个 scoring，更不能套到整链。测不到可信整链变化时明确写未证明。

## 真模型输出验证

复用上轮冻结的原生 GQA 产物作为本轮基线；先逐文件核对上轮摘要的 SHA256，
并核对对应二进制与 `gqa-manifest.json` 相同，再运行本轮候选。
配置沿用相同本地 0.6b 权重 revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4`、narrator、seed 42、32 步、
guidance 2、class_temperature 0。资源不下载、不转换。

| 输入 | codes dump SHA256（基线=候选） | WAV |
| --- | --- | --- |
| 短文本，5.4 s | `d1f037724f6a674505e3e0f52c5ae23d3c341d95cefb6c7ad74d7a7015d829c3` | 4/4 相同 |
| 长文本，51.12 s | `69dd8d8476dcb38aef411a588461245ae39026dc9159c4d6af63d5bdbb42b29c` | 24/24 相同 |

codes 包含参考准备；WAV 包含整段和每个分段的 F32/PCM16，不仅比较量化后的最终音频。
这组运行启用 codes dump，只用于输出验证，不用它的耗时声称提速。

二进制 SHA256：

- 基线 `probe-gqa`：`f0b24630a0cbbe49fbfea8cddc04af62288c29adedf968f143356ea8bacad47b`。
- 候选 `probe-score`：`6a0b4bb94d3afdc68f41ddde781d21ca014eb96bdc902376309d2b3be7834ace`。

两者均为 release，`--no-default-features --features omnivoice-metal,performance-tools`。
没有重新测官方整链，不据此更新官方相对速度。

## 工程验证与后续

- 格式检查、`git diff --check` 通过。
- Metal crate lib 测试 30 passed；无默认 feature 的 CPU lib 测试 22 passed。
- Metal + Accelerate，Clippy all-targets / `-D warnings` 通过。
- Metal rustdoc，library + examples / `-D warnings` 通过。
- `score-manifest.json` 保存代码、二进制、脚本和输出摘要的 SHA256。

下一候选回到 decoder：`stage1_model.rs::Snake1d::forward` 每次重新计算
`(alpha + 1e-9).recip()`，可先测缓存不变量是否有可见局部收益，再决定是否实现。
尚未测量该候选，不将其列为已完成能力；不因结构上重复就承诺整链提速。
没有新增运行时实验开关，没有 push、合并、发包或部署。

## 复现入口

```sh
cargo build --release --locked -p talechime-omnivoice --features metal --example scoring_probe
target/release/examples/scoring_probe
cargo test --locked -p talechime-omnivoice --features metal --lib greedy_confidence_gather_matches_full_reduction
cargo build --release --locked -p talechime-backends --no-default-features --features omnivoice-metal,performance-tools --example performance_probe
```

模型和音频不提交；原始探针、整链日志、输出摘要保存在忽略目录
`target/perf/omni-followup/`。普通测试不下载模型。
