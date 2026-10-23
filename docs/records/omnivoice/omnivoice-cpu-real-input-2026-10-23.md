# OmniVoice CPU 真实输入归因（2026-10-23）

本轮在 `23722ed` 上重新建立官方 CPU 对照，并用相同真实张量定位差距。
**不是缺少 Accelerate 的全局配置问题，也没有证据表明 CPU 已到硬件上限。**
当前主要机会在 decoder 普通卷积/布局、MLP 激活，以及 attention 执行路径。
没有发现本轮夹具范围内的生产行为回归；本轮只增加离线诊断，不改生产算法。

[逐进程原始测量组、环境、输出与张量哈希](omnivoice-cpu-real-input-2026-10-23.json)
与本文配套。原始 stdout/stderr、完整张量及音频保留在工作树
`target/perf/omni-cpu-attribution/`，不提交模型、激活数据或音频。
[探针使用说明](../../../scripts/perf/README.md) 给出构建、冻结、捕获、交错重放及口径。

## 基线与路径核验

- M4 Pro，8P+4E / 24 GB，macOS 26.6.2（25G83）。CPU F32，32 steps，默认采样和 narrator。
- Rust 基线 `23722ed59b5b5ff23e2e238c2e222d60f56c0751`；Candle 固定
  `2266099d92265112ecc0d56ecee1761e3cc2ac00`；官方固定
  `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`，torch 2.8.0 / transformers 5.3.0。
- Rust feature 为 `omnivoice,performance-tools`，无 Metal/CUDA；最终动态链接包含
  Accelerate framework，未解析符号包含 `_sgemm_` / `_dgemm_`，固定 Candle 的
  F32 matmul 使用 Accelerate 分支。离线探针也断言 `has_accelerate()`。
- PyTorch 构建为 `BLAS_INFO=accelerate`，MKL/MKLDNN OFF；实际设备 CPU、F32、
  `torch.get_num_threads()==8`，interop=1。同输入官方层 profiler 三次均记录
  `aten::_scaled_dot_product_flash_attention_for_cpu`，不是仅凭配置名推测。
- Rust/Python 进程均请求 `RAYON_NUM_THREADS=8`、`VECLIB_MAXIMUM_THREADS=8`、
  `OMP_NUM_THREADS=8`。没有测系统 BLAS 的实际内部线程数或 AMX 硬件分派，
  因此不称作“已证明 8 线程 AMX”。编译与推理串行；每 0.5 秒观察编译进程及 RSS。
- 主权重 SHA256 为 `730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa`；
  codec 权重为 `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d`。
  输入、源码、官方运行时代码、各阶段二进制及完整布局指纹见 JSON。

基线二进制为 `ca09bbb9437a509a5a87037c57114e200466c595153e6dba7460d6047245543b`。
最终同 feature 生产构建为 `d49049b174cec2f252e753346e03b475fc6c2f5331b7733e9343380bdcb76b2c`。
捕获钩子与私有重放模块均由 `cfg(test)` 隔离，正式构建没有捕获开关、额外分支或文件 I/O。
两份诊断二进制分别冻结：decoder 使用修正宿主 seed 前的版本，stage0 使用修正后的版本；
修正仅涉及捕获测试的请求 seed 和测试名称，不改变 decoder 算术。JSON 分别记录两者哈希。

## 产品整链：差距可复现，不能当作同算子比较

短文本按 Rust / 官方 / 官方 / Rust 四个独立进程运行，每进程 1 预热 + 3 热轮；
加载与 WAV 导出不计入。Rust 使用生产 session 完成边界，官方使用 `model.generate` 完成边界。

| slot | 实现 | 热轮耗时 / s | 进程均值 / s | 音频时长 / s | 平均 RTF |
| --- | --- | --- | --- | --- | --- |
| 1 | Rust | 7.051079, 7.024408, 6.994110 | 7.023199 | 5.32 | 1.320150 |
| 2 | 官方 | 6.348164, 6.217268, 6.228892 | 6.264774 | 5.40 | 1.160143 |
| 3 | 官方 | 6.219723, 6.209058, 6.193645 | 6.207475 | 5.40 | 1.149532 |
| 4 | Rust | 7.012594, 7.095879, 6.945523 | 7.017998 | 5.32 | 1.319173 |

逐进程等权平均：Rust **7.020599 s**，官方 **6.236125 s**。四个 slot 均未观察到编译活动，
无需补组；没有丢弃轮次或终止其他进程。没有采集频率、温度、功率或带宽计数器，
“未观察到编译活动”不等于系统完全空闲。

宿主的 pinned seed 42 会通过 `SeedPolicy::resolve(call, attempt)` 派生具体请求 seed；
第一段实际是 **7330414270409262323**，官方是 `torch.manual_seed(42)`。
两端 RNG、前端、后处理及输出工作量均有差别；Rust session 还执行首尾静音处理和响度处理。
因此本表只证明当前产品表现，不能把 0.784 s 全部归因给某个算子，RTF 也不能完全校正工作量。

## 相同真实输入：stage0 差距在激活和 attention，不是所有 GEMM 都慢

捕获真实短文本第 0/15/31 步、第 0/13/27 层的完整层输入、输出、mask，以及 final norm 后的
输出头输入。重放由实际私有模型实现加载真实权重，导出归一化、Q/K/V、context、MLP 激活输入等。
完整张量保存为 safetensors，原 strides/offset/dtype 另存 JSON；官方重放恢复原布局。
实际 CFG batch=2、seq=157、hidden=1024、Q heads=16、KV heads=8、head_dim=128、FFN=3072。
没有用跨框架“相同 seed”代替相同张量。

首次筛选直接给底层 Pipeline 设置 seed 42，虽然是有效真实输入，但并非宿主基线的实际请求。
校验完整 codes 时发现此差异后，保留到 `seed42-screen/`，排除其全部 stage0 计时，
修正为宿主派生 seed 并重新捕获及完成三组重放。修正后捕获的完整 code record 与宿主基线一致。
底层 Pipeline 音频与最终 session WAV 不直接比较；两者后处理边界不同。

正式局部顺序为 RP / PR / RP，三个独立进程/框架；第二组反转操作顺序。
每个 case/op 预热一次，随后五组、每组五次调用；先算各 case 的组均值，再算每个进程的
九个真实层 case 均值，最后对三个进程等权。输出头覆盖三个真实步骤，独立平均。
以下均为组件测量，所有正式进程未观察到编译活动。

| 边界 | Rust / ms | 官方 / ms | 判断 |
| --- | --- | --- | --- |
| 完整 decoder layer，含残差与两次 layer norm | 6.9277 | 6.4714 | 存在可重复的层级差距 |
| attention 完整消费者，含 QKV、norm/RoPE、context 布局与 O 投影 | 3.4804 | 3.1717 | 不应只看裸 SDPA 内核 |
| attention context 路径 | 0.7840 | 0.5074 | 官方含 repeat-KV 和 sequence-first 输出物化；完整消费者是主要匹配边界 |
| 完整 MLP | 3.2195 | 2.8014 | 需继续处理非 GEMM 部分 |
| 完整 MLP + 残差消费者 | 3.2534 | 2.8558 | 差距延续到实际消费者 |
| SiLU × up，同一真实 gate/up 值与布局 | 0.5781 | 0.1988 | 当前最清晰的 stage0 局部差距 |
| gate/up 投影：Rust 合并，官方两路 | 1.5376 | 1.5709 | 没有全局 GEMM 缺加速的迹象 |
| down 投影，同一真实激活输入 | 0.8847 | 0.8758 | 接近 |
| K 投影 | 0.2337 | 0.2955 | 当前 Rust 缓存路径有局部优势 |
| V 投影 | 0.2351 | 0.2950 | 同上 |
| Q/K 归一化与 RoPE，含各自必要布局 | 0.5791 | 0.7627 | Rust 这里更快，不能把相邻收益重复计入 |
| 输出头 Linear，完整词表 | 1.9993 | 1.9388 | 本轮不改输出头 |

完整层的三个进程均值，Rust 为 6.9166 / 6.9405 / 6.9261 ms，官方为
6.4858 / 6.4777 / 6.4508 ms。激活为 Rust 0.5791 / 0.5779 / 0.5773 ms，
官方 0.1972 / 0.2014 / 0.1977 ms；原始组值全部保留，未根据离群观感筛除。

固定 Candle 的 SiLU **也已接上 Accelerate 向量路径**：`op::Silu::f32_vec` 调用
`accelerate::vs_silu`，执行负值写入、vForce exp、除法，再单独乘 up。
合并投影拆出的 gate/up 是行间有间隔的视图；`unary_map_vec` 按连续块逐块调用。
这支持继续检查调用粒度和访存，但没有独立消融证明上述某一项独占全部激活差距，
不能写成“漏开了向量 exp”，也不能直接用改变近似/舍入的新内核替换。

当前 Rust CPU GQA 仍物化 scores、缩放、mask、softmax、PV；官方本轮同输入层实际使用
CPU Flash Attention。此前 Candle CPU Flash 探针更慢且改变 bits 的结论不被本轮推翻，
不能把不同实现都叫 Flash 然后假定开 feature 即可追平。

这些边界相互嵌套，不能相加，也不把三个抽样层的差值乘以 28 层 × 32 步当作整链收益。
本轮没有单独归因 embedding mixer；整链 forward 的旧粗段包含它，不声称已闭合完整时间账。

## 固定 codes：decoder 普通卷积值得优先处理

使用宿主基线的真实 8-codebook codes，覆盖每个 codebook 的前 1/7/130 帧，以及长文本的
11 个生成分段（排除单独识别的 48 帧参考 warmup）。两端每进程只有一份目标模型；
官方固定 codes 只加载一份 codec，Rust 只加载 decoder，未额外共存第二份归因模型。
官方 codec 仍含其正常的 encoder/semantic 权重，RSS 不能据此作等内存比较。

完整 decoder 使用 RPPR，每个 case 1 次首轮 + 5 次热轮，权重加载另列。
全部完整 decode 完成后才运行细分探针；第三个 R/P 进程补足局部三进程，不并入整链 RPPR。
`first_decode` 是该进程该形状的首轮，不代表所有 case 都是独立冷模型。

| slot | 实现 | 固定 130 帧热均值 / ms |
| --- | --- | --- |
| 1 | Rust | 443.025 |
| 2 | 官方 | 113.051 |
| 3 | 官方 | 110.081 |
| 4 | Rust | 447.393 |

两臂进程等权：**445.209 vs 111.566 ms**。所有固定 codes case 样本数一致；
首次、热轮、各长分段与局部三进程的完整数据见 JSON。未观察到编译活动。
这项比较已排除随机 codes、输出长度和 stage0 工作量差异；不包含最终 session 后处理。
Rust 边界为 `decode_tokens`，包含小型 I64 codes 到 Candle Tensor 的物化；官方先在计时外
创建 codes Tensor，再调用 `decode`。该准备差异没有单独计时，不能把完整差值全归因于卷积；
同布局真实卷积输入的独立探针提供了额外定位依据。

以下是真实 130 帧 decode 的代表性中间输入，三进程局部均值，均含 bias 与必要布局：

| 实际算子及输入 | Rust / ms | 官方 / ms |
| --- | --- | --- |
| 最后 block 的 res_unit1 1×1 conv，`[1,32,124800]` | 10.7798 | 0.7907 |
| 同 residual unit 的 7-tap conv，`[1,32,124800]` | 26.1516 | 2.9187 |
| 第一个 block 的 deconv，`[1,1024,130]` | 13.4020 | 2.0438 |

完整 block 和 residual 消费者也已测量，差距仍存在，不是只把成本移到下游。
所有生产 CPU conv1d 当前经过 im2col + GEMM + 输出布局复制，包括 kernel=1。
非零 padding/output_padding 的 CPU deconv 则错过已有 GEMM+col2im 路径，落入
逐 kernel/channel 的 dot 路径。Metal 专用快路径没有应用到 CPU。
以上来自固定 Candle `cpu_backend/mod.rs` 的 `conv1d`、`conv_transpose1d` 与
`ConvTranspose1D::f`，不把历史 Metal 收益当作新的 CPU 收益。

局部 `layout_materialize` 是显式 NCL/NLC 往返的控制实验，不是生产真实调用次数证明。
7-tap 目前仅定位，没有引入新算法。Snake 局部开销也保留，但不是这组 decoder 差距的主因。
不能把代表性单元的差值外推为全部卷积或整链提速。

## 正确性、保留与下一步

- Rust 九个真实层的完整重放逐位匹配捕获输出；拆开的 attention、MLP、残差路径也逐位匹配。
  Decoder 生产解码、trace、手动遍历完整 blocks 逐位匹配，固定 codes 的五次热调用逐位一致。
- 跨框架不要求逐位一致。九个共享输入层的最大绝对差上限为 0.00256348，RMSE 上限
  0.000112232，完整误差及输出哈希保留；这不是音频质量验收或新数值容差授权。
- 最终生产构建对照基线：短/长完整 codes 均逐字节一致；短 4 个、长 24 个分段/整段
  F32/PCM16 WAV 全部逐字节一致，样本数由完整 WAV 一致性同时验证。
  短 codes SHA256 为 `1aa283e47c13daf228482390f78d6c5ba46ef934df19248599f60bf4ac8e1096`，
  长为 `283526efd5c460545114bacf001103a66275a75343bd60aa58afcca672d0af3a`。
- 本轮保留私有、仅测试编译的捕获/重放模块及离线脚本；没有新增生产优化、依赖、公共 API、
  资源、采样修改或性能收益声明，不启用 CPU Flash。

下一轮候选顺序：

1. **CPU 1×1 conv 的同 GEMM 几何布局路径**：先验证跳过通用 im2col、保持 GEMM 形状/
   转置语义的等价物化，连同输出复制、bias、Snake/残差消费者计时；完整 bits 改变即撤回。
   不直接把 Metal 的 NCL matmul 特化搬到 CPU 并假定数值相同。
2. **MLP 激活/布局**：用本轮真实 gate/up 比较调用粒度与连续化，保持现有 vForce 算术；
   此前 synthetic 的连续化和 SiLU×up 自定义操作没有可信完整 MLP 收益，不因新发现差距就
   自动恢复旧候选。只有完整 MLP 消费者可重复获益且完整输出不变才接入。
3. **CPU 7-tap im2col/输出布局归因**：普通卷积差距大，先区分物化和 GEMM；涉及 Candle
   fork 或新算法的方案另提，不在本轮夹带实现。
4. **Attention 内核**：已有局部差距，但新分块/融合内核涉及归约顺序、数值和维护成本，
   需独立方案；不以近似、换精度或放宽现有门槛换取速度。

按绝对时间、可重复性、完整消费者和维护成本选候选，不设最低整链改善百分比，
也不把本轮的框架差距直接写成下一轮能获得的收益。

验证：CPU release 单元测试 **36 passed / 3 ignored**；Metal release **49 passed / 3 ignored**；
CPU 与 Metal `--all-targets` Clippy（`-D warnings`）、Metal rustdoc（`-D warnings`）、格式检查通过。
三个 ignored 测试仅在显式提供本地模型/输入时运行，普通测试不下载模型。
未验证长文本 stage0 层级性能、其他 batch/序列形状的性能、其他芯片/AMD 平台、温度/带宽上限，
也没有新增试听或 ASR 质量结论。仅本地交付，不推送、合并或发布。
