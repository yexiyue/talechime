# 案例与证据索引

以下是 Talechime 的带边界记录；首表为截至2026-10-23的历史案例，后续Windows
更新见下节，不是跨设备/跨项目的预期收益。
skill 的操作规则自包含；这里的仓库相对链接在移出 Talechime 后可能不可用，届时不
依赖这些文件继续执行通用流程，也不把缺少原始产物记作重新验证通过。

| 案例 | 能支持的判断 | 不能推出的结论 |
| --- | --- | --- |
| VoxCPM AudioVAE 图 | 八进程比较 CFM-only RTF 0.29465→加 decoder 0.21639，264/264 WAV 相同 | 任意模型套图都快 26.56%；桌面进程驻留时 GPU 完全独占 |
| Vox LocalEncoder 图 | 新基线约 7.52%，输出相同，但未过该轮 10% 门槛，撤回 | 该模块永远不值得优化，或其他任务也必须要求 10% |
| Vox grouped GQA/K view | 小收益或变慢，PCM/部分 EOS 改变，未保留 | 仅凭算子容差就认定生成质量等价 |
| Vox Metal skinny-M GEMM tile | LocDiT `m=22` 从 64 行 tile 改到 32 行 tile：CFM 21.71→18.99 ms/帧（-12.5%）、端到端 RTF -6.9%、WAV 33/33 与 11/11 逐字节一致、未改动模块 ±0.3%；判据是 `ceil(m/32)==ceil(m/64)` 的区间而非整数阈值 | 任意小 M 都该换 tile；10% 门槛可跨案例套用（此例低于门槛，经用户明确授权保留例外） |
| Vox macOS A/B 工具自检 | 不平衡的 2 进程序列会把 +6.9% 显示成 -5.5%；8 进程 ABBA 后段两臂同时漂移 | 未自检的筛选工具，其"无收益"结论不足以撤回候选 |
| Vox 静态 KV 原型 | primitive 通过，Q8 BaseLM 严格对照失败 | 已确认 Candle bug、已有可用图级 LM 或应立即提 PR |
| Qwen Metal depthwise | 特定 causal depthwise 条件下，窗口 batched matmul 避开逐组分派；本机输出相同 | CUDA 同样收益，或任意 groups/stride/dilation 可替换 |
| Qwen CUDA code 打包 | 四进程初筛完整慢 1.04%、流式快 0.46%，输出相同，撤回；说明“输出相同”与“有收益”是两个独立判据 | 打包候选本身等于帧图候选或已取得可靠 CUDA 提速（帧图是独立候选，见下行） |
| Qwen CUDA CP 帧图 | ABBA 八进程完整 3.0840→1.9113 s（-38.0%）、流式 3.6529→2.4353 s（-33.3%）、首 PCM 0.7688→0.5663 s（-26.3%），区间不重叠，60 份 codes/PCM/WAV 与 eager 逐字节一致；dispatch-bound 区域（每帧约 34 ms 主机墙钟的小 kernel 循环）成图收益最大 | 其他模型的帧循环都有同样收益；KV 变体、状态与 dtype 前提可从本例照搬；「平衡」电源计划下测得的数字适用于高性能计划 |
| Qwen CUDA 流式历史解码图 | CP 图两侧开启，ABBA 八进程各 1 预热 + 5 热轮，流式 2.4240→2.1140 s（耗时减少 12.8%），240 份产物逐字节一致；无内部历史状态的子图可以模型级有界缓存，但要串行覆盖输入/回放/复制输出，并显式保活常量 | 带卷积/KV 状态的解码器也能跨请求共享图；局部回放省 60% 等于端到端省 60%；不同实验的 CP 与解码提速能直接相加 |
| Qwen 1.7B Base CUDA | 换模型后重新校验权重、编码 2048 维参考（0.6B 为 1024），独立 12 进程三配置：完整耗时 −34.3%、流式 −39.1%，360 份产物逐字节一致 | 同一优化代码路径自动证明另一模型已验收；直接沿用低规格模型 prompt；Base 结果适用于 CustomVoice/VoiceDesign |
| CosyVoice3 Accelerate | M4 Pro CPU 样例大幅改善；BLAS/Rayon 组合存在争用，波形非逐位一致 | 已证明全部算子用了 AMX，或数值/试听普遍等价 |
| VoxCPM AudioVAE Accelerate 回放 | 真实 Metal patch 分离捕获、独立 target 重放；CPU F32 较普通路径少 37.59%，但含 handoff/PCM 就绪的两进程范围与 Metal F16 重叠，停止于初筛，未验证与其余 GPU 工作重叠的流水线 | 统一内存免除交接、Accelerate CPU 已带来混合/流水线端到端收益、或某个 GEMM 已确认使用 AMX |
| VoxCPM 官方 Python/MPS | 固定官方源码在 MPS 自动改为 F32，`optimize()` 为 CUDA-only 并回退 eager；Candle Q8 对原始 Safetensors F16 的八槽 ABBA 均值观察慢 0.60%、范围重叠且配对翻转，未确认收益 | 官方模型或 Python MPS 天然比 Candle 快；跨 dtype、参考缓存边界和输出长度的 RTF 可归因给语言运行时 |
| IndexTTS depthwise/FIR | 上游 PR 的局部借鉴及大形状算法对照有效，CUDA oracle 有精度配置边界 | 小输入通过即可覆盖长音频，或所有误差都由 TF32 导致 |
| MOSS macOS 基线 | 小模型（Nano，F32 加载）在 M4 Pro 上 Metal RTF 0.208 / CPU 0.308（Accelerate 默认开）、ONNX ORT CPU 单列 0.174、Realtime 1.7B F16 0.854，全部快于实时——「测量后无改动」是合法完成形态；24 GiB 统一内存下 Local 1.7B F16+codec **加载即 OOM**（kIOGPUCommandBufferCallbackErrorOutOfMemory），与 2026-10-08 记录一致 | ONNX 与 Candle 构成同输出提速结论（两者音频时长不同）；Realtime 无热点剖析就断言无优化空间（余量仅 +17%）；其他内存配置同样 OOM |
| CosyVoice3 官方 parity 对照 | 同语料同音色同模式下官方 CPU 流式 RTF 1.52（1.31–1.81）vs 我们 Metal 1.414：**端到端 parity 与组件级大差距（Flow/DiT 慢 4.5–5.6×）可以并存**——设备优势掩盖实现劣势，两个维度必须分开报告；官方无 MPS 路径（硬编码 cuda-else-cpu），官方在 Mac 上流式同样 >1 | 官方 CPU 数当作官方 GPU 能力；跨框架输出音频长度不同时比较绝对耗时；把 parity 均值差 ~7% 报成「超过」（区间重叠不构成超过）。坑：官方 zero-shot 转写必须自带 `<\|endofprompt\|>` 后缀，否则 LLM 线程断言退出、主生成器在轮询睡眠里假死（0% CPU + time.sleep 栈） |
| IndexTTS FIR 融合 conv1d | 共享核单通道 conv1d（`(B·C,1,L)`×`(1,1,K≤12)`）在 Metal 的 im2col+小 GEMM 路径上开销全在搬运：fork 内加每线程一输出的融合核后 FIR 段 6.8×/3.5×，端到端 ABBA -19.4%；**纯 candle 算子重写（逐 tap）与 conv 同速**——先探针证明两条现有路径都在地板上，再决定动 fork；behavior-only 无 API 变更 | 「conv1d 慢=该换逐 tap」；模型级 WAV 对比当等价判据（IndexTTS Metal 采样跨进程本不确定，基线自噪声 > 候选差；等价证据用固定输入探针 + CPU 参照）；fork 改动不加登记（FORK.md 必须登记且优先 behavior-only） |
| IndexTTS DiT rope_i+sdpa | 广播式 RoPE 的 stride-2 视图让全部元素级 kernel 落入非连续慢路径（~2% 带宽，占 block 52%）；**candle-nn 现成的 `rotary_emb::rope_i`（融合 Metal kernel）与 `ops::sdpa` 就在 upstream 公开 API 里**——上游有能力时先换调用、后动 fork（本轮零 fork 改动 ABBA -15.3%）；GEMM 归因健康（3.5–4 TFLOPS）后别急着怀疑 matmul | 「慢=必须写 kernel」：rope_i/sdpa/gemv 等融合 op 可能已存在，先 grep candle-nn 的 CustomOp 实现再立项；(B,T,H) 先 transpose 再 reshape 冒充 (B,H,T,D)（无 d 维可分，CPU 也错，差 100+ 量级）；插桩 scope 守卫不 drop 会把嵌套段重复计入（首测 dit.qkv "71%" 是假的）；run 级逐算子 span 有 ~36% 同步放大，归因以独立形状探针为准 |
| IndexTTS 宿主 custom-op 四内核 | **candle Metal `conv_transpose1d` 的 col2im 快路径仅 padding==0**——带 padding 的 ConvT 全部落入 naive per-output 标量核（L0/L1 仅 0.015 TFLOPS，六级 879ms），k%s==0 且 s 为 2 幂时 ConvT=健康 GEMM`(l,c_in)@(c_in,c_out·k)`+gather 散射核（每输出恰 k/s 次读）→ 14.3ms（~125×，rel≤2.6e-6）；多通道 conv1d 的 im2col+GEMM 编排换直出布局仅 −5~10%——**im2col 物化本身是地板**（k× 写放大+每 call 新 buffer），strided-B/转置回拷不是瓶颈；pad 物化+双相位 FIR conv+snake 五算子链折单内核 ~8×（rel≤4.3e-7，参数加载期打包）；四内核合并 ABAB −25.35% 区间不重叠；零 fork（custom-op 宿主 `CustomOp{1,3}::metal_fwd` + `call_im2col1d_strided` 等 candle-metal-kernels 公开原语直接编排） | 「conv 慢=GEMM 编排烂」——先探针拆 conv vs 等形纯 GEMM 再动手（本轮 GEMM-direct 几乎白做）；越界写**不崩溃而是数值污染**（grid div_ceil 后无 guard，196 线程踩相邻 buffer，对拍 rel 0.85，CPU 标量模拟全对才定位到 Metal 侧）；Metal 内核三元 `cond ? 0 : size_t` 类型不匹配直接编译失败（先算 index 再取指针）；官方同日自身漂移 ~27%（惰性编译深浅），跨臂判定必须当日 O-T-O-T 窗口交错且看配对方向，单窗口/隔日结论皆不可用 |
| IndexTTS wavenet 续轮 | **estimator 插桩各段合计 ≠ 总耗时应先找未插桩模块**（WaveNet 终层 2.29s 此前零归因，补 scope 一举定位）；门控激活 `tanh·sigmoid(+bias)` 与 res/skip 尾双平面折单内核均位级（表达式逐字取 candle：`precise::tanh`、`1/(1+exp(-x))`；`recip` 在 MSL 内核不可用）、wn.act 4.3×、estimator 5.07→4.39s；**candle 批 GEMM 在 rhs 广播（batch_stride 0）时把 batch 折进 M**——自建"直出布局"把广播换到 lhs 侧丢了 collapse，B=2 实测反退 22% 已回退；dual-plane 内核的平面 1 偏移是 `b_len·H·T+plane`（写 `plane+H·T` 在 B=2 越界踩平面 0，位级门当场拦截） | 自建数值参照必须复刻真实调用的全部参数（bias/dilation/padding）——首版 convT/conv 融合漏 bias，门用 `bias=None` 参照互拍"错对错"全绿（性能结论 −25% 复验成立、数值结论作废重测）；B≥2 的 token-major 批别用去广播 lhs 的直出布局 |
| CosyVoice3 DiT rope_i+sdpa | 同一招移植前先核对旋转语义：cv3 是 **partial RoPE**（`RotaryEmbedding(dim_head)` 只旋转扁平 (B,T,1024) 的前 64 通道），rope_i 只能对 64 宽切片以 (B,1,T,64)+(T,32) 唯一频率表应用——「交错配对=可全维迁移」是误读，先看构造处 dim 再定迁移形态。物化注意力（33.5MB scores + mask/softmax 各一遍读写）占单 block 50%，sdpa -87%；mask 需每 solve 展开一次 [B,H,T,T]（内核拒收可广播形状）；端到端 ABBA -12.5%，组件节省 ×占比与端到端可对账。流水线级归因用内建 `CV3_STREAM_PROFILE=1`（串行流水线墙钟五桶，零开销门控）：Flow 56%/HiFT 26%/LLM 18%，逐段条件开销仅 ~1.5%——先排除"逐段零样本"这类直觉病灶再立项 | 探针输入 σ 校准到 post-scale scores O(1)：σ=1 时 scores σ≈10³，ULP 求和序翻转近并列 softmax 项，1e-4 量级"失配"是量级伪影非语义错误（σ=1/32 后同门 1e-6 通过）；相对门的自然舍入底随 softmax 分母项数增长（512 项 ~2e-6），门限设在语义错误（≥1e-2）与舍入底之间并写明理由；引用"过期数字"前先全文检索既有记录——accelerate 复测早已在案（1.19×），差点重复立项 |

## Windows 当前实证（更新至2026-10-25）

以下补充均链接本轮设备、条件和未完成项；不改写上表历史数据或Mac结论。

- [Windows探针导出](../../../docs/records/platform/windows-artifact-export-2026-10-24.md)：WAV未缓冲逐样本写入约2.9s/格式，64KiB缓冲约4ms；36组完整字节与旧工件一致，write/flush失败仍拒绝。只计导出收益，模型A/B需统一导出策略。

- [OmniVoice CUDA](../../../docs/records/omnivoice/omnivoice-windows-cuda-2026-10-24.md)：GQA、请求所有的前向图和保序F32 deconv的局部实证；完整codes/PCM与局部/整链边界分开，官方质量仍有未完成项。
- [MOSS Nano Candle](../../../docs/records/moss/moss-windows-performance-2026-10-24.md)：阶段归因后复用已算embedding，有序相加、1026组逐位oracle及66轮PCM回归；按局部收益保留，稳定整链收益未证明。含有效seed/双声道布局、真实EOS与满队列取消回归；没有阶段证据的RoPE原型继续撤回。
- [Nano forward交错RoPE](../../../docs/records/moss/moss-nano-windows-forward-rope-2026-10-25.md)：从1.7B复用经验重新验证布局/频率/层数，180组oracle、三seed codes/PCM/EOS和旧消费者回归后，短句/段落ABBA约−19.45%/−14.54%，96轮PCM保持；段落首PCM仍inconclusive。区别于此前扩大表长/矩阵形状的失败方案，不相加历史收益。
- [IndexTTS Windows](../../../docs/records/indextts/indextts-windows-production-2026-10-24.md)：参考复用约−63.72%；SF0.8/pread隔离环境恢复官方GPT BF16正式GPU基线，真实EOS正负验证、匹配内容回读与同步阶段归因。Native F32/官方混合精度未强行排名，新F32失败仍单列。
- [CosyVoice3生产资源门禁](../../../docs/records/cosyvoice3/cosyvoice3-windows-production-2026-10-24.md)：HiFT SHA仍不匹配，三种CPU dispatch重转换未解决；不绕过门禁或宣称生产已验收。
- [MOSS Local/Realtime 1.7B](../../../docs/records/moss/moss17-windows-production-2026-10-24.md)：各3×(1+5)生产基线、官方加载/系统提交内存调查；不将非推荐兼容栈当公平官方性能基线。
- [MOSS forward内RoPE复用](../../../docs/records/moss/moss-windows-forward-rope-reuse-2026-10-25.md)：保留原形状、单forward共享表；300组逐位oracle、三个模型9组codes/PCM及owner门禁后，Realtime ABBAABBA约−24.59%，Local整链仍inconclusive。Source稳定不证明Cargo缓存对应源码，旧mtime复制导致相同二进制的伪A已拒绝；比较器off误判修复，原始条件不重标记。
- [MOSS ONNX RTX5070](../../../docs/records/moss/moss-nano-onnx-windows-2026-10-24.md)：DirectML cache/host约−21%；另一完整ORT1.28分发使专用动态构建的Native Registry完成18轮，均值0.694s，状态复用通过。同版本分发/实际模块/动态环境身份须分别验证；官方1.159s条件不同，未强行排名或宣称默认发行配置已恢复。
- [MOSS Nano ONNX](../../../docs/records/moss/moss-nano-onnx-windows-2026-10-24.md)：本机两份CUDA provider缺少sm120/PTX；DLL静态覆盖、真实执行和fallback分开，不升级为所有ORT分发的结论。
- [Qwen Windows](../../../docs/records/qwen/qwen-windows-production-2026-10-24.md)：CustomVoice双方基线、1.7B默认图组合约−49%及48轮PCM一致；EOS正反例、pread加载对照、共享prompt及prefill数值定位，不用长度差直接排名。
- [VoxCPM Windows当前生产回归](../../../docs/records/voxcpm/voxcpm-windows-production-2026-10-24.md)：Q8图组合约−51%、48轮PCM一致，原始BF16另建基线及Job commit实测；漏绑磁盘缓存后新Job复测。固定C++适配器排除WAV写盘，短文本Candle/C++ RTF0.246/0.289，输出长度与RNG不同，质量等价pending。官方初始化失败、启动器RSS漏采及工程门禁另列。
- [VoxCPM Windows官方控制版正式对照](../../../docs/records/voxcpm/voxcpm-windows-official-baseline-2026-10-25.md)：保留12GB门禁恢复3×(1+5)，对齐19patch与真实边界处理；区分WAV时间戳与PCM摘要，先做默认/同步eager逐位门禁再阶段归因。既有BF16 CFM图单变量补验约−33.08%、48轮PCM保持，原始F32 VAE成为下一热点；默认加载与跨框架质量未关闭。
- [VoxCPM Windows F32 decoder图](../../../docs/records/voxcpm/voxcpm-windows-f32-decoder-graph-2026-10-25.md)：将官方对照发现的热点落实为另一dtype的真实状态/模型/owner门禁后扩大Windows BF16默认值；短句/段落21.33%/26.68%、96轮PCM一致，Qwen/Vox六模型组合回归通过。复核开关依赖链，防止默认变化后仍把关闭两张图误称CFM单变量。

## 仓库记录

- [OmniVoice Metal 经验](omnivoice-metal.md)：原生 GQA、序列优先 RoPE、卷积快路径与失败候选；区分已启用、撤回和仅离线验证。
- [OmniVoice CPU 经验](omnivoice-cpu.md)：同真实张量归因、GQA/权重布局、受限 NCL 卷积及 MLP 连续化失败；局部收益与整链增量分开判断。
- [OmniVoice 同期官方对照](../../../docs/records/omnivoice/omnivoice-official-comparison-audit-2026-10-23.md)：免拷贝 Rust 2.672 s、官方 MPS 3.235 s，仅限该短文本 F16 生成路径，不证明跨框架输出或音质等价。
- [CosyVoice3 Windows](../../../docs/records/cosyvoice3/cosyvoice3-windows-performance-2026-10-19.md)：Ryzen 9700X CPU / RTX 5070 CUDA；逐 solve 的 DiT 图局部快但端到端不足 10%，MLP 图及组合也不足，全部撤回；CUDA 转置 view 更慢且改变 PCM；保留跨平台基准与 recorder 修复，不宣称已取得整链提速。
- [通用方法](../../../docs/knowledge/tts-performance.md)：架构与验证策略。
- [VoxCPM 扩展](../../../docs/records/voxcpm/voxcpm-review-extension-2026-10-18.md)：图生命周期、四项真模型测试、接受/拒绝实验。
- [VoxCPM 官方 MPS](../../../docs/records/voxcpm/voxcpm-official-mps-validation-2026-10-19.md)：官方源码、原始 Safetensors、实际 dtype 与可比性边界。
- [Qwen Metal](../../../docs/records/qwen/qwen-performance-optimization-2026-10-18.md)：队列等待、采样、depthwise 及流式状态。
- [Qwen CUDA](../../../docs/records/qwen/qwen-cuda-performance-2026-10-18.md)：固定权重、无插桩初筛和拒绝的打包候选。
- [本 skill 的 Qwen 试用](../../../docs/records/platform/candle-performance-skill-trial-2026-10-18.md)：原始证据复核、capture 前提和由试用发现的参考缺口；不是图已通过的证明。
- [Qwen CP 帧图](../../../docs/records/qwen/qwen-cp-graph-2026-10-18.md)：整帧自回归循环成图、主机簿记不变式、失败策略分档、共享 recorder 提取与复验边界。
- [Qwen CUDA 图分区评估](../../../docs/records/qwen/qwen-cuda-graph-partition-2026-10-18.md)：历史探索数据与后续口径修正；固定容量不等于相同有效 KV 长度。
- [Qwen 解码图与审查](../../../docs/records/qwen/qwen-decoder-graph-2026-10-18.md)：有状态/无内部状态图的生命周期边界、eager oracle 必须独立、表缓存按字节约束、初次录制与多请求稳态计时。
- [Qwen 1.7B Base CUDA 独立验收](../../../docs/records/qwen/qwen17-cuda-acceptance-2026-10-18.md)：另一模型规格的资源与 prompt 身份、eager/单图/双图三配置配对验收；模型统一存储与经哈希验证的重复权重去重。
- [CosyVoice3](../../../docs/records/cosyvoice3/cosyvoice3-performance-2026-10-17.md) §11：Accelerate 线程矩阵、数值和试听边界。
- [IndexTTS](../../../docs/records/indextts/indextts-candle-performance-2026-10-17.md) §10–12：depthwise PR、FIR 大形状、严格 F32 与默认配置的区别。
- [MOSS macOS 基线](../../../docs/records/moss/moss-macos-performance-baseline-2026-10-24.md)：M4 Pro 全产品档位 3×6 协议首份正式基线；含 harness job 冻结备忘（cpu provider 的 hardware/driver 格式、generation.* 默认参数须进 conditions.parameters、baseline 模式 trace_complete 恒 false 属设计）。
- [CosyVoice3 官方 parity 对照](../../../docs/records/cosyvoice3/cosyvoice3-official-parity-2026-10-24.md)：官方流式 RTF 首测（同语料同音色）；endofprompt 假死根因；parity 与组件差距并存的判定方法。
- [IndexTTS DiT rope_i+sdpa](../../../docs/records/indextts/indextts-metal-rope-sdpa-2026-10-24.md)：逐算子归因（GEMM 健康、RoPE 广播链是病灶）、head-major reshape 谬误、插桩嵌套重复计数教训与 ABBA -15.3%。
- [IndexTTS 宿主 custom-op 四内核](../../../docs/records/indextts/indextts-metal-hosted-kernels-2026-10-24.md)：convT padding≠0 落 naive 核的归因与 GEMM+gather 分解（~125×）、im2col 物化地板证伪 GEMM 编排优化、fir_up+snake 单内核、O-T-O-T 交错判定 parity（官方同日漂移 ~27%）。
- [CosyVoice3 DiT rope_i+sdpa](../../../docs/records/cosyvoice3/cosyvoice3-metal-attn-rope-sdpa-2026-10-24.md)：partial RoPE 语义核对、探针量级校准（softmax scores O(1)）、sdpa mask 一次性展开、ABBA -12.5% 与组件-端到端对账。
- [candle fork 转置行块内核](../../../docs/records/cosyvoice3/cosyvoice3-metal-transpose-copy-2026-10-24.md)：转置(1,2)视图 contiguous 的 MultipleBlocks 慢路径（11.6GB/s→113GB/s，9.7×，ABBA -11.7%）；位级门三秒抓偏移漏加 bug 的"值=下标"微型用例法；**Metal 分段计时"先同步再停表"**（异步队列下 istft 假 578ms/真 16ms）；HiFT 真病灶=f0 CPU f64 卷积 50%+resblocks 29%（"卷积轮"前提被探针证伪）。
- [cv3 融合调制 LayerNorm](../../../docs/records/cosyvoice3/cosyvoice3-metal-fused-adaln-2026-10-24.md)：release 单步六区在位探针（派发缝隙 0）定位手写 7-pass LN ×2/block；candle 融合 LN 的 alpha/beta 是 1-D，逐 batch 调制按分支折叠调用；一遍式 vs 两遍式方差 ~2e-7 有门；**ABBA 被机身热态 ±25% 漂移否决两轮 → 改 ABABABAB 严格交替（每对共享热态）后 -18% 区间不重叠**；机身双态（持续负载 1.165/冷态 0.982）是绝对 RTF 数的隐藏变量 |
- [candle fork 广播二元快路径](../../../docs/records/cosyvoice3/cosyvoice3-metal-broadcast-binary-2026-10-24.md)：broadcast_mul 的 rhs 经 broadcast_as 变 stride-0 视图后落 `_rstrided`（每元素多维 div/mod），微基准 25×（0.4ms vs ~10ms）；调度条件必须接受**窄视图 stride（src_stride 是 D 的倍数而非恰为 D）**——首版严格条件漏掉 AdaLN narrow 视图；端到端仅 -4.5% <10% 门槛——**微基准 25× ≠ 端到端 25%，占比决定回收率，子阈值改动与下一轮合并验收** |

- [cv3 DiT 融合内核第六轮：custom-op 宿主 + embed 直卷证伪](../../../docs/records/cosyvoice3/cosyvoice3-fused-dit-customops-2026-10-24.md)：`CustomOp3::metal_fwd` 公开面把三个融合内核全放应用侧（rope+头拆分位级一致 2.9×、门控残差+modLN 双平面、块尾门控加），零 fork 派发；**输出缓冲必须 `Output::new` 绑定**（否则 hazard 追踪旁路、回读全零不报错）、**运行期非 2 幂除法分解 tid 是 4× 性能陷阱**（改 3-D grid）；embed 直卷两版均慢于 im2col 3–4×——"N MB 物化浪费"推算被 L2 常驻+GEMM 计算受限证伪，**数值门过≠性能门过**；合并 ABAB -4.95% 区间不重叠（四对 -4.8~-5.1%）仍 <10% 线，子阈值累积待下轮（adaln 边界融合）合并验收 |

- [cv3 边界 LN 融合第七轮](../../../docs/records/cosyvoice3/cosyvoice3-boundary-ln-fusion-2026-10-24.md)：尾内核同遍产出下一块 modLN 平面（双平面输出模式的延伸——块间依赖也能折进单内核），末块 norm_out 归零；**端到端回收被模块 chunk 份额封顶**（DiT 步 −9.3% → chunk −5.11%，Flow≈90% 为 DiT、DiT≈45% chunk）；candle `Tensor::cat` 偶走转置回退路径返回非连续布局，内核输入要 `.contiguous()` 归一；**"持续负载"读数是环境变量**——同二进制同槽协议不同日差 ±15%（A 基线 1.119→0.985），判定 <1.0 需当日连跑（15×背靠背）确认无降频爬升 |

查原生实现时对应 Candle revision/版本以目标项目 Cargo.lock 为准。本仓库历史基线
为 upstream 0.11.0 `31f35b147389700ed2a178ee66a91c3cc25cc80d`，fork
`bafb03db81c3eae48e63ff28d6b30f5a6576ba38` 仅增加 FORK.md，没有功能性补丁；
查当前依赖不要仅复制这个历史 revision。
