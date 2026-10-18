# candle-performance 的 Qwen CUDA 试用

## 范围与结论

按用户确认的“通用 Candle、以 TTS 为案例、默认保持同后端输出”定位，使用新 skill
试跑一次 Qwen CUDA 调研。源码固定点 `94b4054ea6a1141b39066fcb5ce9e1ae51b6c66d`。
本轮是作者执行的源码与产物复核，不是独立 agent 前向测试；未重新运行模型、GPU
基准或构建，也未实现图回放、改推理默认路径、修改系统配置或创建上游 PR。

结论：现有证据仍支持优先验证完整 CodePredictor 子图，**不证明它可以成功 capture
或达到任何提速比例**。一次 cat 打包的拒绝结论保留，语义采样不作为这轮优先改写点。
skill 的共享测量规则能正确导向这些决定，但 CUDA 参考需要补充主机状态、实际缓存
变体与输出 dtype 的预检，已按下文观察修订。

## 原始证据复核

读取 `target/tmp/voxcpm-opt/qwen-cuda/screen-*/rounds.json`，重新计算四进程 ABBA 的
每进程均值及组均值；每进程一轮预热、两轮热测。结果与已有 summary 一致：

| 指标 | 基线 / s | 打包候选 / s | 改善 |
| --- | ---: | ---: | ---: |
| 完整合成 | 3.0967242 | 3.1288647 | -1.0379% |
| 流式合成 | 3.6744987 | 3.6574397 | 0.4643% |

对 12 份 codes、12 份完整原始 PCM、12 份流式原始 PCM 和 24 份 WAV 共 60 个产物
重新计算 SHA-256，各类型内部均与原 summary 一致；逐轮有限值/EOS 标记与两路径
59 帧记录一致。这是已有产物的复核，不是新推理、内容质量验收或对所有模型的证明。
机器可读复核在忽略目录 `target/tmp/candle-skill-trial/qwen-reanalysis.json`。

此前 trace 的 CodePredictor 约 34.256 ms/帧、Talker 约 13.168 ms/帧来自主机包含式
span，不能相加嵌套项或宣称 kernel 占比。当前复核不重跑 trace，不以此断言瓶颈已
确定为 CPU 分派。完整计时、硬件、权重、参数与精度边界沿用
[Qwen CUDA 原报告](../qwen/qwen-cuda-performance-2026-10-18.md)。

## 源码追踪：图候选的成立条件

| 观察 | 当前证据 | 对原型设计的影响 |
| --- | --- | --- |
| 15 个声学输出不等于 15 次单 token 前向 | `code_predictor.rs::generate_acoustic_codes`：2-token prefill 预测首码，再做 14 次单步前向/argmax | 当前 16-codebook 配置的有效 KV 长度终点为 16；容量 17 只是上限，不能从注释推出节点次数 |
| CP 用 Standard RoPE | `CodePredictor::new` 创建 `RoPEType::Standard`；表在加载时预建 | 不把 Talker MRoPE 的动态主机 positions 上传误归到 CP，仍须实测底层算子的参数上传/capture 支持 |
| 声学选择留在设备 | argmax 与后续 embedding 使用设备 tensor，函数内没有读取 token 值的主机分支 | 具备候选边界条件；Talker、语义采样、EOS、PCM 留在图外 |
| 缓存可能回退 | `new_kv_caches` 在预分配失败时回退 Concat；`PreAllocKVCache` CUDA 更新通过 `InplaceOp2 + copy2d` | 若原型依赖固定预分配地址，需核实所有层实际缓存类型；不声称所有 Concat 图都不可捕获 |
| KV 有主机元数据 | `current_len` 在 update 中递增，reset 仅将其置零；回放不会重新执行 Rust 代码 | 证明每帧固定节点覆盖所有被读取的有效前缀，并隔离缓存；不能只记录一次就认为 reset 自动执行 |
| Vox recorder 不支持整数 | `cuda_graph/replay.rs::device_slice` 只匹配 F16/BF16/F32；Qwen 返回 `[15] U32` | 不能直接搬用现有 recorder；扩展前补整数复制、view 偏移及重复回放对照，不用浮点转换绕过接口限制 |
| 两种生成入口各有 owner | 流式 `StreamingSession.cp_kv_caches`，完整生成使用本次调用的 CP 缓存 | 原型状态要同时覆盖完整/流式入口，各自拥有图与 KV；不能放在共享模型的全局可变缓存中 |

以上为当前实现的静态观察，**不是 capture 原型已验证**。固定节点序列内不同步可以
有不同 KV view 长度；不必先改成填充到最大长度的 attention，这会引入不必要的
归约/舍入变化，与默认保持输出的目标冲突。

## 建议的最小下一实验

先做不接产品默认路径的显式本地真模型原型，仅覆盖单帧 CP：固定模型/dtype/device、
talker_hidden 与 semantic_embed 两份设备输入、该原型独有的 KV，捕获完整
prefill+14 步，输出为 U32。
保留 eager 对照，不叠加已拒绝的 cat 打包、GQA 布局或 BLAS/精度变化。

先验证不同隐藏态连续调用、输入覆盖、scratch 消费、缓存/图析构与失败后的安全退出。
捕获成功不算正确性通过；必须与 eager 的所有声学 codes 严格相同。必要时追踪第一个
logit/隐藏态分歧，而不是只看最终 argmax 恰好相同。然后验证完整/流式 codes、PCM、
EOS、短尾块、取消和同模型复用，最后才进入无 profiling 的独立进程性能比较。

只有原型通过后再决定如何复用 recorder：第二个真实使用者成立时提取公共设备机制，
模型状态和策略仍私有。当前公开 API 看起来有实现路径，未证明需要 Candle 新 API，
因此不先改 fork 或触发暂停发布策略。此处是实验设计，不把性能试用扩展成自动实施。

## 对 skill 的实际修订

- 入口增加“先核对并复算可复用证据”的判断，避免不必要重跑或用旧计时代表新代码。
- CUDA 参考增加主机控制流、实际缓存类型、host/device 状态、整数输出和特化身份检查。
- 明确固定循环可包含不同 shape 的固定节点，固定容量与有效 KV 长度不是一回事。
- 案例索引链接本次试用，保留其静态调研与历史产物复核的证据边界。

未新增通用基准框架、硬编码平台配置或运行时开关；Metal/AMX 规则本轮未作新实机验收。

## 后续：候选已实施并通过

本试用提出的「完整 CodePredictor 子图」最小实验已在
[Qwen CP 帧图](../qwen/qwen-cp-graph-2026-10-18.md) 实现并测量：ABBA 八进程完整
合成 -38.0%、流式 -33.3%，区间不重叠，codes/PCM/WAV 与 eager 逐字节一致；CUDA
默认开启，`QWEN_CP_GRAPH=0` 关闭，profiling 强制 eager。

试用时列为「未证明」的两点现已明确：capture 可行且 codes 逐位一致；缓存变体在
该模型为预分配路径，而 Concat 变体下的 capture 已实测**失败**——其 K/V 在 capture
内分配并被缓存保留，构成未配对分配节点，驱动在 `end_capture`/实例化即以
`CUDA_ERROR_INVALID_VALUE` 拒绝（RAII 恢复流）。因此“不声称所有 Concat 图不可
捕获”应改为“Concat 保留 capture 内分配时不可用；图在 capture 之后读写的一切都
必须预先分配”。回放不重跑 Rust 簿记成立的依据是“每次回放重写全部 16 个 KV
位置”的不变式，已写入验证阶梯；ignored 真 GPU 测试须 `--test-threads=1`
（同进程共享同一条流）。
