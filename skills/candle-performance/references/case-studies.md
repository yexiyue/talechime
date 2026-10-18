# 案例与证据索引

以下是 Talechime 截至 2026-10-18 的带边界记录，不是跨设备/跨项目的预期收益。
skill 的操作规则自包含；这里的仓库相对链接在移出 Talechime 后可能不可用，届时不
依赖这些文件继续执行通用流程，也不把缺少原始产物记作重新验证通过。

| 案例 | 能支持的判断 | 不能推出的结论 |
| --- | --- | --- |
| VoxCPM AudioVAE 图 | 八进程比较 CFM-only RTF 0.29465→加 decoder 0.21639，264/264 WAV 相同 | 任意模型套图都快 26.56%；桌面进程驻留时 GPU 完全独占 |
| Vox LocalEncoder 图 | 新基线约 7.52%，输出相同，但未过该轮 10% 门槛，撤回 | 该模块永远不值得优化，或其他任务也必须要求 10% |
| Vox grouped GQA/K view | 小收益或变慢，PCM/部分 EOS 改变，未保留 | 仅凭算子容差就认定生成质量等价 |
| Vox 静态 KV 原型 | primitive 通过，Q8 BaseLM 严格对照失败 | 已确认 Candle bug、已有可用图级 LM 或应立即提 PR |
| Qwen Metal depthwise | 特定 causal depthwise 条件下，窗口 batched matmul 避开逐组分派；本机输出相同 | CUDA 同样收益，或任意 groups/stride/dilation 可替换 |
| Qwen CUDA code 打包 | 四进程初筛完整慢 1.04%、流式快 0.46%，输出相同，撤回；说明“输出相同”与“有收益”是两个独立判据 | 打包候选本身等于帧图候选或已取得可靠 CUDA 提速（帧图是独立候选，见下行） |
| Qwen CUDA CP 帧图 | ABBA 八进程完整 3.0840→1.9113 s（-38.0%）、流式 3.6529→2.4353 s（-33.3%）、首 PCM 0.7688→0.5663 s（-26.3%），区间不重叠，60 份 codes/PCM/WAV 与 eager 逐字节一致；dispatch-bound 区域（每帧约 34 ms 主机墙钟的小 kernel 循环）成图收益最大 | 其他模型的帧循环都有同样收益；KV 变体、状态与 dtype 前提可从本例照搬；「平衡」电源计划下测得的数字适用于高性能计划 |
| Qwen CUDA 流式历史解码图 | CP 图两侧开启，ABBA 八进程各 1 预热 + 5 热轮，流式 2.4240→2.1140 s（耗时减少 12.8%），240 份产物逐字节一致；无内部历史状态的子图可以模型级有界缓存，但要串行覆盖输入/回放/复制输出，并显式保活常量 | 带卷积/KV 状态的解码器也能跨请求共享图；局部回放省 60% 等于端到端省 60%；不同实验的 CP 与解码提速能直接相加 |
| Qwen 1.7B Base CUDA | 换模型后重新校验权重、编码 2048 维参考（0.6B 为 1024），独立 12 进程三配置：完整耗时 −34.3%、流式 −39.1%，360 份产物逐字节一致 | 同一优化代码路径自动证明另一模型已验收；直接沿用低规格模型 prompt；Base 结果适用于 CustomVoice/VoiceDesign |
| CosyVoice3 Accelerate | M4 Pro CPU 样例大幅改善；BLAS/Rayon 组合存在争用，波形非逐位一致 | 已证明全部算子用了 AMX，或数值/试听普遍等价 |
| IndexTTS depthwise/FIR | 上游 PR 的局部借鉴及大形状算法对照有效，CUDA oracle 有精度配置边界 | 小输入通过即可覆盖长音频，或所有误差都由 TF32 导致 |

## 仓库记录

- [通用方法](../../../docs/knowledge/tts-performance.md)：架构与验证策略。
- [VoxCPM 扩展](../../../docs/records/voxcpm/voxcpm-review-extension-2026-10-18.md)：图生命周期、四项真模型测试、接受/拒绝实验。
- [Qwen Metal](../../../docs/records/qwen/qwen-performance-optimization-2026-10-18.md)：队列等待、采样、depthwise 及流式状态。
- [Qwen CUDA](../../../docs/records/qwen/qwen-cuda-performance-2026-10-18.md)：固定权重、无插桩初筛和拒绝的打包候选。
- [本 skill 的 Qwen 试用](../../../docs/records/platform/candle-performance-skill-trial-2026-10-18.md)：原始证据复核、capture 前提和由试用发现的参考缺口；不是图已通过的证明。
- [Qwen CP 帧图](../../../docs/records/qwen/qwen-cp-graph-2026-10-18.md)：整帧自回归循环成图、主机簿记不变式、失败策略分档、共享 recorder 提取与复验边界。
- [Qwen CUDA 图分区评估](../../../docs/records/qwen/qwen-cuda-graph-partition-2026-10-18.md)：历史探索数据与后续口径修正；固定容量不等于相同有效 KV 长度。
- [Qwen 解码图与审查](../../../docs/records/qwen/qwen-decoder-graph-2026-10-18.md)：有状态/无内部状态图的生命周期边界、eager oracle 必须独立、表缓存按字节约束、初次录制与多请求稳态计时。
- [Qwen 1.7B Base CUDA 独立验收](../../../docs/records/qwen/qwen17-cuda-acceptance-2026-10-18.md)：另一模型规格的资源与 prompt 身份、eager/单图/双图三配置配对验收；模型统一存储与经哈希验证的重复权重去重。
- [CosyVoice3](../../../docs/records/cosyvoice3/cosyvoice3-performance-2026-10-17.md) §11：Accelerate 线程矩阵、数值和试听边界。
- [IndexTTS](../../../docs/records/indextts/indextts-candle-performance-2026-10-17.md) §10–12：depthwise PR、FIR 大形状、严格 F32 与默认配置的区别。

查原生实现时对应 Candle revision/版本以目标项目 Cargo.lock 为准。本仓库历史基线
为 upstream 0.11.0 `31f35b147389700ed2a178ee66a91c3cc25cc80d`，fork
`bafb03db81c3eae48e63ff28d6b30f5a6576ba38` 仅增加 FORK.md，没有功能性补丁；
查当前依赖不要仅复制这个历史 revision。
