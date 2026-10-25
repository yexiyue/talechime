# Qwen Base Windows 三段接续状态验证（2026-10-25）

设备为 Windows 11 / Ryzen 7 9700X / RTX 5070 12 GB，Candle fork
`2266099d92265112ecc0d56ecee1761e3cc2ac00`，CUDA 13 / cuDNN 9.14。
本轮验证当前 `QwenBackend` 的接续与取消恢复，不修改生产算术、采样或缓存格式，
不是新的官方性能对照或听感验收。其他基线见[主报告](qwen-windows-production-2026-10-24.md)。

## 输入与实际路径

显式真模型测试 `qwen::cuda_tests` 使用 Base06 的 `custom:acceptance06`、Base17 的
`custom:acceptance`，初始文本为“你好，欢迎收听今天的故事。”。接下来的文本依次为：

1. 今天的故事开始了。
2. 小女孩推开窗户，看见远处的森林。
3. 她带上地图，沿着小路慢慢向前走去。

每段用上一段完整、未裁剪的 24 kHz 单声道原始 PCM 和对应文本构造
`SpeechContext`，经实际 owner 的 `create_voice_clone_prompt` 和流式合成；
不重复使用第一段参考，也不把全文音频拼接成参考。固定 `Pinned(42)`，按
`resolve(segment_index + 1, 0)` 生成每段 seed，repetition_penalty=1.5、chunk_frames=20、
max_duration=30。两次从相同初始 PCM 独立推进整链，第二遍的中间段先读一个 PCM、
观察容量一的队列已满，丢弃流后用相同完整前段上下文重试。
这只证明观察到满队列，不声称发送线程必然正阻塞于 send。

## 结果与边界

| 模型 | 三段原始 F32 样本数 | 两遍逐位比较 |
| --- | --- | --- |
| Base06 | 48000 / 78720 / 80640 | 三段均一致 |
| Base17 | 42240 / 90240 / 84480 | 三段均一致 |

收集器要求有限 PCM、正确采样率、非空输出及成功 End；中段取消恢复后，第三段仍
逐位一致，整链结束后静态参考输出也恢复一致。逐段 raw F32LE SHA 写入原始日志。
该结果没有导出接续音频、做 ASR 或人工试听，也没有读取底层未裁剪 codes 的自然 EOS，
不能证明文本完整、长期无音色漂移或与官方克隆质量等价；长链、跨度重锚和质量仍待补齐。

同一进程串行四个变体的测试结果为 **3 通过、1 失败，104.60 秒**：两个 Base 和
Custom06 通过，最后 Custom17 初始化报 CUDA OOM。退出后 `nvidia-smi` 显示
947 MiB 已用、10997 MiB 空闲；单独新进程运行同一冻结二进制的 Custom17 测试，
**1 通过，23.42 秒**，换音色、style、取消与预算失败恢复均通过。
不把独立通过改写为原整组通过，也不据此断言泄漏、碎片或驱动根因。

同一冻结二进制的普通测试为26通过、6忽略；上述Windows原生feature组合的
release Clippy（all-targets，-D warnings）、rustdoc（-D warnings）、fmt及diff检查通过。

## 可恢复证据

仓库外根目录：`D:/workspace/talechime-benchmarks/windows-qwen-20261024`。

- `chain-state-build-receipt.json`、`chain-state-build.jsonl/.log` 和
  `chain-state-tests-source.rs` 固定测试源码及构建身份；二进制 SHA 为
  `e9bcdf30430ca59adeedf6bec89372d01d2e1aaa4be2d1540b1d955ad4c2755e`。
- `chain-state-real-tests.log` 保留整组失败和十二条分段摘要，
  `chain-state-custom17-isolated.log` 保留独立复查，二者不覆盖。
- 输入身份沿用 `disk-cache-state-final-evidence.json` 的
  `resource_identities.resources_sha256` 及固定 VoiceDesign 参考 WAV。
  `chain-state-input-before-v2.json` 核对全部 38 文件和原始参考 SHA；初版读取了错误
  JSON 层级、检查零文件，明确无效并保留，不能作为输入核验依据。

复现先配置 CUDA/cuDNN DLL 路径，以及 `TALECHIME_QWEN_STATE_RESOURCES`、
`TALECHIME_QWEN_STATE_REFERENCE`、`TALECHIME_QWEN_STATE_REFERENCE_SHA256`，
具体资源与固定 WAV 见[磁盘缓存验证](qwen-windows-prompt-cache-owner-2026-10-25.md)。
使用收据中的 exe，对每个 `real_cuda_base06_state_reuse`、
`real_cuda_base17_state_reuse`、`real_cuda_custom06_state_and_voice_switch`、
`real_cuda_custom17_state_and_voice_switch` 的完整 `qwen::cuda_tests::` 名称，
分别运行 `--exact --ignored --nocapture --test-threads=1`，串行调度独立进程。
这不会掩盖本轮同进程失败；若调查跨模型资源回收，应另保留同进程矩阵。
