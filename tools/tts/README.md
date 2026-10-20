# 开发对照脚本

统一性能执行、采样、统计、恢复和报告使用 Rust `cargo xtask perf`，参见 [性能工具](../../docs/performance-harness.md)。这些 Python 文件不属于产品运行依赖，也不进入 Rust 性能测量链路。

仍保留的用途：

- 官方 Python/PyTorch 数值及性能对照：Qwen、VoxCPM、CosyVoice3、MOSS、IndexTTS 等 reference/bench，以及 Mac 官方 MPS、ABBA 对照脚本。
- 上游权重转换、ONNX/safetensors 导出和参考夹具生成：只能显式运行，测量阶段不自动调用。
- WAV 数值比对、回读、播放和切换验收：包括被播放验收使用的 `metrics.py`；生成完成不能代替播放完成。
- 既有报告复现：CosyVoice3 矩阵与汇总、VoxCPM Mac 工具仍被历史记录或彼此引用，Mac Rust 原生能力尚待补齐。

已清理：无引用的 Windows `benchmark_resources.py` 草稿，以及由 Rust 原生 RSS/整卡 GPU 采样取代的独立 `probe_metrics.py`。旧记录提到后者时指历史工具；新的性能证据使用版本化 Rust reading 和 artifact。
