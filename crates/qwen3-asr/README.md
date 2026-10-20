# talechime-qwen3-asr

Talechime 的 Qwen3-ASR 本地 Candle 计算库。使用本地模型文件运行 Qwen3-ASR 推理。模型资源准备与回读策略由宿主负责。

Cargo 包名为 `talechime-qwen3-asr`，Rust 导入名为 `talechime_qwen3_asr`。

```toml
[dependencies]
talechime-qwen3-asr = "0.1.0"
```

普通测试不下载模型；模型权重、用户音频与正文不包含在包内。
第三方来源与许可证保留在包内的来源声明和许可证文件中。
