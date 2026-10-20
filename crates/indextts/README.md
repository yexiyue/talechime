# talechime-indextts

Talechime 的 IndexTTS 2.5 本地 Candle 计算库。使用转换后的本地官方权重运行 IndexTTS 2.5，转换与资源要求见仓库 `docs/indextts.md`。

Cargo 包名为 `talechime-indextts`，Rust 导入名为 `talechime_indextts`。

```toml
[dependencies]
talechime-indextts = "0.1.0"
```

默认使用 CPU；CUDA 选择 `cuda`（Windows/Linux），Metal 选择 `metal`（macOS）。
GPU feature 按目标平台显式选择，不同时开启 CUDA 和 Metal。

普通测试不下载模型；模型权重、用户音频与正文不包含在包内。
第三方来源与许可证保留在包内的来源声明和许可证文件中。
