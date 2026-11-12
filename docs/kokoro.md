# Kokoro 中文语音后端

Kokoro 提供 `1.1-zh-g2pw` 中文预置音色，使用 g2pW 预测多音字，支持中英文混排。声学模型使用 Candle，g2pW 在 CPU 上使用 ONNX Runtime。英文使用 crates.io 固定版本 0.4.0 的词典和美式神经未知词后备。

输出为 24 kHz 单声道分段 PCM；支持 `speed`（0.5–2.0）和官方 `zf_*` / `zm_*` 音色，不支持原生流式、参考音色克隆、风格指令或自定义读音。未知字符或不能映射的音素明确报错。原文范围由宿主维护，后端只规范化执行文本。

## 构建与资源

使用 `kokoro` feature；macOS Metal 使用 `kokoro-metal`，CUDA 使用 `kokoro-cuda`。标准发行配置包含 Kokoro，默认后端仍为 MOSS Nano。

```sh
cargo run --release --locked -p talechime --no-default-features \
  --features kokoro --example kokoro_backend -- RESOURCE_ROOT zf_001 cpu
```

资源通过 `Registry::resource_status` 离线盘点，由 `ResolvedModel::prepare_resources` 下载和校验；`load_existing` 只读取已就绪的共享 Hub 缓存，不执行本地转换。固定清单包含模型、音色、前端词典和 g2pW 上下文资源，允许来自不同仓库和 revision。资源身份及许可详见 [SOURCE.md](../crates/kokoro/SOURCE.md) 和[资源准备说明](native-resource-preparation.md)。

推理语义身份由 `ResolvedModel::revision()` 和 `text_semantics_revision()` 共同提供；宿主复用音频时必须冻结和比较两者。取消在模型阶段与英文片段之间检查，单次底层算子或神经 G2P 推理不可抢占。

## 验证边界

自动测试覆盖文本规范化、音素预算和输入拒绝；使用固定资源可运行 `mixed_oracle` 和 g2pW 对照。macOS CPU 上真实加载并合成过包含英文专名与中文多音字的公开短句，生成有效 PCM。部分专名或自造词可能误读；g2pW 按中文片段预测可能失去跨英文的上下文。

真实音频生成和音素测试不等于人工听感通过。原生窗口、其他平台、CUDA 和发行安装包分别验收；不要以一种设备或一种音色的结果代表全部场景。
