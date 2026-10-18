# CosyVoice3 后端

`cosyvoice3` / 模型 `0.5b` 已接入 Talechime Registry、Rust Engine、CLI 与 JSON Lines v7。
当前为实验能力：零样本克隆、24 kHz 单声道原生流式输出；必须先导入参考 WAV 和准确转写。
没有内置音色，不开放 style、执行级 continuation、文本正规化或生成参数。
播放速度使用现有播放层，不声明流式模型原生 speed。

LLM、Flow/DiT、HiFT 使用 Candle，参考语音 tokenizer 与 campplus 使用 ORT CPU。
`cosyvoice3` 启用 CPU，`cosyvoice3-metal` / `cosyvoice3-cuda` 启用相应平台加速；
显式设备不可用会失败。标准发行配置包含 `cosyvoice3-metal`，非 macOS 构建仍可使用 CPU。
默认后端选择不变。

## 准备模型

模型只从本地加载。后端不会自动下载或调用 Python 转换。
以 `--model-dir ROOT` 为资源根，转换结果须放在：

```
ROOT/cosyvoice3/models/0.5b/2512-candle-f64-v1/
```

使用官方 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 权重和固定官方源码 commit
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`，按
[本机对照记录](records/cosyvoice3/cosyvoice3-local-parity-2026-10-17.md)准备 Python 环境和来源文件。
`llm.pt`、`flow.pt`、`hift.pt` 的原始 SHA-256 见该记录末尾。

```sh
python tools/tts/cosyvoice3_convert_weights.py \
  --input OFFICIAL_MODEL_DIR \
  --official_repo PINNED_COSYVOICE_CHECKOUT \
  --output ROOT/cosyvoice3/models/0.5b/2512-candle-f64-v1
cargo build --release --locked -p talechime --features cosyvoice3-metal
```

准备入口逐文件校验 `crates/talechime-backends/src/cosyvoice3/resources.json` 的大小与 SHA-256，
包括 config、三个 safetensors、完整 tokenizer 和两个 ONNX 前端。
校验失败保留原文件，并提示重新转换。`rand_noise.safetensors` 是开发对照文件，运行时不需要。
转换工具与依赖版本会影响导出字节，须使用上述固定环境；更新转换产物时须重新审查 manifest。

## 导入与使用

```sh
talechime --backend cosyvoice3 --model 0.5b --model-dir ROOT \
  voices import reader --name 朗读者 --text '参考录音的准确转写。' reference.wav
talechime --backend cosyvoice3 --model 0.5b --model-dir ROOT voices list
talechime --backend cosyvoice3 --model 0.5b --model-dir ROOT \
  --tts-device metal --voice custom:reader chapter.txt
```

参考 WAV 沿用共享音色存储的校验：1..30 秒、8..192 kHz、单/双声道、有效非静音 PCM。
推理时转换为 24 kHz，再提取参考 token、mel 和说话人向量；转写自动添加官方 end-of-prompt 前缀。
音色 remove、配置隔离与资源根参数复用现有命令。

Rust 库使用 `Engine::prepare(ModelOptions::new("cosyvoice3", root), progress)`，
显式选择 `model = Some("0.5b")` 和实际设备后，传入已导入的 `custom:reader`。
普通测试不下载或加载模型。显式端到端探针：

```sh
cargo run --release --locked -p talechime --features cosyvoice3-metal \
  --example cosyvoice3_backend -- ROOT custom:reader metal
```

探针通过 Engine 验证连续同 seed 请求、收取 PCM、首块后取消、取消后再合成和关闭。
它不打开播放器或写检查点，也不证明听感或文字完整性。

## 所有权与限制

专用推理线程持有一次加载的 Candle/ORT 模型，借用模型创建逐请求流；请求队列与 PCM 队列容量均为 1。
先取得 PCM 队列许可再推理下一块，消费者退出后在块边界取消；同步 ONNX/LLM/Flow/HiFT 运算无法中途抢占。
流销毁清除 KV，请求不共享 mel/token 缓冲，后续请求复用权重。
只有正常 final block 后发布 End；推理错误、长度上限和断连都不能冒充完成。
当前不声明持久音频缓存身份，防止参考文件变化后复用旧结果。

严格非流式整链数值门禁仍未通过，先前最大波形误差约 0.162；组件及流式对照结果见
[审查记录](records/cosyvoice3/cosyvoice3-review-2026-10-17.md)。接入后端不代表完全复刻官方、试听验收或 CUDA 实机验收。
