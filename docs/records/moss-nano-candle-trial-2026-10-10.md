# Nano Candle 试听候选：2026-10-10

## 状态

从 TRNovel `codex/moss-nano-candle` 的固定提交
`8be7d953fd91433569767404e5a7a37534956d5b` 移植 Nano 部分到
Talechime `codex/moss-nano-candle-trial`，没有移植该分支的 Qwen／OmniVoice ONNX 实验。
模型项 `moss/nano-candle`，CPU 可用，Metal／CUDA 使用各自显式 feature。
当前默认 Nano 仍是 ONNX，未合并主线或替换默认；主观听感由用户对照试听决定。

本次原生接续版有过早结束的内容风险，即使听感改善，也应先处理此项再考虑默认替换。
关闭接续的相同末句正常生成 4.64 秒；开启接续只生成一帧（0.08 秒）后模型返回 EOS。
尚未确定是模型采样、前段条件或移植实现导致，不能把它描述为已修复或纯粹的 ASR 误差。

## 实现与资源

- 计算库沿用工作区 Candle 0.11.0，官方 BF16 PyTorch 权重加载后以 F32 推理。
- TTS 固定 revision `44502f80dbf9743528fa921cc544d662c685ebec`；codec 固定
  `6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`，资源按 manifest 大小及 SHA256 校验。
- 目录为 `resources/moss/models/nano-candle/REVISION/`；本次探针显式使用已有缓存根
  `/Users/yexiyue/.novel-tts`，产品没有新增旧目录自动回退。
- 复用原 Nano 的内置／自定义 voice codes。CLI 导入音色仍使用原 Nano ONNX encoder；
  接续中的临时前段音频则由原生 codec encoder 编码，不写入音色缓存。
- 接续使用与 ONNX 共享的官方 prompt 构造函数，完整前段文字与目标文字、用户参考 `None`、
  assistant 音频 codes 前缀。原生 decoder 分批预热，丢弃前缀 PCM，仅交付新帧。
- 上下文和限额复用 core producer；seek、恢复、段落、音色边界与执行隔离规则保持一致。
- 初始化取消先关闭 ready receiver 再 join owner；有界 PCM 发送可被接收端或 owner 关闭打断。
- 保留 Apache-2.0 LICENSE／NOTICE 与固定来源记录。小型合成权重 fixture 是数值测试数据，
  不包含下载的生产模型或用户音频。

## 工程验证

默认工作区 432 项、扩展 CPU 450 项测试通过；格式、扩展 Clippy（warnings denied）及
rustdoc（warnings denied）通过。Metal release 探针和 CLI 均构建成功，实际音色目录查询成功。
CUDA 尚无本机编译／实测结论。

小型 GPT2 fixture 覆盖 prefill、KV cache、取消；原有 24 kHz codec 数值回归通过。
显式真实模型 codec 测试在 CPU／Metal 上均通过：65 帧原 Nano 音色 codes，三帧流式解码，
1024 个采样点与固定 ONNX 参考的最大差小于 `3e-4`，包含流式缓存回绕。
普通测试不下载权重；真实 codec 测试通过显式模型路径单独运行。

## 音频与回读

设备为 Apple M4 Pro，24 GiB 统一内存；release，Metal，Weiguo，固定种子 42。
语料和 SHA256 与[三小模型接续对照](continuation-small-models-2026-10-10.md)相同。
当前默认分块器按实际生成时长学习，开／关分块数量可能不同，但都覆盖同一份原文。
部分运行与编译或回读并行，下列性能仅是本次观察，不是隔离负载基准。

| Candle 模式 | 音频秒数 | 生成秒数 | RTF | 回读 |
| --- | ---: | ---: | ---: | --- |
| 接续开 | 47.65 | 16.292 | 0.342 | 11 段：8 通过、3 疑似 |
| 接续关 | 58.70 | 13.425 | 0.229 | 10 段：8 通过、2 疑似 |

接续开首个 PCM 183 ms；携带前段条件的片段首个 PCM 180–708 ms。
两组疑似均包含标题与“雨停以后……”的发音／转写差异；开启接续的额外疑似是末句过早 EOS，
不能因为报告没有将其升级为 `ConfirmedError` 就当作内容正确。
未发现前段文字重复播出。音频成功生成、decoder 数值通过均不代表主观质量或逐字完整性通过。

本机证据位于忽略目录 `target/nano-candle-trial/`：`candle-on/`、`candle-off/` 含完整及逐段 WAV、
PCM16 回读 WAV、范围与指标；`readback-*-results.json` 保留两种 ASR 的原始结果。
ONNX 接续基线音频位于 `target/continuation-evidence/moss-final-on/comparison.wav`。

## 重现和手工试听

```sh
cargo build --release --locked -p talechime --features moss-nano-candle-metal
target/release/talechime --backend moss --model nano-candle --tts-device metal --voice Weiguo chapter.txt

cargo build --release --locked -p talechime-backends --features moss-nano-candle-metal,qwen-metal,omnivoice-metal --example continuation_probe
target/release/examples/continuation_probe moss-candle RESOURCES_ROOT target/nano-candle-trial/candle-on metal Weiguo on CORPUS_TXT
# 把 on 改为 off，并选择另一输出目录生成无接续对照。

TRNOVEL_MOSS_NANO_CANDLE_DIR=NANO_MODEL_DIR cargo test --locked -p moss-tts --features metal --lib
```

已有缓存可以显式传 `--model-dir`；试验配置和检查点也应使用独立目录。
本次 CLI 已构建到 `target/release/talechime`，普通执行仍由配置选择模型，不自动切到候选实现。
