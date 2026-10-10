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

## 官方实现对照与修正

用户反馈原生候选有音色突变、音量不一致后，对照上述固定 TTS／codec revision 的官方
`prompting.py`、`modeling_moss_tts_nano.py`、`gpt2_decoder.py` 和
`modeling_moss_audio_tokenizer.py`。本次发现并修正两处实现差异：

1. 共享 Nano prompt builder 遗漏用户消息开头的 `im_start_token_id=4`。
   manifest 的 `user_prompt_prefix_token_ids` 只包含模板正文，官方 builder 另行添加控制 token。
   此错误同时影响本分支 ONNX／Candle Nano；首段 voice-clone 和后续 continuation 均修正。
   用官方 builder 生成含两帧固定 dummy codes 的小型 fixture，逐行核对两种模式。
2. 原生 encoder 返回了不足一帧末尾 padding 的 codes。固定官方实现按原始长度逐级向下取整，
   再依据 `audio_codes_lengths` 排除无效帧。原生现在保持 padding 存储、仅返回有效 codes。
   正常原始生成 PCM 按整帧对齐，这个边界修复不能单独解释常规接续中的音色变化。

官方来源：

- [TTS prompt builder](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/blob/44502f80dbf9743528fa921cc544d662c685ebec/prompting.py)
- [TTS generation](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/blob/44502f80dbf9743528fa921cc544d662c685ebec/modeling_moss_tts_nano.py)
- [codec](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano/blob/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68/modeling_moss_audio_tokenizer.py)

### 数值证据

独立官方 Python 探针使用 PyTorch 2.14.1／Transformers 5.19.0，CPU 四线程；生产推理仍为 Rust。
新工具 `tools/tts/nano_reference.py` 只读取显式提供的源文件、缓存权重与音频，不下载模型。
生成的大型参考、官方源文件和音频均留在忽略目录 `target/nano-candle-diagnosis/`。

- 6.84 秒、48 kHz stereo、末尾半帧输入：修正前有效帧后多出一帧。
  修正后 CPU／Metal 均为 85 帧，1360 个有效 code 与官方完全一致。
- 96 行混合文字／音频输入，真实 12 层 global 和 1 层 local：CPU 最大差
  `1.6481e-5`／`1.5140e-5`，Metal 最大差 `1.2636e-5`／`1.3590e-5`。
- 官方 voice-clone 后连续两次 continuation，100／74／80 帧；沿官方 token 轨迹逐帧核对
  text 候选 logits 和每个 audio head 的 16 个均匀取样 logits，Metal 最大差分别
  `2.3842e-5`、`2.8372e-5`、`2.6941e-5`。覆盖真实 prompt、全局 KV cache 与局部 16 路生成。
- 为隔离计算误差，生成参考时仅把官方 `torch.multinomial` 的随机抽签替换为 Candle 的 LCG。
  前两组 100／74 帧采样 codes 完全一致；第三组在第 73 帧第 13 路首次分叉，后续随机轨迹变化。
  因浮点微差在概率边界附近会改变离散选择，验收采用相同 token 轨迹的 logits 对照，
  不要求两个框架完整随机波形逐采样点一致；官方默认 Torch RNG 也与 LCG 不同。

### 修正后的试听证据

相同 M4 Pro／Weiguo／seed 42／固定中文原文重新生成：

| Candle 模式 | 音频秒数 | 生成秒数 | RTF | 片段数 |
| --- | ---: | ---: | ---: | ---: |
| 接续开 | 46.15 | 12.656 | 0.274 | 10 |
| 接续关 | 55.15 | 11.363 | 0.206 | 11 |

运行与部分验证并行，以上不作为隔离负载性能基准。接续开末句不再是旧候选的一帧 0.08 秒，
本次生成 4.08 秒，两种 ASR 均完整识别出“母亲抬起头，笑着替他接过湿透的外衣”。
新接续音频的十段回读报告均为 `passed`，没有把主观音色／响度验收替换为 ASR 判定。

新音频位于 `fixed-on/comparison.wav` 和 `fixed-off/comparison.wav`；`loudness.json` 保留逐段 RMS。
同一 prompt 修正后的 ONNX 对照位于 `onnx-fixed-on/comparison.wav`，CPU，45.57 秒音频、11 段；
默认实现与候选同时修正后再进行试听，避免使用旧 prompt 基线。该 ONNX 对照本次未重新跑 ASR。
接续开首个平静段约 `-28.66 dBFS`，情绪后收尾段约 `-18.60 dBFS`，仍有明显幅度变化。
补齐 prompt 后常规句间的数值变化有所改变，但音色稳定性、情绪与响度变化是否自然仍需试听；
不能认定所有听感问题都由缺失 token 引起，也未增加响度归一化去掩盖生成行为。
默认实现仍保留 ONNX，未切换主线。

复现参考时，`OFFICIAL_SOURCE_DIR` 包含 `official_tts/` 下固定 revision 的五个 Python 文件
（configuration、modeling、gpt2_decoder、prompting、tokenization）与空 `__init__.py`，以及
codec 的 configuration／modeling 两个 Python 文件。工具读取该目录，不把第三方源文件写入仓库。

```sh
uv run --with torch --with torchaudio --with transformers --with sentencepiece --with numpy --with safetensors --with soundfile python tools/tts/nano_reference.py OFFICIAL_SOURCE_DIR NATIVE_MODEL_DIR REFERENCE_WAV OUTPUT_DIR
TRNOVEL_MOSS_NANO_CANDLE_DIR=NATIVE_MODEL_DIR TALECHIME_NANO_ENCODER_REFERENCE=ABS_OUTPUT_DIR/encoder-reference.json cargo test -p moss-tts --features metal real_nano_encoder -- --nocapture
TRNOVEL_MOSS_NANO_CANDLE_DIR=NATIVE_MODEL_DIR TALECHIME_NANO_TTS_REFERENCE=ABS_OUTPUT_DIR/tts-reference.json cargo test -p moss-tts --features metal real_nano_transformers -- --nocapture
TRNOVEL_MOSS_NANO_CANDLE_DIR=NATIVE_MODEL_DIR TALECHIME_NANO_GENERATION_REFERENCE=ABS_OUTPUT_DIR/generation-reference.json cargo test -p moss-tts --features metal real_nano_generation_logits -- --nocapture
```

本轮默认工作区 438 项、扩展 CPU all-targets 454 项、纯库 17 项测试通过；格式、Clippy、
rustdoc（warnings denied）通过。Metal release CLI／探针已重新构建，真实 encoder、Transformer、
逐帧 logits 验证分别记录在忽略目录的 encoder-fixed.log、tts-metal.log、teacher-forced.log。
Python 参考工具实际生成参考和三段 WAV，并通过语法检查；没有 CUDA 本机验证。
