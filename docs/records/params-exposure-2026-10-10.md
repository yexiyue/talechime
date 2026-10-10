# 四后端生成参数与 seed 接入

## 范围

在 `950bd59` 的协议与核心缝上，把生成参数和 seed 真正接入四个后端并声明能力目录：

- **VoxCPM2**：`steps`（2..128，flow-matching 步数）、`cfg`、`temperature`、`max_duration`
  （1..30 秒；AR 帧单位 2560 样本 ≈ 18.75 Hz，默认 200 帧 ≈ 10.7 秒）。适配层解析为
  `voxcpm::Options`，seed 进初始 latent 噪声。
- **Qwen3-TTS**：`temperature`/`top_k`/`top_p`/`repetition_penalty`（目录注明克隆路径内部
  下限 1.5）、`language`（auto 或 10 种，未知名显式报错）、`max_duration`（12.5 Hz，默认
  375 帧=30 秒）、`chunk_frames`（未设置时沿用 CUDA 20 / 其他 10）。seed 进
  `SamplingContext`。
- **OmniVoice**：`num_step`、`guidance_scale`、`language`（约 700 种内建，未知名从上游的
  静默 "None" 改为显式报错）、`speed`（0.5..2.0 生成期原生语速）。seed 经新增
  `Pipeline::set_seed` 逐请求设置。
- **MOSS Local/Realtime（Candle）**：`instruction`（user_inst 模板 Instruction 槽位）、
  `max_duration`（25 Hz，默认 750 帧=30 秒）。`Generation.seed`/`max_frames`/`instruction`
  逐请求填充；Nano/ONNX 只消费 seed（Nano 原本随机，ONNX 原本 None=随机，行为不变，
  来源改为生产者派生）。
- 每后端提供类型化构造器（`QwenParams`/`VoxCpmParams`/`OmniVoiceParams`/`MossParams`，
  `to_generation_params()` 转换），经 `talechime` 门面按后端 feature 导出。
- **原生语速自动路由**：会话启动时若后端声明 `speed` 参数且宿主未显式设置，则把
  `PlanSessionOptions.speed`（≠1.0）写入生成参数、播放 sink 置 1.0（无损变速）；
  运行期 `configure_playback` 在原生语速上按比例换算（新值 ÷ 原生值），不叠加。
  未声明 `speed` 的后端维持播放端变速；宿主显式设置的 `speed` 参数优先于自动路由。
- `continuation_probe` 探针支持尾部 `--seed n` 与 `--param name=value`（可重复），
  summary.json 记录实际 seed 与参数。

## 真模型验证（VoxCPM2 Q8，Apple M4 Pro / Metal）

资源根 `~/.novel-tts`（既有 revision 目录），语料沿用接续记录的两行平静叙述
（源摘要 `ce393482…21117`），`--seed 7 --param steps=16`：

| 运行 | 生成耗时 ms | 交付音频 s | RTF | comparison.wav SHA-256 |
| --- | ---: | ---: | ---: | --- |
| seed7-a | 13943 | 7.36 | 1.894 | `19cc2add…a0733ea` |
| seed7-b | 13943 级 | 7.36 | — | `19cc2add…a0733ea`（与 a 完全一致） |
| seed8 | — | — | — | `640580ce…f65eb99`（与 seed7 不同） |

- 同 seed 两次输出字节级一致：Pinned 派生在该后端可复现。
- 换 seed 输出不同：seed 真实驱动采样。
- steps=16 的 RTF 1.89 对照接续记录中默认 steps=10 的 1.18：步数参数真实生效。
- 首次运行含资源校验与 Metal 编译（load_and_verify 656 s），非稳态数据。
- OmniVoice/Qwen/MOSS 的真模型参数探针本轮未运行（本机时间预算）；无模型测试覆盖
  各自的目录声明、解析与默认值往返。主观听感未验收。

## 无模型验证

参数目录自洽（默认值满足自身类型）、类型化构造器往返、空参数保持各后端原默认值、
OmniVoice 未知语言显式报错、语速路由两方向（声明 `speed` 的后端进生成参数且 sink=1.0、
运行期换算不叠加；未声明的后端维持播放变速且参数不进后端）、协议目录断言
（voxcpm/steps、omnivoice/speed、qwen/temperature 存在且默认合法）。扩展 CPU 特性
483 项测试、逐特性 Clippy（warnings denied）、rustdoc 通过。

对照文件：`target/params-exposure/{voxcpm-seed7-a,voxcpm-seed7-b,voxcpm-seed8}/`。
