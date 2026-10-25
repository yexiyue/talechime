# 各 TTS 后端流式边界审查（2026-10-25）

本轮从已确认的 Qwen 分块噪音继续检查其余后端，关注同一组生成 tokens/latents 的
解码历史、窗口淘汰、lookahead、重叠尾部、最终补尾与取消后重置。不同文本段独立生成的
音色一致性不属于这一类解码缺陷；跨框架采样差异也不作为定位依据。

| 实现 | 结果与处理 | 验证范围 |
| --- | --- | --- |
| Qwen | 已修复：逐层 K/V、前置卷积和有界声码器特征历史替代截短 codes 重算 | [同码隔离、人工反馈及实模记录](../qwen/qwen-streaming-noise-2026-10-25.md) |
| MOSS Nano Candle、Local、Realtime 共用 codec | 新修复：先对旧 K/V 加当前整块计算注意力，再裁剪返回的缓存；旧顺序会丢失当前块早期 query 仍需的历史 | 非零官方阶段 fixture、不同/超窗口块长、固定码 PCM、取消/重置；137 帧真实 codec Metal 回放 |
| MOSS Nano ONNX | 未发现同类缺陷：导出图先合并原始历史与当前块，单独裁剪返回缓存，适配器传递全部状态 | 核验固定图摘要及 52 个状态映射；同图 CPU ORT 1.24.4 的整段与 3/5 帧分块回放 |
| VoxCPM2 | 未发现同类缺陷：逐层卷积历史和转置卷积 overlap tail 持续保留，每请求清零 | 真实捕获的相同 22 个 latent patches，Metal F32/F16，以 1/2/7 patch 分块及取消后 reset，PCM 均与整段逐样本相同 |
| CosyVoice3 | 未发现同类缺陷：完整 token 前缀、累积 mel、固定源相位重放和按偏移输出，与官方协议一致 | [调度/独立标量 oracle、固定官方 tokens、真实同 mel 前缀检查](../cosyvoice3/cosyvoice3-streaming-boundary-audit-2026-10-25.md) |
| OmniVoice | 非原生小块 PCM 流式；每个语义段完整 codec 解码，长文本段按官方规则淡出/停顿/淡入 | 本轮源码及固定官方实现对照，无新模型执行 |
| IndexTTS 2.5 | 非原生小块 PCM 流式；每段完整语义 codec、CFM 和 BigVGAN，段间插入停顿 | 本轮源码及固定官方实现对照，无新模型执行 |

## MOSS 修复与数值边界

原 MOSS Candle 在多 query 一块的注意力之前将 K/V 裁到最终窗口。块早期位置需要的旧
key 会提前丢失；当块比窗口更长时，还可能造成早期 query 没有任何有效 key。
现在保留旧历史与本块全部 K/V 参与计算，由每个 query 的绝对位置与因果窗口 mask 决定
可见范围，之后才裁剪缓存。返回历史仍有界，编码器离线行为、帧分组与所有权不变。
这有意修正上游 pre-attention ring overwrite 的行为，旧上游 fixture 保留来源意义，
不再用其环绕后的错误输出作为正确性标准。

原代码在独立 fixture 的最大误差为 `0.0021459907`，修复后的测试小于 `1e-5`。
实际 Nano codec Metal F32 分块/整段最大误差小于 `9.4e-7`；Local/Realtime 共用 codec
F32 小于 `6.6e-6`。同一共享 codec F16 的 RMSE 为 `2.02e-4`–`2.15e-4`、最大
`0.00878`，仍含半精度分块运算差异，不能称为波形等价或全部听感通过。
ONNX 同图 CPU 分块最大误差小于 `7.9e-7`；该 Python ORT 图级验证不代替每个生产
Rust ORT provider/版本的设备验收。

## CosyVoice3 的正确比较方式

固定官方 tokens 的三块流式 probe 通过既有长度、mel 与 PCM 容差，但最终 PCM 与官方仍有
浮点/阶段传播差异，不能据此宣称严格波形等价。官方自身流式与非流式也会重算不同 mel。

隔离这项差异后，固定同一份 368 帧 mel、源相位及噪声，当前 Metal F32 HiFT 对
25/50/75/100/367 帧前缀和最终短尾所输出的有效 PCM，均与完整 mel 一次解码逐样本相同。
这验证的是该 fixture 的声码器前缀稳定性，而非所有文本的主观音质保证。

## 检查与证据

新增普通测试不下载模型；真实回放均需显式环境变量并设置为 ignored。模型和参考/生成
音频未加入版本控制。每层状态保留在原模型/会话所有者中，没有引入通用跨模型缓存层。
普通模型与适配器测试、全后端 Metal feature 组合的 Clippy/rustdoc、格式检查另有日志。

原始输出在忽略目录 `runs/streaming-audit-2026-10-25/`：Vox F32/F16、MOSS Nano/共用
codec F32/F16、MOSS ONNX、CosyVoice3 固定 token/同 mel 回放及组合检查。
本轮未运行 CUDA 原生编译、图回放、性能或人工长篇试听，不将 Mac 结果外推到 CUDA。
