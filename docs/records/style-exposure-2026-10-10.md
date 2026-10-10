# 逐段风格的开放与组合边界

## 范围

在参数/seed 接入（`ac292b4`）之上，把逐段 `SpeechSpan.style`（协议 `VoiceSpan.style`）
接入三个后端的能力目录；Qwen Custom17 原本已支持。core/协议管道自 T0 起就逐段携带
style，本阶段只打开后端开关并接线：

- **VoxCPM2**：`caps.style=true`。style 按上游 README 的官方约定渲染为
  `(description)text` 目标文本前缀；描述含括号显式拒绝（会模糊前缀边界）。
  上游已核实（revision `f0c787f`）：`generate` 不解析前缀，括号描述就是普通文本，
  且 README 明确展示参考音频＋描述前缀的 Controllable Voice Cloning。
- **OmniVoice**：`caps.style=true`。style 经 `with_instruct` 逐请求注入，前端
  `resolve_instruct` 的词表/互斥校验对朗读路径同样生效，非法词显式报错。
- **MOSS Local/Realtime（Candle）**：`caps.style=true`。span.style 逐请求写入
  `Generation.instruction`（Instruction 槽位），逐段 style 优先于同名执行参数。
- **MOSS language 槽位**：核实官方 `build_user_message` 签名后为 Local 增加
  `language` 参数（官方 v1.5 指南："when the language is known, set it"），
  Realtime 的消息格式无此槽位、目录不声明；`Generation`/模板/Local 链路相应扩展，
  `language=None` 时输出与既有官方格式夹具逐字节一致。Quality / Sound Event /
  Ambient Sound 槽位虽在官方签名中，但仓库与官方均无使用示例与取值语义，
  本轮不开放。
- 探针新增 `--style`，summary 记录实际 style。

## 真模型边界（VoxCPM2 Q8，Apple M4 Pro / Metal）

资源根与语料同参数记录，`--seed 7 --param steps=16 --style "gentle and calm narration"`：

| 组合 | 结果 |
| --- | --- |
| style + 接续 off | 正常：两段 4.16 s + 5.44 s，合计 8.82 s |
| style + 接续 on | **异常**：续读段仅 0.96 s（该句约需 3 s），疑似过早 EOS 截断 |

处置：style×continuation 组合在 VoxCPM 上显式拒绝（Unsupported），直到听感与
正确性验收通过；style 单独使用与接续单独使用均不受影响。这与上游文档一致——
官方示例只展示描述前缀与参考音频克隆的组合，未展示与续读（prompt 转写拼接）的组合。
OmniVoice/MOSS 的 style×接续组合本轮未做真模型验证，听感未验收。

## 无模型验证

官方格式夹具回归（language=None 输出不变）、Realtime 目录不含 language 且传值被
拒、style 优先级（span 覆盖参数）、括号拒绝、组合拒绝矩阵、三家 caps.style 断言。
扩展 CPU 特性 485 项测试、Clippy（warnings denied）、rustdoc 通过。

对照文件：`target/params-exposure/{voxcpm-style,voxcpm-style-off}/`。
