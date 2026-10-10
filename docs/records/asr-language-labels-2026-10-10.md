# ASR 语言自动检测与音频标签捕获

## 范围

生成参数（`ac292b4`）开放各后端 `language` 后，回读侧的对应联动：

- **Qwen3-ASR 主识别改用语言自动检测**：删除强制 `with_language("Chinese")`
  （worker 构造点），`max_new_tokens` 由 256 恢复为 512 以覆盖较长非中文段。
  生成侧 language=english 的段落不再被中文强制识别误判。
- **SenseVoice 音频标签捕获**：`Recognizer` 的返回值从纯文本升级为
  `Transcript { text, labels }`；SenseVoice 解码时的 `<|…|>` 情感/事件/语言
  标签从"丢弃"改为捕获（去重、保持首次出现顺序），经 `ReadbackEvidence.labels`
  （v7 加法式 `#[serde(default)]` 字段）进入报告。文本比对与判定逻辑完全不变。
- `asr_readback` 探针输出逐识别器的 text/labels；Qwen 路径 labels 恒为空。
  这为"读取侧情感标签反哺生成侧逐段 style"（issue #2 方向）提供数据源，
  本轮未接入任何自动策略。

## 验证

无模型测试：确定性转写夹具回归（含 `From<&str>`/`From<String>` 兼容）、
证据构造三分支（Ok/超长/Err）的 labels 透传。扩展 CPU 特性 486 项测试、
Clippy（warnings denied）、rustdoc、格式检查通过。真模型回读行为（自动检测
准确率、标签质量）本轮未重新验收；此前主线验收基于强制中文；改变语言检测后，中文与非中文语料均需重新验收，
不能沿用此前中文语料结论证明当前路径。
