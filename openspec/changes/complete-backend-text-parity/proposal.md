## Why

各后端移植程度与语言默认值不同，可能在跨语言和特殊文本上产生与官方不同的结果；需要逐模型补齐已识别的官方推理能力并验证。

## What Changes

- 按固定官方模型和版本补齐文本前端、语言选择及已识别的适配器能力。
- 以统一公开文本矩阵验证各后端，标识不支持项和未经真实模型验证的边界。
- 保持原文和来源坐标，不将后端专有语法误作通用语义。

## Capabilities

### New Capabilities

- `backend-text-parity`：各后端文本及语言路径对照官方的契约。

### Modified Capabilities

无。

## Impact

影响 Kokoro、Qwen、OmniVoice、IndexTTS、CosyVoice3、MOSS 和 VoxCPM 的适配、验证、依赖及宿主集成；不要求接入官方训练工具或其他模型家族。
