## Why

中文章节中的英文、缩写和型号是正常内容，当前 Kokoro 前端遇到英文字母即失败，阻断整章合成。

## What Changes

- 支持中文与英文片段混合转音素，保持原文及源坐标不变。
- 对未知或无法发音内容显式失败；不承诺其他语言已支持。
- 使用离线英文词典与词外读音模型，记录依赖与资源许可及验证边界。

## Capabilities

### New Capabilities

- `kokoro-mixed-language`：Kokoro 中文声音的中英文混读及失败语义。

### Modified Capabilities

无。

## Impact

影响 Kokoro 前端、依赖、发布二进制大小、最低 Rust 版本及中文/g2pW 验证；宿主无需修改原文或已完成音频。
