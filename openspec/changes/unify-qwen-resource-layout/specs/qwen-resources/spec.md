## ADDED Requirements

### Requirement: Qwen 模型资源目录
系统 SHALL 按模型 ID 和固定 revision 解析每个 Qwen 模型的资源目录，不依赖模型是否为默认值。

#### Scenario: 默认模型
- **WHEN** 用户选择默认 Qwen 模型并准备资源
- **THEN** 使用该模型 ID 和 revision 对应的目录，下载及加载解析同一位置
