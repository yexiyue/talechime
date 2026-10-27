## ADDED Requirements

### Requirement: Fixed official conditioning

系统 MUST 仅使用所选音色的固定参考或官方预置/自然生成方式，不将之前生成的音频、转写或临时说话人向量滚动注入新请求。

#### Scenario: Consecutive chunks

- **WHEN** 连续生成同音色的多个语义片段
- **THEN** 每片复用所选固定参考，PCM 不成为下一请求输入；保留单请求的原生流式状态

### Requirement: Removed strategy surface

系统 MUST 删除滚动及双参考能力、配置、实验开关、开发入口和无用实现。

#### Scenario: Removed protocol fields

- **WHEN** 宿主发送旧 continuation 配置
- **THEN** 未知字段被拒绝，不隐式兼容或开启旧路径

### Requirement: Preserve ownership and evidence

系统 MUST 保留取消、背压、原文范围、内容检查和实际播放进度契约及用户音色资产，明确历史记录与当前行为的差别。

#### Scenario: Validation

- **WHEN** 执行普通检查
- **THEN** 不下载模型或读取私有小说，真实模型/平台验收与 fixture 分开记录
