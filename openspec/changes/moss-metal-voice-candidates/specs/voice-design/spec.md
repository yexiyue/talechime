# Spec Delta

## Purpose

提供通过文字描述创建可复用音色参考的能力，并保证设备能力声明、资源准备、完整生成与取消行为一致。本次扩展让 macOS 用户能够在明确设备和资源边界内使用 MOSS 设计引擎，而不将编译成功误认为真实音频验收成功。

## ADDED Requirements

### Requirement: MOSS Metal 显式设备支持

MOSS 设计引擎 SHALL 在 macOS 编入并可用的 Metal 路径上接受显式 Metal 请求，保持 CUDA 路径；清单 MUST 与实际设备支持一致，不支持的设备 MUST 在模型准备前明确拒绝，不得静默切换设备。

#### Scenario: Mac Metal 设计
- **WHEN** macOS 构建已启用 MOSS Metal 且设备可用，用户指定该引擎与 Metal
- **THEN** 系统使用 Metal 生成完整参考并返回库音色身份，不要求先准备朗读模型

#### Scenario: 未编入 Metal
- **WHEN** 构建未包含 Metal 或设备不可用，用户请求 Metal 设计
- **THEN** 请求在资源准备前失败，清单不将其列为可用设备

### Requirement: MOSS 设计资源与完成边界

设计 SHALL 对模型和 codec 的加载、生成及退出保持明确所有权。OOM、取消、超时、未正常完成或无效音频 MUST 形成失败且不发布半成品；完整产物必须具有有效有限值 PCM、转写和来源。不得通过裁剪、补静音或调速修复候选。

#### Scenario: 取消后继续使用
- **WHEN** 设计加载或生成期间取消，随后发起另一项设计
- **THEN** 前项结束并释放其资源且不新增半成品，后项可独立运行

#### Scenario: 内存不足
- **WHEN** 加载模型或 codec 遇到内存不足
- **THEN** 明确报告失败，保留既有库资产，释放本次资源，不自动重新下载或切换 CPU

### Requirement: Metal 支持的实机证据

项目 SHALL 在声称 MOSS Metal 已验收前记录指定 Mac 的设计成功、音频有效性、固定参考复用与取消后再运行证据；数值或音频差异不得以跨设备逐位一致作为前提。能力存在、执行成功、人工听感确认 MUST 分开报告。

#### Scenario: 仅通过编译
- **WHEN** Metal 编译通过但没有成功完成真实设计
- **THEN** 状态仅为编译通过或实机阻塞，不标记为设计和听感已验收
