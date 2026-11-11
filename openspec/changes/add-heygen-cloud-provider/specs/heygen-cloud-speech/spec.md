# Spec Delta

## Purpose

让 Rust 宿主通过独立云语音库发现 HeyGen 音色并消费指定模型生成的音频流，保留服务端状态、格式和费用的不确定性，同时不引入本地推理、数字人、视频或播放职责。

## ADDED Requirements

### Requirement: 独立 provider 与显式模型
HeyGen 合成 SHALL 独立于本地模型及播放器；调用方 MUST 显式选择 `heygen-voice-1`、`heygen-voice-1-turbo` 或 `heygen-voice-1-turbo-free`，缺失或未知模型 MUST 在网络请求前失败。客户端 MUST NOT 自动切换模型。

#### Scenario: 模型无效
- **WHEN** 调用方无法提供受支持的显式模型
- **THEN** 本地返回输入错误，不发送请求，也不采用服务端默认付费模型

#### Scenario: 免费档不可用
- **WHEN** Turbo Free 返回限流或不可用
- **THEN** 返回对应错误，不切换为付费档

### Requirement: 音色发现与身份保留
客户端 SHALL 提供公开/自有音色游标分页及按 ID 查询，支持公开目录的语言、性别、年龄和文本过滤；结果 MUST 保留远端 ID、原始状态和分页元数据，MUST NOT 自动保存为本地音色资产。

#### Scenario: 继续公开目录分页
- **WHEN** 宿主使用上一页游标和相同过滤条件请求下一页
- **THEN** 返回该页音色及后续游标，未知状态和服务扩展字段保持可读取

#### Scenario: 自有范围不支持过滤
- **WHEN** 宿主对自有范围提供仅公开目录支持的过滤条件
- **THEN** 本地返回输入错误，不静默丢弃过滤条件

#### Scenario: 远端音色不可用
- **WHEN** 详情查询或合成返回无权访问、未就绪或不存在
- **THEN** 返回明确服务错误，不更换音色或宣称已经可用

### Requirement: 显式音频格式与请求校验
合成 SHALL 使用远端音色 ID、1–5000 个 Unicode 标量值的非空白文本，以及显式 PCM16 或 MP3 格式。PCM16 SHALL 为单声道 little-endian 24 kHz；MP3 SHALL 为单声道 44.1 kHz、128 kbps。非法输入和零音频字节上限 MUST 在发送前失败；请求 MUST 禁用时间戳。

#### Scenario: 中文输入与格式
- **WHEN** 宿主提交有效中文文本、语言 `zh` 和 PCM16 格式
- **THEN** 按字符数量校验文本并请求原始 PCM16，而非 WAV/SSE；网络块不被描述为采样帧

#### Scenario: 非法输入
- **WHEN** 文本超过 5000 个 Unicode 标量值、音色 ID 为空或字节上限为零
- **THEN** 不发送合成请求并返回输入错误

### Requirement: 有界流与格式元数据
音频 SHALL 按需拉取消费并受调用方总字节上限约束，不预取整段音频。客户端 MUST 校验响应音频类型、采样率与声道信息，拒绝缺失或冲突的格式信息及 SSE/JSON 成功响应；返回元数据 SHALL 区分请求值和服务端确认值。

#### Scenario: 伪成功响应
- **WHEN** HTTP 200 返回 SSE、JSON、错误声道或冲突的采样率
- **THEN** 返回无效响应，不把响应体交付为音频

#### Scenario: 超过预算
- **WHEN** 后续音频块会使累计字节超过限额
- **THEN** 进入失败终态并释放响应，不交付越界块

### Requirement: 终态与取消
流 SHALL 区分活动、传输完成、失败和取消。空输出、可检测的 HTTP 截断或传输错误 MUST NOT 报告完成；取消/drop MUST 释放本地响应，不保证撤销远端工作或费用。正常 EOF 仅表示非空 HTTP 传输结束，MUST NOT 被描述为内容完整或实际播放完成。

#### Scenario: 连接中断
- **WHEN** HTTP 库检测到音频 body 截断
- **THEN** 返回传输错误，进入失败终态并释放响应

#### Scenario: 主动取消
- **WHEN** 宿主取消或丢弃活动音频流
- **THEN** 本地网络响应释放，后续取消流读取不能报告正常完成

### Requirement: 安全认证与错误
客户端 SHALL 使用调用方显式提供的密钥，只向配置的 HTTPS API 端点发送认证，允许 loopback HTTP 用于测试。MUST NOT 自动跟随重定向、重试合成或从 `.env` 读取凭据。认证、权限、限流、其他服务响应、传输和结果未知 SHALL 可区分；错误、Debug 和默认日志 MUST NOT 暴露密钥、正文或服务响应正文。

#### Scenario: 发送后结果未知
- **WHEN** 合成请求在收到响应头前断连或返回可能已处理的服务端错误
- **THEN** 返回结果未知，不自动重试或声称没有费用

#### Scenario: 敏感错误响应
- **WHEN** 服务端返回含凭据或正文的错误体
- **THEN** 客户端不输出该内容，错误体不会被无界读取

#### Scenario: 请求重定向
- **WHEN** 服务端要求重定向至另一地址
- **THEN** 不跟随跳转，也不向该地址发送密钥

### Requirement: Fish 兼容与验证边界
新增 provider MUST 保持现有 Fish 公共 API 和行为。普通测试 SHALL 只使用本地模拟服务；真实 API、音频可解码性、人工试听和性能 SHALL 分开记录，未执行的验证 MUST 明确标记。

#### Scenario: 无凭据验证
- **WHEN** 在没有云服务密钥的环境运行普通测试
- **THEN** Fish 与 HeyGen mock 测试可运行，且不访问真实服务或下载模型
