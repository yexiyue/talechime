> 2026-10-27：本实验计划已终止，由 `official-reference-synthesis` 替代。未执行任务保持未完成；不再推进滚动、双参考或重锚候选。以下保留历史实验契约。

# Spec Delta

## Purpose

为整篇小说单音色朗读提供可复现的模型策略对照与用户试听决策，分别评价身份稳定、韵律起伏、句间承接和长期漂移。实验结果允许失败或不适用，不将单句自然、自动指标或生成完成冒充整章听感通过。

## ADDED Requirements

### Requirement: Immutable local corpus identity
实验系统 SHALL 只读使用用户授权的本地小说，并记录来源摘要、章范围和选段 UTF-8 字节范围；正文、转写与音频 MUST 仅存于本地忽略目录，且不得修改宿主正文或阅读状态。

#### Scenario: Use the authorized TRNovel novel
- **WHEN** 从 TRNovel 的《蛊真人》准备选段或整章
- **THEN** 系统核验当前源文件和章节快照摘要，保留标点、空白和段落，记录派生关系而不写入 TRNovel

#### Scenario: Existing corpus no longer matches
- **WHEN** 旧清单引用缺失文件或摘要不符
- **THEN** 系统阻止该语料运行并报告原因，不静默采用历史范围或覆盖源文件

### Requirement: Model-specific controlled comparisons
实验系统 SHALL 按 backend、model 和 device 分别比较策略，区分固定原始参考、冻结生成锚点、滚动接续及独立身份双参考。系统 MUST 隔离接续与组段对照，并记录实际生效条件；不支持的组合 SHALL 显示原因而非降级冒充成功。

#### Scenario: Separate continuation from chunking effects
- **WHEN** 比较滚动与固定参考
- **THEN** 两组使用相同正文和固定分块及其余生成条件；另以固定参考比较不同组段，明确分块与 seed 派生差异

#### Scenario: Nano rolling continuation is unsuitable
- **WHEN** Nano 滚动组出现用户不接受的漂移或节奏问题
- **THEN** 允许淘汰该组并比较固定参考或其他受支持策略，不要求所有模型共享同一接续默认

#### Scenario: Backend lacks separate identity and continuation inputs
- **WHEN** 模型不能独立接收身份参考与接续上下文
- **THEN** 双参考组合标记为不支持，不通过拼接目标文本或套用其他模型格式伪造能力

### Requirement: Explicit listening decisions gate dependent stages
实验系统 SHALL 为每个模型提供可直接试听的匿名对照、策略映射和上下文摘录，并记录用户反馈所对应的具体产物。进入依赖该结果的下一阶段 MUST 有明确的用户选择或淘汰决定；沉默、超时和自动指标不得视为接受。

#### Scenario: User has not reviewed the first batch
- **WHEN** 短样批次已交付但尚未收到用户听感反馈
- **THEN** 该批次保持 awaiting_feedback，不启动依赖其胜出策略的细化或长篇生成，可继续不依赖该决定的工具验证

#### Scenario: Every candidate sounds poor
- **WHEN** 用户否定某模型全部候选
- **THEN** 系统记录淘汰理由，允许该模型终止实验或经用户决定追加明确候选，不强选胜者

### Requirement: Long-form conclusions require long-form evidence
模型策略的长篇听感推荐 SHALL 绑定完整章节、不同叙事情境的复核和用户持续聆听反馈。系统 MUST 分开记录短样偏好、整章确认及跨 seed、设备复现范围，允许有条件推荐、失败、阻塞和待验证状态。

#### Scenario: Short excerpt passes but chapter drifts
- **WHEN** 用户喜欢短样但整章出现累积加速、音色变化或反复收尾
- **THEN** 策略不得获得长篇推荐，保留短样结果并记录整章失败

#### Scenario: Result verified only on one device
- **WHEN** 策略只在 M4 Pro Metal 上完成实验
- **THEN** 推荐范围明确限定该设备与构建，不自动声明 CUDA 或 CPU 已验证

### Requirement: Complete evidence without automatic quality claims
每次运行 SHALL 记录构建与模型身份、参考摘要、参数、seed、实际分块、接续重置和原始音频摘要，保留失败与取消证据。内容检查、语速、停顿、响度或声纹指标 SHALL 仅作为诊断，比较样本不得被隐式重试或后处理替换。

#### Scenario: Readback flags a suspicious segment
- **WHEN** ASR 标记疑似漏读、重复或截断
- **THEN** 系统保留原音频与报告，标记该样本需要内容复核，不把疑点自动定为 TTS 错误，也不因声纹稳定判为通过

#### Scenario: Run is cancelled
- **WHEN** 用户停止长篇生成
- **THEN** 系统保留已完成与不完整产物的区分并等待所属推理资源清理，不把部分生成写成整章成功或实际播放完成

### Requirement: Experimental isolation
实验能力 MUST 显式启用且不修改生产默认、用户配置或稳定协议。普通测试 SHALL 不下载模型或读取私有小说；真实权重准备与推理 SHALL 作为独立实验执行并报告。

#### Scenario: Experimental strategy receives positive feedback
- **WHEN** 用户确认某个实验策略听感更好
- **THEN** 系统只更新该模型的实验结论，不自动启用生产默认或改变公共协议

#### Scenario: Ordinary fixture validation
- **WHEN** 运行普通回归测试
- **THEN** 使用合成文本和确定性音频夹具验证对照、决策和生命周期，不下载真实模型或访问 TRNovel 数据
