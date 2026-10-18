# 生成参数与 seed 的协议及后端缝

## 范围

为暴露各后端模型的生成能力建立统一通道，本阶段（S1）交付协议与核心缝，不改变任何
模型行为：

- `talechime-protocol` 新增 `ParamValue`（untagged JSON 值）、`ParamKind`（类型与范围）
  与 `ParameterSpec`（名称/类型/默认/描述）。`Capabilities.parameters` 声明当前模型
  接受的参数目录；`PlanRequest.params` 与 `PlanRequest.seed` 携带宿主设定。均为
  `#[serde(default)]` 加法式字段，协议版本保持 7，不做兼容协商。
- `Backend` 的 `stream` / `stream_with_style` / `stream_with_context` 三方法链合并为
  单个 `stream(SegmentRequest)`；请求对象携带文字、音色、风格、接续上下文、本次尝试
  的具体 seed 与已校验参数。`SegmentRequest::reject_unsupported` 保持"不支持的
  风格/接续显式报错"语义。这是自定义 Backend 实现者的破坏性迁移。
- seed 上移为生产者策略 `SeedPolicy`：`Auto`（默认）每次合成尝试独立随机；
  `Pinned(u64)` 按 hash(seed, 合成序号, 重试次数) 派生，同计划可复现且重试仍变化。
  回读门禁重试因此获得新采样，不再以同参复现同一错读。
- `GenerationParams` 在会话启动（`start_input` / 直合成 `start_with_options`）按
  当前能力目录校验：未声明参数、类型不匹配、越界均显式拒绝，不静默忽略。
  `PlanSessionOptions` / `SynthesisOptions` 增加 `params` 与 `seed`；seek/restart
  沿用原执行参数。worker 在 `plan_input::generation` 完成线上字段到核心类型的转换。
- 各真实后端在本阶段只迁移签名；`parameters` 目录暂为空、模型内部 seed/采样值
  保持原状，参数消费与目录声明在下一阶段接入。

## 可重复验证

无模型测试覆盖：参数校验矩阵（未知/类型/范围，含空目录）、untagged 值线上形状、
协议 round-trip 与 `deny_unknown_fields` 不变、Pinned 复现与派生区分、门禁重试
换 seed 断言（同上下文不同 seed）、参数经 `SegmentRequest` 到达后端的记录断言。
17 个 Backend 实现点（6 真实 + 11 测试/示例）全部迁移到新签名；设备校准改用固定
seed 保持计时可比。

格式、workspace Clippy（扩展 CPU 特性与逐特性组合，warnings denied）、rustdoc
（warnings denied）、无模型 CLI／协议握手通过；未运行真模型，未改变任何生成输出。

## 已知边界与后续

- 本阶段后各后端生成结果与 seed 行为不变（仍为各自内部值）；`parameters` 目录为空
  时任何 params 都会被拒绝，这是有意的显式失败。
- Pinned 复现以"同一工具链、同一计划分段"为前提；后端时长学习可能改变分段边界。
- 下一阶段（S2）：四后端消费 `params`/`seed`、声明参数目录、类型化参数结构体与
  OmniVoice 原生语速路由；随后逐段风格开放（S3）与 CLI/文档收口（S4）。
