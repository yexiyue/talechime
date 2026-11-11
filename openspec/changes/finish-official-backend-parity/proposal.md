# Proposal

## Why

当前固定模型的官方能力仍有移植和验收缺口；实际听验已经发现混排英文漏读、年份误读和音色异常，PCM 与 EOS 成功无法证明朗读正确。需要将剩余工作收敛为逐后端、逐能力、逐用例的完成标准，避免把接通参数或生成样本当作完整交付。

## What Changes

- 补齐现有固定版本的官方推理能力：IndexTTS 中英 TN、日语 UniDic 前端、西语 TN、阿语路由、读音控制及 QwenEmotion；CosyVoice3 模式和文本处理；Qwen、OmniVoice、MOSS、VoxCPM、Kokoro 的已识别差异与实机验证。
- 按 IndexTTS 固定上游移植日语默认路径：日语 TN 透传、UniDic 分词、空格拼接、保留原字、读音标注；固定词典与下载校验。
- 以完整公开文本矩阵验证七类后端及已开放模型变体，分别记录官方对照、预处理、分词、真实生成和人工听验。
- 保持模型适配与宿主业务边界，文本矩阵不成为统一预处理器。中文年份的产品读法策略单列，不能冒充官方行为或误改日文日期。
- 用户最新优先级：先完成剩余官方能力及矩阵；Nano 漏读、年份不连贯和音色异常保留失败记录，后续处理，不以拼接实验宣称修复。

## Capabilities

### New Capabilities

- `official-backend-inference`：固定模型推理能力、资源与公开能力声明的一致性。
- `indextts-japanese-frontend`：固定版本日语分词、标注及资源处理。
- `backend-text-validation`：跨后端公开文本矩阵、失败分类与听验交付。

### Modified Capabilities

无。当前主规格库存为空；既有变更的 delta 不是已归档主规格。

## Impact

实现主要位于 TaleChime 的模型 crate、`crates/talechime-backends`、资源配方及 `tools/tts`。可能新增 Rust 分词/TN 依赖与词典资源，必须保留许可证及固定来源。CastGlean 的原文传递与恢复兼容继续由其 `backend-owned-speech-text` 变更负责，本提案不授权修改宿主业务代码。

本变更是 `complete-backend-text-parity` 的后续收敛提案：复用其已有工作和证据，不删除或重做旧修改；应用阶段先迁移剩余任务及规格归属，避免两个任务集重复宣称完成。新模型家族、训练、加速栈移植、云部署和 SDK 发布不在范围内。


2026-11-12 用户明确调整设备验收范围：先完成本机；CUDA 实机另行安排。
本轮完成本机适用实现与矩阵，原始 BF16 的 CUDA 行保留未验证，后续门禁见
[deferred-cuda.md](deferred-cuda.md)。此调整不将 CUDA、Windows 编译或数值对照改写为通过。
