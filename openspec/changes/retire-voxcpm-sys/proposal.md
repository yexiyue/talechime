# Proposal

## Why

生产 VoxCPM 已使用独立 Candle 实现，`voxcpm-sys` 只保留为开发用 C++ 性能对照。继续将其纳入 workspace 会维护不再用于产品的 native 源码和构建链。

## What Changes

- 删除 `crates/voxcpm-sys`，移除 workspace member 和仅它使用的根级 cmake/cc 声明，更新 Cargo.lock。
- 清理 CI 路径过滤、当前架构、开发、许可与基准说明中的失效入口。
- 将原生 C++ 对照标为已退役；保留历史测量、固定来源 revision 和仍分发组件的许可信息。
- 保持生产 Candle 功能、模型资源、feature、缓存身份及验收状态不变。

## Capabilities

### New Capabilities

无。本次属于开发工具退役和文档清理，设置 `skip_specs: true`。

### Modified Capabilities

无。没有产品行为或公共 API 变更。

## Impact

涉及根 Cargo.toml、Cargo.lock、crates/voxcpm-sys、.github/workflows/qwen-cuda.yml、crates/voxcpm/README.md、crates/talechime-backends/README.md、docs/architecture.md、docs/development.md、docs/licenses.md 和 docs/knowledge/toolchain.md。
开发者不能再从当前 checkout 构建原生对照探针，可从删除前 Git revision 恢复其源码与许可证。严格官方数值、听感和长期播放验收缺口不因本次清理而关闭。
