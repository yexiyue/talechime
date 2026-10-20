# Design

## Context

动机见 proposal.md。当前生产适配器 runtime 使用 `voxcpm::{Model, Options, Outcome}`；正常依赖树不含 voxcpm-sys。后者只有独立 example，且 dev-dependency 反向引用 talechime-backends。cmake/cc 的 workspace 直接声明仅此 crate 使用，但其他依赖仍可能传递依赖它们。

## Goals / Non-Goals

**Goals:** 退役开发基准，清除当前构建及文档中的悬空入口，并保留来源与许可审计线索。

**Non-Goals:** 不改变推理算法、CUDA/Metal feature、模型格式、资源迁移、音色缓存或产品行为；不将尚未通过的完整迁移验收改成通过。

## Decisions

1. 从工作树删除原生 crate，而非改成 excluded 目录。保留目录仍需维护 vendored native 源码，不能达到清理目的；完整源码可经 Git 历史恢复。
2. 删除前把 native/SOURCE.md 和适用的第三方许可证、声明归档到 `docs/legal/retired-voxcpm-native/`，记录删除前可恢复的 Git commit。保留固定上游 revision；不复制模型或音频。当前 Candle 来源以 crates/voxcpm/SOURCE.md 为准，修正后端 README 中过时的 MIT 概括。
3. 历史 docs/records 保留原始结果和命令语境，通过当前说明链接退役来源。知识库中涉及旧 native 链路的段落标注历史，不把旧结论改写为当前事实。
4. 仅移除自有 cmake/cc 声明和失效的锁文件条目，使用 Cargo 解析更新而非手工删传递依赖。CI 保留其他组件仍需的 native 构建工具；不借机修改生成的 release workflow。
5. 保留官方 Python oracle、Candle eager 和现有探针作为后续验证工具。C++ 性能对照退役不是数值、人工听感或 30 分钟播放门槛完成的证据。

## Risks / Trade-offs

- [原生基准无法从当前 checkout 直接重建] → 归档说明提供恢复 commit 和旧构建入口，历史记录保持完整。
- [来源链接与许可证失效] → 删除前归档固定来源、适用许可及声明；扫描当前文档和 notices 脚本的路径引用。
- [误删仍被其他组件使用的工具依赖] → 核查正常及构建依赖树，仅清理不再需要的声明。
- [完整 workspace 检查受其他模型工具链影响] → 记录具体失败与范围，另行验证 Vox CPU 产品组合，不以历史测试代替本次检查。

## Migration Plan

先归档来源与许可，再删除 crate 和构建入口，更新当前文档及 Cargo.lock，最后执行格式、相关测试、Clippy 和 rustdoc。无用户数据迁移；回滚通过恢复同一变更的源码、清单和锁文件完成。
