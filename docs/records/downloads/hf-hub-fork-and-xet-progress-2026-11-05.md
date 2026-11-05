# hf-hub fork 迁移与 Xet 网络进度

2026-11-05，macOS。HTTP 行为沿用已验收实现；本次抽出依赖维护仓库，并单独向上游提交 Xet 进度补丁。

## 固定来源

- 维护分支：[yexiyue/hf-hub:codex/resumable-http-downloads](https://github.com/yexiyue/hf-hub/tree/codex/resumable-http-downloads)。固定 revision `b0de8365ed72d88fb8b7534b9e7561f87d6db49d`。
- HTTP 补丁仍基于官方 1.0.0 的 `dcf865adacf7cbb750038d6521deb16c670ffd42`。源码与 Apache-2.0 许可移入 fork；TaleChime 删除 vendor，两个消费者的 lockfile 只新增 hf-hub 的 Git 来源，没有升级其他包。
- 独立上游 [PR #240](https://github.com/huggingface/hf-hub/pull/240) 基于 main `473de9430b0fbc544fb1d91102180eb713f8dc87`，head `2ff4b976fca32bb786f027e1d329507abf491bf3`，不包含 HTTP 分片或镜像补丁。修复也已回移到维护分支。

## 修复与真实验证

`DownloadEvent::AggregateProgress` 增加网络累计字节、当前已发现的网络总量和网络速率，命名与现有上传事件一致。公共事件使用通用标量，私有 Xet 映射函数适配底层报告；CLI、示例和 Wasm 桥各自负责展示或转发。成功返回前停止并等待 poller，再补发最终快照。网络完成不会替代逻辑重建完成。

真实公开文件：`OpenMOSS-Team/MOSS-TTS-Realtime` revision `75682787d8e2fcc73faca37ba2931453ca9c4022` 的 `model.safetensors`。隔离临时缓存，保留默认 Xet 参数，90 秒 / 8 MiB 网络字节预算，达到预算后停止。

9.251 秒时观察到 10,139,690 网络字节、0 重建字节，50 次聚合报告；首次观察到网络进度约 7.501 秒。预算轮询有超额。这直接验证此前丢失的网络活动现在可见；不证明完整权重、推理或稳定吞吐。

## 上游补丁检查

Rust 1.98.1，nightly rustfmt：217 个库单元测试、2 个 future-size 测试、22 个 doctest、12 个 CLI 进度测试通过；全工作区 default/all-features Clippy `-D warnings`、严格 rustdoc、Wasm library/smoke/tests 编译与格式检查通过。

上游原有 `retry::tests::connect_error_is_transient` 在本机收到 localhost HTTP 502，未得到其预期的连接错误；未修改的 main 也单独复现。上述库套件排除该一项，不计为通过。Rust 1.99 的 Clippy 对上游 `repository/diff.rs` 另有原有借用告警，因此记录实际通过的工具链版本，不增加全局忽略。

PR 已提交且可合并；首次核对上游 Rust workflow 为 `action_required`、没有运行作业，尚无上游 CI 通过结论。

## 应用边界

固定 Git 来源后，TaleChime 27 个后端单元、5 个本地 Hub 集成、5 个分片集成测试通过；CastGlean 38 个宿主测试通过，原有真实下载手工测试保持 ignored。两仓相关 Clippy `-D warnings` 通过。

该次迁移时 TaleChime 和 CastGlean 继续采用 HTTP。后续已经加入 Auto 与基于真实偏移的 Xet 有效断点，见[后续验证](xet-auto-fallback-2026-11-05.md)，不按旧稀疏文件长度推断进度。SDK 发布限制见[维护说明](../../knowledge/hf-hub-fork.md)；未发布 SDK，未验收完整生产模型或 Windows 新行为。
