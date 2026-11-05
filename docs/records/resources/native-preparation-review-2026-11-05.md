# 原生资源准备审查与精简（2026-11-05）

审查范围为两个工作区的未提交变更，基点分别是 CastGlean `1776ff5fd9c541974f0b1e71474c0af49a81fb16` 和 TaleChime `4ed0c6bfb77f6d2d3a864d38665539e332854d58`。两仓库基点至 HEAD 没有额外提交，使用 `git diff <基点>` 加未跟踪源文件审查，未用空的三点比较遗漏工作区内容。规范与需求由两个独立审查代理检查，随后定向复审。

## Standards

1. **P2，旧清单改写**：旧 CosyVoice3 HiFT 清单原为 96,324,868 字节、摘要 `475ac02be5d85f5b13da1e075a217ddb71e14f109f327919609004858c75bd2b`，曾被替换成新序列化产物。根据“保留固定模型清单”的规范应区分历史与当前配方。按用户后续明确的“不考虑兼容性、清理旧实现”授权，删除已无生产引用的旧 CosyVoice3/IndexTTS 清单，当前来源只使用新 native/distribution 配方；原始基线保留在固定 Git 对象与本轮核验记录，不冒充新的数值对照。
2. **P3，判断性精简**：必然存在的确认指纹从 Option 改成必填 String，贯穿宿主 DTO、IPC、owner 请求与执行；只保留真正可能缺失的已计算计划。绑定重新生成，没有空指纹降级。
3. **P3，判断性精简**：两个独立 Option 曾隐含 Verify/Download/Native 三类任务，验证身份依赖“没有下载客户端”。改成 TaskKind 枚举，集中停止策略并排除无效组合；移除重复的缓存活动 guard，没有增加 trait 或兼容框架。

旧清单原文可用 `git show <固定引用>` 重现：
- `crates/talechime-backends/src/cosyvoice3/resources.json`：原文固定在 `4ed0c6bfb77f6d2d3a864d38665539e332854d58:crates/talechime-backends/src/cosyvoice3/resources.json`，完整文件 SHA-256 为 `b3a3effc3f275ae7cca7b203616c04b74b0b66a61a0df4d1c95dcb77576bf4b5`。
- `crates/talechime-backends/src/indextts/conversion-manifest.json`：原文固定在 `4ed0c6bfb77f6d2d3a864d38665539e332854d58:crates/talechime-backends/src/indextts/conversion-manifest.json`，完整文件 SHA-256 为 `086af241d068ac83aaca7867835cbee84ec16700986a77bc27a28ab6337bd26d`。

## Spec

1. **P2，暂停覆盖发布成功**：规范要求“失败和取消 MUST 保留已发布的可用版本”。暂停原先保存旧快照，在等待已经提交的 worker 退出后又覆盖 Ready。现在先等待退出，再读取发布事实；已发布时保持 Ready。阻塞 worker 回归同时证明盘点发生在退出之后。
2. **P2，等大输入的缓存交换**：规范要求“计划或缓存配置变化后 SHALL 重新确认”。原指纹只绑定缺失总字节数；交换两个等大文件的命中状态会沿用旧确认下载另一文件。现在绑定排序后的具体缺失输入身份；本地 HTTP 回归确认旧指纹失败且没有任何请求/准备工作区。完整发布版本不依赖原始输入缓存，故 Ready 计划清空该无关集合，保留真实离线路径。
3. **P2，空间反馈被清除**：规范要求空间不足时说明需求。确认框关闭后 effect 会清掉刚由 InsufficientSpace 设置的提示。现在仅在打开新确认时重置错误，保留执行结果反馈；通过控制流检查和前端构建验证，没有填满用户磁盘做破坏性测试。

## 精简与验证边界

下载进度的多层条件表达式拆成显式的估算分支，统一零字节格式；组件使用明确的 Props 与返回类型，未改变阶段、语言文本或估算优先级。规范复审与需求复审均无剩余阻塞发现。

新回归为 `pause_after_publication_waits_for_exit_and_keeps_ready`、`equal_size_cache_swaps_require_new_confirmation_before_any_download`；原取消等待、共享租约、损坏文件、原子发布、源不可用与就绪离线测试继续通过。普通测试不下载模型。模型输出、资源摘要和推理代码未随此次审查调整；先前 8 个 CPU/Metal 真实合成、Tauri 波形播放证据继续作为独立验收记录，CosyVoice3 原数值差异与人工听感待评估保持原结论。

审查结果：Standards 3 项（1 项明确问题、2 项精简建议），Spec 3 项；最高分别为旧清单基线处理 P2 与暂停/发布竞态 P2，均已处理。
