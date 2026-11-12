# Hub 资源完整性摘要

2026-11-12，`use-hf-cache-directly` 实施中：旧的远程资源清单只固定仓库、commit、文件名和大小，同大小损坏无法被离线盘点识别。本次为 MOSS、Qwen、VoxCPM、OmniVoice 和 ASR 的现有清单补充 SHA-256，供后续统一盘点使用；当前加载和下载流程尚未全部使用这些摘要。

摘要依据各清单固定 commit 的 Hub tree 元数据。LFS 文件采用该元数据中的内容 SHA-256；普通 Git 文件先校验 Git blob 对象 ID 和大小，再计算内容 SHA-256。元数据和缺失小文件通过 Hub 镜像读取；缓存中的普通文件也先经 blob ID 校验。本次核对覆盖 15 个仓库 revision、81 个不同文件，不代表重新下载或运行了所有模型，也不代表实现已支持直接缓存加载。

成品清单本来已有 SHA-256，保持既定摘要。相关实施和未完成验收见[变更任务](../../../openspec/changes/use-hf-cache-directly/tasks.md)。
