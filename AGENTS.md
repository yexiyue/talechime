# Talechime repository guidance

- Read README.md, docs/architecture.md and docs/development.md before changing behavior. Use current source over historical docs/records.
- Keep Talechime independent of TRNovel and CastGlean business types. Hosts own chapter acquisition, role analysis and voice bindings.
- JSON Lines is v7 with one plan-based start path; do not add compatibility negotiation or old wire commands. Alignment remains removed. The only executable is talechime. Paths are centralized in talechime-core/src/paths.rs: ~/.talechime/config.json, checkpoints/, resources/. Only current data formats are supported. Do not scan, copy, convert or delete historical user directories. Unsupported data must fail without overwriting it. Release configuration lives in dist-workspace.toml (cargo-dist 0.32.0); regenerate the release workflow. Custom CI scripts live in .github/scripts.
- User-approved exception (2026-11-11): Kokoro may bundle the fixed eSpeak NG standalone helper and data with its GPL notices and corresponding source. Talechime must not link the GPL library; helper execution is bounded and missing/corrupt resources fail explicitly.
- Use UTF-8 byte ranges bound to immutable source hashes. Generation completion must never substitute for actual playback completion.
- Keep sessions on Tokio LocalSet; inference and audio objects stay with their owners. Cancellation and bounded stream ownership are part of the public behavior.
- Keep Candle versions unified, ORT pinned and Windows CRT dynamic. Read relevant docs/knowledge and docs/records before changing native dependencies.
- Ordinary tests must not download models. Record real-model numeric, audio and performance verification separately.
- Crates use named module files, explicit feature gates and workspace dependencies. Avoid unrelated inference refactors.
- Use Mermaid for diagrams. Distinguish current capabilities, experiments and roadmap in docs.
- Preserve third-party licenses, pinned source revisions and model manifests. Never commit models, credentials, user audio or novels.
- Run formatting, relevant tests, Clippy and rustdoc. CI separates CPU checks from native CUDA/Metal compilation.

## 开发与知识积累工作流

- 开始任务时先读 [知识库索引](docs/knowledge/README.md)，按任务选择相关指南、规划、决策与历史验收，再核对当前源码、锁定依赖和所属规格。历史设备数据、旧命令或旧模型策略不代表当前实现已验证。
- 持续迭代的规划、决策和踩坑经验统一放在 `docs/knowledge/`；新条目按需使用 `plans/`、`decisions/`、`lessons/`。既有专题指南保留原路径，通过索引关联，不为统一目录搬迁历史文档或复制同一份事实。
- 影响公开 API、JSON Lines 协议、资源格式、运行生命周期或跨 crate 边界的工作先落实到对应 OpenSpec。规划明确区分候选、已决定、已实现和已验证；知识笔记不能代替规格、任务勾选或用户授权，也不把宿主的章节分析和角色绑定引入引擎。
- 遇到重要取舍、非直观约束或反复出现的问题，在相关改动中记录日期、适用版本/平台、触发条件、原因、修复方式、验证证据及未验证边界，并更新索引。原生依赖注明来源 revision 和 feature 组合；性能结论注明实际设备、执行路径、输入/权重与测量口径，数值、音频/人工试听和性能分别记录。优先链接源码、测试和整理后的报告，不提交模型、凭据、用户音频、小说或整段运行日志。
- `docs/records/` 按[归档规则](docs/records/README.md)保留特定日期、版本和设备的验收、实验及阶段规划快照；不覆盖旧数据、抹去失败实验或将旧边界改写成新验收。新证据通过新记录或注明日期的补充说明关联旧结论，现行指南另行更新。
- 经验证且可跨任务复用的做法，整理到 `docs/development.md`、`docs/architecture.md` 或对应后端指南；只有长期有效、必须遵守的短约束才提升到 `AGENTS.md`。具体案例与依据留在知识库或验收台账，避免规范无限膨胀。
- 完成改动时同步检查 OpenSpec 任务、当前能力文档、知识库索引及必要验收记录；依赖版本、实现或设备变化时重新核验相关结论。普通测试、编译、真模型验证、宿主集成、SDK/CLI 发行是不同证据，不能互相冒充完成。
