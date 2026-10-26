# 音色资产层重构（voice-studio）落地记录

日期：2026-10-26。OpenSpec change：`openspec/changes/voice-studio/`（含完整 proposal、
两个能力 spec、design 与 tasks）。本记录补充实施边界与验收状态，真模型试听与下载体积
验证按惯例单列，尚未执行。

## 落地内容

- **音色库（唯一事实源）**：`talechime-core::voices::library::VoiceLibrary`，位于
  `~/.talechime/voices/<id>/`（`voice.json` + `reference.wav`，staging+rename 原子写）。
  单文件导出为 zip 容器 `talechime/voice-bundle/1`（Stored，manifest 携带指纹校验，
  版本不符或指纹不符明确拒绝）。旧 `<resources>/voices/` 目录不扫描、不迁移、不删除。
- **设计引擎**：core `VoiceDesigner` 契约（输入约束先于权重校验；future drop 即取消），
  backends 静态登记表四个实现：`qwen-voicedesign`、`moss-voicedesign`（CUDA-only）、
  `voxcpm`、`omnivoice`（后两者复用已就绪朗读模型）。四个旧内核签名补了
  `seed: Option<u64>`（缺省 42，计算逻辑未动）。
- **物化**：只保留 `Registry::install_reference` 异步入口（MOSS Nano 走后台编码）。
  `registry/reference.rs` 负责缓存归属、完整性、替换与失败回滚；相邻 `.library`
  标记包含库 id、内容指纹与缓存摘要。未标记的同名用户音色拒绝替换。
- **facade**：`VoiceStudio`（自动层 = 显式层薄封装）+ `VoiceId` 新类型 +
  `ModelOptions.voices` 覆盖项；`Engine` 持 studio，`synthesize*`/`plan`/
  `single_voice_plan` 按顺序解析（内建 → `custom:` → 库 id）。库 `voice_import.rs`
  已删除（其三处重复 match 中的库侧一份由 `install_reference` 吸收）。
- **协议 v7 加法式**：`DesignVoice`（进度复用 `ModelProgress`，完成 `VoiceReady`）、
  `ListVoices`/`RemoveVoice`、`VoicesListed`。设计任务与模型准备共用单后台槽位互斥；
  `VoiceSnapshot` 持有音色别名，统一处理首次输入和后续 Append。
- **CLI**：`voices` 子命令树改为模型无关（design/import/list/engines/rename/remove/
  export/import-bundle），`--voice <库id>` 直读。

## 验收状态

- 普通测试全绿（不下载模型）：core 74；backends 各 feature 组合 71（qwen+moss+voxcpm+
  omnivoice）与 56（cosyvoice3+indextts）；talechime 默认与 `--features qwen` 全部；
  protocol crate 21。`scripts/voice_studio_acceptance.py` 的无模型部分
  （空库 list/engines/import/export/remove/import-bundle/rename、零下载断言）已通过。
- **真模型验证未执行**（需要下载设计引擎权重）：`voices design` 自动策略实际选择、
  设计→`--voice` 跨模型隐式物化听感、各引擎下载体积、协议 DesignVoice 全链路实机。
  用 `scripts/voice_studio_acceptance.py` 的手工步骤执行后把结果记回本目录。
- 原存量失败已修复，见下文「最小 feature 组合修复」。重新对拍确认：`cli moss`
  实际为两个 runtime 测试失败，裸 `cli` 为三个 protocol 测试失败；原记录少计一个 runtime 失败。

## 本轮审查与简化

以 `6bfa8e7` 为基线审查未提交实现，保留 core / backends / facade 分层。
规范轴发现重复同步/异步物化、Registry 承担缓存事务、静默吞错与设计线程未完整持有；
需求轴发现 Nano 隐式安装失败/缓存不校验指纹、旧数据覆盖风险、损坏缓存不可再生、
首次物化能力快照陈旧、Rust 增量 span 别名缺失，以及设计与播放互斥/取消竞态问题。
上述问题已在当前工作区修复；未引入旧格式转换或兼容路径。

- 删除同步物化副本，显式与隐式入口共用异步路径；Engine 的合成流和计划构造方法改为 async。
- 音色库读取与导出使用加锁、校验指纹的完整快照，WAV 校验复用 hound；删除先 rename 隔离。
- 设计线程先通知取消再 join，完成/取消均等到原生资源释放。
- worker 用最多 16 项的资源命令队列串行设计/准备/播放；CancelPrepare 与完成竞态若完成获胜，返回 VoiceReady。
- 自动选择仅在就绪主模型和专用设计引擎中选择；自动层隐藏原生诊断的显示文本，保留 error source。
- 进度通道关闭后禁用对应 select 分支，避免忙循环阻塞取消和其他 LocalSet 任务。

真实模型试听、GPU 资源释放时延和 CUDA/Metal 实机验证仍未执行。

## Standards 审查（原实现，5 项；已处理）

1. P1，硬约束：Qwen/OmniVoice 丢弃原生线程句柄，取消任务确认早于推理资源释放，违反 owner/取消生命周期约束。现由 `design/thread.rs` 关闭 receiver 后 join。
2. P1，硬约束：物化把未标记的同名目录当作过期缓存删除，违反旧用户数据保护约束。现拒绝未标记冲突，已标记缓存替换失败可回滚。
3. P2，硬约束：Nano 仅凭文件存在判定缓存有效，违反内容指纹约束。现文件型和目录型缓存共用指纹与载荷摘要。
4. P2，硬约束：失败的 Start 已提前更新 worker 别名，后续 Append 可能污染存活会话。现别名归属不可变计划快照。
5. P2，启发式：同步/异步物化重复代码与成组参数（Duplicated Code / Data Clumps）。现只有异步事务入口，借用 `ReferenceVoice` 聚合资产与 WAV。

## Spec 审查（原实现，11 项；已处理）

1. P1：设计/合成允许并行、冲突请求直接拒绝，不满足 design D6「排队而非并行」。现单个 BackgroundJob 枚举和有界资源命令队列维护互斥。
2. P1：取消可能隐藏已保存结果，不满足「取消与完成竞态时结果 MUST 二选一且明确」。完成获胜时返回 VoiceReady。
3. P1：只校验标记而不检查缓存载荷，不满足缓存损坏后「系统 SHALL 能随时重新物化」。现同时校验缓存摘要。
4. P1：Nano 隐式路径调用拒绝工作的同步安装器，不满足「自动物化与显式物化 MUST 走同一条内部路径」。现统一 async。
5. P1：Nano 缓存缺少指纹，不满足 D3「保证库更新后可检测过期并重新物化」。现通过相邻标记记录指纹。
6. P1：首次物化后仍用安装前能力快照校验，不能达到「该音色被物化到本次合成所用模型并用于朗读」。现快照纳入成功安装的音色。
7. P2：计划吞掉物化失败，不满足「错误指明限制」。现传递原错误。
8. P2：未就绪 VoxCPM 可抢占专用引擎，不满足「其次按设备可用性选择专用设计模型」。现按就绪主模型、就绪引擎、专用引擎选择，并筛选请求设备。
9. P2：直接递归删除记录、接受非 WAV、分次导出不一致，不满足原子删除与完整 WAV 资产要求。现 rename 隔离删除，hound 校验，加锁读取完整资产。
10. P1：Rust 多音色计划只改写允许音色而不处理后续库 id span，不满足「计划中的 span 以库音色 id 标注音色并启动会话」。现 core 统一归一化别名。
11. P1：关闭的进度通道反复立即就绪，可能阻塞 LocalSet 的取消与其他工作，不满足设计取消要求。现关闭该 select 分支，并有无模型超时回归。

## 本轮最终验证

- 六个 TTS 后端与 ASR 的 CPU feature 组合：相关四个 crate 的库/集成/示例测试通过。
- 最终生命周期调整后，facade 库与 worker 测试再次通过；补充无默认 feature 的库测试通过。
- Clippy（全部目标、warnings denied）、格式、rustdoc（workspace/私有项/示例、warnings denied）与 OpenSpec 校验通过。
- 默认 CLI 构建与隔离目录无模型验收脚本通过。

验证不包含真实权重推理、试听或 CUDA/Metal 实机验收；旧用户目录未执行迁移或清理。

## 最小 feature 组合修复（2026-10-26）

重新用 `git archive HEAD` 导出干净源码副本，在同一工具链上执行基线测试；
基线为 `6bfa8e7370f8c07c52170c3633d60bd59668f04d`，未包含任何 voice-studio 工作区修改。

- `--no-default-features --features "cli moss" --bin talechime runtime::`：两个失败，
  `device_change_releases_prepared_manager_and_preserves_checkpoint_files` 与
  `default_voice_update_does_not_replace_or_change_current_plan`。前者假设 MOSS CPU 可用，
  后者硬编码 Nano 音色 Weiguo；单独启用 moss 实际只有 GPU Realtime。
- `--no-default-features --features cli --test protocol`：三个失败，query/update、
  EOF/非法输入、未准备计划查询。启动时强制选择不存在的默认后端，握手前即退出。
- 因此确属存量问题，但此前「一个 runtime 测试」的计数不准确；「没人跑」不是由这些证据能证明的结论。

修复：测试从实际模型能力构造默认配置，用 Auto 测试设备切换，不再假设 Nano/CPU。
裸 CLI 的协议模式允许空模型目录；播放偏好通过同一配置事务独立校验数值和 revision，
模型/音色/风格/设备修改仍要求真实后端能力。准备未编译后端时返回结构化错误。
音色资产命令在合成默认值选择前分流，不读取或修改不相关的合成配置。

新增无后端播放偏好、非法 patch 不落盘、音色命令隔离损坏/不可用合成配置的回归。
CI 增加 cli、cli+moss、cli+moss-nano-candle 三组完整库/集成/示例测试，均不下载模型。

验证结果：三个最小组合的完整库/集成/示例测试与 Clippy（all-targets、warnings denied）均通过；
core 的配置事务回归、六个 TTS 后端＋ASR 的 CPU 组合回归、扩展组合 Clippy、rustdoc、格式与 diff 检查均通过。
同时修正了无设计引擎构建的 manifest helper feature gate，以及无模型构建下目录缓存变体的精确 lint 预期。
