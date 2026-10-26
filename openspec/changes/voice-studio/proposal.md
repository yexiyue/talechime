# Proposal

## Why

音色创建目前只存在于 CLI（`voices design` 无法从库调用），同一套「装音色」逻辑在 CLI `voices.rs`、库 `voice_import.rs` 和四个后端 `design.rs` 三处重复且已经漂移（库导入不支持 cosyvoice3/moss）；音色身份绑定 `(backend, model, revision)` 并存放在模型资源目录里，换模型即失效。talechime 作为库供 CastGlean 及外部宿主集成，需要一个简洁、模型无关的音色设计与管理 API，且设计出的音色可跨模型复用。

## What Changes

- 新增核心层统一音色库 `VoiceLibrary`：音色资产（WAV + 转写 + 描述 + 来源）存于 `~/.talechime/voices/`，是唯一事实源；现有 per-model `VoiceStore` 降级为可再生缓存，可随时从库重新物化。
- 新增 `VoiceDesigner` 抽象：把 qwen/MOSS/VoxCPM/OmniVoice 四个现有设计内核包装成可插拔引擎（复刻 Backend/Registry 模式），设计不再要求先准备合成模型。
- 新增库门面 `VoiceStudio`：自动层（`design(description)` 自动选引擎、合成时隐式物化）与显式层（`designers()`/`design_with`/`install_voice`/`export`）两层 API，共享同一条内部路径。
- 合成入口接受库音色 id：`Engine::synthesize` 与计划 `SpeechSpan.voice` 直接引用 `VoiceId`，会话启动时隐式物化并校验各模型限制。
- JSON Lines v7 加法式扩展：新增 `DesignVoice` 命令与 `VoiceReady` 事件，引擎准备复用现有 `ModelProgress` 事件流。
- **BREAKING** CLI `voices` 子命令树围绕音色库重设计，与当前选中后端解耦；库 `import_reference_voice`/`list_voices` 由 VoiceStudio 取代。
- **BREAKING** 不迁移现有 per-model 自定义音色：旧目录原地保留、不扫描不转换，音色库从空开始。
- 设计引擎自动选择策略：优先复用已就绪主模型的描述能力，其次专用设计模型（按设备可用性排序）。

## Capabilities

### New Capabilities

- `voice-library`: 模型无关的音色资产能力——音色的存储、生命周期（导入/设计产出/重命名/删除/清单）、单文件导出与导入、跨模型物化（含隐式物化与各模型限制校验），以及库音色 id 在合成入口（Rust API、计划 span、CLI `--voice`）的解析。
- `voice-design`: 音色设计引擎能力——`VoiceDesigner` 抽象与引擎清单（能力/代价/就绪状态）、自动与显式两种选择方式、设计请求约束（文本/描述限制、设备、seed）与取消语义。

### Modified Capabilities

无。当前 spec 目录为空，这是首个能力 spec。

## Impact

- `crates/talechime-core`：新增音色库模块与 `~/.talechime/voices` 路径（paths.rs）；新增 `VoiceDesigner` 契约。
- `crates/talechime-backends`：四个设计内核包装为引擎实现；Registry 增加设计引擎登记；per-model `VoiceStore` 语义调整为缓存。
- `crates/talechime`：新增 `VoiceStudio` 门面与 `VoiceId` 新类型；Engine 物化路径；删除/替换 `voice_import.rs`；重写 `voices.rs` CLI。
- `crates/talechime-protocol`：`DesignVoice` 命令、`VoiceReady` 事件（v7 加法式）。
- 文档：architecture.md、library.md、README 音色章节重写；旧 per-model 音色目录说明为保留不动。
- CastGlean 集成缝：`VoiceId` 成为角色↔音色绑定的稳定接口（对应既有记录中「宿主编译到 VoiceSpans」的未写胶水）。
- 磁盘：新增 `~/.talechime/voices/`；现有 `<resources>/voices/` 目录不做任何改动。
