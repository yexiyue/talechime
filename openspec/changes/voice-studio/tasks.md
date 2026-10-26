# Tasks

## 1. core：音色库与设计契约

- [x] 1.1 `paths.rs` 新增 `voices()`；实现 `voices/library.rs::VoiceLibrary`（save/get/list/rename/remove，staging+rename 原子写、目录锁沿用现有模式）；单测覆盖 id 字符集与长度、冲突拒绝不覆盖、remove 原子性、路径穿越拒绝
- [x] 1.2 定义 `VoiceDesigner` 契约与 `EngineInfo`（设备、输入约束、额外下载字节数、就绪探测）、`DesignRequest`、`DesignOutput`（WAV 字节 + 转写 + 描述）；纯类型编译验证，无行为
- [x] 1.3 实现导出包格式（zip 容器：`voice.json` + `reference.wav` + 版本字段，读取端版本不符明确报错）与 `export`/`import_bundle`；roundtrip 单测（导出→删库→导入→字节一致）

## 2. backends：设计引擎与物化

- [x] 2.1 包装 qwen 设计内核为 `VoiceDesigner` 实现（`qwen-voicedesign`），输入约束上提进 `EngineInfo`；单测：空描述/超长文本在 prepare 之前被拒（不触下载）
- [x] 2.2 包装 MOSS 设计内核（`moss-voicedesign`，CUDA-only 进 `EngineInfo.devices`）；单测：非 CUDA 设备请求明确报错
- [x] 2.3 包装 VoxCPM 与 OmniVoice 主模型描述能力为引擎（各自现有校验上提）；单测：就绪探测对已转换/未准备目录给出正确状态
- [x] 2.4 新增 `DESIGNERS` 静态登记表与 `Registry::designers()`；单测：feature 组合下清单正确、id 唯一、就绪态与代价字段齐全
- [x] 2.5 实现 `Registry::install_reference`（吸收 `voice_import.rs` 的 match 臂为 adapter 方法，补齐 cosyvoice3 与 MOSS 两家），物化键含库音色内容指纹；单测：物化成功/重复物化幂等/WAV 变更后指纹判过期、时长超限明确报错

## 3. facade：VoiceStudio 与合成入口

- [x] 3.1 新增 `voice_studio.rs`：`VoiceId` 新类型、`VoiceStudio`（design/design_with/engines/install_voice/list/rename/remove/export/import_bundle）、自动选择策略函数、自动层错误不泄露引擎内部名；单测：自动层 ≡ 显式层(默认策略) 组合、slug 自动生成与查重、错误文案检查
- [x] 3.2 合成入口音色解析顺序（内置音色 → `custom:` → 库 id）与隐式物化/过期重物化；用确定性 fixture 后端的库级集成测试验证（不下载模型）
- [x] 3.3 `Engine::voices()` 返回绑定句柄；`voices.rs`（CLI）暂保持编译可用；编译 + 现有引擎测试全绿
- [x] 3.4 删除 `voice_import.rs`，公开导出面切换为 VoiceStudio；全 workspace 编译 + 受影响测试更新

## 4. 协议：DesignVoice

- [x] 4.1 `talechime-protocol` 新增 `Command::DesignVoice` 与 `Event::VoiceReady`（serde 单测：新旧消息互不破坏，旧消息仍可解析）
- [x] 4.2 worker 接线：设计任务与合成互斥排队、`ModelProgress` 转发、future 取消；协议集成测试用 fake designer 验证 命令→进度→VoiceReady 全流程与取消路径

## 5. CLI：模型无关的音色命令

- [x] 5.1 重写 `voices.rs` 子命令树（design/import/list/rename/remove/export/import-bundle/engines），全部走 VoiceStudio、不依赖 `--backend`；`--voice <库id>` 接入朗读入口；命令级集成测试（tempdir 库 + fake 引擎）
- [x] 5.2 手动验收脚本化清单：无 `--backend` 时 design/list/engines 可用、`--voice` 朗读隐式物化；真模型听感与下载体积验证按仓库惯例单独立记录于 docs/records（不进 CI）

## 6. 收尾：文档与全量门禁

- [x] 6.1 更新 architecture.md（音色分层图与旧目录保留语义）、library.md（VoiceStudio 两层 API 示例，示例可照抄运行）、README 音色章节与 CLI 帮助
- [x] 6.2 全量门禁：cargo fmt、clippy、rustdoc、`cargo test -p talechime-core -p talechime-backends -p talechime-protocol -p talechime`（含 qwen/moss/voxcpm/omnivoice/cosyvoice3/indextts feature 组合）、`openspec validate voice-studio`
