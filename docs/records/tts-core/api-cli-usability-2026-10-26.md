# API / CLI 使用面检查与接入技能

日期：2026-10-26。检查基线：`5818714`。范围为公开 facade、CLI 帮助与常见操作路径、
JSON Lines v7 文档，以及给使用方安装的技能；不修改模型生成算法。

## 判断与改动

Rust 保留三个主要入口：Engine（合成/可选播放）、VoiceStudio（独立音色资产）和
model_catalog（只读发现）。便利合成与 with_options 入口共享实现；不再增加会话框架、
另一套声音命令树或将 PCM 输出塞进播放协议。协议继续负责本地播放，不传 PCM。

本轮发现并处理的使用障碍：

- 接入方需要依赖 backends/启动 worker 才能发现模型：新增 facade `model_catalog(resources)`，
  返回 ModelInfo（能力与实际可用设备），CLI 增加 `models`、backend/model 筛选和 JSON。
  目录查询不下载或打开音频，空/未知选择有明确结果。
- 设计扩展的公开签名泄露底层 crate：facade 导出设计契约、设计数据与 DesignerHandle，
  也公开 design 命名空间中的 future 类型，普通接入无须直接依赖 backends。
- CLI 帮助缺少说明：补齐关键参数、voices 入口和示例；路径参数全局化，支持子命令后指定。
  design 名称可省略，保存结果到 stdout、进度到 stderr；不可用设备不再显示为空白。
  --protocol 与仅文件合成使用的 --param/--seed 明确冲突。
- 合法库音色在配置校验阶段被拒绝：共享只读库音色预检，保存库 id，在真正使用时物化。
  CLI、worker 配置/准备均采用该规则；校准使用解析后的模型音色。
- Nano 加载时冻结声音列表，看不到随后安装的参考：目录查询与生成前声音校验改为当前目录。
  首次参考编码仅按需准备 Nano codec 资源，不下载合成权重；仍用原有 CPU codec 和取消 owner。
- --plan 需要用户先维护另一套正确的默认模型/声音：计划只读一次，按自身身份选择模型和
  首个声音；显式身份冲突明确拒绝。计划在参考物化前先验证声音和原文覆盖。
- 文档仍有 v6、退休 MOSS Local、旧 crate 导入名和同步函数中的 await：同步为当前源码，
  VoiceStudio 示例补齐 LocalSet、显式一致的资源根、完成检查与关闭。
  明示工作区新 API 不等于已发布同版本包。

## 新技能

根目录 skills/ 已有 candle-performance，本轮保留不动，新增两个独立目录：

- talechime-integration：按 Rust / worker 选择参考，说明能力发现、数据归属、背压、
  取消/关闭、源文本身份与真正播放完成；worker 与 Rust 两种模式按需读 references。
- talechime-cli：实际帮助优先、模型与设计引擎区分、库音色、保存的默认值、数据路径、
  续播/从头、常见错误；明确 .voice 资产包不是生成 WAV。

两者均可整目录复制安装，不依赖仓库外绝对路径或另一个私有 skill；保持默认自动发现。
只包含 SKILL.md、必要的 references 与 agents/openai.yaml，没有占位目录或通用开发教程。

## 验证

- 两技能通过 skill-creator 的 quick_validate。
- 将 Rust 技能与库指南的 no_run 示例放到仓库外独立 Cargo 消费工程，以 path 依赖当前
  facade、关闭默认 feature、显式启用 Nano/播放，offline cargo check 全部编译通过。
  消费工程未继承仓库的 Candle patch；此结果仅代表所查 CPU 接口与依赖组合。
- 临时目录演练 models JSON、路径参数在子命令后、空库查询、可省略设计名称的输入校验，
  以及原音色库 CLI 验收脚本；无模型下载。
- 回归覆盖库音色配置、计划自动选择身份、只读目录、Nano 加载后的音色发现；最小 CLI
  三组合与扩展 CPU/ASR 测试、Clippy、rustdoc、格式与 diff 校验按项目门禁执行。

未执行真实模型下载、真实参考编码/试听或 CUDA/Metal 实机推理。codec 按需准备和后加载
音色的生产接线有源码/无模型回归验证，不把这些结果作为真实音频验收。
