# Design

## Context

动机见 proposal.md。现有 `continuation_probe` 可复用生产适配器与分段、输出 WAV/JSON，但只覆盖部分模型，单次运行不是多进程热测，加载与校验计时合并。`voxcpm_candle_probe` 有预热与多轮；独立 Qwen/CosyVoice3/IndexTTS 探针的计时与参数各异，不能直接拼表。

`probe_metrics.py` 已采样 Windows RSS 和整卡显存，但没有完整的运行矩阵、恢复协议或离线报告。产品资源解析位于 `talechime-core/src/paths.rs`；IndexTTS prepare 仅检查必需文件非空，CosyVoice3 使用固定大小与 SHA-256，基准必须诚实区分这些校验强度。

盘点时的 Windows 为 Ryzen 9700X / 8 核 16 线程、约 16 GiB RAM、RTX 5070 / 12227 MiB、驱动 616.56；执行时重采环境。当前生产目录有 14 个模型变体：MOSS 4、Qwen 5、Vox 2，以及 Omni/Cosy/Index 各 1。MOSS VoiceGenerator 为额外的 CUDA 音色设计辅助模型，不新增第七个朗读后端。

## Goals / Non-Goals

**Goals:** 在同一源码基线下逐项建立 Windows 数据，保留独立进程和逐轮证据，资源与音频不依赖 target，能从离线报告直接试听并在中断后继续执行。

**Non-Goals:** 不改变采样、精度默认、模型算术、设备门控、缓存语义、播放/协议 API；不比较官方实现、不做优化候选 A/B、不测 Mac/DirectML/AMD GPU，不以生成结果代替实际播放或人工听感结论。ASR 不是本轮 TTS 性能矩阵；可用的回读仅作辅助，缺失回读不触发新增 ASR 下载任务。

## Decisions

### 1. 显式矩阵和产品边界

以当前 Registry catalog、各模型 compiled_devices 和模式能力构建可审阅的冻结计划；工具自动盘点后仍将最终矩阵保存为 run manifest，避免 feature 缺失让模型悄悄消失。

| 后端 | 变体 | CPU | NVIDIA CUDA |
| --- | --- | --- | --- |
| MOSS | nano-candle / nano ONNX | 是 | 是，ONNX 需真实 provider 执行 |
| MOSS | local-1.7b / realtime-1.7b | 不适用 | 是 |
| Qwen | 0.6b-base / 1.7b-base | 是 | 是 |
| Qwen | 0.6b-customvoice / 1.7b-customvoice / 1.7b-voicedesign | 是 | 是 |
| VoxCPM | 2b-q8_0 | 是 | 是 |
| VoxCPM | 2b-bf16（实验） | 不适用 | 是 |
| OmniVoice | 0.6b | 是 | 是 |
| CosyVoice3 | 0.5b | 是 | 是，ORT CPU 前端单列 |
| IndexTTS | 2.5 | 是 | 是，生产 CUDA 使用 cuDNN |

即 28 个设备格中 25 个适用、3 个不适用，另加 VoiceGenerator 辅助任务。适用不等于可成功：缺资源、生产校验失败、ORT 缺 sm120 内核、OOM、超时仍是可见结果。已退休的 Kokoro/ZipVoice、旧 C++ Vox 对照不纳入。

固定短句“你好，欢迎使用叙铃。”和现有 `tools/tts/acceptance-corpus.txt` 原始字节，记录源摘要与实际分段。全部适用行覆盖两个工作负载；支持接续的模型以 off/on 分开，风格默认空；预置音色、克隆参考及设计描述分别固定。支持原生流式的适配器报告 chunk 边界；非原生流式只报其实际交付边界。完整请求耗时可由同次生产 stream 排空获得，不能伪称为另一个已测的非流式推理 API。已有独立库 full/stream 诊断结果若复用须另标层级，不进入产品基线。

Base/克隆使用现有准确转写的可复用参考；首选同一 WAV 跨支持模型复用，受模型约束时记录不同条件。CustomVoice 采用一个明确预置音色，Qwen VoiceDesign 采用一个固定描述；不对所有预置说话人/风格做组合穷举。MOSS VoiceGenerator 只测固定描述生成一次参考的成本及声音产物，后续 Local/Realtime 合成与该成本分开。

### 2. 开发资源导入，不扩展产品迁移

新增工具从 `AppPaths` 的解析结果接收资源根，不在 Python 重写另一套 HOME 规则；允许显式根覆盖。模型运行目录仍由现有模型 directory 规则确定；原始 checkpoint/转换输入放模型命名空间的 `official/<revision>/`，接收转换产物与 provenance receipt。参考音色用既有 VoiceStore 方式登记，保留 WAV/准确文字/描述，不迁移用户配置或检查点。

导入流程：盘点 → 身份验证 → 身份级锁 → 同文件系统暂存复制 → 摘要复核 → 无覆盖原子发布 → 收据。已存在且一致则复用；不同则报冲突，不自动重命名、替换、清空或调用可能把现有文件移为 `.corrupt` 的下载路径。补下载先落暂存，再遵循相同发布规则；中断后保留原件和有效目标，仅清理工具拥有的 staging。无需对大目录做整棵原子替换，按文件发布，完成标记/收据在全集验证后发布，未完成模型不能测量。

明确导入源：target 中 Qwen Base06、旧 `.novel-tts` 中 Vox BF16、`D:/workspace/index-tts/checkpoints{,-converted}`，以及新资源根已有模型与其参考。MOSS target 重复资源只做身份核对后复用现有副本。已授权缺失变体补下载，但不做无关文件系统全盘扫描或删除重复资源。

有生产 SHA 清单的按清单校验；IndexTTS 保留上游固定 revision、转换脚本身份、转换输入/输出完整 SHA 收据，再运行现有 prepare，不将本地产出的 SHA 自证为官方 oracle。额外子模型随 manifest 统一保留。

### 3. CosyVoice3 冲突独立登记

现有 HiFT 文件实际 `4cde9c7b…`，生产预期 `475ac02b…`；原始 checkpoint 已在持久目录。准备阶段对照 safetensors 元数据与逐张量值、转换器及固定原始输入，判断是否仅容器序列化差异，并尝试用已有固定转换流程复现生产清单对应产物。调查不改变算法、生产清单或校验阈值，也不为测量跳过 Engine prepare。

若无法获得符合现有清单的文件，该组合必须保持 blocked，并保留诊断及可选独立库开发测量，后者单独标层级，不进入产品成功统计。需要修改转换实现、清单或产品行为才能解决时，另行提出范围变更；本次完成测量台账可以有已尝试并说明原因的阻塞行，但不能称为所有模型均测量成功。

### 4. 一个生产适配器探针，加薄编排工具

在 backends examples 新增专用开发基准探针，复用 Registry、Backend 请求与生产文本分段/参考处理，覆盖现有探针缺少的 CustomVoice、VoiceDesign、CosyVoice3 和 IndexTTS。使用 Tokio LocalSet、既有 owner 线程及有界流，统一输出版本化事件/结果；不引入通用模型框架或公开产品 benchmark 命令。

按需通过开发入口暴露准确计时或设备同步：优先在 owner 执行路径测量已有边界，不使用另一个 CUDA context 代表模型同步，不为方便测量传出 inference/audio 对象。无法拆开的阶段输出 null 和原因；prepare 校验与 load 使用分离的外层计时，懒参考编码明确记为首请求准备，不能再次计入热请求。同步不能改变正常 chunk 内容或切断生产背压。

编排、采样、恢复、CSV/JSON、统计与 HTML/SVG 报告统一复用 rust-performance-harness 的 Rust/xtask 框架；不实现第二套 Python runner，不以 Python 绘图作为报告运行依赖。Python 仅允许用于已有上游转换的显式准备步骤并保留来源收据。CPU 与 CUDA 分别构建、记录实际 features；ORT GPU 所需组合可独立二进制，避免 cuDNN/ORT feature 统一后悄然改变 CPU 或其他模型基线。记录 graph 生效状态及所有相关环境开关，保持生产默认，不开展关闭图的 A/B。

### 5. 基线协议、预算和恢复

每个 case = 模型 × 设备/实现 × 语料 × 模式 × 条件/参数摘要。正式基线 3 个独立进程，每进程 1 预热 + 5 热轮；冷指新进程/新模型实例，不宣称清空 OS 文件页缓存。轮内多段先累计生成时间和实际音频时长计算 RTF，再算进程热均值，最后 mean-of-process-means；保留区间、逐轮分布和样本数，p50/p95 仅作为描述性统计，不将少量样本称为置信区间。加载与参考准备单列不摊入热合成。

同机全部串行。CPU 默认明确设置 Rayon 8 线程与 ORT 8 线程（可由冻结计划覆盖，分别记录），不做线程调优搜索；在相同配置下记录真实执行精度。每一配置/轮次的 seed 明确记录，默认 42，并接受模型支持范围，不把同 seed 当跨设备波形一致。

先用固定短句跑 readiness/screen，screen 不充当正式热轮。默认每请求超时 30 分钟、每进程 4 小时，均写入 manifest；退出后等待/验证 owner 释放，必要时终止所属进程树，未释放不能启动下一条。资源不足或预算耗尽保存 partial/timeout，不自动缩短语料、减少重复次数、降低采样参数来凑完整表。允许另建标记清楚的 screen，不替代正式 case。

记录当前电源计划、CPU/GPU 状态和桌面竞争进程；默认只观测，不改系统设置或锁频。每个进程前后采环境，运行中 500ms 采 RSS/peak working set、可用的 private bytes、整卡 VRAM、GPU 时钟/功率/温度。WDDM 无进程显存则 null；整卡前置基线和峰值同时保留，不把差值称为独占显存。profile/ASR/浏览器报告审阅与计时分开。

版本化 run/case/attempt 元数据含全部身份与路径收据；提交状态用原子文件发布。恢复仅复用身份相同且证据完整的终态成功 case；失败保留原 attempt，新 attempt 不覆盖，参数/源码/二进制改变则新 case 或批次。checkpoint 只用于基准自身，不读写用户播放进度。测试失败后不会跳过剩余后端，系统级设备不可用时逐行登记原因。

### 6. 报告与音频持久交付

批次位于 `AppPaths.root()/benchmarks/windows/<run-id>/`。目录包括 `run.json`、`resources.json`、`cases/…/attempts/…`、`results.json`、`results.csv`、`charts/`、`report/index.html`、`report/audio/`。原始日志、每轮整段/逐段 WAV 长期保留。报告选“第一个成功进程的第一个成功热轮”作为默认播放器样本，提供全部其他轮次索引；失败截断样本标记清楚，不放入成功播放器。

WAV 按原采样率保留 float32 PCM，播放兼容需要时另产 PCM16，记录派生关系和裁剪/量化方式，计时外写入。所有参考、文本、参数和摘要跟随元数据；HTML 转义文本与错误消息，路径只接受批次内资产，防止路径穿越。HTML 采用本地相对链接，数据内嵌或直接渲染，不依赖 file:// fetch、CDN 或模型文件。打包导出不包含模型和非必要参考原件，不自动发布或上传。

图表按语料/模式分组：CPU/CUDA RTF（1.0 实时线）、首 PCM、加载/参考耗时、RSS 与整卡 VRAM；明确后者不同单位与口径，不混为一个内存排行。显示失败/不适用格、覆盖率及重复次数，播放边界不同的模型分面呈现。图表链接原始表，跨模型对比只表明这些条件下的耗时，不暗示听感、音色或漏字等价。

仓库仅提交工具、普通夹具、方法和无音频/敏感内容的 `docs/records/platform/windows-backend-baseline-<date>.md` 汇总，可包含静态图表；完整试听包与批次数据在用户目录。报告初始 listening_acceptance=pending，不自动宣称实际播放、内容完整或质量合格，也不自动在 issue 上发消息。

## Risks / Trade-offs

- [25 个适用设备格乘模式/语料/重复，CPU 可能需长时间] → 先 readiness，冻结超时预算，支持恢复，完整/部分覆盖分别报告，不减少正式重复冒充完成。
- [16 GiB RAM / 12 GiB VRAM 无法容纳部分大模型] → 单实例串行、前后内存采样、保留 OOM；不改变精度或产品 CPU 门控。
- [准备、懒缓存与生成计时混杂] → 阶段可测则独立计时，无法分离则公开 null/包含关系，首请求与热请求不混算。
- [资源复制和逐轮音频占空间] → 导入前统计缺失/复用字节与可用空间，失败安全退出；不删除原件，导出包不带模型。
- [ORT rc.13 的 sm120 内核缺失] → 用实际 Nano 生成验证 provider，失败单列；不把初始化成功或 Candle CUDA 成功当作 ORT 可用。
- [外部 GPU 客户端造成波动] → 记录观测环境、分布和进程均值，不称独占，不从新旧异条件数字推导优化收益。
- [现有 Cosy 转换摘要冲突] → 保持生产门禁，完整诊断/blocked 行；不在基准项目中悄悄修模型或放宽验收。

## Migration Plan

仅增加开发工具和报告，不迁移用户配置/播放检查点。先实现小夹具验证导入、计时结果和恢复，再准备真实资源、编译所需 release 组合，完成 readiness、正式串行测量和报告审阅。运行中断可恢复；回滚工具改动不会删除用户模型或批次。恢复或清理由显式命令选择自有 staging/批次，不递归删除资源根。
