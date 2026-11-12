# 直接读取 Hub 缓存：阶段验证（2026-11-12）

## Qwen 1.7B CustomVoice

- 平台：Apple M4 Pro、arm64、macOS 26.6.2；CPU，release 构建。源码为 `09fa62631582aee3386fb1168bdb1d76e198b971` 加本地未提交改动，不代表发行版本。
- 输入：固定清单 `crates/talechime-backends/src/qwen/models/1.7b-customvoice.json` 对应的已有 Hub 缓存；资源身份为 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 的固定 revision。应用数据使用临时目录，端点指向不可用的 `127.0.0.1:9`，测试未下载模型。
- 路径：`Registry::resource_status` → `ResolvedModel::load_existing` → 已解析文件加载 → 固定短句“你好，欢迎回来。”、`uncle_fu`、seed 42 合成。测试检查有限、非静音的单声道 PCM 和正常结束，并确认临时应用数据下没有创建 Qwen 模型目录。
- 结果：24 kHz、46,080 个样本，峰值 0.4214；整个测试 63.44 秒，包含多次完整摘要校验、加载和合成，不是推理性能基准。

执行入口为 `crates/talechime-backends/tests/cache_direct_real.rs` 的显式 ignored 测试，需设置已有缓存路径；普通测试不下载模型。此结果不证明其他 Qwen 变体、GPU、离线损坏修复下的真实权重重下载、跨平台行为、官方数值一致性或人工听感。其他后端的直接加载和完整验收仍在 [OpenSpec 任务](../../../openspec/changes/use-hf-cache-directly/tasks.md)中。

## MOSS Nano

同一台 macOS / M4 Pro 以 CPU 和 release 构建，从固定的 Nano 与 Nano codec 两个 Hub 仓库的本地缓存离线加载，在临时应用数据目录中使用内置 `Weiguo`、seed 42 合成“你好，欢迎回来。”；未创建模型目录。输出 48 kHz 双声道、215,040 个交错样本，峰值 0.3473，检查了有限、非静音 PCM 和正常结束。

第一次测试错误地预期 24 kHz 单声道而失败；核对固定 codec 配置为 48 kHz 双声道后修正断言并复验通过。这个短句检查不是人工听感或官方数值对照；此前 Nano 漏读、年份连贯性和音色异常仍未解决，见[后端验收记录](../tts-core/official-backend-parity-2026-11-11.md)。

另在同机 debug 构建中，用 1 秒合成正弦 WAV 和隔离 Hub 缓存验证参考音色导入：codec 的固定文件从本机缓存硬链接至临时目录，端点不可用；编码后的音色记录写入临时应用数据目录，未创建模型目录。测试通过（2.48 秒），不代表参考音色质量或真实录音效果。

## Kokoro 1.1 中文 + g2pW

同一台 macOS / M4 Pro 以 CPU 和 release 构建，固定清单覆盖 121 个 Hub 文件，跨多个仓库及 revision；使用本机已有缓存、不可用端点和临时应用数据目录，经过缓存完整性盘点后直接加载并用 `zf_001`、seed 42 合成“你好，欢迎回来。”。输出 24 kHz 单声道、58,200 个样本，峰值 0.4172，检查了有限、非静音 PCM、正常结束及未创建模型目录；测试主体用时 7.58 秒，未下载资源。入口为 `crates/talechime-backends/tests/cache_direct_real.rs` 中显式 ignored 测试。

该检查不证明其他设备或平台、逐音色结果、人工听感及官方数值一致性；旧目录式开发探针和原生转换代码仍待清理。

## WeText 文件准备

改造前 CosyVoice3 和 IndexTTS 的原生准备从固定 WeText 0.1.0 wheel 提取四个 FST。调查时，`yexiyue/talechime-resources`、CosyVoice3 官方模型及其当前 Candle 成品仓库未列出这四个独立文件；`aireel/wetext` 的 FST 名称也与目标不同。该调查不代表其他仓库不存在同摘要文件。后续已按下述固定提交发布，运行时不再需要解包 wheel。

从清单固定 wheel 校验源摘要后，在 Git 忽略的 `.local/hf-resources/wetext-0.1.0/` 准备四个文件及原许可证；大小与 SHA-256 均与既有配方一致。显式 ignored 测试从各文件路径加载真实图，并对中英文数字文本执行归一化，通过。

IndexTTS 的 Linux 和西语图从仓库已有固定归档提取，日语词典复制自本机已准备的固定输出，连同对应许可证暂存至 `.local/hf-resources/indextts-frontend-v6/`；所有文件均按现有配方核对大小和 SHA-256。`TextFrontend` 可接收独立图文件：显式 ignored 测试使用上述暂存文件，分别对照已有 macOS、Linux 及西语官方前端夹具，全部通过。该结果验证文本图路径与夹具，不证明 IndexTTS 的完整模型加载或生成；文件随后已发布至下述固定提交。

同机 macOS / M4 Pro 的 debug CPU 构建中，显式测试 `model::tests::independently_resolved_model_files_load_on_cpu` 使用已有固定权重、独立暂存的 WeText 和日语词典路径，成功加载完整模型、执行中文文本归一化并读取日语词典（44.44 秒）。该测试未生成语音，权重仍来自此前本机准备的目录，不证明 Hub 直接读取、音色参考、听感或其他设备平台。

引擎线程随后改为按逻辑名称接收独立路径，显式测试 `indextts::runtime::resource_tests::owner_loads_files_from_independent_locations` 在相同平台上用分离的模型、WeText 和文本前端路径完成 CPU 线程加载和退出（22.61 秒），临时应用数据目录未创建。该测试不生成音频；Registry 随后切换到固定 Hub 清单，尚未进行该模型的端到端合成验证。

## CosyVoice3 分离文件加载

在同一台 macOS / M4 Pro 的 release CPU 构建中，使用本机已有的固定模型成品和另一个暂存目录中的 WeText 图，显式运行 `cosyvoice3::runtime::tests::owner_loads_independently_resolved_files`。推理线程成功初始化并正常退出，未创建临时应用数据目录；测试主体用时 12.76 秒，未下载。此时 Registry 仍走旧成品准备目录，本次只验证模型、ONNX 前端和文本图的独立路径加载，不证明 Hub 直接加载、参考音色合成或听感。

线程输入调整为可由已解析的逻辑路径映射构造后，同一测试在 debug CPU 构建再次通过（63.93 秒）；缺项映射的独立单元测试也通过。随后已切换 Registry 并进行了下述验证。

固定文本图随后发布到 Hub 提交 `aec04d67e67f8e309ec8996f161d134b43e4ecd7`，远端 15 个 Git/LFS 对象的大小和身份与本地文件核对一致。切换 Registry 后，使用隔离 Hub 缓存、不可用网络端点及 2 秒合成正弦参考音频执行真实 CPU 加载和“你好。”合成：输出 24 kHz 单声道、5,760 个样本，峰值 0.0882；成功结束且未创建模型目录（45.80 秒）。参考音频不是人声，结果仅证明直接缓存路径和非静音 PCM，不代表音色或人工听感。另在隔离缓存以官方端点下载已发布 WeText 文件并完成摘要盘点（35.04 秒）。

IndexTTS 在同一隔离缓存中以官方端点下载已发布的日语词典及其他前端文件，并对 33 项固定资源完成摘要盘点（83.84 秒）；该检查不执行该模型合成。按用户 2026-11-12 的要求，本变更只要求至少一款真实模型的直接缓存合成验收，其他模型保留自动化和资源完整性验证。
