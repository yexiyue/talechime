# Kokoro 官方权重原生后端

当前工作区默认使用 `01e7505-runtime-f32-v3` 成品配方，用户直接下载并校验，无需重复转换。
下文涉及 `01e7505-native-f32-v2` 的原生构建与耗时为源配方验证记录。
固定分发、许可与两类配方边界见[公开成品记录](records/resources/runtime-distribution-2026-11-05.md)。

Kokoro 是可选的中文预置音色后端，当前源码提供 `1.1-zh` 和显式启用的
`1.1-zh-g2pw`。声学网络直接对照 hexgrad 官方 Python 实现移植到 Candle；
运行时不调用 Python。0.6.0 标准发行配置已编入该后端，本机 CPU/Metal 对照结果见
[实现记录](records/kokoro/kokoro-native-validation-2026-10-27.md)。

输出为 24 kHz 单声道，每个模型片段生成完毕后交付 PCM。没有原生流式、音色克隆、
风格指令或自定义读音接口。仅支持安装的官方 `zf_*` / `zm_*` 中文音色，
参数 `speed` 范围为 0.5–2.0。当前源码支持中英文混读；无法处理的符号明确报错，
英文词典与独立未知词后备的边界见下文。

## 编译与执行

facade 和 backends 的开关一致：`kokoro`、`kokoro-metal`、`kokoro-cuda`、
`kokoro-g2pw`。Cargo 默认 feature 不启用 Kokoro；标准发行配置启用 kokoro-g2pw，macOS 另启用 kokoro-metal。默认后端仍为 MOSS Nano。
CPU 使用 F32；在 macOS 上启用 Accelerate。Metal/CUDA 开关选择解码器设备，
编码器、时长/F0 预测和 FFT 保留 CPU，因此 GPU 路径是混合执行。
Windows CPU 使用保留原 GEMM 布局的并行卷积窗口构造；CPU 与 CUDA 混合路径的
输出保持、性能及未通过的官方数值门禁分别见
[Windows 记录](records/kokoro/kokoro-windows-performance-2026-10-27.md)。
后续输出布局优化与 MKL/AVX2/AVX-512 调查见
[Windows CPU 第二轮](records/kokoro/kokoro-windows-cpu-round2-2026-10-27.md)。
不能由 feature 存在或输出保持型提速推断跨平台数值/音质已验收。

```sh
cargo run --release --locked -p talechime --no-default-features \
  --features kokoro --example kokoro_backend -- RESOURCE_ROOT zf_001 cpu

cargo run --release --locked -p talechime --no-default-features \
  --features kokoro-metal --example kokoro_backend -- RESOURCE_ROOT zf_001 metal

cargo run --release --locked -p talechime --no-default-features \
  --features kokoro-g2pw --example kokoro_backend -- RESOURCE_ROOT zf_001 cpu 1.1-zh-g2pw
```

库入口使用 `ModelOptions::new("kokoro", resource_root)`，可设置
`options.model = Some("1.1-zh-g2pw".into())`。通过 Engine 在 LocalSet 上运行；
内建音色与参数从准备后的 capabilities 查询。推理对象由专用线程持有，
取消检查覆盖资源哈希、模型阶段和有界 PCM 发送；单次底层算子执行不能抢占。

## 准备固定运行资源

显式准备通过 `ResolvedModel::native_preparation_plan()` 获取当前缓存需求，确认指纹后
调用 `prepare_native_resources`。默认下载已验证的完成文件，普通中文与 g2pW 变体的
`conversions` 均为空；`load_existing` 只校验和加载已发布资源，不隐式下载。
完整取消、磁盘预算及发布协议见[准备接口](native-resource-preparation.md)。

当前目录分别为 `ROOT/kokoro/models/1.1-zh/01e7505-runtime-f32-v3/` 与
`ROOT/kokoro/models/1.1-zh-g2pw/01e7505-runtime-f32-v3/`。必须匹配编译清单和发布标记，
不能把自行导出的旧 `native-v1` 目录放入新版本冒充就绪。正式准备不调用外部脚本，也不
迁移/删除旧模型、用户音色或原始资产。

维护方的 v2 原生源配方以固定官方/核验社区 F32 输入经 Rust 折叠权重归一化，补齐固定
官方 AdaIN 参数并组装完整资源。被忽略的准备测试显式设置 `TALECHIME_BUILD_NATIVE_SOURCE=1`
才选择源构建；默认用户入口不自动回退。运行资源、配方与许可固定在
`crates/talechime-backends/src/kokoro/` 的 native/distribution 清单，来源对照另见
[SOURCE.md](../crates/kokoro/SOURCE.md) 和公开成品记录。

## 维护方独立参考导出

以下 Python 工具只为开发者生成官方数值对照，终端用户不需要它们。需要固定官方
Kokoro/Misaki 源码及原始模型、音色；分词字典完整许可须明确提供。导出目标必须是新目录，
不会自动注册为当前发布版本。具体依赖版本及完整资产核验见本轮资源记录。

```sh
python crates/kokoro/scripts/export_official.py \
  --kokoro-source KOKORO_CHECKOUT --misaki-source MISAKI_CHECKOUT \
  --assets OFFICIAL_ASSETS --voice-file OFFICIAL_ASSETS/zf_001.pt \
  --output NEW_MODEL_EXPORT

python crates/kokoro/scripts/export_frontend.py \
  --misaki-source MISAKI_CHECKOUT --jieba-license JIEBA_LICENSE \
  --output NEW_FRONTEND_EXPORT
```

## 可选上下文读音

`1.1-zh-g2pw` 在相同声学模型前增加作者官方 g2pW v2 ONNX CPU 推理。
它先对规范化后的字符预测多音字，再应用已有变调、儿化和音素映射；
原版 `1.1-zh` 保留官方 Misaki 词典行为。

g2pW 是另一个模型，不能保证所有姓名或上下文正确。本阶段 24 个手工目标读音
答对 23 个，“解先生”仍误读为 jie3。该小语料结果不是通用正确率。
相对原版有 6 例音素结果改变，含重复的姓氏上下文；未以训练或规则硬编码修补测试句。

生成独立对照的开发环境还需要作者官方 g2pW 固定源码和验证过的模型压缩包、BERT 中文 tokenizer。
只需 tokenizer 文件，不另下载 BERT 权重。

```sh
python crates/kokoro/scripts/export_context.py \
  --source G2PW_CHECKOUT --archive G2PWModel-v2-onnx.zip \
  --model-dir EXTRACTED/G2PWModel --tokenizer-dir BERT_TOKENIZER \
  --output NEW_CONTEXT_EXPORT

```

## 独立对照

普通测试不下载模型。以下命令只读取开发者明确提供的本地导出，数值门槛失败会返回错误。
声学对照注入完全相同的 IDs、style、speed 和随机激励，要求时长一致、各阶段/PCM
SNR 至少 40 dB；固定官方中间特征的独立解码对照要求 60 dB。

```sh
cargo run --release --locked -p talechime-kokoro --example official_compare -- NEW_MODEL_EXPORT cpu
cargo run --release --locked -p talechime-kokoro --example frontend_compare -- NEW_FRONTEND_EXPORT
cargo run --release --locked -p talechime-kokoro --features g2pw \
  --example context_compare -- NEW_CONTEXT_EXPORT NEW_FRONTEND_EXPORT
KOKORO_OFFICIAL_FIXTURE_DIR=/absolute/path/NEW_MODEL_EXPORT cargo test --release --locked \
  -p talechime-kokoro --lib -- --ignored --nocapture
```

g2pW 对照要求输入张量完全一致、概率绝对误差不超过 1e-5，同时报告人工目标读音。
数值对照和 Engine 生命周期测试不能替代完整章节试听、跨平台和实际播放验收。

`official_compare NEW_MODEL_EXPORT cpu 5` 可在数值门禁通过后增加一次显式预热与
五轮无 trace 的声学计时，并导出本地 WAV/JSON。前面的数值预检也执行过一次
同形状 forward。此计时使用固定官方激励，不包含前端、加载和随机数生成；
单进程结果只能作为性能筛查，不能代替正式的进程交错对比。

固定官方输入的优化 A/B 可用以下独立探针，输出目录必须不存在。它每轮保存完整
F32 PCM，并检查同 owner 重复和中途取消后的逐位恢复；计时包括获取主机 PCM。
它不代替前述官方 SNR 门禁，也不包含中文前端、原生噪声生成、加载或实际播放。

```sh
cargo run --release --locked -p talechime-kokoro --example performance -- \
  NEW_MODEL_EXPORT cpu NEW_EVIDENCE_DIRECTORY 5
```

`performance_probe` / `performance_catalog` 已支持 Kokoro 的生产资源验证与 owner
加载，编译时显式启用 `kokoro,performance-tools` 或 `kokoro-cuda,performance-tools`；
可选 g2pW 仍需单独准备对应资源。生产整链与固定激励声学探针的 RTF 分别报告。

M4 Pro / CPU F32 开启 g2pW，最新八线程对照中长文原生 RTF 为 0.06522、官方组合为
0.06637；短句分别为 0.07010 / 0.06944。公共 Engine RTF 为 0.07259，约 13.8 倍实时。
完整 PCM 保持逐字节一致，见[缓冲、归一化与向量优化记录](records/kokoro/kokoro-cpu-buffer-vector-2026-10-27.md)。
该样本的模型内窗口缓冲保留约 297 MiB；内存峰值与线程条件在记录中分别列出。
历史阶段见[第一轮 CPU 优化](records/kokoro/kokoro-cpu-optimization-2026-10-27.md)、
[四线程官方对照](records/kokoro/kokoro-cpu-parity-2026-10-27.md)和
[原始性能记录](records/kokoro/kokoro-g2pw-performance-2026-10-27.md)。

### 原生资源准备进展（2026-11-05）

中文前端数据已在公共、无访问审批的
[yexiyue/talechime-resources](https://huggingface.co/yexiyue/talechime-resources/tree/8eb001f5498e099e6ac929454e7fb28e457bf035/kokoro/misaki-zh-v1)
发布并逐文件匿名下载校验。运行时为 Rust 词典/规则读取，不需要 Python；生成工具仅供开发环境使用：先运行 `scripts/export_frontend.py`（显式传入固定 Misaki checkout、jieba v0.42.1 LICENSE 和新输出目录），再运行 `frontend_compare`，最后用 `scripts/package_frontend.py` 形成带来源与许可证的分发包。参数以工具的 `--help` 为准。

原生权重转换已与官方完整 599 个张量逐位对齐，并通过 CPU/Metal 固定噪声声学对照。应用内一键准备、g2pW 配方和桌面试听验收仍在接入，不以数据发布或上述数值对照代替。
详细证据见 [资源核验记录](records/resources/native-preparation-2026-11-05.md)。

#### 已登记的普通中文原生配方

SDK 的 `kokoro/1.1-zh` 已登记 `kokoro-1.1-zh-native-f32-v2`。
`ResolvedModel::native_preparation_plan().await` 提供离线计划，确认指纹后调用
`prepare_native_resources`，原生折叠 F32 权重、复用 100 个经官方逐元核对的中文音色，并下载上述固定前端数据。输出版本目录为 `kokoro/models/1.1-zh/01e7505-native-f32-v2`；旧手工导出目录不迁移。

完整目录为 114 个已固定摘要的运行资源，共 396,981,297 字节（另含准备发布标记）。在已核验输入的隔离资源根中，原生准备/校验/发布实际完成约 9.67 秒，随后计划返回就绪、缺失下载量和额外准备空间均为零。推理只接受与编译配方相同的清单及发布标记，修改清单或混用读音变体会被拒绝。缓存中的完成组件按固定输出摘要复核后共享；未完成文件不复用。

宿主模型中心的新准备确认、进度和任务日志尚在接入；此 SDK 记录不表示桌面流程已完成。

#### 可选 g2pW 原生配方

启用 `kokoro-g2pw` 后，`kokoro/1.1-zh-g2pw` 同样通过 SDK 原生准备计划执行。上下文表包固定到资源仓库提交 `ead7e24684eaea42af3dc0fec0bc5c7d9c51bfdf`，ONNX 读音模型从固定官方 ZIP 原生解包，BERT 仅下载词表。声学权重、100 个音色和普通中文前端与普通变体共享完成组件；不切换 ONNX 声学执行。

两变体均已在 CPU、Metal 完成真实加载、短文本合成和有效 WAV 校验。g2pW 24/24 官方输入/输出对照通过；人工目标读音为 23/24，保留上游同一处模型误读，听感尚未人工评价。新配方不接收用户自建的清单或混合版本。公开数据包匿名可下载，运行时不依赖开发阶段 Python 环境。

### 中英混排（开发中，2026-11-11）

Kokoro 中文前端先规范化数字，再使用固定的 `talechime-loqui-g2p` 美式词典处理英文；真正无法由词典和形态规则处理的词交给独立 eSpeak NG 1.52.0 程序。宿主必须提供可信 bundle；缺件、校验失败、超时及取消明确报错，不回退到神经读音。可通过 `TALECHIME_ESPEAK_DIR` 指定 bundle，或在可执行文件旁的 `talechime-helpers/espeak-ng-1.52.0/` 自动发现；开发构建也检查 `target/talechime-helpers/espeak-ng-1.52.0/`。词典词不要求启动外部程序。启用 Kokoro 编译要求 Rust 1.92 或更新，其他语言和专名读法仍需听验。

2026-11-11：使用 macOS Apple Silicon 的独立 bundle，对固定 Misaki 美国英语 callable/eSpeak 音素对照，普通版和 g2pW 各有 11/13 条相同，三个独立 OOV 样本全部相同。剩余两类差异为带空格年份的产品规则和未知符号明确报错；不能把它们计作官方一致。来源、精确方法、校验及边界见 [SOURCE.md](../crates/kokoro/SOURCE.md#mixed-language-oracle-2026-11-11)。g2pW 按中文片段预测可能丢失跨英文的上下文；听感、其他平台和真实桌面验收仍未完成。

全角百分号边界：固定 Misaki 对 `你好％` 返回 `ㄋㄧ2ㄏㄠ3❓`，原生前端明确拒绝 `％`；独立 fixture 记录的是差异，不算官方一致。先前真实合成中该输入的失败仍按失败记录。

Kokoro 的 `ResolvedModel::revision()` 使用独立推理语义版本 `01e7505-runtime-f32-v3-text-v3-espeak-1.52.0-4870adfa25b1a32b4361592f1be8a40337c58d6c`，包含 OOV 回退所用的固定 eSpeak 源码身份。已校验权重目录仍为 `01e7505-runtime-f32-v3`，无需重复准备权重；bundle 清单校验用于检查本地文件完整性，并非签名认证。


2026-11-12 完整语料复核：已修复中英文片段多余空格，并将独立后备改为直接调用
官方相同的 `espeak_TextToPhonemes`，避免通用 CLI 给 `de` 等功能词增加句末重音。
20 条功能词/OOV 全部相同。19 条公开文本的两变体各 9 条音素完全相同、7 条明确符号拒绝、
3 条产品执行策略差异（空格年份、换行、连续空格）。两个变体的 Metal 矩阵共 38 行：
24 段有效音频、14 条预先声明的符号拒绝，无缺行。原有 v11 的六条失败不改写；
新一轮拒绝已在运行前按精确错误登记。当前声音仍待用户听验，g2pW 年份的历史反馈保留为失败。
后备包同时包含独立程序、对应 C 源码、构建脚本及 eSpeak 的固定源码和许可；Rust 不链接 GPL 库。
