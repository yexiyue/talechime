# 开发与验证

Rust 性能开发入口、Windows 任务复用及 Mac 接续边界见 [性能工具](performance-harness.md)。

## 工具链

Rust 2024，声明最低 Rust 1.89；当前 stable 验证不等于最低版本实机验收。Linux 安装 `libasound2-dev libssl-dev pkg-config`。CUDA 可选构建需要匹配 Toolkit；显式设备不可用时不能静默替代。ORT 固定 rc.13，Candle core/nn/transformers 固定 0.11.0；保持 Windows 动态 CRT。

TTS 热路径修改先按 [性能优化方法](knowledge/tts-performance.md) 固定基线与输出门禁。
当前性能主线见[双设备优化 #5](https://github.com/yexiyue/talechime/issues/5)：Windows
RTX 5070 的 CUDA，与 MacBook Pro M4 Pro 的 Metal / CPU Accelerate 分别实机验收。
性能报告在结果表旁注明设备和执行路径，不把 Accelerate feature 启用当作逐算子 AMX 验证；
固定音色与可选 VoxCPM2 双参考见[低优先级规划 #4](https://github.com/yexiyue/talechime/issues/4)。
Candle core/nn/transformers/metal-kernels 的 fork 来源统一在根 `[patch.crates-io]`，
CUDA/Metal feature 仍由 `tts-candle-platform` 按 target 路由；详见
[来源与发布策略](legal/candle/SOURCE.md)。组合启用模型时须检查 feature 合并后的依赖与测试。

```bash
cargo test --locked --workspace --lib --tests --examples
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items --workspace --examples
```

不带 GPU Toolkit 的机器可验证 CPU 适配器组合：

```bash
cargo test --locked --workspace --features talechime/qwen,talechime/voxcpm,talechime/omnivoice,talechime/moss-candle,talechime/asr --lib --tests --examples
cargo build --locked -p talechime --bins
```

完整 all-features 检查需要相应平台工具链；CUDA 与 Metal 编译分别在原生 CI 中检查。普通测试不下载大模型；真实模型测试使用显式环境变量，保留历史 `TRNOVEL_*` 变量以兼容已有验收脚本。旧 `tools/` 脚本已删除；性能工具使用 Rust `cargo xtask perf`。历史对照和转换脚本可从 Git revision `e591c94` 取回，不是当前工具依赖。

## 编译 feature 分层

CLI 与 backends 使用同名开关，计算库只提供 `cuda` / `metal` 平台开关：

| 层次 | 开关 | 含义 |
| --- | --- | --- |
| 模型适配器 | `moss`、`qwen`、`voxcpm`、`omnivoice`、`cosyvoice3`、`indextts` | 启用对应模型；默认构建启用 `moss-nano-candle`；单独启用 `moss` 保留 ONNX；IndexTTS 仅源码构建 |
| 回读模型组 | `asr` | Qwen3-ASR 0.6B Candle CPU + SenseVoiceSmall INT8 ORT CPU；合成策略仍默认关闭 |
| Nano 默认实现 | `moss-nano-candle` | 在 MOSS 目录优先选择原生 Nano；ONNX 仍可显式选择 |
| MOSS 实验实现 | `moss-candle` | 在 MOSS 目录增加 Local / Realtime；生产入口目前仅开放 GPU |
| Candle 加速 | `qwen-cuda` / `qwen-metal`、`voxcpm-cuda` / `voxcpm-metal`、`omnivoice-cuda` / `omnivoice-metal`、`cosyvoice3-cuda` / `cosyvoice3-metal`、`moss-candle-cuda` / `moss-candle-metal`、`moss-nano-candle-cuda` / `moss-nano-candle-metal` | 启用对应适配器及计算库加速；按目标平台门控 |
| IndexTTS 加速 | `indextts-metal`、`indextts-cuda-cudnn` | 启用 IndexTTS 及对应设备；CUDA 使用 cuDNN 卷积，需要原生工具链 |
| Candle cuDNN 卷积 | `indextts-cuda-cudnn` | indextts 的 cuda 变体：卷积走 cuDNN，绕开 candle unfold 回退对跨步布局的限制（普通 `cuda` 行为不变） |

| ORT provider | `ort-cuda`、`ort-coreml` | 为 Nano 提供 provider，不自动启用模型适配器 |

macOS 上 `cosyvoice3` 与 `qwen` 的 CPU matmul 默认经系统 Accelerate（AMX BLAS）执行
（2026-10-18 决定；target 门控依赖，无开关）。数值与其他平台的纯 Rust
`gemm` 路径非位级一致，详见[性能记录](records/cosyvoice3/cosyvoice3-performance-2026-10-17.md) §11
与 [Qwen 记录](records/qwen/qwen-tts-acceptance.md)。

适配器层不提供通用 `metal` / `cuda` 开关；选择加速时必须明确模型。旧 `metal` / `coreml` 名称已移除，改用 `qwen-metal` / `ort-coreml`。加速开关包含对应模型开关，发行配置无需重复列出模型。`directml-probe` 仅是 backends 的实验 example，不暴露产品设备。

Cargo 会统一同一计算库的 feature，可能使依赖同时拥有多个加速实现；各适配器仍按自己的开关报告设备，不能以底层 Candle 的统一 feature 推断模型可用性。MOSS 两种实现共用一个后端目录，`moss-candle` 依赖 `moss` 是当前设计；CPU Nano 与 GPU 实验模型的差异保留。

## Rust 库发布

`talechime-libs-v*` 标签触发独立库发布流程，不触发应用安装包构建。流程先完成 CPU
模型组合的测试、Clippy、格式、rustdoc 和纯库检查，再按依赖顺序执行 `cargo publish`。
协议 v7 使用 `talechime-protocol 0.3.0`（新增加法字段 `PlanRequest.continuation_boundary`
与 `continuation_span`，缺省值保持 0.2.0 行为；旧宿主发出的计划仍被接受），不兼容已发布
的 0.1.0 / 协议 v5。
Qwen 计算库使用 `talechime-qwen3-tts` / `talechime-qwen3-asr`，避免占用其他作者的包名；
Rust 导入名不变。旧 `voxcpm-sys` 开发对照已退役，历史复现见[来源与许可证归档](legal/retired-voxcpm-native/README.md)。

发布失败后只重跑原提交的工作流；脚本查询同版本并核验 `.cargo_vcs_info.json` 的提交和包目录。
已从该提交发布的包跳过；来自可验证祖先提交的版本，仅在该包目录与根 `Cargo.toml`
相对当前发布提交完全未变时跳过，允许新的库版本复用未修改的计算库。
同版本源码已变、来源未知或不属于当前发布历史、包目录不符及 yanked 版本明确失败。
发布及恢复 checkout 保留完整历史；上传结果未知时不会自动换版本。
若只在上传阶段失败，可调用 `Resume validated library release`，传入原发布 run ID；流程
通过 GitHub API 核验该提交的全部检查成功，再 checkout 原提交并继续发布，不重复已通过的检查。
恢复流程仅对明确拒绝上传的 429 新包限流按服务端时间等待，再重新核验已发布版本；
其他错误停止。等待总预算一小时、最多十次，不变更原版本或移动原标签。
Cargo 发布验证会从打包源码构建，不依赖兄弟 crate 的文件系统路径。共享资源清单由计算库
导出常量，核心测试夹具包含在核心包内。

## 原生验收

迁移不升级模型、不改 PCM、EOS、取消、用户数据和检查点语义。历史数值与人工试听结论位于 `docs/records/`，平台差异、未通过项与实验入口必须如实记录。迁移后仍应执行独立协议握手、配置隔离、默认/扩展 CPU 构建及 GPU 编译检查。

Qwen ICL 直接使用调用方的 repetition penalty 和帧上限，不再隐式提升至 1.5 或按文本长度截断。
12Hz codec 默认使用 72 帧滑动注意力；流式保持 10 帧新块与最近 25 帧 codes 上下文，首块保留完整参考。
这会改变旧输出并增加流式解码开销；上下文样本只用于条件计算，不重复播放。
参考 encoder 加载 HF 权重时适配半组/交错 RoPE 布局；自定义音色使用新的
`prompt-encoder-rope-v2.json` 缓存，保留旧 `prompt.json`，从原参考音频自动重建。
参考与逐帧的 BF16/F16 codebook 向量统一在 F32 中累加，最后转换回原 dtype。
官方开发对照工具默认保护空历史 repetition penalty，规避本机 Torch 2.7.1 MPS
返回复用缓冲区内容的问题；历史模式需显式 `--empty_repetition original`，两者分别记录。
固定 codes 的官方 F32 交叉验证与人工试听反馈见 [Qwen 正确性记录](records/qwen/qwen-performance-review-2026-10-18.md)。
Metal depthwise ConvNeXt fast path 及实现内性能边界见
[Qwen 性能优化记录](records/qwen/qwen-performance-optimization-2026-10-18.md)；
该路径只适用于符合条件的 causal depthwise 卷积，路由测试要求与原路径逐元素一致，
不改变固定 seed codes 或 PCM。
Qwen 的设备传输与 CPU 采样策略分别负责；depthwise fast path 封装在 causal conv
模块，流式 decoder 状态独立负责 codes 历史、PCM 裁剪和成功后提交，会话负责生成与取消。
修改这些边界时，应同时验证多 batch mask 广播、旧 CDF fallback、解码失败重试、
末尾短块与真实模型输出 hash；不能仅用两条调用路径共用同一实现的测试证明等价。

构建产物不进入 Git；也不提交模型权重、用户参考音频、正文、凭据和临时运行输出。第三方源码更新保留固定 revision、许可证和对照夹具。PR 使用 Conventional Commit，说明行为、兼容和验证。

## 发行

只保留 dist-workspace.toml，使用 cargo-dist 0.32.0 的原生 Cargo 构建。运行 `dist generate`、`dist generate --check` 和 `dist plan`。`.github/workflows/ci.yml` 在三个平台构建标准包、解压至隔离目录并执行无模型、无 CUDA 的帮助/握手检查；Mac 标准包编译全部 Metal 路径。许可证由 `.github/scripts/prepare-notices.py` 汇集到 target/third-party-licenses 并随包分发。

协议库单独发布 `talechime-protocol-v*` 标签到 crates.io，应用标签使用 `talechime-v*`。CARGO_REGISTRY_TOKEN 与 HOMEBREW_TAP_TOKEN 由 GitHub Secrets 管理；Homebrew 发布到 yexiyue/homebrew-tap。正式应用标签需发布时再创建，普通验证不下载模型、不代替试听。

GNU release runner 为 Ubuntu 24.04，最低 glibc 2.39，ORT 1.28 预编译库不能在 Ubuntu 22.04 链接。Mac 标准包 macOS 15+，构建设置 MACOSX_DEPLOYMENT_TARGET=15.0；Metal 探测先枚举设备，避免无 GPU 的 hosted VM 调用 Candle 0.11 构造函数时 abort。非空 HOME 可用于隔离主目录。许可收集包括 Cargo registry 依赖和 pinned ONNX Runtime 的完整 notices。

## 文档归档

使用与架构指南直接放在 `docs/`，历史验收与实验放在 `docs/records/` 的主题子目录（划分与归档规则见 [records/README.md](records/README.md)），开发知识放在 `docs/knowledge/`，原生第三方声明放在 `docs/legal/`。[文档索引](README.md)是统一入口。搬迁记录时只更新路径引用，保留验收日期、设备、未通过项及原始结论；当前行为变化写入指南和知识库。


## 仅库构建

`talechime` 默认启用 cli 与 moss-nano-candle。嵌入库用 `default-features = false` 再选择模型 feature；cli 门控 executable、clap/crossterm 与终端控制依赖。无模型库测试可运行：

```sh
cargo test --locked -p talechime --no-default-features --test library
cargo clippy --locked -p talechime --no-default-features --all-targets -- -D warnings
```

无默认 feature 的二进制构建需显式加 cli，例如 `--features cli,qwen-metal`。core 当前仍编译播放器依赖，直接合成运行时不打开设备。

ASR 普通测试使用确定性转写和合成夹具，不下载权重；原生前端用固定整数合成波形对照独立 kaldi-native-fbank 数值。手工真实模型探针：`cargo run --release --locked -p talechime-backends --features asr --example asr_readback -- QWEN_DIR SENSEVOICE_DIR CORPUS_JSON OUTPUT_JSON`。语料、模型、结果只放 Git 忽略目录；验证范围见[原生验收](records/asr/asr-readback-mainline-2026-10-09.md)。

## 段落内接续探针

独立探针使用固定 manifest 校验模型，固定种子 42；真实模型、输入、对照 WAV 和报告都放忽略目录。

```sh
cargo build --release --locked -p talechime-backends --features qwen-metal,omnivoice-metal --example continuation_probe --example moss_continuation_codec
target/release/examples/continuation_probe moss NANO_DIR target/continuation-evidence/moss-on cpu Weiguo on CORPUS_TXT
target/release/examples/continuation_probe qwen BASE06_DIR target/continuation-evidence/qwen-on metal custom:reader on CORPUS_TXT
target/release/examples/continuation_probe omnivoice OMNI_DIR target/continuation-evidence/omni-on metal narrator on CORPUS_TXT
target/release/examples/moss_continuation_codec NANO_DIR
```

将 `on` 换成 `off` 并使用另一输出目录生成对照；Qwen Base 先通过 `voices import` 导入准确转写的参考音色。
探针支持尾部旗标：`--seed n` 钉住采样（默认 42）、`--param name=value`（可重复，按当前模型
目录校验）、`--style 描述`（逐段风格）。summary.json 记录实际 seed、参数与 style。
探针输出完整与逐段 float32 WAV、PCM16 回读 WAV、原文范围、首个 PCM 延迟、合成耗时及 RTF；
用 `/usr/bin/time -l` 额外记录进程峰值内存。`readback-corpus.json` 可传入上面的 `asr_readback`。
取消、错误不会自动退回独立生成；失败保留已产生的音频与错误摘要。
Nano 数值探针对比分批前缀预热与整段预热后同一组新 codes 的 PCM，前缀音频本身不交付。
[本次结果与限制](records/continuation/continuation-small-models-2026-10-10.md)包含固定语料、权重 revision、回读及试听边界。

其他模型复用上述探针：`qwen-base17` 的目录是 Base 1.7B revision 目录；`voxcpm` /
`voxcpm-bf16` 使用对应 revision 目录；`moss-local` / `moss-realtime` 使用资源根目录，
由各自固定 manifest 校验模型与 codec。构建探针时增加 `voxcpm-metal,moss-candle-metal`，
CUDA 可替换为相应 `-cuda` feature 与 `cuda` 设备。使用 `on` / `off` 生成对照；
BF16 仍只开放 CUDA。听感与真实平台数值验收单独记录，不由能力标记推断。

CosyVoice3 的本地转换、资源校验、音色导入与端到端探针见 [CosyVoice3 后端](cosyvoice3.md)。

## Mac 性能工具契约

`cargo test --locked -p xtask` 包含 Mac 原生进程/采集测试。构建免模型 performance_probe 后设置 `TALECHIME_PERF_PROBE_BUILD` 为 build receipt 的绝对路径，再运行 `cargo test --locked -p xtask --test cancellation_cli`，才能验证实际 SIGINT/SIGTERM CLI 取消；未设置时测试仅提示跳过。Mac CI 运行相同显式流程，不下载模型。命令与平台保证见 [性能工具](performance-harness.md#mac-原生工具接线)。
