# 开发与验证

## 工具链

Rust 2024，声明最低 Rust 1.89；当前 stable 验证不等于最低版本实机验收。Linux 安装 `libasound2-dev libssl-dev pkg-config`。CUDA 可选构建需要匹配 Toolkit；显式设备不可用时不能静默替代。ORT 固定 rc.13，Candle core/nn/transformers 固定 0.11.0；保持 Windows 动态 CRT。

```bash
cargo test --locked --workspace --lib --tests --examples
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items --workspace --examples
```

不带 GPU Toolkit 的机器可验证 CPU 适配器组合：

```bash
cargo test --locked --workspace --features talechime/qwen,talechime/voxcpm,talechime/omnivoice --lib --tests --examples
cargo build --locked -p talechime --bins
```

完整 all-features 检查需要相应平台工具链；CUDA 与 Metal 编译分别在原生 CI 中检查。普通测试不下载大模型；真实模型测试使用显式环境变量，保留历史 `TRNOVEL_*` 变量以兼容已有验收脚本。`tools/tts/` Python 仅供开发对照。

## 编译 feature 分层

CLI 与 backends 使用同名开关，计算库只提供 `cuda` / `metal` 平台开关：

| 层次 | 开关 | 含义 |
| --- | --- | --- |
| 模型适配器 | `moss`、`qwen`、`voxcpm`、`omnivoice` | 启用对应模型；默认 `moss` 使用 Nano ONNX |
| MOSS 实验实现 | `moss-candle` | 在 MOSS 目录增加 Local / Realtime；生产入口目前仅开放 GPU |
| Candle 加速 | `qwen-cuda` / `qwen-metal`、`voxcpm-cuda` / `voxcpm-metal`、`omnivoice-cuda` / `omnivoice-metal`、`moss-candle-cuda` / `moss-candle-metal` | 启用对应适配器及计算库加速；按目标平台门控 |
| ORT provider | `ort-cuda`、`ort-coreml` | 为 Nano 提供 provider，不自动启用模型适配器 |

适配器层不提供通用 `metal` / `cuda` 开关；选择加速时必须明确模型。旧 `metal` / `coreml` 名称已移除，改用 `qwen-metal` / `ort-coreml`。加速开关包含对应模型开关，发行配置无需重复列出模型。`directml-probe` 仅是 backends 的实验 example，不暴露产品设备。

Cargo 会统一同一计算库的 feature，可能使依赖同时拥有多个加速实现；各适配器仍按自己的开关报告设备，不能以底层 Candle 的统一 feature 推断模型可用性。MOSS 两种实现共用一个后端目录，`moss-candle` 依赖 `moss` 是当前设计；CPU Nano 与 GPU 实验模型的差异保留。

## 原生验收

迁移不升级模型、不改 PCM、EOS、取消、用户数据和检查点语义。历史数值与人工试听结论位于 `docs/records/`，平台差异、未通过项与实验入口必须如实记录。迁移后仍应执行独立协议握手、配置隔离、默认/扩展 CPU 构建及 GPU 编译检查。

构建产物不进入 Git；也不提交模型权重、用户参考音频、正文、凭据和临时运行输出。第三方源码更新保留固定 revision、许可证和对照夹具。PR 使用 Conventional Commit，说明行为、兼容和验证。

## 发行

只保留 dist-workspace.toml，使用 cargo-dist 0.32.0 的原生 Cargo 构建。运行 `dist generate`、`dist generate --check` 和 `dist plan`。`.github/workflows/ci.yml` 在三个平台构建标准包、解压至隔离目录并执行无模型、无 CUDA 的帮助/握手检查；Mac 标准包编译全部 Metal 路径。许可证由 `.github/scripts/prepare-notices.py` 汇集到 target/third-party-licenses 并随包分发。

协议库单独发布 `talechime-protocol-v*` 标签到 crates.io，应用标签使用 `talechime-v*`。CARGO_REGISTRY_TOKEN 与 HOMEBREW_TAP_TOKEN 由 GitHub Secrets 管理；Homebrew 发布到 yexiyue/homebrew-tap。正式应用标签需发布时再创建，普通验证不下载模型、不代替试听。

GNU release runner 为 Ubuntu 24.04，最低 glibc 2.39，ORT 1.28 预编译库不能在 Ubuntu 22.04 链接。Mac 标准包 macOS 15+，构建设置 MACOSX_DEPLOYMENT_TARGET=15.0；Metal 探测先枚举设备，避免无 GPU 的 hosted VM 调用 Candle 0.11 构造函数时 abort。非空 HOME 可用于隔离主目录。许可收集包括 Cargo registry 依赖和 pinned ONNX Runtime 的完整 notices。

## 文档归档

使用与架构指南直接放在 `docs/`，历史验收与实验放在 `docs/records/`，开发知识放在 `docs/knowledge/`，原生第三方声明放在 `docs/legal/`。[文档索引](README.md)是统一入口。搬迁记录时只更新路径引用，保留验收日期、设备、未通过项及原始结论；当前行为变化写入指南和知识库。


## 仅库构建

`talechime` 默认启用 cli 与 moss。嵌入库用 `default-features = false` 再选择模型 feature；cli 门控 executable、clap/crossterm 与终端控制依赖。无模型库测试可运行：

```sh
cargo test --locked -p talechime --no-default-features --test library
cargo clippy --locked -p talechime --no-default-features --all-targets -- -D warnings
```

无默认 feature 的二进制构建需显式加 cli，例如 `--features cli,qwen-metal`。core 当前仍编译播放器依赖，直接合成运行时不打开设备。
