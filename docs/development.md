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

## 原生验收

迁移不升级模型、不改 PCM、EOS、取消、用户数据和检查点语义。历史数值与人工试听结论位于 `dev-notes/`，平台差异、未通过项与实验入口必须如实记录。迁移后仍应执行独立协议握手、配置隔离、默认/扩展 CPU 构建及 GPU 编译检查。

构建产物不进入 Git；也不提交模型权重、用户参考音频、正文、凭据和临时运行输出。第三方源码更新保留固定 revision、许可证和对照夹具。PR 使用 Conventional Commit，说明行为、兼容和验证。

## 发行

只保留 dist-workspace.toml，使用 cargo-dist 0.32.0 的原生 Cargo 构建。运行 `dist generate`、`dist generate --check` 和 `dist plan`。`.github/workflows/ci.yml` 在三个平台构建标准包、解压至隔离目录并执行无模型、无 CUDA 的帮助/握手检查；Mac 标准包编译全部 Metal 路径。许可证由 `.github/scripts/prepare-notices.py` 汇集到 THIRD_PARTY_LICENSES 并随包分发。

协议库单独发布 `talechime-protocol-v*` 标签到 crates.io，应用标签使用 `talechime-v*`。CARGO_REGISTRY_TOKEN 与 HOMEBREW_TAP_TOKEN 由 GitHub Secrets 管理；Homebrew 发布到 yexiyue/homebrew-tap。正式应用标签需发布时再创建，普通验证不下载模型、不代替试听。

GNU release runner 为 Ubuntu 24.04，最低 glibc 2.39，ORT 1.28 预编译库不能在 Ubuntu 22.04 链接。Mac 标准包 macOS 15+，构建设置 MACOSX_DEPLOYMENT_TARGET=15.0；Metal 探测先枚举设备，避免无 GPU 的 hosted VM 调用 Candle 0.11 构造函数时 abort。非空 HOME 可用于隔离主目录。许可收集包括 Cargo registry 依赖和 pinned ONNX Runtime 的完整 notices。

