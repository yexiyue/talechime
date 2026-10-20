# Tasks

## 1. 来源与许可归档

- [x] 1.1 将 native 固定来源和适用第三方许可证、声明保留到 docs/legal/retired-voxcpm-native，并记录删除前 Git commit；逐文件比较确认原文与 revision 保留完整。
- [x] 1.2 更新 docs/licenses.md 和后端 README 的当前来源、许可与退役说明；核验每个新链接存在，Candle 和 tokenizer 的独立许可未被笼统覆盖。

## 2. 构建链退役

- [x] 2.1 删除 crates/voxcpm-sys，移除根 workspace member 及仅它使用的 cmake/cc 声明，使用 Cargo 更新锁文件；通过 cargo metadata --locked --no-deps 和 Vox 正常依赖树确认不再包含该包。
- [x] 2.2 清理 qwen-cuda.yml 的旧路径过滤、架构/开发说明、voxcpm README 基准入口以及知识库中的当前路径；rg 核验剩余提及仅为明确的历史或退役说明，保留其他组件需要的 native 构建工具与 docs/records 原始证据。
- [x] 2.3 运行 cargo test --locked -p voxcpm --lib -j 1，以及 talechime-backends 的无默认 feature、voxcpm feature 单测；确认不下载模型，记录通过与 ignored 数量。

## 3. 集成验证

- [x] 3.1 执行 cargo fmt --all -- --check、cargo check --locked --workspace -j 1，以及 talechime 的无默认 feature、cli,voxcpm 组合检查；记录实际结果和任何工具链阻碍。
- [x] 3.2 对 voxcpm、启用 voxcpm 的 talechime-backends 和 cli,voxcpm 产品组合运行相关 all-targets Clippy（-D warnings）及 rustdoc（-D warnings）；核验没有通过隐式 native 回退解决构建失败。
- [x] 3.3 核对最终 diff、来源归档、锁文件与引用扫描，确认无模型/音频/凭据入库，生产代码、资源身份和未完成的数值/听感/长期播放验收状态不变。

## 验证记录（2026-10-20，Windows）

- 所有 Cargo 验证使用 `--locked --offline`，构建使用 `-j 1`，未下载或运行模型。
- `cargo test -p voxcpm --lib`：10 passed / 5 ignored；ignored 为本地真实模型/oracle 测试。
- `cargo test -p talechime-backends --no-default-features --lib`：0 tests；增加 `--features voxcpm`：14 passed / 0 ignored。
- `cargo fmt --all -- --check`、`cargo check --workspace`、`cargo check -p talechime --no-default-features` 以及增加 `--features cli,voxcpm`：全部通过。
- `cargo clippy -p voxcpm -p talechime-backends -p talechime --no-default-features --features talechime/cli,talechime/voxcpm --all-targets -- -D warnings`：通过。
- 相同包与 feature 的 `cargo doc --no-deps --document-private-items`，`RUSTDOCFLAGS=-D warnings`：通过。
- Cargo metadata、正常依赖树、当前引用扫描与 git diff 检查通过；来源、两份许可证和 AUTHORS 与删除前 Git blob 逐字节一致，42 个相关文档链接存在。
- 集成日志保存在忽略目录 `target/retire-voxcpm-sys-checks/`。本次不宣称 CUDA/Metal 运行时、严格数值、听感或长期播放验收通过。
