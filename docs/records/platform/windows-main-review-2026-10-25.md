# Windows 改动合并、审查与简化（2026-10-25）

将本地 Windows 优化保存为 `002bfde`，随后合并最新远端 main
`57f90010c760ee7db7ba00cdb8ed99abaeda2d70`；合并提交 `859a277`。
审查范围为 `git diff 57f90010c760ee7db7ba00cdb8ed99abaeda2d70...HEAD`，
并覆盖本次审查后的修复。设备为 Windows / Ryzen 7 9700X / RTX 5070，
CUDA 13.0、cuDNN 9.14；Candle revision 保持 `884b359`。

## Standards

独立规范审查发现一项模块组织问题：MOSS 新增的内联 `cuda_mask_tests`
不符合 AGENTS.md 的 named module files 要求，已移到
`crates/moss-tts/src/transformer/cuda_mask_tests.rs`，保留原平台门控及可见性。
VoxCPM 时长换算常量另重命名为 `PATCHES_PER_SECOND`，明确单位，计算不变。
无剩余已确认规范问题。初审对 WaveNet guard 生命周期的疑问经完整上下文复核撤回；
原代码已经及时 drop，不修改该计时范围。

## Spec

静态需求审查未发现实现偏差。合并保留 main 的 Qwen 会话级因果解码状态，
本地 eager 对照改为只绕过 vocoder 图，不绕过因果历史。

随后真实 CUDA 对照发现一项输出契约问题：首次 vocoder 图回放时，codes 相同，
但 63,360 个 PCM 样本中 63,355 个位模式不同，最大绝对差约 0.00028455，
RMSE 约 0.00001644。图输入的连续副本改变了 eager 使用的转置视图布局。
修复将 time-major 底层连续布局作为固定图输入，在图内恢复原视图；缓存键区分
连续与转置 vocoder 模式，其他布局回退 eager，不改变原有数学或降低逐位门禁。
严格回归通过后关闭此项。旧版本性能与 PCM 证据仍只属于各自冻结版本。

## 验证

- `cargo test --locked --workspace --lib --tests --examples`：761 通过，41 ignored。
- 扩展 CPU 模型组合的 workspace/all-targets Clippy、workspace/examples/private-items
  rustdoc 均以 `-D warnings` 通过；格式与 diff 检查通过，15 个 Python 脚本语法通过。
- release CUDA/cuDNN 组合的六个 crate 普通库测试：480 通过，102 ignored。
  布局修复后再运行受影响的 Qwen/backends 普通库测试通过。
- 同一 CUDA 组合的 Qwen/backends Clippy（lib/tests，`-D warnings`）通过。
  Qwen CUDA 的 private-items rustdoc 另以 `-D warnings` 通过。
- 单独显式运行本地 Qwen Base 0.6B 真模型测试：
  `decoder_graph_large_clone_preserves_codes_pcm_and_eos` 通过，33 帧 codes、
  63,360 个 F32 PCM 位模式在 eager 与 miss/capture/replay 间完全相同，且自然 EOS。
  `decoder_graph_survives_cancelled_session` 通过；普通测试不读取这些权重。

日志位于本机忽略目录 `target/review-*.log`。初次 CUDA 链接因当前终端缺少
cuDNN LIB 搜索路径失败；补齐本次命令环境后成功。release 测试链接仍报告
LNK4098（LIBCMT 冲突）警告，未通过忽略链接库掩盖它；Qwen 产物直接 imports
包含 VCRUNTIME140 与 UCRT DLL。这不是完整传递依赖或干净机器验收。

本次没有重新测量性能排名，没有执行 Mac 验收或人工试听。CosyVoice HiFT、
长篇及未覆盖模型变体的历史待办仍保持，不把本轮检查扩展为全部音频质量验收。
