# 已退役的 VoxCPM 原生开发对照

2026-10-20 退役 `crates/voxcpm-sys`（llama.cpp-omni / ggml C++ 开发基准）。生产适配器继续使用独立的 [Candle 计算库](../../../crates/voxcpm/SOURCE.md)，不提供旧引擎选择入口。

完整旧源码、构建脚本和探针保存在删除前的 Git commit：
`30339e67c44a3e000f7d1259ec038bf7daff18e3`，路径为 `crates/voxcpm-sys/`。
需要复现历史基准时，在该 commit 的独立 checkout 中构建
`cargo build -p voxcpm-sys --features cuda --example voxcpm_native_benchmark --release`；
设备、工具链、模型和参数以对应的[历史记录](../../records/voxcpm/voxcpm-candle-acceptance.md)为准。

以下文件逐字节保留删除前内容及固定上游 revision：

- [SOURCE.md](SOURCE.md)：原始来源、补丁范围和权重身份。
- [vendor/LICENSE](vendor/LICENSE)：vendored llama.cpp-omni 许可证。
- [vendor/AUTHORS](vendor/AUTHORS)：上游作者声明。
- [vendor/vendor/cpp-httplib/LICENSE](vendor/vendor/cpp-httplib/LICENSE)：cpp-httplib 许可证。

本目录不含可执行实现、模型或音频。历史测量仍保留于 `docs/records/voxcpm/`；
退役开发对照不表示严格官方数值、人工听感或长期播放验收已经通过。
