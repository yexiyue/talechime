# 许可与上游来源

Talechime 原创项目代码沿用仓库根目录 MIT 许可证。第三方代码与权重分别按各自许可分发；模型不随源码仓库分发。

| 组件 | 来源记录 |
| --- | --- |
| Qwen3-TTS Rust 计算库 | `crates/qwen3-tts/SOURCE.md` 与组件许可证 |
| MOSS Candle 计算库 | `crates/moss-tts/SOURCE.md`、LICENSE / NOTICE |
| MOSS Nano 资源 | `crates/talechime-backends/src/moss/assets/LICENSE.OpenMOSS` 与固定资源清单 |
| VoxCPM Candle 计算库 | `crates/voxcpm/SOURCE.md` 与许可证 |
| VoxCPM 原生开发对照 | `crates/voxcpm-sys/native/SOURCE.md` 与 vendored 许可证 |
| OmniVoice | `crates/omnivoice/SOURCE.md` 与许可证；tokenizer 的 BOSON/Higgs/Llama 授权单独保留 |
| 已移除的 Qwen 对齐（历史归档） | [许可](legal/retired-qwen-alignment/LICENSE)、[固定资源清单](legal/retired-qwen-alignment/SOURCE.md) |
| ONNX Runtime 1.28 | [许可证](legal/onnxruntime/LICENSE)、[第三方声明](legal/onnxruntime/ThirdPartyNotices.txt)、[固定来源](legal/onnxruntime/SOURCE.md) |

历史验收资料中的项目名、模型实现和版本反映当时状态，以当前代码及固定资源清单为准。独立制品应携带所编译组件的 LICENSE / NOTICE / SOURCE.md，不能只附根 MIT。

发行前运行 `.github/scripts/prepare-notices.py`，将工作区和 Cargo 依赖声明汇集到 Git 忽略的 `target/third-party-licenses/`。cargo-dist 将其作为 `third-party-licenses/` 放入压缩包；ONNX Runtime 的完整原文随 `docs/legal/onnxruntime/` 分发。生成文件不在仓库根目录保留占位目录，固定上游声明仍纳入版本管理。
