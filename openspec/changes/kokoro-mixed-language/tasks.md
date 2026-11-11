剩余引擎官方对照、矩阵及听验交付由 [finish-official-backend-parity](../finish-official-backend-parity/ownership.md) 负责；宿主行为仍归宿主变更，既有状态保留。

## 1. 实现

- [x] 1.1 审核并固定离线英文依赖及许可证，更新 Rust 版本要求。
- [x] 1.2 实现规范化后中英转音素、词外错误与 g2pW 路径，确保切分与源范围正确。
- [ ] 1.3 宿主保留未知字符供引擎检查；其他后端是否静默忽略未知字符尚未验证。

## 2. 验证与记录

- [ ] 2.1 英文、缩写、未知词、非法文本、长文切分和 g2pW 已有测试；独立官方英文前端对照尚未完成。
- [ ] 2.2 真实模型与宿主集成（含 S600 / Apple）已完成；人工听验未完成。
- [x] 2.3 格式、相关测试、Clippy、rustdoc 和能力文档已更新；验证边界见 docs/kokoro.md。

2026-11-12 后续：本机能力及 285 项矩阵交付见 `finish-official-backend-parity` 和
`docs/records/tts-core/official-backend-parity-2026-11-11.md`；本文件保留阶段快照，
不重复勾选后续负责项。用户听验失败和 CUDA 实机延期仍按各自后续清单跟进。
