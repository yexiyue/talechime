后续实施统一由 [finish-official-backend-parity](../finish-official-backend-parity/ownership.md) 负责；本清单保留原状态和证据，不作为重复实施队列。

## 1. 后端文本边界

- [x] 1.1 将现有 Kokoro 章节文本规范化移入 Kokoro 后端；验证源坐标及中英文/标点。
- [ ] 1.2 制定并运行跨后端文本矩阵，记录官方输入、token/音素、生成、音频听感及平台边界。

## 2. 已确认能力缺口

- [ ] 2.1 已实现官方自动语言及方言前缀，通过单元测试和隔离 Qwen 0.6B Base CPU 英文自动路径真实合成；CustomVoice、VoiceDesign、其他语言及听感仍待验。
- [ ] 2.2 已按官方未指定语义实现 OmniVoice 自动语言与设计路径；真实多语言输入待验。
- [ ] 2.3 已暴露 IndexTTS 时长倍率及文本规范化开关；官方语言、文本规范化、读音和情绪等剩余能力、资源许可及真实验证待完成。官方词表的多语言输入字节往返已验证，受损词表缺字节时拒绝加载。
- [ ] 2.4 已接入 CosyVoice3 跨语言模式并测参数转发；官方文本处理能力和真实模型待验证。
- [ ] 2.5 对照 MOSS Nano/Realtime 和 VoxCPM2 官方同版本，修复发现的遗漏并验证特殊文本与结束边界。
- [ ] 2.6 完成 Kokoro 官方混读音素对照、未知词和听感验收。

## 3. 交付

- [ ] 3.1 更新公开能力、依赖和知识库；所有相关测试、Clippy、rustdoc、真实模型及平台边界如实记录。

2026-11-12 后续：本机能力及 285 项矩阵交付见 `finish-official-backend-parity` 和
`docs/records/tts-core/official-backend-parity-2026-11-11.md`；本文件保留阶段快照，
不重复勾选后续负责项。用户听验失败和 CUDA 实机延期仍按各自后续清单跟进。
