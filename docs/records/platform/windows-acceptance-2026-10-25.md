# Windows 优化版本验收与交付（进行中，2026-10-25）

当前目标是完成优化版本验收、回退检查和交付；仅处理已经证明影响体验的性能瓶颈。
不再以逐个模型寻找新优化作为完成条件，也不要求为已有明确阻塞反复重试OOM。

验收基线：main `1746d7232ab79367f1a8b9411a7f8e917a0a4fe5` 加本地Windows优化，
Candle统一`884b359895a16c8fbd3647789b1f7c7974894708`。每次构建另固定dirty源码
身份与二进制SHA；历史实验保留原版本。设备为本机Ryzen7 9700X / RTX5070。

## 有限验收范围

| 路径 | 验收工作 | 当前证据 |
| --- | --- | --- |
| MOSS Nano Candle | 常用短句/段落完整请求、首PCM、RTF和输出回退检查 | [合并后36轮通过](../moss/moss-nano-windows-merged-acceptance-2026-10-25.md)，180音频资产与旧控制一致；长文/试听仍待验 |
| Qwen CustomVoice/Base 0.6B/1.7B | 常用朗读性能与既有取消/接续/缓存状态回归 | [0.6B合并后36轮及状态回归通过](../qwen/qwen-windows-merged-acceptance-2026-10-25.md)，144音频资产与旧控制一致；1.7B因提交余量暂缓，长篇/试听待验 |
| VoxCPM2 Q8/BF16 | 短句/段落及既有decoder/CFM图、取消与复用 | [合并后72轮及状态/图门禁通过](../voxcpm/voxcpm-windows-merged-acceptance-2026-10-25.md)，360音频资产与旧控制一致；长篇/试听待验 |
| MOSS Local/Realtime 1.7B | 常用朗读及既有owner状态回归 | 合并前已有完整记录；合并后待补 |
| IndexTTS 2.5 | 常用参考朗读、情绪参数与缓存共同工作 | [修复Windows RoPE输出回退及情绪广播错误](../indextts/indextts-windows-merged-acceptance-2026-10-25.md)，18轮/72音频资产与旧对应轮一致，四种情绪与缓存联合通过；长文/试听待验 |
| OmniVoice | 常用朗读及CUDA图/decoder输出回退检查 | [54轮及四项CUDA门禁通过](../omnivoice/omnivoice-windows-merged-acceptance-2026-10-25.md)，288音频资产与旧控制一致；空间恢复后当前组合Clippy/rustdoc补齐通过，试听待验 |
| MOSS Nano ONNX CUDA | 动态ORT入口、实际运行库身份及完整请求 | 合并前主CLI播放/checkpoint通过；交付范围待核验 |
| CosyVoice3 | 正确HiFT资源与生产校验 | 本地6/7资源匹配；HiFT摘要不符，保持门禁，不冒充通过 |

## 验收原则与交付

- 复用固定语料、参考、seed、参数和资源清单，生成新收据/Job/结果；不重写历史证据。
  正式绝对基线每case三独立进程，各1预热+5热轮；先编译后串行测量。
- 首先核对完整输出、成功完成、实际流式边界，再比较逐进程均值/范围、首PCM、
  音频时长、RTF及内存。跨日期旧计时只作回退线索，发现可疑回退再做同期交错对照；
  输出一致不代表听感已人工验收，ASR也不能替代试听。
- 合并后的原生构建、274普通测试、Clippy/rustdoc通过见
  [main合并记录](windows-main-merge-2026-10-25.md)。其他构建组合按实际改动补门禁，
  已通过且源码/依赖一致的门禁不无条件重复。
- 最终交付包含当前源码身份、验收表、可复现命令、可恢复构建/补丁及明确阻塞，
  核对优化skill引用与适用范围。模型、音频、原始结果留仓库外；不自动发布PR或发行。
- 若新基准证明体验瓶颈，单独限定输入、假设、预算和退出条件再处理；
  不因局部慢就恢复全后端微优化。需要人工试听或外部资源的项目明确待办。

## 静态交付核验

- 当前OmniVoice探针与旧探针直接PE imports相同，含动态CRT依赖；源码包清单含
  CUDA源码、build.rs及对应许可，见[合并验收](../omnivoice/omnivoice-windows-merged-acceptance-2026-10-25.md)。
  这是直接依赖/源码包检查，不是完整传递依赖、标准发行构建或干净机器验收。
- 优化skill主文件74行，133个本地Markdown链接全部存在；skill-creator的
  quick_validate退出0。默认Python缺PyYAML，复用既有官方对照Python环境完成校验，
  未安装依赖。结构/链接检查不代替经验内容与模型结果审阅。
- 空间不足曾阻止续跑构建。用户明确授权两个CastGlean incremental目录后，原生
  PowerShell重试仍被自动审批以blocked by policy拒绝，工具未删除文件。随后外部
  清理恢复空间，OmniVoice未完成的工程检查已续跑；保留原失败和重试记录。

本轮外部收据/结果根：`D:/workspace/talechime-benchmarks/windows-acceptance-20261025`。
此表是待执行清单，不表示全部路径已验收或目标已完成。
