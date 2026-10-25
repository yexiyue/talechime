# Windows 工作区合入最新 main（2026-10-25）

按用户要求将当前main从`42a7887e28f095684266eac43a68bd8fcc303d11`快进到
`1746d7232ab79367f1a8b9411a7f8e917a0a4fe5`，随后恢复本地未提交的Windows优化。
新提交为`90cf8d0`（Mac Metal优化及复核）和`1746d72`（CosyVoice风格、IndexTTS情绪参数）。
未推送、未创建PR，Windows改动仍未提交。

合并前保存tracked binary diff、文件SHA清单和含untracked文件的Git stash，
外部目录为`D:/workspace/talechime-merge-backup-20261025`；保留stash，不自动删除。
恢复审计对非远端重叠的135个本地文件逐个比较内容（统一CRLF/LF后），无非预期变化；
11个本地文件与远端改动重叠，Git报告其中5文件冲突，均已解决。

解决方式：

- IndexTTS BigVGAN保留main的FIR系数预缩放和Metal快路径，保留Windows细分profiling；
  不重复执行原末尾affine，避免音量放大两次。历史`bv.fir_up.interleave_scale`名称
  暂保留，但此合并后范围只包含stack/reshape，缩放已在系数中。
- DiT保留两端的平台实现、测试和阶段诊断。外层`dit.wavenet`继续包含此前Windows
  报告定义的整个范围，新增内层名为`dit.wavenet.core`，避免同名嵌套累计。
- IndexTTS adapter同时保留Windows参考缓存和新版情绪参数；两个真模型测试请求
  适配到`params::Resolved`，包含明确style=None及原seed。
- 优化skill的Mac和Windows经验均保留。

Candle四个根patch统一更新到main的`884b359895a16c8fbd3647789b1f7c7974894708`。
相对旧`2266099d`的本地Git差异仅涉及Metal实现、Metal kernels及FORK说明，
没有CUDA源码差异；不能因此把旧二进制的性能结果直接记作当前源码测量。

Windows 11 / Ryzen7 9700X / RTX5070，CUDA13、cuDNN9.14下验证：

- release原生组合构建通过：IndexTTS CUDA/cuDNN、CosyVoice3 CUDA、OmniVoice CUDA、
  backends/performance-tools/ort-dynamic及xtask。
- 普通库测试：backends27、IndexTTS108、CosyVoice3 37、OmniVoice42、xtask60，
  共274通过；显式ignored不计入普通通过。
- BigVGAN真实权重、合成非连续mel的8→128→224→8帧、各1+5轮，24份原始F32LE
  文件SHA与合并前冻结控制全部相同；不是完整TTS内容/EOS验收。
- IndexTTS真实参考缓存的uncached/首次/命中/原地替换/返回、无效WAV、取消与复用
  完整PCM回归通过，34.94秒。旧Job同一参考路径列出两次，核验摘要后选唯一文件，
  不重写原Job。
- 同一原生组合Clippy all-targets及rustdoc均以-D warnings通过；fmt/diff检查通过。

`native-receipts.json`保存五个冻结测试exe及SHA，构建JSON/log、五套普通测试日志、
FIR输出与对照、缓存回归、Clippy/rustdoc和恢复审计均在上述外部目录。
初次编译发现旧测试字段未适配，其失败日志保留；后续r3构建为通过版本。
Mac硬件本轮未测，其他模型的完整性能矩阵和人工质量验收沿各记录保持待办。
