# Kokoro 原生移植与官方对照：2026-10-27

状态：源码实验的本机 CPU/Metal 阶段验收完成；未提交、推送、发布。主工作区已有其他未提交修改，
本次不回退或归并这些修改。官方研究起点见
[可行性记录](kokoro-official-feasibility-2026-10-27.md)，当前用法见 [Kokoro 指南](../../kokoro.md)。

## 来源与实现

唯一声学参考是 [hexgrad/kokoro](https://github.com/hexgrad/kokoro/tree/dfb907a02bba8152ca444717ca5d78747ccb4bec)，
前端参考 [hexgrad/misaki](https://github.com/hexgrad/misaki/tree/fba1236595f2d2bf21d414ba6e57d25256afada3)，
权重是 [官方 v1.1-zh 固定版本](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/tree/01e7505bd6a7a2ac4975463114c3a7650a9f7218)。
没有使用社区 Rust Kokoro 实现作为源码或正确性参考。完整 SHA-256 与许可证在
[crate SOURCE.md](../../../crates/kokoro/SOURCE.md)。

`talechime-kokoro` 实现共享 12 层 ALBERT、双向 LSTM、时长/韵律/文本编码器、
AdaIN 残差解码器、Snake、NSF 激励与官方默认 TorchSTFT 约定。
开发导出折叠 weight normalization；第一例官方转换前后输出逐位一致。
运行时只读原生 safetensors、前端数据及清单，不依赖 Python。

CPU 使用 F32。Metal 是 CPU 编码/韵律/F0/FFT + Metal 解码器的混合路径。
完全 GPU 编码的微小 F0 误差会经过相位累计放大，原候选未达到门槛，故未保留为生产路径。
不同框架的 seed 不能生成相同随机张量；独立对照注入官方实际 phase/Gaussian draws。
生产请求使用本地 seed，不承诺跨框架逐位复现。

中文数据来自官方实际使用的 jieba/POS-HMM、pypinyin 大词典、数字规范化和
变调/儿化规则。运行时执行规则与词典，不使用测试句结果缓存。前端 22 例中
21 个支持的中文样本音素完全一致；1 例嵌入英文明确不支持。长文分段保留原始
UTF-8 范围并覆盖全部输入，不把规范化字符位置当成原文字节偏移。

## 独立数值门槛

同一官方 IDs/style/speed/随机激励，要求时长和输出长度完全一致，
各已捕获阶段及全模型 PCM SNR ≥40 dB。另用固定官方 F0/N/文本特征独立检查
解码器，要求 PCM SNR ≥60 dB。所有失败保留，不降低门槛。

开发环境实测为 Apple M4 Pro、24 GiB、macOS 26.6.2（25G83），
Python 3.11.17、PyTorch 2.7.1、Transformers 4.51.3、
Misaki 0.9.4、jieba 0.42.1、pypinyin 0.55.0、pypinyin-dict 0.9.0、cn2an 0.5.24。
Rust 使用 workspace 固定 Candle 0.11 fork 与 ORT rc.13 / Runtime 1.28。

原始资产、官方/原生中间张量、WAV 和完整日志位于忽略目录
`target/kokoro-official-research/`，不随源码提交。`export-v1` 是女声短句标准语速，
`case-female-slow` / `case-female-fast` 分别为 0.8 / 1.2 倍，`case-male-short`
为 zm_010 短句，`case-male-long` 为约 12 秒男声长句。导出工具和对照入口随源码保存。

### 已发现并保留的失败

- 最初完整 GPU Metal 候选 39.753 dB；强制 CPU InstanceNorm 的中间候选
  36.543 dB，均撤回。混合执行保留 CPU F0 路径。
- 最初 CPU 短句 40.666 dB、固定特征解码器 77.034 dB、男声短句 47.013 dB。
  扩展慢速样本失败于 38.941 dB，证明短句通过不能推断所有速度通过。
- 双精度统计归一化候选不能稳定改善全链，未保留。随后直接按官方 PyTorch
  CPU Welford/级联合并重做归一化，并添加独立无模型 PyTorch 测试行。
- GEMM 先算乘积再加 bias 与官方先填 bias、beta=1 的 GEMM 存在 F32 舍入差异。
  macOS CPU 显式使用 Accelerate beta=1 与官方卷积列布局后，慢速样本提升至
  42.504 dB；当时长男声仍为 39.169 dB，继续定位，未当作通过。

阶段张量误差、固定特征误差和整模型 PCM 误差分别报告。F0 的高 SNR 不保证
PCM 同样高，因为相位累计会放大有符号误差。没有以对齐、截短、重新采样或
舍入 F0 的方式提高对照数字。

## 可选 g2pW 消歧

参考作者 [GitYCC/g2pW](https://github.com/GitYCC/g2pW/tree/b9328ce8129c4717a70f16a2688542a0197541dd)
及[作者发布的模型](https://huggingface.co/YC-Chen/g2pW/tree/fb4784f7dce8466a53dd4f3552f5a4c5b657b90c)，
独立导出实际 ONNX 输入/概率，原生端使用 ORT CPU、2 intra threads、顺序执行、
graph optimization all。字符窗口 32、batch 32；tokenizer/字表/简繁单字映射均固定来源。

24 例包含姓氏、多音字、超过 32 字的上下文和空格。原生所有输入张量完全一致，
概率误差 ≤1e-5，所有多音字预测与官方一致。人工目标读音 23/24，
“解先生”仍误读 jie3，不能宣称通用姓名纠错。相对原版音素有 6 例改变，
含两种重复姓氏上下文，其他这些样本没有额外音素变化。

独立入口 `1.1-zh-g2pw` / `kokoro-g2pw` 明确选择；原版 `1.1-zh` 不加载该模型，
官方声学/前端夹具不被消歧结果污染。未训练模型，未针对测试句硬编码读音。

## 生命周期与边界

专用线程持有并销毁推理对象，请求和 PCM 通道容量均为 1。加载 future 取消时
owner 先 cancel 后 join；资源哈希每 64 KiB 检查取消。完整底层算子执行不可抢占。
PCM 接收端退出或 owner 关闭不会伪造 End。

CPU 实测加载取消含 join 约 4.5 ms；保持 PCM 接收端存活且队列已满时销毁
后端，约 38.5 ms 内 join，保留一个已缓存完整 PCM 后断开，未发送成功 End。
公开 Engine 的同 seed 重复、首次 PCM 前取消、取消后复用和 close 已在 CPU
原版/g2pW 与 Metal 原版执行；完整数值复测继续进行。

独立 Metal feature 的公共设备探测曾漏列 `kokoro-metal`，导致设备错误显示不可用，
已修复并通过真实 Engine 流程。Windows SSH 本轮连接超时；没有 CUDA
编译或 RTX 5070 验证证据。普通测试和本地模拟不能替代跨平台硬件验收。

性能日志中的单轮 trace 耗时包含中间张量保留；官方导出的热模型耗时不含同样 trace。
两者不能用于宣称提速。完整章节试听、主观音质、ASR/内容完整度和实际音频设备播放
本轮尚未验收。

## 最终复核

最后按官方 PyTorch 的标量/向量舍入顺序修正归一化：Welford 的标量合并、
LayerNorm/InstanceNorm 向量 affine 分别遵循官方操作顺序。独立无模型夹具验证
这些算子，不以整链分数挑选任意舍入方案。

同时确认 Transformers 4.51.3 的 ALBERT 默认使用 CPU SDPA，非 eager attention。
原生端跟随其 32/64-query、512-key 单块路径，在未归一化 exp 权重与 value
矩阵相乘之后再按行缩放；Kokoro 上限 512 IDs，无需跨 key-block 状态。
独立 PyTorch SDPA 夹具覆盖短序列与 193 长度的切块边界，最大误差门槛 1e-6。

最终完整声学对照如下，所有整数时长及输出长度完全一致，所有具有独立官方夹具的阶段亦通过
40 dB 门槛。Metal 为上述混合路径，不能记作纯 GPU 推理。

| 样本 | 音色 / speed / seed | 输出样本 / 秒 | CPU PCM SNR | Metal PCM SNR |
| --- | --- | --- | --- | --- |
| 春天短句 | zf_001 / 1.0 / 7 | 81,000 / 3.375 | 43.112 dB | 43.111 dB |
| 慢速短句 | zf_001 / 0.8 / 7 | 100,800 / 4.200 | 42.740 dB | 42.740 dB |
| 快速短句 | zf_001 / 1.2 / 11 | 70,800 / 2.950 | 40.987 dB | 40.978 dB |
| 男声短句 | zm_010 / 1.0 / 7 | 84,000 / 3.500 | 45.025 dB | 45.053 dB |
| 男声长句 | zm_010 / 1.0 / 19 | 284,400 / 11.850 | 41.953 dB | 41.950 dB |

最终固定官方中间特征解码器 SNR 77.0372 dB，全模型标准短句 43.1115 dB，
两个 opt-in 真模型测试通过。以上不等于全音色、全部输入或跨平台逐位一致。

普通验证结果：模型库 10 项通过、2 项真模型测试默认忽略；Kokoro/g2pW/Metal
后端组合 21 项通过；默认 MOSS + Kokoro/g2pW 组合 50 项通过；独立 CLI
组合的 facade、会话与协议测试合计 56 项通过；CI Python 脚本 16 项通过。
`cargo fmt --all --check`、相关三个包全 targets Clippy `-D warnings`、严格私有项
rustdoc、OpenSpec strict validate 通过。CI 已加入 CPU/独立 feature 与原生
CUDA/Metal 编译组合，但没有把尚未运行的远端 CI 当作本轮通过证据。

最后代码的加载取消含 join 15.04 ms，满 PCM 队列且接收端仍存活时 drop/join
39.54 ms，无假 End。原版 Metal 与 g2pW CPU 的 Engine 重复/取消后复用/关闭
再次通过。真实协议使用临时配置与显式资源根完成 Hello → PrepareModel →
ModelReady → GetConfig → Shutdown；未播放扬声器。

Engine 示例因分段与边界处理输出 2 块、66,240 样本；整句声学夹具为 81,000
样本，两条路径不能拿样本数直接作官方整链等价比较。文本范围覆盖通过独立测试；
音频内容完整度仍需后续试听/ASR，未以 generation completed 冒充 playback completed。

### 性能筛查

在未观测到外部构建的最终筛查中，原生路径固定 `VECLIB_MAXIMUM_THREADS=4`、
`RAYON_NUM_THREADS=4`。先执行一次带 trace 的数值预检，再一次同形状显式预热，
五轮不保留 trace。相同官方随机激励；排除加载、中文前端、随机数生成及设备播放。

| 执行路径 | 热轮耗时（秒） | 均值 / RTF |
| --- | --- | --- |
| 原生 CPU | 0.581295, 0.575310, 0.573480, 0.573950, 0.580250 | 0.576857 / 0.170921 |
| 原生 CPU+Metal 解码 | 0.670106, 0.662610, 0.671821, 0.656713, 0.637052 | 0.659660 / 0.195455 |

同机此前官方 folded F32、4 Torch threads 的三次热轮为
0.245520、0.243244、0.241618 秒，包含官方随机数生成，排除前端。
这些不是同边界 ABBAABBA 八进程比较，不能据此确认精确的官方相对性能比例；
本轮没有证明原生端比官方更快，Metal 也没有优于原生 CPU 的性能证据。
默认仍为 CPU。独立 WAV 和逐轮 JSON 保存在上述忽略的开发目录。

提交前的后续审查修复了资源准备取消、全角数字和 `eR` 儿化问题；新增对照与最终提交
边界见[审查记录](kokoro-review-2026-10-27.md)。本记录前面的未提交状态和 22 例前端结果
保留为当时阶段证据，后续结果不覆盖原来的失败过程。
