# IndexTTS FIR 布局候选撤回（Windows，2026-10-25）

Windows 11 / Ryzen 7 9700X / RTX 5070 12 GB，Candle 0.11 fork
`2266099d92265112ecc0d56ecee1761e3cc2ac00`，CUDA13 / cuDNN9.14。
沿[细分诊断](indextts-windows-fir-stages-2026-10-25.md)验证两个保持数值的候选，
均未采用。没有新增生产优化、默认开关或精度取舍。

## 局部初筛

原 `pad_with_same` 把左右边界逐个加入 cat。候选把边界 narrow 后广播为左右块，
再与输入做三块 cat。另一候选把 odd/even phase 先 stack 到倒数第二维，
transpose+contiguous 后 reshape/affine；原来直接 stack 到最后一维。

测试为独立 ignored CUDA 测试，三独立进程；五个形状
`(1,512,32)/(1,128,2048)/(1,32,14336)/(1,8,57344)/(2,9,111)`，
输入为较宽 tensor 的 narrow 视图，保留相位裁剪的行步幅，包含正负零及小值。
每种操作1预热+5热轮，每臂20次，轮间交替顺序，边界同步；读回和比较在计时外。
三种操作、五形状均逐位一致，但这些不是实采的模型中间激活。

15个 shape/operation 组合在三个进程中：padding 两种边界5/5、5/6的耗时
下降约36.5%–58.1%；相位交错候选约−5.7%至+4.4%，没有稳定收益，直接拒绝。
不从cat输入数量推断kernel数，也不将padding局部改善外推为vocoder提速。

## 真实权重完整 vocoder 复查

仅将padding候选接入 **cfg(test)线程内guard**；非测试构建从未启用候选。
使用同一 BigVGAN 模型实例、F32权重、ImplicitGemm算法；参数/配置SHA与前述
细分诊断相同。输入为确定性合成mel、narrow后非连续，按8→128→224→8帧切换。
三个独立进程，每case每臂1预热+5热轮，轮间交替执行顺序；
`INDEXTTS_PROFILE=0`，不把同步细分范围当作候选完整阶段计时。
先编译再串行推理；每个输出在计时后写raw F32LE。

| mel帧数 | 进程 | 原路径均值ms | 候选均值ms | 耗时改善 |
| --- | ---: | ---: | ---: | ---: |
| 8，首次 | 1 | 62.514 | 61.486 | +1.64% |
| 8，首次 | 2 | 61.885 | 58.164 | +6.01% |
| 8，首次 | 3 | 69.048 | 68.947 | +0.15% |
| 128 | 1 | 100.068 | 106.129 | −6.06% |
| 128 | 2 | 96.390 | 103.672 | −7.56% |
| 128 | 3 | 101.279 | 105.297 | −3.97% |
| 224 | 1 | 153.298 | 172.632 | −12.61% |
| 224 | 2 | 151.749 | 176.330 | −16.20% |
| 224 | 3 | 151.195 | 175.657 | −16.18% |
| 8，再次 | 1 | 78.599 | 69.595 | +11.46% |
| 8，再次 | 2 | 79.033 | 68.836 | +12.90% |
| 8，再次 | 3 | 71.315 | 62.996 | +11.67% |

144次完整输出均与相应原路径基准逐位一致，外部72对原/候选文件SHA再次独立匹配。
各输出长度为mel帧数×256且有限。短输入存在明显抖动和前后热状态差异；
这里是组件初筛，不是生产Registry基线、正式ABBA、置信区间或根因证明。
长形状三个进程方向均不利，停止扩大；没有为了局部收益接入生产模型，
也没有以整链10%硬门槛作为拒绝理由。

下一步若继续追查，需实采真实中间layout及分配/复制行为，解释为何局部复制收益未能
在完整阶段复现；当前不将其归因于已证明的带宽、分配器或cuDNN选择变化。
不为这两个候选新增框架或持久缓存。官方同精度对照及质量未完成项保持原状态。

## 撤回、工程门禁与证据

临时 `bigvgan/layout_tests.rs`、guard、测试分支和fir_pad包装全部撤回。
`bigvgan.rs` SHA恢复为前轮已通过真实诊断/Clippy/rustdoc的
`9481cc8c55f331e0b843e8b285b12687f50cc1d93836e29a78dcb7b7cbc7c489`。
冻结的对应二进制重新运行普通测试：106通过、23忽略。
源码恢复身份和fmt/diff检查通过；Clippy/rustdoc沿用与该源码匹配的上一轮通过证据，
不将已删除候选的测试二进制冒充当前生产构建。

外部根目录 `D:/workspace/talechime-benchmarks/windows-indextts-20261024`：

- `fir-layout-build-receipt.json`、`fir-layout-test-source.rs`、
  `fir-layout-bigvgan-source.rs`，三个独立stderr日志及`fir-layout-summary.json`。
- `fir-padding-real-build-receipt.json`、两个源码快照、三份stderr日志及三个output目录，
  `fir-padding-real-summary.json`保留逐轮、进程均值和72对PCM SHA。
- `summarize_fir_layout.py`与`summarize_fir_padding_real.py`为冻结离线汇总，
  `fir-layout-withdrawal.json`及`fir-layout-withdrawal-ordinary-tests.log`记录恢复身份。

恢复实验应使用相应收据中的冻结exe，不重启已撤回代码的生产开关。
局部完整测试名为 `bigvgan::layout_tests::cuda_layout_candidate_screen`；
真实权重为 `bigvgan::layout_tests::cuda_real_vocoder_padding_screen`，
使用 `--exact --ignored --nocapture --test-threads=1`。后者要求本地已核验的
`INDEXTTS_CONVERTED_DIR`、全新 `INDEXTTS_PROFILE_OUTPUT`、`INDEXTTS_PROFILE=0`，
CUDA/cuDNN DLL路径同前轮；无下载。原始mel不是有效文本合成，不能拿这些输出做语音质量验收。
