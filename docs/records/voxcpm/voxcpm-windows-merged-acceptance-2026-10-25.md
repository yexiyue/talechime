# VoxCPM2 合并后验收（Windows，2026-10-25）

本轮只验收合并版本和既有优化，不增加性能候选。Windows / Ryzen7 9700X / RTX5070，
CUDA13、驱动616.56；main `1746d7232ab79367f1a8b9411a7f8e917a0a4fe5`加本地Windows优化，
Candle统一`884b359895a16c8fbd3647789b1f7c7974894708`。

冻结xtask `08cafd5a641a321ad0179cf3d7da13a63e3f022f19c9f3e25926d288713252c1`
执行`perf build`，features为performance-tools、voxcpm、voxcpm-cuda，源码guard通过。
新performance_probe SHA为`8d41742fb814c4652a6b55c1fab6697ae6ad4b6229397eab2717a1f9d9a52057`。
构建3m18s；复用相同target目录预编译测试后才开始测量，不与GPU推理并行编译。
沿用Rayon8和原baseline采集预算、环境及参数，不切换为profile/eager模式。

## 生产请求

Q8沿用[复制优化B臂](voxcpm-windows-key-packing-2026-10-25.md)的custom:acceptance；
BF16沿用[F32 decoder图B臂](voxcpm-windows-f32-decoder-graph-2026-10-25.md)的narrator。
两种精度各短句/段落，固定原语料、资源、seed、10步、CFG2、temperature1、
29.92秒预算、无style和continuation。四case各三独立进程，1预热+5热轮，共72轮。
Q8为base-q8_0/acoustic-f16/codec-f16，BF16为main-bf16/codec-f32；不跨精度比较排名。

| case | 三个进程热均值s | 均值s | 首PCM均值ms | 音频s | RTF均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| Q8短句 | 0.58926 / 0.61517 / 0.57727 | 0.59390 | 119.94 | 2.4 | 0.24746 |
| Q8段落 | 2.81618 / 2.70070 / 2.71081 | 2.74257 | 112.48 | 12.4 | 0.22118 |
| BF16短句 | 0.67460 / 0.64497 / 0.63028 | 0.64995 | 69.70 | 2.93 | 0.22183 |
| BF16段落 | 3.20155 / 3.22963 / 3.21194 | 3.21437 | 32.22 | 15.0 | 0.21429 |

对应历史四进程B臂均值依次0.63087、2.73370、0.69218、3.24812秒。
本轮没有持续性性能退化迹象；Q8段落点值约+0.32%，其余点值较低。
跨日期、不同进程矩阵不是同期A/B，不据此报告新提速或普遍无回退。

全部12进程success、exit0、cleanup_verified，采集flush_complete且无raw/trace丢弃。
Windows Job commit高水位Q8约5.453–5.460 GiB，BF16约6.413–6.423 GiB；
这是contained Job提交高水位，不是RSS或独占显存。

## 输出、状态与图路径

- 72轮输出摘要、音频时长与各自历史控制一致；360份完整/分段F32及PCM16 WAV
  的SHA/长度与旧资产一致。另72份ranges JSON按自身SHA审计，稳定元数据、
  文本SHA、分包样本边界和source_range与对应旧attempt一致，连续覆盖完整音频。
  合计432资产；consumer时间和raw_owner_events中的时序保留，不要求旧新相等。
- 两种精度分别在新进程回归owner状态：未消费取消、观察到容量1满队列后取消、
  未知音色拒绝、1秒预算部分PCM→Err→关闭且无End，随后完整PCM逐位恢复。
  预算负例均发出46080样本；Q8还验证custom:acceptance→narrator→custom:acceptance。
  两项分别8.65秒/8.05秒通过，不把满队列观察当作已证明线程正阻塞于send。
- 原始BF16模型做decoder off/off/on/on/off五次完整请求，CFM图固定启用；
  断言实际graph状态匹配、没有fallback。19个latent patch、145920原始PCM样本、
  chunk长度、自然EOS逐位保持，6.53秒通过。该raw音频为3.04秒，生产短句
  2.93秒含既有边界处理，不能据二者时长差猜测提前结束。
- Q8参考WAV/metadata/prompt cache三文件在状态测试后仍匹配冻结conditioning摘要。
  BF16使用内置narrator，不伪造参考文件。

新CUDA组合普通测试：backends17通过/2忽略，模型10通过/17忽略。
两crate release all-targets Clippy与rustdoc均-D warnings通过，fmt/diff检查通过。
既有loader的IntStorage提示和LNK4098链接警告保留，没有改加载策略或屏蔽警告。
人工试听、长篇接续/style和跨框架质量未由本轮关闭，官方历史对照不改写为新验收。

## 可恢复证据

外部目录`D:/workspace/talechime-benchmarks/windows-acceptance-20261025`：

- `voxcpm-build.json`固定生产二进制/源码；`voxcpm-tests-receipts.json`固定两个测试exe。
  backends测试SHA `fe5a17f8465d6c1d4a498662a3488993cff7cc1ba68010b485c2f4e7d737fd20`，
  模型测试SHA `608c09a070ed224ac831201b6588646ee5659d6355a3532d580d70c4b85b26ae`。
- 四case的`voxcpm-*-job.json`、plan/run、`voxcpm-acceptance-summary.json`及
  `summarize_voxcpm_acceptance.py`保留逐轮/进程统计和432资产SHA。
- `voxcpm-state-{q8,bf16}-*.log`、`voxcpm-bf16-graph-correctness.json`及日志，
  `voxcpm-conditioning-after.json`、普通测试/Clippy/rustdoc日志保存独立门禁。

```powershell
$r = 'D:/workspace/talechime-benchmarks/windows-acceptance-20261025'
$tool = Get-Content 'D:/workspace/talechime-benchmarks/windows-indextts-20261024/component-diagnostic-final-tool.json' -Raw | ConvertFrom-Json
# 先核对tool/build SHA并配置CUDA DLL路径；输出必须全新，保持Job环境和资源身份。
& $tool.path perf run --job "$r/voxcpm-q8-short-job.json" --output "$r/voxcpm-q8-short-replay-NEW"
# 其余case：q8-paragraph、bf16-short、bf16-paragraph，串行执行。
```
