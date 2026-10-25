# Qwen 0.6B 合并后验收（Windows，2026-10-25）

本轮验收既有优化，不新增性能候选。设备为 Ryzen7 9700X / RTX5070，CUDA13 /
cuDNN9.14 / 驱动616.56；main `1746d7232ab79367f1a8b9411a7f8e917a0a4fe5`
加本地 Windows 优化，Candle统一 `884b359895a16c8fbd3647789b1f7c7974894708`。
1.7B按用户“先完成可运行的检查”暂缓，不将0.6B的结果外推至1.7B。

## 构建与完整请求

冻结xtask `08cafd5a641a321ad0179cf3d7da13a63e3f022f19c9f3e25926d288713252c1`
执行perf build，features保持indextts、indextts-cuda-cudnn、performance-tools、
qwen、qwen-cuda；没有去掉cuDNN改变历史比较的依赖路径。源码guard通过，dirty
身份 `c5a776c9e8947b793ebc5325feb023211d25d2c6246b8d82f4a42c46f49135e6`，
新probe SHA `f448f1cfeeb310f446a4f60ad53948de86a8dc8245c489271c96636ae550f0ac`。
先完成测试预编译，再串行测量；CUDA图沿用生产默认，没有强制eager或改变精度。

固定历史中文短句、seed42经会话解析、Rayon8、无style/continuation、原资源清单。
CustomVoice使用uncle_fu、repetition_penalty1.05；Base使用custom:acceptance06、
penalty1.5、chunk_frames20。两者都是talker/code-predictor BF16、codec F32。
各三独立进程、各1预热+5热轮，共36轮，均success、exit0、cleanup_verified；
采集flush_complete且raw/trace无丢弃。

| case | 三个进程热均值s | 均值s | 首PCM均值ms | 音频s | RTF均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| CustomVoice 0.6B | 1.121936 / 1.082538 / 1.094098 | 1.099524 | 632.75 | 2.19 | 0.502066 |
| Base 0.6B | 1.242566 / 1.176239 / 1.183399 | 1.200735 | 718.47 | 2.52 | 0.476482 |

历史CustomVoice三进程均值1.113747秒，Base既有large-decoder B臂四进程均值
1.213668秒。本轮未出现明显性能回退线索；跨日期矩阵不能证明新的提速或普遍无回退。
Windows contained Job提交高水位分别5.159–5.197 GiB和7.602–7.604 GiB，
不是RSS、物理内存或独占显存。

36轮输出摘要/音频时长与各自历史控制一致。144份完整/分段F32及PCM16 WAV的
SHA/长度与旧控制相同，另36份ranges JSON核对自身SHA、稳定元数据、source范围、
连续分包边界及完整音频覆盖；合计180资产。旧资产按确切attempt_id定位，时序字段
保留原值，不要求consumer时间或raw_owner_events时序跨次相等。

## 状态回归

两个owner测试分别新进程串行执行，CustomVoice 10.62秒、Base 39.86秒通过：

- 未读取消、观察到容量1满队列后取消、非法音色及预算失败后，完整63360原始PCM
  位模式恢复；预算失败保留24960样本部分输出，随后Err和关闭，没有伪造End。
- CustomVoice验证uncle_fu→serena→uncle_fu。带style状态门禁属于Custom17，本轮未执行。
- Base三段接续两次逐段PCM一致，分别48000、78720、80640样本；链中取消后恢复。
  静态参考与临时续接上下文分别检查，不以一次完整请求代替多段状态门禁。
- Base第二参考59520样本，A/B/A、未读/满队列取消后恢复；旧owner释放并join，
  仅隐藏测试拥有的WAV副本，新owner实际从磁盘prompt缓存恢复B及原A，逐位一致。
  临时参考通过存储API移除；四份既有参考/metadata/cache及固定生成夹具摘要不变。

上述状态与输出一致不代表长篇音色稳定或人工听感已验收。

Base06另以两个独立模型测试验证decoder图：形状/会话切换、容量界限、旧输出保活及
常量表淘汰后回放保持逐位输出，15.54秒通过；完整克隆请求33帧codes、63360原始
F32位模式在eager与图miss/capture/replay间一致，确认自然EOS及106帧图实际录制，
6.57秒通过。该直接模型输出包含宿主边界处理前的样本，不能拿它和生产WAV时长
差异推断提前结束。源码断言129帧不录制，未提高图预算掩盖退化。

## 检查与未完成项

新组合模型普通测试221通过/12忽略，backends30通过/6忽略，共251项通过。
两crate release all-targets Clippy和rustdoc均以-D warnings通过。
fmt和git diff --check通过。图测试命令、退出状态及日志分别保存在
`run_qwen_base06_numeric.py`、`qwen-base06-numerical-status.json`和对应日志。
1.7B的新Job/plan已准备但没有启动；历史Base17 Job提交峰值约9.445 GiB，
本轮提交余量不足所需安全空间。保持原模型、图预算及精度，不通过降低工作量过门禁。
VoiceDesign、1.7B、长篇和人工试听仍待验，本轮不是新的官方Python性能比较。
既有LNK4098 CRT冲突警告保留，未屏蔽或宣称解决。

外部证据根为 `D:/workspace/talechime-benchmarks/windows-acceptance-20261025`：
`qwen-build.json`固定源码/二进制，`qwen-tests-receipts.json`固定模型测试SHA
`bbc3df4fc73d2c775132f791dffc02262799f8fcc2276da77718d619b4aa5090`
和backends测试SHA `e86013f158e5de49754f0503645865b0b877e233446bf97865c8e0d5187c4d14`。
`qwen-{custom06,base06}-{job,plan}.json`、run、state日志及
`qwen-acceptance-summary.json`保存逐轮统计与资产审计；
`run_qwen_feasible_checks.py`记录串行命令，`qwen-feasible-checks-status.json`
记录退出状态，`qwen-conditioning-*-state.json`与`qwen-17-deferred-preflight.json`
记录输入、清理与未启动条件。模型、音频及原始结果保留在仓库外。

```powershell
$r = 'D:/workspace/talechime-benchmarks/windows-acceptance-20261025'
$tool = Get-Content 'D:/workspace/talechime-benchmarks/windows-indextts-20261024/component-diagnostic-final-tool.json' -Raw | ConvertFrom-Json
# 核对tool/build SHA及CUDA/cuDNN DLL环境，保持Job资源与参数，使用全新输出。
& $tool.path perf run --job "$r/qwen-custom06-job.json" --output "$r/qwen-custom06-replay-NEW"
& $tool.path perf run --job "$r/qwen-base06-job.json" --output "$r/qwen-base06-replay-NEW"
```
