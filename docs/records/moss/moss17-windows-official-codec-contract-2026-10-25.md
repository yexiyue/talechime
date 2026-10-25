# MOSS Local 1.7B：Windows 官方 codec 精度与容量对照

本轮修正官方参考适配器，不改变 Rust 推理或精度。尚无完整官方 TTS 热轮，不能宣布
达到或超过官方。外部证据根为
`D:/workspace/talechime-benchmarks/windows-moss17-20261024`。

## 官方入口的实际精度

固定 Local revision `12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e`、codec revision
`3cd226ba2947efa357ef453bcad111b6eafba782`。官方 processor 第237行调用
`AutoModel.from_pretrained` 加载 codec，没有默认传 dtype；README_zh 第384–388行
加载 processor 后只 `.to(device)`，没有 autocast。codec checkpoint 为 F32，主模型
单独选择 BF16。旧参考脚本强制 codec F16，属于非默认精度偏离；此前关于加载控制
“不改变精度”的表述仅指保持该旧适配器精度，不能解释成与官方默认相同。

官方 quantizer 的 decode_code 明确产生 F32，再进入卷积。codec 全部 F16 且没有
autocast 时，固定 codes 解码报 Float 输入与 Half bias 不匹配。这解释的是本轮
codec-alone 控制的失败；不能倒推早前尚未加载完权重的 MemoryError 原因。

## 最小组件控制

用冻结 Rust xtask `ce6f4d64ceb23a8488c358ae1935f3c2eb24f4169f60e5cba4971db850997f02`
串行运行 `codec-alone-{float32,float16}-job.json`。保留原 HF allocator warmup，
仅沿用隔离 Safetensors header IO 补丁。Torch 2.8.0+cu128、Transformers 5.0.0、
Safetensors 0.8.0+windows.headerio1，仍非官方推荐 Torch 2.9.1+cu128。

两个组件均逐一核对1600张量与 checkpoint 转目标 dtype 后的 CPU 位模式完全一致；
无 buffer。固定 codes 为 arange(128).reshape(32,4)%16，32层 RVQ、10秒上下文。
每组1次预热和1次热轮，均成功解码、cleanup通过。没有文本主模型、生成或 EOS，
不能当作端到端 TTS 基线或性能达标证据。

| 控制 | F32、无 autocast（官方默认 codec） | F16、显式 autocast（非默认） |
| --- | ---: | ---: |
| 参数字节 | 7,098,265,600 | 3,549,132,800 |
| CUDA allocated（加载与权重核验后） | 7,098,296,320 | 3,549,164,544 |
| Windows Job 提交峰值 | 9,685,209,088 | 6,232,064,000 |

Job 峰值是 contained-job commit，不是 RSS 或显存。每轮输出7680样本；跨精度
全部样本不同，max abs 0.0102166608、RMSE 0.0006011516、相对参考 RMS 的比值
0.003775934。两轮结果相同。固定 codes 不是可代表音质的完整语音，尚无听感门槛，
不能因此启用降精度。详细数据为 `codec-alone-precision-analysis.json`，加载指纹、
runtime、WAV 和 capture 位于两个对应 run/process-0 目录。另保留无 autocast 的
F16失败 `codec-alone-hf-control-run`，不从证据中删去。

Rust 当前 codec 使用 F32 quantizer 并在 decoder 前转换为运行 dtype，不能仅据
“F16”标签推定与官方整模型 F16+autocast 算法相同。本轮没有做跨框架 codec 等价验证。

## 完整常驻 GPU 基线的容量下界

Local checkpoint 实际556张量、3,060,606,464参数，BF16为6,121,212,928字节；
codec为1,774,566,400参数，F32为7,098,265,600字节。合计13,219,478,528字节，
超过本机 CUDA 报告的独立显存12,792,037,376字节，尚未计 CUDA context 和激活。
这是常驻独立显存方案的容量下界，不证明 WDDM 共享显存或 offload 必然不能运行；
本轮未测这些方案，也不把 CPU/offload 测速称作同条件 GPU 对照。

仓库 `scripts/perf/moss_local_cuda_reference.py` 恢复 F32 codec，并在加载权重前
读取分片头计算上述下界，保存 capacity-preflight.json，超限时明确拒绝本方案。
单文件、分片、重复张量拒绝、目录越界拒绝四项合成检查通过，真实头计数也与独立
结构审计一致。`default-codec-capacity-job.json` 经 xtask 实际执行并在权重加载前
拒绝，Job 提交峰值1,527,377,920字节、cleanup通过；capture记录 failure /
disconnected probe without terminal evidence，具体拒绝原因见 stderr 与容量文件。
该失败不是合成计时，不进入性能统计。

后续完整对照需明确选定更大显存机器、官方支持的 offload 或经单独质量验证的
非默认精度配置。默认入口、兼容栈、控制版及原生实现继续分别记录。
