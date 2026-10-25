# Windows Safetensors pread 的元数据提交峰值调查

2026-10-24，Windows11 / Ryzen7 9700X / RTX5070 12GB，Torch2.8.0+cu128、
Transformers5.0.0；原隔离环境Safetensors0.8.0。仅本机Windows证据，没有Mac测量。
外部根：D:/workspace/talechime-benchmarks/windows-moss17-20261024。

## 问题与定位

官方MOSS Local先前在codec safe_open报1455。适配器的pread patch实际同时覆盖主模型
和codec，报错调用行没有backend参数，不足以认定漏出作用域或退回mmap。
这轮先独立打开真实codec分片，仅读元数据和各3个小切片，没有加载模型或生成音频。
显式framework=pt、device=cpu、backend=pread，记录GetPerformanceInfo系统提交、
GetProcessMemoryInfo的PrivateUsage及PeakPagefileUsage；后者是本进程提交高水位，
不是RSS、VRAM或Windows Job全子树峰值。

[固定v0.8.0源码](https://github.com/huggingface/safetensors/blob/a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6/bindings/python/src/lib.rs#L642-L676)
在选择pread存储之前先建立整文件copy-on-write映射以解析元数据。由此推断该临时
映射是Windows额外提交峰值的来源，和以下独立进程实测相符；不能把pread标签解释
为整个打开过程完全不映射文件。

## 独立IO控制

从上述revision保留完整Apache许可证；隔离源补丁仅在Windows+pread的元数据阶段
改用只读文件映射，其他backend与平台保持原行为。轮子显式标记
0.8.0+windows.headerio1，原python-pread环境不改；新python-header-io借用相同Torch/
Transformers依赖。Rust构建31.93s成功，Cargo.lock、patch、轮子SHA、许可证路径
保存在header-io-receipt.json。dumpbin确认动态VCRUNTIME140及UCRT依赖。
它是官方对照的外部IO实验，不进入Talechime生产依赖或发行配置。

| codec分片字节 | 原pread打开峰值增量 | 补丁峰值增量 |
| ---: | ---: | ---: |
| 4,998,259,168 | 5,009,014,784 | 10,637,312 |
| 2,100,202,560 | 2,104,459,264 | 4,468,736 |

每个表格数据来自新进程，以after_open峰值减before常驻私有提交；不是逐时刻采样，
也不把系统提交的背景波动算作本进程增量。原pread在打开返回后的常驻提交已很低，
只看返回后值会漏掉约整分片大小的高水位。轮子重新构建涉及编译器/解析依赖版本，
未另构建同锁文件stock控制；这不是无其他二进制差异的严格单因素性能排名。

12类合成张量含BF16/F32特殊位，全量及切片与原mmap字节一致；两个真实codec分片
各3个小切片、张量数量与原pread一致。没有宣称比较了全权重所有张量或完成模型
数值/听感验收。原资源文件和SHA门禁不修改。

最初混合pread/mmap诊断保留了末个切片，mmap文件存储随Tensor继续存活，导致后续
8GB预检拒绝；其结果不用于上表。v2清理引用但峰值计数不能重置，最终v3每分片新
进程得到可归属峰值。不能将同进程第二个文件继承的高水位当成该文件的新峰值。

## 完整官方加载仍未恢复

两次均由冻结Rust xtask perf reference串行管理、输入SHA固定，未并行编译/推理：

- local-official-header-io-smoke-run：8GB可用提交预检通过。新pread实际来源和参数已
  写入safe-open-trace.jsonl。主模型文件打开通过；codec在Transformers allocator
  warmup试分配约3.30GiB时CUDA OOM，无prepared/有效轮次，cleanup=true。
  失败前系统可用提交约2.29GB，错误同时报告约5.07GiB显存空闲，不能只归咎显存碎片。
  Job提交峰值11,490,668,544字节。
- local-official-header-io-no-codec-reserve-run：10GB预检通过，只在codec加载作用域
  跳过分配器预分配，不改其dtype/权重/计算。codec推进至约1293/1600张量后主机
  MemoryError，伴随bytearray导出缓冲异常；无prepared/有效轮次，cleanup=true。
  Job提交峰值11,432,972,288字节。codec加载前GPU allocated=6,122,016,768字节，
  系统可用提交2,307,317,760字节；详见codec-load-memory.json。

weights头核实Local556张量全部BF16，3,060,606,464参数、6,121,212,928字节；
codec1600张量全部F32，1,774,566,400参数、7,098,265,600字节。头摘要和统计保存在
checkpoint-header-dtypes.json。不能用1.7B标签估算全部权重，也不能因GPU约6.12GB
就断言BF16退回F32；这里未取得完整参数运行时dtype直方图。按声明codec F16估算，
两者仅参数就约9.67GB，另需上下文、激活、CPU暂存与驱动提交；并非容量保证。

当前结论：元数据瞬时提交已在隔离IO控制中大幅降低，但完整官方加载仍受内存约束，
没有官方合成基线或达标排名。没有修改系统页面文件、强制CPU推理、降低RVQ层数或
改变当时适配器的模型精度；也不反复在相同资源条件重试。下一步应量化逐张量物化/传输暂存，
或在有足够提交预算的环境重建同条件官方基线。源补丁目前只保留为诊断资产。

## 复现资产

- diagnose_codec_pread_commit_v3.py / diagnose_codec_header_io_commit.py：每次传入一个
  model-0000N-of-00002.safetensors文件名，独立进程运行；仅元数据/小切片。
- header-io-receipt.json及safetensors-header-io.patch：上游pin、隔离轮子/许可证/源码SHA。
- 两个local-official-header-io*job.json与对应run：真实失败capture、参数/内存trace。
- header-io-evidence.json：本轮脚本、IO统计、加载失败和构建证据的SHA目录。

普通测试没有引入模型下载；本轮未改Talechime运行时代码，不重复无关GPU性能测试。

后续精度审计发现旧MOSS适配器的codec F16并非官方默认F32。此前“不改变精度”
仅指与旧控制保持一致，不能扩称官方默认。默认参数容量及codec组件控制见
[MOSS codec合约](../moss/moss17-windows-official-codec-contract-2026-10-25.md)。
