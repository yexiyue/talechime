# Windows GPU 进程采集的编码故障

设备 Windows 11 / Ryzen 7 9700X / RTX 5070。Qwen Base 图候选的 500 ms
采样中，`nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory`
返回非 UTF-8 字节；原 `from_utf8` 拒绝整份列表，status 为 unavailable。
旧计时与限制保留于 [Qwen 图记录](../qwen/qwen-decoder-reference-graph-windows-2026-10-24.md)，
不因工具修复改写为“旧测量没有竞争”。

`xtask/src/perf/sampling.rs` 现在先接受有效 UTF-8；否则将非 ASCII 字节按原始顺序
显示为 `\xNN`，并在 reason 中记录编码说明。它没有猜测 Windows code page，
名称不能视为正确解码的本地路径；PID、行结构和显存字段仍严格解析。
不可用显存仍为 null，命令失败、超时或损坏数值仍使采集失败。
空列表只有在命令成功、格式有效时才表示此查询未列出 compute 进程；它不覆盖
所有桌面图形负载，也不保证后续合成期间没有新竞争。

普通测试覆盖非 UTF-8 名称、合法中文名称、损坏 PID/显存和空输出。
独立 ignored live_gpu_process_collection 使用真实 nvidia-smi，不加载模型。
4 项普通采集测试通过、1 ignored；显式实机测试通过，返回 20 个进程，
编码说明保留，逐进程显存全部为 null。旧轮次仍不追溯改写。
测试二进制 SHA256 `f704734f0a394457afb3800e48bffad5fdd4c29d8730a274df2ce245d172b026` 已冻结到证据目录 binaries。
release all-targets Clippy -D warnings、rustdoc -D warnings、格式与 diff 检查通过。
冻结的是测试二进制；后续正式 perf 编排仍须重新构建 xtask 并冻结新工具收据，
不能把旧 target/release/xtask.exe 视为已含此修复。证据存于
`D:/workspace/talechime-benchmarks/windows-qwen-20261024/gpu-process-encoding-*`。


## 后续正式报告暴露的数值 Reading 接口遗漏

VoxCPM 新生产基线的两种权重各 3×(1+5) 都完成且 PCM 与旧基线一致，但
新父工具把进程名称编码说明同时复制到 available 数值 count 的 reason。
离线 report 按 Reading::validate 拒绝“可用数值带失败原因”，因此首轮 run 不能
当成已通过完整报告门禁的交付，原目录及失败的部分 report assets 保留。
证据在 windows-voxcpm-20261024/budget-recheck-{q8,bf16}-run，父工具
budget-recheck-xtask-tool.json。不能通过删除原始日志字段或放宽 Reading 校验修复历史证据。

修正在 executor::competing_processes：当 count 可用时，数字 Reading 的 reason 为 null；
编码说明仍完整保留于 frame 顶层 competing_gpu_processes 明细。不可用 count 的
原因继续进入 Reading。新增跨接口单元测试同时检查数字 validate 成功、count=1、
明细的转义名称与编码说明未丢失。后续使用新父工具重跑同一个冻结生产探针与请求，
并要求 xtask 离线 report 成功；不重命名原始失败为成功。

这说明真实 collector 返回成功及 collector 自身测试不足以证明完整采集链验收：
还要通过数据模型校验和实际报告消费者。56 项 xtask 单元测试通过、1 ignored，all-targets Clippy -D warnings 与 rustdoc
通过。新父工具的 Q8/BF16 各 3×(1+5) 完成，两份离线 report 均成功；数值 count
可用且 reason=null，编码说明仍在每帧明细中。正式报告与冻结工具身份见
[VoxCPM 预算复核](../voxcpm/voxcpm-windows-owner-state-2026-10-24.md)后续生产基线节。
