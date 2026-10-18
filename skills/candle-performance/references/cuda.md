# CUDA / cuDNN / CUDA Graph

## 算子候选

确认当前 Candle、cudarc、Toolkit、cuDNN 和设备架构的组合。编译、设备可创建、库能
加载与目标算子能执行是不同证据。检查 grouped conv 是否逐组分派、FIR 是否逐 tap
重复遍历、KV 是否反复复制、固定位置/mask 是否重建，再选择小实验。

IndexTTS 曾参考 Candle depthwise PR 做局部实现，保留其几何与 workload 选择条件；
不能只抄数学公式而丢掉 fallback、bias、padding 或选择门槛。FIR 可把 (B,C,L) 变为
(B*C,1,L) 复用同一个核，但 kernel 方向、相位、裁剪和大形状都要对照。

cuDNN auto 在小输入通过，不保证长音频容差。算法选择影响精度与性能，先做固定输入
算法探针，再明确局部选择；不因误差变化就断言由 TF32 导致。
NVIDIA_TF32_OVERRIDE=0 可用于单独的严格 F32 诊断，记录为不同运行配置，不能把其
验证结果冒充默认配置验收，或悄悄将全局精度修改混入性能优化。

## 图边界与所有权

优先固定形状、无主机读回的子图；固定循环可展开，跨帧动态 KV 需要独立设计。
先核对当前 API；下列经验在 Candle 0.11.0 / cudarc 0.19.10 下验证过，不承诺其他版本
接口或 allocator 行为相同。

首选目标是 **dispatch-bound 区域**：每帧数百个极小 kernel 的子图，其成本主要是
主机分派而非算力，一次 launch 即可整体消除。判断方法是在无插桩构建里按 API 边界
统计主机墙钟与每帧 kernel 数，而不是看 GPU 利用率——被 CPU 饿着的 kernel 队列
利用率同样低。已验证实例：Qwen CodePredictor 每帧约 34 ms 主机墙钟（2-token
prefill + 14 步），整帧成图后端到端完整合成 -38.0%、流式 -33.3%，输出逐字节一致。
反之，帧内大算子主导的区域成图只省少量 launch，不要先做。

录制前沿实际调用链检查以下前提，而非只看入口 shape：

| 检查 | 决策依据 |
| --- | --- |
| 主机控制流 | 循环次数、offset、分支是否对该特化固定；设备值驱动的 Rust 分支或读回不能直接留在图内 |
| 实际缓存类型 | 若设计依赖固定 KV 地址，确认分配没有静默回退其他实现；不能只看到 new_kv_caches 就假定是预分配。已实测：Qwen 的 `Concat` 回退（K/V 由 `cat` 增长、capture 内分配且被缓存保留）会让驱动在 `end_capture`/实例化就以 `CUDA_ERROR_INVALID_VALUE` 拒绝录制；预分配变体正常。结论是“图在 capture 之后读写的一切都必须预先分配”，不是“所有 Concat 图都不可能” |
| 主机与设备状态 | reset/current_len 等 Rust 更新只在录制时执行，replay 不会重新运行；用不变式判定：**每次回放是否重写图内稍后读取的每个字节**。全量重写则无需图内重置（Qwen CP 每帧重写全部 16 个 KV 位置，容量 17 只是上限）；跨帧卷积历史不被重写，必须改为图内更新的持久状态，否则只能终止请求 |
| 缓冲接口 | 查 recorder 的具体 dtype 分支和逻辑字节长度，包括 token 的 U32/I64；浮点图可用不代表整数输出可复用。整数输出需要 recorder 的 storage-slice 支持，不要用“转浮点再转回”绕过接口限制——那是数值-changing 绕路 |
| 特化身份 | shape/dtype/device 之外，捕获的分支、模型资源、缓存实现等变化是否要求重录；优先用 owner 隔离，避免庞大通用缓存键 |

固定步数循环可以展开成不同长度的固定节点序列，不要求每个节点具有相同 shape。
反过来，固定容量也不代表有效 KV 长度固定。若完整子图每次覆写其全部有效前缀，
可另行证明无需重放主机 reset；不要机械地把所有主机字段改为设备标量。

- 固定输入、输出、状态缓冲在图外分配。recorder 持有录制流，使复制与 launch 同流。
  连续 view 仍可有 storage offset，按 start_offset 和逻辑字节数复制并校验设备/布局。
- Candle h2d 参数缓存在 enable_cuda_graph_htod_cache guard 生效期间填充，预热也
  要在 guard 内。首次 capture 只录制不执行，返回真正算过的 warmup，不交付未执行输出。
- 捕获的临时 alloc/free 应配对。Vox 将图内结果复制进图外输出、在 end_capture 前释放
  临时结果，解决重复 launch 问题；不先假设必须私有 mempool 或修改 Candle allocator。
  CUDA AUTO_FREE_ON_LAUNCH 处理下一 launch 前遗留分配，不是在上一 launch 末尾。
- 配平的对偶：**capture 内分配、capture 后仍被主机状态持有的张量同样破坏配平**。
  症状可能是实例化即 `CUDA_ERROR_INVALID_VALUE`（Qwen Concat KV），也可能是
  capture 成功但只能 launch 一次（Vox 早期输出缓冲）；失败 capture 后必须验证流
  已退出捕获态（RAII），否则后续录制静默损坏。
- 图持有 schedule、权重、流及其他被捕获地址的资源；steps 切换和 LRU 淘汰不能释放
  其他有效图仍引用的张量。销毁顺序必须保证这些地址直到图销毁仍有效。
- scratch 输出在下一 replay 会被覆盖，消费或复制后再回放。状态快照与恢复使用同一
  状态枚举；有状态 warmup、capture、replay 只推进一次，请求开始明确 reset。
- 取消和 host profiling 同步留在图外合适的边界，验证取消后可复用及公共延迟语义。
  profiling eager 与 graph 的计时不能混为一组。
- RAII 收尾失败 capture。launch 失败若无法证明状态未推进，终止当前请求，不能重算
  同帧并重复推进。上下文被异步非法访问破坏时，不承诺静默恢复 eager。
- 失败策略按状态性分档，不要一刀切：
  | 区域状态 | launch 失败 |
  | --- | --- |
  | 每帧自覆写、无跨帧持久状态（本帧 CP） | 警告 + 本 owner 永久停用图 + 本帧 eager 重算 |
  | 跨帧持久状态、无法证明未推进（AudioVAE decoder） | 终止当前请求；下一请求显式 reset 后 eager |
- `profiling`/插桩 feature 下停用图：逐步 span 会把 replay 压成一次 launch，
  trace 失真。正式基准用无插桩 release 构建，两组分开记录。

## 归属范围与共享边界

- 图和被捕获的状态归**每次生成**：完整合成的每次调用、流式的每个 session。
  不放进全局模型的可变缓存（跨请求共享地址），也不放进每帧新建的临时值
  （每帧重录）。每调用捕获一次仍在该调用的所有帧上摊销。
- 出现第二个真实使用者且各自通过性能门槛后，才把设备机制提取为独立 crate：
  已验证形态是 `talechime-candle-cuda-graph` 只含 capture/replay、输入覆盖、
  输出回拷、缓冲与流生命周期；模型策略（缓存键、状态保活、取消、失败分档）
  留在模型私有模块。提取本身是纯移动，用消费者原有测试（最好有免权重真 GPU
  测试）复验。
- 机制 crate 依赖 Candle 的 `cuda` feature 来获得 cudarc 的库加载模式，而不是
  自己挑 `static-linking`/`dynamic-loading`；并导出小 helper（如 `is_capturing`），
  让消费者测试无需直接命名 cudarc，消费者因此不必直接依赖 cudarc。

最小测试阶梯（Qwen CP 帧图按此顺序全部通过）：

1. CPU 单测：session 包装器与直接调用的 eager 输出逐位一致（无 GPU、无模型）；
2. 真模型真 GPU 测试：多组输入状态 replay 与 eager 严格相同；跑完若干帧后回放
   早期帧仍一致（验证全量重写不变式）；故意用错误宽度输入触发 launch 失败，
   断言图停用且后续帧仍与 eager 一致；
3. 端到端产物：codes、原始 PCM、WAV 与改动前 eager 基线逐字节一致；
4. 正式 ABBA（锁频、丢弃预热、报 mean-of-means 与区间、区间重叠即判无收益）；
5. 提取共享机制后复跑消费者的免权重真 GPU 测试 + workspace 编译/Clippy/fmt，
   并记录哪些真模型测试因缺权重未复验。

另有：特化 A→B→A、LRU 淘汰、非零偏移 view、录制失败清理、取消、跨请求 reset
按适用性补充。忽略的真 GPU 测试要用 `--test-threads=1`：同一进程内多次
`Device::new_cuda` 共享同一条流，并发 `begin_capture` 会互相破坏（并行失败、串行
通过的现象要先怀疑流共享，而不是被测代码）。缺少硬件时不声称通过。

## 不把布局变化视作免费

分组 GQA、去 K transpose 的 contiguous、填充静态 KV 虽数学等价，可能更慢或改变
GEMM 形状和舍入。量化模型下少量隐藏态误差也可能改变 EOS；先做独立候选，不一次
叠加多种改动。Qwen 的固定 15 步 CodePredictor 图已实现并通过（见案例索引），
但这只证明该模型该输入；同一循环在别的模型上仍需重新核对缓存变体、状态与输出
dtype。

图接口与释放语义核对 [NVIDIA CUDA Graph Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)，
并读目标版本的 Candle/cudarc 源码，特别是分配、event 和 storage 生命周期。
