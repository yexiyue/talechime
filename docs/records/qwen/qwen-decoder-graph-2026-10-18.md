# Qwen 流式 CUDA 解码图与优化审查

## 范围与设计

实现固定点为 `11ceaba`。本轮承接 `11df568`，审查 CP 图、解码器常量缓存、图 recorder 和测量探针，
实现 12Hz 解码器的流式图回放。CodePredictor 图继续按 session 持有 KV；
12Hz 解码器每次输入已经包含历史 codes，没有跨调用的内部卷积/KV 状态，
因此图可由模型持有并跨会话复用。这两种生命周期不能合并。

按 KISS、DRY、YAGNI、SOLID 检查后保留的边界：

- `decoder_12hz.rs` 保留模型算术和 eager 入口；完整合成不进入图缓存。
- 私有 `decoder_12hz/tables.rs` 管 RoPE/滑窗掩码的 LRU，最多 4 项、8 MiB；
  超过字节上限的单次表可以使用，但不会留在缓存中。
- 私有 `decoder_12hz/graph.rs` 管形状准入、串行回放、淘汰和失败策略。
  只接受 `[1,16,T] I64`、`1 <= T <= 64`，最多保留 4 项；同形状第二次出现
  才预热和录制，返回实际计算过的 warmup，第三次开始回放。
- 复用已有 `candle-cuda-graph::Replay`，没有新增通用 trait、依赖或 Candle API。
  图固定持有 cos/sin/mask，先于权重销毁。模型锁覆盖输入写入、launch 和输出复制，
  返回独立张量，调用方不会拿到下一次回放即被覆盖的 scratch。

`QWEN_DECODE_GRAPH=0` 关闭解码图；CUDA 默认开启，profiling 强制 eager。
CPU/Metal 保持 eager 解码。capture 失败后关闭本模型的解码图并验证流已退出
捕获态，返回 warmup；launch/copy 失败关闭图并传播错误，避免承诺已损坏 CUDA
上下文仍能继续工作。流式取消仍在解码前由 session 检查；历史提交规则未变。

## 审查发现与修正

1. CP 图真模型测试的“eager”也创建了可启用图的 session，后续轮次可能比较
   两份图。改为直接调用 `generate_acoustic_codes`，保持 eager 基线独立。
2. 旧表缓存只限 16 项，`T²` 掩码仍可长期占用大量显存。改为同时限制条目与
   字节，按 LRU 淘汰；图显式保活常量，不依赖表恰好还在缓存中。
3. 旧取模查表只有 3072 项，引入了解码器与 Talker 词表的耦合，缩小了原来
   I64 输入可表达的范围。CUDA 改用整数 `c - (c / n) * n`，其他设备沿用原来
   的主机取模，不改变浮点累加或采样规则。
4. 旧 Talker 计时的 eager KV 长度增长、回放长度固定，不能直接宣称同工作量
   提升 70%。修订探针每次重新准备固定长度缓存，准备在计时外。历史记录已标注。
5. 旧八进程汇总仅取最后一轮，误称进程均值。已用所有热轮复算并修正文档；
   “区间”统一指进程均值的最小/最大值，不是置信区间。
6. 录制不是执行第二遍 GPU 图，不能断言其成本等于两遍 eager。收益须由实际
   首次使用和稳态请求测量确定，不能据此先排除单次会话。

真模型 CUDA 探针移到 `synthesis/cuda_tests.rs`，主合成模块不再混入长段调查代码。

## 数值与状态验证

显式真模型 CUDA 测试串行运行：多组不同 codes、5 种短块形状、LRU 淘汰、
表缓存淘汰后回放、旧返回张量保留、取消已使用图的会话后再合成。比较原始 F32
位模式，不以浮点容差代替逐位一致。

Talker 的探针限定为数值调查：16 头、head_dim 128、BF16 合成 Q/K/V，
31→32、35→64、127→128 三组长度各 4 份输入，合计 24576 个输出元素，
本机补齐前后全部逐位一致。它没有覆盖真实模型隐藏态、分组头映射、设备 RoPE
和动态 KV 写入，也没有证明完整 Talker 生成等价。本轮没有修改生产 Talker，
不预设需要 fork Candle；先评估现有算子扩展接口是否足够。

局部 headroom 探针的固定长度 Talker 与解码回放仍显示进一步减少分派的空间，
但其运行未锁频，仅作诊断，不作为端到端性能结论。

## 正式基准协议

Windows、RTX 5070、Qwen3-TTS 0.6B Base ICL，本地模型、prompt、文本及 seed
沿用 [CP 图记录](qwen-cp-graph-2026-10-18.md)。两侧 CP 图均开启；同一无 profiling
release 二进制只切换 `QWEN_DECODE_GRAPH`。8 个独立进程按 E/G/G/E/E/G/G/E
顺序串行运行，各 1 轮预热、5 轮热测；每进程复用同一模型，覆盖跨 utterance
的图复用。预热与正式调用都包含可能发生的录制，不从调用计时中扣除录制成本。

请求锁 SM 2700 MHz，运行采样约 2692 MHz，内存约 14001 MHz；保留系统原来的
平衡电源计划，结束解除锁频。桌面 GPU 进程仍驻留，不能称完全独占，也不与旧记录
高性能电源计划下的绝对秒数直接拼接。每轮导出在计时外，采样记录和进程快照留档。

原始轮次、GPU 采样、分析脚本与输出摘要在忽略目录
`target/tmp/voxcpm-opt/qwen-cuda/decoder-final/`。按进程全部热轮求均值，再报告
四进程均值的均值及范围。

## 实测结果

| 指标 | eager（CP 图开启） | 解码图 + CP 图 | 耗时变化 |
| --- | ---: | ---: | ---: |
| 流式合成 | 2.423958 s [2.403281, 2.443626] | 2.113950 s [2.090455, 2.139444] | **减少 12.8%** |
| 流式 RTF | 0.514 | 0.448 | 减少 12.8% |
| 完整合成 | 1.905687 s [1.889154, 1.919381] | 1.932733 s [1.902749, 1.977327] | 增加 1.4%，范围重叠 |
| 首 PCM | 0.572088 s [0.567045, 0.580395] | 0.570311 s [0.558775, 0.588278] | 减少 0.3%，范围重叠 |

流式达到本轮 10% 收益门槛，两组范围不重叠；完整和首 PCM 不报告明确收益或劣化。
完整路径没有进入解码图，波动不能直接归因于图开关。范围不重叠也不替代统计显著性检验。

首次使用轮（不纳入热测）含录制的流式进程均值约为 eager 2.4678 s、图 2.3989 s；
仅 4 次独立冷样本，不作为稳定提速结论。这说明必须实际计入录制，而不能把成本
简单视为两遍 GPU 解码后推断本次短句必然无收益。

八进程 × 六轮 × 五类产物共 **240 份**文件，codes、完整/流式 F32LE、完整/流式
WAV 各类 SHA-256 唯一，并与旧 CP 帧图基线对应类别相同。全部 59 帧、113280
samples、EOS、有限输出，流式 6 块完整覆盖。检查包含预热轮，没有靠丢弃录制轮
来跳过结果一致性门禁。

基准二进制 SHA-256：
`b1df019f44923b03bcc00f42620fd1f638bb5151e268a72489121de0c5d07c8b`。
该组是 CP 图已开启后的解码图增量收益，不能与其他电源计划/轮次的 CP 提速
直接叠加成“相对官方”或整体累计加速百分比。

## 检查与复现

- CPU 库测试：221 passed、3 ignored；普通测试不加载模型。
- CUDA 显式测试：新调查/回归 5 项及修正后的 CP eager 对照均通过，串行执行。
- 全 feature、all-targets Clippy `-D warnings`、私有项 rustdoc `-D warnings`、
  workspace 格式检查和 Qwen CUDA 后端集成编译通过。
- Candle 性能 skill 校验与本轮文档的本地链接检查通过。
- Windows release 链接仍有既有 `LNK4098` CRT 库冲突警告；本轮未改原生依赖或
  CRT 配置。Metal/AMX 没有在本机重新实测，不把 CUDA 数字推广过去。

```powershell
$env:QWEN_CP_TEST_MODEL = 'D:\workspace\talechime\target\drift-experiment\round3\qwen-base06'
$env:QWEN_CP_TEST_PROMPT = '<已有本地 ICL prompt.json 的完整路径>'
cargo test --locked --release -p talechime-qwen3-tts --features cuda --lib -j 1 `
    synthesis::cuda_tests -- --ignored --nocapture --test-threads=1
cargo test --locked --release -p talechime-qwen3-tts --features cuda --lib -j 1 `
    cuda_code_predictor_replay_matches_eager -- --ignored --nocapture --test-threads=1
```
