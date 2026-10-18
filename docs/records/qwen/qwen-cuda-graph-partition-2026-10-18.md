# Qwen CUDA 图分区评估：Talker 步与 12Hz 解码器

## 问题与范围

CodePredictor 帧图合入后（见
[Qwen CP 帧图](qwen-cp-graph-2026-10-18.md)），完整合成 1.90 s、流式 2.45 s。
本轮评估剩下的两个循环是否也值得成图：Talker 自回归步与 12Hz 解码器
（AudioVAE decode）。范围只有"评估"，实现按结论另行决定。

## 方法

新增被忽略的真模型探针 `synthesis::tests::cuda_graph_headroom_probe`
（`QWEN_CP_TEST_MODEL`，CUDA，串行运行）。它对每个候选做同形状 capture，
要求回放与 eager **逐位一致**，并按 ABBA 顺序报每次调用的墙钟时间。
锁频 2700 MHz / 内存 14001 MHz，RTX 5070，模型与 [Qwen CUDA 调研](
qwen-cuda-performance-2026-10-18.md) 相同。

Talker 步的图只在固定 KV 长度上有效，因此每个阶段各用一份独立预热到同一
offset 的 cache：capture 用一套、计时 eager 用一套。capture 前的预热必须在
`enable_cuda_graph_htod_cache()` 之内，否则 capture 会因为 h2d 参数缓存未命中
而被拒；预热和 capture 必须读不同长度的 cache，否则记录到的 KV 长度与预热时
不同，同样被拒。

## 结果

| 候选 | eager | 回放 | 变化 | 逐位一致 |
| --- | ---: | ---: | ---: | --- |
| Talker 步（KV 长度 31） | 12.06 / 13.12 ms | 3.59 / 3.79 ms | **-70%** | 是 |
| 12Hz 解码 35 帧（流式块形状） | 130.16 / 119.50 ms | 49.91 / 50.57 ms | **-60%** | 是 |
| 12Hz 解码 185 帧（完整合成形状） | 313.96 / 332.67 ms | 241.56 / 237.91 ms | -27% | 是 |

每格两个数字是交错的两段计时。Talker 步的回放只对录制时的 KV 长度有效，
解码器的回放对该帧数有效。

## 结论

### 1. 解码器此前根本无法 capture

首轮探针报 `to_cpu_storage during CUDA graph capture`：解码器每次调用都把
语义 token 读回主机做 `% 2048` 再上传。改成设备端查表
（`Decoder12Hz::semantic_code_lut`）后，这一往返消失。

紧接着又报 `cannot upload uncached host data`：`run_transformer` 每次调用都在
主机拼一个 `frames × frames` 的滑窗 mask 并上传，35 帧 4.9 KB、185 帧
136.9 KB，都超过 candle 捕获期 4096 字节的 h2d 缓存上限。同时每次调用还重建
RoPE 的 `inv_freq`。三者都只依赖帧数，改为按帧数缓存的常量
（`Decoder12Hz::tables`，含 cos/sin/mask），`inv_freq` 变成加载期字段。

这两处是纯搬迁：索引与掩码取值不变。配对 ABBA 八进程（CP 帧图两侧都开，
各 1 预热 + 2 热轮）证明没有回归也没有收益：

| 指标 | 改前 | 改后 |
| --- | ---: | ---: |
| 完整合成 | 1.8817 s [1.8445, 1.9182] | 1.8739 s [1.8462, 1.9188] |
| 流式合成 | 2.4100 s [2.3688, 2.4381] | 2.4204 s [2.3817, 2.4551] |

八个进程的 codes、完整/流式原始 PCM、两份 WAV（预热轮与测量轮共 10 类产物）
SHA-256 各自唯一，跨改前改后完全一致。保留这两处改动的理由是去掉每块一次的
设备→主机同步并让解码器可捕获，不是性能数字。

### 2. 流式解码成图有收益，但只在长驻进程里成立

流式解码每块输入 25 帧左上下文 + 10 帧新帧，首块之后形状恒为 35 帧，每次
回放省约 75 ms。完整合成只解码一次，-27% 抵不过一次 capture 的两遍解码，
所以完整路径不适用。

块内 capture 的代价约等于两遍 eager 解码，session 级"第二次出现才 capture"
在 6 块 utterance 上只是勉强打平；只有把图按帧数缓存在模型上、跨 utterance
复用，才有稳定收益。这是实现前必须先确认的工作负载假设，尚未实现。

### 3. Talker 步收益最大，但需要 candle 侧改动

-70% 的余量是三个候选里最大的，但自回归步的 KV 长度每帧加一，回放只对录制时
的长度有效。要复用需要三件事同时成立：

1. 按桶补齐注意力长度并加掩码（改变 matmul 的 K 维，数值会变）；
2. RoPE offset 变成设备端输入（否则一个桶内不同帧的位置编码被录死）；
3. KV 写入位置变成设备端偏移——当前 `KVCacheAppend` 用主机 `dst_pos` 调
   `copy2d`，图里只能写固定位置，这需要在 candle 里换成按设备索引写入的算子。

第 3 项属于 candle fork 的行为级改动并需要向上游提 PR，第 1 项会改变数值，
需要先定容差。因此本轮只记录，不实现。

## 复现

```powershell
$env:QWEN_CP_TEST_MODEL = 'target\drift-experiment\round3\qwen-base06'
cargo test --locked --release -p talechime-qwen3-tts --features cuda `
    --lib cuda_graph_headroom_probe -- --ignored --nocapture --test-threads=1
```

解码器改前/改后的两个探针二进制、ABBA 脚本与全部产物在
`target/tmp/voxcpm-opt/qwen-cuda/`（`probe-old.exe`、`probe-new.exe`、
`run-decoder-abba.ps1`、`decoder-abba/`），该目录被忽略。
