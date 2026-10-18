# VoxCPM2 Candle macOS 性能实测（2026-10-08）

> 同步自 TRNovel [c5d7daa](https://github.com/yexiyue/TRNovel/commit/c5d7daa60b435293d77f1ab886e22fdd56a214c0)。以下为上游 macOS 实机记录；产物路径属于原检出，crate 路径和命令已映射到 Talechime，未在此次同步中重做实机试听。

## 结论

M4 Pro 上 Q8 Candle Metal 完整语料平均 RTF **0.954**，旧原生 Metal 为 **1.737**，按实际输出时长归一后的吞吐约提升 **1.82 倍**。热首 PCM 从 518ms 降至 213ms。
Candle Metal 接近实时线，平均生成速度约为播放速度的 1.05 倍，余量较小；本次没有证明实际长时间播放零断流。
相同短句上，CPU 平均 RTF 3.689，Metal 0.922，Metal 吞吐约为 CPU 的 4.00 倍。CPU 无法持续实时生成。

## 环境和条件

- 分支 `tts-gpu-model-integration`，快进到 `cd8d71051840d063cdfb6f57bbaae4f387c6a3fa`；拉取前工作区干净。
- MacBook Pro，Apple M4 Pro，12 CPU 核（8P+4E），24GB 统一内存；macOS 26.6.2 (25G83)。
- Cargo release / locked，Candle 0.11.0。Candle 探针 feature 为 `voxcpm-metal`，CPU 测试使用同一二进制的显式 CPU 设备；未额外开启 Accelerate feature。
- `2b-q8_0`，revision `169f64d8b98bbaab1761e4ca3a83e6af653456cc`；沿用 `~/.novel-tts/` 两份 GGUF，大小和 SHA-256 与资源清单一致，没有重新下载。
- 相同 seed 42、CFM 10 步、CFG 2.0、温度 1.0、最大 200 帧。GPU Acoustic 为 F16，CPU 为 F32。
- 完整语料 `tools/tts/acceptance-corpus.txt`，生产 180 字节分段，共 11 段；短句为“山风吹过松林，星光照亮归途。”。
- 同一 `target/tts-integration/onnx-reference.wav`，参考文字“你好，欢迎使用听书功能。”，均测试带文字的参考续接。
- 各模型各预热一轮；表格排除预热。Candle 完整语料五轮，原生完整语料三轮，CPU/Metal 短句各三轮。设备测试串行，没有同时运行两个推理基准。
- 原生日志确认 `custom component backend=MTL0` 且 BaseLM GPU offload，未回退 CPU。

## 结果

RTF = 生成耗时 / 实际输出音频时长，越低越好，RTF < 1 才快于实时。首 PCM 是每轮第一段的首音频延迟，不是全体段落的延迟分布。

| 测试 | 热轮数 | 每轮音频秒 | 平均首 PCM ms | 平均 RTF | RTF 范围 | 峰值 RSS GiB | peak memory footprint GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candle Metal 完整语料 | 5 | 64.64 | 212.8 | 0.953823 | 0.938–0.982 | 2.92 | 5.51 |
| 旧原生 Metal 完整语料 | 3 | 70.08 | 518.0 | 1.736678 | 1.736–1.738 | 5.02 | 3.47 |
| Candle Metal 短句 | 3 | 4.00 | 212.0 | 0.921933 | 0.920–0.923 | 3.06 | 5.32 |
| Candle CPU 短句 | 3 | 4.00 | 979.3 | 3.688558 | 3.499–3.856 | 5.67 | 6.05 |

内存来自 macOS `/usr/bin/time -l`，覆盖加载、预热、热测和探针回归；RSS 与 footprint 是不同记账口径，不能相加，也不能当作独占 GPU 内存。Candle 的 RSS 较低，但其 footprint 较高，不能据此断言总内存占用降低。

Candle 完整语料加载 2578ms（设备初始化 16ms）、参考重采样/编码 466ms；首轮预热首 PCM 4890ms、RTF 1.009。加载数字受到文件缓存影响，不表示冷磁盘启动。后续短句加载 1560ms、预热首 PCM 221ms，不能与第一次首次运行开销直接等同。

Candle 每轮完整音频 64.64s，原生为 70.08s；随机数/数值路径不同，不强行对齐 WAV。性能变化不是音质或数值一致性的证明。

## 运行验证和边界

- Candle Metal 完整语料、Metal 短句和 CPU 短句探针均正常退出，所有测量段达到 EOS；取消后再请求达到 EOS。
- 空文字、非法采样、帧上限截断、prefill/CFM 内取消和初始化取消检查通过。
- PCM 回调内取消到返回均记为 0ms；这仅表示小于毫秒计时精度，不代表任意 GPU kernel 执行期间的外部取消延迟。Metal 完整探针释放/同步 93ms。
- 原生完成预热和三轮热测后，监测脚本主动 SIGTERM 停止多余重复，exit 143 是预期的基准截停，不是生成失败，也不作为正常退出或取消验收。
- 未测试 BF16（目前产品限定为 CUDA 实验）、无参考/无文字克隆、音色设计、worker/TUI 实际播放或 30 分钟持续播放；未重做官方 F32 数值 oracle，原有失败和待验收项保持不变。
- 没有生产代码修改；原生开发探针临时添加 Metal match arm，构建后已恢复原文件。未运行全工作区质量门槛，此次验证是实际 release 推理与探针自带回归。

## 原始材料和复现

全部产物位于 `target/tts-integration/voxcpm-mac-20261008/`：

- `metadata.json`：版本、条件、输入和二进制 SHA-256。
- `comparison.json`：汇总指标；`wav-validation.json`：逐文件非空/有限数值/48kHz 单声道 F32 校验及 SHA-256。
- `candle-metal/summary.json`，`candle-cpu-short/summary.json`，`candle-metal-short/summary.json`。
- `native-metal/summary-three-hot-rounds.json`：原生完成轮次及主动停止原因。
- 各目录 `round-*/` 保存逐段 WAV 和原文 `index.json`；各设备 `*.stdout.log` / `*.stderr.log` 保存计时及日志。

```bash
cargo build --release --locked -p talechime-backends --no-default-features --features voxcpm-metal --example voxcpm_candle_probe
model_dir="$HOME/.novel-tts/voxcpm/models/2b-q8_0/169f64d8b98bbaab1761e4ca3a83e6af653456cc"
bench_dir="target/tts-integration/voxcpm-mac-20261008"
VOXCPM_BENCH_ROUNDS=5 /usr/bin/time -l target/release/examples/voxcpm_candle_probe "$model_dir" metal "$bench_dir/candle-metal-retest" tools/tts/acceptance-corpus.txt target/tts-integration/onnx-reference.wav "$bench_dir/transcript.txt"
```

CPU/Metal 短句复现：将 `VOXCPM_BENCH_ROUNDS` 改为 3，语料改为 `$bench_dir/short.txt`，CPU 设备参数改为 `cpu`，输出使用新目录。
原生复现：在 `crates/voxcpm-sys/examples/voxcpm_native_benchmark.rs` 的设备 match 中临时加入 `"metal" => Device::Metal,`，运行 `cargo build --release --locked -p voxcpm-sys --features metal --example voxcpm_native_benchmark`；参数顺序与 Candle 完整语料相同，但没有轮数环境变量，默认预热一次、热测五次。测试后恢复源文件。
