# Rust 性能工具实现验证（2026-10-20）

本文记录框架验证，不是正式性能基线。所有生成数据位于忽略的 `target/perf/`，没有把模型、音频或语料加入版本库。产品发行入口和原生依赖 revision/CRT 设置不变。

## 本机通过的检查

- xtask 单元、入口和 Windows 进程树测试；覆盖事件截断/预算、孙进程回收、不影响无关进程、超时/取消后继续、锁、恢复摘要、比较器和导出白名单。
- Core 77 个库测试通过，包括新增的 permit 释放和部分预约取消测试。
- performance-tools 免模型库/探针测试通过；跨 owner 线程关联、trace 溢出、流错误/截断/NaN/超时及延迟 Backend 导出。
- xtask/Core all-targets Clippy `-D warnings`；全部 CPU 后端开发 feature 的 lib/tests/examples Clippy；rustdoc `-D warnings`。
- Criterion `text_performance -- --test` smoke 通过，无共享机器墙钟门禁。
- CUDA 13.0 下 `cargo check --locked -p talechime-backends --no-default-features --features performance-tools,qwen-cuda --example performance_probe` 通过。仅编译，未声称 CUDA 实测。

## 免 Python 端到端证据

`fixture-v3-run` 使用确定性 Backend，3 进程 ×（1 warmup + 5 hot），PATH 置空运行。执行、resume、compare、report、pack 通过；resume 复用已核验提交。`overhead-v3` 独立执行 off/baseline/trace，每组覆盖完整，三组 PCM 摘要相同；真实模型开销验收仍 unverified。

`fixture-v4-report.zip` 解包到新目录后，逐资产 SHA256 和 HTML 相对 src/href 全部通过，无 script/fetch/远程 URL。浏览器工具拒绝 `file:` URL，因而**视觉打开验收尚未完成**；不以文件检查冒充浏览器验收。任务 7.2 保持未勾选，后续需人工离线打开 index.html、检查图表和播放代表音频。

`profile-v3-run` 是独立 profile 模式，使用明确标注 synthetic 的不透明 `.etl` 夹具验证导入、来源收据、报告及打包。它不是 WPR 实际采集结果。本机 nsys 不可用，WPR 帮助可用；缺工具不阻塞 baseline。单元测试确认 baseline 不接受原生 profile 附件。

## 显式真实模型短 smoke

使用本机现有、生产 manifest 全部校验通过的 `moss/nano-candle` 资源（revision `44502f80dbf9743528fa921cc544d662c685ebec`），CPU、内置 Weiguo、seed 42、文本“你好。”、continuation=false、Rayon 1 线程。无下载、无权重转换。通过 Rust runner 的生产 Registry/Backend/LocalSet 路径，PATH 置空，1 进程、0 warmup、1 请求，正常 Completed、事件完整、资产核验和树回收通过。

当次观察：合成 0.6109038 秒，输出 0.48 秒音频，consumer 首 PCM 0.4988722 秒。原始 PCM 摘要 `da1cab5e404b09c00021691ffae13f8731eae3acf6923cc041074816c16f1ae5`。原始证据为 `target/perf/nano-smoke-run/results.json` 和对应 committed attempt。**这是接线 smoke，不是稳定性能结论或质量验收**；不参与全量矩阵排名。Job 为一次性手工冻结的 JSON，运行编排、采样和报告全部是 Rust。

## 生产完成边界审查

探针通过生产 SynthesisStream drain 到显式 Completed，不通过“函数返回了”推定已播放。各模型在 owner 内生成 host PCM 后发送，接收端采集实际 PCM；不新增每 span 同步，不创建无关 CUDA context。

| 路径 | 源码边界 | 当前验证范围 |
| --- | --- | --- |
| Qwen | `synthesis`/audio 输出转 host Vec，owner 流结束发送 End | CPU 编译/探针夹具；CUDA 编译；真实设备完成与 graph 状态待模型实测 |
| VoxCPM | `model.rs` 输出 `to_vec1` 后 owner 交付 | CPU 编译；真实 CPU/CUDA 待实测 |
| OmniVoice | 生产 synthesize 返回 host PCM，owner 交付并 End | CPU 编译；真实设备路径待实测 |
| CosyVoice3 | pipeline 输出 host samples，async 发送保留原 await/取消 | CPU 编译；ORT/Candle CUDA 完成需原生证据 |
| MOSS Nano Candle | codec.decode 返回 host samples，原有取消感知发送 | 本文 CPU 短 smoke；CUDA 待实测 |
| MOSS ONNX / Local / Realtime | 原生产 ORT 输出/codec host PCM 路径 | CPU 编译；适用设备真实矩阵待旧 Windows change |
| IndexTTS | 生产 synthesize host PCM 后 End | CPU 编译；缺生产不可变资源/转换来源收据时测量 blocked |

host PCM 可用时间和通道排空证明的是请求输出契约，不证明 GPU 所有无关工作完成，也不等于 kernel duration。GPU kernel、graph replay 和设备空闲状态须由对应 owner/原生 profiler 单独确认。公开报告保留不适用、缺资源、失败和 unverified，不能把表格中的源码审查标成设备实测。

## 交付边界

新增 CI 是免模型契约和微基准 smoke；native CUDA/Metal 与固定硬件真实基线分开。Mac native lifecycle、CPU Accelerate/线程、RSS/高水位、Metal 完成与统一内存、Instruments、观测开销和真实模型测试继续 deferred；字段映射及接续清单见 `docs/performance-harness.md`。

旧 `benchmark-windows-backends` 已按用户授权改为复用 Rust/xtask；其资源安全、全量 Windows 批次和人工试听仍由原 change 交付。未删除历史 Python 对照脚本，也未归档两个 change。

## 提交前双轴审查与清理

基点为 `f1528b98dd7e9a963637d28c0d49b35c490ef6b8`，审查其后的本次工作区改动及新增文件。

### Standards

发现 1 项实际问题：线程配置可为空或使用未生效键，与记录有效线程数的规范不符。现要求唯一、显式、正整数 Rayon 配置，并在执行前覆盖子进程继承值；空配置、零值和未知键有回归测试。

另有 1 项简化建议：删除采样间隔的重复校验，展开采样循环，并统一报告资产扩展名、校验及复制逻辑。复核未发现新增阻塞项。

### Spec

发现 2 项：上述线程兼容性缺口，以及 CLI 未接入 executor 取消状态。现已接入 Windows Ctrl-C/Ctrl-Break，并在提交边界和末轮后检查取消，保留部分结果、停止后续执行且非零退出。真实新进程组 CLI Ctrl-Break 测试已显式运行通过：仅一个 cancelled attempt、cleanup_verified=true、无成功 committed 标记。CI 在免模型探针构建后执行同一测试；单独运行时需设置 TALECHIME_PERF_PROBE_BUILD 为构建收据的绝对路径。最终复核原有 2 项均已修复。

按用户要求清理无引用的 `benchmark_resources.py` 草稿和已由 Rust 取代的独立 `probe_metrics.py`。保留官方对照、转换、播放验收及 Mac 历史复现脚本；`metrics.py` 仍被播放验收使用。用途清单见 `tools/tts/README.md`。
