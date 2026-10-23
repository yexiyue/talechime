# Rust 性能工具

实现验证、免模型端到端证据、已准备 Nano CPU 短 smoke 与未完成的浏览器视觉验收，见 [验证记录](records/2026-10-20-rust-performance-harness.md)。

## 当前入口

`cargo xtask perf --help` 提供计划、构建、只读准备、执行、恢复、比较、报告、打包、观测开销对照和外部 profiler 证据接口。xtask 本身不链接模型、ORT 或音频设备，不发布、不进入发行安装包，默认 Cargo member 仍为 talechime。

```powershell
cargo xtask perf doctor
cargo xtask perf build --output target/perf/catalog-build.json
cargo xtask perf prepare --build target/perf/catalog-build.json --app-root target/perf/custom-app-root --output target/perf/catalog.json
cargo xtask perf plan --platform windows --machine windows-9700x --catalog target/perf/catalog.json --output target/perf/windows-catalog-plan.json
cargo xtask perf plan --platform windows --machine windows-9700x --output target/perf/windows-plan.json
cargo xtask perf plan --platform macos --machine mac-m4-pro --output target/perf/mac-plan.json
```

计划创建不会覆盖已存在文件、下载模型、转换权重或启动推理。计划中的 unverified/deferred 不是运行成功。catalog 没有编入的模型/设备保留 missing_feature，不从矩阵消失；Mac 仍需本机模型 readiness 验证，Windows catalog 不证明 Metal 可执行。

build 在 `target/perf-builds/<配置摘要>/` 用显式 feature 构建 release 探针，自动包含开发专用 `performance-tools`；默认不包含模型 feature。定位使用 Cargo artifact 消息，收据保留有效 features、源身份和二进制 SHA。构建期间源码改变会拒绝发布收据；增加 feature 用不同收据文件，不覆盖。

prepare 消费构建收据并校验可执行文件摘要。无模型默认只返回生产 AppPaths/Registry 目录和 catalog，不创建自定义资源根。指定 `--backend qwen --model 0.6b-base` 可在编入对应 feature 后对已有模型进行只读生产清单校验，生成资源收据；失败保留文件，不下载、不重命名 `.corrupt`。IndexTTS 缺少生产摘要清单/转换来源收据时 blocked；缺 importer 不调用 Python 兜底。资源已验证不等于设备执行、精度和性能已验证。

Windows 资源准备使用三个显式命令，只复制或下载 pinned 文件，不做目录迁移、扫描或转换：

```powershell
cargo xtask perf resource-inventory --plan resource-plan.json --output resource-inventory.json
cargo xtask perf resource-import --inventory resource-inventory.json --resources-root "$HOME\.talechime\resources" --output resource-import.json
cargo xtask perf resource-download --plan resource-plan.json --resources-root "$HOME\.talechime\resources" --output resource-download.json
cargo xtask perf resource-download --plan resource-plan.json --resources-root "$HOME\.talechime\resources" --offline --output resource-offline.json
```

`resource-inventory` 记录大小、SHA-256、复用字节和来源身份；`resource-import` 按身份加锁、暂存复核并原子无覆盖发布匹配目标，冲突保留原件。`resource-download` 仅接受 `production_manifest`，先暂存校验摘要，离线模式只复用已验证文件。转换收据和未验证文件不能替代生产清单；准备完成标记是下载/导入收据的前提，测量 Job 只消费已验证资源收据，不下载或转换。

计划中的公共朗读矩阵保持 28 个 CPU/CUDA 格（25 个适用、3 个不适用）；`moss/VoiceGenerator` 是单独 `auxiliary=true` 的 CUDA 辅助行。Windows 所需 feature 记录在构建收据中：Candle 使用 `qwen-cuda`、`voxcpm-cuda`、`omnivoice-cuda`、`moss-candle-cuda`、`moss-nano-candle-cuda`、`cosyvoice3-cuda`、`indextts-cuda-cudnn`，Nano ONNX 使用 `ort-cuda`；CPU 探针使用对应基础模型 feature。

基准参考音色使用生产 `voices import --text` 保存 `reference.wav`、准确转写和模型身份；探测记录只引用 `custom:<id>` 与文件摘要，不复制用户音频入仓库。缓存身份来自生产 `Backend::cache_identity()`，按实现、revision、设备、平台和目录隔离，不能仅凭同名音色跨模型复用；可证明身份时，每个进程仅在正式轮次前用一个有超时的短请求预热懒参考编码；其完整合成成本单列为 `reference_warmup_seconds`，不冒充纯参考编码成本，也不混入每轮计时。否则首次懒准备保留在首个正式轮并如实呈现。首 PCM 按能力分为 `native_streaming_first_pcm` 和 `segment_delivery_first_pcm`；后者是整段交付延迟，不能当作原生流式首块。

正式测量链路的目标为纯 Rust：资源检查、构建、执行、恢复、采样、统计和报告。已有上游 Python-only 转换仅允许在显式准备阶段运行，必须记录解释器/脚本 revision、输入输出摘要，并通过生产校验；测量不隐式执行转换。历史官方 Python 对照脚本仅保留在 Git 历史中，当前树已删除 `tools/`。

## 与 Windows 基线的任务映射

`rust-performance-harness` 拥有唯一的 Rust 开发框架。`benchmark-windows-backends` 消费该框架，保留自己的模型矩阵、资源安全契约、真实测量与试听报告交付责任：

| Windows 任务组 | 归属和衔接 |
| --- | --- |
| 1：资源导入 | 保留复制、锁、摘要、无覆盖发布要求，Rust 实现；不依赖未完成的 Python importer |
| 2：探针 | 复用本框架事件/计时/音频契约，扩展全部生产变体 |
| 3：编排采样恢复 | 复用 xtask，禁止再建第二套 Python runner |
| 4：报告 | 复用 Rust JSON/CSV/HTML 输出；ECharts 以仓库内固定 SHA 资产随报告离线分发，不调用 Python 绘图或 CDN |
| 5/6/7：资源与全量批次 | Windows change 独立执行，框架 smoke 不代替全量基线 |

旧 Windows design/tasks 已同步改为复用 Rust 工具，保留全量覆盖、样本协议和资源安全规则。旧 `tools/` 脚本现已全部删除；历史官方对照、转换与复现脚本可从 Git revision `e591c94` 取回。固定语料保留在 `crates/voxcpm/tests/fixtures/acceptance-corpus.txt`，当前执行、采样和报告使用 Rust 工具。

## Mac 接续边界

Mac 计划保留 CPU 和 Metal，适用模型行仍为 unverified，缺编译 feature 由 catalog 标记 missing_feature，不适用组合保留原因。原生工具已接入受控进程组及 SIGINT/SIGTERM 取消回收、RSS/physical footprint 和 OS footprint 高水位；模型 readiness 与工具能力独立。真实 CPU/Metal 基线和真实观测开销另行验收，Instruments 仍 deferred。

统一内存不是进程独占显存。native 编译通过不等于真实模型性能验收，Windows 数据不激活 Mac 能力。

## 指标字典

每个 reading 包含 value、unit、scope、source、interval_ms、status、reason。available 必须有有限值且无缺失原因；unavailable/unsupported/deferred 必须 value=null 并给原因。

| 指标 | 单位 | 边界 |
| --- | --- | --- |
| request / load / verify / reference_prepare | seconds | host 单调墙钟，父子重叠不求和 |
| owner / adapter / consumer first_pcm | seconds | 相对同一请求起点，各位置独立 |
| channel_wait / chunk_gap | seconds | 明确生产端或消费端 |
| queue_audio | seconds 或 null | 纯排队时长目前 unavailable；不能用总 permit 占用冒充 |
| prefetch_time_budget_peak | seconds | pending、active 和 consumer 持有的时间 permit 总占用；按毫秒取整，非纯队列 |
| RSS / private_bytes | bytes | process；按 interval_ms 采样 |
| peak_working_set | bytes | process；OS high-water，不冒充采样最大值 |
| device_vram | bytes | whole_device，不称进程独占 |
| system_available_private_bytes | bytes | system commit limit 减 commit total，不称为空闲物理内存 |
| competing_gpu_processes | processes | whole_device_compute_apps 数值 reading；同帧顶层 `competing_gpu_processes` 另保留 status、reason、PID、进程名及不支持内存原因 |
| Metal allocation | bytes 或 null | 后续 native collector 声明来源与范围 |

schema 为 `talechime/performance/1`。未知版本必须拒绝恢复与比较，不能静默解释。原始逐轮结果是权威证据；metrics 聚合与 trace 仅用于补充诊断。

## Windows 免模型工作流

以下命令仅生成确定性合成音频，不加载或下载模型。输出不可覆盖，重复验证请换目录。

```powershell
cargo xtask perf build --probe performance_probe --output target/perf/probe-build.json
cargo xtask perf fixture-job --build target/perf/probe-build.json --output target/perf/fixture-input
cargo xtask perf run --job target/perf/fixture-input/job.json --output target/perf/fixture-run
cargo xtask perf resume --job target/perf/fixture-input/job.json --output target/perf/fixture-run
cargo xtask perf report --run target/perf/fixture-run --output target/perf/fixture-report
cargo xtask perf pack --report target/perf/fixture-report --output target/perf/fixture-report.zip
cargo xtask perf overhead --job target/perf/fixture-input/job.json --output target/perf/overhead
```

构建完成后可直接运行 `target/debug/xtask.exe perf ...`；执行、恢复、报告和打包不需要 Cargo/Python PATH。离线报告只使用相对资产，不发起 CDN、fetch 或原资源目录访问；本地 ECharts 脚本和许可证从 `docs/legal/echarts/` 复制并按 SHA-256 复核。ZIP 仅包含 manifest 中经摘要复核的资产，不递归复制资源根。试听始终 pending，确定性音频相同不代表真实模型质量验收。

正式 Job 参见 `xtask/src/perf/runner.rs::Job` 和生成的 fixture job：固定 plan/build 的绝对路径、case_id、完整 Conditions、生产 request、资源收据、参考文件摘要、进程超时和采集 Budget。用 `doctor` 获取机器指纹；CUDA 必须记录实际 UUID/驱动。生产请求必须使用 catalog 中实际 backend/model/device、有效默认参数与覆盖值、准备收据，不能把 fixture 标记改成真实模型。threads 必须是唯一显式正整数 `{"rayon":N}`，runner 总是覆盖子进程继承的 RAYON_NUM_THREADS；其他 BLAS 线程环境在 environment 中冻结。math/precision 必须依实际构建与执行路径填写，不以 CPU/CUDA 名称推定。`native_stream` 和 `segment_delivery` 分开比较。测量准备不运行转换器。

每次独立进程默认 1 次 warmup + 5 次 hot，共 3 个进程。统计保留原始轮次、进程均值和 mean-of-means，RTF 使用实际输出时长，warmup 不进入 hot 均值。多段请求使用总合成秒数/总音频秒数，不平均段 RTF。阶段可能重叠，不能相加；首轮保留独立证据。优化实验另行安排 ABBA 批次，不把固定顺序观测开销校准解释为随机化因果实验。

比较需要显式 policy：

这里的阈值按具体指标与测量噪声配置，用于本次比较判定，不是优化是否保留的统一门槛。
项目不设置默认的整链改善百分比；局部显著收益的保留与组合复测遵循
[优化保留判断](knowledge/tts-performance.md#优化保留判断)。

```json
{"schema":"talechime/performance/1","metric":"rtf","lower_is_better":true,"absolute_threshold":0.02,"relative_threshold":0.05,"expected_processes":3,"require_equal_output":true}
```

```powershell
cargo xtask perf compare --baseline target/perf/fixture-run/results.json --candidate target/perf/candidate-run/results.json --policy policy.json --output comparison.json
cargo bench --locked -p talechime-core --bench text_performance
```

判定保留 pass/regression/inconclusive/incomplete/incomparable，使用绝对及相对阈值与进程均值的观测范围（不是置信区间），缺轮、失败或输出等价证据不足不会声称保持输出加速。允许源码变化，平台、硬件、驱动、math、参数、资源、语料、参考、线程、观测配置改变则不可比。共享 CI 只运行 Criterion smoke，不对墙钟时间设硬门禁。

## 生命周期、预算与恢复

Windows 在 suspended 创建后先加入独立 KillOnJobClose Job Object，再恢复主线程；退出、超时、取消和解析失败都回收所属树。只终止本工具 Job；无法证明回收则停止后续推理。CLI 的 Ctrl-C/Ctrl-Break 通过共享取消标志通知 executor，回收后保留 cancelled attempt 和部分 results，停止后续调度并以非零状态退出；正常控制台信号测试验证仅目标进程组受影响。父进程被强制终止也关闭 Job，此时可能只保留未提交目录，resume 不复用它。

stdout/stderr 同时排空。默认单事件 1 MiB、单日志 64 MiB、最多 100 万事件，资源采样日志使用相同字节预算，采样默认 500 ms（10–60000 ms），GPU 采样需显式开启。进程超时最多 4 小时，请求超时最多 30 分钟。探针每请求 PCM 最多 512 MiB，每进程资产上限 8 GiB，执行前校验保守预算；chunk/segment/trace 均有独立容量上限。事件缺序、缺终态、截断、非有限音频、预算超限不是成功。导出 WAV/摘要在计时外；F32 是原始样本，PCM16 是有明确转换说明和派生关系的试听资产。

每 run 有 OS 文件锁，每 attempt 唯一目录。成功仅在音频、日志及事件完整性核验后以原子 no-clobber committed 标记发布。resume 重新核验源码/二进制/输入条件/重复协议/全部资产及日志摘要，失败目录保留，输入变化创建新 run。不要手工补 committed 文件。执行器串行的是同一 run；不同 run 请勿同时使用同一 GPU，工具尚不提供跨 run GPU 调度。

## tracing、metrics 和原生诊断

开发 feature `performance-tools` 接入 observation；库不安装全局 subscriber/Recorder。探针在入口安装并在工作结束后 flush。owner 请求携带 span 上下文，async 使用 Instrument，不持 entered guard 跨 await；不改变推理数学、流缓冲或 worker stdout 协议。off 关闭采集，baseline 仅必要阶段，trace 额外详细模型 span，profile 单独承载原生诊断。各模式构成独立比较身份。

metrics 使用最多 256 series、白名单标签和固定 32 桶 histogram；不从 histogram 重建逐轮时间。详细 trace 最多 8192 事件/16 MiB，溢出记录 dropped 和 trace_complete=false；异常进程不能提交为成功。原始轮次及 owner/adapter 事件是首 PCM、等待等指标的依据。计时为 host 单调时间，owner 返回 host PCM 及生产流显式 Completed 构成请求边界，不为每个 span 增加设备同步。没有原生 profiler 证据时不得把 host span 叫作 CUDA kernel 时间；各真实后端的设备完成与 graph 状态仍需真实模型单独验证。

Windows 使用原生进程 RSS、private bytes、OS peak working set；低频 nvidia-smi 是整卡 memory.used/clock/power/temperature。资源文件保留单位、source、scope、interval 和 null 原因，报告不将整卡显存与进程 RSS 排成统一内存优劣榜。

```powershell
cargo xtask perf profile-tools
cargo xtask perf fixture-job --build target/perf/probe-build.json --observation profile --output target/perf/profile-input
# 执行 profile run，并在外部工具完成采集后导入
cargo xtask perf attach-profile --run target/perf/profile-run --tool nsight_systems --version "实际版本" --artifact capture.nsys-rep -- nsys profile <实际命令>
```

WPR 使用 `--tool wpr` 与 `.etl`。导入要求 profile run、工具版本和实际调用记录；版本/调用由调用者声明，工具不擅自启动全局 WPR 会话。报告保存原生文件和来源收据。工具缺失不阻塞 baseline；Mac Instruments deferred。overhead 用同条件 off/baseline/trace 独立进程，先核验音频摘要，再输出分布和描述性开销；真实模型开销验收始终明确 unverified，夹具不支持固定开销承诺。

## Mac 验收映射

| 后续实机工作 | 证据与能力字段 |
| --- | --- |
| 原生进程组、取消、孙进程回收 | 原生工具已接通；逐 attempt 必须 cleanup_verified，失败禁止继续 |
| RSS 与高水位 | reading source/scope/interval/status/reason，分清 sampled max 和 OS peak |
| CPU Accelerate/math/线程 | Conditions math_backend/features/threads/environment，独立 CPU 基线 |
| Metal 完成边界和内存 | timing_contract、reading scope，统一内存不称独占显存 |
| Instruments | profile-mode 原生附件、工具版本/调用；目前 deferred |
| 开销与真实模型 | 独立 off/baseline/trace 分布、资源摘要、CPU/Metal 分组和人工试听 |

编译检查、免模型夹具、已准备资源短 smoke、全量真实基线和试听验收分别记录。Mac 只能依据本平台证据激活能力，Windows 框架验证不能代替 Mac 验收，也不能完成旧 Windows change 的全量实测任务。


## Mac 原生工具接线

```sh
cargo xtask perf doctor
cargo xtask perf plan --platform macos --machine mac-m4-pro --output target/perf/mac-plan.json
cargo xtask perf build --probe performance_probe --output target/perf/mac-fixture-build.json
cargo xtask perf fixture-job --build target/perf/mac-fixture-build.json --output target/perf/mac-fixture-input
PATH='' target/debug/xtask perf run --job target/perf/mac-fixture-input/job.json --output target/perf/mac-fixture-run
PATH='' target/debug/xtask perf resume --job target/perf/mac-fixture-input/job.json --output target/perf/mac-fixture-run
PATH='' target/debug/xtask perf report --run target/perf/mac-fixture-run --output target/perf/mac-fixture-report
```

相同输出目录不可覆盖；重跑用新目录。Mac fixture 自动选择本机平台。doctor 不加载模型、无需 Python 或音频设备；性能 CI 在 macOS 上运行无模型契约、进程回收和显式 CLI 取消测试，不下载权重。Windows 仍由 Windows 原生 CI 验证。

Mac 受控进程组面向不主动脱离会话的可信探针；已观察换组、PID 身份变化或回收不确定时保存失败并停止后续推理。正常退出后仍清理遗留后代。SIGINT/SIGTERM 触发有界回收；SIGKILL/断电不保证实时清理。轮询前逃逸且重新托管的恶意后代不在保证内。

Mac `proc_pid_rusage(RUSAGE_INFO_V4)` 提供 rss_bytes、physical_footprint_bytes 和 os_peak_physical_footprint_bytes（bytes/process）；最后一个是 OS lifetime footprint 高水位，不是 RSS 峰值。报告对普通 RSS 只显示采样最大值。os_peak_rss_bytes 与 metal_allocation_bytes 当前为 null+reason；统一内存/RSS 不称独占 VRAM。查询失败/短命进程仍保留 unavailable，不填零。

真实 Mac job 的 math_backend 必须匹配所构建 `performance_probe --execution-info` 返回的 cpu_math/metal_math；它由该二进制 Candle 的 has_accelerate 报告有效构建能力，ORT 仅在选用时参与，不宣称逐算子 AMX/纯 GPU。rayon 为显式正整数，启动前覆盖；VECLIB_MAXIMUM_THREADS 为冻结的继承环境配置并核对；设置时必须为正的 32 位十进制整数，不承诺观测到的 BLAS 工作线程数。其他未知线程键拒绝。OS/build、硬件、设备、feature、math、线程、环境与二进制改变时不得复用旧 run。

Metal 计时继续采用 production_synthesis_stream_consumer_start_to_completed/1：owner 产出 host PCM，consumer drain 到显式 Completed，导出写盘不计合成。源码审查确认 Qwen/VoxCPM/MOSS codec 输出 host samples、OmniVoice/IndexTTS 返回 host PCM、CosyVoice3 pipeline 交付 host chunks 后才发送成功 End；这是请求输出边界，不是 GPU kernel 时间、整设备空闲或播放完成。各模型实机验证状态独立，不以源码审查标全模型通过。Instruments 保持可选 deferred。

接通证据见 [Mac 实施记录](records/2026-10-20-macos-performance-harness.md)。完整矩阵、资源补齐、正式重复协议、真实模型开销与人工听感属于后续独立基线任务。

恢复前会在 run 锁内核对所有历史 attempt 的 cleanup 证据：缺失、损坏或 cleanup_verified=false 时拒绝 resume，不会忽略旧失败再启动推理。需先在工具外确认遗留进程已清理，再以新的 run 记录重新开始；工具不提供静默覆盖此门禁的开关。
