# Mac 性能工具接通（2026-10-20）

本记录跟随 `connect-macos-performance-harness` 实施更新。仅记录工具与短 smoke，完整 CPU/Metal 基线另行交付。尚未执行的验收不视为通过。

## 生命周期保证

Mac 可信探针在 spawn 时建立独立进程组。原生 libproc 快照发现组内成员和仍可观察的后代，身份包含 PID 与启动时间；已观察到的换组导致验收失败。终止有宽限及强杀预算，父先退出也清理后代；reader 在无法确认清理时可中断，防止继承 pipe 的后代挂住 scoped join。清理失败保留 capture，runner 拒绝下一次推理。

它不是 Windows Job Object 或恶意进程沙箱：轮询前脱离并被重新托管的后代可能不可观察，可信探针不得主动 setsid/换组。SIGINT/SIGTERM 请求取消，正常控制流清理并保存失败证据；SIGKILL/断电无法保证实时回收。PID 启动时间核对降低复用风险，不提供内核级原子信号身份句柄。

## 原生采集与条件

Mac 原生进程组/后代/管道/逃逸测试和最后一进程 terminal 后 SIGTERM 取消及 resume 保留旧 attempt 测试通过。RSS 与 physical footprint 使用 proc_pid_rusage V4；footprint lifetime peak 单列，RSS peak 不可用。无效 PID 的 RSS 为 null 而非零。OS/build、CPU、逻辑核、hostname 使用绝对系统命令读取，Metal 身份只提取 GPU 型号/vendor/device-id/Metal 支持，避免显示器或硬件序列号进入证据。

生产探针的 --execution-info 报告实际链接 Candle has_accelerate 和 host/math 计时契约；runner 核对实际探针，不信任手填 math 标签。未知 math/provider 与不匹配构建拒绝，既有比较/恢复结构已包含 OS、feature、环境、math、线程及二进制，不需改变 schema。Rayon 工作线程配置显式覆盖；Accelerate 环境配置冻结但不宣称实际线程观测。Instruments/Metal allocation 仍不可用，不阻塞普通工具。

当前 stable 编译器对已有 AtomicU64::fetch_update 给出弃用告警；计数器改用同等饱和 compare_exchange_weak 循环，保留 Relaxed 语义与现有 owner 采集回归测试，不提高最低 Rust 版本。

## 免模型端到端证据

`target/perf/mac-tool-fixture-build.json` 为 feature=performance-tools 的独立 release 探针收据。显式 TALECHIME_PERF_PROBE_BUILD 下 CLI SIGINT/SIGTERM 测试通过。空 PATH 执行 run/resume/overhead/report/pack/compare 成功；`mac-tool-fixture-run` 3 个进程各 1 warmup+5 hot，共 15 hot；resume 未新增 attempt。`mac-tool-compare.json` 自比较为 pass。`mac-tool-overhead/overhead.json` 三种观测模式均完整且 PCM 相等，这只是合成接线验证，不代表真实模型开销。

`mac-tool-fixture-report.zip` 解包到 `mac-tool-offline-copy` 后 35 个资产与原报告逐字节摘要一致，manifest JSON 内容一致（打包重序列化去除末尾换行），HTML 相对链接均存在，无 script/fetch。浏览器工具拒绝 file: 协议（仅允许 http/https），未尝试绕过；实际离线图表显示及音频控件验收仍未完成，任务 5.2 保留未勾选。人工可在本机离线打开 `target/perf/mac-tool-offline-copy/index.html` 检查图表、代表音频和逐轮链接。

## 已有模型 smoke

本机 MOSS Nano Candle revision `44502f80dbf9743528fa921cc544d662c685ebec` 的 6 个文件通过生产 manifest；收据 `target/perf/mac-tool-nano-prepared.json`。资源只读检查，无下载、转换或迁移。CPU 使用独立 moss-nano-candle release 构建，内置 Weiguo、seed 42、文字“你好。”、continuation=false、Rayon 1、0 warmup/1 请求，空 PATH 经生产 Registry/Backend/LocalSet 运行，Completed、非空有限 PCM、资产核验及 cleanup_verified 均通过。证据 `target/perf/mac-tool-nano-cpu-run/results.json`，构建 math 为 Candle Rust（非 Accelerate）；一次性 job JSON 在准备阶段手工冻结，测量/恢复/统计/报告不调用 Python。

Metal 使用独立 moss-nano-candle-metal release 构建，其余请求条件与 CPU smoke 对齐；原生身份为 Apple M4 Pro、Metal 4 支持，未回退 CPU。`target/perf/mac-tool-nano-metal-run/results.json` 记录 Completed、有限非空 PCM、asset commit 和 cleanup_verified。Metal allocation 与 RSS 高水位继续 unavailable，footprint 高水位不冒充 RSS/VRAM。这两次 smoke 仅证明各自接线，不建立稳定性能或跨路径输出等价结论，listening_acceptance=pending。

## 回归与状态

并行启动身份查询子进程时暴露了 RunLock 仅靠最后一个文件描述关闭才释放的窗口。复制文件句柄的夹具稳定复现失败；owner Drop 显式 unlock 后回归通过，仍拒绝同时持有两个 RunLock。增加 admission 身份查询失败时直接 child 的有界 Drop 兜底；不能确认后代清理时继续拒绝下一次推理。

本地已通过 xtask 29 个单元测试、显式 SIGINT/SIGTERM CLI 测试、入口测试、2 个 Mac runner 集成测试和 4 个原生进程测试；core 78 个库测试；performance-tools backends 3 个库测试和探针 4 个测试。cargo fmt、xtask all-targets Clippy、backends performance-tools 库/探针/catalog Clippy、私有项 rustdoc 均以 -D warnings 通过。CPU/Metal 的 release 构建与上述真实 smoke 分别记录。Windows 原生执行未在本机运行，保留原 Windows CI；新增 macOS CI 配置尚未在远端运行，不将本地成功等同远端 CI 成功。

当前剩余工具验收仅 5.2：浏览器策略阻止 file: 离线打开。报告资产/链接检查已通过，实际视觉/控件检查未通过，未修改旧 rust-performance-harness 的 7.2 状态，未归档本 change。

## 后续独立基线任务的输入

- 复用当前 xtask/production probe、Mac plan 和 typed evidence；工具能力可用不代表所有模型行 ready。
- 先只读盘点全变体资源，另行执行显式准备/导入/必要转换；本任务没有补齐资源或放宽生产清单。MOSS Nano 的已验证收据可复用为资源起点，不能替代后续逐次生产校验。
- 为每个适用模型/设备构建隔离 release 探针，冻结机器/OS/build、math、线程、资源/音色/语料/参数及重复协议，CPU 与 Metal 分组；缺项/不适用/失败保留，不能把这两个短 smoke 外推到全矩阵。
- 正式多进程基线、真实 off/baseline/trace 开销、统计比较、报告与人工试听在新任务验收；Instruments 仍是可选后续工作。
- 工具合成报告可以验证 UI 和音频控件，但不能用合成 timing 作模型性能排名。

CPU 与 Metal 的真实 smoke 均用最终 runner 执行 resume，核验原有 committed attempt 后复用，各自仍只有 1 个 attempt，没有再启动模型测量。`target/perf/mac-tool-plan.json` 已经本机 CPU catalog 展开并通过 check，28 行全部保留，适用模型维持 unverified/missing_feature，不标全模型 ready。OpenSpec 严格校验和最终范围审查通过；发行、v7、模型算术、旧 change 任务状态及原有 docs/learning 均未修改。代码在本地 `codex/connect-macos-performance-harness`，远端 CI 尚未执行。

## 提交前审查与简化

审查基点 `e591c945f5fcd24d4b167fd20883e70a4bff7541`，覆盖本次未提交实现和随后用户明确要求的整个 tools/ 删除。Standards、Spec 两轴分别复核，完整基线继续不在范围内，离线视觉任务 5.2 保留阻塞。

### Standards

初审发现两项：reader 初始化错误没有进入 fault 通知路径；单个失效/复用 PID 会过早中止其他可确认后代的清理。前者改为启动读取线程前完成非阻塞管道初始化，失败直接由 Process guard 回收；后者聚合身份不确定性、跳过不安全目标并继续清理已验证成员，最终仍报告 cleanup unverified。父进程退出使用 waitid WNOWAIT 观察，直到后代清理结束才 reap，固定原进程组身份，避免 PID/PGID 复用误判。

### Spec

初审发现两项：resume 可能跳过 cleanup 未确认的历史 attempt；VECLIB_MAXIMUM_THREADS 缺少合法值校验。现于 RunLock 内检查全部历史 capture，false/缺失/无效均阻止新 attempt；Mac 显式 Accelerate 线程配置须为正 32 位十进制整数，仍不宣称实际算子线程观测。新增回归先复现错误，再验证修复，包括旧失败不被覆盖、无新推理、失效身份不阻止安全目标回收且无关进程存活。

简化：Mac 管道实现移动到独立命名模块；进程状态改用明确的可变借用，移除 RefCell；恢复流程共享受路径约束的 attempt 目录遍历。按用户后续要求完整删除 tools/，固定语料原字节移入 VoxCPM 测试夹具，现有转换/对照来源和许可证说明保留 Git 固定快照。现有个人 TTS 学习笔记按用户“全部提交”的指令一并纳入，保持原有日期和内容。

两轴最终静态复核均无剩余发现。最终 xtask 31 个单元测试、显式双信号 CLI 测试、入口、2 个 Mac runtime 和 4 个原生进程测试通过；fmt、all-targets Clippy 与私有 rustdoc（-D warnings）通过。使用既有已冻结 probe/build/job，在最终 runner 上重跑空 PATH 合成批次与 resume（仍 3 attempts），并重跑已有 Nano CPU、Metal 各 1 次短请求，均正常 Completed、资产核验及 cleanup_verified；证据分别为 `target/perf/mac-tool-review-fixture-run`、`target/perf/mac-tool-review-nano-cpu-run` 和 `target/perf/mac-tool-review-nano-metal-run`。这些是接线复核，不是新基线。离线报告视觉/控件验收仍明确未完成。
