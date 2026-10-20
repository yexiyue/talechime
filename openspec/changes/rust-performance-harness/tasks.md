# Tasks

## 1. 开发入口与变更协调

- [x] 1.1 在本 change 交付 Windows 基线任务映射，明确原 change 的资源契约、Rust 工具复用和真实全量执行归属；实施前确认相关任务冲突已协调，验证没有两套权威 runner。
- [x] 1.2 新增不发布的轻量 xtask member 与 Cargo alias，接入 perf 子命令骨架；验证 help/doctor/report 不链接模型、ORT 或打开音频设备，default-members 和发行 executable 不变。
- [x] 1.3 编写 Rust 工作流与 Python-only 显式准备例外文档；在无 Python PATH 的测试环境验证帮助与免模型执行，不删除既有历史对照脚本。

## 2. 平台计划与证据 schema

- [x] 2.1 实现版本化 plan/case/attempt/round/observation/artifact 数据及源码 dirty/untracked 摘要；用夹具验证未知版本拒绝、身份变化、有效参数、原始字节摘要与序列完整性，文档给出兼容策略。
- [x] 2.2 实现 Windows/macOS 平台计划和能力状态展开；夹具验证 CPU/CUDA/Metal、Accelerate 区分、缺 feature、不适用、deferred/unverified 保留且不回退，输出 Mac 示例计划。
- [x] 2.3 通过生产 AppPaths/catalog 解析持久目录与已准备资源，消费校验收据；夹具验证不重写 HOME、不触发自定义路径迁移、缺导入功能明确 blocked、测量不下载或转换。
- [x] 2.4 编写指标字典和平台采样契约，含 source/unit/scope/interval/status、OS 高水位与采样最大值；schema 测试验证缺失原因、Mac 统一内存与独占显存不可混用。

## 3. 显式构建与隔离执行

- [x] 3.1 使用 Cargo artifact JSON 与隔离构建目录定位 CPU/Candle CUDA/ORT CUDA 探针；夹具验证缺 artifact、错误 feature/binary 身份拒绝，并记录命令与构建后冻结流程。
- [x] 3.2 实现串行子进程状态机、版本化事件排空、日志/帧/磁盘预算；假子进程验证 stdout/stderr 大量输出无死锁、序列缺失/无终态/超长帧不是成功，文档说明预算。
- [x] 3.3 实现 Windows 受控进程树、超时与取消回收；含孙进程的夹具验证退出/强杀/未回收停止后续推理，不影响非本工具进程；Mac lifecycle 能力明确 deferred。
- [x] 3.4 实现 run 锁、音频与日志核验、原子 committed 标记和恢复；夹具验证双 runner、中断、缺文件、binary 改变、旧失败保留及不兼容版本拒绝，文档提供恢复示例。

## 4. 生产探针与统一测量

- [x] 4.1 接入生产 Registry/Backend 的无模型 catalog 和具名请求探针，保留 LocalSet/owner/有界流；确定性 Backend 验证模型/模式拒绝、正常 End、截断、错误及取消不混淆。
- [x] 4.2 实现 verify/load/reference/首请求/warmup/hot 边界和类型化原始 observation；延迟 Backend 验证懒参考归属、父子阶段重叠不可求和、导出 I/O 不计时及缺指标理由，更新测量文档。
- [x] 4.3 在可观测 owner/adapter/consumer 边界接入首 PCM、chunk gap、发送等待和队列音频秒数；背压夹具验证排队不冒充推理、取消不死锁，阶段不适用保留原因。
- [x] 4.4 审查并验证 Windows CPU/CUDA 请求完成边界，复用模型 owner 的生产执行路径；验证不引入逐 span 同步、不以无关 CUDA context 证明完成，记录各后端支持与未验证边界。
- [x] 4.5 实现有界 PCM 持有/暂存及计时外 F32 WAV/PCM16 派生导出；夹具验证完整样本、非有限值拒绝、摘要关联与预算超限明确失败，不改变输出 chunk 内容。

## 5. tracing、metrics 与平台采样

- [x] 5.1 在开发入口组合 subscriber/Recorder，贯通请求和 owner 线程，复用 Qwen span 并补必要公共阶段；跨线程/async 测试验证相关 ID、不跨 await 持 guard、库层不抢全局配置与 worker stdout 不受污染。
- [x] 5.2 实现有界 counter/gauge/histogram 采集与逐 attempt snapshot；夹具验证 owner 线程采集、标签白名单、无历史累积污染、原始结果不由 histogram 反推，文档列指标单位与语义。
- [x] 5.3 实现 baseline/trace/profile 配置及本地 trace writer 生命周期；压力夹具验证缓冲有界、丢失披露、正常 flush/异常退出完整性、诊断不并入 baseline，记录采集模式身份。
- [x] 5.4 实现 Windows 原生进程采样及 Rust 调用 nvidia-smi 的低频设备采样；夹具验证命令超时/缺失、RSS/private bytes/整卡 VRAM 单位与范围，记录采样间隔和 OS peak 来源。
- [x] 5.5 提供可选 Nsight/WPR 诊断 artifact 接口和外部工具版本记录，Mac Instruments adapter 明确 deferred；无工具夹具验证不阻塞 baseline，文档区分 host span 和设备原生时间。
- [x] 5.6 实现同条件无采集/基础/详细采集开销对照流程；免模型夹具验证输出条件核验、独立进程与分布报告，未做真实模型开销验收时明确 unverified，不承诺固定开销上限。

## 6. 统计、比较与微基准

- [x] 6.1 实现可配置重复协议与原始轮次/进程均值/mean-of-means/RTF 汇总；夹具验证预热排除、多段时长加权、缺轮与失败覆盖，文档区分 baseline 和 ABBA 优化协议。
- [x] 6.2 实现比较兼容性检查和显式 metric policy；夹具验证 source 改动允许、平台/驱动/math/参数/观测变化拒绝，五种判定、双阈值/区间规则及音频等价证据不足不声称保持输出加速。
- [x] 6.3 为已有真实热点建立小规模 Criterion 无模型基准，优先分段或重采样；验证稳定输入及 benchmark 可执行性，CI 不以共享机器墙钟硬门禁，文档给出本地比较入口。

## 7. Rust 报告与跨平台接续

- [x] 7.1 实现 Rust JSON/CSV/HTML/SVG 汇总、覆盖和确定性代表音频选择；成功/失败/缺值夹具验证实际数值、单位、平台分面、listening=pending 与原始结果链接。
- [ ] 7.2 实现报告安全路径与不含模型的打包；夹具验证文本转义、路径越界/符号链接逃逸拒绝、资产完整性，并在新路径离线打开验证无需 Python/CDN/fetch/原 home。
- [x] 7.3 交付 Mac 后续接续清单：native lifecycle、RSS/高水位、CPU math/线程、Metal 完成边界/内存可用性、可选 Instruments、开销和真实模型独立基线；验证每项对应 capability/证据字段且未实现项不标成功。

## 8. 集成验证与交付

- [x] 8.1 接入无模型 CI 契约/假子进程/微基准 smoke 与 CPU 检查；运行 cargo fmt、相关测试、all-targets Clippy、rustdoc（-D warnings），保留 native CUDA/Metal 编译和真实性能分离的记录。
- [x] 8.2 在 Windows 完成无 Python 端到端夹具运行/恢复/比较/报告；有已准备资源时显式执行短真实模型 smoke，否则披露缺失，不下载、不冒充全量矩阵或真实开销验收。
- [x] 8.3 审查最终工具与规划一致性、发行边界、用户数据保护及与原 Windows change 的复用关系；运行 OpenSpec 严格校验，交付命令、报告样例、限制和 Mac 接续文档。

## Workflow follow-up

- 2026-10-20：7.2 的安全路径、白名单打包、解包摘要和相对链接检查已通过；浏览器策略拒绝 `file:` URL，离线视觉打开/音频控件验收尚未完成，故保留未勾选。详见 `docs/records/2026-10-20-rust-performance-harness.md`。
- Mac native collector、生命周期实现与 CPU/Metal 真实模型验收由后续 Mac change 在实际机器上完成，不属于本 change 未完成任务。
- `benchmark-windows-backends` 使用本框架继续资源准备、全量模型矩阵与人工试听；其计划调整需协调，不能本次悄然覆盖。
- 本 change 完成实现与复核后再归档；不将规划文档完成当作性能基础设施已实现。
