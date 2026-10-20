# Tasks

## 1. Mac 原生执行与有界回收

- [x] 1.1 增加 macOS target-gated Process 模块和必要依赖，spawn 时建立独立进程组，接入 pipes/id/wait；以无模型子进程测试验证组隔离、spawn 失败无泄漏及其他平台仍显式拒绝，补充平台保证范围文档。
- [x] 1.2 实现组终止、宽限后强杀、直接 child wait、存活核对及 Drop/错误路径兜底；测试正常退出留孙进程、父先退出/后代持 pipe、忽略终止信号及无关进程存活，确认总清理预算有界。
- [x] 1.3 接入可观察后代身份/换组异常和清理不确定失败路径；已知逃逸夹具验证不虚报 cleanup_verified、不提交成功且不启动下项，测试夹具自行清理逃逸进程，并记录不可观察逃逸的保证限制。
- [x] 1.4 将 executor 共享状态机接入 Mac 并解除 runner 对已实现平台的硬门禁；在 Mac 运行 timeout/flood/截断/无终态/取消后下一成功夹具，验证日志证据、reader 排空、错误回收和串行调度，保留 Windows 原生回归测试。

## 2. CLI 取消

- [x] 2.1 实现 Mac SIGINT/SIGTERM 原子通知和 handler 生命周期，执行清理留在正常控制流；隔离进程测试验证两个信号均保留 cancelled attempt、非零退出、无成功 commit、无下一轮及子孙回收，文档说明 SIGKILL 不提供实时清理保证。
- [x] 2.2 覆盖最终轮次后取消与异常退出恢复；CLI/存储测试验证取消不会返回成功、未提交 evidence 不被 resume 当成功复用，并保留旧 attempt。

## 3. 原生采集与执行条件

- [x] 3.1 增加 Mac 原生 RSS 查询和可用高水位/缺失原因输出；分配内存的无模型进程及查询失败/短进程夹具验证字节单位、process scope、interval、sampled max 与 OS peak 不混用，更新指标字典。
- [x] 3.2 接入 Mac 机器、OS/build、CPU/逻辑核与 Metal 设备身份采集；验证相同环境稳定、必要身份缺失显式失败、命令调用有界且不依赖用户 PATH、不采集硬件序列号，补充 doctor 示例。
- [x] 3.3 从实际 probe/catalog 或构建元数据接通 CPU math/feature 证据；按需扩展 Mac 受支持线程控制的验证及子进程覆盖，测试非法键/零值/环境继承冲突/不匹配构建被拒绝，Windows rayon-only 仍有效；文档区分配置值和实际算子/线程观测。
- [x] 3.4 将 Mac 身份、math、线程和 provider 纳入冻结、比较及恢复校验；夹具验证身份变化拒绝复用、原证据保留、无静默设备回退；若 schema 解释变化则增加版本拒绝测试并记录版本策略。

## 4. Metal 边界与能力激活

- [x] 4.1 审查生产探针各 Metal 路径的 host PCM/End 边界并登记 timing_contract，无法确认的行保留 unverified；用确定性探针验证错误/截断/取消不是 Completed，不新增逐 span 同步，更新完成边界表。
- [x] 4.2 输出 Metal allocation 等未实现指标的 null/source/scope/reason，保留 Instruments deferred；夹具验证统一内存/RSS 不标作独占 VRAM、缺 profiler 不阻塞基础执行，更新指标说明。
- [x] 4.3 将 Mac 工具能力与模型行 readiness 分开，接通 plan/doctor/fixture-job；矩阵测试验证适用项保留、BF16 Metal 等不适用项不改变、缺 feature/资源不消失、工具通过不冒充全模型已验证，给出 Mac 计划示例。

## 5. 无模型端到端与报告接线

- [x] 5.1 在 Mac 执行合成 fixture 的 plan/build/run/resume/compare/report/pack，并验证 off/baseline/trace 模式接线；测量阶段使用不含 Python 的 PATH，核对事件、资产摘要、兼容复用、失败 attempt 保留及无模型下载，保存忽略目录内证据和可复现命令。
- [ ] 5.2 将报告解包/复制到新路径并核对相对链接、摘要和无远程依赖，实际离线打开图表及代表音频控件；记录工具级播放检查与 listening=pending，若本地文件策略阻塞则保留本项未完成并明确人工步骤，不代勾旧 change 的验收。
- [x] 5.3 在 performance-contracts CI 增加 Mac 无模型契约与 CLI 取消/fixture 闭环，保留 Windows 原生回归任务；核对工作流不下载模型、无需 GPU/音频设备，更新开发说明中的平台检查命令。

## 6. 已有资源短 smoke

- [x] 6.1 只读盘点本机已有资源，选择 CPU/Metal 各一个最小短请求所需的已校验模型和已存在音色/内置条件；交付资源状态与生产 manifest 校验记录，不复制迁移、下载、转换或修改资源清单，缺项明确 blocked。
- [x] 6.2 通过独立 release CPU 构建和 runner 执行一次短生产请求；记录二进制/资源/参数/math/线程身份、有限非空 PCM、显式成功终态及 cleanup_verified，标注 wiring smoke；缺资源不勾选成功。
- [x] 6.3 通过独立 release Metal 构建和 runner 执行一次短生产请求；记录实际 provider、owner→host PCM 完成边界、输出/终态及回收，不回退 CPU；资源或 Metal 不可用时保留 blocked，不勾选成功。
- [x] 6.4 交付 Mac 接通记录与后续基线任务输入清单，核对工具、CPU smoke、Metal smoke、离线报告验收状态分别呈现；完整矩阵、资源补齐、正式重复测量、真实开销与听感均明确属于独立后续任务，不将短 smoke 解释为基线。

## 7. 集成检查

- [x] 7.1 运行 cargo fmt --all --check、相关 xtask/backends 无模型测试、all-targets Clippy -D warnings 与 rustdoc -D warnings；单列本机 native Metal 编译、Windows CI 原生执行和实际 smoke 证据，失败或未运行不标通过。
- [x] 7.2 执行 OpenSpec 严格校验并对照 spec 全部场景审查任务与证据，确认只激活已验证能力、不覆盖旧 change 任务、不改发行/v7/模型算术，所有工具产物和资源仍未进入 Git。

## 验收状态（2026-10-20）

实现、免模型闭环和已有 MOSS Nano 的 CPU/Metal release 短 smoke 已完成；证据见 `docs/records/2026-10-20-macos-performance-harness.md`。仅 5.2 未完成：Browser Use 的 URL 策略拒绝 file: 协议（只允许 http/https），因此没有实际离线视觉/音频控件验收，也未绕过限制。解包后 35 个资产摘要、manifest 内容和本地相对链接检查已通过。人工验收入口为 `target/perf/mac-tool-offline-copy/index.html`。不以静态检查替代实际打开，不归档本 change。
