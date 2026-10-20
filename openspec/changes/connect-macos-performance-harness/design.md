# Design

## Context

动机及交付范围见 proposal.md。`rust-performance-harness` 已有共享版本化证据、生产探针、恢复与报告实现，但尚未归档，正式 specs 库为空。本 change 新增单一 Mac 平台能力，并依赖该 change 的共用契约；不复制或抢先同步其 specs，也不要求其尚未完成的离线视觉验收先结束。

源码现状：`process.rs` 非 Windows spawn 拒绝，`executor.rs` 非 Windows execute 拒绝，`runner.rs` 独立限制 Windows；`cancellation.rs` 非 Windows 不安装信号处理；`sampling.rs` 非 Windows 仅 deferred RSS；`environment.rs` 缺 Mac 真实硬件/OS 信息；`plan.rs` 将 Mac 适用项一律 deferred。`Conditions` 当前仅接受一个正整数 rayon 设置，`VECLIB_MAXIMUM_THREADS` 只是环境字段，不能冒充已验证的 BLAS 有效线程数。性能 CI 仅 Linux/Windows，原生进程和取消测试仅 Windows。

## Goals / Non-Goals

**Goals:** 在现有执行接口内加入 Mac adapter；先证明隔离、终态与资源证据，再解除对应工具门禁。CPU 和 Metal 使用独立 feature 构建和独立条件，复用同一事件/资产/恢复链路。

**Non-Goals:** 不做全模型资源导入、下载/转换、完整基线、真实模型 overhead 验收或性能优化；不新增第二个 runner、产品命令或发行二进制；不建设 Linux 原生执行，不将 Instruments 作为本次必需实现。

## Decisions

### 1. 保留共用执行状态机，平台只负责原生资源

抽取/复用 executor 的日志排空、预算、采样、取消、终态与提交规则；Mac 平台模块实现与 Windows Process 等价的 spawn/pipes/id/wait/cleanup 能力，runner 按已实现平台路由。明确区分 macOS 与其他非 Windows target，不能把原有 `not(windows)` 全部当作 Unix 支持。

替代方案是复制 Mac executor，容易让预算、恢复与异常清理分叉，故不采用。Windows 原生实现和证据语义保持不变。

### 2. 受控进程组加有界回收，避免宣称 Job Object 等价保证

使用 spawn 时原子建立的新进程组，避免 spawn 后再 setpgid 的竞争。保存直接 child handle、组身份和必要的进程身份信息；取消/超时先请求终止，经过短宽限后强杀受控组，等待直接 child 并在总清理预算内核对存活成员。父进程先退出也必须清理组，且清理先于可能等待 EOF 的 reader join。Drop/错误路径具有同样的有界清理兜底，清理失败不得启动下项。

该工具运行可信生产探针，不提供对恶意 setsid/换组进程的 OS 沙箱保证。支持的探针及辅助子进程必须保留受控组；记录可观察到的后代身份和换组异常，不能证明回收时 fail closed，测试覆盖已知逃逸夹具。仅 kill 直接 child、全局按进程名杀进程和 shell `pkill` 均不满足契约。PID/PGID 复用和进程枚举竞争需保守处理，身份不确定即停止，不能为清理杀无关进程。

### 3. 信号只触发取消，状态机持有清理权

Mac SIGINT/SIGTERM handler 只做信号安全的原子通知，正常控制流写日志/终态、终止组并停止排程；guard 恢复先前处理方式。优先复用合适的已审查信号库或最小 target-gated 绑定，不在回调内分配、等待或写文件。CLI 测试对隔离 runner PID 发信号，验证最终一轮后的取消检查同样生效。

SIGKILL/系统断电不能执行 handler；保留未提交 attempt 并由现有恢复校验识别，不宣称这类终止也完成了实时清理。

### 4. 采集原生可验证指标，未知值不填零

Mac 进程只读查询优先使用系统原生进程信息接口，采样当前 RSS；OS 提供且可验证的 lifetime peak 单列。优先评估 libproc 的进程资源查询及可用高水位字段，权限/进程退出/系统版本不足给出 null+reason，不使用当前 RSS 或 sampled max 填充 OS peak。限定查询成本、保留 interval/source/scope，避免外部命令每个采样周期启动进程。

机器身份采集 OS 版本/build、架构、CPU 型号、逻辑核数、Metal 设备标识；不采集硬件序列号/凭据。不修改电源或线程系统设置。少量系统命令如确需使用，使用绝对系统路径、有界输出和超时，使测量不依赖用户 PATH 或 Python。

### 5. 构建身份与线程配置必须可归因

复用 Cargo artifact JSON、feature 隔离目录和摘要。通过实际 probe/catalog 或构建元数据声明 Accelerate/Rust/混合实现，不能只由 target 或用户填写 `math_backend` 推断。rayon 正整数仍由 runner 显式覆盖。Mac 新增受支持的 Accelerate 环境线程控制时，扩展 Conditions 的类型化允许集合与验证，并在子进程初始化前应用；记录“配置值”与“实际执行线程观测”的区别，未测后者不承诺精确线程数。旧 Windows rayon-only 条件仍有效。

若序列化结构或旧证据解释改变，按现有版本机制显式升级/拒绝，不悄然解释旧证据。比较和恢复核对 OS/build、math、feature、线程与 provider；无法验证真实执行条件时阻塞真实 job，合成 fixture 使用明确 synthetic 身份。

### 6. Metal 以生产 host PCM 边界接通，不插入逐阶段同步

探针继续通过 Registry/Backend/LocalSet/owner 交付实际 host PCM，drain 到显式成功终态。逐后端审查 host 转换与 End 的关系，保存已验证路径的 timing_contract；无法验证的路径保留 unverified。host span 不是 kernel 时间，host PCM 完成不代表整设备空闲或播放完毕。

本次不承诺原生 Metal GPU 内存 collector。RSS、OS footprint、设备 allocation 如果有来源则分别标注，未实现项 null+reason，不能将统一内存称作独占 VRAM。Instruments 维持 deferred 且不阻塞工具，无需为通过工具验收新增 profiler adapter。

### 7. 按能力激活，模型验证保持独立

在 plan/doctor/报告中保留工具生命周期和采集能力的证据状态，以及模型行的 feature/资源/provider readiness；避免一个全局 Mac 布尔值代表所有模型。验证通过的原生能力可 available，未实测模型仍 unverified，缺资源明确 blocked，不适用组合保持原样，包括 BF16 Metal 不开放。不得用移除所有 deferred 的测试作为验收。

### 8. 工具闭环与基线任务分开

先完成确定性 fixture 的 build/run/resume/compare/report/pack，以及 off/baseline/trace 的接线和产物核验；无 Python 测量路径，不承诺真实开销阈值。报告在复制/解包后的新路径核对资产和相对链接，离线打开图表和代表音频；若浏览器限制拒绝本地文件，明确留待人工验证，不以 HTTP 预览冒充完全离线打开。

然后只盘点本机已有生产校验通过的资源，选最小可用 CPU 和 Metal 各一个短请求，可使用同一支持两种设备的模型；记录源码/二进制/资源/条件、有限非空 PCM、成功终态、回收。无合适资源则记录具体 blocked，不隐式下载；工具夹具通过与 smoke 未完成分别报告，未过的验收任务保持未勾选。无需跑全部后端、正式重复协议或人工听感，报告中 listening 继续 pending。

## Risks / Trade-offs

- [进程组无法提供恶意后代的强隔离] → 限定可信探针契约、测试逃逸与异常关闭、身份不确定时 fail closed；不宣称覆盖不可观察的逃逸。
- [OS 高水位或 Metal 指标不可用] → 能力和 reading 单列 unavailable，不阻塞已验证的基础执行，不补造数据。
- [Accelerate feature 不等于逐算子 AMX/线程实测] → 保存有效构建与配置来源，限制结论范围。
- [共享状态机重构影响 Windows] → 保留 Windows 进程/取消回归测试，由原生 Windows CI 验证，Mac 本地编译不能代替。
- [主机无资源或 hosted Mac 无 Metal] → CI 只做无模型契约，实机 smoke 单独记录 blocked/通过，不触发模型下载。
- [原 change 尚未归档] → 引用共用契约，不修改其任务状态；归档时复核能力边界，完整基线留给后续独立 change。

## Migration Plan

先添加 target-gated adapter 和无模型测试，再启用 Mac runner/plan 能力，最后验证本机已有资源 smoke。产物均放 Git 忽略目录。更新开发文档与 Mac 接续记录，列出已激活能力、缺指标和后续基线任务输入。回滚代码仅恢复工具 deferred，不删除既有模型、音色、证据或原始资源。
