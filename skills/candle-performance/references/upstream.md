# 上游补丁、依赖与发布

检索 issue/PR 时以当前源码、官方仓库和目标 revision 为准。先从算子/错误与最小复现
找对应实现，记录 PR 的 base/head、crate 版本、支持设备/dtype、选择阈值、测试与
合并状态。旧报告中的 open/merged 状态不是今天的事实。

比较 PR diff 与已锁源码：补丁可能已合并、被其他实现替代，或者只支持 CUDA 而目标
是 Metal。仅将依赖改成 main 不保证获得目标修复；PR head 的 Candle 版本也可能早于
工作区，不能为一个算子无意回退全部模型。先判断公开 API 能否表达局部修复。

适合上游的行为补丁应有小范围复现、选择条件/fallback、原生目标数值对照和准确的
性能口径。不能因为私有模型隐藏态不一致就断言框架存在缺陷。未经任务授权不创建
公开 fork、不推送或提交 PR；本 skill 本身不授权这些外部动作。

## 依赖接线

统一 Candle 家族的版本与来源，以及直接 cudarc 与 Candle 使用的包实例。
检查 metadata/tree 的实际 feature 组合和目标平台；同版本也可能来自 registry 与
git 两个来源。Talechime 曾漏 patch Metal kernels，Windows CPU/CUDA 树未暴露问题。
fork 内部路径 crate 与工作区直接依赖都要检查，避免重复编译与不兼容类型。

直接依赖 cudarc driver 时只声明需要的能力，核对其 default-features 是否额外激活
runtime/加载策略；与 Candle 的链接/加载策略一致。不要因此强行升级隔离链路中不同
版本的 cudarc。目标 gating 同时应用于 manifest、源码模块与原生测试。
Windows 保持目标项目约定的 CRT；不通过全局 NODEFAULTLIB 掩盖混用。

## fork 与发行契约

固定可审计 revision、保留许可与来源、补丁台账、上游关联和删除条件。
工作区 patch 不随发布库成为下游默认来源，publish dry-run 应验证去 patch 后的
实际 registry 依赖。工作区编译成功不代表发布库依赖 fork 独有 API 时也能编译。

若新增 fork 独有 API，先确定受影响库的发行策略；暂停发布、等待上游、显式替代
依赖等由目标项目约定决定。Talechime 的 publish=false 轨道是当地政策，不强加给
其他仓库。dry-run 不是正式发布授权；不为掩盖其他包的发布顺序问题更改无关版本。
