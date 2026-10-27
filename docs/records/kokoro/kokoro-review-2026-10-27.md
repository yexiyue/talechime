# Kokoro 提交前审查与简化：2026-10-27

按项目规范和 OpenSpec 两个维度分别审查。原始基线为
`23c7d3e21fa8f8c550f9df5839ab07f81e9ed619`；用户确认先提交固定参考接口前置改动，
再提交 Kokoro。其他 VoxCPM 诊断实验和旁白实验文档不纳入这两笔提交。

## Standards

- **取消所有权问题，已修复。** 原资源准备阶段 `spawn_blocking` 在 future 被取消后仍会
  后台哈希。现在 `VerificationTask` 持有专用校验线程，Drop 先取消、再 join；哈希循环
  检查取消。普通回归验证 abort future 返回前线程已经退出；真实资源探针也覆盖 prepare。
- **可能的重复代码，已简化。** 资源准备只在出口构造一次完整错误上下文，删除重复读取
  manifest 和 `Option<callback>` 分支。共享非取消校验入口按实际调用 feature 编译。

另外将上下文模型的内部 `Data` / `Query` 改为 `ContextData` / `PronunciationQuery`，
展开 benchmark JSON；用 Black 25.1.0 格式化六个开发导出脚本，逐文件验证 Python
AST 前后完全一致。没有改动已验证的声学算术顺序或降低数值门槛。

## Spec

- **[P2] 全角数字，已修复。** 固定版 cn2an 支持 `１２３` 和混合宽度日期，原生端仅用
  ASCII 正则会拒绝这些输入。现在只转换 ASCII/全角数字，源范围分段和后端数字边界
  复用相同判定；不对全文 NFKC，也不扩大到官方未支持的其他 Unicode 数字。
  独立官方样例覆盖价格、日期、前导零、百分数、分数、范围及六位年号边界。
- **[P2] `eR` 儿化，已修复。** 原生把 `eR` 映射成 `er` 后还追加 `R`，导致“这儿”
  多一个 token。现在转换同时消费该标记，其他 finals 的儿化规则保持不变。
  普通回归和官方“这儿/哪儿/那儿”样例均通过。

两个维度复核均无剩余发现。Standards 原有 1 项硬问题与 1 项判断性问题均已处理；
Spec 原有 2 项 P2 均已修复。

## 验证与提交边界

- 第一提交使用从暂存区直接导出的独立源码快照：默认 workspace lib/tests/examples
  **790 passed / 0 failed / 41 ignored**；workspace all-targets Clippy `-D warnings`、
  包含 examples/private items 的严格 rustdoc、fmt 和 OpenSpec strict 均通过。
- Kokoro 相关三个包的最终组合（CLI、g2pW、Metal）：lib/tests
  **89 passed / 0 failed / 2 ignored**，其中模型库 11 项、适配器 22 项、facade/CLI 56 项。
  相关 all-targets Clippy、严格私有项 rustdoc、fmt、Python 语法检查通过。
- 扩展官方中文前端夹具 **29 例 / 0 失败**：28 例中文完全一致、1 例英文明确拒绝。
  新增的 7 例来自独立官方执行，不缓存原生结果。
- g2pW 官方输入/概率对照 **24/24**；人工目标读音仍为 **23/24**，已知“解先生”误读
  保留在报告中，不声明通用姓名正确率。
- 真资源取消探针：prepare 校验取消含 join **2.57 ms**；load 取消含 join **15.46 ms**；
  满 PCM 队列且接收端仍存活时 drop/join **52.13 ms**，没有成功 End。

原始日志在 `target/kokoro-review/` 与 `target/kokoro-official-research/review-*.log`，
均为忽略的本地开发证据。原有官方声学 CPU/Metal 结果见
[实现验收](kokoro-native-validation-2026-10-27.md)；本轮不将命名/排版简化当作新的性能优化，
也不新增 CUDA、完整章节试听或实际播放验收声明。
