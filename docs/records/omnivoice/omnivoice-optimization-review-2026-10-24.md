# OmniVoice CPU / Metal 全链路审查与简化（2026-10-24）

固定审查范围为 `git diff ec3b862796362123a9313607bad6a5503af5eaaf...d13385b`，
即当时 main 到 CPU 7-tap 的 24 个提交。标准轴与需求轴由独立审查并行执行，
范围包括生产实现、离线探针、macOS Accelerate 接线及实验结论；数值 JSON 用作证据，
不将数万行原始数据当作需要手工重构的代码。
需求来源为本任务明确计划与对应 `docs/records/omnivoice/` 验收记录。
仓库缺少 issue tracker 配置，本次没有依赖外部 issue 状态或创建新 issue。

## Standards

1. **P2，已修复：decoder 归因路径与生产 conv7 分叉。**
   `stage1_model/cpu_attribution.rs` 仍通过通用 `Conv1d` 测 res_unit1 的 conv7，
   但生产已使用受限 `forward_decoder_conv7`。逐位输出相同不能发现这个路由偏差；
   当前探针改用生产入口。历史 conv7 原路径控制与已提交计时不重命名为新测量。
   对应标准：Candle 技能要求先确认实际计算路径，隔离组件不能冒充生产阶段耗时。
2. **P2，已修复：诊断 canary 的全局缓存跨 device 实例。**
   原 canary 缓存首个设备的张量，后续 runtime 可能执行首个设备的矩阵乘法、
   却同步自身设备。缓存改归 `Stage0RuntimePlan` 的私有 `OnceLock`，只在诊断开启时初始化。
   CPU 继续跳过不支持的 F16 canary；Metal 测试验证两个独立 device 实例的张量归属与同步。
   对应标准：measurement 规定同步实际执行的设备，资源随 owner 生命周期管理。

手工重复的 decoder 遍历存在 possible Duplicated Code 风险；保留其独立数值 oracle，
通过复用生产 primitive 入口修复路由分叉，没有引入新的公共遍历框架。

## Spec

1. **P2，已修复：K/V 转置缓存未排除非连续权重。**
   `simplify` 要求“All original features, outputs, and behaviors must remain intact”，
   本次需求审查据此检查受限缓存的活权重/非连续权重回退与原错误行为。
   原 `CpuProjection::load` 排除了活权重，但没有连续性条件，可能缓存被 narrow 的跨行权重，
   将原 BLAS 不支持的 stride 从错误改为成功，并引入未经验收的布局。
   缓存增加 `weight.is_contiguous()` 条件；测试覆盖转置权重、带行间距且非零 offset 的权重，
   同时比较原输出或原错误，不通过放宽容差接纳该布局。

审查计数为 Standards 2 项、Spec 1 项，全部修复；不是“原分支无问题”。

## 简化与文档

- 将五个内嵌测试模块移入 `stage0_qwen3/{cpu_tests,metal_tests}.rs`、
  `stage0_model/{scoring_tests,metal_tests}.rs`、`codec_layers/tests.rs`。
  保留测试 module 路径、feature 条件与算术，比较确认除 rustfmt 空白/可选尾逗号外内容不变。
- 删除未调用的私有 `try_flash_attention_per_row` 空实现，attention 调度改为早返回。
  公共函数签名、GPU/CPU 路由条件和张量运算顺序不变，没有启用 CPU Flash。
- 不合并 CPU/Metal 卷积、GQA、缓存的不同回退前提，不将早期授权数值变化改写为逐位等价。
- 新增 [当前路径概览](../../knowledge/omnivoice-performance.md)，更新 README 和文档索引，
  修正“OmniVoice 缺少官方对照”的过时说法，明确 CPU 最终组合没有新的配对官方排名。
- 更新 [Candle 技能](../../../skills/candle-performance/SKILL.md)：新增 CPU 经验、
  设备归属/真实入口规则，区分 CPU 已保留 7-tap 与 Metal 已撤回候选。

本轮是审查、错误边界修复和结构整理，不宣称新的吞吐提速；沿用有版本和设备条件的历史性能证据。
普通测试不下载模型，真实输出另由显式本地模型验收。

## 普通验证与独立复审

- CPU release：40 passed / 7 ignored；Metal release：55 passed / 7 ignored。
- 宿主 OmniVoice + performance-tools 单元测试：11 passed。
- 宿主 CPU feature 组合 `omnivoice,qwen,voxcpm,moss-candle,cosyvoice3,indextts,asr` 的
  `cargo check --locked` 通过，覆盖本分支 macOS Accelerate feature 合并。
- CPU/Metal 两套 `--all-targets` Clippy（`-D warnings`）、Metal rustdoc
  （`-D warnings --document-private-items`）、格式检查通过。
- `skill-creator` 的 `quick_validate.py` 通过；性能脚本语法及新增概览/技能链接检查通过。
- 两位原审查者独立复看修复 diff，分别确认所属轴的原 P2 已修复，没有新可执行发现。
  审查者只读源码，编译、测试、真实模型验收由本地执行，两种证据不混写。

这些检查仅为当前 Mac 环境，不代替 Windows/CUDA、其他 BLAS 版本或真实其他模型的性能验收。

## 最终真实模型复验

全部编译完成后串行运行，与普通测试分开，模型与请求使用原不可变归档。
CPU 对照 `d13385b` 已验收归档，Metal 对照免拷贝版本归档；后续 CPU 专用优化没有
将 Metal 的早期数值变化重新作为本轮“逐位保持”的基线。

| 链路 | 文本 | 完整 codes | 全部分段/整段 F32、PCM16 WAV |
| --- | --- | --- | --- |
| CPU | 短 | 逐字节一致 | 4 个，逐字节一致 |
| CPU | 长 | 逐字节一致 | 24 个，逐字节一致 |
| Metal | 短 | 逐字节一致 | 4 个，逐字节一致 |
| Metal | 长 | 逐字节一致 | 24 个，逐字节一致 |

共 56 个 WAV，按同后端比较；没有比较 CPU 与 Metal 彼此逐位相同。
完整 codes dump 包括参考预热和所有生成分段。WAV 字节对照同时覆盖样本数与全部波形，
不将完整 SHA256 对照改成抽样误差或 SNR。

另使用当前 CPU release 测试二进制运行真实 forward 重放，9 个 step/layer 的完整输出
与原归档逐位一致；修正后的 decoder 归因探针覆盖原有14组固定 codes，
重复解码逐位检查通过。其中130帧另对照 trace 和独立消费者，conv7 分段使用实际生产 helper。
该同步/分段诊断结果不作为新的整链吞吐报告，也不覆盖原来的历史实验数字。

六个真实模型验证进程均成功，编译活动采样为 0；没有据此保证系统完全空闲。
最终源码、CPU/Metal 二进制、请求、WAV/codes 哈希与运行 receipt 保存在
[配套 JSON](omnivoice-optimization-review-2026-10-24.json)，原日志和冻结产物在
`target/perf/omni-review/`，不提交模型、张量或音频。

完成后将本轮修复/简化、文档和技能一并提交，按用户要求快进合并并推送 main；
没有发包、创建 release 或部署应用。
