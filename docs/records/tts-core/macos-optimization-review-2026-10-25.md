# macOS 优化审查与简化（2026-10-25）

审查分为两个固定范围：已提交部分使用 `git diff ec3b862...HEAD`，HEAD 为
`42a7887e28f095684266eac43a68bd8fcc303d11`，覆盖 OmniVoice 的 macOS CPU / Metal
优化及 macOS Accelerate 依赖接线；未提交部分以该 HEAD 为基线，包含 CosyVoice3、
IndexTTS 与新增宿主融合内核。修改前工作区已存在这些优化，本轮在其上修复和简化。

标准轴与需求轴分别由独立审查执行。需求依据对应模型的优化记录、
[parity 计划](../plans/official-parity-optimization-goal-2026-10-24.md) 与
[IndexTTS 计划](../plans/indextts-exceed-official-goal-2026-10-24.md)。
仓库缺少 `docs/agents/issue-tracker.md`，未读取外部 Issue 的最新状态；若需配置技能的
远端 Issue 工作流，可运行 `/setup-matt-pocock-skills`。

## Standards

1. **P2，已修复：Candle 来源声明落后于实际 pin。** 更新
   `docs/legal/candle/SOURCE.md` 的 revision 与补丁清单，标明未接入调度的实验内核。
   依据 `docs/development.md` 和 `docs/knowledge/tts-performance.md` 的依赖来源规则。
2. **设计判断，已简化：调制张量的匿名元组及布局知识分散。**
   `embeddings::ModulatedInput` 用命名字段替代六元组和独立 Option/expect；
   最终 norm 的通道重排与打包由 normalization 模块负责，DiT 负责块间串接。
3. **设计判断，已简化：重复的 eager 回退。** BigVGAN 复用已有 up/snake helper；
   WaveNet 共用激活及 residual/skip 回退，删除用虚构 CPU 错误选择路径的方式。
4. **设计判断，已简化：OmniVoice 模型与诊断代码承担不同职责。**
   将 codes dump、同步探针移入私有 `stage0_model/diagnostics.rs`；canary 缓存
   仍由 `Stage0RuntimePlan` 持有，不引入跨设备全局张量缓存。

此外，将 Metal shader 文本移入各模型的 `.metal` 文件，Rust 文件负责资源、校验和调度；
依赖、模块和仅供 Metal 使用的缓存统一按 feature 与 macOS target 门控。
两个模型的融合不建立跨模型公共框架，CPU / Metal 不同的数值回退前提保持独立。
这些调整针对单一职责、依赖方向、明确接口及避免重复，不构成对整个仓库的形式化设计保证。

## Spec

1. **P1，已修复：CosyVoice3 非流式 padding mask 展开失败。** 原代码从
   `[B,1,1,T]` 的 query 轴取长度，再展开到 `[B,H,1,1]`；改为完整 `[B,H,T,T]`。
   新测试覆盖流式/非流式与两个 batch，检查被屏蔽 key。
2. **P2，已修复：IndexTTS im2col 忽略输入 slice offset。**
   原 `BufferOffset::zero_offset` 改为布局的字节 offset；新增连续 batch slice
   与独立 CPU Conv1d 对照，验证实际读取选中 batch。
3. **P2，已修复：非标准 DiT 几何误入固定融合。** CosyVoice3 的整个 boundary
   carry 链在创建前校验 F32、1024 维及连续 Metal 布局；不适用时走参考路径。
   两个模型的 attention 排除 SDPA 不支持的 head dimension。新增小模型 Metal
   测试曾揭示 head_dim=16 的 SDPA 错误，修复后通过。
4. **P2，已修复：raw float 内核未完整校验次要输入 dtype。**
   RoPE cos/sin、门控 residual/embedding、FIR phases/activation 等均显式检查 F32，
   避免将 F16 buffer 当作 float 读取。两套 RoPE 包装均有混合 dtype 拒绝测试。
   对向上取整的 RoPE dispatch 也限制其支持的通道整除条件。
5. **P2，已修复：OmniVoice SDPA 忽略 mask storage offset。** Candle 0.11 的
   mask binding 传递 buffer/stride 而未传 offset。仅对非零 offset mask 在 head
   展开前调用 `force_contiguous()`；零 offset 路径保持共享。`contiguous()` 和
   `copy()` 均不能保证重置这种视图的 layout offset，不能用其替代。
   新用例覆盖共享单头/完整 16 头 mask，已观察修复前失败、修复后通过。
6. **P2，已修复：IndexTTS ConvTranspose 短输入输出长度错误。** 原算法先对
   `(L-1)*stride - 2*padding` 做 saturating subtraction 再加 kernel，可能把
   合法短输出算长。改为先加 kernel 再减 padding，并以 checked arithmetic
   拒绝非法长度和 stride=0。新增单帧输入与独立卷积/裁剪数学结果对照（Candle 通用 CPU 路径在该边界自身下溢）。

前四项对应优化计划的“输出保持优先”及“位级或 ≤1e-5，附 CPU 锚”约束；
OmniVoice 掩码问题对应其融合记录的共享 mask 语义。普通模型形状的快路径
保持既有计算，不把错误边界修复声明为新的性能优化。

## 验证与边界

- CPU 普通测试：CosyVoice3 36、IndexTTS 108 通过。
- Metal feature 的普通测试：CosyVoice3 36、IndexTTS 108、OmniVoice 56 通过；
  真实权重测试仍为显式 ignored，不下载模型。
- 显式 Metal 数值测试：CosyVoice3 5、IndexTTS 9 通过，包括融合与参考链、
  slice offset、混合 dtype 拒绝、padding/small geometry 和 ConvTranspose 短输入。
  OmniVoice 的新增 mask offset 测试包含在上述 56 项中。
- 三个模型 CPU / Metal 的 all-targets Clippy（`-D warnings`）通过。
- 三个模型 Metal rustdoc（`-D warnings`、private items、examples）通过。
- `cargo fmt --all --check`、`git diff --check` 通过。
- 两位需求轴审查者独立复看各自原问题，确认对应修复成立；复审仅检查源码，
  测试结果由本地执行产生。

本轮没有运行真实模型音频、人工试听或 ABBA 性能基准，不更新历史 RTF 或官方排名。
Windows/CUDA 未实机复验。提交和推送另按用户指令执行；本轮未发布应用或库。

计数：Standards 4 项（1 项文档标准问题、3 项设计判断），Spec 6 项；均已处理。
标准轴最高 P2；需求轴最高 P1（padding mask）；两轴不合并排序。
