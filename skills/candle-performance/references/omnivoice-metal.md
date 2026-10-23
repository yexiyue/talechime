# OmniVoice Metal：从已验证实现提炼的经验

适用于排查 Candle Metal 注意力、布局与音频 decoder 的额外开销。这里是截至
2026-10-23 的案例；新模型或新 Candle revision 先重新核对内核前提，再做局部对照。
无需为使用这些经验加载模型或重跑全部历史实验。

## 先还原真实路径

案例环境为 M4 Pro / macOS 26.6.2，Candle fork
`2266099d92265112ecc0d56ecee1761e3cc2ac00`；OmniVoice 0.6B stage0 Metal F16，
decoder Metal F32。短文本真实 Q 为 `[2,16,157,128]`，K/V 为 `[2,8,157,128]`。
CFG 拼在 batch，不能将其误算成双倍序列长度；287 是额外合成测试长度。

确认构建包含模型自己的 Metal feature，再沿源码找到真实算子。启用 Metal 不会
自动融合手写 attention；Accelerate 加速 CPU BLAS，不会替换 Metal 内核；ORT/CoreML
也不会自动接管 Candle 模型。Python 通常调度 ATen 原生算子，语言本身不是归因。

## 已采用的改动与迁移条件

### 直接调用融合算子，单独验收数值变化

手写 QK → scale → mask → softmax → PV 和展开 RMSNorm 可以利用已有融合实现。
先检查固定版本的 API、dtype、mask、head dimension、stride 与实际 dispatch。
融合可能改变舍入和离散采样轨迹；初期 SDPA/RMSNorm 改动确实改变了 codes，
不能与后来的逐位保持优化混写为同一正确性结论，也不能由此推断所有优化都会改输出。
这类改动遵循主 skill 的数值授权规则，不能因速度好就自动扩大精度容差。

### 原生 GQA：不要替内核预先重复 K/V

固定 Candle SDPA 直接按 Q/KV 头比例映射，原输入 8 KV 头无需复制成 16 头。
检查的是头数整除关系；源码中名为 `q_head`/`k_head` 的变量可能指最后一维的
head dimension，不能只凭变量名推断头数限制。检查 full-SDPA 分支，不能只读 vector 分支。

同探针比较原生 GQA 与 repeat_kv，包含复制、分配和真实 mask/V 布局；再核完整
生成输出。CPU 等其他设备保留原路径。本案例不等于“照搬官方”：固定 transformers
在有 mask 时仍重复 K/V，Candle Metal 与 PyTorch MPS 的路由不同。

### 顺着消费者布局消除复制，而不是逐个删除 contiguous

投影自然产生 `[B,S,H,D]`。在此连续布局做 RMSNorm 和 `rope_thd`，随后仅 transpose
成 `[B,H,S,D]` 视图交给 SDPA，避免先复制成头主序再执行 `rope`。
确认旋转约定、cos/sin 序列索引一致，比较完整 Q/K 与有/无 mask 的 attention 输出。
SDPA 的共享 mask 可通过 expand 的零 stride 广播，V 也可以保持转置视图。

消费者并非都支持任意 stride。固定版本 Metal gather 曾对窄词表视图读错行跨度；
argmax 读视图通过不能证明 gather 也支持它。分别检查索引布局与源布局，不能将
“算子接受 Tensor”当成 stride 正确的证明。最终 context 的复制也不能凭直觉省掉。

### 转置卷积：先查是否错过现有快路径

固定 Candle 的非零 padding 会错过 GEMM + col2im。对于已验证的 dense Metal F32、
连续输入/权重、groups=dilation=1、padding>0 且 output_padding<=padding：
先以 padding=output_padding=0 计算完整结果，再裁左侧 padding、右侧
`padding-output_padding`，最后加 bias。检查裁剪长度并测试短输入与奇偶 stride。

该变换数学等价，但实现路径改变不保证逐位相同。本轮原始 GPU 解码对照报告过
波形误差，并经用户授权保留；不能把其 SNR 写成逐位一致或试听证明。
CPU/CUDA、量化和不合格配置不沿用本例的性能或数值结论。

### 1×1 卷积：输出布局也是算法成本

固定 conv1d 走 im2col → GEMM → 输出布局复制。decoder 私有入口可对 dense Metal
F32、连续输入/权重、kernel=stride=dilation=groups=1、padding=0、非空输入尝试
`W[Co,Ci] × X[B,Ci,T]`，直接得到连续 NCL，再加 bias。
保留通用 forward 与参考编码路径；其余配置回退原卷积。

同时测 GEMM 与必要的布局物化、bias、后续 Snake/残差消费。保持原 GEMM 方向但
来回转换 BTC 的候选反而更慢。完整固定 codes 解码是比组件等权均值更强的验收；
7-tap 需要真正的窗口展开，不能把 1×1 方案直接推广过去。

### 贪心 confidence：复用索引，保留随机采样语义

默认 class_temperature=0 时，已有 argmax 索引可从 log_probs gather 最大值，
省掉第二次全词表 max 归约。使用 U32 索引 gather 后再转换 token 类型。
随机采样仍需要全局最大置信度，不能改成采样 token 的概率。
覆盖同分、mask、实际词表宽度与不同布局，比较完整 tokens/confidence 位模式。
不要为数学简化代数消去 CFG 中的 log_softmax，舍入改变会影响迭代轨迹。

## 不应重复当成已成功优化的候选

| 候选 | 实验结果与后续判断 |
| --- | --- |
| Snake 加载时缓存 `(alpha+1e-9).recip()` | 输出一致，但各块快慢混合，固定 codes 变化与漂移同量级；撤回缓存状态。少几个算子不保证关键路径缩短 |
| 先 contiguous 再 cast，或整批 cast 后切 logits | 含后续 scoring 仍分别慢约 2.3% / 4.3%；保留原切片后转型，拒绝的只是这两个候选 |
| 通用 unfold + GEMM 实现 7-tap | 两种布局输出一致但更慢；不能把预先准备好的窗口计时当成原生 im2col 成本 |
| 原生 im2col + GEMM 直接输出 NCL | 固定 codes decoder 局部约快 4.94%，短/长输出一致；仅离线 Candle 补丁，未改生产依赖，整链遇外部编译未确认提速 |
| 自定义 NCL im2col | 完整 decoder 慢约 9.77%，撤回 shader；省输出复制不代表总访存更省 |
| 用有效词表窄视图替代 mask 拼接 | 修正 gather 布局问题后仍无可重复收益，撤回生产候选 |
| 深层 K/V 缓存或合并投影 | 双向注意力中输入前缀不变不证明深层 K/V 不变；合成投影等价不代替真实生成验收，不能列为已采用成果 |

## 怎样报告收益，何时停止

固定版本、同 feature 的 release 二进制先冻结。局部探针复用真实权重与实际形状，
区分合成激活和捕获激活；在进程内交错原/候选，并重复独立进程。
固定 codes 解码与短文本整链再配对；正确性另外比较短/长完整 codes、样本数及
所有分段/整段 F32、PCM16 WAV，不能只比最终 PCM16 或使用相同 seed。
profile 边界同步只用于归因，不将其相加冒充生产吞吐，也不称为 GPU timestamp。

按进程 basename 监控外部 cargo、rustc、clang、ld、clippy-driver 等；历史上错误的
路径匹配曾漏报编译。受干扰数据保留并标记，没有合适窗口就不报整链提速。
局部显著收益可以独立保留，不设整链最低百分比；组合后重新测，不直接累加比例。

1×1 案例固定 codes decoder 176.20→161.85 ms（-8.15%），整链观测仅 -0.37%，
后者小于波动，不能宣称整链收益。随后免拷贝注意力的同期三路 ABCCBA 才测得
Rust 3.02383→2.67203 s（-11.63%），官方 MPS F16 为 3.23450 s。
这些是不同基线；不能把它们和初期融合收益相加。

官方对照须重新核版本、实际 dtype/device、采样、参考准备、warmup 与输出长度。
上述 2.67203 s 在该短文本生成配置下比官方少 17.39%，已经满足该范围的性能目标；
不证明长文/参考音色/CPU 也达标，更不证明跨框架音质等价或 Metal 已到硬件极限。
达到用户指定范围的目标后可停止继续抠小收益；不要把这个停止条件强加给其他任务。

## 源码与原始记录

仓库中查看 `crates/omnivoice/src/stage0_qwen3.rs`、`stage0_model.rs`、`codec_layers.rs`。
现有离线探针包括 `gqa_probe`、`attention_copyfree_probe`、`scoring_probe`、
`decoder_ops_probe`、`stage1_probe`、`conv7_layout_probe`；普通测试不下载模型。
移出本仓库后，下列链接可能不可用；上面的流程不依赖历史产物才能执行。

- [初期融合与数值变化](../../../docs/records/omnivoice/omnivoice-optimization-2026-10-22.md)
- [转置卷积与 decoder 差距](../../../docs/records/omnivoice/omnivoice-backend-experiments-2026-10-23.md)
- [原生 GQA](../../../docs/records/omnivoice/omnivoice-native-gqa-2026-10-23.md)
- [贪心 gather 与转型候选](../../../docs/records/omnivoice/omnivoice-scoring-2026-10-23.md)
- [Snake 与 1×1 卷积](../../../docs/records/omnivoice/omnivoice-decoder-pointwise-2026-10-23.md)
- [免拷贝最终审计与同期官方比较](../../../docs/records/omnivoice/omnivoice-official-comparison-audit-2026-10-23.md)
- [7-tap 离线候选与失败实验](../../../docs/records/omnivoice/omnivoice-native-conv7-2026-10-23.md)
