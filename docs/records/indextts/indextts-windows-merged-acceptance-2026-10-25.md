# IndexTTS 合并后验收与回退修复（Windows，2026-10-25）

本轮验收main `1746d7232ab79367f1a8b9411a7f8e917a0a4fe5`加本地Windows优化，
Candle `884b359895a16c8fbd3647789b1f7c7974894708`，设备为Ryzen7 9700X /
RTX5070、CUDA13/cuDNN9.14、驱动616.56。既有参考缓存保留，不新增提速候选。
外部证据根 `D:/workspace/talechime-benchmarks/windows-acceptance-20261025`。

## 合并后发现的两项问题

首先复用已冻结的Qwen/Index CUDA探针`f448f1cfeeb310f446a4f60ad53948de86a8dc8245c489271c96636ae550f0ac`。
初批18轮合成都成功、进程exit0且清理通过，但历史Job未包含main新增的情绪参数
默认值，preflight_identity_verified=false，工具正确拒绝提交成功。保留完整
`indextts-run`，从实际prepared能力目录生成新的预期参数清单，未更改请求或旧结果。
新`indextts-v2-run`三进程通过身份门禁，却揭示了输出回退。

main将DiT的token-major显式乘加RoPE换成head-major `rope_i`，通用fallback也作用于
Windows CUDA。相同轮次的长度保持，但PCM改变；首轮57000样本中56998个F32值不同，
max_abs约0.007199、RMSE约0.0002242、SNR约50.12dB。这里的SNR只是数值差异，
不是听感评分。此前24份BigVGAN组件逐位相同，不能覆盖完整DiT/CFM路径。
未将此差异视为已获准的容差，未把v2的1.39145秒计为可接受提速。

修复只让Windows CUDA恢复原有RoPE布局与分开舍入的乘加链。Mac fused路径及其他
设备的main路径保留；没有修改采样、模型精度、seed或CFM随机策略。固定构建后，
新18轮全部与合并前对应轮次逐位一致，证明该回退检查覆盖了实际整请求。

另扩展现有ignored真模型测试，将情绪与参考缓存联合执行。初版在named-happy
的首次uncached请求报`shape mismatch in mul, lhs: [], rhs: [1, 1280]`。
模型情绪向量混合的标量Tensor乘法未广播，改成显式`broadcast_mul`；保留原标量
构造/dtype和乘后加法，不改成融合affine。此修复属于共享模型正确性，但本轮实机
仅验证Windows CUDA，不能宣称Mac实机已测。

## 修复后的生产请求

新probe SHA `c02f8a7bfd05e4298835d5b43eb1e8c2f35807196fbf1a8b431234fb719b5383`，
收据`indextts-fixed-build.json`固定源码/dirty身份与源码guard结果。features为
indextts、indextts-cuda-cudnn、performance-tools、qwen、qwen-cuda；复用相同依赖
缓存，额外编译Qwen不表示本轮测量Qwen。测试和GPU计时串行，无并行编译。

固定原中文短句、custom:acceptance、Pinned42、Rayon8、F32、无style/continuation，
原转换权重及参考身份不变。三独立进程，各1预热+5热轮，共18轮：

| 项目 | 实测 |
| --- | --- |
| 三进程热均值s | 1.546826 / 1.536485 / 1.522001 |
| 均值s / 首PCM均值s | 1.535104 / 1.535042 |
| 音频时长s | 各轮2.585034或2.592154，与对应旧轮一致 |
| RTF均值 | 0.592534 |
| contained Job提交高水位 | 6.759–6.761 GiB，非RSS或独占显存 |

三进程均success、exit0、cleanup_verified；diagnostics flush_complete，无raw/trace
丢弃。baseline模式未开启trace，trace_complete=false不冒充采集了完整trace。
18轮输出摘要一致对应旧轮，72份完整/分段F32及PCM16 WAV的SHA/长度与旧控制相同；
另18份ranges JSON核对自身摘要、稳定元数据、逐块样本/source范围及完整覆盖。
CFM生产RNG不随Options.seed逐轮重置，因此按对应process/round比对，不能把首轮
作为所有轮次的音频oracle，也不能要求这18轮互相相等。

历史buffered-cache三进程热均值1.505700 / 1.554374 / 1.513148秒，均值1.524407秒。
新旧范围重叠，本轮没有明显持续回退线索。跨日期且构建features不同，不能视为
同期严格A/B或新的官方排名；既有缓存的收益与官方精度差异沿历史记录保持原口径。

## 情绪与缓存联合门禁

扩展`indextts::runtime::tests::reference_cache_matches_uncached_pcm`，仍显式设置
本地权重和参考，不在普通测试下载模型。临时store导入原参考A与测试拥有的逆序B。
每个控制重置测试拥有的CUDA随机状态20261024，host sampling seed保持原值；
这个额外重置仅在测试里，不更改生产随机策略。

先回归uncached/首次/命中、同路径换内容B及返回A、预关闭、非法WAV、生成中取消
后恢复。再对下列四种条件分别比较关闭缓存、缓存命中及重复命中的完整PCM位模式，
每组之后普通请求恢复60672样本原PCM；所有完整请求均有限PCM及正常End。

| 条件 | 每次完整原始PCM样本 |
| --- | ---: |
| named happy | 58112 |
| happy/calm混合向量，alpha0.6 | 57088 |
| 同向量、固定seed随机锚点 | 58880 |
| 独立逆序情绪参考，alpha0.6 | 59648 |

四者均实际区别于普通请求，未把解析参数成功当作模型已生效。联合门禁69.72秒通过，
这些是直接owner输出、宿主边界处理前的样本，不与生产WAV时长混算性能。
条件请求本轮验证完整重复/返回普通状态；取消分支仍是原普通请求子集，不能据此
宣称所有情绪方式的取消或情绪质量已独立验收。

新测试二进制固定在`indextts-fixed/indextts-test-binaries`：backends SHA
`2b82371a1a220dd47b59561e7eccfacac4d0771d93ae112c9a7126a2eefea133`，模型SHA
`5c32819554aa23f72ac39b0cf272a2b22e1df937c17c00b0d446ea665670ceae`。
普通测试backends30通过/6忽略、IndexTTS108通过/23忽略，共138通过。
同组合两crate release all-targets Clippy和rustdoc均以-D warnings通过；
fmt、git diff --check通过。原生链接的既有LNK4098没有屏蔽或宣称解决。

## 证据与未验范围

- `indextts-initial-preflight-rejection.json`保存新旧参数目录差异；失败attempt不改写。
- `indextts-v2-summary.json`明确output_gate_passed=false；
  `indextts-merged-rope-output-difference.json`保存六轮数值差异，旧代码副本另存。
- `indextts-fixed-{build,job,plan,summary}.json`与run固定修复后的结果；
  `summarize_indextts_acceptance.py`按确切attempt/round审计90资产。
- 生产基准仍使用原静态参考；联合测试的逆序参考仅在临时目录创建，
  不将合成夹具当作自然说话人或情绪质量证据。
- `indextts-fixed/indextts-new-checks-status.json`、测试receipts、普通/真模型/
  Clippy/rustdoc日志保存独立退出状态，`run_indextts_fixed_checks.py`保留串行命令。
  首次情绪失败日志保持在外层`indextts-emotion-cache-real.log`。

人工试听、长文、主CLI交付及官方F32完整常驻对照仍未由本轮关闭。
不重试此前已确认资源不足的官方F32整模型，不放宽资源摘要门禁。
