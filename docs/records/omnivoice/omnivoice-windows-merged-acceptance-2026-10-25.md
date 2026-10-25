# OmniVoice 合并后验收（Windows，2026-10-25）

本轮只验收既有Windows优化与main合并，不增加性能候选。main
`1746d7232ab79367f1a8b9411a7f8e917a0a4fe5`加本地Windows优化，Candle统一
`884b359895a16c8fbd3647789b1f7c7974894708`；Ryzen7 9700X / RTX5070，CUDA13、
驱动616.56。相对旧main，OmniVoice变化为诊断代码整理及Metal mask偏移处理，
仍通过实际Windows请求与CUDA门禁验证，不凭源码范围直接判定通过。

冻结xtask SHA `08cafd5a641a321ad0179cf3d7da13a63e3f022f19c9f3e25926d288713252c1`
执行perf build，显式features为omnivoice-cuda，自动加performance-tools；
实际Cargo收据还包含隐式omnivoice。复用原请求对应的D盘target缓存，不创建C盘链接。
新probe SHA `0cff5746400f8ead92e3a95c2aaa8f8ecd95895294c5507fa8e9cf65ae953615`，
`omni-build.json`固定源码dirty身份且guard通过，构建5m54s。测试先预编译，
三套正式基准和CUDA门禁串行执行，不与编译并行。

## 完整请求与输出回退

沿用[既有保序decoder记录](omnivoice-windows-cuda-2026-10-24.md)的short/long
ordered-current及clone-1-ordered Job，保持语料、资源、参考、seed42、Rayon8、
32步、CFG2、stage0 F16/decoder F32和原采集配置。各三独立进程、1预热+5热轮，
共54轮；新Job只更新source/build、case身份及进程矩阵，不缩短输入或修改采样。

| case | 三进程热均值s | 均值s | 首PCM均值ms | 音频s | RTF均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| short narrator | 0.305498 / 0.292258 / 0.297739 | 0.298499 | 298.45 | 1.63 | 0.183128 |
| long narrator | 1.889315 / 1.884465 / 1.886792 | 1.886857 | 461.39 | 16.4 | 0.115052 |
| short reference clone | 0.749064 / 0.742504 / 0.751306 | 0.747625 | 747.57 | 2.0 | 0.373812 |

9个进程全部success、exit0、cleanup_verified；flush_complete，无raw/trace丢弃。
短句与clone首PCM接近整段完成；long按原三个文本段交付，不能称作原生实时音频流。
contained Job提交高水位分别2.581–2.613、2.996–3.027、2.579–2.611 GiB，
不是RSS或独占显存。旧对应均值0.318765、2.175120、0.782940秒；跨日期结果较低，
本轮未发现明显回退线索，但不是同期交错A/B，不把差值记为新的优化收益或官方排名。

全部54轮输出摘要和时长与各自旧控制一致。288份完整/分段F32及PCM16 WAV的
SHA/长度相同；另54份ranges JSON按自身SHA审计，稳定元数据、文本范围、
分包样本区间一致且连续覆盖完整音频，保留consumer及owner时序原值。
总计342资产；long按实际三个segment枚举，旧文件由对应attempt_id定位。

## CUDA与普通测试

新组合普通测试backends11通过、模型42通过/26忽略，共53通过。四项显式CUDA门禁
各用独立进程运行，确认每个过滤器实际执行一项测试，均成功：

- `request_graph_matches_eager_and_recovers_after_cancel`：两种文本形状eager与生产
  图完整PCM逐位相等，A/B/A、录图后取消、取消后恢复通过；实际graph_records=5，
  不以fallback冒充图回放。4.11秒。
- `production_ordered_decoder_keeps_raw_bits`：固定既有48/92/222帧、8个codebook
  codes；外部先核对文件数量、形状、合法code值和SHA，避免空目录造成零次比较。
  每形状原/逐code+1模1024/原三次，生产保序decoder与eager完整raw PCM位一致。
  0.86秒。
- `failed_capture_exits_stream_and_valid_graph_replays_again`：故意录制失败后退出
  capturing状态，合法图能重新录制，拒绝错误输入形状并保持A/B/A输出。0.09秒。
- `strided_output_copy_is_warmed_before_capture`：非连续输出复制在录制前预热，
  多次输入变化的完整F32回放保持。0.08秒。

模型测试SHA `bd9e9b962af29c16c4eb00b0f2f2adde289aaa4c08c776c55901dd7fc3869462`，
backends测试SHA `73c35e7c90ac594301377045abd51e9b2ac3843454f14df74e590897d85e5a9b`。
本轮没有人工试听、情绪/style/长链续接质量验收，也未新跑官方Python矩阵。

## 磁盘阻塞与保留证据

补充静态交付检查：新旧已冻结probe的直接PE import表完全相同，仍含
VCRUNTIME140/UCRT、cuBLAS13、curand10及系统库，没有新增直接NVRTC import。
这不是已加载模块或全部传递/延迟依赖枚举，也不能据此消除既有LNK4098问题。
`cargo package --list --allow-dirty --no-verify --offline --locked -p talechime-omnivoice`
退出0，包清单包含build.rs、ordered_deconv.cu、CUDA wrapper、SOURCE、项目/
Higgs及Candle Apache/MIT许可。dist仍只发布talechime，msvc-crt-static=false，
没有把CUDA或ort-dynamic自动加入标准发行features。此处仅核对源码包和配置，
没有打包或发布安装器，不能据此宣称干净机器安装通过。

随后release all-targets Clippy在windows-sys写full.rmeta时因磁盘空间不足
（os error112）退出101；不是通过，也不能据此判定代码lint失败。rustdoc尚未执行。
fmt/diff检查已通过；最终交付仍未完成，不以其他feature组合旧门禁代替。
释放空间后只续跑未完成的Clippy/rustdoc并使用新日志名，不重新执行整个脚本覆盖
已有日志，也不重复已成功的54轮和CUDA门禁。后续空间恢复后，两项均-D warnings
通过，见`omni-space-retry-status.json`与`omni-{clippy,rustdoc}-space-retry.log`；
初次os error112失败保留，不改写原状态。当前组合工程检查已补齐，最终交付仍未完成。

用户授权检查workspace可清理缓存。盘点确认CastGlean target约16.10 GiB，其中
`D:/workspace/castglean/target/debug/incremental`约2.24 GiB、
`D:/workspace/castglean/target/audiobook-test/debug/incremental`约1.84 GiB可再生；
两目录无reparse点，文件仅bin/lock/o/rmeta。对这两个明确目录执行原生PowerShell
清理时，自动审批再次以blocked by policy拒绝，命令未启动、未删除文件。
已请求用户手动释放空间；不改用其他删除方式绕过，也不删除模型或验收资产。
用户随后明确回复“帮我删除”，按原生PowerShell校验流程重试一次，仍在进程启动前
被同一policy拒绝，两个目录未删除；独立拒绝记录保留。
随后外部清理使两个目录消失，D盘余量恢复；本线程没有成功执行缓存删除，
不将外部清理归为工具执行成功。

外部证据根 `D:/workspace/talechime-benchmarks/windows-acceptance-20261025`：
`omni-build.json`、`omni-tests-receipts.json`固定二进制；三个case的Job/plan/run、
`omni-acceptance-summary.json`与`summarize_omni_acceptance.py`保留54轮和342资产审计；
`omni-codes-fixtures.json`固定非空codes输入；`run_omni_acceptance_checks.py`及
`omni-checks-status.json`保存串行命令与真实退出状态。四项CUDA日志分别独立保存。
空间盘点与删除拒绝记录见workspace-cache-inventory-20261025.json、
castglean-incremental-candidates.json及workspace-cleanup-rejection.json。
后续直接PE/包许可检查见omni-merged-pe-imports.json、omni-package-list.txt、
omni-package-license-audit.json；具体授权重试见workspace-cleanup-explicit-retry-rejection.json。

```powershell
$r = 'D:/workspace/talechime-benchmarks/windows-acceptance-20261025'
$tool = Get-Content 'D:/workspace/talechime-benchmarks/windows-indextts-20261024/component-diagnostic-final-tool.json' -Raw | ConvertFrom-Json
# 先核对tool/build SHA与CUDA DLL环境，保持Job身份，输出使用新目录。
& $tool.path perf run --job "$r/omni-short-job.json" --output "$r/omni-short-replay-NEW"
# long、clone同理串行执行；现有成功批次无需因Clippy空间失败无条件重跑。
```
