# Windows 性能探针 WAV 导出修复

本项是测量工具开销优化，不是TTS推理提速。Windows本机YEXIYUE，Ryzen 7 9700X，
项目位于D盘；RTX5070/驱动616.56是本目标设备，但导出对照没有调用GPU或加载模型。
源码HEAD `42a7887e28f095684266eac43a68bd8fcc303d11` 加当前工作区修改，精确源码、
Cargo.lock及不可变测试二进制摘要见外部 `export-build.json`。

## 问题、边界与实现

Nano长文本测量中，热合成约3秒，但逐轮事件间隔远大于此值。复核
`performance_support/probe.rs::round`：`synthesis_seconds`在所有序列化/摘要之前固定；
WAV导出使用 `WavWriter::new(File)`，逐样本直接写入。不能把未计时导出算成模型耗时，
也不能只凭间歇就认定其造成GPU频率漂移。

`write_wav`改为64KiB BufWriter；编码循环提取为私有泛型 `write_wav_to`，保持F32与
PCM16转换、采样率/声道、有限值检查、create_new禁止覆盖，以及显式finalize。
未修改模型算法、计时边界、JSON事件版本或播放完成语义。
检查固定Hound3.5.1源码确认finalize更新头后显式flush；不能依赖Drop吞掉I/O错误。

## 导出对照

输入是此前冻结Nano基线的完整16.08秒、48kHz双声道F32 WAV，1543680个样本：
`paragraph-native-run/attempts/1792825401878173700-19488/round-1-full-f32.wav`。
先在计时外读入并验证SHA，分别用同一编码循环的File直写与64KiB缓冲写入。
三个独立进程，各1预热+5热轮；每轮包含两个格式，两种写入顺序按轮交替。
创建与写入/显式finalize/关闭均计时；事后的读回、摘要与逐字节比较不计时。
直写分支使用create_new，缓冲分支还保留已有路径预检；后者额外的元数据查询也计时。
这是Windows文件缓存下的host导出时间，不是sync_all持久化吞吐或GPU时间。

| 格式 | 进程 | 直写热均值 s | 缓冲热均值 s |
| --- | ---: | ---: | ---: |
| F32 | 0 | 2.93256252 | 0.00509820 |
| F32 | 1 | 2.93113406 | 0.00329668 |
| F32 | 2 | 2.91775122 | 0.00315472 |
| PCM16 | 0 | 2.92433292 | 0.00399896 |
| PCM16 | 1 | 2.91921752 | 0.00397888 |
| PCM16 | 2 | 2.90437844 | 0.00399416 |

F32组均值2.92714927→0.00384987s，PCM16为2.91597629→0.00399067s。
所有36组格式轮次的直写/缓冲输出逐字节相同；进一步对照修改前的原始完整F32和
PCM16文件，两种输出均保持原摘要，而不只是在新实现两种writer之间自洽：

- F32：6174788 bytes，`a53e59913922905bfeb60d69b01901dfa9b1bd19580b8cb6b6410829c3b377d5`。
- PCM16：3087404 bytes，`ec0ffb2fdd5aebd04cb47e37d394fe489e7e6429c76d8df7fb959a9e02ee1e2e`。

目录 `D:/workspace/talechime-benchmarks/windows-nano-20261024` 保留：
`export-build.json/jsonl/log`、`export-ordinary-tests.log`、`export-summary.json`，
以及 `export-cost-process-[0..2]/results.json`、逐轮全部WAV和同名前缀日志。
结果kind固定为 `artifact_export_cost_not_inference`，不作为任何模型排名输入。

## 错误与集成门禁

新增测试在 `performance_support/wav_tests.rs`：缓冲与内存writer的完整字节对照、
已有文件不覆盖、注入底层write失败和flush失败、非有限PCM拒绝。连同已有有限流、
取消/预算、分段范围、计时后导出、遥测有界性等探针测试，8项通过（2.39s）。
不下载模型；慢速导出对照默认ignore，必须显式指定本地输入和新输出目录。

Release Clippy `--example performance_probe -D warnings`（7.30s）、私有项rustdoc
`-D warnings`（4.85s）通过。Rust xtask重新冻结CUDA Nano/performance-tools组合探针，
49.56s编译通过，原有LNK4098警告仍记录；二进制SHA256：
`20d2ec2d28d710c9414eb1c8b77d6e449e823d89b8775d6923113405cfb89394`。
使用该探针生成xtask fixture-job并运行3×(1+5)，18轮成功、三个进程清理通过。
夹具显式标记synthetic；CUDA编译和CPU夹具成功不代表本轮运行了GPU模型。
原始收据、Job/Plan、results及HTML分别为 `buffered-production-build.json`、
`buffered-fixture-input`、`buffered-fixture-run`、`buffered-fixture-report`。

## 恢复和后续比较

使用export-build.json中冻结测试exe，普通检查：
`--skip real_audio_export_cost_and_byte_equivalence --test-threads=1`。
慢速对照先设置TALECHIME_PERF_EXPORT_INPUT为上述WAV、
TALECHIME_PERF_EXPORT_INPUT_SHA256为其固定摘要、TALECHIME_PERF_EXPORT_OUTPUT为新目录，
再执行 `real_audio_export_cost_and_byte_equivalence --ignored --nocapture --test-threads=1`。

已存在的Nano模型优化ABBA两臂都使用原直写探针，保留其原始身份，不回填新收据。
缓冲改变未计时的轮间空档，可能改变设备热态；本轮没有证明频率因果关系。
后续模型候选应使用相同导出策略重建并冻结两臂，不把旧直写A与新缓冲B之间的
模型耗时差异直接归因推理算法。其他平台的导出性能未实测。
