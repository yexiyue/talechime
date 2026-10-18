# IndexTTS 官方、社区 Rust 与本地实现对照

日期：2026-10-17。范围：独立 `crates/indextts` 的 IndexTTS 2.5 计算路径和开发探针；不新增生产适配器。

## 来源与实现边界

| 实现 | 核验来源 | 结论 |
| --- | --- | --- |
| 官方 Python | [index-tts](https://github.com/index-tts/index-tts/tree/d9e41aac89fd00b3d71497fddb287b7f24613712)，本机 checkout 同 revision；本次查询远端 HEAD 也相同 | 2.5 真实推理基线，使用 eager、F32、默认 beam-sample，不启用 accel / 自定义 CUDA kernel |
| Talechime Candle | `crates/indextts`，权重版本见 `SOURCE.md` | 真正的 w2v-BERT → CAMPPlus / GPT → EnhancedCodec → S2Mel CFM → BigVGAN 路径；无需 Python 运行；仍是独立计算 crate |
| 社区 Rust | [8b-is/IndexTTS-Rust](https://github.com/8b-is/IndexTTS-Rust/tree/621b39822bb4a86923f189280fd018b56fc16932)，本机检出于 `target/indextts-review/community/` | ORT 实验框架；`src/pipeline/synthesis.rs::generate_mel_spectrogram` 忽略 token / speaker / emotion 输入，用随机数和正弦构造 Mel；`generate_with_cache` 直接调用无 cache 版本。不能作为真实 2.5 音质或性能基线 |
| 社区 HF 发布 | [ThreadAbort/IndexTTS-Rust](https://huggingface.co/ThreadAbort/IndexTTS-Rust)，核验 README / Cargo 源码 | 同名 ORT 路线；README 仍列 full GPT integration / KV cache 为未完成。没有据此运行或宣称获得 2.5 对照结果 |

社区代码在本机核验源码，但未运行随机占位的音频流水线，也未为它转换或下载模型。实机 A/B 由官方 Python 与本地 Candle 完成。

官方支持的多语言前端、可选 QwenEmotion 文本情绪 LLM、流式返回及加速引擎不等于本 crate 已支持：当前 Rust 前端只覆盖 zh/en 子集，支持参考音色、独立情绪参考 / 情绪向量及 duration_factor，未移植上述扩展。CPU 有数值验收，CUDA 有本机音频验证，Metal 未验证。

另纠正 `SOURCE.md` 的来源描述：固定官方 revision 的 `LICENSE` 是 [bilibili Model Use License Agreement](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE)，其 Model 定义包括发布的代码与权重；不能写成官方代码是 MIT。本次不改自有代码的许可证声明，也不重新分发模型。

## 本次修复与简化

1. **采样候选不足误判完成**：原先对概率做不放回采样，用 `total -= selected_weight` 更新剩余质量。分布集中时，小概率被舍入或下溢吞掉，候选数少于 `2 * num_beams`；scorer 又把不足的非 EOS 候选当成正常完成。现在每次抽取都从剩余 log scores 重新归一化，并显式检查数量；scorer 对不足候选报错，与官方 scorer 的失败语义一致。增加集中概率分布、未完成不得伪装完成的回归测试。
2. **移除随机重抽补丁**：删除不足 8 码时换系统随机种子重抽的逻辑；它掩盖采样错误、破坏固定 seed，且最终仍允许空码进入 CUDA。当前空码在 codec 前返回明确错误，短的非空输出不按任意阈值重抽。
3. **补足 oracle**：原 `gpt_conditioning_and_first_step_logits_match_python` 实际止于 embeds，未检查 logits。现在执行 GPT / final_norm / mel_head，并按原阈值比较 8194 个首步 logits，最大绝对误差 **1.383e-5**。
4. **帧数与工具口径**：Mel 目标帧数按官方 `int(...)` 截断，替换四舍五入；两种报告的 RMS 统一到归一化 PCM；Rust 读取语料不再用 `map_while(Result::ok)` 吞掉读取错误；报告写入尝试数、失败数、失败耗时、逐段文本和 seed。
5. **收简探针**：WAV 写入和设备选择共用 `examples/support/probe.rs`；显式 CUDA 使用 `new_cuda`，不可用时失败；Python 增加 `--device`，两边增加 `--no-text-normalization` 以支持固定规范化文本对照。修复 rustdoc 的私有项目链接和重复注释。
6. **修复分层测试自身**：启用全部真实夹具后，length regulator 的逐层诊断暴露了测试中的多余 transpose（把 Conv1d 的 BCT 输出按 BTC 比较后又转回 BCT，导致 mask 形状错误），以及最大绝对误差漏 `.abs()`。测试已纠正，未改生产 regulator 或放宽容差。
7. **前端、参考截断、情绪与加载收简**：`front.rs` 的拼音声调与人名占位替换统一改为按字节长度降序去重（共用 `dedup_longest_first`），消除短串先替换破坏长串的隐患，并用 `OnceLock` 与预编译不区分大小写正则减少重复编译；`Reference::from_pcm`、`prepare_reference` 和 `prepare_emo_reference` 对齐上游 `_load_and_cut_audio(..., 15)` 增加 15 秒上限截断；补齐 `normalize_emo_vec` 辅助函数、`emo_vector` 按 `emo_alpha` 缩放（4 位小数截断）、`emo_reference` 特征向量缓存与 `clear_emo_reference` 清理入口，并将 `emotion_vector_matrix` 的余弦相似度由逐行标量同步简化为按组张量运算；`Weights::load_dir` 改为单次加载 `feats.safetensors` 并移除未使用的重复 `w2v_bert` 映射。

## 实机对照

Windows、RTX 5070 12 GB；CUDA 13.0 / cuDNN 9.14；Candle 0.11.0；Python torch 2.8.0+cu128 / transformers 4.52.1。官方 checkout 和模型已经在本机，运行设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。权重由相同官方 checkpoint 转换到 F32 safetensors。

语料是仓库 `tools/tts/acceptance-corpus.txt` 的 **6 个非空行**，两轮 **12 次**；不是历史记录写的 11 段 / 22 次。两端使用 `examples/voice_01.wav`、seed 基值 7，逐段 seed 为 `7 + round * 1000 + segment`。两个 GPU 进程串行运行；计时含各实现正常调用路径，官方首段含参考准备，Rust 参考准备在循环外。CPU 检查 / 编译有后台运行，因此此表是本机开发验收，不是独占机器的性能排名。

修复前重新构建并运行一次完整语料：**5/6 成功**，日期段重抽后仍 `CUDA_ERROR_INVALID_VALUE`。只修采样及重抽路径后的隔离轮次：**12/12 成功**，日期段两轮各 **6.908 s**；该次统计在 `target/indextts-review/fixed-first/index.json`。最终包含帧数和探针口径修正的结果，以 `target/indextts-review/rust-cuda/index.json` 与官方 `python-cuda/index.json` 为准。

| 最终版本 | 成功 / 尝试 | 音频总时长 | 合成调用总耗时 | 整体 RTF |
| --- | --- | --- | --- | --- |
| 官方 Python CUDA | 12/12 | 96.891 s | 43.579 s | 0.450 |
| Rust Candle CUDA | 12/12 | 98.562 s | 69.031 s | 0.700 |

RTF 越小越快；这次官方更快。日期段第 0 轮两端均为 6.908 s，第 1 轮 Rust 6.908 s / 官方 6.699 s。两轮同文本结果接近不代表声学输出逐位一致。

补充隔离：将日期句固定为“今天是二零二六年十月七日,车票售价一百二十八元,发车时间是下午三点.”，两端均 `--no-text-normalization --seed 9`，各成功一次，均为 6.908 s；产物在 `normalized-rust/` / `normalized-python/`。该单句官方首调用包含参考准备，不与前述两轮整体 RTF 混算。

原始证据目录 `target/indextts-review/` 包括 `before/`、`fixed-first/`、`rust-cuda/`、`python-cuda/` 的 WAV / index.json，以及汇总 `comparison.json`。语料 SHA-256 为 `af0b4d9b69c9e53bb4b1ae2e2e1283a952ae2de97427d0cbe52200ffe4b3f2c2`；参考 WAV SHA-256 为 `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`。

本轮检查：94 个无模型单测及两个 example 测试目标通过；Clippy `--all-targets -- -D warnings`、全仓 `cargo fmt --all --check`、rustdoc `RUSTDOCFLAGS=-D warnings ... --document-private-items --examples` 通过。CUDA release 两个探针构建成功（链接警告见下）。真权重完整检查在修复分层测试后以 release 模式 **16/16 通过，0 ignored，24.98 s**；日志 `oracle-release-tests-final.log`。其中包含 smoke / 形状测试和数值 oracle，不能把全部 16 项都视为独立 Python 数值对照。debug 全量检查在 CFM 等 CPU 路径耗时较长，已中断，不能把该次中断记作通过；首次 release 的 15/16 失败日志 `oracle-release-tests.log` 保留，失败原因见测试修复条目。

## 复现

以下 PowerShell 命令在仓库根目录执行；原始音频和报告均留在 Git 忽略目录。

```powershell
$env:CUDNN_LIB_PATH = 'D:\dev-tools\cudnn\v9.14-cuda13\lib\x64'
$env:LIB = $env:CUDNN_LIB_PATH + ';' + $env:LIB
$env:PATH = 'D:\dev-tools\cudnn\v9.14-cuda13\bin;' + $env:PATH
cargo build --release --locked -p indextts --features cuda-cudnn --example generate --example corpus
target/release/examples/corpus.exe --dir D:/workspace/index-tts/checkpoints-converted --reference D:/workspace/index-tts/examples/voice_01.wav --corpus tools/tts/acceptance-corpus.txt --out-dir target/indextts-review/rust-cuda --rounds 2 --seed 7 --device cuda

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
D:/workspace/index-tts/.venv/Scripts/python.exe tools/tts/indextts_corpus_py.py --checkpoints D:/workspace/index-tts/checkpoints --reference D:/workspace/index-tts/examples/voice_01.wav --corpus tools/tts/acceptance-corpus.txt --out-dir target/indextts-review/python-cuda --rounds 2 --seed 7 --device cuda

# 本机现有 oracle 及三个逐层夹具；全部设置，避免手工 ignored 测试缺参直接返回。
$env:INDEXTTS_CONVERTED_DIR = 'D:/workspace/index-tts/checkpoints-converted'
$env:INDEXTTS_TIKTOKEN_FILE = 'D:/workspace/index-tts/checkpoints/multilingual_zh_ja_yue_char_del.tiktoken'
$env:INDEXTTS_ORACLE_JSON = "$PWD/target/indextts-oracle/oracle.json"
$env:INDEXTTS_CAMPPLUS_LAYERS = "$PWD/target/indextts-oracle/campplus_layers.json"
$env:INDEXTTS_EMO_LAYERS = "$PWD/target/indextts-oracle/emo_layers.json"
$env:INDEXTTS_REGULATOR_LAYERS = "$PWD/target/indextts-oracle/regulator_layers.json"
cargo test --release --locked -p indextts --lib -- --ignored --nocapture --test-threads=1
```

## 限制

- 相同 seed 不保证 torch / Rust 采样轨迹相同；Rust 的 Options.seed 控制 token / 情绪随机选择，CFM 仍使用设备随机源，不能据此承诺 PCM 逐位复现。CUDA 归约顺序也可能产生差异。
- 时长 / RMS / 首步 logits 不能证明逐字完整性和主观音质相同；本次未运行 ASR 回读或人工听感验收。
- 本次没有核验真实长会话、取消、生产适配器、Metal、显存峰值；不能以独立 crate 探针代替这些验收。
- Windows CUDA release 链接仍报告已有的 LNK4098（LIBCMT 冲突）；本次未切换 Rust CRT 或升级原生依赖。CPU Clippy / rustdoc 单独验证。
