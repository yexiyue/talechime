# Qwen 性能改动审查与准确性验证（2026-10-18）

审查基线是 `59696d567e2ebb3b64d7794cff19cff0d1498189` 加上一轮未提交改动；本轮同时修正实现、简化设备数据流并重新验证。需求来源是本次用户指令：审查 Qwen 整体实现与性能，官方代码仅作参考，优化必须保证结果准确。仓库没有 `docs/agents/issue-tracker.md`，本次不依赖外部 issue。

**最终样本试听：用户反馈“目前感觉听感差不多了”。** 流式音色突变此前也已得到用户确认不再出现；这只验收本轮样本，不代表所有文本、音色或设备的质量等价。下文按阶段保留历史问题、待试听状态及追加修复，最终结论以末节为准。旧 A/B/C 的 ICL repetition penalty 实际为 1.5，尽管请求值是 1.05。“官方匹配”D 仅匹配了请求参数，没有匹配被隐藏覆盖后的实际行为。不得再引用为完全相同采样策略的对照。

**追加发现：原官方 D/E 的本机 PyTorch 2.7.1 MPS 路径还存在空历史 repetition penalty 异常。** 首帧分数受复用缓冲区内容影响，见末节的逐阶段追踪和预填缓冲区复现。原音频及用户偏好记录保留；修正后的官方对照另存，不把旧音频作为正确性标准。

## Standards 轴

1. **已修复，P1：MOSS 无条件 F32 主机往返。** 上一轮在共享 attention 中把 F32 tensor 取回 CPU 再上传，Metal F32 也会经过该路径；附带多处没有解决问题的 reshape。违背 AGENTS.md 的避免无关推理重构要求。全部移除。真正失败的位置是 codec 输入投影前的非连续 chunk 切片；只在 `codec.rs` 的 transpose 后加一次设备内 `contiguous()`，组合启用 Qwen Accelerate 的 fixture 已通过。
2. **已修复，P2：基准 revision 硬编码。** Python 工具现在从实际导入的官方源码目录读取 Git HEAD，并记录源码 diff 的 SHA-256、dirty 状态、模型目录和参考 WAV 的 SHA-256；避免不同源码被标成同一 revision。对应仓库的来源固定与实机证据边界要求。
3. **已修复，P3，简化判断：帧布局隐含约定。** 删除 `as_chunks::<16>().0` 的截尾路径；设备帧直接组成 decoder 的 `[1,16,T]` I64 tensor，显式校验帧宽度，不再经过 GPU→Vec→GPU。新增参考前缀与 codebook 顺序测试、错误宽度拒绝测试。

## Spec 轴

1. **已修复，P1：旧官方百分比比较不成立。** 旧 Python 主模型 `top_p=1.0`、声学随机采样，Rust 为 `top_p=0.9`、声学 argmax。相同音频时长不能证明计算量相同。README 撤回“RTF 低 62%”行，历史数值保留并注明失效原因；Python 工具改为显式相同采样策略。固定 seed 仍不保证 PyTorch 与本实现的随机序列相同。
2. **已修复，P2：预热与计时边界不实。** 旧 Python 的 sync 只处理 CUDA，旧三个进程均未预热，不能称作后端热路径。补 MPS 同步；Rust 探针加入双方预热、实际设备同步、首 PCM 与逐块时间，导出在计时外进行。零测量轮返回 null 而非伪造 NaN 观测。
3. **已补验证，P1：原来只有样本数和 EOS。** 新探针保存完整 codes、原始 F32 PCM、WAV 和 JSON。设备缓冲简化的独立 ABBA 对照要求输出逐字节一致，见下表。此证据证明该优化保留原输出，不等同于官方数值一致、内容正确或人工试听通过。
4. **已修正，P2：codec 开销解释错误。** `reference_frames.take()` 只在首块使用参考 codes，后续块独立解码，既不重放全部参考也不累计解码全部生成前缀。本轮保留该音频行为。流式与整段 PCM 本来就不同，块边界质量需独立验收。
5. **已修复，P1：GPU top-p 截止值同分扩大 nucleus。** 旧 GPU 用 `logits >= min_kept` 重建 mask，导致边界上所有同值 token 都保留；CPU 按排名只留前缀。现在统一使用 CPU 排名过滤，按原词表 ID 还原候选，并明确累计边界为“超过 p 的首个 token 仍保留”。CPU 测试覆盖同分边界、恰好等于 p、非排序词表 ID、多行；Metal 显式覆盖实际 3072 词表及 16 个 seed 的 CPU 策略一致性。直接 GPU scatter 的初稿只通过小词表测试，实模型生成反复选择 token 0，完整探针达到 114 帧上限，流式未正常结束，已主动停止并拒绝该 Metal 路径。随后新增 3072 宽度的稀疏 logits 测试，确认该版本的 Metal 排序/过滤组合产生空候选。最终删除重复的 GPU top-k/top-p 分支；加速器每次语义采样只回传一次 F32 logits，调用同一 CPU 策略，再把选出的 token 放回设备。Metal 已实机验证；CUDA 未做原生性能或质量验收。该修复可能改变旧 seed 的输出，是采样正确性修复，不能伪装成保持输出不变的性能优化。相同 F32 logits 与 seed 使用同一采样规则；不同设备模型前向的浮点误差仍可能改变逐 token 结果。

## 保持 PCM 的设备缓冲简化

Apple M4 Pro，Qwen 0.6B Base ICL，Metal BF16，同一本地 prompt、中文短段、seed 42、chunk_frames=10。独立进程串行 ABBA，每进程一轮完整+流式预热，再测一轮；计时期间无本任务的并行编译或推理。A 是上一轮延迟传输版本，B 是本轮设备 tensor 版本，两者均使用修复前的同一个 sampler，以独立验证数据搬运优化。

| 进程 | 完整合成 / s | 流式合成 / s | 首 PCM / s |
| --- | ---: | ---: | ---: |
| A1 | 4.9349 | 9.1995 | 1.1766 |
| B1 | 4.9125 | 9.1470 | 1.1465 |
| B2 | 4.9346 | 9.2394 | 1.1608 |
| A2 | 4.9163 | 9.1169 | 1.1457 |

流式 A 中位数 9.1582 s，B 中位数 9.1932 s（B 慢约 0.38%），处于本组波动范围，**没有证明进一步提速**。保留 B 是因为数据流更简单、消除了主机中转，并有严格输出证据。

四个进程的 86 帧、165120 samples、9 chunks、EOS 一致，以下 SHA-256 全部相同：

- 完整 PCM F32LE：`e72fe61a112cc0c8ebf254b71b66fe9ab3e4d755726d40a98d9b2b22f33f0533`
- 流式 PCM F32LE：`b117747c61eb36815a03e3875ce6bf1a125d9a6a6381420f9c815d260abfb1b9`
- 完整生成 codes JSON：`05aa775e661d6df7a34ec3701511bc75d881493f672d82ac686aebbfe8334f90`

原始证据在 Git 忽略目录 `runs/qwen-review/{a1,b1,b2,a2}/`，汇总 `buffer-comparison.json`。普通测试不加载模型。

## 后续性能方向

- **采样与固定张量复用。** 加速器现已使用单次 logits 回传后的 CPU 采样，GPU 内全词表 sort/cumsum 的方案须先解决真实词表宽度测试。可继续合并 CPU top-k/top-p 的排序、复用固定 shape 张量；另已移除 multinomial 中仅用于获取 shape 的无用 F32 转换。任何合并必须保留原词表 ID、同分与累计边界规则，并通过固定输入分布测试与实模型记录。
- **CodePredictor 15 步的小张量开销。** 可评估缓存位置 tensor、减少 `slice_assign` 与 kernel launch；不应仅为了吞吐改成并行预测或更换采样策略。需要同一隐藏态的 logits/codes 对照。
- **decoder 的语义 code 取余。** `Decoder12Hz::decode` 仍为语义 code 的 `% codebook_size` 读取 CPU；帧缓冲的设备化不等于整个 decoder 没有主机同步。可在明确输入 code 合法范围后移到设备端，并用相同 codes 验证 PCM。
- **流式 codec 上下文。** 官方 `chunked_decode` 默认使用 300 帧块和 25 帧左上下文，与本实现 10 帧独立块不是同一种策略。可用固定 codes 比较整段、独立块、有上下文块的 PCM、块边界与回读；在质量证据充分前不增大默认 chunk 或引入不完整状态缓存。
- **CPU Accelerate。** 已有单轮收益，但跨平台数值非位级一致、feature 会合并影响同进程其他模型；组合回归必须保留，不能以逐包测试替代。

## 自动验证

- Qwen Metal 单元测试：206 passed，2 ignored；其中原生 Metal 模型词表采样测试随后显式启用并通过（1 passed）。另一个忽略项是既有随机波动测试。
- `cargo test --locked --workspace --lib --tests --examples -- --test-threads=1 --skip default_home_migration_keeps_hello_lightweight_and_legacy_voices_visible`：通过；包含共享 Accelerate 下的 MOSS codec fixture。日志汇总见 `workspace-tests.log`（含子进程测试调用）。一次并行重跑出现无关配置锁测试的 WouldBlock/RevisionConflict 失败，9 个配置测试串行重跑全部通过；最终组合测试使用单测试线程避免该锁竞争，不改配置实现。
- 跳过项仍是已有的遗留数据迁移 5 秒门限用例；未改门限，未称作无跳过的完整门禁。前一轮单独复测约 9 秒，范围外证据见 CosyVoice3 性能记录。
- workspace Clippy `-D warnings`、workspace rustdoc `-D warnings`、rustfmt、`git diff --check`、Python `py_compile` 通过。

## 复现

`MODEL_DIR` 指向固定 Qwen 0.6B Base revision 目录，`PROMPT_JSON` 指向既有导入音色的 prompt 文件；不使用资源根目录。仅显式开发探针读取本地权重。

```sh
cargo build --release --locked -p talechime-qwen3-tts --features metal --example qwen_timing_probe
target/release/examples/qwen_timing_probe "$MODEL_DIR" "$PROMPT_JSON" '山风吹过松林，星光照亮归途。两个人沿着小路慢慢往前走。' metal runs/qwen-review/reproduce 3 10

PYTHONPATH=runs/reference/Qwen3-TTS runs/qwen-tts-venv/bin/python tools/tts/qwen3_tts_bench.py \
  --model_dir "$MODEL_DIR" --reference "$REFERENCE_WAV" \
  --reference_text '希望你以后能够做的比我还好呦。' \
  --text '山风吹过松林，星光照亮归途。两个人沿着小路慢慢往前走。' \
  --out_dir runs/qwen-review/official-reproduce --dtype bfloat16 --device mps --threads 8 --warmup --rounds 2
```

Rust 探针的轮次 0 是预热，后续才是测量；Python `--rounds` 不含额外预热。原始 F32LE 用于逐样本/哈希比较，WAV 仅供试听与回读。启用 `profiling` feature 可输出 `trace.json`，profiling 结果与正常计时分开；tracing 的主机 span 不等于 GPU 独占耗时，不能把所有异步 kernel 时间准确归给单个 span。

## 最终版本的新基线与回读

在上述修复后重新串行运行：每个 Rust 进程一轮预热，中文短段两轮测量、中英混读一轮；官方一轮预热加两轮测量。无并行编译/推理，模型与参考准备不计入合成耗时，原始 PCM/WAV/JSON 均在计时外导出。

| 路径 | 合成中位数 / s | 音频 / s | RTF | 首 PCM / s |
| --- | ---: | ---: | ---: | ---: |
| Rust Metal 中文整段 | 4.2360 | 6.64 | 0.6380 | — |
| Rust Metal 中文流式 | 8.5487 | 6.64 | 1.2875 | 1.0799 |
| Rust Metal 中英混读整段 | 4.0696 | 6.32 | 0.6439 | — |
| Rust Metal 中英混读流式 | 7.8598 | 6.32 | 1.2436 | 1.0701 |
| 官方 PyTorch MPS 中文，显式匹配采样策略 | 20.1763 | 6.88 | 2.9326 | 未提供 |

Rust 中文为 83 帧、9 chunks，混读为 79 帧、8 chunks；流式均正常 EOS。每种文本各路径的预热与测量原始 PCM SHA-256 均各自一致，证明本版本固定 seed 可复现；整段和流式 PCM 不相同。采样修复后中文输出由旧 86 帧变成 83 帧，不能把这张表与旧表的耗时直接换算成保持输出的提速。

官方匹配的是显式采样选项：语义 top-k=50、top-p=0.9、temperature=0.9、repetition_penalty=1.05，声学贪心。双方随机数实现、输出样本数与 codec 精度仍不同：Rust Talker/CodePredictor BF16，codec 保留 checkpoint F32（本地 safetensors 头验证：主模型 478 个 BF16 tensor，speech tokenizer 496 个 F32 tensor）；官方传入 BF16 并传播给 tokenizer 加载。因此本表只列绝对测量，不宣称逐算子同精度、官方数值一致或统一框架提速。官方 revision/补丁/输入与配置证据在 `runs/qwen-review/official-matched/summary.json` 与 `official-source.patch`，新版本汇总在 `final-summary.json`。

本地 Whisper small 缓存为未完整下载的 25 MB 文件，读取时报缺少 zip central directory；保留原文件，未把该模型用于验收。改用已有完整 Whisper tiny 对旧/新整段、旧/新流式、中英混读与官方 PCM16 WAV 回读。新中文整段回读为“山峰吹過松林 新光照亮規圖兩個人嚴著小路慢慢往前走”，新流式为“山峰吹過松林 行光著亮歸圖兩個人原著笑路 慢慢往前走”；旧与官方回读也存在相似的同音字错误。混读为“今天是10月18日温度25摄石度埃累斯说哈囉欢迎回家”。完整文本在 `readback.json`。

这些 tiny 回读只能辅助检查大致句子覆盖，不能判断同音字、块边界和英文发音是否正确，**不记作内容/听感验收通过**。本轮已经证明缓冲优化保持原 PCM、修复后的采样满足明确规则和本机复现；没有把“有限音频 + EOS”当作逐字准确。下一阶段应先用固定 codes 检查 codec 块边界，再补可靠的双模型回读或人工试听，之后才改变默认上下文/分块策略。

## 采样策略说明与试听实验

“官方匹配采样”是开发对照配置，不是官方默认值，也不表示两套实现产生相同随机序列。

| 项目 | 旧试听 Rust A/B/C | 官方 D | 官方 E 默认 |
| --- | --- | --- | --- |
| 语义 top-k / top-p / temperature | 50 / 0.9 / 0.9 | 50 / 0.9 / 0.9 | 50 / 1.0 / 0.9 |
| 实际 repetition penalty | **1.5（隐藏覆盖）** | 1.05 | 1.05 |
| 声学码 | 贪心选择 | 贪心选择 | 随机采样，50 / 1.0 / 0.9 |
| max_new_tokens 上限 | 请求 2048，被 ICL 文本长度启发式进一步限制 | 2048 | 8192 |

随机采样提供更多候选变化，贪心声学码减少一层随机性；两者都不能仅凭参数推断发音准确率或听感优劣。即使显式匹配，PCG/PyTorch RNG、浮点边界、codec 精度与分块方式仍不同。固定 seed 只能分别验证实现内复现，不能对齐双方的抽样结果。对照音频沿用同一文本和参考音色，但没有实现间 PCM 等价承诺。

新增 `qwen_codec_probe` 读取已经生成的同一组 codes，分别整段解码、10 帧独立块解码、10 帧块加 25 帧左上下文解码。第一块均保留完整 ICL 参考，随后上下文实验仅保留有限历史；按 1920 samples/frame 去除上下文，并检查完整样本数。这是独立的开发试听实验，不修改默认流式行为，也不是完整复现官方的 300 帧分块。

```sh
target/release/examples/qwen_codec_probe "$MODEL_DIR" "$PROMPT_JSON" \
  runs/qwen-review/selection-final/codes-1.json metal runs/qwen-review/codec-audition

# 其他参数沿用上文；选择 checkpoint 解析出的官方默认采样配置。
# --policy matched 为原脚本的缺省行为。
# tools/tts/qwen3_tts_bench.py ... --policy official-default
```

试听文件保存在忽略目录 `runs/qwen-review/audition/`，原始 F32LE 保留在各运行目录。试听副本按整段 RMS 统一音量，每个文件只施加一个常量增益；共同目标为 -23 dBFS RMS，若任何文件会超过 -1 dBFS peak 则降低共同目标。不裁剪、不淡入淡出、不压缩动态、不重采样，以保留块边界问题。此处理不是 LUFS 响度归一化。`manifest.json` 记录来源、原始哈希、增益与输出峰值。

| 文件 | 路径 | 音频时长 |
| --- | --- | ---: |
| A.wav | Rust 整段 | 6.64 s |
| B.wav | Rust 默认独立流式块 | 6.64 s |
| C.wav | 同一 codes，25 帧左上下文实验 | 6.64 s |
| D.wav | 官方匹配采样 | 6.88 s |
| E.wav | 官方默认采样 | 6.08 s |

五份均为 24 kHz 单声道 PCM24 WAV，重新读取验证样本数、有限值、RMS 误差小于 0.01 dB、无削波；本次共同目标保持 -23 dBFS。A/B/C 固定 codes，D/E 则改变了采样/实现条件，不能把听感差异归因到单一采样参数。官方默认运行在一轮预热后测得 19.0647 s，输出 6.08 s；预热/测量 PCM 哈希一致，但单轮、不同 token 数，不能与匹配策略换算成纯采样开销或质量优劣。

CPU top-k 改为 `select_nth_unstable_by` 查找阈值，仍在原词表顺序上按阈值保留同分候选；含 NaN 的输入沿用旧排序路径。3072 词表、多行、同分、负无穷及不同 k 的旧排序对照通过。独立 release 微基准（3072 元素、k=50、两万次、ABBA）旧排序约 0.355 s，阈值选择约 0.075 s；仅该阈值查找约快 4.7 倍，每语义步只省约 14 微秒，不能换算为整段合成的同等提速。

修改后串行重跑一轮预热、两轮测量：整段中位数 4.2375 s，流式 8.4898 s，首 PCM 1.0719 s；仍为 83 帧、159360 samples、9 chunks、正常 EOS。与上一基线 4.2360 / 8.5487 s 相比，没有证明整段性能改善。`selection-final/` 的 codes、整段和流式 F32LE 与 `final/` 的对应输出逐字节一致；`codec-audition/` 的整段和独立块输出也分别与实际整段/流式路径逐字节一致。实验新增的 25 帧上下文版本具有相同输出长度，PCM 有变化，需人工试听，未宣称改善音质。单次无独立预热的 decoder 探针约为整段 1.27 s、独立块 5.52 s、上下文块 6.90 s，仅用于识别重复解码开销，不作为正式性能对照。

本次增量验证：Qwen Metal 单元测试 207 passed、2 ignored，两个开发 example 编译通过；原生 Metal 3072 词表采样测试显式启用后另有 1 passed。该包 Metal all-targets Clippy `-D warnings`、examples rustdoc `-D warnings`、workspace 格式和 diff 检查、Python 编译检查通过。此前 workspace 组合验证见上文，本次未重复全 workspace 测试。试听 manifest 与逐字节比较见 `audition/manifest.json`、`selection-equivalence.json`，测试日志为 `audition-{tests,metal-test,clippy,rustdoc}.log`。

## 人工试听反馈后的正确性修复

用户反馈 D/E 整体听感优于 Rust，B 流式存在音色突变。这是本次样本的人工反馈，不扩展成所有文本或音色的质量排名。追加审查发现并修复：

1. **P1：ICL 隐藏改写采样参数。** 整段和流式把请求的 repetition penalty 1.05 抬到 1.5。删除该覆盖，直接使用 `SynthesisOptions::to_gen_config()`。D 同样是声学贪心，不能据此把官方优势归因于随机声学采样。此前声称 D 与 Rust 的采样参数完全匹配有误，已在本记录开头及对照表更正。
2. **P1：流式 codec 丢弃历史。** 第一块使用参考 codes，之后每个 10 帧块都从空历史重新解码。现保留最近 25 帧参考/生成 codes，拼到下一块前，解码后去除上下文样本；只返回新块，不重复播放历史。首块继续使用完整 ICL 参考。历史长度有界，取消和会话所有权不变；解码成功后才提交缓冲与上下文更新。保留有限 codes 历史不是完整的神经网络状态缓存，也不承诺与整段位级相同。
3. **P1：codec attention 忽略 72 帧窗口。** checkpoint 的 `decoder_config.sliding_window=72`，官方 decoder 层使用 sliding attention；本实现只屏蔽未来，允许访问全部过去。修正为当前帧及之前 71 帧，窗口外和未来均屏蔽。新增第 71/72 帧边界、过期历史、未来及零窗口拒绝测试。
4. **P2：整段 ICL 隐藏截断。** 删除 `min(max_length, max(75, text_token_count*6))` 的启发式限制，尊重调用方帧上限。该项是独立的完整性问题，旧短句未达到该截断点，不能用于解释这一次听感。

用同一组旧 codes 隔离 codec 后，旧 B 相对旧整段 A 的波形相关性约 0.5162、RMSE 0.03963；25 帧上下文实验 C 的相关性约 0.9891、RMSE 0.006237。相关性说明固定输入的数值偏离，不能证明听感验收通过。

交叉 decoder 检查：把完全相同的 Rust 参考和生成 codes 交给官方 tokenizer 解码，原 Rust 整段相对官方 F32 的相关性约 0.99816、RMSE 0.002408；差异在超过 72 帧后出现，此前两个 0.8 秒窗口误差约 6.5e-8。该定位将随机采样与 decoder 差异分离，不把官方代码本身当作正确性证明。输入、脚本与原始输出位于忽略目录 `runs/qwen-review/{codec-cross-check.py,codec-cross/}`。

修正窗口后，同一组 codes 的 Rust F32 / 官方 F32 decoder 最大误差 **1.24425e-6**、RMSE **7.36380e-8**，相关性约 0.999999999998；不是位级一致。另导出 D 的完整 codec 输入，用 Rust F32 解码，和官方 BF16 输出的相关性约 0.999795、RMSE 0.0008921，剩余差异含精度影响。导出过程未预热，不引用该运行的时间作性能对比。原始证据为 `codec-window-fixed/`、`official-export/`、`official-codes-rust/`。

### 参考编码器 RoPE 配对修复

参考 codes 的交叉检查揭示另一项 **P1**：直接加载 HF checkpoint 到 Candle Mimi `ProjectedTransformer` 时，没有适配 RoPE 的布局。HF Mimi 对前后半组分量旋转；Candle 0.11 Mimi 使用 `rope_i`，对相邻分量旋转。现仅在加载参考 encoder 的 Q/K projection 时，把每个 head 的输出权重行从半组布局交错排列；V、输出投影和 checkpoint 文件不变。

同一 24 kHz 参考 WAV：修复前 Rust F32 与官方 F32 编码结果相同的比例仅 **43.75%**，语义 code 为 **50%**；与官方 BF16 的总比例为 29.40%。重新调用当前旧编码器确认与旧 `prompt.json` 完全相同，排除了单纯缓存过期。转换后重新生成完整 prompt，44 帧 × 16 codebook 的全部 codes 与官方 F32 **100% 逐项相同**。该离散结果是这一条参考音频上的实测，不扩展为所有输入的编码或音质保证。

应用改用 `prompt-encoder-rope-v2.json` 缓存名，下次使用自定义音色时从原参考音频与 transcript 自动生成。旧 `prompt.json` 保留，不覆盖旧缓存或参考源文件。直接库调用仍可显式加载已有有效 codes；开发探针必须传入重新导出的 prompt，不能误用旧缓存评价本修复。

新增开发入口 `qwen_reference_probe MODEL REFERENCE TRANSCRIPT DEVICE OUTPUT_JSON`，读取本地资源，输出完整新 prompt，不合成语音。`reference-cross/{official-float32.json,official-bfloat16.json,rust-current.json,fixed-prompt.json,fixed-comparison.json}` 保存隔离对照。投影转换新增多 head 顺序及非法 shape 回归测试。

### 使用新参考 prompt 的最终验证

一轮预热、两轮测量，M4 Pro Metal、Talker/CodePredictor BF16、codec F32、seed 42、10 帧新块、25 帧历史；仍用前述短句、同一参考 WAV 和 transcript。三轮生成均为 88 帧、168960 samples（7.04 s）、9 个流式块并正常 EOS；每种路径的三轮原始 codes/PCM 分别完全一致。旧 A/B 及中间 `quality-final/` 仍用了修正 RoPE 前的参考 codes，不作为当前完整修复版本的质量结果。

| 路径 | 合成中位数 | 首 PCM | 音频时长 |
| --- | ---: | ---: | ---: |
| 修复后整段 | 4.4681 s | — | 7.04 s |
| 修复后流式 | 10.1101 s | 1.0694 s | 7.04 s |

本轮优先修正数值与连续性；token 数和解码历史改变，不能与旧版本的绝对秒数换算为保持输出的提速或退化。上下文仍增加 codec 的计算量，尚未实现状态缓存。当前流式与其整段波形相关性约 0.9855，RMSE 0.006978，有限历史仍不能替代完整历史，也不能凭此宣布听感问题已全部消失。

新试听目录 `runs/qwen-review/corrected-audition/`：`ours-full.wav`、`ours-stream.wav`、`official-D.wav` 和诊断用 `official-codes-rust.wav`。全部采用一个整段常量增益匹配 RMS，24 kHz PCM24，重新读取检查样本数、有限值、无削波及 RMS 误差小于 0.01 dB；没有裁剪、淡入淡出或动态压缩。D 的重新导出 PCM 哈希与原 D 完全相同。来源与处理证据见 `manifest.json`，汇总见 `corrected-summary.json`。**本轮新音频尚待用户试听，不声明已追平 D/E。**

离线 Whisper tiny 回读新整段为“山峰吹过松林 新光照亮规徒两个人沿着小路慢慢往前走”，新流式为“山峰吹过松林 新光照亮规徒两个人言着小路慢慢往前走”。记录在 `corrected-readback.json`；同音字与“沿/言”等仍不足以用该模型判断内容正确性，不据此记作准确率或听感验收通过。

增量组合门禁：Qwen 210 passed、2 ignored，backends 43 passed，相关 examples 全部编译通过；两包 Metal all-targets Clippy 与 examples rustdoc（包含 private items）均 `-D warnings` 通过；格式、diff 和 Python 编译检查通过。Clippy 首次发现手写整除判定，已改为标准 `is_multiple_of` 后重跑通过。没有重复前述整 workspace 门禁。日志为 `corrected-{tests,clippy,rustdoc}.log`。

实际后端运行已自动生成新的参考缓存，其完整 JSON 与开发探针的新 prompt 相同，旧 `prompt.json` 仍存在。额外的 debug 运行用于检查取消及模型复用，不把 debug 耗时与上表 release 性能混用。该过程的一秒主机采样集中在 codec ConvNeXt 的 depthwise/grouped conv 经 Candle 分组分派 Metal kernel 的路径；这是后续可单独测量的开销候选，不据单次 debug 栈推断 release 占比或直接改写算子。原始采样为 `backend-debug.sample.txt`。

debug 探针因执行缓慢主动终止，改为 release `native_probe` 验证。启用 `NOVEL_TTS_PROBE_CANCEL=1`，收到首块后丢弃接收端，再在同一后端实例完成下一请求：1 个 segment、168960 samples、24 kHz、7.04 s、正常 End。输出 PCM 与独立库新流式逐样本一致。该运行首 PCM 1389 ms、生成 10507 ms，含实际 adapter 边界，不与上表纯模型路径混用。证据为 `corrected-backend-release.log`、`corrected-backend-comparison.json`；没有播放到音频设备。

## 首句表现力的人工复核与隔离检查

用户对 `corrected-audition/` 的反馈：听感已接近，流式没有再出现音色突变；但官方 D 的第一句话更有感情，Rust 较平缓。因此，这条样本的流式音色连续性有了人工复核证据，首句表现力差距仍然开放，不记作与官方听感完全等价。

继续用同一参考 WAV 检查条件输入：Rust / 官方 F32 speaker embedding 余弦相似度约 0.999999745、RMSE 0.0002322；F32 参考 codes 完全一致。相对官方 BF16 speaker embedding 的余弦相似度约 0.999997831，但参考 codes 仅 52.27% 相同，说明比较 D 时仍包含参考编码精度的差别。原始输出在 `prosody-prompt/`，检查脚本为 `prosody-prompt-check.py`；这不是情绪优劣的数值评分。

### 编码向量累加精度

BF16 下，原实现逐组相加并在每一步舍入，且生成时先加 15 个声学向量、最后加语义向量；官方对所有 16 组求和。真实参考向量的独立 Torch 检查：逐步累加相对整体求和的最大误差约 0.00390625，先声学后语义约 0.0078125；F32 累加并只在最后转换 BF16 与整体求和逐项相同。

现由共享的 crate 内部 helper 实现整组求和，BF16/F16 在 F32 累加后返回原 dtype。参考条件和整段/流式生成都使用同一规则，公共声学 embedding 求和入口也统一实现。新增 `[256, 1, -256]` 的 BF16 消去回归，避免丢失小贡献；原始数值验证见 `prosody-sum-comparison.json`。这会改变生成概率与 PCM；修复不等于已证明情绪改善。

### 首帧逐阶段对照与官方 MPS 异常

使用完全相同的官方 BF16 prompt，导出首步计算。双方目标文本 token、参考文本 token、prefill 长度 54 与 prefix shape 均相同；修正累加后的参考 embedding **逐项一致**。最后 hidden 的余弦相似度约 0.999836，原始 logits 余弦相似度约 0.999929；原始分数最高的语义 token 均为 **898**（Rust 24.0，官方 24.125），不是结构完全不一致。`probe_voice_clone_prefill` 是普通测试默认忽略的显式本地模型探针，需要设置 `QWEN_PREFILL_{MODEL,PROMPT,TEXT,DEVICE,OUTPUT}`，不会默认访问模型或下载资源。输出见 `prosody-prefill-{rust,official,comparison}.json`。

但原官方贪心运行第一帧却选择 1537，后续逐阶段追踪发现分叉在 **RepetitionPenaltyLogitsProcessor 对空 input_ids 的处理**：空历史理应原样返回 scores，而本机 MPS scatter 返回复用缓冲区内容。一次跟踪首帧分数变为约 1.00 的数值；加入 top-k 跟踪后，复用分数又变成排在索引 0/1/2 的 24.125/23.0/21.5，并选择 token 0。记录为 `prosody-greedy-trace-class.log`、`prosody-greedy-stage-trace.log`，说明跟踪自身也能改变分配器状态。

首次朴素的独立空历史测试在 CPU/MPS 上均误差为零，不能证明不存在该异常；补充“预填并释放同形状缓冲区”的检查后稳定复现：CPU 四轮均原样保留分数；MPS 四轮输出依次成为预填的 -900/-901/-902/-903，最大误差分别为 924.125/925.125/926.125/927.125。真实词表宽度为 3072，score dtype F32。`empty-penalty-poison-comparison.json` 与 `empty-penalty-poison.py` 是独立证据；不推广到未经验证的其他 Torch 版本或设备。

开发对照脚本现在默认 `--empty_repetition guard`，仅对空历史跳过重复惩罚，非空历史仍调用原实现；上下文退出时恢复原 class method。`--empty_repetition original` 显式复现历史行为。summary 记录该选项和 benchmark 脚本 SHA-256，官方 checkout/依赖文件不被此 runtime guard 改写。空历史的正确语义与 CPU 独立验证一致；Rust 无需模拟 MPS 缓冲区异常。

原 2048 帧上限的 Rust 贪心诊断曾出现 Metal buffer 分配失败；改用显式 128 帧上限后可记录触达上限的结果。原官方异常路径 78 帧结束，Rust 触达 128 帧，这组差异混入了首帧处理错误，不能用于判断模型实现、情绪或贪心质量。贪心诊断与首步短序列仅用于定位，均不作为完成的试听候选。

开发入口 `qwen_prosody_probe MODEL PROMPT TEXT DEVICE OUTPUT_DIR TEMPERATURE MAX_FRAMES SEED...` 导出不同 seed 的 codes、PCM、参数和 `hit_frame_limit`。拒绝空输出和非有限 PCM，试听只使用未触达上限的完整候选。产品默认采样参数不变，不根据单条样本挑选“最佳 seed”作为产品优化。

guard 的独立验证覆盖 CPU/MPS、空历史和单个历史 token，均与 CPU 正确结果相同；正常退出及异常退出均恢复原 processor method，见 `empty-penalty-guard-validation.json`。修正后的官方贪心短序列前 3 帧语义 code 与 Rust 相同，全部声学/语义 code 合计匹配约 41.67%，仍不承诺声学数值或整段 waveform 等价，见 `prosody-guard-greedy-comparison.json`。

### 新试听与增量验证

`runs/qwen-review/prosody-audition/` 采用同一句正文、同一参考 WAV，统一整段 RMS 到 -23 dBFS，PCM24/24 kHz，仅施加一个常量增益；不裁剪、淡化块边界或压缩动态。manifest 记录源/输出 SHA-256、增益、样本数，重新读取校验无削波、有限值、样本数及 RMS 误差小于 0.01 dB。

| 文件 | 内容 | 时长 |
| --- | --- | ---: |
| A-default42.wav | F32 累加后的当前策略，固定 seed 42 | 6.88 s |
| B-seed0.wav | 同采样策略，seed 0 | 6.96 s |
| C-seed1.wav | 同采样策略，seed 1 | 6.88 s |
| D-seed2.wav | 同采样策略，seed 2 | 7.04 s |
| E-official-guard.wav | 官方匹配策略，修正空历史异常，seed 42 | 6.96 s |
| F-original-D.wav | 保留原 D，含原 MPS 路径 | 6.88 s |
| G-stream42.wav | 当前策略的流式 seed 42 | 6.88 s |

三份新 seed 候选均未触达 256 帧上限。seed 42 的完整/流式分别运行预热及一次测量：均为 86 帧、165120 samples、9 chunks、正常 EOS，各路径 codes/PCM 分别完全复现；本次整段约 4.3171 s，流式约 10.2059 s，首 PCM 1.0731 s。该单次结果不作为新的提速结论。官方 guard 运行两次，均 87 帧且 PCM SHA-256 相同；未单独预热，时间不用于与 Rust 量化比较。旧 D 文件始终保留。

新版默认策略、seed 0/1 和官方 guard 的离线 Whisper tiny 回读均记录在 `prosody-readback.json`，仍有同音字与“沿/严”等错误，不能据此判定发音准确率或情绪优劣。新版候选的首句表现力等待人工试听；不依据一次 seed 偏好更改产品默认参数。

本次两包组合门禁：Qwen 211 passed、3 ignored，backends 43 passed，相关 examples 编译通过；两包 Metal Clippy all-targets 与 private-items/examples rustdoc `-D warnings` 通过，格式、diff、Python 编译检查通过。显式本地 prefill 探针另有 1 passed。未重复完整 workspace 门禁。日志见 `prosody-final-{tests,clippy,rustdoc}.log`，本轮没有改动模型权重或外部依赖文件。

## 最终试听接受与提交审查

用户对上述新试听反馈“目前感觉听感差不多了”，并要求审查、简化、记录后提交推送。结合此前“流式没有突变音色了”的反馈，本条样本的连续性与整体听感获得人工接受；不宣称官方波形相同、普遍情绪表现一致或所有模型变体验收通过。产品默认 seed 与采样参数不因候选偏好调整。

本次 simplify 删除 `StreamingSession.current_token`：设备 token tensor 的 `Option` 已完整表达待生成状态，不再同时维护一个只用于判断是否存在的 CPU token ID。首 token/EOS 判定、重复惩罚、生成顺序、帧缓冲和解码算术保持不变。

最终并行复审以 `git diff HEAD` 和本轮新增探针/工具/记录为范围，基线仍为 `59696d5`，需求以本次会话和本记录为准：

- **Standards：0 项新的可操作问题。** 仓库规范及 12 项代码异味启发式审查未发现需追加修复的事项。
- **Spec：2 项 P2，均已修复并复审。** 官方工具原先把帧上限产生的非空音频也纳入成功统计，且未记录 seed 和参考转写。现捕获 Talker 的生成 token tensor，在计时结束后检查末 token 是否 EOS；不新增计时内主机传输。逐轮记录 `eos`、`termination`、token 数；完整合成中位数只包含非预热且 EOS 的轮次，无有效轮次时为 null。summary 增加 seed、参考转写及完成测量轮数。EOS 恰好在上限处仍按完成处理，其他终止保守标为 incomplete。

提交前门禁：workspace 串行 lib/tests/examples 共 657 passed、24 ignored，55 个测试二进制；仅跳过既有 `default_home_migration_keeps_hello_lightweight_and_legacy_voices_visible` 五秒时限测试，其独立历史超时见前文，不宣称完整无跳过通过。Qwen/后端 Metal 组合 254 passed、3 ignored；显式原生 Metal 3072/4 词表采样测试另有 1 passed。两包 Metal all-targets Clippy、private-items/examples rustdoc 均以 `-D warnings` 通过，格式及 Python 编译检查通过。日志为 `commit-{workspace-tests,metal-tests,metal-native,clippy,rustdoc}.log`。忽略的真模型测试没有作为普通回归运行；本次没有新增 CUDA/Windows 实机验收。

官方工具的显式本地真模型检查：4-token 上限输出 5760 samples，但 `eos=false`、完成测量轮数为 0、中位数为 null；375-token 上限运行在第 88 个 token 正常 EOS，输出 167040 samples，完成测量轮数为 1。完整运行原始 PCM SHA-256 与之前受保护的官方对照完全相同（`bf174054…97ebe4`），没有改变该候选的声音。两组 summary 均包含 seed 42 与参考转写；证据位于 `commit-official-{cap,eos}/` 和 `commit-official-validation.log`。均未预热，仅验证工具与输出，不新增性能排名。

保留下一步性能候选：流式 codec 的显式卷积/attention 状态缓存，以及 CodePredictor 的小算子分派开销。前者须先对固定 codes 验证跨块数值、样本覆盖、取消后复用和听感；本次不为降低耗时缩短已接受的历史上下文，也不依据旧 D/E 的异常输出调节模型。当前缺少修复后的同精度、同边界、多轮配对官方速度矩阵，不恢复官方提速百分比。
