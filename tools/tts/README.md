# 固定后端文本矩阵

开发验收工具，不是通用文本预处理器。宿主传原文，每个模型拥有自己的 TN/G2P。真实运行输出放在 `.local/`，不得提交音频、权重、私有正文或参考。

`backend-variants.json` 登记 14 个当前推理/辅助设计变体及源码、资源清单、设备边界和剩余缺口。`text-matrix.json` 为 19 条公开输入，包括原有 13 类、全角数字、无假名日文日期、连续空白、长词、长段和日语读音标注。每个变体均有独立计划行；参考语言、设计、方言、速度、时长和情绪模式的附加矩阵另列。2026-11-12 用户明确将 19 条原始 BF16 CUDA 实机行另行安排，本机结论不代表 CUDA 通过，见验收记录及 `finish-official-backend-parity/deferred-cuda.md`。

```sh
python3 -m unittest discover -s tools/tts -p 'test_*.py' -v
python3 tools/tts/run_text_matrix.py .local/matrix-config.json .local/matrix-run-001 \
  --probe target/release/examples/native_probe --timeout 600
```

配置为 JSON 数组，每项需 `id`（目录中的变体 ID）、`backend`、`directory`、`voice`。可选 `device`（默认 cpu）、`params`（协议原生标量，如 `{"language":"auto"}`）、`style` 和 `reference_source`（公开来源说明）。`env` 只允许 `TALECHIME_KOKORO_MODEL`、`MOSS_MODEL`。程序不会把任意环境变量或未知配置字段存入报告；子进程只继承路径、用户目录和临时目录基础变量，不继承凭据。参考音色须先由现有资源/音色流程准备，本工具不导入、不下载、不修改资源。VoiceDesign 使用独立的 `qwen-design` / `moss-design` / `voxcpm-design` 探针入口和必填 `style` 描述，调用真实设计 API；其长度限制与朗读不同。

输出目录必须全新，避免覆盖旧证据。`plan.json` 先于执行写入；`results.jsonl` 逐行落盘；`resources.json` 记录实际模型文件的大小和 SHA-256（不收录可变音色目录）；`build.json` 记录提交、工作区跟踪 diff 摘要、probe 二进制摘要和执行平台。探针另报编译 feature、构建时 rustc 版本与实际加载的模型 ID；参考记录/WAV 摘要和模型内部阶段快照独立记录。加载的模型 ID 必须与计划变体匹配。这些证据不等于官方数值对照。

`resource_directories` 可显式列出当前模型及共享 codec 的目录，必须位于配置根目录内；
避免把其他模型和准备缓存混入本轮摘要。参考记录按 backend/model/runtime_revision 过滤。
`case_params` 为已声明用例设置模型参数，不修改公共原文。
预期拒绝须在目录中声明 `reason` 与 `error_contains`，仅进程失败且匹配确切模型错误时
记录 `expected_rejection`；缺件、崩溃、超时仍是独立失败，预期拒绝不计为朗读成功。

`aggregate_text_matrix.py NEW_OUTPUT REPORT...` 按参数顺序合并完整报告，后面的
报告提供同一用例的最新生成事实；原文或摘要不一致、缺行、重复行会拒绝合并。
已有听验失败仍保留，每条音频重新校验模型身份、来源/PCM 覆盖及原始 WAV 摘要。
合并与 `verify_text_matrix.py` 都不会悄悄接受已经改变的 WAV。
`sources.json` 保留各来源报告的摘要和位置，控制参数音频等额外矩阵另行列出。

每条输出含原文 SHA-256、UTF-8 范围、设备、音色、参数、原生日志及分段。生成通过需退出成功、成功流结束或设计 API 完成记录、非空有限且非静音 float32 PCM、有界分段和原文及 PCM 完整覆盖和有效 UTF-8 边界。它们不证明逐字完整；矩阵也不会从时长推测发音。

状态独立：

- `generation`：generated / expected_rejection / failed / timeout / cancelled / blocked。配置或资源缺失为 blocked，计划行不能被省略。超时保存部分 stdout/stderr，终止进程树并等待主进程退出。POSIX 后代清理有本机测试，Windows 清理代码尚未实机验收。
- `official`、`numeric`：默认 pending。当前 runner 不伪造官方对照或数值通过证据。
- `listening`：默认 pending；用户已确认的失败保留为 failed。新音频生成不自动清除历史反馈；单词独立成功、手工改写或拼接实验不算产品修复。
- `expected`：带来源的预期范围说明，不把任意加载错误当成“预期拒绝”。实际拒绝原因尚须核对；字符可编码不表示支持该语言。

`summary.json` 核对计划与实际结果的差集、重复和各层统计；`listen.html` 为本机音频索引。`coverage_complete` 只代表所有计划行都有记录，含 blocked/failed 时整体仍未完成。命令发现未生成行会返回非零。测试使用假探针和公开短句，不下载模型，不作真实声音证据。

## 日语 oracle

`indextts_ja_oracle.py` 仅在开发环境运行官方固定脚本，不进入产品依赖。环境固定 `fugashi==1.4.0`、`unidic-lite==1.0.8`（UniDic 2.1.2）；脚本 SHA、词典内容 SHA 和 14 条公开样本见 `crates/indextts/tests/fixtures/japanese-oracle.json`。同一环境重复生成逐字节一致；Rust 默认分词已通过这 14 条对照；固定 IndexTTS front-v6 的真实 Metal 日语样本已生成，听感仍待人工确认。完整证据见 `docs/records/tts-core/official-backend-parity-2026-11-11.md`。

```sh
python tools/tts/indextts_ja_oracle.py /path/to/pinned/ja_g2p.py /new/path/oracle.json
```

官方脚本版本为 IndexTTS `d9e41aac89fd00b3d71497fddb287b7f24613712`，`ja_g2p.py` SHA-256 在工具内固定，默认 ratio=0，保留连续 ASCII 空格及原字，不对日语日期执行中文数字展开。日语 oracle fixtures 的词典许可副本为相邻 `UNIDIC-BSD`。

模型 EOS 未由公共适配器 API 暴露时记录 `model_eos=not_exposed`；`AudioChunk::End` 只证明流正常结束，不能伪称模型 EOS。模型内部的截断/结束回归另列。

## 独立官方证据

`annotate_official_matrix.py SOURCE NEW_OUTPUT --receipt RECEIPT_JSON` 将已执行的
官方前端/条件门禁附加到完整报告。收据包含命令、成功退出、日志 SHA 和 fixture SHA，
逐项原文 ID/hash 必须对应公开语料；缺证据保留 pending，失败收据或变更过的日志/fixture
直接拒绝。聚合、再次复验和注释沿用真实音频路径，并重新校验 WAV、原文、模型及分段。
试听页可展开查看模型、设备、音色、参考来源、参数和执行文本；完整 token/音素诊断见 JSON。

`conditioning_checked_synthetic` 仅指固定官方 AST 的合成条件结构：Qwen/OmniVoice/VoxCPM
的 token/embedding/参考特征并非全部来自实际 checkpoint。MOSS 的
`prompt_matched_actual_tokenizer` 使用固定实际 tokenizer，但参考 codes 仍是合成输入。
`frontend_matched` 使用实际 TN/词典/tokenizer 所需资源；`phonemes_matched` 指 Kokoro
音素对照。`policy_rejection` 是明确符号拒绝，`scope_declared` 是支持范围声明，
`difference` 保留执行规范化与官方输入的已记录差异，均不等于音频或人工听验通过。
三种 Kokoro 产品策略差异（空格年份、换行、连续空格）保留在逐项记录；
原始失败、后续显式语言成功和当前音频的关系见验收台账，不能只拿合并后的生成数字推断修复。
