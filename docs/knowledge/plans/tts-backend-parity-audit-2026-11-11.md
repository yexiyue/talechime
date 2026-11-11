# TTS 后端官方能力差异初审

后续实施与任务归属见 [finish-official-backend-parity](../../../openspec/changes/finish-official-backend-parity/ownership.md)。

### 必须保留的用户听验失败

| 后端/样本 | 内容完整性 | 读音/连贯性/音色 | 后续状态 |
| --- | --- | --- | --- |
| Nano，中英日期混排 | 用户确认漏读 hello | 未通过 | 后置修复；单独 hello 有声不能覆盖本失败 |
| Nano，逐位年份分段拼接实验 | 未确认 | 用户确认 2026 不连贯，二六音色异常 | 失败；实验不是产品修复 |
| Qwen Base，中英日期混排 | 未确认 | 用户确认 2026 读英文 | 失败；参考语言影响未证实 |
| VoxCPM Q8，中英日期混排 | 未确认 | 用户确认读成二零二十六年 | 失败；手工改写样本不代表自动修复 |
| Kokoro g2pW，中英日期混排 | 未确认 | 用户确认读成二千零二十六年 | 原样本失败；代码已修复空格年份，重生成待听验 |

上述结论独立于生成成功、EOS 和有效 PCM；后续重跑默认待听验，不能自动清除用户失败。私人文本及原始音频不入库。

日期：2026-11-11。TaleChime `3f5f1d527c23801b50f4a881fbe688c82029d3ce`；宿主 CastGlean `8d14408e36e0f6539b068faeb97a127bf3e20019`。本记录是静态源码、既有台账与官方材料的初审，未重新执行真实模型或听验。固定版本来源见各 crate `SOURCE.md`。上游最新版新增能力不等于当前固定模型缺陷。

后续工作中的源码已开始修复：宿主保留原文并将原有映射迁入 Kokoro；Qwen、OmniVoice 自动语言和 CosyVoice3 跨语言模式已调整；IndexTTS 暴露时长倍率。各项源码变更及测试/真实验收状态以 `complete-backend-text-parity` 和宿主 `backend-owned-speech-text` 变更为准；下列仍保留初审基线，不应误认作当前工作区全部未修复。

## 优先核验

1. **宿主跨后端文本改写**：CastGlean `chapter_audio/execution.rs` 所有章节合成均经过 Kokoro 导向的 `execution_text`：`[`/`]` 变全角括号、`·` 变空格、单引号及换行被改写。这可能破坏其他后端支持的控制或读音标记；原文没有改，但送入引擎的文本变了。应逐后端定义可验证的规范化和覆盖测试，不对用户原文中的标记默认解释为控制命令。
2. **Qwen 自动语言**：`crates/talechime-backends/src/qwen/runtime.rs` 缺省/`auto` 遇任意汉字选中文，否则选英文；日语假名、韩文和多种其他语言不会自动选择自己的语言，含汉字的日文也会被选为中文。参数目录明确称启发式，但与[官方 auto](https://github.com/QwenLM/Qwen3-TTS) 不同。显式设置语言可绕过该分支；需核对官方模型前缀语义和 CastGlean 是否能传参，并做真实多语言听验。
3. **OmniVoice 默认语言**：`crates/talechime-backends/src/omnivoice/params.rs` 未传 `language` 时固定 `zh`；`design.rs` 的设计路径也固定 `zh`。支持显式语言并校验；非中文缺省输入可能选错条件，实际效果未测。不能以官方[模型卡](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md) 的 646 种语言宣传代替本地实际路径验收。
4. **IndexTTS 2.5**：`crates/indextts/SOURCE.md` 明确未移植官方 wetext/TensorFlow TN、日语 G2P 和西/日规范化，`front.rs` 仅近似中英数字日期；`crates/talechime-backends/src/indextts/params.rs` 只允许 `zh`/`en` 且默认 `zh`。官方[模型卡](https://huggingface.co/IndexTeam/IndexTTS-2.5) 标注五种语言和语速/读音控制。未移植的不全是当前公开能力的实现错误，但可能造成读法差异；当前不是端到端等价。低层 tokenizer `byte_pair_encode` 对缺失 token 使用 `filter_map`；固定资源是否保证所有字节需再核验，不应据此声称现网一定漏字。
5. **CosyVoice3**：计算库存在 `CrossLingual` 模式（`crates/cosyvoice3/src/pipeline.rs`），产品适配器 `runtime.rs` 只选择 ZeroShot/Instruct，未暴露该模式；[官方固定版本前端](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/frontend.py) 的文本处理与本地直接 tokenizer 的边界需逐例对照。官方说明模型本身支持数字和特殊文本处理，因此不能简单认定缺少 wetext 即必然漏读。当前数值对照存在历史未通过项，短句生成不等于完整听感验收（`docs/cosyvoice3.md`）。
6. **MOSS Nano / Realtime**：只实现两个产品档位，其他官方家族模型不是当前档位的缺陷；适配器未开放 style/instruction/language/读音。`moss/text.rs` 会清理装饰行、移除部分格式字符并按 CJK 拼接跨行文本，属于产品策略，需针对原文内容与官方行为做对照，不能把音频成功当作朗读覆盖证明。历史长文 segment seam 失败须对当前代码复测，不能直接认定仍未修复。
7. **VoxCPM2**：固定参考克隆、风格、声音设计和原生流式已有；不能只凭 README 判定漏功能。量化 Q8 与官方原始权重不是同一数值路径，严格数值和听感验收仍有未完成项（`crates/voxcpm/README.md`）；跨语言、特殊符号、长文及停止边界需明确实测。`runtime.rs` 已把 Truncated 转成错误，不能把其描述为静默成功。
8. **Kokoro**：本次离线中英混读已接入；英文 OOV 使用非官方神经回退，官方混读音素对照、人工听感及其他语言没有完成验收。已完成公开样章 CPU 合成不等于朗读正确。

## 建议验证顺序

先做每个已开放后端的端到端公开文本矩阵：中英夹杂、日/韩、英文缩写和品牌、金额/日期/百分数、特殊符号、数字型号、长词/跨单元、控制标记、未知字符；分别记录原始范围、执行文本、token/音素、错误与实际声音。明确模型/权重 revision、设备、feature、参考音色，并区分静态代码证据、官方预处理对照、真实生成及人工听验。随后按结果修复宿主文本边界和语言选择，再决定哪些官方可选能力值得接入。续跑需纳入规范化/引擎版本兼容策略并保留旧产物；设计前不自动改写用户任务。

## IndexTTS WeText 候选探针（2026-11-11）

`wetext-rs` 0.1.2（Apache-2.0）在隔离临时工程读取 Python `wetext` 0.1.0 wheel 内自带的中英文 FST，与 Python `wetext` 0.1.0 / kaldifst 1.9.0 对同三条公开文本的输出一致：中文日期和小数金额、`S600` 型号与速度、英文金额和日期。Python 3.9 的 namespace 包加载需要在隔离环境补空 `wetext/fsts/__init__.py`；该临时处理不进入产品。此证据仅说明三例可运行并一致，不证明覆盖官方 IndexTTS 的预处理选项、所有语种、FST 来源一致性、性能或跨平台兼容。Python wheel 不含日语 FST。ModelScope 文件列表 API 在本机返回 403，PyPI wheel 可下载；接入前需核验固定版本/许可/摘要及官方实际配置，设计可复现的资源准备与缺失失败路径，不得静默回退至近似规则。

IndexTTS 固定权重 revision `c39ce5ba981572cb187443877ff559dfb246ce63` 的词表实查包含全部 256 个单字节 token，上述静默丢字风险不适用于该完整词表。构造或加载缺失单字节 token 的词表现在会报错，避免受损资源静默丢失 UTF-8 字节；使用官方词表对中英混排、日韩、数字符号、标注及 emoji 的普通文本编码均可还原原始字节。这不证明文本规范化、日语前端或语音输出与官方一致。

## 部分真实声音探针（2026-11-11）

新增 `tools/tts/text-matrix.json` 统一公开用例及 `tools/tts/run_text_matrix.py` 执行器；真实探针已扩展接入各后端，参数通过 `TALECHIME_PROBE_PARAMS` 按协议 JSON 显式传入。执行器保留每条用例、配置、分段、输出或失败状态，`generated` 不代表听验通过；尚未完成全部后端执行。IndexTTS 适配器补充 `ar` 参数转发，参数及语言 embedding 路由测试通过，真实阿拉伯语声音未验。

日语前端追加核实：固定上游 `infer_v2_5.py` 使用 `JapaneseG2PProcessor(g2p_ratio=0)`；`ja_g2p.py` 用 fugashi/MeCab 与 UniDic 分词后空格拼接，此配置不随机改为假名。`nemo_tn.py` 未登记 `ja`，因此日语 TN 实际透传。移植需要匹配分词边界和词典版本，不应新增自创数字日期读法。已开始隔离参考环境，尚未接入 Rust 日语分词器及资源配方。

macOS Apple Silicon CPU，公开文本 `早上好，Alice 说 hello。今日はいい天気です。`：Qwen 0.6B Base revision `5d83992436eae1d760afd27aff78a71d676296fc`、默认自动语言和隔离的系统合成英语参考音频正常结束，输出 3.6 秒 PCM；MOSS Nano revision `44502f80dbf9743528fa921cc544d662c685ebec`、Weiguo 内置音色正常结束，输出 5.2 秒 PCM。同一句中英和日期文本还由 Qwen Base、MOSS Nano、Kokoro g2pW 和 VoxCPM Q8 生成；VoxCPM revision `169f64d8b98bbaab1761e4ca3a83e6af653456cc` 的适配器报告正常结束、4.96 秒音频。音频保存在本机未纳入 Git 的 `.local/parity-audio/`；有效 PCM 和结束事件不证明朗读完整、语言正确或听感合格。其余模型、矩阵条目与人工听验未完成。

共同输入的矩阵应至少覆盖下列类别；“编码通过”与“声音正常结束”分别记录，任何一项都不能替代人工听验或官方对照：

| 输入类别 | IndexTTS 词表字节往返 | 已生成声音 |
| --- | --- | --- |
| 中英混读、人名 | 通过 | Qwen Base、MOSS Nano、Kokoro g2pW、VoxCPM Q8 |
| 日韩 | 通过 | Qwen Base、MOSS Nano、VoxCPM Q8（日语，均待听验）；Kokoro 中文 g2pW 对 `は` 明确报不支持，未生成 |
| 型号、日期、金额、百分比 | 通过 | Qwen Base、MOSS Nano、Kokoro g2pW、VoxCPM Q8（日期，待听验） |
| 方括号、读音标注、emoji | 通过 | Qwen Base、MOSS Nano、VoxCPM Q8 对包含方括号和 emoji 的样例正常结束；MOSS 短文本生成 22 秒，移除方括号和 emoji 后同一数字文本生成 4.4 秒，差异待听验，不能据此确定原因；Kokoro 中文 g2pW 对 emoji 明确报不支持，未生成；读音标注未测 |
| 长词、跨单元、结束边界 | 未测 | 未测 |

用户听验同一条中英和日期文本后反馈：MOSS Nano 漏读 `hello`；Qwen Base 将 `2026` 按英文读；VoxCPM 读作“二零二十六年”；Kokoro 读作“二千零二十六年”。这些样本不通过朗读完整性或中文年份读法验收。Kokoro 的直接原因是原年份规则要求数字紧邻“年”，样本为 `2026 年`；现补充空格边界和回归测试，修复后的声音仍须听验。Nano 固定词表的实际编码及解码包含完整 `Alice` 和 `hello`，因此这次漏读不是输入规范化或词表删除了英文；生成行为的根因尚未证实。Qwen 的参考音频是英文系统语音，可能影响表现但尚无对照证据，不能直接归因。其他模型的生成错误不能仅凭 PCM 和 EOS 判断原因。

用户进一步确认 Nano 单独合成 `hello` 可听见。已用同一模型、音色和种子生成混排标点、英文两侧空格及缩短上下文三条对照；仅验证音频和 EOS，尚未验证有无漏读。不能从单词独立合成成功推断任意混排切分后都能正确发音，也不能把时长作为朗读完整性的证据。宿主当前章节合成不向引擎传递文本语言，单靠出现汉字无法可靠区分中文与无假名的日文日期。
