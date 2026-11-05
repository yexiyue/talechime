# 原生资源准备核验（进行中）

日期：2026-11-05。变更 `prepare-native-model-resources`，中文运行数据包已公开发布，SDK 原生配方在本地接入中，尚未发布 SDK 或应用。本记录只描述实际执行结果，不将文件头、模型卡或编译当作完整模型验收。

## Kokoro 社区 F32 与官方基线

固定输入：

- 社区 `1038lab/Kokoro-82M-zh-MLX@4bd6c9644da381fee105f37fcd8cb63d038ba7e8`，`model.safetensors` 327115152 字节，SHA-256 `c3e1448dbd823dc54bb950a4f0650975c2385e789e2684fea37c93185bc104a0`。
- 官方 `hexgrad/Kokoro-82M-v1.1-zh@01e7505bd6a7a2ac4975463114c3a7650a9f7218`，`kokoro-v1_1-zh.pth` 327247856 字节，SHA-256 `b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b`，与 SOURCE.md 固定摘要相同。
- 官方配置原文件摘要 `bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890`；社区配置序列化不同，但解析后的全部 JSON 字段相等。

隔离开发环境使用 PyTorch 2.7.1、safetensors 0.8.0、CPU，`torch.load(weights_only=True)` 展开官方组件和 `module.` 前缀后，548 个张量与社区文件逐个比较：名称无缺失/新增，dtype、shape、全部数值精确相等（548/548）。社区文件保留 weight normalization 的 g/v，不是现有加载器直接消费的折叠结果；仍须原生折叠及声学对照。

代表中文音色 `zf_001`、`zm_010`：官方 `.pt` 与社区 `voice` 张量均为 F32 `[510,1,256]`，逐元素精确相等，最大绝对差为 0。社区两个文件摘要依次为 `19aba9f273067860f5e490bcbbefd6e531b462109413432b80d9d15c53c05748`、`2a6f163f8e32cb8223437419195039056d7840b3a86d033b496e14ccc345610b`。其他音色未由这两个结果替代，完整配方前仍需核验所有纳入的音色。

来源声明：官方模型及 Kokoro/Misaki 代码为 Apache-2.0，固定来源和版权见 `crates/kokoro/SOURCE.md` 及 LICENSE；社区声称 Apache-2.0。已比对的模型数值与官方完全一致，分发时仍须保留官方来源/许可证，社区声明不替代上游许可。

开发下载最初通过裸 HTTP 读到了提前结束的 200 响应（仅 20058385 字节），摘要不符，因此拒绝作为基线；改用 HF 客户端共享缓存取得完整文件，并重新核验上述大小与摘要。损坏诊断输入未进入运行时目录，不将“HTTP 成功”视为文件完整。

原始权重、音色和详细 JSON 输出在 Git 外的审计缓存中；没有写入正式音色库。Kokoro 保留 Candle 声学路径。此项仅证明可复用原始权重/所列音色，尚未证明原生准备、中文前端、声学合成或听感验收完成。

## 共享准备边界（尚未接入模型）

已新增固定配方计划、缓存/摘要核验、阶段事件、受控 worker、取消/发布线性化边界及暂存发布基础。测试覆盖计划只读、缓存命中/损坏、空间算术溢出、转换取消等待退出、成功阶段不误取消后续阶段、忙锁、旧版本保留、校验失败不发布和受管目录符号链接拒绝。三类后端的实际配方尚未登记，公开计划查询对它们暂返回无可执行配方；不得据此声称已支持一键准备。

## Kokoro 全部中文音色补充

已对 BUILTIN_VOICES_JSON 中全部 100 个 `zf_*` / `zm_*` 声音重复执行官方 `.pt` 原始摘要校验，并与同一社区 revision 的 `voice` 比较：100/100 的 dtype、shape 和全部元素精确相等。没有将未核验的英语音色纳入中文配方。逐文件摘要记录在 Git 外的 `kokoro-voices-comparison.json`，后续配方可使用这些固定结果。

## CosyVoice3 四件复用与差异

已实际下载并核验社区固定 revision 的 LLM、Flow、campplus.onnx、speech_tokenizer_v3.onnx，四个完整文件的 SHA-256 与现有 resources.json 一致；未仅采用远端元数据。其 HiFT 246 个张量与官方 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512@29e01c4e8d000f4bcd70751be16fa94bf3d85a18` 的 hift.pt（SHA-256 `b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65`）折叠结果无名称缺失/新增，但 82 个张量的精度或数值不相等。

使用固定 PyTorch 2.7.1 和历史转换规则：F0 先转 F64，再调用 `_weight_norm`，其余使用 F32 融合。重建文件 96324868 字节、SHA-256 `475ac02be5d85f5b13da1e075a217ddb71e14f109f327919609004858c75bd2b`，与当前已核验基线完全相同。固定官方源码 `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` 的 CausalConvRNNF0Predictor/CausalConv1d 在 CPU、`linspace(-1,1)` F64 `[1,80,17]` 上，对照原始参数化网络：上述 F64 重建最大输出差为 0；社区折叠结果转成 F64 后仍有最大绝对差 0.0016396911555602856。不能以简单转 dtype 代替转换。

社区 tokenizer 只有 21 个 added tokens，缺少官方 special token 列表中的 260 项。固定源码给出 280 个 additional special tokens；开发环境用 Transformers 4.51.3、tokenizers 0.21.4 从官方 vocab/merges/config 重建后得到 10573953 字节、SHA-256 `caa0dadcb8c40c79a3244aaf290492e298a49c5c25e0db16cad12f2f8ff639c5`，与当前基线相同。除 added_tokens 外，重建图与社区 JSON 所有字段精确相等，因此可复用基图再原生补齐。ID：endofprompt=151646、[AA]=151664、[ǜ]=151923。

以上为开发环境对照，不是应用运行时调用 Python 的实现。HiFT、完整 tokenizer、配置已在本地实现原生准备，完整配方在经核验缓存上约 64.79 秒完成并复核 Ready；尚未把组件对照当作整链音频或旧 CosyVoice3 数值门禁通过。

## Kokoro 中文前端公开数据包

公开仓库 `yexiyue/talechime-resources`，固定提交 `8eb001f5498e099e6ac929454e7fb28e457bf035`，无 gated 访问审批。经独立 `token=False` 客户端、隔离缓存逐一下载 10 个文件并验证大小与 SHA-256，共 17,543,422 字节。复核 API 返回 `private=false`、`gated=false`。这是数据资源发布，不是 SDK 或应用发布。

数据位于 `kokoro/misaki-zh-v1/`，包含 453,885 条词条、1,654 种音节、jieba 字典及 POS-HMM、规则、独立测试语料、来源/校验清单与 5 份完整许可证。固定 Misaki `fba1236595f2d2bf21d414ba6e57d25256afada3`；依赖为 jieba 0.42.1、pypinyin 0.55.0、pypinyin-dict 0.9.0、cn2an 0.5.24。生成记录仅含模块相对路径，不上传本机路径、凭据、用户音频或小说。

开发生成使用已有 `export_frontend.py`（新增固定 jieba 许可证输入），独立校验使用 `cargo run -p talechime-kokoro --example frontend_compare`，再经 `package_frontend.py` 检查版本、数据摘要和完整对照记录后形成上传目录。29 条语料中 28 条与固定 Misaki 音素串一致，1 条嵌入英文明确不支持，0 失败；覆盖多音字、数字、全角数字、变调和儿化。终端消费者仅下载数据，Rust `Frontend` 不调用 Python；Python 仅用于开发导出。

许可证保留 Misaki/PaddleSpeech Apache-2.0、jieba/pypinyin/pypinyin-dict/cn2an MIT。jieba 许可证来自官方 v0.42.1，SHA-256 `18ba0984839f85853b29fadaf992f7dba8fd0ca0fbeae34de2b8735222dc7a37`。已通过 Hugging Face CLI 上传，专用细粒度 Token 仅有该资源仓库写权限，存于标准本机 HF 凭据位置，不写入代码、命令参数或日志。首次网络 TLS 中断后使用现有本机代理重试成功，全程保留 TLS 校验。

## Kokoro 原生折叠权重与声学对照

社区 548 个原始 F32 张量原生折叠后，补充固定官方 `AdaIN1d` 所声明的 70 个 `InstanceNorm1d(affine=True)` 的确定性初始化，共 140 个未训练常量（unit weight / zero bias）。这些常量不在原始 checkpoint 中，但存在于官方实例化模型；名称和维度编入固定清单，不为未知缺失参数填默认值。完整输出为 599 个 F32 张量，与固定官方 PyTorch 模型折叠导出逐位一致。

Rust 输出 327,178,648 字节，SHA-256 `41ed7fe28189035125f50721b7d456391bd59f5c31b125da8b1ec111d6317c93`。固定输入、音色与官方独立捕获随机噪声的 `official_compare` 在 CPU、Metal 均通过所有阶段 40 dB SNR 门禁及精确时长门禁，输出 81,000 样本/24 kHz。各预热后 1 次计时为 CPU 3.9949 s、Metal 1.1500 s；这是单样本观测，不表示完整性能评估或听感已通过。Metal 路径保留既有 CPU 编码/韵律/频谱及所选设备解码器划分。未切换 ONNX 声学推理。

## g2pW 数据、原生解包与双变体复用

公开上下文表包 `kokoro/g2pw-context-v1/` 固定到 `yexiyue/talechime-resources@ead7e24684eaea42af3dc0fec0bc5c7d9c51bfdf`。6 个文件共 170,575 字节，隔离缓存中全部匿名下载、逐文件摘要与大小验证通过。规则表仅含数据；完整 Apache-2.0 许可证、BERT 来源、官方对照记录单独保留。g2pW ONNX 权重不在该仓库重发，原生准备从固定官方 ZIP 中只解出指定成员，以固定文件大小和 SHA 校验；BERT vocab 从固定官方仓库复用。

固定官方 source、archive、BERT revision 见 `crates/kokoro/SOURCE.md`。开发环境 ONNX Runtime 1.22.1 生成独立输入/输出 fixture；Rust `context_compare` 比较全部 24 例 tokenizer 输入、位置/音素掩码和概率：24/24 官方等价，浮点概率误差门限 1e-5，离散输入完全相等。人工目标读音 23/24，官方与 Rust 保留同一处错误；6 例相较普通前端改变读音，不声称上下文模型永远正确。

SDK 同时登记普通与 g2pW 原生配方。g2pW 目录 121 个固定运行文件，共 1,032,463,384 字节（另有发布标记），原生准备在已验证输入上约 24.70 秒完成并复核 Ready。首次真实解包暴露 ZIP feature 未启用 deflate，已限定在 `kokoro-g2pw` 启用 Rust zlib-rs 解码；重试复用此前完成且已验证的组件，不运行 Python。普通目录为 114 个固定文件，共 396,981,297 字节。共享完成组件以输出 SHA 寻址，仅在字节校验通过后建立硬链接；实测两个变体的 `model.safetensors` 位于同一设备、同一 inode。旧普通目录移至隔离审计备份后再次准备也成功，未丢弃旧证据。

真实 SDK 加载与短文本合成（固定 zf_001、seed=7，24 kHz 单声道，81,000 个有限样本，RMS 约 0.051887）：

| 模型 | 设备 | 加载时间 | 加载+合成总时间 | WAV | 听感 |
| --- | --- | --- | --- | --- | --- |
| 1.1-zh | CPU | 2.779 s | 7.189 s | 有效 | 未评估 |
| 1.1-zh | Metal | 2.633 s | 4.128 s | 有效 | 未评估 |
| 1.1-zh-g2pw | CPU | 4.360 s | 8.972 s | 有效 | 未评估 |
| 1.1-zh-g2pw | Metal | 4.115 s | 5.678 s | 有效 | 未评估 |

Metal 只表示既有声学后端的实际设备划分，g2pW 仍在 ORT CPU。上述结果不包括 Tauri 中确认→准备→试听验收，完整变更与 `complete-voice-center` 的剩余验收继续保留。

## CosyVoice3 固定参考与真实后端补充

使用原有 `runs/cv3-components` 和 `runs/cv3-real-ort124` 官方参考，所有新输出写入独立副本，未覆盖旧 fixture/摘要。固定参考为官方 `CosyVoice/asset/zero_shot_prompt.wav`；组件对照通过，ORT 前端的 speech tokens 完全一致，prompt mel 最大差 6.485e-5、speaker embedding 最大差 2.205e-5，均通过既有门禁。

完整真实权重对照仍有历史数值边界，未放宽：Metal 的 LLM、time/input embedding、estimator、flow mel、共享官方 mel 的 HiFT 全通过，tokens→PCM 最大差 0.39034 超过 0.01；CPU estimator 最大差 0.003486 超过 0.003，tokens→PCM 最大差 0.16024 超过 0.01，其余项通过。CPU/Metal 共享 mel 的 HiFT 最大差分别为 1.293e-5 / 3.330e-4，均低于 0.003。转换后的 246 个 HiFT 张量仍与官方 F64/F32 导出逐字节一致，不能把成功生成 WAV 当成端到端逐样本对照通过。

另用已公开生成、准确转写的 Kokoro 参考录音，真实 CosyBackend 在无 Python 命令的 PATH 下合成“你好，欢迎回来。”：Metal 加载 28.26 秒、总计 34.85 秒；CPU 加载 66.13 秒、总计 150.30 秒。均为 24 kHz 单声道、47,040 样本/1,960 ms 的有效 WAV，RMS 分别 0.069132 / 0.069134。未做人工听感；耗时来自开发构建及并行本地工作，不作为性能基准。

## IndexTTS 完整准备与后端补充

原生 F32 配方 14 输入、13 输出，运行资源 6,676,865,606 字节；在已核验输入缓存下完整准备 173.89 秒，额外空间计划为输出加 512 MiB 工作预算（7,213,736,518 字节），此为保守预算，不宣称已采样测得实际峰值。原始输入下载、组件缓存与硬链接另按当前卷情况核算。mean/sqrt(var)、说话人/情绪矩阵与 5 个权重组件均包含于 2,707 张量逐字节核验。

无 Python 命令的 PATH 下，真实 IndexBackend 使用相同生成参考合成短句：Metal 加载 26.83 秒、总计 69.47 秒；CPU 加载 34.95 秒、总计 222.00 秒。两者输出均为 22.05 kHz 单声道、37,632 样本/1,707 ms 的有效 WAV，RMS 为 0.063033 / 0.067459。未做人工听感，未声明跨设备逐样本一致或跨平台验收。
