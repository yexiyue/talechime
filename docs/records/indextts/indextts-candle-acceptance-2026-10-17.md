# IndexTTS 2.5 Candle 移植验收记录与问题台账

日期：2026-10-17（前两轮历史记录）。关联：[移植过程记录](indextts-candle-port.md)、[独立生成计划](../plans/independent-tts-candle-plan.md)、`crates/indextts/SOURCE.md`（固定来源）。

> 同日后续更正：历史语料实际为 6 段 × 2 轮，10/12 成功，原文的 20/22 统计错误；seg2 早停已定位到采样候选丢失及 scorer 误判完成并修复。首步 logits 测试此前未比较输出，本轮已补足。后续实机结果、与官方及社区比较见[对照复核记录](indextts-official-community-review-2026-10-17.md)。以下保留历史台账，不能把历史猜测当作当前结论。用户试听仍未执行。

## 1. 验收环境

| 项 | 值 |
| --- | --- |
| 硬件 | RTX 5070 12GB（本轮 CUDA 验证）；CPU 对照同机 |
| 工具链 | Rust（workspace 统一 candle 0.11.0）、CUDA 13.0（`D:\dev-tools\cuda\v13.0`）、cuDNN 9.14（`D:\dev-tools\cudnn\v9.14-cuda13`）、MSVC 2022 |
| Python 对照 | `D:\workspace\index-tts`（commit `d9e41aa`）+ venv：torch 2.8.0+cu128、transformers 4.52.1 |
| 权重 | 官方 `checkpoints/`（IndexTeam/IndexTTS-2.5 `c39ce5ba` 等，见 SOURCE.md）；Rust 侧 `checkpoints-converted/`（safetensors，键名原样） |
| 数值 oracle | `target/indextts-oracle/oracle.json`（`tools/tts/indextts_reference.py` 产出）+ 分层 dump（campplus/emo/regulator 各一份） |

## 2. 验收结果总表

| 验收项 | 结果 | 数据 |
| --- | --- | --- |
| 无模型单测 | ✅ 92 通过 / 0 失败 | tokenizer（语言表/special 布局/BPE/预切分）、config、dsp、front、model 及各模块 VarMap 形状测试 |
| Clippy / rustfmt / workspace | ✅ | `clippy -p indextts --all-targets -- -D warnings` 零告警；fmt 全仓；workspace 编译通过 |
| 真权重 + oracle（`#[ignore]`） | ✅ 16/16 | tokenizer **逐位一致**（4 语料）；w2v.hidden_17 1.56e-4（12.4 万值）；w2v.normalized 1.02e-4；gpt.emovec 5.96e-7；gpt.embeds 9.54e-7；campplus.style 5.34e-6；mel.ref_mel 6.96e-5；codec codes 0/61 错配、decoded 6.91e-6；regulator.prompt_condition 4.05e-6 |
| 端到端 CPU 探针 | ✅ | 5.75s 音频 / 生成 66.9s，RTF 11.6（seed 42；修复前为 26.1 且音色异常） |
| 端到端 CUDA 探针 | ✅ | 5.43s 音频 / 生成 3.88s，**RTF 0.71**（实时）；加载 2.3s、参考 2.7s |
| 端到端音频特征 vs Python | ✅ | 过零率 0.140 vs 0.145、频谱质心 2317 vs 2358 Hz、RMS 0.0995 vs 0.0950（修复前过零率 0.327） |
| 语料轮次（11 段 × 2 轮，GPU，seed 7） | ⚠️ 20/22 段 | RTF 0.59–1.17；逐段时长 vs Python CPU 基线（RTF 6.49）：2.03/2.00、3.99/4.03、6.07/5.67、4.03/3.55、26.28/24.44 秒；seg2 两轮均失败（见 §4-A） |
| 试听门槛 | ❌ 未执行 | 待用户对 `target/tts-integration/indextts/` vs `D:\workspace\index-tts\runs\py-corpus\` 同文本 A/B |

复现命令（Git Bash）：

```sh
# 单测 + oracle（需真实权重与 oracle JSON）
INDEXTTS_CONVERTED_DIR=/d/workspace/index-tts/checkpoints-converted \
INDEXTTS_ORACLE_JSON=/d/workspace/talechime/target/indextts-oracle/oracle.json \
INDEXTTS_TIKTOKEN_FILE=/d/workspace/index-tts/checkpoints/multilingual_zh_ja_yue_char_del.tiktoken \
cargo test -p indextts --lib -- --ignored --nocapture

# CUDA 构建（Git Bash 必须带 MSYS2_ARG_CONV_EXCL，否则 nvcc 参数被路径转换破坏）
MSYS2_ARG_CONV_EXCL="*" CUDNN_LIB_PATH='D:\dev-tools\cudnn\v9.14-cuda13\lib\x64' \
LIB='D:\dev-tools\cudnn\v9.14-cuda13\lib\x64' \
cargo build --release -p indextts --features cuda-cudnn --example generate --example corpus

# GPU 探针 / 语料（运行时 PATH 需含 cudnn bin）
PATH="/d/dev-tools/cudnn/v9.14-cuda13/bin:$PATH" ./target/release/examples/generate \
    --dir /d/workspace/index-tts/checkpoints-converted \
    --reference /d/workspace/index-tts/examples/voice_01.wav \
    --text "测试文本" --lang zh --out out.wav --seed 42 --device cuda
```

## 3. 已修复 bug 台账

按发现轮次与类别列出；「数值影响」列为修复前 → 修复后的对 Python 偏差。

### 3.1 语义移植错误（真权重/oracle 暴露）

| # | 位置 | 根因 | 修复 | 数值影响 |
| --- | --- | --- | --- | --- |
| B01 | `campplus.rs` `out_nonlinear` | Python `DTDNN.py:97` 用 `config_str='batchnorm-relu'`：affine BN **+ ReLU**；移植版用 eval-only BN 且无 ReLU，负值泄入统计池化 | `bn_affine` + `forward` 补 `.relu()` | 0.37 → 5.3e-6 |
| B02 | `conformer.rs` 位置编码 | Python `RelPositionalEncoding` 把 `pe` 注册为 buffer，`load_state_dict` 会用检查点里的**低精度表**（1/256 量化副本）覆盖解析公式；移植版重算精确公式 | 加载 `pos_enc.pe` buffer，缺失才回退解析式 | emovec 3.9e-3 → 6.0e-7（与 B03 叠加） |
| B03 | `perceiver.rs` FF | Python `FeedForward` 是带偏置 `nn.Linear`（检查点有 `layers.d.1.{0,2}.bias`）；移植版 `linear_b(..., false)` 丢偏置 | 换 `candle_nn::linear` | 同上 |
| B04 | `length_regulator.rs` Mish | 手写 softplus `y.exp()?.affine(1.0, 0.0)?.log()`：`affine(mul,add)` 是 `x*mul+add`，少了 `+1`，Mish 退化为 `y·tanh(y)` | 改 `candle_nn::ops::mish` | 0.0757 → 4.1e-6 |
| B05 | `gpt.rs` `get_emovec` | 上游调用方与函数体内各有一次 `.transpose(1,2)` 相互抵消；移植版照搬了内层那次，把 (F,T) 当 (T,F) 喂给 conformer（真权重下 matmul 形状直接报错） | 删除多余 transpose | 阻断级 |
| B06 | `dit.rs` 注意力/FFN | 检查点的 `wqkv/wo/w1/w2/w3` 均无 bias（llama 风格）；移植版按带 bias 加载 | `linear_no_bias` | 阻断级（加载失败） |
| B07 | `generation.rs` mel 位置 | HF `kv_cache=True` 字面公式使第 i 个生成 token 取位置 **i+1**（位置 1 永不使用；start 取 0）；初版按 i 实现 | 按 Python 字面语义实现（与 accel 引擎的 i 不同，以 oracle 路径为准） | 首步之后全偏 |
| B08 | `generation.rs` beam scorer | 一次采样中 eos 候选数 > num_beams 时移植版直接 `bail!`；HF 语义是取前 num_beams 个入假设、该批完成并填充 | 按 HF 语义修复 | 特定条件下阻断（CUDA 探针首次触发） |

### 3.2 权重名映射（真权重加载暴露）

| # | 位置 | 根因 | 修复 |
| --- | --- | --- | --- |
| B09 | `conformer.rs` 子采样 | 检查点是 Sequential 索引名 `embed.conv.0.*` / `embed.out.0.*`；移植版用 `embed.conv.*` | 补 `.pp("0")` |
| B10 | `campplus.rs` BN（多处） | `get_nonlinear` 返回包了 `.batchnorm` 子模块的 Sequential；`nonlinear{1,2}`/transit 的 `nonlinear`/`out_nonlinear` 四类键名都差这一层 | 逐一补 `.pp("batchnorm")` |
| B11 | `wavenet.rs` weight-norm | torch 存 `weight_g` 为 `(out,1,1)`/`(out,1)` 带尾随单位维；移植版按 `(out,)` 加载 | 双形状兼容后 flatten |
| B12 | `model.rs` w2v 归一化 | candle 的 `/` 不广播，`(1,T,1024) / (1024)` 报形状错 | `broadcast_div` |
| B13 | 导出脚本 `s2mel.pth` | 官方结构是 `state['net']` 按 ModuleDict 名分段（cfm/length_regulator/gpt_layer 各一扁平子表），非 'model' 包裹 | 导出时展平为 `models.<name>.*`；bigvgan 的 'generator' 包裹同理 |

### 3.3 candle CUDA 路径限制（GPU 首跑暴露；修复后 oracle 全绿复核，无数值影响）

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| B14 | candle conv 回退 | 无 cudnn 时 conv 走 unfold+matmul 回退，拒绝部分跨步布局（`DriverError`/contiguity 报错） | 新增加法式 `cuda-cudnn` feature（`tts-candle-platform`/indextts 两层；普通 `cuda` 行为不变） |
| B15 | `length_regulator.rs` k=1 卷积 | candle CUDA 对 k=1 Conv1d 特例化为 matmul，拒绝上游跨步视图 | `out_proj_forward` 显式连续 2D matmul（数值等价） |
| B16 | `dit.rs` 全部 3D×2D Linear | 同类：3D lhs × 2D 权重在 CUDA 内部生成跨步视图后自拒（merge/AdaLN 投影/wqkv/wo/FFN/跳连/res_projection/final_linear/cond_projection） | 新增 `lin2d` 辅助统一改为显式连续 2D matmul |
| B17 | 构建环境 | Git Bash 把含空格路径拆进 nvcc 参数（`nvcc fatal: Don't know what to do with 'Files/Git/MD'`）；cuDNN 无链接路径（LNK1181 cudnn.lib） | `MSYS2_ARG_CONV_EXCL="*"`；`CUDNN_LIB_PATH`/`LIB` 指向 cudnn lib，运行 PATH 加 cudnn bin |

### 3.4 健壮性（语料轮次暴露）

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| B18 | `model.rs` 生成 | 采样可能首 token 即停（或极短，<8 码）：空/微小张量进 CUDA kernel 报 `CUDA_ERROR_INVALID_VALUE` | 空码重抽守卫（≤3 次、阈值 ≥8 码），探针改逐段容错记录 |

### 3.5 工具/测试自身缺陷（顺带修复）

validation.rs oracle 路径多处多带 `"data"` 段、tokenizer ids 按字符串索引数组、campplus fbank 少 unsqueeze、mel shape 路径错；gpt smoke 测试输入张量方向写反；示例 `write_wav` 构造错误。

## 4. 遗留问题与风险（开放）

### A. seg2 生成异常早停（开放缺陷，优先级最高）

- 现象：语料 seg2（原文「今天是2026年10月7日，车票售价128元…」，正则化后「今天是二零二六年十月七日,车票售价一百二十八元,…」36 tokens）生成分布异常偏早停——同 seed 单跑仅 0.72s 音频，**Python 用完全相同的正则化文本生成 6.55s**；语料中连续两轮失败（重抽 3 次仍极短后触发 B18 守卫外的驱动错误路径）。
- 特征：文本相关、确定性复现；其余 10 段（9–148 tokens）全部正常且时长对齐。该段 token 序列含大量短重复 token（「二」「零」等多次出现）。
- 隔离结论：非前端问题（正则化文本与 token 数已核对）、非参考条件（其他段共享同一参考正常）、非数值精度（oracle 全绿）。
- 疑点：`generation.rs` 的 repetition_penalty / warper 链或 beam scorer 对该 token 模式的边界行为与 HF 有差异。
- 下一步：对同文本做逐步 logits 分层 dump（Python `indextts_reference.py` 扩展 vs Rust 临时标记），比对第一步及停止前若干步的 top-k 分布。

### B. 文档化边界（非 bug，验收对照时须排除）

1. **文本正则化近似**：上游 wetext/TensorFlow-TN 未移植；`front.rs` 实现中英基数词/小数/百分比/日期(年)子集，**非逐字节可比**（seg2 的 Python 侧文本即来自 wetext，需像 §4-A 那样固定同一份文本对照）。
2. **重采样**：自实现 Kaiser sinc，非 torchaudio 逐位一致；oracle 全部从重采样后的固定 PCM 输入开始比对。
3. **采样 RNG**：torch 与 Rust StdRng 无法逐位对齐；同 seed 只保证可复现、不保证与 Python 同轨迹；GPU 上 cuBLAS/cuDNN 归约顺序还引入非确定性（同 seed 两次运行 codes 可不同）。
4. **流式模式**（`stream_return`/`quick_streaming_tokens`）、QwenEmotion 文本情绪 LLM、ja/es（NeMo）、说话人 conformer 条件（检查点缺失）：均未移植（见 SOURCE.md）。
5. **Metal / CPU 长会话 / 显存峰值**：未验证；CUDA 显存未精确计量（探针运行时 ~1.5–3GB 量级，未采样存档）。
6. **哈希绑定的参考编码缓存**：计划阶段三的「与资源哈希绑定的缓存」目前仅为进程内 `Option<ReferenceConditioning>`（无哈希键/持久化/陈旧保护）——延期至阶段四适配器层（VoiceStore 已有 per-revision 目录惯例）。
7. **ONNX 导出脚本**（`tools/tts/indextts_export_onnx.py`）：本机 venv 无 onnx/onnxruntime，未执行过；作为对照路径首验前不得引用其产物。

### C. 验收状态判定

- 数值验收：**通过**（§2 表；阈值未放宽）。
- 听感/产品验收：**未通过也未失败——未执行**；在 §4-A 修复并完成用户 A/B 试听前，不得宣称移植完成，也不得接入 `talechime-backends`（阶段四）。
