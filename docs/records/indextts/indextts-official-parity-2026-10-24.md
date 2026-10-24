# IndexTTS 2.5 官方对照（2026-10-24，M4 Pro）

[官方对照 parity goal](../plans/official-parity-optimization-goal-2026-10-24.md)
第 3 项：建立 IndexTTS 首个同机官方对照并判定差距。官方 IndexTTS2
**原生支持 MPS**（device 自动检测含 mps 分支，与 CosyVoice 不同），因此这是
GPU 对 GPU 的对照。

## 一、判定

| 设备 | 官方 PyTorch | 我们 Rust（例程边界 2026-10-24 首测） | 差距 |
| --- | ---: | ---: | --- |
| **MPS / Metal** | **1.42–1.50**（稳态 3 轮 8.0–8.9 s / 5.7–6.1 s 音频；r4 分阶段 gpt 3.72 + s2mel 3.21 + bigvgan 1.00 s） | **2.289** | **落后 ~36–56%，双端均 >1 实时线** |
| CPU | **3.912**（5 轮 3.87–3.96，极稳） | 4.789 | 落后 ~18% |

**结论：IndexTTS 双设备均落后官方，GPU 差距显著（约 1.6×）**。对照成立的
前提是优化跟进——差距方向与 CosyVoice3 组件级差距一致（我们 GPU 路径的
算子/实现落后 PyTorch 后端），但 CosyVoice3 端到端 parity 而 IndexTTS
端到端落后：IndexTTS 官方没有"只能 CPU"的借口，这是 goal 内最明确的
未达标项。

## 二、方法与身份

- 官方：固定检出 `runs/reference/index-tts@d9e41aa`（GitHub clone），
  `uv sync` 独立 venv（torch 2.8.0，MPS 可用）；权重=已就位的
  `runs/indextts-checkpoints/`（pinned HF 下载 + 本地转换同源）；本轮补下载
  `w2v-bert-2.0/preprocessor_config.json`（官方 loader 需要，此前只取了
  model+config 两文件）。
- 直调脚本 `runs/indextts-parity/official_rtf.py`：`IndexTTS2(cfg, dir,
  device)` + `infer(spk_audio_prompt=bench 参考, text=山风句, lang=ZH)`，
  预热 1 + 5 热测，逐轮 WAV 落盘计时（**含 wav 写盘**，比我们的例程边界
  略宽，方向对官方不利，如实登记）；每轮重播种。
- 同语料（corpus-short）同参考（voice_bench reference.wav，官方内部重采样）。
- 我们的数字沿用 [Mac 首测](indextts-macos-first-measurement-2026-10-24.md)
  （crate 例程：单段合成 Instant，含文本正则化不含加载）。

## 三、必须登记的官方 MPS 预热假象

官方 MPS 首轮巨慢：warmup 72.7 s（RTF 12.9）、热轮 1 达 117 s（RTF 19.95）、
热轮 2 19.8 s（3.62），热轮 3–5 进入稳态 8.0–8.9 s（RTF 1.42–1.50）。
模式与 MPS 惰性编译/统一内存逐级驻留一致；对照采用稳态轮并在表中披露全部
轮次（`runs/indextts-parity/official-{mps,cpu}.json`）。我们的例程边界没有
对应稳态问题（首测 4 轮 2.26–2.33 稳定）。

## 四、分阶段 profiling（同日，`INDEXTTS_PROFILE=1`，Metal 3 轮）

| 阶段 | 我们 Metal（3 轮范围） | 官方 MPS（r4 自打印） | 差距 |
| --- | ---: | ---: | --- |
| GPT 循环 | 3.02–3.34 s | 3.72 s | **我们更快** |
| CFM/s2mel 合计 | 6.42–6.96 s（cfm.estimator 6.13–6.66 + codec 0.26） | 3.21 s | 慢 ~2× |
| **BigVGAN** | **7.24–7.72 s**（fir_down 2.11–2.24 + fir_up 2.04–2.15 + res_conv 1.46–1.62 + snake 0.72 + up_conv 0.83–0.90） | **1.00 s** | **慢 ~7×，最大缺口** |

**优化路线图（按收益排序）：BigVGAN FIR 卷积（~4.2 s）>> CFM estimator
（~3.3 s 差距）>> GPT（已领先，不动）**。与 Windows 轮经验一致（卷积是
Candle Metal 弱项；codec 小张量卷积 CPU 更快的反常当时已实测）。原始 profile
JSON 在 `target/perf/itb-profile-metal/`。

### 4.1 FIR 路线定性（同日离线探针，ignored 测试 `fir_metal_conv_vs_tap_timing`）

真实六级形状（1536×492 … 48×62976）上对比现役批量 conv FIR 与逐 tap 参照：
**两者在 Metal 上几乎同速**（逐 tap 相对 conv 0.76–1.38×，大形状略快），
数值差 ≤ 9.5e-7。结论：candle Metal 的 conv1d（im2col 12× 物化 + 批量小
GEMM + 转置拷贝，`metal_backend/mod.rs:911`）与 12 次 strided elementwise
遍历**都在各自的地板上**，距带宽下限还有 ~10–30×；纯 candle 算子重写无法
关上 FIR 的 7× 缺口。**下一步是 candle fork 内的融合 FIR Metal kernel**
（每线程一个输出、K=12 MAC 单遍，fork 已有 metal-sdpa 先例）：预期 FIR 段
4.2 s → ~0.3–0.5 s，BigVGAN 7.5 s → ~3.7 s。res_conv/up_conv 的 conv 项
（2.3 s）与 CFM（6.5 s）是后续两档。

## 五、对 goal 的动作

- 第 3 项"建对照"完成；"若慢于官方则优化跟进"转为主线：**IndexTTS Metal
  优化立项**，profiling 已完成（§四：BigVGAN FIR >> CFM >> GPT 已领先），
  候选方向沿用 Windows 轮已验证的经验（卷积路径/tile/布局），按 ABBA/
  输出保持门禁执行。
- 值得注意：官方 GPU 稳态也 >1（1.42）——"超过官方"与"进入实时"在此
  是同一条线（把 Metal 压到 <1 即同时达成两者）。BigVGAN 一项若压到
  官方水平，总时长 17.5→11 s（RTF ~1.9）；再加 CFM 对齐则 ~7.8 s
  （RTF ~1.35）；进入实时还需要更多（GPT 已无余量可让）。

## 六、原始数据

`runs/indextts-parity/`：`official-{mps,cpu}.json`（逐轮+阶段）、
`official-*.wav`（逐轮输出）、脚本本体；`/tmp/itb-*.log` 为终端日志副本
（含官方自身打印的 gpt_gen_time/s2mel_time/bigvgan_time）。
