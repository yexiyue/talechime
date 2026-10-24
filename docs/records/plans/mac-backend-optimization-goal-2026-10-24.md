# Mac 后端优化 goal（2026-10-24 起）

> **状态：已完成（2026-10-24 凌晨，§五 0–7 全勾）**，由
> [官方对照 parity goal](official-parity-optimization-goal-2026-10-24.md)
> 接续。本文件保留为状态记录，勾选与登记不再更新。

本文件是 issue #5 Mac 侧（Apple M4 Pro）的执行 goal 与工作约定，不替代 issue
#5 的验收约束；两者冲突时先在 issue 评论里说明再执行。范围：MOSS、OmniVoice、
IndexTTS 三个尚无 Mac 优化的后端为主，兼 Qwen 变体补测、CosyVoice3 最终路径
复测与 VoxCPM 回归。Windows 侧不在本 goal 内。

## 一、Mac 现状基线（2026-10-24 凌晨更新）

| 后端 | Mac 现状 | 本 goal 内的工作 |
| --- | --- | --- |
| OmniVoice 0.6B | **优化已合入 main**（Metal RTF 0.783，-28%；2026-10-24 `42a7887`） | 已完成；遗留试听/ASR 回读与可选 next-hotspots |
| MOSS | 基线已建：Nano Candle Metal 0.208 / CPU 0.308 / ONNX 0.174 / Realtime 0.854；Local 1.7B OOM 阻塞 | 已完成（已测量无改动）；切分缝隙单独立项 |
| IndexTTS 2.5 | **Mac 首测完成**（例程边界：Metal 2.29 / CPU 4.79；资源已就位） | 首测完成；统一协议被设计性阻塞，优化立项待 profiling |
| Qwen3-TTS | 0.6B Base 优化验收 + **变体矩阵补齐**（1.7B Base 0.992 / 1.7B CV 0.758/0.751 / 0.6B CV 0.836 / CPU 对照） | 已完成 |
| CosyVoice3 0.5B | 热路径优化已验收 + **统一协议复测完成**（Metal 短 1.414 / 长 2.012 / CPU 1.921） | 已完成；流式路径超实时是后续优化方向 |
| VoxCPM2 | Metal tile 优化已保留 | 回归 + CPU 基线（最后一项，进行中） |

## 二、资源与磁盘政策（用户 2026-10-24 授权）

1. **模型统一放 `~/.talechime/resources/`**，走生产 manifest/SHA-256 校验；模型
   永不进仓库（AGENTS.md 约束不变）。下载前核对目标盘余量。
2. 2026-10-24 实测盘点：
   - 系统盘（`~`，模型落点）：52 GiB 可用，resources 已 37 GiB
     （moss 18 / qwen 13 / voxcpm 3.3 / omnivoice 3）。
     **cosyvoice3、indextts 资源缺失**，IndexTTS 还需准备本地转换权重。
   - 仓库卷（`/Volumes/yexiyue`）：**已 100% 满，仅 3.3 GiB 可用**；`target/` 94 GiB
     （debug 61 / asr-spike 12 / release 8.3 / perf-builds 4.1 /
     continuation-evidence 2.6 / voxcpm-official-mps 1.8 / voxcpm-codec-* 各 0.98 /
     tmp 0.6 / perf 0.5）。
3. **空间不够时清理构建缓存**，优先级从高到低：
   1. `target/debug`（纯构建缓存，最大头）；
   2. `target/perf-builds`、`target/tmp`、`target/voxcpm-codec-*` 等实验构建目录；
   3. `target/release` 谨慎：清了要整轮重编译，优先清陈旧子目录；
   4. `target/continuation-evidence`、`target/perf`、`target/asr-spike` 可能是
      records 引用的证据或 spike 环境——**清理前先在 `docs/records/` 里检索引用**，
      被引用的保留或在记录里注明清理。
4. 清理后重建的代价（重编译时间）不算测量开销，但不与正式基准并行。

## 三、CPU 支持政策（用户 2026-10-24 决策）

**CPU 执行很慢的模型不必强行支持和优化。** 工作判定：Metal 可用且 CPU 路径
RTF 明显超出实时（参考线 ≥3，现有数据点：OmniVoice 5.2–5.8、VoxCPM 3.7、
CosyVoice3 优化前 4.3–11.3）时，CPU 只做一轮对照测量（或资源/时间不允许时
标记不适用并记录原因），不投入 CPU 专项优化；产品默认档位指向 Metal。
这是对 issue 正文「分别优化 Metal 与 CPU Accelerate 路径」的执行放宽：
Metal 仍是必测必优化路径，CPU 从「优化对象」降为「对照或登记不适用」，
逐例把决定和理由写进该后端记录。跑不动的 CPU 轮次记阻塞，不勾成已测量。

## 四、candle skill 共同演进

`skills/candle-performance/` 与后端优化同步迭代，让后面的模型少踩坑：

- 每个后端完成一轮优化后，把**实测验证过**的坑与规则沉淀进 skill：新后端可加
  参考文件（如 `references/moss-metal.md`、`references/indextts-metal.md`），
  通用规则并入 `references/metal.md`、`accelerate-amx.md`、`case-studies.md`。
- 已知待沉淀候选：OmniVoice 的 argsort 列数 >1024 Metal 回退 CPU + 主机往返；
  已拒绝的候选写成反面案例。假设未验证前不写进 skill。
- skill 修订沿用其自身验收（案例须有证据链接，不新增运行时开关）。

## 五、执行清单

- [x] 0. 准备（2026-10-24 完成）：仓库卷经并行工作区清理后 75 GiB 可用，
      `target/` 未动（evidence 目录 asr-spike/continuation-evidence/perf 均被
      records 引用，按 §二 政策保留）；资源盘点完成——cosyvoice3 生产格式权重
      已在本机 `runs/cv3-converted/`（待导入 `~/.talechime/resources`），
      **indextts 本机完全没有**（runs/reference 无官方检出，需取源+权重+转换）。
- [x] 1. OmniVoice（2026-10-24 由并行会话完成并合入 main `42a7887`）：profiling
      已做（92–93% 时间在 step 循环；argsort 实测 0.18%/步，按 KISS 不做）；
      Metal 融合 SDPA + rms_norm 落地，短句 RTF 1.088→0.783（-28%，两次 ABBA），
      CPU 侧另有 MLP/KV/conv 系列优化；candle skill 已同步更新。遗留：人工试听、
      ASR 回读（本机无 ASR 模型）、可选的 next-hotspots 候选
      （`omnivoice-next-hotspots-2026-10-23.md` 优先级 1–5）。
- [x] 2. MOSS Nano Candle（2026-10-24 完成，见
      `docs/records/moss/moss-macos-performance-baseline-2026-10-24.md`）：
      Metal RTF 0.208 / CPU 0.308 / 接续 0.207，全部快于实时——**已测量无改动**，
      无优化必要。Nano ONNX CPU 单列 RTF 0.174。长语料因「`next_segment` 与
      `stream()` 切分不一致」blocked，已登记为独立排查项。
- [x] 3. MOSS Local / Realtime 1.7B（同上记录）：Realtime Metal RTF 0.854
      （快于实时，余量 +17%，未做热点剖析）；**Local 1.7B GPU OOM 阻塞**
      （与 2026-10-08 一致），产品无 CPU 档，按 §三 政策不强行。
- [x] 4. IndexTTS（2026-10-24 完成，见
      `docs/records/indextts/indextts-macos-first-measurement-2026-10-24.md`）：
      pinned 源下载（7.7G）→ 固定快照脚本转换（张量数与 Windows 逐项一致）→
      资源就位 + 共用 bench 音色。**Mac 首测：Metal RTF 2.29 / CPU 4.79**
      （crate 例程边界——统一 harness 资源预检按设计拒绝无 SHA 清单的
      indextts，已登记）。CPU 超 §三 参考线不优化；Metal 为后续唯一优化
      候选路径（需先 profiling）。例程 probe 补 metal 设备臂（dev-only）。
- [x] 5. Qwen（2026-10-24 完成变体矩阵，见
      `docs/records/qwen/qwen-macos-17b-baseline-2026-10-24.md`）：1.7B Base
      Metal 0.992（贴线）、1.7B CV 预置 0.758 / 风格 0.751、0.6B CV 0.836
      （旧平铺布局补齐单文件 config.json 后 `resources_verified`）；
      1.7B Base CPU（Accelerate）对照一轮完成（结果登记于记录与
      `target/perf/qwenb-17bbase-cpu-short/`）。1.7B VoiceDesign 未测
      （非必须项）。
- [x] 6. CosyVoice3（2026-10-24 完成，见
      `docs/records/cosyvoice3/cosyvoice3-macos-final-path-2026-10-24.md`）：
      本机转换权重经 harness 导入（7/7 SHA 吻合）+ 2 音色；统一协议复测
      **Metal 短 1.414 / CPU(Accelerate) 1.921 / Metal 长 2.012**——生产流式
      最终路径在实时线之上，与 10-17 非流式 bench 0.936 的口径差异已逐项
      列明（流式+分段+零样本每段 prompt）。CPU 按 §三 不再优化。
- [x] 7. VoxCPM（2026-10-24 完成，见
      `docs/records/voxcpm/voxcpm-macos-regression-2026-10-24.md`）：共同版本
      `42a7887` 统一协议 **Metal RTF 0.782（回归通过，优于历史的 ~0.86）**；
      **CPU 基线首次正式建立 1.760**（Accelerate 默认启用带来大幅改善，
      仍超实时按 §三 不优化）。

## 五点五、完成状态（2026-10-24 凌晨）

§五 0–7 全部勾选。六后端 Mac 结论速览：OmniVoice 优化已合入（0.783）；
MOSS 已测量无改动（Nano 0.21 无需优化）+ Local OOM 阻塞登记；IndexTTS 首测
（例程边界 2.29/4.79，统一协议被设计性阻塞登记）；Qwen 变体矩阵补齐
（0.76–0.99）；CosyVoice3 复测（1.41–2.01，流式超实时）；VoxCPM 回归通过
（0.78）+ CPU 基线 1.76。遗留（不阻塞 goal 完成口径）：各后端人工试听、
OmniVoice ASR 回读（本机无 ASR 模型）、MOSS nano 切分缝隙单独立项、
IndexTTS 统一协议资源身份方案。本轮代码改动仅
`crates/indextts/examples/support/probe.rs`（metal 设备臂，dev-only；
fmt/Clippy/tests/rustdoc 门禁通过）。

每项产出：`docs/records/<backend>/` 设备明确的报告 + 汇总表登记
（已测量无改动 / 优化已验收 / 候选已撤回 / 阻塞 / 不适用），原始数据在忽略目录。

## 六、验收与完成口径

沿用 issue #5：输出型候选默认 ≥10% 且区间不重叠（VoxCPM 6.9% tile 例外不扩大）；
ABBA 正式协议、codes/PCM/EOS 等价、取消与复用不变；先 profiling 定位再无插桩
A/B；普通测试不下载模型，真模型测量独立记录。本 goal 完成 = §五 全部勾选且
每个后端在 Mac 上有可追溯结论（含「测量后无需优化」与「不适用」两种完成形态）。
