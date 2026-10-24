# CosyVoice3 官方对照 parity 判定（2026-10-24，M4 Pro）

[官方对照 parity goal](../plans/official-parity-optimization-goal-2026-10-24.md)
第 1 项：补上从未测过的**官方流式 RTF**，同语料同音色同模式判定。

## 一、判定（先说结论）

| 口径 | 官方 PyTorch CPU | 我们 Rust Metal（统一协议 3×6） | 判定 |
| --- | --- | --- | --- |
| **流式 RTF**（生产同模式） | **1.5205**（5 轮 1.314–1.806） | **1.414**（进程均值 1.403–1.419） | **parity**：区间重叠，均值我们快 ~7%；不构成"超过" |
| 流式首块（请求起算） | 3.5 s（新鲜模型态）/ 5.51 s（热轮均值） | **2.39 s** | 我们显著更快 |
| 非流式 RTF | **1.1509**（5 轮 1.073–1.284） | 同口径未测（历史 bench 0.936 为不同语料/音色，仅方向参考） | 登记边界，不判定 |

**端到端层面我们与官方 parity**。官方在 Mac 上同样流式 RTF>1（双方都不能裸奔
流式听书）；"差距"收敛到两个明确的剩余维度：(a) 受控组件级 CPU 差距
（Flow/DiT 慢 4.5–5.6×，10-17 记录，与端到端 parity 并存——Metal 设备优势
掩盖了组件劣势）；(b) 流式 RTF 本身仍 >1。goal 第 2 项的优化目标据此
重新定位：**把流式 RTF 压回 1 以下是"超过官方"的唯一路径**（官方做不到）。

## 二、方法与身份

- 官方：固定检出 `runs/reference/CosyVoice@074ca6d`（+ matcha），
  `runs/cv3-venv`（torch 2.7.1，CPU——官方硬编码 cuda-else-cpu，无 MPS），
  原始权重 `runs/cv3-model`。`torch.set_num_threads(8)`、进程级 seed 42。
- 直调脚本 `runs/cv3-parity/official_stream_rtf.py`（忽略目录；官方高层
  `frontend_zero_shot + model.tts(stream=True/False)` 逐块计时，预热 1 +
  5 热测）；分诊日志与逐轮 JSON 同目录。历史对照工具在 git 快照
  `e591c94^:tools/tts/cosyvoice3_bench.py`（fixture 驱动，本轮未用）。
- 同一语料（corpus-short 山风句）与同一参考音色（voice_bench reference.wav
  3.48 s + 转写）；**官方要求转写自带 `<|endofprompt|>`（151646）后缀**，
  frontend 不代加——不带则官方 LLM 线程断言退出、主生成器在轮询睡眠里
  挂死（首次运行的假死根因；HiFT f0 conv 崩溃是其下游症状）。
- 我们的数字沿用 [统一协议复测](cosyvoice3-macos-final-path-2026-10-24.md)
  （生产请求边界，含每请求参考特征提取；官方 `rtf_request` 同边界）。
- 注意：双方输出音频长度不同（我们 7.10 s / 官方各轮 5.84–6.40 s——
  跨框架采样轨迹不同，RTF 按各自音频时长归一，不做内容等价声明）。
  官方热轮首块 5.5 s vs 新鲜模型态 3.5 s：官方流式 hift hop 状态跨请求
  保持，块化行为随状态变化，如实登记。

## 三、对 goal 第 2 项的影响

Flow/DiT 深度优化的验收参照从"追官方端到端"改为：**(1)** 生产流式路径
RTF < 1.0（超越官方在 Mac 上的能力）；**(2)** 组件级 CPU 差距收敛
（受控同负载对照复测，Flow/DiT 从 4.5–5.6× 收窄到 ≤2× 或给出不可行归因）。
沿用 ABBA/输出保持门禁。

## 四、原始数据

`runs/cv3-parity/`：`official-stream.json`、`official-offline.json`（逐轮）、
`probe*.log`（分诊与 endofprompt 根因）、脚本本体。
