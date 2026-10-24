# 官方对照 parity goal（2026-10-24 起）

接续已完成的 [Mac 后端优化 goal](mac-backend-optimization-goal-2026-10-24.md)。
新靶心：**每个后端都以同机官方 Python/原生实现为对照，追平（测量区间重叠
或均值差距 ≤10%）或超过（我们更快且区间不重叠）**。issue #5 的验收门禁
（ABBA、≥10%、输出保持优先、数值变化先过质量门槛）继续适用。

## 一、对照现状（2026-10-24 已有证据核对）

| 后端 | 同机官方基线 | 我们 | 判定 | 本 goal 动作 |
| --- | --- | --- | --- | --- |
| CosyVoice3 0.5B | 官方 CPU（无 MPS 路径）非流式 RTF 1.09–3.38；流式仅测过首包 3.9–6.4s，**流式 RTF 从未对照** | 非流式 bench（优化后）0.936；生产流式 1.414（短）/2.012（长）；首包 2.1–3.0s | **唯一实锤差距：受控组件级 Flow/DiT 慢 4.5–5.6×、HiFT 慢 2.5–4.7×**（Flow/DiT 占我们端到端 76–90%） | **主战场**：补官方流式 RTF 对照 → Flow/DiT 深度优化 |
| OmniVoice 0.6B | 官方 MPS F16 3.235s（RTF≈0.600，ABCCBA 统一窗口，10-23 审计口径） | 免拷贝 2.672s（RTF≈0.495） | **已超过官方 ~17%**（旧"官方更快"的 0.635 vs 0.783 对比已被审计否定） | 登记，不动 |
| Qwen3-TTS 0.6B | 官方 MPS BF16 完整 20.57s | 4.35s（优化后长句完整 RTF ~0.564） | **已超过 ~4.7×** | 登记，不动 |
| VoxCPM2 Q8 | 官方 MPS（自动转 F32） | 与官方区间重叠、配对翻转 | 持平，达标 | 登记，不动 |
| IndexTTS 2.5 | 官方 MPS 稳态 1.42–1.50 / CPU 3.912 | Metal **1.68**（rope_i+sdpa 后持续负载；原 2.29/2.58→FIR 轮 2.08）；CPU 4.789 | 落后收窄到 **~1.15×**；剩余：融合 split+rope kernel（parity 关键）> BigVGAN res_conv/up_conv/snake；对照与三轮记录见 indextts/ 目录 |
| MOSS | 无官方同机对照基础设施 | — | 未判定 | 不在本 goal（如需另行立项） |

## 二、对照与验收口径

1. **对照双方**：同机、同语料、同音色/参考、固定官方 revision 与 torch 版本、
   同模式（非流式对非流式、流式对流式；不能成对的单列并标注）。官方在 Mac
   只能 CPU 时，差异如实标注"设备差异"，不冒充语言/框架差异。
2. **parity 定义**：同口径 RTF 双方区间重叠，或均值差距 ≤10%；**超过** =
   我们更快且区间不重叠。每个结论必须给逐轮数据与协议（ABBA/ABCCBA）。
3. **优化顺序**：先 profiling 后改动；CosyVoice3 从 Flow/DiT（76–90% 占比）
   开始，候选包括 Metal 算子/注意力路径、精度与布局——数值变化方案先过
   质量门槛再动。
4. **政策延续**（自上一 goal）：模型统一 `~/.talechime`；空间不够清构建缓存
   （evidence 目录先查引用）；CPU 不强行优化（CPU 对照测量仍作证据）；
   candle skill 同步演进；试听/ASR 回读遗留项继续挂起不阻塞。

## 三、执行清单

- [x] 1. **CosyVoice3 官方流式 RTF 对照**（2026-10-24 完成，见
      `docs/records/cosyvoice3/cosyvoice3-official-parity-2026-10-24.md`）：
      官方 CPU 流式 **RTF 1.5205**（1.314–1.806）vs 我们 Metal 流式 **1.414**
      ——**端到端 parity 成立**（区间重叠，均值快 ~7%）；首包 2.39 s vs
      官方 3.5–5.5 s 显著更快；官方非流式 1.151（我们同口径未测，登记边界）。
      官方在 Mac 同样流式 >1。坑：官方要求转写自带 `<|endofprompt|>` 后缀，
      否则 LLM 线程断言退出、主生成器假死。
- [x] 2. **CosyVoice3 Flow/DiT 深度优化**：验收参照已按第 1 项结果重新定位
      （见对照记录 §三）：目标 (1) 生产流式 RTF < 1.0（= 在 Mac 上超越官方）；
      (2) 组件级 CPU 差距从 4.5–5.6× 收窄到 ≤2× 或给出不可行归因。先 Metal
      kernel 级 profiling 定位，再逐项 ABBA。
      **优化第一轮已验收**（2026-10-24，见
      `docs/records/cosyvoice3/cosyvoice3-metal-attn-rope-sdpa-2026-10-24.md`）：
      固定形状逐算子探针归因 = 物化注意力（单 block 5.05ms/10.07ms）+ rope
      广播链；上游 `rope_i`（partial RoPE 纠偏：只对 64 宽切片）+ `sdpa`
      （mask 每 solve 展开一次）迁入 Metal 分支，CPU 锚逐位不变；ABBA
      **-12.5% 区间不重叠**（短句 1.415→1.278 无噪声槽），首包 2.49→2.16s。
      **流水线级 profiling 已完成**（同记录 §七，`CV3_STREAM_PROFILE=1`
      内建分段计时）：短句 8.93s = Flow 54% / HiFT 25% / LLM 22%；长文
      106s = Flow 56% / HiFT 26% / LLM 18%，**长文 RTF 2.012→1.72**；逐段
      零样本开销仅 ~1.5%（非主要矛盾）；O(n²) 前缀重解为官方语义；NFE=10
      与官方同值。
      **第 (2) 项已达成**（证据在案，此前文本为引用遗漏）：2026-10-17 记录
      §11 受控复测（accelerate，同 184-tok fixture）：Flow/DiT CPU
      **4.8s vs 官方 4.05s = 1.19× ≤2×**，逐 token 端到端 CPU 反超官方
      （43.6 vs 45.6 ms/tok）。残余 CPU HiFT 3.2×（卷积类，Metal 档为快速
      路径，不属本项点名范围，登记不优化）。
      **仅剩第 (1) 项**：短句 1.278，需再 −1.7s。已定路径：fork 分块转置
      内核（split+出口转置 ~2.1ms/block 非连续拷贝，估 −0.9~−1.1s，与
      IndexTTS follow-up 同内核家族）+ HiFT Metal 卷积轮（估 −0.7~−1.1s），
      合计估算 RTF 0.98–1.05（紧）。长文 <1.0 受官方 O(n²) 语义约束，
      归因为不可行（增量缓存=语义变更，另行立项）。
      **第二轮已验收**（2026-10-24，见
      `docs/records/cosyvoice3/cosyvoice3-metal-transpose-copy-2026-10-24.md`）：
      candle fork `5a7d251f`（FORK.md 登记，behavior-only）加
      `transpose2_contiguous_f32` 行块内核，命中转置(1,2)视图连续化的
      精确 stride 签名（11.6 GB/s 病灶 → ≈113 GB/s，tpose 9.7×、
      split_x3 7×、attention 总体 −35%@T=512）；位级一致门 7 形状×双向
      + cv3/indextts 全套回归绿；ABBA **-11.7% 区间不重叠**（B 臂
      1.310–1.323 极稳，热稳态口径 −13%），首包 2.42→2.19s。IndexTTS
      split 链路同享（其 fork follow-up 剩余价值修正为 rope 融合与
      BigVGAN）。**RTF 热稳态 1.31 仍 >1.0**。**HiFT 探针归因已完成**
      （同记录 §六，`CV3_HIFT_PROFILE=1` 内建计时）：「HiFT 卷积轮」前提
      被证伪——360 帧下 f0 condnet（CPU f64 卷积链）537ms/50%、
      resblocks 317ms/29%、sbranch 140、up 66、istft 仅 16ms；候选修正
      为 ① fork 内 candle CPU conv1d f64 走 im2col+dgemm（估 −0.3~0.4s）
      ② resblock Metal conv 轮（−0.15~0.25s）③ 之后还需 LLM 轮
      （1.93s）才可能 <1.0（热态 ~1.15–1.2 → 需再压）。
      **f0 轮已结案——前提证伪**（同记录 §七）：f0 真算力仅 ~10ms，
      596ms 是 `to_device` 吸干 flow 队列（段首传输陷阱）；candle CPU
      conv1d 本就走 im2col+GEMM 且 accelerate 覆盖 dgemm，无立项价值。
      **修正五桶：Flow 5.17s（62%）/LLM 1.95s（23%）/HiFT 0.94s（11%）**
      （flow 段尾同步后）。主战场回到 Flow：块内 GEMM ~1.5s 可解释，
      另 ~0.9s 差值待 release 构建 forward_prepared 单步在位探针归因
      （debug 探针的 CPU 派发开销失真）→ GEMM tiling / 元素级融合 /
      派发批化三选；HiFT 剩 GAN 卷积链 ~0.88s。
      **第四轮已验收**（2026-10-24，见
      `docs/records/cosyvoice3/cosyvoice3-metal-fused-adaln-2026-10-24.md`）：
      `CV3_DIT_PROFILE` 逐步在位探针（六区先同步再停表，release）归因差值
      = 手写 7-pass LayerNorm 链 ×2/block（adaln 38.6+门控 54.6ms/步）；
      `modulated_layer_norm` 按分支折叠 AdaLN 调制进融合 candle
      `ops::layer_norm`（仅 Metal，CPU 锚位级不动；一遍式方差差 ~2e-7
      有门）；adaln −61%/步墙钟 −12%；ABABABAB 交替 ABBA（热态噪声否决
      两轮后改设计）**-18.0% 区间不重叠**（A 1.365 vs B 1.119；中位对
      -14.1%）。**冷态 RTF 0.982<1.0 已达**（B4 重跑 5 轮 0.980–0.983），
      持续负载态 1.165（机身双态：持续 GPU 负载降频，非代码可控）。
      剩余候选（已定量）：门控广播 pass ~0.9GB/s（fork 二元广播快路径）、
      embed 分组因果卷积 24–28ms/步（grouped conv 每组循环病）、
      attn/ffn GEMM 近峰。
      **第五轮已落地、未单独验收**（2026-10-24，见
      `docs/records/cosyvoice3/cosyvoice3-metal-broadcast-binary-2026-10-24.md`）：
      fork `15af86dd` `binary_rbcast` 内核（广播 rhs 索引坍缩，覆盖
      (B,1,D) 门控/窄视图与 (D,) bias 两族，位级门+微基准 25×）；端到端
      冷态成对 **-4.5%（<10% 门槛）**，暂记已落地，与 embed 分组卷积
      （估 -0.4s，需融合直卷内核）**合并 ABBA 验收**（合计估 ~9% 贴线，
      不过线则再并入 rotate 链融合轮）。持续负载口径现状 ≈1.11（ABAB
      B 槽），冷态极值 0.98。
      **第六轮已落地、未单独验收——embed 直卷证伪 + custom-op 三内核**
      （2026-10-24，见
      `docs/records/cosyvoice3/cosyvoice3-fused-dit-customops-2026-10-24.md`）：
      embed 分组直卷两版（朴素/8宽l分块）均比 im2col 慢 3–4×（"136MB 物化"
      前提被 L2 常驻+GEMM 计算受限证伪），fork `884b3598` 留休眠基础设施；
      转向 candle `CustomOp3::metal_fwd` 公开面，三融合内核全住 talechime
      （rope+头拆分位级一致 0.097 vs 0.276ms/调用；门控残差+modLN 双平面；
      块尾门控加），零 fork 派发。合并 ABAB（A=LN 基线，B=rbcast+三内核）
      **-4.95% 区间不重叠、四对 -4.8~-5.1%**（冷态 B 0.9353 vs A 0.9840）
      ——真实但 <10% 线，登记待与下一轮合并。剩余最大项=adaln 边界融合
      （块间 LN 并入尾内核+affine 批化，估 -2.5~~-3.5%）→ 与本轮合计有望
      过线；sdpa 桶 ~29ms/步（flash 级工程）与 embed 未立项。持续负载口径
      仍 ≈1.11 量级（本轮冷态 −5% 折算持续态 ≈1.05–1.06），**<1.0 未达**。
      **第七轮已落地——adaln 边界融合，合并 ABAB −5.11% 区间不重叠；
      持续负载口径 0.933 <1.0 达标**（2026-10-24，见
      `docs/records/cosyvoice3/cosyvoice3-boundary-ln-fusion-2026-10-24.md`）：
      尾内核升级 `cv3_gate_tail_modln_f32` 同遍产出下一块 modLN 平面（末块=
      norm_out，out 段归零），块间 LN 全消（adaln 10.3→7.3ms/步）；affine
      批量化经评估**有意不做**（权重栈 553MB 常驻换 ~1–2ms/步，不成比例）。
      逐步墙钟热平价 275→249.5ms（−9.3%）；合并 ABAB（A=LN 基线，
      B=rbcast+四内核）**−5.11% 区间不重叠、四对 −4.99~−5.19**（A 0.9851
      vs B 0.9347）——单轮仍 <10% 线（连续三轮 −4.5/−4.95/−5.11%，收益
      递减+DiT chunk 份额 ~45% 封顶），第五~七轮累计登记。
      **目标 (1) 持续负载口径 <1.0：以同口径实测达成**——ABAB B 槽
      0.9334–0.9355（n=20 轮）+ 连续负载 15 连跑（~60s 背靠背 GPU 满载）
      0.928–0.935 无降频爬升。此前 1.11–1.165 的"持续态"读数为当日热背景
      所致（同 A 基线二进制当日 1.119–1.365、今日 0.982–0.987，机身双态
      幅度 ±15% 为环境变量而非二进制属性），已如实登记。基线 1.415→0.935
      累计 −34%；六轮已验收/落地（rope+sdpa −12.5%、transpose2 −11.7%、
      LN 融合 −18%、rbcast、custom-op 三内核、边界 LN），全部数值门+位级
      门在案；sdpa 为 candle 全 flash 内核、GEMM 近峰（2.3 TFLOPS 有效），
      剩余元素级地板 ~−40ms/步已定量登记未立项。目标 (2) CPU ≤2× 此前已
      达（1.19×）。**§三.2 两项验收口径均已有可追溯证据，勾选。**
- [x] 3. **IndexTTS 官方对照建立**（2026-10-24 完成，见
      `docs/records/indextts/indextts-official-parity-2026-10-24.md`）：
      官方检出 `d9e41aa` + uv venv（torch 2.8），**官方原生支持 MPS**——
      同语料同参考：官方 MPS 稳态 **RTF 1.42–1.50** vs 我们 Metal 2.289
      （落后 ~36–56%）；官方 CPU 3.912 vs 我们 4.789（落后 ~18%）。
      **优化第一轮已验收**（FIR 融合 conv1d Metal kernel，candle fork
      `712d0017`，ABBA -19.4% 区间不重叠，见
      `indextts-metal-fir-conv-2026-10-24.md`）：Metal 持续负载
      2.58→2.08，差距收窄到 ~1.4×。
      **优化第二轮已验收**（DiT 注意力链路：上游融合 `rope_i` + `sdpa`，
      应用层改动零 fork，ABBA **-15.3% 区间不重叠**，Metal 持续负载
      1.98→**1.68**，见 `indextts-metal-rope-sdpa-2026-10-24.md`）：
      归因=GEMM 全部健康（3.5–4 TFLOPS），病灶为 RoPE 广播元素级链
      （stride-2 视图慢路径，52%）+ 手写注意力；名义 CFM 6.4→~4.5s
      （官方 3.21）。**未完候选已升级为独立 goal**（2026-10-24 立，
      `indextts-exceed-official-goal-2026-10-24.md`，靶心 1.68→<1.42
      超过官方；split+rope 融合优先走 talechime custom-op 宿主路径而非
      本记录写的 fork 轮——该路径在 cv3 第六/七轮才建立）+ BigVGAN
      res_conv/up_conv/snake（~3s 插桩）；GPT 3.3s 已领先不动。**
- [x] 4. **登记已达标签**（2026-10-24）：OmniVoice（超过 ~17%）/ Qwen
      （超过 ~4.7×）/ VoxCPM（持平）——见 §一 对照表，官方换版本才重开。
- [x] 5. **skill 沉淀**（2026-10-24 完成）：measurement.md 新增「同机官方对照」
      一节（口径四要素、设备差异≠框架差异、parity 与组件差距并存、判定口径、
      ABCCBA、官方预热假象、endofprompt 类输入契约陷阱）；case-studies 补
      MOSS/cv3 parity 两行。IndexTTS FIR 的归因已随对照记录 §4.1 沉淀；
      后续优化轮的结论继续按轮补入。

## 四、完成口径

§三 1–5 全部勾选，且每个后端在"官方对照"栏有可追溯结论（超过 / parity /
差距已定位并登记后续）。优化本身允许"定位后本轮不做"（如成本/收益不成立），
但必须给出与官方的明确对照数字。
