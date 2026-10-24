# IndexTTS 超过官方 goal（2026-10-24 起）

承接 `official-parity-optimization-goal-2026-10-24.md` §三.3（对照已建立、
两轮已验收、落后收窄到 ~1.15×）。新靶心：**IndexTTS Metal 持续负载 RTF 从
1.68 压到 < 官方稳态下沿（<1.42），并以同日双臂区间不重叠的对照实测确认
「超过官方」**。

## 一、现状与靶心（2026-10-24 已有证据）

| 项 | 官方 MPS（稳态 r4） | 我们 Metal（rope 轮后） | 差距 |
| --- | ---: | ---: | --- |
| 端到端 RTF（持续负载） | **1.42–1.50** | **1.68** | 落后 ~15% |
| GPT 循环 | 3.72 s | 3.02–3.34 s | **我们更快，不动** |
| CFM/s2mel | 3.21 s | ~4.3–4.7 s（名义推算） | 慢 ~1.4×，GEMM 已近峰 |
| BigVGAN | 1.00 s | ~3.5 s（res_conv 1.43–1.62 + up_conv 0.83–0.90 + snake 0.72 + FIR 残余） | **慢 ~3.5×，最大缺口** |

历史轨迹：2.289 → FIR 融合卷积轮 **−19.4%**（`indextts-metal-fir-conv-2026-10-24.md`，
fork `712d0017`）→ rope_i+sdpa 轮 **−15.3%**（`indextts-metal-rope-sdpa-2026-10-24.md`，
零 fork）→ **1.68**。要「超过官方」需再 **−17~20%**（≈ −2.2 s）。

## 二、判定与验收口径

1. **超过** = 同日、同语料、同参考、固定官方 revision（`d9e41aa` + uv venv
   torch 2.8），我方更快且 RTF 区间不重叠；**官方只取稳态轮**（MPS 惰性
   编译，前几轮无效——见 parity 记录 §三）。
2. **机身双态纪律**（cv3 第七轮教训）：绝对 RTF 是环境变量（同二进制不同
   日 ±15%）。最终对照必须当日双臂交错（ABCCBA 或 ABABABAB），我方结论
   另做 ≥15 次背靠背连跑确认无降频爬升。
3. 每优化轮沿用既有门禁：固定形状探针归因 → 数值门（位级或 ≤1e-5，
   附 CPU 锚）→ ABAB 交错 ≥10% 且区间不重叠单轮验收；子阈值改动与下一轮
   合并验收。
4. **WAV 级对照不可用作等价门禁**（Metal 采样跨进程非确定性，FIR 记录在
   案）：以固定输入探针 + 数值门为准；人工试听另行挂起不阻塞。

## 三、执行清单（按预期收益排序）

- [x] 1. **融合 split+rope 内核**（预计 −0.9 s，CFM 4.5→~3.6）：
      单内核读 qkv 直写 roped `(B,H,T,D)`，消 6 次跨步拷贝（split 0.88s +
      attn_out/rope 残余）。**首选 talechime 侧 custom-op 宿主路径**
      （`crates/cosyvoice3/src/flow/metal_fused.rs` 是完整模板：零 fork；
      三铁律 = 输出缓冲 `Output::new` 绑定、免除非 2 幂整除的 3-D grid、
      cat 结果 `.contiguous()` 归一）——rope 记录写"fork 轮"时该路径尚未
      发现，优先级反转。仅当 custom-op 路径受阻才走 fork（cv3 rope+头拆分
      内核即同族，可整体移植，interleaved rope 表达式逐字对齐 rope_i 保位
      级一致）。数值门：位级（cv3 同款已验证可行）。
      **✅ 2026-10-24 落地（宿主路径，零 fork）**：`itb_qkv_split_rope_f32`
      位级门全过（q/k/v+sdpa 视图路径 max_abs=0），区域探针 1.344→0.157 ms
      （8.6×），单段 ABAB −8.69% 子阈值，并入第 2 项合并验收。见
      `indextts-metal-hosted-kernels-2026-10-24.md`。
- [x] 2. **BigVGAN res_conv / up_conv 轮**（conv 项 ~2.3 s，预计 −1.2~1.5 s）：
      先插桩复测现状（FIR 轮后各子项占比已变，勿沿用旧比例），再逐个归因
      conv 形状。已知背景：candle Metal conv1d 在此类形状距带宽地板
      10–30×（parity 记录 §4.1 定性结论）；cv3 的直卷内核经验**不直接
      移植**（那边 im2col 实际 L2 常驻、前提被证伪——先探针后动手），
      但 IndexTTS 的 conv 是大通道非分组、且 Windows 轮已两次实测卷积是
      Candle 弱项，预期成立。内核走 custom-op 宿主。
      **✅ 2026-10-24 落地**：归因发现 **up_conv 是重灾区**（BigVGAN 六级
      ConvT padding≠0 全部落入 candle naive 标量核，L0/L1 仅 0.015
      TFLOPS）→ `itb_convt_gather_f32` GEMM+gather 分解（六级 879→14.3 ms，
      L0/L1 各 ~125×，rel≤2.6e-6）；res_conv 走 im2col+GEMM 直出布局
      （`itb_im2col1d`+matmul，位级一致，全级仅 −5~10%——**探针证明
      im2col 物化本身是地板**，直卷内核登记为候选未做）。合并 ABAB
      （含第 1/3 项）：A 1.8175 [1.7387,1.9098] vs B **1.3568
      [1.3155,1.4194]，−25.35% 区间不重叠**，单轮验收通过。
- [x] 3. **snake 激活融合**（0.72 s，预计 −0.4~0.5 s）：逐元素三角函数，
      与相邻 conv/残差折成单遍；若第 2 项内核顺手吸收则并入其验收。
      **✅ 并入第 2 项验收**：`itb_fir_up_snake_f32` 把 pad 物化+双相位
      conv+snake 五算子链折成单 dispatch（chain 4.5–8.7→0.77–0.96 ms，
      ~8×，rel≤4.3e-7）；插桩 fir_up+snake 1.37→0.213 s。
- [x] 4. **CFM 残余与 codec**：~~第 1 项后 CFM ~3.6 vs 官方 3.21，GEMM 近峰
      无易果；仅登记不立项，除非第 1–3 项完成后仍未过线~~。
      **登记完成，且因第 5 项判定未过线按条款升级并部分执行（同日续轮）**：
      补插桩归因出 estimator 5.07s 中 **WaveNet 终层未插桩 2.29s**（此前
      完全未归因）；融合门控激活+res/skip 尾双平面内核（均位级）后
      wn.act 0.58→0.13s、dit.wavenet 2.51→1.57s、**estimator 4.39s**
      （对官方 3.21 差 −1.18s，续轮前 −2.9s）。剩余：ffn 1.47（GEMM
      地板）/ wn.conv 0.91+res_skip 0.39（candle 批 GEMM rhs 广播时
      batch 折 M 已近优，B=2 直出布局实测反退已回退登记）/ adaln 0.40
      （modulated-RMS 融合内核候选，未做）。GPT 3.44 vs 3.02–3.34 近平。
      详见记录 §七。
- [x] 5. **最终对照与判定**：当日官方稳态复测 + 我方 ABAB 连跑，出
      「超过 / parity / 未达」结论与全部槽值；若未达，按剩余定量缺口更新
      本清单而非放宽口径。
      **✅ 判定 = parity（同日两次交错，第二次略领先；未达"超过"）**。
      第一次（上午，四内核轮）：官方稳态两窗口 [1.067, 1.205]（O1）/
      [1.378, 1.629]（O2）——官方自身同日漂移 ~27%（上午 1.42–1.50 亦在
      案）；O-T-O-T 交错窗口 1 官方 1.145 vs 我方 1.208（慢 5.5%）、
      窗口 2 官方 1.447 vs 我方 1.405（反超 2.9%）——两臂同步漂移、方向
      翻转，全量 T +0.84% 区间重叠。我方 16 连跑稳态 1.2093 无爬升。
      第二次（晚间，bias 修复+wavenet 融合续轮，修复后 ABAB −25.16%
      区间不重叠复验）：窗口 3 官方 1.1862 vs 我方 1.1936（打平），
      窗口 4 官方 1.3849 vs **我方 1.3280（快 4.1%）**，聚合 T −1.92%
      略领先、区间重叠——两窗口不败+聚合领先，但仍不满足"更快且区间
      不重叠"。起步落后 ~15%（1.68 vs 1.42–1.50）→ 当日同机收敛到
      ±1–5%。原始数据 `target/perf/itb-final/`、`runs/indextts-parity/
      official-mps-{o1..o4}.json`。
- [x] 6. **skill/memory 沉淀**：每轮结论入 `skills/candle-performance`
      （case-studies 加行；新方法论入 metal.md），memory 同步；记录按
      docs/records/indextts/ 归档规则落子目录。✅
      `indextts-metal-hosted-kernels-2026-10-24.md` + skill 两处 + memory。

## 四、完成口径

§三 1–6 全部勾选且第 5 项给出可追溯判定。允许"定位后本轮不做"（成本/
收益不成立需写明）。预算参考：第 1+2+3 项若都兑现，估算总时长
~10.6→~9.1 s（RTF ~1.33）< 官方 1.42 下沿——**第 1、2 项是过线双关键，
第 3 项提供余量**；只完成其一大概率停在 parity 区间。

## 五、约束与复用资产（沿既有政策）

- 模型只在 `~/.talechime`；空间不够先清构建缓存，删 evidence 目录前查
  docs/records 引用；普通测试不得下载模型；GPU ignored 测试
  `--test-threads=1`；talechime 仓库仅在用户要求时提交（fork 推送除外）。
- fork：`github.com/yexiyue/candle` talechime-perf 分支，workspace [patch]
  四行 rev 同步流程；每补丁 FORK.md 登记；behavior-only。
- 复用设施：`INDEXTTS_PROFILE=1` 分段计时、`target/perf/itb-*` 探针目录、
  cv3 的 `metal_fused.rs` 模板与 ABAB 脚本（`target/perf/cv3-dc-abba/
  run_abab.sh` + `analyze.py` 可拷改）、candle fork 内既有内核
  （transpose2 / rbcast / FIR / GEMM tiles）。
- 教训清单（每条都有实测出处，写在新会话最前面重读）：
  输出缓冲 `Output::new`；3-D grid 免整除；数值门≠性能门（region 探针才是）；
  traffic 估算先问 L2 驻留；微基准 N× ≠ 端到端 N×；async 队列先同步再停表；
  机身双态是环境变量、结论当日复测。
