# IndexTTS BigVGAN FIR 融合 conv1d Metal 快路径（2026-10-24）

[官方对照 parity goal](../plans/official-parity-optimization-goal-2026-10-24.md)
IndexTTS 主线的第一步优化：candle fork 新增共享核单通道 conv1d 融合 Metal
kernel，消除 FIR 的 im2col+小 GEMM 开销。**通过 ABBA 八进程验收（-19.4%，
区间不重叠）**，candle 与 talechime 侧门禁全绿。

## 一、改动

1. **candle fork**（`yexiyue/candle` `talechime-perf`：`2266099` →
   **`712d0017`**，FORK.md 已登记）：
   - `candle-metal-kernels`：`conv.metal` 新增 `conv1d_shared_*` 融合核
     （每线程一个输出，K-tap 单遍 MAC；f32/f16/bf16 模板）；
     `kernels/convolution.rs` 新增 `call_conv1d_shared`。
   - `candle-core` Metal `conv1d`：当 `padding==0 && F32 && c_in==c_out==1 &&
     k≤64 && 双方连续` 时走融合核，其余形状原路径。**behavior-only，
     无公共 API 变更**（FORK.md 约束）。
   - 新增 ignored Metal 测试 `conv1d_shared_matches_cpu_reference`
     （6 组形状 × stride 1/2，对 CPU conv ≤1e-5，`--test-threads=1`）。
2. **talechime-indextts `bigvgan.rs`**：`FirFilter` 相位核预乘
   `AA_RATIO`，`fir_upsample` 去掉末尾全张量 affine。2 的幂缩放与舍入
   可交换，逐位等价（CPU 交叉验证测试继续通过）。

 talechime 侧**无调用点改动**：FIR 本就以 `(B·C,1,L)`×`(1,1,K)` conv1d
 形式发出，自动命中快路径；分派在 candle 设备层完成，不新增分支逻辑。

## 二、证据链

### 2.1 离线探针（ignored 测试 `fir_metal_conv_vs_tap_timing`，真实六级形状）

| 形状 | down conv→fused | up conv→fused | 数值（vs 逐 tap 参照） |
| --- | --- | --- | --- |
| 1536×492 | 1.64→0.57 ms | 3.64→0.95 ms | ≤7.2e-7 |
| 384×7872 | 6.74→1.05 ms | 15.04→3.59 ms | ≤9.5e-7 |
| 48×62976 | 6.65→0.94 ms | 14.87→3.28 ms | ≤9.5e-7 |

### 2.2 ABBA 八进程（A=旧 rev 二进制，B=新 rev，1 预热+5 热测 × 8 slot 串行）

| | mean-of-means RTF | 进程均值区间 |
| --- | ---: | --- |
| A（基线） | 2.5784 | [2.5745, 2.5839] |
| B（融合核） | **2.0794** | [1.9383, 2.1306] |

**-19.4%，区间不重叠**；逐 slot 全部 B 快（相邻对差 21–33%）。绝对值与
首测 2.289 的差异来自持续负载状态（本 ABBA 为 8 进程连跑），A/B 交错内部
自洽。二进制 sha：A `b1fd2876…`、B `8dee6f66…`（file:// 中间 rev，与最终
rev 仅差格式化/测试形状修复；最终 rev 重建复测每轮 2.08–2.12 一致）。

### 2.3 分阶段复测（最终 rev，`INDEXTTS_PROFILE=1`，3 轮）

| 阶段 | 改前 | 改后 |
| --- | ---: | ---: |
| BigVGAN 合计 | 7.24–7.72 s | **3.90–4.13 s（-46%）** |
| fir_down | 2.11–2.24 s | **0.32 s** |
| fir_up | 2.04–2.15 s | **0.59–0.61 s** |
| res_conv / snake / up_conv | 1.5 / 0.7 / 0.87 s | 不变（真实 conv/elementwise） |
| CFM / GPT | 6.2–6.7 / 3.0–3.3 s | 不变 |

### 2.4 输出等价性（如实登记）

模型级 WAV 对比**不能**作为本轮等价判据：基线自身跨进程就不确定
（A1/A4/A8 同 seed 的 sha 互不相同；基线对基线 max_abs 100–127%、
RMS 相对差 0.75–4.3%，**大于** A vs B 的 72.6%/0.32%——IndexTTS 的 Metal
采样存在跨进程不确定性，与本改动无关）。等价性证据 = 固定输入探针
（≤9.5e-7，§2.1）+ candle-core 对 CPU conv 测试；样本数/时长在所有
ABBA 轮完全一致。

## 三、对 parity goal 的影响

- IndexTTS Metal 持续负载 RTF **2.29 → 2.08–2.12**；对官方 MPS 稳态
  （1.42–1.50）的差距从 ~1.6× 收窄到 **~1.4×**，仍未 parity。
- 剩余缺口分解（每段单轮量级）：**CFM 6.4 s（最大单项）** > res_conv
  1.54 s > up_conv 0.87 s > snake 0.69 s > FIR 残余 0.93 s。下一优先级
  CFM（官方 3.21 s，差 ~3.2 s）；res_conv/up_conv 同为 conv 路径但为
  多通道真实卷积，GEMM 形态合理，需单独归因。

## 四、门禁

talechime：fmt clean、`clippy -D warnings`（metal all-targets）通过、
crate 106 测试 + 18 ignored 通过、rustdoc 通过。fork：新增代码 0 clippy
告警（上游既有 21 条新 clippy 版本告警不动）、Metal ignored 测试通过。
普通测试不下载模型。

## 五、原始数据

`target/perf/itb-fir-abba/`（8 slot index.json/逐轮 WAV/日志/双二进制）、
`target/perf/itb-profile-after2/`（分阶段）、探针输出在测试日志。
