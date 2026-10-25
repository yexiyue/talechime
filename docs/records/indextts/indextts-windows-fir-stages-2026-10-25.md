# IndexTTS BigVGAN FIR 细分诊断（Windows，2026-10-25）

Windows 11 / Ryzen 7 9700X / RTX 5070 12 GB，Candle 0.11 fork
`2266099d92265112ecc0d56ecee1761e3cc2ac00`，CUDA 13、cuDNN 9.14。
本轮沿[生产调查](indextts-windows-production-2026-10-24.md)的 vocoder 差距继续归因，
不修改模型数学、权重、精度或 FIR 的 ImplicitGemm 算法，不宣称性能优化已验收。

## 方法与身份

在原有 `INDEXTTS_PROFILE` 下细分上采样的 pad、odd_conv、even_conv、
interleave_scale，以及下采样的 pad、conv。关闭时不新增设备同步；开启时每个
子范围同步。conv 范围含 reshape/crop，interleave_scale 含 stack/reshape/affine，
均不是独占 GPU kernel 时间；外层 `bv.fir_up/down` 与子范围嵌套，不能相加。

显式 ignored `bigvgan::tests::real_cuda_fir_stage_diagnostic` 加载实际转换权重：

- `bigvgan.safetensors` SHA256：`054e00342ad121da0b93f1fd2ecb0fcf99f7da6b5a0c03969d6b602e2c08e517`。
- `bigvgan.config.json` SHA256：`88a1f47acf747db0b21e97a389d838566147f7a5464583ff5c8d819d870f03ee`。
- 冻结测试 exe SHA256：`35b4e9daee5ad85127e94728d0ecb2c4ea48805e19bf1deec47d708bdecf41d9`。

mel 是确定性的合成平滑输入，narrow 后非连续，依次 8→128→224→8 帧；
不是从真实 CFM 请求截取的激活。每 case 1 预热+5 热轮，三个独立 profiling
进程串行运行，另一个独立进程关闭 profiling 作输出控制。全部先编译再运行，
没有并发编译/推理。原始 F32LE 在计时后导出，不提交到仓库。

## 结果

下表为每 case 五个热轮的进程均值，再对三个进程取均值，单位 ms；
每个范围每次完整 vocoder 调用均有 109 次，短形状存在热状态漂移。

| mel 帧数 | up pad | odd conv | even conv | interleave/scale | down pad | down conv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8，首次 | 8.788 | 7.775 | 6.677 | 3.022 | 9.807 | 5.996 |
| 128 | 12.719 | 23.289 | 19.738 | 9.853 | 11.644 | 16.408 |
| 224 | 12.323 | 31.672 | 29.424 | 38.071 | 11.593 | 24.616 |
| 8，再次 | 10.024 | 7.950 | 6.663 | 3.527 | 11.276 | 5.931 |

224 帧 interleave/scale 的三个进程均值范围 37.271–39.396 ms，128 帧为
9.682–9.949 ms。大形状不是首进程单轮尖峰，但尚不能据此断言带宽、分配或
派发根因。padding 也非可忽略：128 帧 up/down 合计约24.36 ms。
这些同步范围受插桩扰动，不是生产完整耗时的分解，也不能直接当作可节省时间。

三个 profiling 进程的 72 份输出与关闭 profiling 的对应 24 份 raw F32LE
SHA 全部相同；每进程还逐位检查同 case 重复输出、有限值和 samples=frames×256。
该组件实验没有 GPT codes/EOS、生产 Registry 请求、取消状态、ASR 或试听验收。

## 决策与日志陷阱

接下来优先验证保持数值的布局/复制候选，再考虑变更 FIR 计算。当前 Candle 的
`pad_with_same` 对每个边界副本分别加入 cat：上采样的左右各5形成11个输入，
下采样左右5/6形成12个输入。广播左右两块再三块拼接是待测候选，不能因输入
数量减少就认定 kernel 更少或更快。大形状的相位拼接也应先做局部 A/B，保持实际
strides、输出位模式和完整阶段后，再决定是否进入生产请求验证。

第一版标签写 stdout、profile 写 stderr，PowerShell 合并后乱序；原日志保留，
不作逐轮归因。第二版统一诊断 JSON 到 stderr，并分开保存 stdout/stderr，
汇总器要求 profile 与标签严格一一配对，24轮齐全且无悬空范围才接受。
第一版输出一致性仍可独立核验，但不挪用其乱序阶段数据。

## 复现与工程门禁

外部目录 `D:/workspace/talechime-benchmarks/windows-indextts-20261024`：
`fir-stage-r2-build-receipt.json`、源码快照、build jsonl/log、三个 on 的独立 stderr
和输出目录、off 控制、`fir-stage-r2-three-process-summary.json`、
`summarize_fir_stages.py` 与 `fir-stage-inputs.json` 保留可恢复身份及逐轮证据。

```powershell
# 先按输入清单核对本地权重及配置，并配置 CUDA 13 / cuDNN 9.14 DLL 路径。
$r = 'D:/workspace/talechime-benchmarks/windows-indextts-20261024'
$receipt = Get-Content "$r/fir-stage-r2-build-receipt.json" -Raw | ConvertFrom-Json
$env:INDEXTTS_CONVERTED_DIR = "$r/app/resources/indextts/models/2.5/c39ce5ba-candle-v1"
$env:INDEXTTS_PROFILE = '1' # 独立控制进程设为0
$env:INDEXTTS_PROFILE_OUTPUT = "$r/fir-stage-replay-NEW" # 必须不存在
& $receipt.exe bigvgan::tests::real_cuda_fir_stage_diagnostic --exact --ignored --nocapture --test-threads=1
```

release CUDA/cuDNN 原生构建通过；106普通测试通过、23显式忽略，Clippy all-targets
及 rustdoc 均 -D warnings 通过。fmt/diff 检查通过。没有新增生产优化开关或默认图策略。
