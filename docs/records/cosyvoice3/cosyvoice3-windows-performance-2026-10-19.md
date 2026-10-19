# CosyVoice3 Windows CPU / CUDA 性能调查

本轮从 2026-10-18 开始准备，于 10-19 继续测量。设备为 Windows 11 / Ryzen 7 9700X
（8 核 16 线程，约 16 GB 系统内存）/ RTX 5070（约 12 GB 显存，驱动 616.56），CUDA 13.0，Candle
0.11.0 fork `bafb03db81c3eae48e63ff28d6b30f5a6576ba38`。模型采用 F32，F0
沿用既有实现的 CPU F64；不是 Mac M4 Pro 的 Metal / Accelerate 结果。

Mac 已有优化记录见 [2026-10-17 报告](cosyvoice3-performance-2026-10-17.md)。
实验基于主线 `dfa0291` 与本轮未提交的探针/候选；候选已撤回，最终保留边界见后续节。

合并说明（2026-10-19）：本轮实现提交 `1ebcf45` 已与 Mac 侧主线 `b790eae` 合并。
工作区 Candle 统一更新至 `2266099d92265112ecc0d56ecee1761e3cc2ac00`；上文环境及下文
性能数字仍对应测量时的 `bafb03db…`，不将合并后的版本视为已重跑真实模型性能验收。
Mac 侧结果见 [VoxCPM Metal 报告](../voxcpm/voxcpm-metal-performance-2026-10-18.md)，
其收益不外推至 Windows CUDA 或 CosyVoice3。

合并后补跑：CosyVoice3 默认 CPU 库测试 36 项、all-targets Clippy 与私有项/examples
rustdoc（均拒绝警告）、工作区格式检查、VoxCPM Python 工具测试 16 项与 WAV 比较测试
2 项、skill 结构校验通过。CUDA 真实模型基准与 Mac 实机测试未在合并后重跑；下文
CUDA 验证记录仍指合并前版本。

## 官方权重与准备边界

从 Hugging Face `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 固定 revision
`29e01c4e8d000f4bcd70751be16fa94bf3d85a18` 下载原始权重、ONNX 与 tokenizer。
原始文件、转换模型、开发 Python 环境和官方参考 WAV 均位于
`~/.talechime/resources/cosyvoice3/`，未写入仓库。保留逐文件下载摘要收据。

原始 `llm.pt`、`flow.pt`、`hift.pt` 的 SHA-256 均与此前 Mac 原始权重记录一致。
转换规则沿用仓库脚本，固定官方 tokenizer 源码 `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`。
Windows Torch 2.7.1+cpu 与 2.8.0+cu128 转换的 HiFT 摘要相同；最终开发基线使用 2.7.1+cpu。

| 转换/复制产物 | 与现有生产清单比较 |
| --- | --- |
| config.json、LLM、Flow、tokenizer.json、campplus、speech_tokenizer_v3 | 大小与 SHA-256 全部一致 |
| HiFT | 大小 96,324,868 bytes 相同；Windows SHA-256 为 `4cde9c7be7e744390012774875aa157945a3ce757ec8fa01c279f1290ab9e240`，生产清单为 `475ac02be5d85f5b13da1e075a217ddb71e14f109f327919609004858c75bd2b` |

HiFT 摘要差异尚未定位到具体张量，不能仅凭相同原始权重断言跨平台转换逐位一致。
没有修改生产清单、绕过资源校验或宣称 Engine prepare 通过；以下为计算库开发基准。
转换脚本的 JSON 原先在 Windows 写入 CRLF，现固定 UTF-8/LF；配置恢复到既有摘要
`455c574d7ccefeed2419d3ad67c87d17545eed2644f65ca46733d126d5790b26`，不更改配置内容。

## 探针与测量修正

- 既有 `bench` 只接受 CPU/Metal，补充显式 CUDA 设备，禁止静默使用 CPU。
- 阶段同步改用实际 Device 的 synchronize，补齐组件预热结束同步；记录 Rayon 环境与 ORT 线程各自口径。
- 新增 `prepare` 模式，从官方参考 WAV 通过本地 ORT 前端准备固定特征和分词结果。
  这些是实现内 A/B 输入，不是独立官方数值 oracle。
- 可选逐轮 WAV 导出在计时外完成，阶段 JSON 保留完整 speech tokens，避免只比较长度或首轮。
- 旧 warm 模式同时保留 staged 与完整 pipeline 两份权重。新增 `--no-stages`，
  单实例测完整合成，报告明确 `stage_attribution=false`；此时不伪造未采集的阶段/token 指标。
  原归因模式保留用于诊断，不能把其额外内存压力下的耗时当作生产单实例结果。

原始日志、候选二进制测量及 WAV 位于忽略目录 `target/tmp/cosyvoice3-*`。
基准期间不并行编译或推理。初筛不锁频、不变更电源计划，桌面程序仍驻留；
不能称为 GPU 独占或正式统计显著性证据。设备间输出不要求逐位一致。

## 初筛与候选

初筛固定文本“你好，欢迎使用叙铃。”、官方参考，seed 42，10 步 CFM，
不重新提取参考，预热与加载不计入热轮。初筛转写尾字误写为“哦”，后续已更正官方的“呦”，
原输入保留在 `short-transcript-oh/`。下表只用于同输入实现内筛选，不作为准确转写下的
正式音频验收或官方对照；初筛 CPU / CUDA 均生成 66 tokens / 2.64 秒音频。

| 设备 / 路径 | 完整合成热轮秒数 | Flow 阶段秒数 | 状态 |
| --- | --- | --- | --- |
| Windows CPU，Rayon 8 | 14.4825 / 15.8134 | 11.0229 / 10.9019 | 初筛，Flow 为主要热点 |
| Windows CPU，Rayon 4 | 16.6245 / 16.1122 | 13.8430 / 12.0332 | 初筛，未见收益；与 8 线程 WAV 一致 |
| Windows CPU，Rayon 16 | 18.1037（单热轮） | 12.5411 | 初筛，不用单轮推定稳定线程排名 |
| RTX 5070 CUDA eager | 0.9407 / 0.9916 | 0.3783 / 0.3838 | 初筛基线 |
| RTX 5070 CUDA，逐 solve 的 DiT 步图 | 0.8773 / 0.9572 | 0.3288 / 0.3302 | 输出一致，但完整合成仅约 5% 改善，不足以转正 |

DiT 局部探针使用真实 Flow 权重、合成条件，两种长度（52/112）、两种掩码，
不同 timestep 和输入连续回放逐位一致；十步 eager/replay 初筛约改善三到四成，
录制约 10–15 ms。它不能代替上表端到端结果。

初次 capture 失败于 h2d 布局参数缓存缺失。定位为输出转置 view 在 recorder 内首次
物化，未被模型 eager 预热覆盖。共享 recorder 在 capture 前预热同一输出复制后，
局部真模型回放通过；补充免权重 CUDA 回归，防止仅依赖 CosyVoice3 模型复现。

## 候选撤回与最终保留范围

四配置各一个独立进程、每进程一轮预热与两轮热测，分完整/流式测量。
这不是正式 ABBA；只用于判断是否值得扩大验证。固定 eager 基线，完整/流式的均值如下：

| RTX 5070 CUDA 配置 | 完整合成秒数 | 流式秒数 | 相对 eager 耗时减少 | 输出 |
| --- | ---: | ---: | --- | --- |
| eager | 0.9525 | 1.3429 | 基线 | 自然完成 |
| 每 solve 的 DiT 步图 | 0.9002 | 1.2506 | 5.5% / 6.9% | 各热轮 WAV 与 eager 逐字节一致 |
| 每层单 token MLP 图 | 0.9349 | 1.2971 | 1.9% / 3.4% | 各热轮 WAV 与 eager 逐字节一致 |
| 两项图同时启用 | 0.8877 | 1.2588 | 6.8% / 6.3% | 各热轮 WAV 与 eager 逐字节一致 |

均未达到本项目 10% 的候选门槛，因此没有运行昂贵的正式 ABBA，也没有保留运行时代码、
环境开关或新增依赖。不能将局部十步回放的三到四成改善宣传成整链收益。

另测加载时保留线性权重转置 view、交给 cuBLAS 处理布局：同一二进制 eager 两轮
0.9224 / 0.9450 秒，候选 1.0003 / 0.9800 秒；66 个 token 数相同，但 WAV 已不同，
63,360 样本 max_abs 0.25209 / RMSE 0.01364。性能与输出两项都未通过，已撤回。
没有更改 GEMM 布局、归约、精度、采样或帧上限来获取收益。

最终保留的是 Windows 基准入口、同步/记录修正、免权重 CUDA 算子测试、转换 JSON
固定 LF，以及共享 recorder 的输出复制预热修复。CosyVoice3 模型热路径和 CPU / Metal
默认不变，CPU 线程初筛也没有变成库级全局线程设置。当前尚无可宣称的新增端到端提速。

下一轮 CPU 候选是 Windows BLAS/GEMM 路径；这可能改变浮点累加与音频，需先独立
约定数值/质量门槛，不套用 Mac Accelerate 的收益或已接受数值范围。CUDA 则需继续区分
算子实际计算/权重带宽与分派成本，不为低收益子图增加缓存框架。

## 修正转写、单实例后的可复现基线

参考转写更正为“希望你以后能够做的比我还好呦。”（带官方示例的 end-of-prompt
前缀），重新提取固定特征；完整合成使用 `warm --no-stages`，流式原本即只有一个模型。
各一个独立进程、一次预热，CPU 三热轮、CUDA 各五热轮。未锁频，仍有桌面程序驻留，
加载/参考准备/音频导出不计入合成时间。以下是当前计算库绝对基线，不是优化前后对照。

| 设备 / 模式 | 音频秒数 | 热轮均值秒数（轮次范围） | RTF | 首 PCM 秒数 | Windows 进程峰值工作集 |
| --- | ---: | --- | ---: | ---: | ---: |
| Ryzen 9700X CPU，Rayon 8，完整 | 2.44 | 14.8607（14.6088–15.1036） | 6.0905 | — | 4.55 GiB |
| RTX 5070 CUDA，完整 | 2.44 | 0.9120（0.8936–0.9470） | 0.3738 | — | 2.37 GiB |
| RTX 5070 CUDA，流式 | 2.40 | 1.2849（1.2619–1.3050） | 0.5354 | 0.7171（0.6967–0.7361） | 2.37 GiB |

峰值工作集由 Windows GetProcessMemoryInfo 采集，含加载期，只表示 host 进程驻留内存，
不是 GPU 显存；没有把两者相加。各模式内逐轮 WAV 摘要相同；流式每次两块、收到正常
final 且未达 token 上限。完整与流式时长不同，不能声称二者对应相同 token 负载，
也不要求 CPU 与 CUDA 逐字节相同。没有通过真实播放、ASR、听感或长文本验收。

原始轮次、完整元数据、源 diff、二进制/权重/WAV 摘要与 Windows 内存计数收据位于
`target/tmp/cosyvoice3-opt/final-baseline/`。二进制 SHA-256 为
`7cff7bf54dd1ce010df4d11069b3a2fc7a5ad393700483ac5049684e110e35a6`。
不把此表与前面转写不同、双模型归因模式下的初筛相减后宣传为提速。

## 工程验证与未完成项

- 默认 CPU `cargo test --locked -p cosyvoice3 --lib -j 1`：36 passed，不下载模型。
- CUDA/ORT 组合的库测试：36 passed，GPU 测试默认 ignored；bench 另通过真实 CLI 运行和 Clippy 检查。
- 显式串行运行免权重 CUDA 测试：DiT 算子布局与 recorder 转置输出回归各一项通过。
- CUDA/ORT 组合两包 all-targets Clippy、私有项/examples rustdoc（warnings denied）、
  格式与 diff 空白检查通过；CUDA release bench 构建成功。
- Python 转换脚本语法检查、config 摘要核对、skill 校验与本地引用检查通过。

release 链接仍有既有 LNK4098 LIBCMT 冲突警告，未用 NODEFAULTLIB 或修改全局 CRT
掩盖它。本轮没有发行、提交上游 PR 或复测 Mac，也没有修改依赖来源/版本。
此前原有的官方整链波形门禁与本轮 HiFT 生产清单差异均保持未通过状态。
