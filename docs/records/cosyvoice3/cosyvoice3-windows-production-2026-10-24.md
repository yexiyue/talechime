# CosyVoice3 Windows 生产资源复核（未通过）

设备 Ryzen 7 9700X / RTX 5070 12 GB / Windows / 16 GB RAM。工作区 main42a7887
加当前 Windows 修改；本轮 fetch origin/main 后无新增主线提交。历史 10-19 计算库
数字不是当前生产基线，本轮没有加载不合格资源运行 Engine，也没有更改生产清单。

证据根：`D:/workspace/talechime-benchmarks/windows-cosyvoice3-20261024`。
原模型仍在 `~/.talechime/resources/cosyvoice3/`。逐文件读取当前 production manifest，
重算本地全部大小和 SHA256，结果 production-manifest-check.json：

| 文件 | 当前生产清单 |
| --- | --- |
| config.json / llm.safetensors / flow.safetensors | 全部匹配 |
| tokenizer/tokenizer.json / campplus.onnx / speech_tokenizer_v3.onnx | 全部匹配 |
| hift.safetensors | 大小 96324868 bytes 匹配，SHA 不匹配 |

HiFT 本机 SHA `4cde9c7be7e744390012774875aa157945a3ce757ec8fa01c279f1290ab9e240`；
生产期望 `475ac02be5d85f5b13da1e075a217ddb71e14f109f327919609004858c75bd2b`。
已检查本地目标目录与忽略目录，没有找到另一份生产摘要匹配的 HiFT。
这使问题从“资源可能过期”缩小到一个转换产物，但不证明其与 Mac 产物数值等价。

## CPU dispatch 转换实验

从固定 Git e591c94 取回 cosyvoice3_convert_weights.py，存为 frozen_converter.py，
SHA `b6201b74e11ecff4de09c51115944a6f14f6c23939e7977708dda9bb3ac24187`。
使用原官方 hift.pt（原来源 revision29e01c4…）、原 convert_hift/fuse_weight_norm
规则；F0 predictor 的 weight norm 在 F64 融合，其余 F32，未手改融合公式或容差。
核验脚本 check_hift_conversion.py SHA
`bce6de6e67bc2876f5a3b071d726ba15a74e8321644f33120d25e5ab98a2772c`。
独立进程各设 ATEN_CPU_CAPABILITY，torch 2.7.1+cpu、1 线程：

| 设置 | 实际报告 capability | 相对本机原 HiFT 的张量/元素位差 | 输出文件 SHA |
| --- | --- | --- | --- |
| default | NO AVX | 0 / 0 | 4cde9c7b… |
| avx2 | AVX2 | 0 / 0 | 4cde9c7b… |
| avx512 | AVX512 | 0 / 0 | 4cde9c7b… |

覆盖全部 246 张量，其中 5 个 F64；key、shape、dtype 一致。三个新文件均与本机
原转换文件 SHA 相同，没有复现生产期望 SHA。逐张量结果、脚本、日志和独立转换
产物留在证据根，不覆盖原模型。没有下载或修改系统 CPU 设置，环境变量仅在该次
子进程链生效。不同 capability 标签不等于每个算子的实现/结果都改变；本轮实验
排除了这三种设置能修复当前摘要差异，不能据此定位原跨平台差异的具体原因。

## 未完成与后续边界

原始官方权重相同、Windows 重转换稳定，仍不能证明未取得的 Mac 转换产物的每个
张量相同；也不能区分其数学融合与序列化差异。严格生产门禁仍失败，不把新算 SHA
加入清单来冒充解决，不将已有计算库探针改名为生产验收。
要解锁需取得与生产摘要匹配的 HiFT 做逐张量对照，或在能展示误差/质量证据后审查
明确的新转换版本；不能只用相同文件大小或原始 checkpoint 来源代替这一步。
本轮没有新性能结论、试听验收或上游发布，继续推进其他可验证后端。

## 10-25：Torch版本与序列化的独立排查

固定e591c94的官方参考requirements实际要求Safetensors0.8.0。新增实验复用隔离的
Torch2.8.0+cu128环境，但转换仍在CPU、1线程执行，原converter和hift.pt不变。
torch28-threads1.{json,log,safetensors}中246张量相对本机原转换物的位差仍为0，
文件SHA仍为4cde9c7b…，没有恢复生产期望475ac02b…。

该2.8环境实际使用Safetensors0.5.2，因而另做只重序列化、不重算权重的控制：
在隔离0.8.0+windows.headerio1运行时加载现有文件并另存，重新逐张量核对全部位模式。
reserialized-sf08.json记录246张量精确相同，文件大小和SHA也仍不变；该IO fork的
来源/许可证见Windows元数据提交记录。它没有修复当前摘要差异，不能把发现的版本
不一致直接宣布为根因，也不能据此断言所有跨平台融合结果相同。

两个新产物仅保存在既有仓库外证据根，未覆盖模型原件或修改manifest。
re-serialization脚本reserialize_hift_sf08.py、原转换检查脚本及摘要冻结保存。
当前仍需取得Mac上匹配生产摘要的实际文件做张量/序列化对照；路径问题已异步提出，
不会阻塞其他后端推进。未新增生产加载、性能或质量结论。
