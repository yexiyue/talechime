# MOSS Nano ONNX：CUDA 节点 profiling 控制

Native Runtime的独立profiling、GPU cache binding与原始PCM对照已补齐，见
[Native后续](moss-nano-onnx-native-profile-2026-10-25.md)；两个执行环境不混算。

Windows11 / Ryzen7 9700X / RTX5070。沿用固定官方源码及ONNX资源，TTS revision
f52645cb467506d8e18e746ddd59482685b74e58、codec revision
ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae；具体输入文件SHA由外部job固定。
证据根`D:/workspace/talechime-benchmarks/windows-moss-onnx-20261024`。
没有修改Rust模型、ORT provider参数或生产分块；本轮是诊断，不是提速结果。

## 先固定会受计时影响的控制流

为官方ONNX适配器增加显式profile_sessions选项，默认关闭。只在创建InferenceSession
时沿用原SessionOptions并开启ORT profiling，结束后记录各profile文件名/大小/SHA。
xtask继续管理输入、进程、预算、超时、清理及原音频资产。profile文件另由
summarize_cuda_node_profiles.py逐项核对索引，原始文件保持不变。

初次采用官方动态codec预算，prompt与35帧codes和旧匹配输入基线一致，但PCM摘要
不相同。该profiling运行两次均每块1帧，旧基线codec未固定，且旧逐轮PCM本就不同。
源码预算依赖已交付时长与首音频后的实际耗时，可在1/2/4/8帧间选择；不能用相同
seed/codes推断同分块，更不能直接把PCM差异归因于profiling或当作模型数值回退。

随后在同一当前适配器/资源/环境中，分别开启和关闭profiling，各1进程、1预热+1轮，
显式固定codec预算3。两边prompt、35帧codes、停止条件及分块计划均相同：
11块×3帧加末尾2帧；四份原始F32 PCM逐位一致，摘要
`2cedf4c6a47c51622f9c3a6114ace0bbfc5d5e3eed8e7466879f0c234b7abde3`。
它是诊断控制，不替换原官方动态策略或正式基线，不把固定3帧作为产品优化。

实际ORT1.28；Python官方路径仍先导入Torch，cuDNN根DLL来自既有Torch目录，
路径/SHA记录于runtime.json。与Native cuDNN9.14组件分发未完全对齐，本轮不宣称
其绝对耗时或PCM就是Native结果。两者既有差异见[原报告](moss-nano-onnx-windows-2026-10-24.md)。

## 固定3帧的节点跨度

八个session profile均保存，其中四个graph本轮没有Node执行事件。下面合计包含
预热与另一完整请求；Session/Node跨度不是GPU独占kernel耗时，不能求和当整链墙钟，
也不能从节点数或注册CUDA推断全图均在GPU。没有CUPTI kernel分类轨迹。

| graph | session run次数 | CUDA Node数 | CPU Node数 | CUDA Node跨度合计（µs） | CPU Node跨度合计（µs） |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 2 | 2,176 | 200 | 154,845 | 1,731 |
| global decode | 70 | 65,170 | 3,640 | 807,011 | 24,379 |
| local fixed sampled frame | 72 | 136,368 | 12,384 | 1,746,553 | 88,993 |
| codec step | 24 | 30,216 | 11,952 | 553,415 | 82,968 |

local CUDA跨度前列为Add 153,616µs、Concat117,481µs、MatMul116,723µs，另有
大量Reshape/Unsqueeze/Slice等节点；codec前列Conv142,745µs。prefill的冷Equal
跨度突出，但未分离冷启动，不能直接当作稳态优化目标。局部72次包含终止判定，
不能将它误写为72个最终音频帧。

这支持下一步先在Native路径复核local分派及数据边界，不能仅因codec有Conv就
改cuDNN算法，亦不能凭小算子多就宣布dispatch-bound已证明。CPU节点可能是明确
分区的形状操作；error_on_failure只约束EP注册，并不禁止这种分区。ORT另报告
插入复制节点，整图CUDA Graph的可捕获性需要独立检查，不能盲目开启。

## 验证与恢复

三个诊断进程全部正常结束/cleanup通过，普通运行不下载模型。有/无profiling的
固定3帧控制由official-cuda-fixed3-profile-output-audit.json保存精确对照；节点
汇总official-cuda-fixed3-node-summary.json，初始动态分块汇总单独保存。
复现用冻结工具运行official-cuda-profile-job.json或fixed3的control/profile job，
每次选择新输出目录；profiling耗时不混入原0.767秒Native或官方热基线。
Python语法、fmt/diff检查通过；未改Rust运行时，不重标历史工程测试为本轮新测试。
