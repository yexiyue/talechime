# MOSS Nano ONNX：local batch1 特化撤回

Windows11 / Ryzen7 9700X / RTX5070，ORT1.28动态分发 / ort rc.13 API21，CUDA13、
cuDNN9.14。外部证据根windows-moss-onnx-20261024。结论：节点减少且已测输出
保持，但无profiling的生产路径初筛未显示收益，撤回生产特化，不做正式ABBA。

## 假设与数值验证

[Native profiling](moss-nano-onnx-native-profile-2026-10-25.md)已验证local首维符号
batch，而Runtime所有输入均batch1。原型仅对local fixed sampled frame session
设置with_dimension_override("batch",1)，不改模型文件、权重、精度、输入或采样。
ORT将一部分MatMul/Add融合成Gemm，所以不以“只固定形状”代替输出对照。

先在Native Runtime独立原/候选两个session组间各重复两次：35帧codes、自然EOS及
原始PCM一致。随后扩大到下表，每组原/候选各两次，16次完整请求全部通过严格
codes、原始PCM、长度和EOS对照，无新增数值容差：

| 输入 | 截断seed | 最终帧数 |
| --- | ---: | ---: |
| 短句“你好，欢迎使用叙铃。” | 3560097011 | 35 |
| 同短句 | 42 | 34 |
| 同短句 | 7 | 36 |
| 段落“窗外的雨慢慢停了……一个新的故事。” | 3560097011 | 148 |

矩阵冻结测试exe SHA `ec7959bd419d03ce2c4f30daec9145ce0e1e3c4ec3756d355a0b340b2e79f7e2`，
48.35秒仅为正确性运行。完整文本、codes、逐轮PCM摘要及generation记录均在
local-batch1-matrix-*中；seed为Runtime直接值，不冒充公开Pinned42的跨请求策略。

## 节点变化不等于实际收益

同一短句两请求的local执行Node数148,752→101,808（−31.56%），CPU Node
12,384→0，CUDA Node136,368→101,808。出现4,896次Gemm事件；节点与Node跨度
不可当成GPU独占kernel时间。其他session未加维度覆盖。

之后冻结同feature的全新A/B生产探针。A SHA
`3ef1794d0549906bffbc14917a197f51bb38c4083c4c3566a57aa4896af7c190`，B SHA
`f3d58008e459edbe86028a82326ef79fa400d16f356451b9b6f53d569bdcf18a`，Source guard
均通过，二进制不同。候选仅Windows CUDA local graph生效；其余CPU/Mac/DirectML
保持原策略。测量以实际Registry/Backend/SynthesisStream请求进行，不带profiling，
两输入各一进程、1预热+2热轮，编译与GPU计时串行，观测off。

| 生产输入 | A两热轮均值 s | B两热轮均值 s |
| --- | ---: | ---: |
| 短句 | 0.805866 | 0.899951 |
| 段落 | 2.488505 | 2.656808 |

12轮均完整成功、cleanup通过，每个输入A/B生产输出摘要一致。这里是单进程
初筛，不能宣称稳定回退百分比，更不能凭节点少宣布提速。未定位变慢的单一根因，
不把Gemm融合、频率、分配器或缓存假设写成结论。按初筛结果停止扩大矩阵并撤回。

## 最终状态与恢复

撤回session_builder提取、生产维度覆盖和专用batch候选测试；原型完整保存在仓库
外local-batch1-prototype-source及冻结测试/生产exe中。没有留下失败候选的环境
开关或平台特例。撤回时git diff HEAD --binary摘要与A构建Source记录精确一致，
证据local-batch1-withdrawal-source-audit.json。未回退其他已有优化。

仅保留通用profiling门禁的test-only codes采集与对照：未启用测试时无额外字段或
复制。最终exe SHA `11fc43929f08a7f36ac20b4cbff6a25978c80b472b91f918a1790191d67d47d0`，
有/无profiling的codes、完成记录及四轮原始PCM一致，摘要仍为5321dbe4…；21普通
测试通过、2忽略。真模型13.99秒仅作正确性，不更新性能数字。

复现撤回候选使用冻结矩阵exe或A/B build/job及新输出目录；当前源码已无此生产
候选，不能用当前新构建冒充B。原有模型资源、DLL、构建、顺序及输入收据保留。

最终相同release feature组合all-targets Clippy、rustdoc -D warnings及fmt/diff
检查通过。12生产轮的72份资产SHA/大小全部复算，完整F32 WAV另按Native的
rate/channels前缀域核对PCM摘要；16请求矩阵的codes/完成记录与原始PCM摘要
独立复核。新增证据保留在local-batch1-final-summary.json及最终证据清单中。
