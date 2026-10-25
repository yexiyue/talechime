# MOSS Nano ONNX：Native Runtime CUDA profiling

后续batch1特化已完成数值与生产初筛并撤回，见
[候选结论](moss-nano-onnx-local-batch1-2026-10-25.md)；当前生产不启用该特化。

Windows11 / Ryzen7 9700X / RTX5070，ORT1.28动态分发、API21/ort rc.13，CUDA13与
cuDNN9.14。外部证据根windows-moss-onnx-20261024。本轮不改生产推理、provider
参数、模型或分块；新增默认ignored的实机诊断及test-only profiler结束接口。

## 执行路径与输出控制

测试在独立owner线程构造现有Runtime，session factory沿用生产的intra_threads4、
默认CUDA provider及error_on_failure。使用Placement::Cuda，保留生产run()的
GPU cache I/O Binding；这不是官方Python的host cache，也不是完整Registry/
SynthesisStream或播放路径。测试新增的finish_profiles接口只在测试/CUDA或已有
DirectML诊断feature下编译，未增加产品配置或运行时环境开关。

固定生产normalize/tokenizer、Weiguo、截断seed3560097011、无续接。先独立加载
不带profiling的Runtime执行两次，再释放并独立加载带profiling的Runtime执行
两次，全部走同一生成代码。每轮35帧、自然EOS、2.8秒、268,800个48kHz stereo
F32值；四轮原始PCM逐位相同，SHA均为
`5321dbe4748b24ff23b2dd3768e2fc068a2524ef13db4f058e5e88adf73b9fb8`，
也与既有Native原始块审计摘要一致。不是宿主处理后WAV或量化PCM16。

冻结测试exe SHA `edf46eaea82effe630d07fc170e1569e253b0cf452a8ebf8bc1908c0628ce3aa`，
测试13.23秒通过，仅正确性/诊断。实际加载模块路径与版本另外采样；ORT三库来自
cli-dynamic-stage，cuDNN来自独立9.14目录，没有Torch导入的混合组件条件。

复现使用native-profile-test-receipt.json的exe，设置已有
TALECHIME_NANO_ONNX_ROOT、全新TALECHIME_NANO_ONNX_PROFILE_OUTPUT与固定ORT_DYLIB_PATH/
依赖PATH，过滤real_cuda_runtime_profile_preserves_raw_pcm，并传
--ignored --nocapture --test-threads=1。普通测试不会创建目录或加载模型。

## 两请求合计的Node跨度

| graph | Session次数 | CUDA Node数 | CPU Node数 | CUDA跨度µs | CPU跨度µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 2 | 2,176 | 200 | 33,577 | 1,407 |
| global decode | 70 | 65,170 | 3,640 | 779,655 | 25,022 |
| local fixed sampled frame | 72 | 136,368 | 12,384 | 1,588,714 | 86,754 |
| codec step | 24 | 30,216 | 11,952 | 425,303 | 81,974 |

encoder本轮没有Node调用。与[官方profile](moss-nano-onnx-cuda-profile-2026-10-25.md)
一样local节点跨度最大；本表是ORT节点跨度，包含预热及同步/分派相关成本，不能
当GPU独占kernel时间或整链墙钟。Native与官方的cuDNN组件条件也不同，不能按
两表点差宣称某个库更快或给出跨框架质量等价结论。

Native local前列为CUDA Add138,585µs、Concat111,078µs、MatMul105,757µs；codec
前列Mul55,011µs、Concat38,034µs、Conv34,080µs。这进一步否定“先改codec卷积
一定最划算”的假设，不证明所有小算子均dispatch-bound。CPU形状节点仍存在，
不能直接假定整个图符合CUDA Graph限制。

## 下一个可证伪候选

使用独立onnx1.19.1检查环境，仅加载graph protobuf，load_external_data=False，
没有改官方Python环境或权重。实际local graph四个输入都以符号batch为首维，
其他形状依次为768、16×1024、标量随机数组与16随机值；Runtime始终构造batch1。
原始local图4,949节点，含255个Shape、255个Gather和397个Unsqueeze。

因此可以单独检验session维度覆盖batch=1是否使形状处理被常量折叠。当前只是
候选依据，未修改生产session，未声明节点已消除或测到收益；必须先用同输入
对照完整codes/原始PCM/EOS，再检查优化后图与无profiling正式计时。不能以
“有Shape节点”代替实测，也不修改模型文件或放宽资源SHA门禁。

## 工程门禁

21项普通测试通过、2忽略，既有可选无配置测试不扩称真模型验证。release相同
feature组合all-targets Clippy和rustdoc -D warnings通过。初次编译因ORT
Error<SessionBuilder>含非Send/Sync恢复状态而拒绝直接?转换到anyhow；按现有
设备工厂在owner内格式化错误，不增加unsafe Send、不跨线程移动builder。
fmt/diff检查通过，原失败编译日志与最终构建/运行日志分别保留。
