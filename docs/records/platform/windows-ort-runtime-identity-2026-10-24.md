# Windows ORT 运行库身份与 Resume 回归

2026-10-24，Windows 11 / Ryzen7 9700X / RTX5070，当前工作树。
本轮是 xtask 工程验证，不是新的 GPU 合成或模型性能结果。

## 问题与改动

[ONNX 实测](../moss/moss-nano-onnx-windows-2026-10-24.md) 已证明同版本的不同 ORT
分发包含不同内核，显式动态构建可以恢复 CUDA 生产请求。原 Native 环境快照未包含
ORT_DYLIB_PATH，更未检查同路径 DLL 替换，不能据旧条件安全 Resume。

现在显式非空 ORT_DYLIB_PATH 必须是绝对路径、现存普通文件。环境快照同时保存：

- ORT_DYLIB_PATH 原路径字符串。
- ORT_RUNTIME_BUNDLE_SHA256：主库及同目录 onnxruntime_providers_* /
  libonnxruntime_providers_* 动态库的内容摘要，再对排序后的文件名/摘要表计算 SHA256。
  这个字段由工具计算，不从同名环境变量读取或注入子进程。

识别 dll、so/版本化 so 和 dylib。主库或 provider 内容替换、provider 新增/删除都会
改变身份；无关日志文件不改变。相对、缺失或目录路径失败，读取错误不记作空身份。
未设置或空路径不增加字段，保留旧静态任务原环境身份。

run 初始校验及每次进程调度/Resume 复用前都会重新计算；指纹变化拒绝当前任务，
不能拿已经提交的旧结果冒充新库。原始结果不改写，恢复完全相同的内容后可复用。
这是文件快照，不是运行时锁或完整 DLL 搜索路径证明；外部 CUDA/cuDNN/System 库及
实际加载模块仍需单独固定和审计。不要把它称为所有传递依赖的完整身份。
哈希发生在子进程计时前，可能预热文件缓存；加载耗时不能据此称为冷磁盘读取。

## 验证

两项无模型单元测试覆盖：同路径主库/provider 替换、恢复原字节、provider 增删、
无关文件、未选择路径兼容、相对/缺失/目录路径拒绝和跨平台库名识别。

显式 CLI 集成测试使用已冻结 performance_probe 的 synthetic fixture，创建伪 DLL
仅作身份输入，从不加载为 ORT：

1. 固定指纹运行夹具，生成实际已提交 attempt。
2. 同路径替换 provider 字节，Resume 以 performance flags 不符拒绝。
3. attempt 数及原 results.json SHA 均不变。
4. 恢复原 provider 字节，Resume 成功且不创建新 attempt，旧 results.json 不变。

回归2.93s通过。这验证真实 CLI/提交恢复流程，不代表伪 DLL 的 ABI 验收，也不是 GPU
推理。另对真实 PyPI ORT1.28 分发创建 fixture Job，记录到预期绝对路径，bundle SHA
a8cea5959da4acaf4f7c7e368880610ac4c601e0ab63326558f6827731456d36；只采集身份，未跑模型。

全部53单元测试通过，普通 CLI/help/process 检查通过；新的显式测试默认 ignored，
另行开启后通过。旧取消 CLI 测试未配置其环境时提前返回，不扩称本轮实测过信号。
release xtask 构建、Clippy --all-targets -D warnings、rustdoc -D warnings通过。

## 复现与证据

```powershell
$env:TALECHIME_PERF_PROBE_BUILD = 'D:\workspace\talechime-benchmarks\windows-nano-20261024\buffered-production-build.json'
# 该冻结探针需要已安装的 CUDA13 DLL 搜索路径；夹具不执行 CUDA 模型。
cargo test --locked -p xtask --test dynamic_ort_identity -- --ignored --nocapture
```

当前 Windows 外部 ONNX 证据根保留 ort-runtime-identity-tool.json（冻结 release xtask
路径/SHA及源码摘要）、xtask-dynamic-identity-cli-test.log、真实库身份 fixture-input。
没有修改旧工具/旧 Job，也没有宣称旧数据被本次检查重新验证。

以下后续验证关闭正式构建收据缺口；动态分发打包、完整官方条件排名和试听仍未完成。

## 后续：正式动态构建与 Native 请求

同日 Windows 11 / RTX 5070、上述固定源码和 ORT 分发上，perf build 现在允许显式
`ort/load-dynamic`。工具检查 Cargo 输出中实际 ort 库的 compiler-artifact features，
确认 load-dynamic 后才记录该依赖 feature；顶层探针请求字符串不能代替依赖构建证据。
只增加这一明确支持项，不放开任意依赖 feature；原收据 schema 和默认生产配置不变。

```powershell
# 先编译 release xtask；构建期间保持源码和文档不变。
target/release/xtask.exe perf build --probe performance_probe --features 'moss-nano-candle,ort-cuda,ort/load-dynamic' --output D:/workspace/talechime-benchmarks/windows-moss-onnx-20261024/xtask-native-dynamic-build.json
# 设置上述 ORT_DYLIB_PATH 及 CUDA13/cuDNN9.14 DLL 搜索路径后执行。
target/release/xtask.exe perf run --job D:/workspace/talechime-benchmarks/windows-moss-onnx-20261024/xtask-native-dynamic-job.json --output D:/workspace/talechime-benchmarks/windows-moss-onnx-20261024/xtask-native-dynamic-run
```

全新专用 release 构建6m41s成功，Source guard通过。合并 feature 为 moss、
moss-nano-candle、ort-cuda、ort/load-dynamic、performance-tools。synthetic fixture
通过后，使用标准 Native Job 直接运行生产 Registry/Backend/SynthesisStream，无 Python
桥接。3独立进程各1预热+5热轮，18轮成功、全部清理成功，音频均2.8s；PCM SHA256
均为 3cdb198c371fc7ae5dadf31ffb556f51bc63ddfd919ce6307a1630b15fec49a6。

热轮逐进程均值0.77490348、0.76871418、0.75805572s，组均值0.76722446s；首PCM
组均值0.07672371s。这比早期桥接0.694s更慢，构建、观测配置及运行时段不同，不能
归因于单一改动；本次是工具可复现性验证，不宣称模型提速或完成官方性能排名。

外部证据根保存 xtask-native-dynamic-build.json、同前缀 job/plan/summary.json、
xtask-native-dynamic-run/results.json（Native数组格式）和 xtask-native-dynamic-report/index.html。
dynamic-build-tool.json 指向冻结 xtask。新增单元测试拒绝错误库、test/example artifact
及缺失依赖 feature；全套 xtask 测试、Clippy all-targets、rustdoc 均通过，日志为
 dynamic-build-tests.log、dynamic-build-clippy.log、dynamic-build-rustdoc.log。
