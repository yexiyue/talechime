# Windows 动态 ORT：主 CLI 部署验证

Windows11 / Ryzen7 9700X / RTX5070，ORT rc.13 / API21保持固定。这是交付路径验证，
没有模型算子或性能提升声明；外部证据根windows-moss-onnx-20261024沿用既有报告。

## 显式构建入口

backends及主CLI新增opt-in `ort-dynamic`，转发现有ort/load-dynamic，不改变默认
构建、ORT版本或发行feature。首次--locked拒绝后离线解析锁文件，仅给ORT增加
已存在的libloading0.9.0依赖边；没有新增、删除或升级任何package。随后--locked
release主CLI构建通过，实际ORT compiler-artifact包含load-dynamic与cuda。

```powershell
cargo build --locked --release -p talechime --no-default-features --features cli,moss,ort-cuda,ort-dynamic -j1
# 使用已固定并验证整套来源、架构与许可证的 ORT GPU 分发。
$env:ORT_DYLIB_PATH = 'D:/your-runtime/onnxruntime.dll'
# CUDA/cuDNN 的依赖目录也须在本次进程的 DLL 搜索路径中。
target/release/talechime.exe --backend moss --tts-device cuda --config YOUR_ISOLATED_CONFIG --model-dir YOUR_RESOURCES --checkpoint-dir YOUR_ISOLATED_CHECKPOINTS YOUR_TEXT
```

该仅ONNX构建在catalog中沿用model=None，省略--model选择Nano ONNX；同时编译
Candle的多模型构建才使用显式nano标识。首次smoke误用--model nano，按目录规则
在加载前拒绝，日志保留。没有新增兼容协商、改变模型标识或绕过资源检查。

## 暂存与真实请求

主程序SHA `052c86c6c6af0acad7990ebb4366843140a88531c49b1b8682dbd96b7af58b5f`。
仓库外cli-dynamic-stage暂存exe及已实测PyPI ORT1.28 GPU分发中的主库、CUDA及
shared provider三个DLL，同时保留项目已有ORT1.28 LICENSE/ThirdPartyNotices。
没有挑换单一provider。CUDA13/cuDNN9.14仍从本机独立目录加载，未宣称全依赖自包含。

使用已有~/.talechime/resources，短句“你好，欢迎使用叙铃。”，seed42，显式CUDA，
关闭续接。配置volume0与checkpoint在全新外部目录，不读取/覆盖用户设置，不触发
默认路径迁移。主CLI正常资源校验、创建模型、生成并实际消费音频，进程exit0。
checkpoint原文SHA正确，resume_byte=30等于UTF8总长，completed=true；不是只凭
generation结束宣布播放完成。静音只影响播放音量，不改合成输入或精度。

进程存活期间采样实际加载模块，三个ORT库均来自cli-dynamic-stage，版本
1.28.0.20260722.3.0368187；cuDNN9.14.0.64、cuBLAS13路径另记。采样是已加载模块
证据，不宣称枚举了每个瞬时/传递依赖；模块文件摘要随交付证据冻结。

## 边界

此次证明主CLI显式动态构建能完成真实CUDA请求，关闭了“只在performance_probe
构建中跑过”的缺口。标准cargo-dist仍CPU runtime，未改dist-workspace或发布流程，
也未生成/发布CUDA安装包。完整依赖打包、干净机器安装、CUDA/ORT许可证分发清单
及非此GPU的兼容性仍须独立验收；手工暂存成功不能替代发行验收。

工程门禁：主CLI/backends的release普通测试合计35通过、1忽略；既有可选真模型
测试的专用环境未设置，不计为本轮模型验证。实际真模型证据是上述独立主CLI请求。
同feature组合all-targets Clippy与rustdoc均-D warnings通过，fmt/diff检查通过。
日志cli-dynamic-{tests,clippy,rustdoc}.log与构建原始JSON保留；失败的首次--locked
和错误model smoke也保留，不覆盖或改标成功。新增feature本身不自动下载运行库
或模型；ORT已有build依赖行为保持原约束。
