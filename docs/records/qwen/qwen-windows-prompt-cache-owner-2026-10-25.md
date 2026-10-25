# Qwen Base：新 owner 磁盘 prompt 缓存验证

Windows11 / Ryzen7 9700X / RTX5070，Candle fork2266099d、CUDA13/cuDNN9.14，固定
Base06/17资源与参数沿用[生产报告](qwen-windows-production-2026-10-24.md)。本轮只
扩展ignored测试，没有生产推理或缓存格式改动，不更新性能数字。

## 确认确实经过磁盘路径

沿用前轮唯一临时B记录、生成参考SHA
`f9f9c4c5d47cf00538ba0b19895a3e85a45c1ecf116cf7d3381e252c0604595a`。
先完成首次编码、A/B/A、取消和同owner缓存恢复。然后：

1. 记录旧owner线程ID并drop Backend。现有Drop关闭工作队列并join线程，旧模型
   与prompt HashMap结束之后，才开始新owner加载。
2. 仅将测试记录内reference.wav副本重命名为reference.offline.wav，固定输入
   fixture及原A参考不动。保留voice.json与prompt-encoder-rope-v2.json。
3. 新Backend加载完成，断言线程ID不同；源码runtime::run为新owner创建空prompt
   HashMap。B请求完成时原WAV仍不可用，排除重新编码WAV替代磁盘缓存读取。
4. 新owner的完整B输出与旧owner逐位一致，A恢复也逐位一致；磁盘cache字节不变。
   恢复测试副本后显式删除临时记录，并按VoiceStore返回的精确目录路径断言不存在。

两模型均通过：Base06线程3→20，A/B原始F32样本63,360/59,520；Base17线程22→23，
A/B为76,800/65,280。完整请求有限24kHz mono并有正常End。这验证所测固定参考
的序列化/重载及owner状态，不证明任意缓存损坏可恢复、跨说话人质量或人工听感。

## 清理审计修正

前轮外部目录扫描使用裸ID模式cuda-reference-*，而VoiceStore实际目录为
voice_cuda-reference-*，因此不能将旧空扫描视作有效目录覆盖。原始记录不改写。
本轮按实际前缀重新检查，前后均零；测试内精确路径删除断言也通过。38份原模型/
参考/cache和固定B fixture的SHA前后不变。没有覆盖原音色，测试操作仅涉及自己
导入的唯一临时记录。

## 验证与复现

四种narration变体串行4 tests通过（104.45秒，仅正确性），同时回归既有取消、
部分预算失败恢复、单段续接、内置音色切换和Custom17 style；VoiceDesign独立
测试明确过滤。26普通测试通过、6忽略，相同release feature组合all-targets
Clippy及rustdoc -D warnings通过。

冻结exe SHA `dd1fc6720e089c35ed8606eeaa4a6e004485c5ca15f8775930ca89f08b8140d3`。
外部windows-qwen-20261024目录保存disk-cache-state-build-receipt.json、
disk-cache-state-real-tests.log、-before.json、-after.json及工程日志。复现沿用
原资源、CUDA/cuDNN PATH与TALECHIME_QWEN_STATE_REFERENCE/_SHA256，运行该exe：

```text
qwen::cuda_tests::real_cuda_ --skip real_cuda_voice_design_production_repeat --ignored --nocapture --test-threads=1
```

普通测试不加载模型或下载数据。新进程冷加载性能、多段漂移及跨框架质量仍待验证，
不将新owner正确性测试总耗时与官方热基线比较。
