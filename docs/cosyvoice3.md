# CosyVoice3 后端

当前工作区使用 `2512-runtime-f64-front-v4` 固定资源配方；本轮文本前端改动尚未发行。
下文涉及 `2512-native-f64-v2` 的原生构建与耗时为源配方验证记录。
固定分发、许可与两类配方边界见[公开成品记录](records/resources/runtime-distribution-2026-11-05.md)。

`cosyvoice3` / 模型 `0.5b` 已接入 Talechime Registry、Rust Engine、CLI 与 JSON Lines v7。
当前为实验能力：零样本克隆、24 kHz 单声道原生流式输出；必须先导入参考 WAV 和准确转写。
没有内置音色；支持逐段 style 指令及显式 `cross_lingual` 模式，二者互斥；不开放执行级 continuation。`text_frontend` 默认 true，使用官方公开 WeText 路径；跨语言模式的新版前端真实模型与听感验收仍在进行。
播放速度使用现有播放层，不声明流式模型原生 speed。

LLM、Flow/DiT、HiFT 使用 Candle，参考语音 tokenizer 与 campplus 使用 ORT CPU。
`cosyvoice3` 启用 CPU，`cosyvoice3-metal` / `cosyvoice3-cuda` 启用相应平台加速；
显式设备不可用会失败。标准发行配置包含 `cosyvoice3-metal`，非 macOS 构建仍可使用 CPU。
默认后端选择不变。

## 准备模型

SDK 使用原生固定配方准备资源，不启动 Python。先调用
`ResolvedModel::native_preparation_plan()`，确认指纹后调用 `prepare_native_resources`；
`load_existing` 只校验并加载已发布的完整版本，不隐式下载。
详见[原生准备 API](native-resource-preparation.md)。

当前版本目录为 `ROOT/cosyvoice3/models/0.5b/2512-runtime-f64-front-v4/`。
新版增加固定 WeText FST 与许可，缺件、摘要错误或混用版本均不能就绪。
LLM、Flow、campplus 和语音 tokenizer 四个文件使用已逐字节核验的固定社区版本；
HiFT 由维护方从固定官方 checkpoint 原生导入后分发，保持 F0 的 F64 路径；完整 tokenizer 补齐官方
281 个特殊 token（含 EOF），配置采用固定版本。输入、输出摘要与许可见
`cosyvoice3/native.rs`（源构建）和 `cosyvoice3/distribution-inputs.json`（成品输入）及[核验记录](records/resources/native-preparation-2026-11-05.md)。
HiFT 共 246 个张量与独立 PyTorch 对照逐字节一致；社区 HiFT 存在 82 个差异张量，
没有替换原有精度要求或改写历史摘要。发布必须完整验证输出及配方标记，缺件、混合版本
和损坏资源均拒绝加载。固定组件和 ORT 前端对照通过；完整 CPU/Metal 对照仍分别有 2/1 项历史数值门禁差异，未放宽要求。两设备真实短文本合成均已生成有效 WAV，人工听感未评估，详见同一核验记录。

参考音色由 `ROOT/voices/cosyvoice3/` 下的模型音色存储管理，独立于不可变模型版本。
开发对照可使用 Python 环境，最终用户准备和加载不依赖它。权重与参考音频不进入 Git。

## 导入与使用

```sh
talechime --backend cosyvoice3 --model 0.5b --model-dir ROOT \
  voices import reader --name 朗读者 --text '参考录音的准确转写。' reference.wav
talechime --backend cosyvoice3 --model 0.5b --model-dir ROOT voices list
talechime --backend cosyvoice3 --model 0.5b --model-dir ROOT \
  --tts-device metal --voice custom:reader chapter.txt
```

参考 WAV 沿用共享音色存储的校验：1..30 秒、8..192 kHz、单/双声道、有效非静音 PCM。
推理时转换为 24 kHz，再提取参考 token、mel 和说话人向量；转写自动添加官方 end-of-prompt 前缀。
音色 remove、配置隔离与资源根参数复用现有命令。

Rust 库使用 `Engine::prepare(ModelOptions::new("cosyvoice3", root), progress)`，
显式选择 `model = Some("0.5b")` 和实际设备后，传入已导入的 `custom:reader`。
普通测试不下载或加载模型。显式端到端探针：

```sh
cargo run --release --locked -p talechime --features cosyvoice3-metal \
  --example cosyvoice3_backend -- ROOT custom:reader metal
```

探针通过 Engine 验证连续同 seed 请求、收取 PCM、首块后取消、取消后再合成和关闭。
它不打开播放器或写检查点，也不证明听感或文字完整性。

## 所有权与限制

专用推理线程持有一次加载的 Candle/ORT 模型，借用模型创建逐请求流；请求队列与 PCM 队列容量均为 1。
先取得 PCM 队列许可再推理下一块，消费者退出后在块边界取消；同步 ONNX/LLM/Flow/HiFT 运算无法中途抢占。
流销毁清除 KV，请求不共享 mel/token 缓冲，后续请求复用权重。
只有正常 final block 后发布 End；推理错误、长度上限和断连都不能冒充完成。
当前不声明持久音频缓存身份，防止参考文件变化后复用旧结果。

严格非流式整链数值门禁仍未通过，先前最大波形误差约 0.162；组件及流式对照结果见
[审查记录](records/cosyvoice3/cosyvoice3-review-2026-10-17.md)。接入后端不代表完全复刻官方、试听验收或 CUDA 实机验收。

Windows Ryzen 7 9700X CPU / RTX 5070 CUDA 的首轮计算库基准与被撤回候选见
[Windows 性能调查](records/cosyvoice3/cosyvoice3-windows-performance-2026-10-19.md)。
HF 原始权重摘要与 Mac 记录一致，但本轮 Windows 转换的 HiFT 与现有生产清单摘要
不同，尚未通过 Engine 资源准备门禁；没有替换清单或将开发计算库测试当作后端验收。

## 逐段风格

CLI 的 `--style`、Rust 的 `SpeechSpan.style` / 直接合成 style，以及 JSON Lines v7
的 `VoiceSpan.style` 进入同一 Instruct 链路。例如：

```sh
talechime --backend cosyvoice3 --voice custom:reader --style '请用悲伤的语气说。' chapter.txt
```

传入普通自然语言描述即可，不添加 `<|endofprompt|>` 等模板标记（显式拒绝）。
后端将指令包装为官方 assistant 前缀：LLM 使用指令并去掉参考语音 token 前缀，
Flow / HiFT 仍使用原音色参考，保持原生流式与取消边界。未提供 style 时继续使用原零样本路径。
指令是否达到预期语气需模型试听确认；普通接线测试不代表情绪质量验收。

## 文本前端（2026-11-11，未发行）

固定官方公开 WeText 0.1.0 路径、英文 inflect 7.5.0 数字规则、中文清理及
80/60/20 token 分组已移植到原生 Rust；25 条独立官方前端 fixtures 在 macOS CPU
通过。参考转写按官方 CosyVoice3 示例添加 assistant/end-of-prompt 前缀，
带此控制标记的转写按官方透传。CrossLingual 删除 LLM 转写和音频提示，
将官方默认前缀放入目标正文，因此该模式也走控制标记透传；Flow 仍保留音色参考。
含官方控制 token 的正文按上游跳过规范化；`text_frontend=false` 可显式关闭。
不声称复刻可选专有 ttsfrd 路径。来源和许可见计算库 [SOURCE.md](../crates/cosyvoice3/SOURCE.md)。

一个宿主语义段可包含多个模型内部组；仅最后一组正常结束后发送 End，错误仍传播。
生成有效 PCM 不证明每个词均已读出，也不关闭前述历史波形数值差异。
官方 WeText 的数字读法与产品中文年份策略分别核验，不做跨后端统一替换。

本轮真实三模式测试发现并修复跨语言缺少目标前缀，以及 Metal 无 LLM 音频提示时
零字节 token 缓冲区无法创建的问题；空条件现在使用有效存储的零长度视图。
显式 Metal 空条件测试通过，真实重跑结果与旧失败分开记录。
front-v4 在固定旧官方数值 fixture 上的 CPU 复核仍有两项差异：estimator 最大差
0.003486（门限 0.003），tokens→PCM 最大差 0.16024（门限 0.01）；其他阶段通过。
没有放宽门限或用新文本前端对照替代声学数值验收。
