# CosyVoice3 后续实现审查与修复

日期：2026-10-17。范围：本轮未提交的数值诊断、增量 LLM 与流式计算库；
需求依据为本次对话的阶段 A/B 提示词和仓库 AGENTS.md。未接入 Backend/CLI，未提交或发行。
源码审查分为需求符合性与实现/证据可靠性两条线，下面的问题均已修复。

## 需求符合性

- **P1：失败后复用半提交状态。** 原流式解码先更新 mel cache，再运行 HiFT；失败重试会重复
  拼接同一段 mel。LLM 也可能在更新 decoded/step 后遇到 KV 解码错误。现在 PCM 成功后才
  提交 mel/offset；任何推理错误将会话置为失败终态，后续调用拒绝重试。
- **P2：请求状态与模型 KV cache 没有绑定。** 旧的 `last_sampling` 在模型对象上，
  但 `IncrementalState` 可以属于旧请求或另一个模型。现在采样参数随状态保存，并用不透明的
  run identity 核验 KV 所有权；重新 prefill、清缓存或手动解码会使旧状态失效。
- **P2：终态和 token 输入不封闭。** 现在 source end 一旦声明就保持；拒绝继续追加、
  非法 token 和超出固定 CFM 噪声表容量的批次。空 token 源报错；最终块只发射一次。
  同时对非最终 Flow/HiFT 的不足预读输入提前报错，避免长度下溢。
- **P2：取消/慢消费者示例不证明成功。** 原 `while let Ok(...)` 吞掉推理错误并丢失长度
  上限标志。现在统一生产者逻辑传播错误，拒绝截断伪装成功，先取得队列容量再推理，
  块间让出 LocalSet。首块前接收端已关闭时不构建模型。同步在途块不能被抢占，取消粒度
  仍为块；模型历史缓存和工作区也不是恒定内存。

API 调整：`StreamingTts::next_pcm` 返回 `Result<Option<StreamChunk>>`；`None` 表示已结束。
`TokenStreamDecoder::extend_tokens` 返回 `Result<()>`，输入批次校验失败不部分追加。
原始 LLM 状态的终态字段改为只读访问器，防止外部把失败状态改回运行态。

## 实现与证据可靠性

- **P2：流式对照可能假通过。** 缺参考块/缺 mel 原先只打印；数值比较用 min 长度截断，
  还把日志字符串反解析成门禁。现在严格核对块集合、final、全部 mel 与维度，
  拒绝空输入和非有限数值，缺失或额外块均失败；使用数值变量直接判断。
  保留原流式 `mel RMSE≤1e-3`、`PCM RMSE≤0.1` 门禁，不代替原整链严格门禁。
- **P2：f64 探针污染后续模型。** `estimator.double()` 原地修改同一个模型；
  现在用 finally 恢复 f32。实跑 f64 锚点之后，后续官方 f32 estimator/mel/PCM
  相对原 fixture 的误差均为 0。
- **P2：诊断探针未调用实际实现。** 旧 `sine_probe` 手抄旧算法、没有使用构造的 SineGen2。
  现在直接运行实际 SineGen2；DiT detail 补齐官方 `after_attn/out` 导出，并修正 block 0
  的输入。新增 `--vocoder_stages`，替换原文档中的空导出脚本占位。
- **P2：正常推理无条件保留诊断张量。** HiFT 原来为普通调用也收集全部中间张量，直到返回
  才释放。现在诊断为可选参数；普通调用不收集、不格式化诊断名称。DiT 普通/traced
  forward 合为同一计算路径；移除 HiFT 的 `(Tensor, ())` 和 F0 探针的纯转发包装。

## 新定位的声源数值偏差

Candle 的标量 `/ 24000` 实现为乘一个已舍入的倒数，而官方 PyTorch 对该张量使用真除法。
两者单点差异很小，但谐波相位累积后可测。

在相同官方 F0、初相和噪声上做 Python 替换实验：真除法相对官方声源输出误差为 0；
仅替换成乘倒数，最大误差 **1.873821e-4**、RMSE **1.839522e-5**，复现旧 Rust 探针。
将实际 SineGen2 改为标量 tensor 的 `broadcast_div` 后，本机 CPU 探针最大误差降为
**1.490116e-8**，RMSE **2.059278e-9**，合并声源最大误差同为 **1.490116e-8**。

新增小型独立官方 PyTorch oracle `torch-sine-source.json`，不下载模型、不分配 300 秒
噪声表，验证生产 SineGen2 路径；保留此前 f64 cumsum 与 F0 f64 融合修复。

## 验证

- CPU 无 feature、`ort,metal` 两种配置：34 个库测试 + 5 个 example 回归测试通过。
  新覆盖 hop/预读边界、空源、封闭输入、容量、旧/异模型 KV 状态、解码失败重入、
  预关闭接收端、队列背压、推理错误及 token 上限传播。
- 格式、Clippy `-D warnings`、rustdoc `-D warnings`、五个 Python 工具语法检查通过。
- 官方参考新增 `--vocoder_stages` 实跑通过；DiT `--f64_out` 后继续 f32 推理实跑通过；
  Rust detail block 0/18 均运行完成，未发生缺键 panic。
- CPU 原组件对照通过；实际 SineGen2 对照通过。
- 加严后的 Metal 流式基准与长样例对照通过，块集合/长度与官方相同：
  baseline 为 32640 + 35520 = 68160 样本；long 为 32640 + 48000 + 96000 = 176640 样本。
- 实际 LLM 流式连续请求相同 seed 输出一致；首块前取消、首块后退出和慢消费者实跑通过。
- 原非流式整链门禁仍失败：`real_tokens_to_pcm` 最大误差约 **0.1617**，
  RMSE **0.00983**，保留 `1e-2` max-abs 门禁失败及非零退出码。
  共享官方 mel 的 Metal HiFT 最大误差约 **3.33e-4**。本轮不宣称整体数值等价。
- 未运行 CUDA、跨平台、播放器集成或发行验证。

证据均保存在 git 忽略的 runs：`cv3-review-tests-all.log`、`cv3-review-tests-cpu.log`、
`cv3-review-clippy.log`、`cv3-review-rustdoc.log`、`cv3-review-reference/`、
`cv3-review-dit-v2/`、`cv3-review-sine.log`、`cv3-review-stream-{baseline,long,short}.log`、
`cv3-review-lifecycle.log`、`cv3-review-stream-full.log`、`cv3-review-real-parity.log`。
本轮修改前的 crate 快照在 `runs/cv3-review-before/`，仅供本机审查，不进入 Git。
