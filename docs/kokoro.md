# Kokoro 官方权重原生后端

Kokoro 是可选的中文预置音色后端，当前源码提供 `1.1-zh` 和显式启用的
`1.1-zh-g2pw`。声学网络直接对照 hexgrad 官方 Python 实现移植到 Candle；
运行时不调用 Python。0.6.0 标准发行配置已编入该后端，本机 CPU/Metal 对照结果见
[实现记录](records/kokoro/kokoro-native-validation-2026-10-27.md)。

输出为 24 kHz 单声道，每个模型片段生成完毕后交付 PCM。没有原生流式、音色克隆、
风格指令或自定义读音接口。仅支持安装的官方 `zf_*` / `zm_*` 中文音色，
参数 `speed` 范围为 0.5–2.0。嵌入英文和无法处理的符号明确报错。

## 编译与执行

facade 和 backends 的开关一致：`kokoro`、`kokoro-metal`、`kokoro-cuda`、
`kokoro-g2pw`。Cargo 默认 feature 不启用 Kokoro；标准发行配置启用 kokoro-g2pw，macOS 另启用 kokoro-metal。默认后端仍为 MOSS Nano。
CPU 使用 F32；在 macOS 上启用 Accelerate。Metal/CUDA 开关选择解码器设备，
编码器、时长/F0 预测和 FFT 保留 CPU，因此 GPU 路径是混合执行。
CUDA 尚未完成本阶段硬件验收，不能由 feature 存在推断为已验证。

```sh
cargo run --release --locked -p talechime --no-default-features \
  --features kokoro --example kokoro_backend -- RESOURCE_ROOT zf_001 cpu

cargo run --release --locked -p talechime --no-default-features \
  --features kokoro-metal --example kokoro_backend -- RESOURCE_ROOT zf_001 metal

cargo run --release --locked -p talechime --no-default-features \
  --features kokoro-g2pw --example kokoro_backend -- RESOURCE_ROOT zf_001 cpu 1.1-zh-g2pw
```

库入口使用 `ModelOptions::new("kokoro", resource_root)`，可设置
`options.model = Some("1.1-zh-g2pw".into())`。通过 Engine 在 LocalSet 上运行；
内建音色与参数从准备后的 capabilities 查询。推理对象由专用线程持有，
取消检查覆盖资源哈希、模型阶段和有界 PCM 发送；单次底层算子执行不能抢占。

## 从官方资产准备新资源

此后端使用显式的本地开发转换，prepare 只校验，不自动下载或迁移。
完整固定版本、SHA-256 和许可证见 [SOURCE.md](../crates/kokoro/SOURCE.md)。
需要官方 Kokoro/Misaki 固定源码、原始 config/模型、官方中文 voice `.pt`，
以及独立的 Python 3.11 开发环境（依赖在 `crates/kokoro/scripts/requirements.txt`）。
导出工具校验实际导入源码与固定 commit，
转换时折叠 weight normalization，并对照官方输出。

```sh
python crates/kokoro/scripts/export_official.py \
  --kokoro-source KOKORO_CHECKOUT --misaki-source MISAKI_CHECKOUT \
  --assets OFFICIAL_ASSETS --voice-file OFFICIAL_ASSETS/zf_001.pt \
  --output NEW_MODEL_EXPORT

python crates/kokoro/scripts/export_frontend.py \
  --misaki-source MISAKI_CHECKOUT --output NEW_FRONTEND_EXPORT

python crates/kokoro/scripts/prepare_runtime.py \
  --model-export NEW_MODEL_EXPORT --frontend-export NEW_FRONTEND_EXPORT \
  --voice OFFICIAL_ASSETS/zf_001.pt --voice OFFICIAL_ASSETS/zm_010.pt \
  --output RESOURCE_ROOT/kokoro/models/1.1-zh/01e7505-native-v1
```

所有输出路径必须是新目录。CLI 的 RESOURCE_ROOT 是 `~/.talechime/resources`；
嵌入宿主可指定独立目录。转换保留许可证、模型来源报告、音色身份、大小与 SHA-256
清单。目录缺失、格式不支持或校验失败会报错，原文件不会被覆盖。
模型、词典和用户音频不随源码提交。

## 可选上下文读音

`1.1-zh-g2pw` 在相同声学模型前增加作者官方 g2pW v2 ONNX CPU 推理。
它先对规范化后的字符预测多音字，再应用已有变调、儿化和音素映射；
原版 `1.1-zh` 保留官方 Misaki 词典行为。

g2pW 是另一个模型，不能保证所有姓名或上下文正确。本阶段 24 个手工目标读音
答对 23 个，“解先生”仍误读为 jie3。该小语料结果不是通用正确率。
相对原版有 6 例音素结果改变，含重复的姓氏上下文；未以训练或规则硬编码修补测试句。

开发环境还需要作者官方 g2pW 固定源码和验证过的模型压缩包、BERT 中文 tokenizer。
只需 tokenizer 文件，不另下载 BERT 权重。

```sh
python crates/kokoro/scripts/export_context.py \
  --source G2PW_CHECKOUT --archive G2PWModel-v2-onnx.zip \
  --model-dir EXTRACTED/G2PWModel --tokenizer-dir BERT_TOKENIZER \
  --output NEW_CONTEXT_EXPORT

python crates/kokoro/scripts/prepare_runtime.py \
  --model-export NEW_MODEL_EXPORT --frontend-export NEW_FRONTEND_EXPORT \
  --context-export NEW_CONTEXT_EXPORT --voice OFFICIAL_ASSETS/zf_001.pt \
  --output RESOURCE_ROOT/kokoro/models/1.1-zh-g2pw/01e7505-native-v1
```

## 独立对照

普通测试不下载模型。以下命令只读取开发者明确提供的本地导出，数值门槛失败会返回错误。
声学对照注入完全相同的 IDs、style、speed 和随机激励，要求时长一致、各阶段/PCM
SNR 至少 40 dB；固定官方中间特征的独立解码对照要求 60 dB。

```sh
cargo run --release --locked -p talechime-kokoro --example official_compare -- NEW_MODEL_EXPORT cpu
cargo run --release --locked -p talechime-kokoro --example frontend_compare -- NEW_FRONTEND_EXPORT
cargo run --release --locked -p talechime-kokoro --features g2pw \
  --example context_compare -- NEW_CONTEXT_EXPORT NEW_FRONTEND_EXPORT
KOKORO_OFFICIAL_FIXTURE_DIR=/absolute/path/NEW_MODEL_EXPORT cargo test --release --locked \
  -p talechime-kokoro --lib -- --ignored --nocapture
```

g2pW 对照要求输入张量完全一致、概率绝对误差不超过 1e-5，同时报告人工目标读音。
数值对照和 Engine 生命周期测试不能替代完整章节试听、跨平台和实际播放验收。

`official_compare NEW_MODEL_EXPORT cpu 5` 可在数值门禁通过后增加一次显式预热与
五轮无 trace 的声学计时，并导出本地 WAV/JSON。前面的数值预检也执行过一次
同形状 forward。此计时使用固定官方激励，不包含前端、加载和随机数生成；
单进程结果只能作为性能筛查，不能代替正式的进程交错对比。

M4 Pro / CPU F32 开启 g2pW，最新八线程对照中长文原生 RTF 为 0.06522、官方组合为
0.06637；短句分别为 0.07010 / 0.06944。公共 Engine RTF 为 0.07259，约 13.8 倍实时。
完整 PCM 保持逐字节一致，见[缓冲、归一化与向量优化记录](records/kokoro/kokoro-cpu-buffer-vector-2026-10-27.md)。
该样本的模型内窗口缓冲保留约 297 MiB；内存峰值与线程条件在记录中分别列出。
历史阶段见[第一轮 CPU 优化](records/kokoro/kokoro-cpu-optimization-2026-10-27.md)、
[四线程官方对照](records/kokoro/kokoro-cpu-parity-2026-10-27.md)和
[原始性能记录](records/kokoro/kokoro-g2pw-performance-2026-10-27.md)。
