# CosyVoice3 后端接入验收

日期：2026-10-17。设备：本机 macOS，Metal；代码基于 `930c6bf` 后续接入。
当前用法与限制见 [CosyVoice3 后端](../../cosyvoice3.md)。

本次增加 `cosyvoice3` / `0.5b` 能力目录、CPU/Metal/CUDA feature 路由、本地转换清单校验、
共享音色导入/删除、Registry 准备、Engine 与 CLI/worker 接入。标准包启用该适配器，
默认后端不变。未开放风格、continuation、模型原生流式 speed 或持久音频缓存。

专用 owner 线程只加载一份模型；新增借用模型的 StreamingTts 路径，与原有拥有模型的探针
共用解码逻辑。流销毁清 KV，逐请求持有 token/mel 缓冲。请求与输出通道均为容量 1，
取得输出许可后才生成下一块。正常 final 后才发 End；长度上限、错误与断连不作为完成。

## 本机真实模型

使用已有官方参考音频，`voices import --text` 导入临时资源根，`voices list` 返回该音色。
所有模型、音频、配置、资源链接和日志位于 Git 忽略的 `runs/`。

```sh
cargo build --release --locked -p talechime --features cosyvoice3-metal \
  --example cosyvoice3_backend --bin talechime
target/release/examples/cosyvoice3_backend runs/cv3-backend-resources custom:review metal
```

通过 Talechime Engine（非单独计算库）执行：

- 两次固定 seed 42 请求均正常完成，各 3 个对外 PCM 块、82080 个样本（24 kHz 单声道），PCM 完全一致。
- 长请求首块后取消成功，取消后再次合成成功，Engine 正常关闭。
- 独立 CLI `--version`、`--help`、JSON Lines v7 握手与 shutdown 通过，无模型下载或配置写入。

日志：`runs/cv3-backend-real.log`、`runs/cv3-backend-build-release.log`。
这些结果只证明接入、输出格式、重用和取消路径；未进行播放器试听或文字回读验收。

## 无模型检查

新增 5 项适配器测试覆盖无资源能力查询、缺失本地转换拒绝、关闭消费者避免推理、
队列满时暂停推理，以及错误/上限/缺失 final 不发布 End。已有协议目录测试增加 CosyVoice3，
保留旧模型目录顺序，并明确该模型不声明 continuation。

验证包括全工作区 CPU 适配器组合测试、Clippy 与 rustdoc；Metal 相关库/例子测试；
独立 `cosyvoice3` 无 CLI 构建、库测试与无模型构建 Clippy；格式、发布脚本测试及 dist
生成一致性/计划。日志前缀为 `runs/cv3-backend-`，dist 日志为 `runs/cv3-dist-`。

全工作区 rustdoc 发现计算库与 facade 的 synthesize 例子重名，计算库例子改名为
`cosyvoice3_synthesize`，避免生成文档路径冲突。第一次全工作区测试发现目录断言未包含新后端，修复后通过；另一次在并行真模型运行时，
原有迁移握手测试超过 5 秒，保留时间门限，独立重跑及最终组合复验通过。

CUDA 仅接入编译 CI，未在本机编译或实跑。整包跨平台安装未在本次重跑。
严格非流式整链波形等价门禁仍是此前已知失败（max error 约 0.162），未放宽阈值；
本次不宣称官方完全等价或正式模型听感验收完成。
