# talechime

文件朗读 CLI 与 JSON Lines worker。模型装配、准备、设备校准和音色命令归程序层；会话与播放归 `talechime-core`，模型实现归 `talechime-backends`。

## 文件朗读

```sh
cargo run -p talechime -- --help
cargo run --release -p talechime -- --backend moss --tts-device cpu book.txt
cargo run --release -p talechime -- --restart book.txt
cargo run -p talechime -- voices list
```

`Space` 暂停/继续，`s` 停止，`q` 退出。片段进度来自实际播放时钟；只有播放完的片段提交完成检查点。合成失败停止会话，不跳过正文。

默认只编入 MOSS Nano。其他模型与加速按需选择：

```sh
cargo build --release -p talechime --features qwen,voxcpm,omnivoice
cargo build --release -p talechime --no-default-features --features qwen-cuda
cargo build --release -p talechime --features qwen-metal,voxcpm-metal,omnivoice-metal,moss-candle-metal
cargo build --release -p talechime --features ort-coreml
```

Candle 加速开关包含对应模型开关；`ort-cuda` / `ort-coreml` 仅为 MOSS Nano 启用 ORT provider。`auto` 通过完整合成校准选择设备；显式不可用设备报错。切换模型时同时校验目标模型与设备。

## Worker

```sh
cargo run -p talechime -- --protocol
```

stdin/stdout 使用协议 v6，stderr 为日志。先发送 `hello`：

```json
{"protocol_version":6,"request_id":"hello-1","session_id":null,"type":"hello"}
```

协议模式不接管终端。握手期限 5 秒，单行上限 16 MiB；未知版本关闭连接。EOF / shutdown 停止播放并保存可靠进度。控制命令保持串行，等待期间继续消费会话事件，避免有界队列死锁。

当前只提供片段级进度；对齐模型、对齐配置及句子事件已移除。客户端与 worker 必须使用同一协议版本。

## 配置与资源

配置位于 `~/.talechime/config.json`，检查点位于 `~/.talechime/checkpoints/`，模型与音色位于 `~/.talechime/resources/`。保留 `--config`、`--checkpoint-dir`、`--model-dir` 显式参数。不读取旧主目录，不自动迁移退休后端。

模型在实际准备或朗读时按需下载。帮助、握手和音色目录查询不下载模型；参考音色按模型隔离。配置使用修订检查与原子替换，损坏数据不覆盖。

完整用法见[仓库 README](../../README.md)、[架构](../../docs/architecture.md)、[开发与验证](../../docs/development.md)、[协议](../talechime-protocol/README.md)。真实模型验收与普通测试分开记录。
