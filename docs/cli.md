# 命令行使用

本指南对应当前源码。先用 `talechime --version` / `--help` 核对当前安装支持的入口。

## 查询与选择

```sh
talechime models
talechime models --backend qwen --json
talechime models --backend moss --model nano-candle
talechime voices list
talechime voices engines
```

`models` 查询合成模型、当前可用设备、内建音色及参数名；JSON 含完整参数约束、默认值、
编入设备和可用设备。`voices list` 只列用户音色库，`voices engines` 只列设计引擎及代价。
这些命令不下载权重、不初始化播放器，也不要求有效的合成配置。

## 文件朗读

```sh
talechime chapter.txt
talechime chapter.txt --backend moss --model nano-candle --tts-device cpu
talechime chapter.txt --voice reader
talechime chapter.txt --restart
talechime chapter.txt --after-chapter
talechime chapter.txt --plan chapter-plan.json
```

文件须为 UTF-8。默认按实际播放检查点续读，`--restart` 从头；空格暂停/继续，s 停止，q 退出。
非交互终端不接管键盘。`--after-chapter` 先完整生成到暂存，再播放，失败不播放部分章节。
CLI 提供文件朗读；若需要导出生成的 WAV，使用 Rust PCM API，不要使用音色资产 export。

选择 backend/model/voice/style/tts-device 会保存为后续朗读默认值。一次性试验用独立
`--config`。`--voice` 可以直接给库 id，无需手动安装模型缓存；首次使用 Nano 参考可能
先准备编码器，合成模型资源随后按需准备。显式设备不可用会报错，不静默换设备。

计划文件自带 backend/model 和固定声音集合，CLI 自动选择计划模型及首个音色作为准备
默认值；显式 `--backend`/`--model` 必须与计划一致。计划必须 sealed=true，正文与传入文件
完全一致；`--plan` 不与 --voice/--style/--after-chapter 混用。

`--param name=value` 可重复，按模型目录校验；`--seed` 固定采样输入。
各段使用所选音色的固定参考；无滚动参考或双参考开关。
`--verify report|gate` 需要 ASR feature；默认关闭。

## 音色资产

```sh
talechime voices design --description "低沉温和的中年男声"
talechime voices design --description "温和清晰" --name "朗读者" --id reader --engine qwen-voicedesign
talechime voices import reference.wav --name "朗读者" --text "参考录音的准确文字" --id reader
talechime voices rename reader --name "新的名字"
talechime voices export reader --output reader.voice
talechime voices import-bundle reader.voice --id reader-copy
talechime voices remove reader-copy
```

设计名称可省略；自动 id 以返回结果为准。设计引擎默认优先就绪模型资源，其次专用引擎；
缺失资源可能数 GB，先查 engines。已就绪表示资源可复用，不代表复用已加载的推理对象。
导入需准确转写；可用参考时长等限制由目标合成模型决定，超限明确报错。

`.voice` 是原始参考 WAV＋元数据的搬运包，不是生成的朗读音频。冲突 id 拒绝覆盖，
需要保留两条时指定新 id。remove 删除库资产；模型缓存不是资产的替代备份。
旧模型音色目录不会自动扫描、迁移或清理。

## 路径与机器接入

默认配置 ~/.talechime/config.json；资源 ~/.talechime/resources/；
音色 ~/.talechime/voices/；检查点 ~/.talechime/checkpoints/。
--config、--model-dir、--checkpoint-dir 是全局路径参数，可放在子命令前后：

```sh
talechime voices list --config ./speech-data/config.json --model-dir ./speech-data/resources
talechime models --json --model-dir ./speech-data/resources
```

音色库跟随 config 文件，位于其父目录的 voices/；只改 model-dir 不会搬动用户音色。
音色资产命令不读合成配置，不会用默认设置覆盖损坏配置。

非 Rust 宿主用 `--protocol` 启动 JSON Lines v7 播放控制，stdout 只含协议消息，stderr 为日志。
宿主持续消费两条输出流，五秒内 hello，等待 model_ready 后提交计划，按真正 completed
终态判断播放结束。详细契约见 [协议](../crates/talechime-protocol/README.md)。
