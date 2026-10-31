# 旁白候选设计与试听

先设计短参考、保存准确转写，再固定参考合成其他正文。无参考的 `narrator` 不是具名旁白资产；候选生成成功也不代表人工认可。

## Mac 构建与运行

```sh
cargo build --release --locked -p talechime --no-default-features --features cli,qwen-metal,voxcpm-metal,moss-candle-metal --bin talechime --example voice_candidates
target/release/talechime voices engines
python3 scripts/voice_candidates.py --output target/voice-candidates/20261031/review-v2
```

`voice_candidates` 示例通过公共 Engine API 输出 WAV，不打开播放设备；试听固定使用 VoxCPM2 Q8、Metal、seed 7。生成脚本显式准备所需设计资源，普通测试不会下载权重。默认库为当前用户 `~/.talechime/voices/`，不会更换当前朗读配置。

批次默认 8 画像、2 seed，VoxCPM 与 Qwen 各 16 项。MOSS 设计层现支持编入且可用的 CUDA/Metal：CUDA 模型 BF16，Metal 模型 F16，codec 均为 F16；CPU 请求在资源准备前拒绝。MOSS 的 Metal 实机设计、取消后复用与固定参考试听门禁通过后，才在批次运行命令增加 `--moss-verified`。此开关是操作方对实测证据的声明，不能代替门禁。

## 交付与恢复

产物位于 `target/voice-candidates/20261031/review-v2/`：`manifest.json`、`index.html`、`feedback.csv`，每项包含原始参考 WAV、转写、可搬运音色包和另一文本的克隆试听。音频不进 Git；索引可直接离线打开。完整性失败、克隆失败、未运行和 MOSS 阻塞单独列出；所有候选仍待用户试听。

批次 ID 限制为 1–32 个 ASCII 字母、数字、连字符或下划线，避免文件路径和反馈 CSV 注入。重新运行相同命令会核验内容、配置与构建指纹后继续；`--retry-failed` 显式重试失败阶段。已经成功设计但导出失败的音色不会重新设计。构建或配置变化必须使用新的 `--batch` 和 `--output`，不能覆盖旧证据。脚本用排他锁阻止同批并发；每阶段每次运行只尝试一次，设计 600 秒、试听 300 秒、批次累计最多 6 小时。超时后先结束子进程组再继续，保留旧用户资产。

```sh
python3 scripts/voice_candidates.py --output target/voice-candidates/20261031/review-v2 --retry-failed
python3 -m unittest discover -s scripts -p test_voice_candidates.py -v
```

参考目标约 5–8 秒，实际时长由模型决定；不剪切、补静音或调速凑长度。设计转写需结合实际录音检查，自动指标只标记疑点。批次短试听不是中文整章、多设备或所有后端的验收结论；实机状态写入 `docs/records`。

当前实机状态：MOSS Metal 在 2026-10-31 的首次完整设计尝试中报 `invalid MOSS logits`，尚未通过门禁；不能将设备清单中的 Metal 解释为已有可用设计音频。Qwen/VoxCPM 批次独立推进。

本次 review-v2 保存了与其清单指纹相同的 runner.py；若需要复现旧执行，使用 `python3 target/voice-candidates/20261031/review-v2/runner.py --output target/voice-candidates/20261031/review-v2 --retry-failed`。当前工具新增了批次 ID 校验，不能直接用新工具冒充旧执行版本。
