# 手工搬迁旧听书数据

关闭旧 worker 并备份。新版不读取、不删除旧目录，不自动迁移退休后端。

1. 将 `~/.novel/tts_config.json` 复制到 `~/.talechime/config.json`。
2. 将 `~/.novel/tts/checkpoints/` 内容复制到 `~/.talechime/checkpoints/`。
3. 将 `~/.novel-tts/` 的全部内容整体复制到 `~/.talechime/resources/`，保留模型、音色、编码缓存和校准文件的相对结构，不增加旧目录名这一层。
4. 若已有新配置或资源，先人工核对/合并，不覆盖新数据。旧 Kokoro / ZipVoice 配置需手工选择支持的模型、音色和设备，或移开配置后重新设置。

显式 `--config`、`--model-dir`、`--checkpoint-dir` 仍然有效。检查点格式、正文摘要和字节坐标保持原样。仅路径变化不会要求重新下载模型；完整资源仍需校验。旧文件可在确认后自行删除。

TRNovel 的阅读数据、外观、阅读偏好和 worker 路径由 TRNovel 自己管理，见其[迁移指南](https://yexiyue.github.io/TRNovel/guides/migration/)。JSON Lines 继续使用 v5，应用版本无需一致。
