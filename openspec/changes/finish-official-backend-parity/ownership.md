# 任务归属与既有证据

2026-11-11 应用基线。旧任务保持未完成复选框，以下映射是后续唯一实施归属；旧记录不代表新门禁通过，不重复实现或删除已有修改。

| 原任务 | 后续负责项 | 保留证据与边界 |
| --- | --- | --- |
| complete-backend-text-parity 1.1（已完成） | 无新增实施 | Kokoro 前端及宿主原文边界，旧任务保留 |
| complete-backend-text-parity 1.2 | 本变更 2、7.1、7.4 | tools/tts；已有局部 WAV 不代表完整矩阵 |
| complete-backend-text-parity 2.1 | 本变更 5.1、5.2 | Qwen Auto/方言代码与 Base 探针 |
| complete-backend-text-parity 2.2 | 本变更 5.3 | OmniVoice 参数代码，真实验证未完成 |
| complete-backend-text-parity 2.3 | 本变更 3、4 | IndexTTS 词表、时长及阿语代码，TN/G2P/情绪未完成 |
| complete-backend-text-parity 2.4 | 本变更 5.4、5.5 | CosyVoice3 跨语言参数测试 |
| complete-backend-text-parity 2.5 | 本变更 6.1、6.2、6.4 | 既有 MOSS/VoxCPM 记录和未通过项 |
| complete-backend-text-parity 2.6 | 本变更 6.3、6.4 | Kokoro 混读及未知词边界 |
| complete-backend-text-parity 3.1 | 本变更 7.3、7.5 | 普通门禁与真实验收分开 |
| kokoro-mixed-language 1.3 引擎矩阵部分 | 本变更 2、6.4、7.1 | 宿主原文传递部分由宿主变更负责 |
| kokoro-mixed-language 2.1 官方前端部分 | 本变更 6.3 | 已有普通测试不重算为官方对照 |
| kokoro-mixed-language 2.2 人工听验部分 | 本变更 7.4 | 待用户听验；真实样章/宿主既有事实保留 |
| CastGlean backend-owned-speech-text 2.1 引擎矩阵部分 | 本变更 2、7.1、7.4 | 原生窗口验收仍由宿主负责 |
| CastGlean backend-owned-speech-text 2.2 | 宿主变更 | 本变更只复核原文及恢复契约（7.2），不更改宿主 |

旧任务入口：[文本能力](../complete-backend-text-parity/tasks.md)、[Kokoro 混读](../kokoro-mixed-language/tasks.md)。宿主入口为 CastGlean 的 `openspec/changes/backend-owned-speech-text/tasks.md`，不在本仓库内伪造完成状态。完成时统一核对归档和规格同步，当前不归档。
