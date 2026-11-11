# eSpeak 辅助程序的安装器打包

发行归档包含独立的 `talechime-helpers/espeak-ng-1.52.0/`，其中有程序、数据、manifest 和 `source/` 对应源码及 GPL 许可。运行时从 Talechime 可执行文件所在目录查找该版本目录；仅在 cargo-dist 的 `include` 中放入文件并不能让原生安装器复制它。

固定 cargo-dist 0.32.0 的全局构建完成后、收集上传列表前，运行 `python3 .github/scripts/espeak-installers.py dist-manifest.json`。脚本只修改生成的 shell、PowerShell 和 Homebrew 安装器，将完整目录安装到可执行文件旁；Homebrew 公式注明 MIT 与独立辅助程序 GPL-3.0-or-later。它更新全局 SHA-256 清单及 manifest 中的安装器摘要（存在单独 sidecar 时一并更新），保留原安装器的下载、归档校验和其他安装逻辑。版本、模板锚点或上传清单不符合预期时直接失败，不能上传未打补丁的安装器。对同一批产物只能执行一次；重试应先重新生成全局产物。

`generate-release.py` 将这一调用固定在全局 job 内，发布 host 的全部严格前置门禁保持不变。`distribution-smoke.py` 自行生成全局安装器时也对该次生成的 manifest 执行同一脚本，并使用 builder 的 `verify_bundle` 核对归档和安装后的 helper 文件及源码摘要，再各自执行 helper 的离线语音前端探针。

2026-11-11 验证：cargo-dist 0.32.0 生成的 macOS shell 安装器使用隔离 HTTP 归档完成首次安装及覆盖安装，真实 helper 的程序、数据和源码文件摘要一致；缺失 `source/COPYING` 时失败且旧程序、helper 保留。Homebrew 公式通过 Ruby 语法与隔离 DSL 安装路径检查，尚未进行真实 `brew install`；PowerShell 仅做生成内容与补丁测试，Windows 尚未实机安装验证。

同日用 cargo-dist 0.32.0 实际构建并安装 macOS aarch64 归档，通过了归档校验、程序版本/模型目录/协议检查及安装前后的 helper 摘要和 CLI 检查；隔离构建为绕开当时 ASR API 签名不匹配而关闭了 `asr` feature，因此不是完整发行配置的验收。eSpeak 1.52 的数据路径有长度上限，smoke 在 Unix 使用较短的 `/tmp` 隔离目录并保留空格路径覆盖。


随后修复 ASR 门面调用的下载参数，完整标准 feature（含 `asr` 和 macOS Metal）
归档重新构建并通过同一隔离安装探针：版本、模型目录、v7 协议、主归档 SHA、
安装前后 helper 完整清单/对应源码与真实 CLI 均通过。
此项是本机 cargo-dist 构建和 shell 安装，未执行发布；Windows 与真实 Homebrew
安装仍按上述未验证边界记录。

2026-11-12：最终完整 macOS 标准配置再次通过真实 cargo-dist 归档与隔离 shell 安装，
新包含 `kokoro-espeak-phonemize` 独立程序及对应 C 源码/构建脚本。安装后真实
`espeak_TextToPhonemes` 探针将 `de` 转为官方相同的 `də`；通用 CLI 仍单独 smoke，
不再把 CLI 句末重音当作官方音素结果。Windows 和实际 Homebrew 安装仍未实机验收。
