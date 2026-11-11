# 原生文本资源：权重比较与跨卷准备

2026-11-11，macOS Apple Silicon；rustfst 1.3.1、WeText 0.1.0 /
WeTextProcessing 1.3.0 / NeMo 1.2.0。此记录适用于固定 FST 前端和原生准备流程，
不代表所有平台、模型声音或产品读法通过。

官方语法含 1e-5..1e-4 的候选路径偏好。rustfst 的 TropicalWeight 默认相等容差
为 1/1024，直接使用其 shortest path 会把这些偏好当作相等，导致日期或金额候选
选错。问题发生在路径选择，不应通过修改模型正文或放宽 oracle 来绕过。
共享 [FST 执行器](../../../crates/talechime-text/src/fst.rs) 改用严格 F32 代价比较，
并在路径搜索中检测取消与负环。两套平台中英各 20 条及西语 16 条独立官方 fixtures
通过；CosyVoice3 另有 25 条完整前端 fixtures。具体来源与许可仍由各模型负责。

另一个故障是共享下载缓存与准备目录跨卷时，下载器可能以符号链接物化输入。
文件内容与摘要正确，但准备器要求转换输入是普通文件，因而报完整性失败。
[输入物化](../../../crates/talechime-backends/src/native_preparation.rs) 现仅解析准备器
拥有的输入视图，将内容硬链接或复制为普通文件；不修改共享缓存、不跟随任意输出链接。
符号链接输入回归及 IndexTTS front-v6 完整准备、无下载离线复验通过。

原始运行产物保存在 Git 忽略目录。模型权重、签名下载地址与用户正文不进入知识库。
可复现测试入口见模型 SOURCE、`tools/tts` 和显式 ignored 真实资源测试；
普通测试继续使用本地夹具，不自动下载。


MOSS VoiceGenerator 的独立 prompt 对照进一步发现，正则写法相同不等于 Unicode
行为相同：固定 Python 3.9 的 Unicode 13 将 U+1FAE8 视为未分配并清除，新版 Rust
正则会保留。原生前端改用固定的 Unicode 13 C 类别范围；生成器与官方 UCD 对
全部码位交叉核对，61 条规范化用例和四条实际 tokenizer prompt 对照分别验收。
Python `splitlines()` 接受 U+2028/U+2029，Rust `lines()` 不接受，也需单列回归。
数据来源、摘要及 Unicode 许可见 [MOSS SOURCE](../../../crates/moss-tts/SOURCE.md)。
这类移植必须固定隐含的分类数据版本，不能只固定脚本文件或放宽比较。
