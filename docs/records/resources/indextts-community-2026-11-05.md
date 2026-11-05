# IndexTTS 固定资源核验（2026-11-05）

社区包 `mlx-community/IndexTTS-2.5-fp16@65644cd70da15309ffeb74aa03f3686bb04e3eb1` 的 GPT、codec、s2mel、BigVGAN 已完整下载并按 LFS 摘要和大小核验。固定官方原始输入来自 `IndexTeam/IndexTTS-2.5@c39ce5ba981572cb187443877ff559dfb246ce63`，辅助件来源见 `crates/indextts/SOURCE.md` 与 `crates/talechime-backends/src/indextts/native-assets.json`；全部原始资产已实际下载验证，不进入 Git。

## 社区布局与精度

采用固定官方 F32 导出为逐张量参考，按发布转换源码的 GPT/S2Mel/BigVGAN 名称与布局规则及 Codec Conv1d/weight-norm 规则进行逐项映射。审阅的源码 revision 为 `vanch007/mlx-indextts2@4a32c967accd4d212c7cfedeb42e06667645e3e0`（MIT），仅在开发环境抽取已审阅的纯转换函数，未引入 MLX 运行依赖。完整逐张量记录和资产留于 Git 外审计缓存。

| 组件 | 官方张量 | 映射/社区张量 | 名称缺失/新增 | 形状差异 | 实际 dtype | 相对映射 F32 有差异的张量 | 最大绝对差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT | 456 | 457 | 0/0 | 0 | 全部 FP16 | 457 | 0.00381374 |
| codec | 243 | 241 | 0/0 | 0 | 全部 FP16 | 241 | 0.00386143 |
| s2mel | 284 | 264 | 0/0 | 0 | 全部 FP16 | 261 | 0.00192642 |
| BigVGAN | 783 | 449 | 0/0 | 0 | 全部 FP16 | 449 | 0.000968456 |

GPT 的 457 项全部与映射结果转 FP16 相等；S2Mel 264 项、BigVGAN 449 项同样全部相等。Codec 240/241 与上述折叠规则转 FP16 相等，`quantizer.quantizers.0.in_project.weight` 存在额外舍入差异，未掩盖或替换原记录。

- GPT：GPT-2 attention/MLP 二维权重转置；卷积从 OIK/OIHW 改为 OKI/OHWI；`embed.conv.0`/`embed.out.0` 更名；Perceiver `to_kv` 拆为 K/V，FF 和 norm 名称改变；省略 1 项位置编码缓冲。原始 F32 方案保留 456 项，不以遗漏的缓冲从其他版本补齐。
- Codec：2 组 `weight_g/weight_v` 折叠并改为 MLX Conv1d 布局，243 项变为 241 项。原生方案保持全部原始 F32 参数，使用既有 Candle 加载/运算路径。
- S2Mel：扁平化 `models.<module>.*` 后，MLX 方案再去掉 `models.`，更名 gpt_layer/timestep/adaLN/嵌套卷积，折叠归一化并转置卷积；报告忽略 `input_pos`，共 284→264 项。原生方案保留全部 284 项。
- BigVGAN：报告省略 218 项 alias-free 上/下采样滤波器，折叠 116 组 g/v；Conv1d/ConvTranspose1d 使用不同排列，783→449 项。当前 Rust 语义需要原始滤波器并在现有加载器执行归一化，不能直接消费这个包。

发布清单声明来源 revision `d0aa86e75bb6f3437f3831e95056fa72842d89ef`，与当前批准官方 revision 不同；README 所称 `convert_v25.py` 不在它声明的转换源码 revision 中，实际存在的纯转换基础函数与上述逐张量映射相符。未仅凭报告的 `status=pass` 批准它。四个组件均拒绝作为本轮 F32 输入，FP16 转回 F32 不能恢复原精度。

## 完整官方来源与必要辅助件

- 主模型、feat1/feat2、w2v stats、配置与 tiktoken：固定 IndexTeam revision `c39ce5ba…`；bilibili Model Use License Agreement。准备目录保留官方 LICENSE 和派生修改声明，不把模型权重上传到本资源仓库。
- CAMPPlus：`funasr/campplus@e4b6ede7ce16997aff4ae69fbca1f0175e2afede`，937 项官方张量，Apache-2.0。社区包未包含此必需组件。
- BigVGAN 原始生成器及其专用配置：`nvidia/bigvgan_v2_22khz_80band_256x@633ff708ed5b74903e86ff1298cf4a98e921c513`，MIT；不能用社区总体 config.json 代替专用配置。
- w2v：`facebook/w2v-bert-2.0@da985ba0987f70aaeb84a80f2851cfac8c697a7b`，MIT，直接复用其原始 F32 safetensors 与配置。完整文件实际 SHA-256 `eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043`，不会从 FP16 恢复。
- feat1=说话人矩阵 F32 `[73,192]`，feat2=情绪矩阵 `[73,1280]`；stats 的 mean 为 `[1024]`，std 严格为对原始 F32 var 求平方根。都从固定原始 `.pt` 原生导入。

## 原生导入对照

独立开发环境 PyTorch 2.7.1、safetensors 0.8.0 生成 F32 oracle，Rust 受限导入并写出六个组件。合计 2,707 个张量，dtype、形状与所有元素逐位一致；只改变确定性 safetensors 头部序列化顺序，不替换旧清单摘要。原生实际转换约 55.66 秒。

GPT/codec/S2Mel/CAMPPlus/BigVGAN 的开发导出文件与既有旧摘要完全相同。配置 JSON 的旧 3,413 字节是 Windows CRLF，当前 LF 版本为 3,269 字节；按 CRLF 重建可精确得到旧摘要 `41c6d33f…`，解析数据未改变。旧 feats 文件摘要与当前 oracle 不同，未获得旧文件，不能确定其字节差异来源；本轮按固定原始 feat/stats 独立验证，Rust 与当前 PyTorch oracle 全部逐位相等，使用独立版本和新摘要，不宣称恢复了旧文件。

以上是来源/布局/张量对照。完整资源发布、加载、短文本合成和桌面验收分别执行，不以本记录替代。
