# OmniVoice Rust / 官方 Python 对比复核（2026-10-23）

**当前短文本基准不支持“Rust 仍慢于官方”的前提。** 同窗口三路测量：
已提交 `a43e46a` 为 3.024 s，完成收敛的免拷贝候选为 2.672 s，官方为 3.235 s。
保留免拷贝改动；修正此前用微基准外推整链、沿用旧官方数字及无效编译监控的问题。

## 同期测量与限制

M4 Pro / 24 GiB / macOS 26.6.2；固定本地 0.6b revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4`。Rust Candle revision
`2266099d92265112ecc0d56ecee1761e3cc2ac00`；官方源码 `08be0b4`，
torch 2.8.0 / transformers 5.3.0。官方模型源码无本地修改，只有 uv.lock 有本地差异。
这里的“官方”是本地计时脚本调用固定官方实现，不是引用官方发布的性能数字；
明确传入 MPS 和 float16，与 Rust 主干精度对应，不代表官方所有默认配置。

相同短文本 SHA256 `f29b49d81cce2947186d9b6f6d5fc719a2aaee26e7993c14e394db0afb313657`，
两边 text-only、中文、32 步、guidance=2、class_temperature=0、
position_temperature=5、t_shift=0.1、layer_penalty_factor=5、seed=42。
Rust narrator 不附加参考音频或 prompt；仅 custom voice / context 分支附加参考。
Rayon / VecLib / OMP 设为 8，实际 torch CPU threads=8。

顺序 ABCCBA，A 为已提交 Rust，B 为免拷贝 Rust，C 为官方；各路径两个独立进程，
每进程 1 预热 + 3 热轮。release 无诊断，加载和 WAV 导出不计时，GPU 完成纳入计时。

| 路径 | 两进程热测均值（s） | mean-of-means（s） | 平均 RTF |
| --- | --- | --- | --- |
| A：a43e46a | 3.05472 / 2.99293 | 3.02383 | 0.55997 |
| B：免拷贝候选 | 2.66903 / 2.67503 | 2.67203 | 0.49482 |
| C：官方 Python | 3.22400 / 3.24500 | 3.23450 | 0.60010 |

B 相对 A 耗时减少 11.63%，相对 C 的这组生成路径减少 17.39%。
这不是全模型、全长度的排名；样本数每侧两个进程，没有给出置信区间，也未锁频或独占系统。
Rust 输出 5.4 s；官方各进程热轮为 5.4 / 5.37 / 5.4 s，因此同时报告逐轮 RTF。
跨框架 RNG 和浮点策略不同；官方脚本在进程开始设 seed，Rust 按请求设置 seed，
不是逐张量相同输入的内核比较，也不是音质等价验收。

第一组三路测量捕获到了一个官方进程的外部编译活动，未作为最终比较依据。
随后重新完成上述完整三路；每 0.5 s 按进程 basename 监控，六个进程均无
cargo/rustc/clang/ld/clippy-driver 活动观测。
原始数据 `target/perf/omni-audit/quiet-comparison.json`、`quiet-summary.json`；
受干扰原数据单独保存在 `comparison.json`，不删除也不改名为无干扰结果。

## feature 和实际路径核查

| 检查项 | Rust | 官方 |
| --- | --- | --- |
| 主干设备 / dtype | Candle Metal / F16；不是 CPU 回退 | 实例属性为 mps:0 / torch.float16 |
| 注意力 | 已用 Metal SDPA、原生 GQA；候选用 rope_thd 减少 q/k 布局复制 | 实例 `_attn_implementation=sdpa` |
| 音频 tokenizer / decoder | 本轮 decoder Metal F32 | 实例为 CPU / torch.float32；官方 MPS 分支明确如此路由 |
| Accelerate | 本轮 release 链接 Metal，未链接 Accelerate.framework | build config 为 BLAS_INFO=accelerate、LAPACK_INFO=accelerate |
| CUDA / FlashInfer | 不是本轮 Mac 执行路径 | 本轮 MPS 不使用 CUDA FlashInfer |

运行时属性由额外加载官方模型读取，未在正式计时中插入探针。
证据为 `official-runtime.json`、`torch-build.txt` 和 Rust 二进制 `otool -L`。
Accelerate 影响 CPU BLAS，不会把 Candle Metal 主干换成 PyTorch MPS 内核；
链接库也不能证明所有 CPU 算子使用 AMX，更不能证明 ANE。
历史 CPU 实验确认 Accelerate 有帮助，但没有解决全部 CPU 解码差距；本轮未重测 CPU A/B。

普通 Cargo 默认仅 `cli + moss-nano-candle`，`omnivoice` 与 `omnivoice-metal` 是不同 feature。
本轮基准显式启用 `omnivoice-metal`；发布源码配置也列有该 feature。
因此本轮没有漏开 Metal。源配置不能代替对用户另一份已安装二进制的实际验证。
若构建最小 OmniVoice Metal CLI，配置入口为：

```sh
cargo build --release --locked -p talechime --no-default-features --features cli,omnivoice-metal
```

这只是 CLI feature 入口；本轮实际构建和计时的是 performance_probe，没有发布或安装 CLI。
ORT/CoreML feature 不会自动替换当前 OmniVoice Candle 实现。

## 之前为什么慢：已证明的实现效率问题

1. 早期手写 attention 没有调用融合 SDPA；Metal feature 开启也不会自动把手写算子链融合。
2. 重复 K/V、q/k 头主序布局物化等造成额外 GPU 分配与搬运。本轮保留的改动利用现成
   rope_thd，在投影原布局上归一化和旋转，让 SDPA 读取转置视图；不是新增编译开关。
3. 非零 padding 的转置卷积曾错过 Candle 快路径，1×1 卷积也经过 im2col 和输出复制。
   前几轮已分别验证并优化；它们不是当前仍然存在的同一份差距。
4. PyTorch 的 Python 层主要调度 ATen/MPS/CPU 原生算子，Rust 则调度 Candle 原生算子。
   性能由具体内核、布局、dtype、融合、批处理和同步路径共同决定，不能按实现语言排名。

这些证据支持“原实现存在效率问题”，不是当前发现了导致功能错误的新 bug。
也没有证据证明打开一个遗漏的 feature 就能解决所有差距。当前跨框架逐层数值和
音质等价尚未由本轮性能测量证明；不能从 Rust 新旧输出一致推导与官方完全相同。

## 未提交实验如何收敛

保留免拷贝生产分支及三个离线探针；新建补丁快照保存在 `incoming.patch`。
重新构建当前源码，候选二进制 SHA256 为
`8cfb1b4c82d0bbc56013da5cac0f10ed34add04bf21b104fd2c629b1d303f0ef`。
该重建产物另跑短/长文本，与冻结基线的完整 codes 和全部 28 个 F32/PCM16 WAV
逐字节一致；不是只相信先前记录。34 项 Metal、23 项 CPU 测试，
Metal + Accelerate Clippy all-targets、rustdoc、格式检查均通过。

已纠正以下证据越界：同步墙钟不叫 GPU timestamp；不能由单个 GEMM 判断整个
forward 仅受 GPU 限制；不能把独立算子耗时乘层数当成生产阶段归因；输入前缀不变
不能推出双向注意力的深层 K/V 不变；合成投影逐位等价不能替代真实权重整链验证。
以后按固定版本、同期整链对照、再做可证伪的局部实验推进，不沿用过时官方数字。

没有修改依赖、采样配置、公共 API、会话或取消行为，没有推送、合并、部署或发布。
