# Qwen 与 VoxCPM 优化主线合并

## 范围

将主线 `704e819`（Qwen 的 Metal/采样/流式修复）与 VoxCPM 分支 `a675f92`
（CFM + AudioVAE CUDA Graph、安全修复与性能记录）合并到本地 main。
保留两边完整实现与验收记录；没有把 Metal 性能外推到 CUDA，也没有重新宣称官方/C++
在相同协议下的排名。本次不发布 crates、不推送远端、不改模型或用户音色。

## 审查与修正

- P2：原三包 Candle patch 遗漏 `candle-metal-kernels`。fork core/nn 的路径依赖
  来自 git，`tts-candle-platform` 的直接依赖来自 registry，lockfile 出现同版本双来源。
  补充同一 revision 的第四个 patch，删除 registry kernels 条目。CUDA kernels 和
  candle-ug 的 fork 内部路径来源保留，不增加重复声明。
- 原 Vox 直接 cudarc 依赖还启用了默认 runtime/fallback-dynamic-loading，而 Candle
  自身使用关闭默认 feature 的 dynamic-linking 配置。追加关闭直接依赖的默认 feature，
  仅声明 std/driver，CUDA 加载方式及其他库能力统一由 Candle feature 激活。
- P2：Qwen depthwise 的 Metal 测试仅检查 feature，Windows all-features 会执行
  不存在的 Metal 设备。补充 `target_os = "macos"`，与既有采样测试和平台 crate 一致。
- 文档索引冲突合并保留 Qwen 和 VoxCPM 两组完整链接；其余模型源码自动合并，无算术冲突。
- 组合测试发现 Windows `autocrlf=true` 把 narration 夹具转为 CRLF，固定字节偏移
  baseline 因而在 26/27 处失败。两个 narration 夹具在 `.gitattributes` 显式固定 LF，
  不改正文或断言，不把真实输入规范化；已有 CRLF 原文偏移测试保留。
- Qwen 的旧 CDF 等价测试在 Windows 3072 词表、seed 53 下失败：顺序 F32 结果
  `[232, 1978]`，旧 matmul cumsum 为 `[232, 1977]`。源码确认 Candle cumsum
  使用三角矩阵乘法；任意 softmax 概率的不同累加次序不能保证逐位等价。
  保持 main 的顺序 F32 策略，旧 CDF 对照改用累加可精确表示的 dyadic 概率，
  仍覆盖两个 batch、4/3072 词表、512 个 seed 和 RNG 推进；既有阈值/空分布测试保留。
  修正注释与证据边界，不把 Mac 上有限夹具通过外推为所有平台保持旧采样结果。

## 架构决定

不新增 crate/trait，不移动模型私有状态。`tts-candle-platform` 继续做平台 feature
和 Metal 可用性探测；Qwen decoder 的有限 codes 历史与 Vox AudioVAE 的图状态各自
管理。Vox replay 的设备复制与 capture 生命周期、CFM 的 schedule/LRU、decoder 的
history/overlap 分开，Qwen 保留统一 CPU 采样边界与设备帧缓冲。

可共享的方法写入 [TTS 性能优化方法](../../knowledge/tts-performance.md)，从 KISS、DRY、
YAGNI、SOLID 检查模块边界，并统一测量、输出门禁、失败回退、来源与发布规则。
没有为形式统一提取两种不同状态的通用 decoder，也不为清除无关版本重复升级第三方包。

## 验证

依赖解析：Windows CUDA 与 macOS Metal 的 all-features metadata 中，各 Candle 包均
只有一个 0.11.0，来源统一为 `bafb03db81c3eae48e63ff28d6b30f5a6576ba38`。
Candle/Vox 共用 cudarc 0.19.10；其他依赖的 0.17.8 不属于这条设备数据通路，保留。
这只验证跨平台解析，不代表本机执行过 macOS Metal 编译或性能测试。

- `cargo test --locked --workspace --lib --tests --examples -j 1 -- --test-threads=1`：
  修正上述夹具后 56 个测试二进制，667 passed / 24 ignored / 0 failed，未跳过遗留
  `default_home_migration_keeps_hello_lightweight_and_legacy_voices_visible` 时限用例。
- workspace all-targets Clippy、workspace private-items/examples rustdoc：`-D warnings` 通过。
- Qwen/VoxCPM/backends 同时启用 qwen-cuda、voxcpm-cuda、Qwen profiling 的
  all-targets Clippy 与 private-items/examples rustdoc：`-D warnings` 通过。
- Qwen CUDA + profiling timing/conv 探针 release 构建通过；既有 CUDA SDK 静态 loader
  的 LNK4098 保留，不用 `/NODEFAULTLIB` 压制。该构建本身不是性能验收。
- `cargo build --locked --release -p talechime --no-default-features --features cli,qwen-cuda,voxcpm-cuda -j 1`：
  主程序同时链接两种 CUDA 后端通过，生成的 `talechime --help` 冒烟通过。
- 格式与 staged diff 检查通过；本地日志存于忽略目录 `target/tmp/voxcpm-opt/merge-*.log`。
- VoxCPM `cargo publish --dry-run --locked --allow-dirty -p voxcpm -j 1`：通过，
  使用 registry Candle 验证打包；未上传。
- `prepare-notices.py`：通过，产物包含 Candle Apache/MIT 许可副本。
- 拒绝候选撤回后，Qwen 标准 CUDA release 探针重新构建、运行，10 份最终制品与基线
  对应类型逐字节相同；没有把忽略目录中的候选二进制作为最终标准探针。

CUDA 普通库组合测试：Qwen 219 passed / 3 ignored，Vox 10 passed / 10 ignored。
CUDA sampler 对照另行显式执行 1 passed；Vox 真模型图对照另行串行执行 4 passed。
关闭 cudarc 默认 feature 后的组合 CUDA Clippy 与 rustdoc 再次通过。
不把普通测试中的其他 ignored 项统计为执行通过。
本轮 Windows 验证不能代替 Linux CUDA 或 macOS Metal 原生门禁。

## 双轴复审

| 轴 | 结果 |
| --- | --- |
| Standards | 复查固定点 `704e819` 以来的集成差异，未发现新的规范违反或需要提取的公共抽象 |
| Spec | 文档纳入提交与 Qwen CUDA 追加范围两项检查已落实；模型私有状态未被合并，性能与平台证据边界保留 |

## 后续追加：Qwen CUDA 调研

实测与候选状态见 [Qwen CUDA 性能调研](../qwen/qwen-cuda-performance-2026-10-18.md)。
当前 `qwen-cuda` 经模型库 `cuda` feature 与 `tts-candle-platform/cuda` 启用统一
Candle CUDA；Qwen 没有默认启用 cuDNN，其他后端的 cuDNN feature 合并会改变同进程
卷积实现，应另建对照。本地完整 0.6B Base 权重与 ICL 缓存用于显式探针，未下载模型。

主机 profiling 指向 CodePredictor，随后完成一次 cat 打包的无插桩 ABBA 初筛：
输出一致但未过收益门槛，候选源码已撤回；固定 15 步图尚未实现，不用 Vox 或 Metal
的收益代替 Qwen CUDA 的实际结果（该候选的实现与结果见文末追加）。

## 后续追加：CodePredictor 帧图已实现并合入主线

上述“固定 15 步图尚未实现”已于同日完成：commit `8ab5962`（把
`voxcpm::cuda_graph::replay` 提取为 `crates/candle-cuda-graph`，整数输出支持）、
`59a7d7b`（Qwen CP 帧图）、`94fada2`（Concat 变体实测）。正式 ABBA 八进程
（锁频 2692 MHz、1 预热 + 2 热轮、59 帧 / 4.72 s 音频、区间不重叠）：

| 指标 | eager | 帧图 | 改善 |
| --- | ---: | ---: | ---: |
| 完整合成 | 3.0840 s（RTF 0.653） | 1.9113 s（RTF 0.405） | -38.0% |
| 流式合成 | 3.6529 s（RTF 0.774） | 2.4353 s（RTF 0.516） | -33.3% |
| 首 PCM | 0.7688 s | 0.5663 s | -26.3% |

八进程 × 三轮 × 五类产物（codes、完整/流式原始 PCM、两份 WAV）跨变体 SHA-256
唯一，`qwen-cuda` 默认启用；完整边界见
[Qwen CP 帧图](../qwen/qwen-cp-graph-2026-10-18.md)。
