# VoxCPM CUDA 单 query 注意力候选：数值拒绝与复制初筛

本文保存筛选阶段证据；后续Windows CUDA运行时接入、完整输出/状态及交错测量见
[K复制运行时验证](voxcpm-windows-key-packing-2026-10-25.md)。下文test-only描述指当时阶段。

Windows 11 / Ryzen 7 9700X / RTX 5070，Candle 0.11 fork 2266099d、CUDA13，
源码基点 `42a7887e28f095684266eac43a68bd8fcc303d11`。本轮没有改变生产注意力算术。
新增模块及输入捕获只在 cfg(test, cuda, Windows/Linux) 编译，Mac 未优化或验收。
所有证据在 `D:/workspace/talechime-benchmarks/windows-voxcpm-20261024`。

## 拒绝：把 GQA group 折入 query 行

VoxCPM 原 single-query 为16个Q头、2个KV头、head_dim128。折入group后可以减少
repeat-KV，但 GEMM 的行数由1变8，数学等价不代表 CUDA 归约逐位相同。
合成夹具覆盖 batch1/2、16种KV长度1..512、连续与转置K/V、三种dtype，共192组。

| dtype | 位差组数 / 组数 | 最大绝对误差 | 改变元素总数 / 元素总数 |
| --- | ---: | ---: | ---: |
| F32 | 60 / 64 | 4.76837e-7 | 114840 / 196608 |
| F16 | 20 / 64 | 0.00048828125 | 70 / 196608 |
| BF16 | 6 / 64 | 0.00390625 | 6 / 196608 |

例如F16在31/32/40/64/77/128/191/512均有失败，不能只选几个偶然相同的长度。
没有放宽容差，也没有进入整模型；原型已从活跃测试移除，源码/失败二进制/逐组数据
保留为 single-query-gqa-rejected-*、single-query-gqa-{oracle,summary}.json。
最初测试还发生JSON错误类型不兼容的编译失败，修正测试返回类型后才取得上述实测，
两个日志分开保留。此数值结果不能外推 OmniVoice ratio2、多query 的已验实现。

## 候选：只合并 K 的布局复制

保留 GEMM 形状及算术：原先先将K展开为重复头，再转置并contiguous，产生两次
物化；候选先transpose，再广播/reshape，直接得到相同的连续矩阵，省去中间副本。
V、softmax、缩放和context恢复不改变。长度1时原transpose已经无需复制，且singleton
stride不同，因此显式走原路径，不为没有收益的形状改变 GEMM dispatch。
首次对照正是通过stride断言发现该边界，日志 packed-key-oracle.log 保留；不是
把非零数值误差改成容差通过。

修正后192组的K值/有效strides与最终attention位模式一致，最终重跑
packed-key-final-oracle.{json,log}。合成成本初筛，每种dtype/长度/边界做32次预热，
单进程ABBAABBA，各block512次，CUDA前后同步，计host+GPU完成成本：
K准备点值减少约44%–50%；含context恢复的注意力减少约13%–33%。
这是合成筛选，不是产品提速，也不是3进程的完整请求基线。

## 真实模型输入回放

用当前Model类加载本机固定资源，Q8使用已有Candle参考cache，BF16为narrator。
资源/音色/cache/文本共10项重新校验，见 packed-key-real-inputs.json。
每种模型3独立进程，原路径合成一次63字段落并自然EOS，从单query调用中有界
捕获ordinal 0/27/35、576/603/611、2304/2331/2339的9组Q/K/V。
每组先逐位比较，再32次预热，ABBAABBA各256次，包含context transpose/reshape，不含o_proj。
旧/新路径均包含同等的inactive test hook开销，正式版本用u32位模式比较。
先前初次回放保留为preliminary，不与最终6进程汇总混用。

| 模型 | 实际KV长度 | 三进程局部改善点值范围 | 进程均值范围不重叠的输入 |
| --- | --- | --- | --- |
| Q8/F16 | 127、143、191 | 17.18%–32.15% | 9 / 9 |
| 原始BF16 | 66、82、130 | 14.41%–25.93% | 7 / 9 |

54组真实输入context全部逐位一致；各模型原路径三次raw PCM长度分别恒为
614400 / 837120样本，并自然EOS。该生成仍使用旧注意力，不能称为候选完整PCM验收。
每个block及进程均值在 packed-key-real-final-{q8,bf16}-{0..2}.json 与
packed-key-real-summary.json。范围不是置信区间；本地replay没有独立采样GPU时钟/
竞争进程，正式完整请求比较仍须使用xtask冻结两臂和资源观测。

## 补正：真实模型输入不等于生产宿主的相同请求

本次Model组件直接把全文一次送入模型，seed为7330414270409262323。
核对raw长度后发现，[此前段落记录](voxcpm-windows-paragraph-2026-10-25.md)漏列了
生产宿主内部拆分；已从两种模型36份ranges manifest逐一核对UTF-8范围和文本SHA：

| 生产内部调用 | UTF-8字节范围 | 实际seed |
| --- | --- | --- |
| call0 | [0,42) | 7330414270409262323 |
| call1 | [42,189) | 15690801507913080125 |

边界清单 paragraph-production-boundaries.json 绑定原始corpus SHA；seed用当前
rustc1.98.1按SeedPolicy原Hash类型重算，call0与既有收据复核，不视为跨编译器常量。
C++对照和本轮组件回放都是一次全文；Native首PCM及总时长还包含宿主拆分与后处理。
因此本轮只能证明该真实Model输入下的局部收益，不能直接代表生产两段的相同工作量，
也不能把raw BF16 17.44秒与生产15秒的差异当成候选少生成。

随后已按冻结段界和各自seed补齐组件回放，结果见下节。接候选到Windows CUDA之前仍须
验证完整latent patch/PCM/长度/EOS、取消与失败后复用并做完整请求ABBA。当前候选仍只有
test-only实现，不把局部数值称为TTS整链改善。

## 按生产段界补测

2026-10-25在同一Windows设备上，每种精度3独立进程，模型及参考只加载一次，
按call0/call1的顺序完成两次原路径生成，随后才回放所捕获的attention输入。
启动模型前检查原始corpus SHA、UTF-8范围、各段文本SHA与当前SeedPolicy计算的seed；
错误corpus SHA、非法UTF-8范围、错误seed三种负向均在加载前拒绝。
不能把此组件驱动称为实际Registry回调验收，它仅重建其冻结模型请求边界。

| 模型 | 精确context对数 | 每进程选中输入 | 局部改善点值范围 | 进程均值范围不重叠 |
| --- | --- | --- | --- | --- |
| Q8 | 36 | call0六组、call1六组 | 12.63%–21.70% | 10 / 12 |
| BF16 | 45 | call0六组、call1九组 | 9.86%–22.36% | 14 / 15 |

每组仍为32次预热及ABBAABBA各256次，包含context恢复，不含o_proj。
81组context逐位相同，所有原路径请求自然EOS。三进程各段raw样本数恒定：
Q8为145920/476160，BF16为176640/552960。这些生成使用旧注意力，且未经过宿主
后处理，不能据此宣称候选完整PCM相同，也不与生产导出长度直接混比。
原始记录为packed-key-segmented-final-{q8,bf16}-{0..2}.{json,log}，
汇总为packed-key-segmented-summary.json，新测试二进制及源身份在
packed-key-segment-tests-build.json；旧的全文单请求证据继续保留。

首次正向尝试在模型加载时CUDA OOM，保留packed-key-segments-q8-0.log。
当时系统余量较低；用户释放内存后GetPerformanceInfo可用commit约9.93GB，
六个正式补测进程前分别检查至少9GB余量后成功。该值是本轮native组件的启动预算，
不是内存预留或官方Python预算，也没有用它降低官方12GB门禁。
资源恢复快照不是失败瞬间采样，不把这次OOM原因确认为单纯显存不足。
普通测试10 passed、14 ignored，CUDA release all-targets Clippy、rustdoc及格式门禁
分别验证；本轮未改变运行时实现。

## 工程与复现

冻结单元测试exe与源摘要为packed-key-final-tests-build.json；普通测试10 passed、
14 ignored，release CUDA all-targets Clippy -D warnings、rustdoc -D warnings与格式
检查通过。LNK4098既有警告保留，未改变CRT。

```powershell
cargo test --locked --release -p talechime-voxcpm --features cuda packed_key_cuda_oracle -j1 -- --ignored --nocapture --test-threads=1
```

真实回放还须设置TALECHIME_VOX_GQA_MODEL、VARIANT、PROMPT（仅Q8）、TEXT、OUTPUT
对应的完整环境名（共享前缀TALECHIME_VOX_GQA_），filter为real_cuda_packed_key_replay。
OUTPUT必须是新文件，资源已准备；普通测试不下载模型。新源码改动后生成新测试收据，
不以Cargo可变路径覆盖已冻结二进制。
