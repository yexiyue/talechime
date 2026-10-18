# Metal

## 判断真实成本

先查正在使用的 Candle revision 的 Metal 算子实现和目标设备能力。
大词表排序、grouped conv 的每组循环、小张量 command encoding、布局物化和
CPU 读回后的队列同步是不同问题，分别做最小复现。
不套用 CUDA 的 fast path 或调度阈值，也不因统一内存就假定主机往返没有同步成本。

## 排序与采样

Candle 历史版本的大词表排序曾出现线程组/索引问题；这不是所有新版本或所有 Metal
设备的永久限制。核对 kernel 的线程数、padding、真实词表和返回索引，不只测试 4/32
列；验证稀疏 logits、同分、非排序词表 ID 和过滤后至少有候选。

Qwen 曾小词表测试通过、3072 词表过滤失败，真实生成反复选 token 0 并触达帧上限。
CPU 排名策略与一次 logits 回传是其已验证修复，不是所有模型必须使用 CPU 采样。
改变 top-p 边界属于准确性修复，另立性能基线。

若 trace 的回传 span 很长，先测已经同步完成的 logits 回传。Qwen 的历史约 19 ms
主要包含前序队列等待；不能据此承诺把 12 KiB 复制改小就会提速。
查 cumsum/排序等 API 的当前实现；矩阵乘三角矩阵的 CDF 可能比顺序扫描更贵，但
更换累加顺序也可能改变 token，必须先遵守主 skill 的数值门槛。

## 卷积与布局

grouped conv 若逐通道分派，先定位真实调用几何，再评估窗口堆叠和批量 matmul。
Qwen 的局部替换仅适用于 unit stride/dilation、每组单输入、groups=channels。
验证因果左 padding、kernel 方向、bias 广播、短输入、多 batch 和浮点归约。
不合格配置明确走 stock；性能阈值按当前设备重新测量。

逐 tap 广播相加可能改变求和顺序，即使误差很小也不能冒充逐位一致。
transpose/narrow 的非连续布局只在必要的消费边界物化，实测连续化成本和下游内核
选择；删除 contiguous 不是天然优化。一次设备 tensor 简化若无明确收益，只能作为
单独的可读性改动解释，不作性能宣传。

## 平台验证

依赖与源码测试都按 macOS + feature 门控。编译通过、设备枚举成功、实际 kernel
执行和端到端输出是不同层次。无 GPU 的原生 CI 不强行构造 Metal 设备。
macOS 统一内存峰值、进程 RSS 与独占 GPU 显存不可混为一个指标。
