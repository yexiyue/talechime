# 原生资源准备接口

`ResolvedModel::native_preparation_plan().await` 返回已登记配方的离线需求，资源摘要检查在受跟踪 worker 上完成，不阻塞引擎 owner 的事件循环。`None` 表示该模型尚无可执行的原生配方，不能将它当作资源就绪。

Kokoro 两变体、CosyVoice3 和 IndexTTS 均已在 SDK 登记，CastGlean 宿主流程仍在验收；下列代码说明 SDK 调用契约，不宣称桌面流程已验收。普通测试使用固定合成数据和本地 HTTP 服务，不下载权重；真实输入核验见 [2026-11-05 记录](records/resources/native-preparation-2026-11-05.md)。

```rust,no_run
# async fn example(model: &talechime::ResolvedModel) -> anyhow::Result<()> {
let Some(plan) = model.native_preparation_plan().await? else {
    anyhow::bail!("model has no verified native preparation recipe");
};
// 将 recipe、download_bytes、conversions 和 peak_additional_bytes 展示给用户。
// 用户确认后传回 fingerprint；不接受前端任意资源路径或转换程序。
let control = talechime::PreparationControl::default();
let (stages, mut stage_rx) = tokio::sync::mpsc::channel(32);
let (files, mut file_rx) = tokio::sync::mpsc::channel(32);
let execution = model.prepare_native_resources(plan.fingerprint, control, stages, files);
let consume = async {
    let stage_stream = async { while let Some(_stage) = stage_rx.recv().await {} };
    let file_stream = async { while let Some(_file) = file_rx.recv().await {} };
    tokio::join!(stage_stream, file_stream);
};
let (result, ()) = tokio::join!(execution, consume);
result?;
# Ok(())
# }
```

## 确认与空间

指纹绑定固定来源/revision/摘要、输出、转换版本、目标目录、下载缓存、端点、凭据/传输策略的不可逆身份以及当前准备需求。执行前重新计算；命中状态或配置改变后旧指纹失效。三类模型准备能力不会绕过宿主确认；下载继续使用同一个显式 `Downloads` 的镜像、令牌、传输选择和共享缓存。

`download_bytes` 只计算缺失输入；峰值空间保守覆盖目标输入视图、完整暂存输出和声明工作空间，同卷可复用的缓存硬链接不重复计输入。资源完成标记与输出全部验证后，`ready` 为 true、额外准备需求为零。不同卷的输入复制也计入预算。预估值是资源准备上界，不是文件最终体积或推理内存用量。

## 取消与发布

`PreparationControl` 可克隆到控制端；调用 `cancel()` 后等待执行返回，期间继续显示取消中。下载取消保留共享客户端断点；本地 worker 在 I/O/张量边界检查，终态前等待线程及暂存退出。发布与取消共享提交边界，已经提交成功后迟到取消不能改写成功事实。

准备命名空间为引擎资源根的 `.native-preparation/`：`inputs/<recipe>/` 是经固定摘要核验的输入视图，`components/<sha256>` 复用经固定输出摘要校验的完整组件（仅同卷硬链接缓存，不将部分文件当作完成）；`staging-*` 仅为本次完整转换暂存；目录受全局准备租约保护，不扫描用户音色、正文和历史目录。输出发布到模型配方版本隔离目录。`preparation.json` 记录固定配方输入/输出契约，不能用用户生成清单替代编译时信任的来源和输出摘要。

相同大小的损坏输入也会被拒绝；共享缓存命中但 SHA 错误时，经现有 HF 客户端执行至多一次强制刷新，不直接删除共享 blob。失败、空间不足或取消不覆盖已发布目录；未知或不一致的现有版本保留并报错。

阶段为下载、转换、校验和发布；转换用组件/张量计数或不定进度，不用字节速率伪造 ETA。阶段通知可合并，执行返回与宿主快照为终态权威。资源就绪、设备可用、加载和生成验收分别表达。

## 暂存恢复与故障边界

只有显式执行准备、且取得全局租约后，才清理当前配方的已标记暂存。清理要求 `staging-<recipe>-*` 名称、与编译配方完全相同的 `preparation.json`，以及目录中仅包含该配方声明的输出和父目录。未知配方、额外文件、符号链接和共享下载缓存一律保留。Registry 重建不恢复网络/转换，也不自动清理；宿主中断日志的显示由宿主负责。

发布标记在受控暂存创建时写入，但它不使暂存成为就绪版本；推理只解析版本目录。全部输出与标记校验、文件 flush/sync 后，整目录 rename 才是提交边界。已有目录的输出和标记必须都与当前配方一致，否则保留并报错。rename 前的路径/空间/写入错误不影响旧版本。Unix 在提交后同步父目录；如果该持久化检查失败，错误明确说明版本已发布但目录持久性未确认，迟到取消不能把已经提交的事实逆转成取消。

取消的执行 future 必须等待底层 worker 和暂存退出才返回；源码级阻塞 worker 测试在取消中尝试第二个准备租约，确认仍返回 Busy，释放原 worker 后才可重新获得租约。阶段 Sender 随执行退出关闭，不会有底层迟到写入；宿主仍须以任务身份与终态拒绝已经排队的旧阶段事件。相同 `PreparationControl` 只对应一次执行，不能在完成后复用到新任务。

每个完整组件先验证编译输出摘要再存入受控内容缓存，因此后续组件失败仍可在用户重试时复用已完成件。组件中断、摘要不符或未知缓存文件不能跳过转换或发布。测试另覆盖无下载计划、确认变化、损坏输入一次强制刷新、并发租约、空间不足、发布路径失败、已就绪旧版本可读及限定清理。

## 受限张量导入范围

这不是任意 pickle 转换器。原生配方只接受编译时批准的完整文件大小和 SHA-256，先核验整个输入，再符号化解析固定 PyTorch ZIP 元数据；不调用 Python，也不执行 GLOBAL/REDUCE 所携对象。未知对象不会被当作 tensor 接受；显式对象归约测试验证构造表达式不被执行。

仅支持固定 checkpoint 里的字符串映射、选定子映射、张量和受约束的 `_metadata` 描述；支持连续 C/Fortran view，正确处理 byte offset 到 element offset 的转换。目录名与 storage 路径必须安全，shape、stride、storage 端点和文件实际长度都先验证。元数据最多 8 MiB、条目最多 100,000、嵌套/张量 rank 最多 8、单张量最多 512 MiB、单组件输出最多 8 GiB。512 MiB 上限覆盖固定 IndexTTS 的 309,806,080 字节 F32 文本 embedding；不加载整套模型再转换。磁盘空间计划与推理 RAM 需求是不同概念，不把此单张量上限当作整个进程内存峰值。

普通测试覆盖嵌套映射、F32/F64、连续/转置 view 与非零 storage offset、未知对象、溢出/非连续 stride/越界 storage、输出预算与取消。真实对照另记录：Kokoro 599 项、CosyVoice3 HiFT 246 项、IndexTTS 六组件合计 2,707 项。IndexTTS 的原生文件已经与固定官方 PyTorch 导出按实际 dtype、shape、逐元素原始字节对齐；配置与复制资产另外校验，不以文件序列化摘要不同掩盖数值变化。

## 成品分发与源配方

默认 v3 配方直接取得经过逐张量核验的 F32 / F64 成品，`conversions` 为空，不下载原始 `.pt/.pth` 再执行转换。相同字节的组件继续使用固定上游 HF revision；缺少的成品使用公开非 gated 的 `yexiyue/talechime-resources`。源码固化每个输入的 40 位 revision、大小和 SHA-256；成品源不可用时报告失败，不静默切换源构建或扩大下载。

v2 原生源配方与固定张量导入器保留在各后端私有模块，用于维护方复现资源构建。明确运行被忽略的准备测试并设置 `TALECHIME_BUILD_NATIVE_SOURCE=1` 才选该路径；终端用户的公开准备入口默认使用 v3。已完成 v2 组件以摘要存储，v3 可核验复用；新发布目录独立，旧版本不删除。

成品复制/硬链接表达为校验阶段，不伪报转换或网络速度。发布仍检查所有输出和精确配方标记；模型推理器与原生源配方生成的张量契约保持一致。分发记录、上游许可和数值门禁另见资源核验文档，不把成品上传视为 SDK 或应用版本发布。
