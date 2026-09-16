# RandLoRA-Damage Encoder v1.1.0 → v1.2.0 审计报告

## 总结

v1.1.0不是空壳，且已修复v1.0的大部分关键问题。对上传ZIP复核后，原21项测试可复现通过，参数预算147,456→rank 70→147,488也正确。

但原测试没有覆盖若干会影响正式训练、敏感度分析、checkpoint安全性和复现实验的边界情况。v1.2.0新增破坏性测试并完成修复。代码级结论是：**已达到可进入真实SAM工程100–200 step smoke的状态，但不能把本地测试等同于RTX3090和车辆损伤数据集正式验证。**

## v1.2新发现并修复的问题

### 1. eval缓存可能切断梯度（严重）

v1.1在`eval()`中缓存detached ΔW。`eval()`只切换Dropout/BN等模块行为，并不关闭autograd。因此先做一次`torch.no_grad()`验证，再在仍处于eval模式下做敏感度反传时，可能复用detached缓存，导致RandLoRA梯度静默消失。

修复：只有同时满足以下条件才缓存：

```text
module.training == False
cache_eval_delta == True
torch.is_grad_enabled() == False
```

任何eval-mode backward都会重新构造带图ΔW。

### 2. strict checkpoint可混入基础SAM权重（严重）

v1.1会把checkpoint中“当前模型也存在”的键视为可加载键，未强制限制为RandLoRA键。恶意或误打包文件可以包含`base_layer.weight`并覆盖冻结主干。

修复：adapter loader只接受λ、γ和注册随机基底键。普通SAM权重即使名称、shape和digest均合法，也会被列入`forbidden_non_adapter_keys`并在strict模式拒绝。

### 3. weakref导致deepcopy和整模型序列化不可靠（严重）

v1.1系数模块通过weakref引用共享基底：

- `copy.deepcopy`后的系数可能仍指向原模型基底；
- `torch.save(model)`会因weakref不可pickle而失败。

修复：改为强引用但不注册为子模块，避免每层state_dict重复基底，同时让Python deepcopy/pickle memo机制保持副本内部共享关系。`merged_copy`额外审计所有系数是否指向副本registry。

### 4. `adapter_dtype=float32`没有真正保证FP32（严重）

v1.1注入时使用FP32，但随后调用`model.half()`会把λ、γ和基底一起转成FP16；ambient autocast也可能让组合矩阵乘法降精度。

修复：

- basis bank和coefficients覆写`_apply`，父模型half/bfloat16后恢复adapter为FP32；
- FP32 adapter组合和factorized路径显式关闭CPU/CUDA autocast；
- 输出最后转换回基础层输出dtype。

### 5. occupancy-sensitive分配实际常常不生效（严重的实验设计Bug）

v1.1候选排序首先最小化预算误差，再最大化敏感度奖励。只要均匀rank能精确命中预算，它几乎总会胜出，导致所谓problem-driven allocation退化为原均匀设置。

修复：在`budget_tolerance`内先最大化damage priority，再用预算误差打破平局；若无候选满足容差，才退回最接近预算。增加组合数上限，防止指数搜索失控。

### 6. merge的“数值恢复”不等于精确恢复（中高）

v1.1保存历史ΔW并做减法，避免了系数变化后减错ΔW，但FP16/BF16下`(W+Δ)-Δ`不保证bitwise等于原W。

修复：merge时保存实际合并前weight或Q/V切片，unmerge通过`copy_`精确恢复。若已merge后调用`train(True)`，自动unmerge，避免无adapter梯度的静默训练。

### 7. checkpoint完整性仍不够完整（中高）

v1.1缺少：

- 文件内容摘要；
- 随机基底指纹；
- 原子保存；
- metadata完整性覆盖；
- alpha/scaling/dropout/sparse/dtype等完整配置比较；
- adapter state真正CPU clone快照。

修复：格式升级v3，加入state digest、basis fingerprints、完整manifest/config/resolved rank、metadata摘要、临时文件+`os.replace`、CPU clone。digest用于发现意外损坏，不是带密钥签名。v1.0/v1.1真实生成的checkpoint均已独立加载验证。

### 8. optimizer安全边界不足（中）

v1.1 duplicate检查会迭代现有参数组。若`params`是generator，检查会把它消费，后续构造optimizer时头部参数可能消失。

修复：先将一次性iterable物化为list；新增`add_randlora_to_optimizer()`，调用PyTorch公开`Optimizer.add_param_group`给已创建optimizer添加adapter组。

### 9. 注入与配置预检不完全（中）

新增修复：

- 重复Q/V或extra target提前拒绝；
- `auto_match_budget=True`时同时设置rank和target budget视为歧义并拒绝；
- 所有目标层先完整预检并构造wrapper，再改变原encoder；
- 非官方融合QKV形状、已包装目标和rank越界不留下半冻结模型。

### 10. 诊断与gate输入可静默污染结果（中）

新增：

- mask必须含batch和至少二维空间维；
- threshold必须在[0,1]；
- rank诊断拒绝NaN/Inf；
- gate拒绝NaN/Inf、非法指标范围和非法epoch。

### 11. 第一optimizer step的gamma梯度预期需要纠正（实验说明错误）

lambda初始化为0以保证no-op。此时：

```text
∂L/∂lambda 可以非零
∂L/∂gamma 包含lambda因子，因此第一步为零
```

不能把“第一步lambda和gamma都非零”作为smoke通过条件。v1.2脚本和文档已改为：第一步验证lambda非零，lambda更新后再观察gamma。

### 12. 默认 scaling 不能声称是论文视觉超参数（实验说明风险）

v1.1延续了 `alpha_multiplier=2.0`，其实际更新缩放恒为 `alpha/r=2`。该设置便于非均匀 rank 分配时保持每层尺度一致，也保留旧checkpoint/config语义，但它不是论文视觉实验中报告的 `10/r`。v1.2不擅自改变默认值，以免破坏与v1.1的配对和兼容性；新增文档要求正式实验明确记录实际 `rank/alpha/scaling`，并将“基线scale对齐”和“论文式alpha=10”视为可选消融，而不是混入首轮主比较。

## 保持不变的核心设计

- 完整RandLoRA逐基底公式；
- 共享固定随机A/B基底；
- 官方SAM1融合`nn.Linear(768,2304)`的Q/V切片注入；
- K严格不变；
- 初始no-op；
- Blocks 4–11首轮默认；
- rank 70和147,488参数参考预算；
- Prompt Encoder、Mask Decoder、TQ-MOTP、QJ-2、loss和评估器不变；
- 第二阶段可选proj/MLP；
- factorized/materialized/auto；
- 训练后merge与unload。

## 仍不能声称已经完成的内容

本环境无Meta官方`segment_anything`、SAM ViT-B checkpoint和CUDA GPU，因此未完成：

- 官方1024×1024完整Image Encoder真checkpoint前后向；
- RTX3090 AMP、峰值显存和step time；
- 你的实际SAM代码命名与TQ-MOTP/QJ-2优化器接入；
- 车辆损伤数据集20 epoch正式训练。

这些必须在你的`/workspace/sam44`真实环境运行随包脚本和100–200 step smoke后才能确认。
