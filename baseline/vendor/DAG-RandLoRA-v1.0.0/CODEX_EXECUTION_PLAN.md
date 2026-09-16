# Codex执行指令：DAG-RandLoRA（Damage-Aware Gradient RandLoRA）正式实验

请在现有 `/workspace/sam44` 工程中，基于已经验证有效的 **Uniform RandLoRA-Damage Encoder**，完成下一阶段 **DAG-RandLoRA** 实验。新增代码位于：

```text
/workspace/sam44/experiments/RandLoRA-Damage Encoder/
```

目标不是简单调RandLoRA超参，而是在**严格相同RandLoRA可训练参数预算**下，根据车辆损伤困难实例产生的真实Encoder梯度，把适配容量重新分配到更需要的SAM1 ViT-B层及Q/V投影。只使用一张GPU，不允许DDP/DataParallel/torchrun多卡。

## 0. 固定正式基线与实验协议

先审计工程并找到当前已经验证有效的Uniform RandLoRA正式run及其best checkpoint、真实adapter参数量、resolved basis rank、seed、训练初始化、optimizer groups和完整评估器。模型固定为：

```text
SAM1 ViT-B
+ RandLoRA Image Encoder
+ 当前最新且已验证的 PiCOPlus 分类头（DETACH）
+ QJ-2质量头（DETACH）
+ Oracle/GT Box Prompt（保持当前阶段协议）
```

分类loss和质量loss不得向Image Encoder回传；Encoder的DAG profiling只使用与正式训练一致的mask监督（原BCE/Dice及其权重），不要偷偷加入classification/QJ-2梯度。Prompt Encoder、Mask Decoder、分类头、质量头、数据划分、augmentation、loss、checkpoint选择规则和评估器全部沿用基线。

## 1. 两个必须再次审计的问题

### 1.1 SAM resize/坐标链
必须保持当前已验证链路：原图H×W → `ResizeLongestSide`最长边1024 → pad到1024×1024 → Encoder → 256 mask logits → SAM官方`postprocess_masks`去pad并恢复原图尺寸。Box必须使用同一套坐标变换。禁止直接拉伸1024×1024、box/image不同resize、256 mask直接与原图GT评估、pad区域进入指标。正式run前随机抽样断言最终mask尺寸等于原图尺寸。

### 1.2 CPU不能拖慢训练
训练step不要频繁`.cpu().numpy()`、逐实例RLE、Boundary-F1、可视化或Python循环。BCE/Dice/resize/threshold尽量留GPU。DataLoader合理使用`pin_memory=True`、`persistent_workers=True`、`non_blocking=True`和适量workers，不能盲目开大量CPU进程。完整COCO/RLE和重型指标仅在validation阶段做。先观察100–200 step GPU利用率；若GPU长期空闲且CPU满载，先修复瓶颈。

## 2. Phase P：只用训练集做梯度校准，禁止验证集泄漏

使用Uniform RandLoRA **best checkpoint作为只读reference/probe model**。创建独立模型副本，不更新任何参数；代码会临时只对Blocks 4–11融合QKV的base weight打开`requires_grad`以提取真实Q/V全权重梯度，结束后自动恢复。

从**training split**固定生成一个calibration manifest并保存image/instance IDs、seed和occupancy，后续R1/R2复用同一批数据，禁止使用val/test选择rank。

至少收集两套profile：

```text
hard_profile:
  occupancy <= 0.10的实例
  loss = 正式mask loss在这些实例上的mean

general_profile:
  同一calibration图像中的全部有效实例
  loss = 完全相同mask loss的mean
```

建议至少累计约512–1024个hard实例；若训练集中不足则全部使用。general使用相同calibration图像，不需要人为类均衡。每个batch loss必须按选中实例取mean，并把`selected_count`传给profiler。Probe阶段不要使用GradScaler，不调用optimizer.step，不更新BN/参数；模型用eval模式做确定性profile。

调用最新版接口：

```python
from randlora_damage import collect_qv_gradient_profile, save_gradient_profile
```

为hard/general分别运行profile并保存：

```text
profiles/hard_occ_le_010.pt
profiles/general_same_calibration.pt
profiles/calibration_manifest.json
profiles/PROFILE_REPORT.md
```

报告每个`block:q/v`的Frobenius gradient norm、样本数和是否存在NaN/Inf。

## 3. R1：OC-RandLoRA，先验证block级问题驱动分配

R1只做**block级capacity allocation**，同一个block中的Q/V共用rank，不使用GID。以Uniform RandLoRA的真实adapter参数量作为固定预算，不使用理论147456替代实际值。

若Uniform resolved rank≈70，第一轮候选只用保守集合：

```text
[64, 70, 80]
```

若实际rank不是70，则围绕真实rank选择3个相邻候选，必须包含baseline rank，且不要一次变化过大。

用：

```python
from randlora_damage import build_r1_oc_plan
```

hard gradient高、general gradient相对低的block获得更多RandLoRA scaling容量（更小basis rank）；低优先block获得更少容量。要求：

```text
|Params_R1 - Params_R0| / Params_R0 <= 1%
alpha/r保持与R0一致
Blocks仍固定4–11
仍只Q/V
```

保存`allocations/R1_OC_allocation.json`并审计每block rank、num_bases、实际参数量、budget error。

然后**重新从与R0相同的正式初始化开始训练R1**，不要从R0 best继续finetune；R0 best只用于训练集gradient profiling。训练20 epochs、相同seed、数据顺序、batch size、augmentation、optimizer头部分组、loss和评估器。

## 4. R2：DAG-RandLoRA，Q/V级GID + hard-specificity

R1完成后继续R2。若R1出现灾难性退化（例如D-AP75下降>0.010或box内FPR绝对增加>0.05），先检查profile/resize/预算，不盲目继续；若R1正向或基本中性，则正常进入R2。

R2直接复用Phase P的hard/general完整Q/V梯度矩阵，不需要重新采样。对每个`Block × {Q,V}`计算：

```text
GID = entropy effective rank of hard-instance full gradient
hard-specificity = ||G_hard||F / (||G_general||F + eps)
priority = z(log GID) + 0.5 * z(log hard-specificity)
```

当前正式分类头为DETACH，因此第一轮R2固定：

```text
classification_penalty = 0
```

不要为了“保护分类”临时把classification loss混进Encoder训练或主profile。若后续确实观察到分类明显下降，再单独设计context-protection消融。

如果R0 rank≈70，R2候选：

```text
[56, 64, 70, 80, 96]
```

调用：

```python
from randlora_damage import build_r2_dag_plan
```

R2允许同一block的Q/V使用不同rank，例如`8:q=64, 8:v=80`。仍要求：

```text
总RandLoRA可训练参数与R0误差 <= 1%
每个目标仍满足full-rank reachable
alpha/r与R0保持一致
不增加额外训练分支或推理模块
```

保存`allocations/R2_DAG_allocation.json`。然后同样从与R0/R1相同正式初始化重新训练20 epochs，不从R1 best继续训练。

## 5. 代码接入

本包v1.3支持：

```python
DAGAllocationPlan.load_json(...)
config_from_allocation_plan(...)
inject_randlora_damage_encoder(...)
```

R1使用`block_rank_pattern`；R2使用`target_rank_pattern`（键如`"8:q"`、`"8:v"`）。只替换原Uniform RandLoRA注入配置，其他训练器逻辑不要重写。

优化器必须继续保留现有PiCOPlus/QJ-2各自LR、weight decay、bias/LayerNorm no-decay规则，仅把新的RandLoRA参数接入原Encoder PEFT参数组。CLS/QJ-2 detach梯度审计必须继续PASS。

## 6. 单GPU和smoke test

先`nvidia-smi`选择一张空闲/负载最低的RTX3090，只设置一个`CUDA_VISIBLE_DEVICES=<id>`。禁止占用第二张GPU。

每个正式run前做100–200 optimizer-step smoke，确认：

- 只有一张GPU有本实验进程；
- R1/R2实际rank pattern与JSON完全一致；
- base SAM Encoder冻结；
- Q/V RandLoRA lambda有有限非零梯度；初始lambda=0时gamma第一步允许为0，后续应有梯度；
- K切片严格不被adapter更新；
- CLS/QJ-2 detach不向Encoder回传；
- AMP/GradScaler无NaN/Inf；
- resize/box/postprocess尺寸链PASS；
- optimizer参数组未被破坏；
- 峰值显存、step time、GPU utilization正常。

## 7. 完整评价指标

R0、R1、R2必须在**完整validation set**使用完全相同评估协议，至少输出：

```text
COCO Mask: AP, AP50, AP75, APs, APm, APl,
           AR1, AR10, AR100, ARs, ARm, ARl
现有正式协议: D-primary AP/AP50/AP75、B-QJ2等已有指标
Mask: Mean IoU, Dice, Boundary-F1
Classification: Accuracy, Macro-F1, per-class Precision/Recall/F1, confusion matrix
Difficulty: occupancy<=0.10的AP/AP75(若支持)、Mean IoU、Dice、box内背景FPR、over-segmentation
Per-class: 全类别AP，重点列Glass_crack、Scratch/Scratches、Dislocation等困难类
Efficiency: trainable params、peak GPU MiB、平均step time、GPU利用率
```

同时至少按occupancy分桶保存`<=0.10 / 0.10–0.30 / 0.30–0.50 / >0.50`结果，判断收益是否真集中在目标困难实例，而不是只改善分类。

## 8. 判定与最终报告

不要用训练中间波动冒充最终提升，checkpoint仍按现有正式D-primary规则选。R1作为低成本信号实验：若D-AP75约提升>=0.004或low-occupancy Mean IoU提升>=0.010，同时overall D-AP基本不降、Accuracy下降<=0.5个百分点且box FPR不恶化，则认为值得继续。R2若进一步稳定优于R0/R1，则再考虑三seed正式验证。

最终生成：

```text
summary/FINAL_DAG_RANDLORA_REPORT.md
summary/R0_R1_R2_metrics.csv
summary/rank_allocation_table.csv
summary/profile_statistics.csv
summary/smoke_audit.json
```

报告必须明确：R0/R1/R2模型差异、profile仅来自training split、实际参数预算、每个block/Q/V rank、GID与hard-specificity、完整指标差值以及失败案例。只报告真实运行结果，不伪造。

## 9. 本轮明确不做的事情

不要加入动态input gate、AdaLoRA式训练中rank裁剪、Fourier/卷积/拓扑模块，也不要把“gradient-informed basis selection”作为本轮创新；已有GaRA-SAM、AdaLoRA类工作以及2026 GiVA会造成明显方法重叠。本轮唯一研究变量是：**车辆损伤困难实例梯度驱动的RandLoRA full-rank capacity allocation**。

完成R0审计→Phase P→R1→R2（满足上述安全条件）后停止，不自动继续新的方法或多seed实验。
