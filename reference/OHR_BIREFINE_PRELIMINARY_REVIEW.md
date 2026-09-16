# OHR-BiRefine 接入准备

状态：已核查基线、数据和 CUDA；缺少引用对话中的原始模块 ZIP，尚未审查具体实现、接入或训练。

## 已核实

- 基线：`/workspace/sam44/formal/coco_bj823_three_experiments/experiment_1_uniform`。
- SAM1 ViT-B，59 类；Uniform RandLoRA blocks 4–11、q/v、预算 147456、seed 2026；TQMOTP 分类反向系数 0.25、QJ2 detach、oracle GT box。
- 基线完成 20 epoch；best epoch 18，D_legacy AP 0.6761107389269907。
- SAM wrapper 已返回 `upscaled_embedding [N,32,256,256]`，可在 decoder 输出处接入。
- torch 2.1.0、torchvision 0.16.0；独立 CUDA 前向及反向通过；该环境没有 scipy。
- 两个划分的标注 SHA256 与基线一致，所有图片存在，全部实例类别已映射。categories 中额外的 id=0 是无实例的背景类。详见 READINESS_AUDIT.json。

## 设计审查项目（未审查源码）

1. 独立 sigmoid gate 可同时处理两类错误；SDF 辅助监督需通过实际 residual 或输出约束影响最终 mask。
2. GT mask、GT occupancy 只能用于监督；推理输入遵守现有 oracle box 协议。
3. FP 距 GT >8px 说明远处误检，不能单独证明语义混淆，需可视化或邻接实例标注确认。
4. SDF 和 far-FP 在 256 mask 坐标计算后映射到 ROI，不能在 resize 后重新解释 8px。
5. negative-only Region 不能恢复 FN，需报告 recall/completeness、FN、empty-mask rate。
6. 若最大负 residual 小于 base positive logit，Region 不能删除该高置信 FP；高置信保护也可能保护错误区域。需核查 logit 分布和删除能力。
7. 高分辨率 RGB ROI 不代表高分辨率最终输出；若最终仍为 256 mask，需明确输出限制。
8. 验证 ROI paste、空实例、极小/越界 box、实例与图片索引、AMP、梯度、optimizer coverage、checkpoint strict roundtrip。
9. 原始 package 缺失时，不以重新编写的替代实现冒充原模块。

## 初步实验参数（待源码审查和真实 smoke 后定稿）

- 主实验从相同 common_init 开始，20 epoch、seed 2026、batch 2、accumulation 4、effective batch 8、cosine scheduler、5% warmup、clip 1.0。
- 保留基线 optimizer 学习率和 loss 权重，见 BASELINE_CONFIG.json；新增模块 AdamW LR 暂定 1e-4、weight decay 1e-4。
- ROI context 暂定 20%、feature backward scale 暂定 0.25；ROI 尺寸、SDF、suppression、gain 参数根据源码和 smoke 确定。
- 每 epoch 全验证集验证，沿用 D_legacy_AP 选择 best，补充 Boundary F1、远距 FP、FN、前景召回和小 occupancy 子集。
- 从训练后的 best.pth 微调只作为另行标注的探索实验，不作为相同训练预算的对比。
- 先实现审查，再真实训练/验证 smoke，再正式训练。4 张 GPU 当前均有负载，运行前检查余量。

## 参考

- [官方 SAM MaskDecoder](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py)
- [Boundary loss](https://arxiv.org/abs/1812.07032)
- [PointRend](https://arxiv.org/abs/1912.08193)

文献支持相关构件的可行性，不能替代当前模块在 coco_bj823 上的实验验证。
