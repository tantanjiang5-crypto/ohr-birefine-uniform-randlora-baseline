from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mq_rs_sam.quality_targets import boundary_f1_score

from .io_utils import atomic_json_dump


COCO_STAT_NAMES = [
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
]


def _output_field(output: Any, name: str) -> Tensor:
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, dict) and name in output:
        return output[name]
    raise AttributeError(f"model output lacks {name}")


def _forward_prepared(model: torch.nn.Module, prepared: Any, *, training: bool = False, epoch: int = 0) -> Any:
    """Route optional experiment-head inputs without changing the V0 call path."""
    if getattr(model, "requires_hrda_inputs", False):
        return model(
            prepared.images,
            prepared.boxes_list,
            raw_images=prepared.raw_images,
            boxes_original_list=prepared.boxes_original_list,
            original_sizes=prepared.original_sizes,
            labels_list=prepared.labels_list,
        )
    if getattr(model, "requires_pico_inputs", False):
        labels = torch.cat(prepared.labels_list, dim=0) if training else None
        valid_image_sizes = torch.tensor(
            prepared.input_sizes, dtype=torch.float32, device=prepared.images.device
        )
        return model(
            prepared.images,
            prepared.boxes_list,
            labels=labels,
            valid_image_sizes=valid_image_sizes,
            epoch=int(epoch),
        )
    if getattr(model, "requires_dcb_inputs", False):
        return model(prepared.images, prepared.boxes_list, epoch=int(epoch))
    return model(prepared.images, prepared.boxes_list)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _confidence_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)), "max": float(values.max()),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if x_array.size < 2 or x_array.shape != y_array.shape:
        return float("nan")
    x_centered = x_array - x_array.mean()
    y_centered = y_array - y_array.mean()
    denominator = math.sqrt(float(np.square(x_centered).sum() * np.square(y_centered).sum()))
    if denominator <= 1e-15:
        return float("nan")
    return float((x_centered * y_centered).sum() / denominator)


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if x_array.size < 2 or x_array.shape != y_array.shape:
        return float("nan")
    return pearson(_average_ranks(x_array), _average_ranks(y_array))


def hard_ece(confidence: np.ndarray, correctness: np.ndarray, bins: int = 15) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if confidence.size == 0:
        return float("nan")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= boundaries[index]) & (confidence <= boundaries[index + 1])
        else:
            selected = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1])
        if selected.any():
            value += selected.mean() * abs(confidence[selected].mean() - correctness[selected].mean())
    return float(value)


def soft_ece(confidence: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    return hard_ece(confidence, target, bins=bins)


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    bins: int,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = len(class_names)
    predictions = probabilities.argmax(axis=1) if probabilities.size else np.empty(0, dtype=np.int64)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[target, prediction] += 1
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted_support, out=np.zeros_like(tp), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    total = max(int(confusion.sum()), 1)
    accuracy = float(tp.sum() / total)
    weights = support / max(float(support.sum()), 1.0)
    confidence = probabilities.max(axis=1) if probabilities.size else np.empty(0)
    correctness = (predictions == labels).astype(np.float64)
    if probabilities.size:
        one_hot = np.eye(classes, dtype=np.float64)[labels]
        multiclass_brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
        negative_log_likelihood = float(-np.log(np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
    else:
        multiclass_brier = float("nan")
        negative_log_likelihood = float("nan")
    return {
        "count": int(labels.size),
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * weights).sum()),
        "micro_f1": accuracy,
        "balanced_accuracy": float(recall.mean()),
        "ece": hard_ece(confidence, correctness, bins=bins),
        "multiclass_brier": multiclass_brier,
        "negative_log_likelihood": negative_log_likelihood,
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            class_names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(classes)
        },
    }


def _encode_binary_mask(mask: np.ndarray) -> tuple[dict[str, Any], list[float], float]:
    from pycocotools import mask as mask_utils
    binary = np.asarray(mask, dtype=np.uint8, order="F")
    rle = mask_utils.encode(binary)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    bbox = [float(value) for value in mask_utils.toBbox(rle).tolist()]
    area = float(mask_utils.area(rle))
    return rle, bbox, area


def _extract_per_category(coco_eval: Any, category_names: list[str]) -> dict[str, Any]:
    precision = coco_eval.eval.get("precision")
    recall = coco_eval.eval.get("recall")
    if precision is None or recall is None:
        return {}
    iou_thresholds = np.asarray(coco_eval.params.iouThrs)
    index_50 = int(np.argmin(np.abs(iou_thresholds - 0.50)))
    index_75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))
    result: dict[str, Any] = {}
    for category_index, name in enumerate(category_names):
        all_precision = precision[:, :, category_index, 0, -1]
        p50 = precision[index_50, :, category_index, 0, -1]
        p75 = precision[index_75, :, category_index, 0, -1]
        category_recall = recall[:, category_index, 0, -1]
        valid_all = all_precision[all_precision > -1]
        valid_50 = p50[p50 > -1]
        valid_75 = p75[p75 > -1]
        valid_recall = category_recall[category_recall > -1]
        result[name] = {
            "AP": _safe_mean(valid_all),
            "AP50": _safe_mean(valid_50),
            "AP75": _safe_mean(valid_75),
            "AR100": _safe_mean(valid_recall),
        }
    return result


def run_coco_segm_eval(
    coco_gt: Any,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    prediction_path: Path,
    category_names: list[str],
) -> dict[str, Any]:
    from pycocotools.cocoeval import COCOeval
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    if not predictions:
        return {
            **{name: 0.0 for name in COCO_STAT_NAMES},
            "per_category": {name: {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0} for name in category_names},
            "prediction_count": 0,
        }
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.params.imgIds = sorted(set(int(value) for value in image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {name: float(evaluator.stats[index]) for index, name in enumerate(COCO_STAT_NAMES)}
    metrics["per_category"] = _extract_per_category(evaluator, category_names)
    metrics["prediction_count"] = len(predictions)
    return metrics


def _coco_with_annotation_ids(dataset: Any, annotation_ids: set[int]) -> Any:
    """Build an in-memory COCO GT view for the fixed low-occupancy subset."""
    from pycocotools.coco import COCO
    subset = COCO()
    source = dataset.coco.dataset
    subset.dataset = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key not in {"annotations", "images", "categories"}
    }
    subset.dataset["images"] = copy.deepcopy(source.get("images", []))
    subset.dataset["categories"] = copy.deepcopy(source.get("categories", []))
    subset.dataset["annotations"] = [
        copy.deepcopy(annotation) for annotation in source.get("annotations", [])
        if int(annotation["id"]) in annotation_ids
    ]
    subset.createIndex()
    return subset


def _batch_mask_instance_metrics(predicted: Tensor, target: Tensor) -> Tensor:
    """Compute pixel and boundary metrics for all instances on the tensor device.

    RLE encoding and COCOeval remain CPU-only.  The old evaluator moved every
    mask to CPU and ran six reductions plus four morphology operations once per
    instance; this batched form is algebraically identical and avoids that
    host-side bottleneck and repeated CUDA synchronizations.
    """
    if predicted.ndim != 3 or target.ndim != 3 or predicted.shape != target.shape:
        raise ValueError(f"expected equal [N,H,W] masks, got {tuple(predicted.shape)} and {tuple(target.shape)}")
    predicted = predicted.float()
    target = target.float()
    dimensions = (1, 2)
    intersection = (predicted * target).sum(dim=dimensions)
    union = ((predicted + target) > 0).float().sum(dim=dimensions)
    pred_area = predicted.sum(dim=dimensions)
    gt_area = target.sum(dim=dimensions)
    iou = intersection / union.clamp_min(1e-6)
    dice = 2.0 * intersection / (pred_area + gt_area).clamp_min(1e-6)
    completeness = intersection / gt_area.clamp_min(1e-6)
    boundary_f1, boundary_precision, boundary_recall = boundary_f1_score(
        predicted[:, None], target[:, None], width=1, tolerance=2
    )
    return torch.stack((iou, dice, completeness, boundary_f1, boundary_precision, boundary_recall), dim=1)


def evaluate_model(
    *,
    model: torch.nn.Module,
    preprocessor: Any,
    loader: Any,
    output_dir: str | Path,
    split_name: str,
    score_cfg: dict[str, Any],
    calibration_bins: int = 15,
    max_images: int = 0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = loader.dataset
    model.eval()
    if hasattr(model, "set_lora_dropout"):
        model.set_lora_dropout(False)

    protocols: dict[str, list[dict[str, Any]]] = {
        "B_qj2": [],
        "B_native_iou": [],
        "C_legacy": [],
        "C_class_only": [],
        "D_legacy": [],
        "D_balanced": [],
        "D_class_only": [],
        "D_qj2_only": [],
    }
    low_occupancy_d_predictions: list[dict[str, Any]] = []
    low_occupancy_annotation_ids: set[int] = set()
    instance_rows: list[dict[str, Any]] = []
    probabilities_all: list[np.ndarray] = []
    labels_all: list[int] = []
    image_ids_seen: list[int] = []
    processed_images = 0

    with torch.inference_mode():
        for batch in loader:
            if max_images and processed_images >= max_images:
                break
            if max_images and processed_images + len(batch) > max_images:
                batch = batch[: max_images - processed_images]
            prepared = preprocessor.process(batch)
            output = _forward_prepared(model, prepared)
            logits = _output_field(output, "class_logits").float()
            pred_masks = _output_field(output, "pred_masks")
            native_iou = _output_field(output, "original_pred_iou").float().clamp(0, 1)
            qj2_raw = _output_field(output, "effective_quality")
            qj2_available = qj2_raw is not None
            qj2 = qj2_raw.float().clamp(0, 1) if qj2_available else None
            if logits.shape[0] != sum(labels.numel() for labels in prepared.labels_list):
                raise RuntimeError("evaluation instance alignment failure")
            probabilities = logits.softmax(dim=-1)
            predicted_labels = probabilities.argmax(dim=-1)
            class_probability = probabilities.gather(1, predicted_labels[:, None]).squeeze(1)
            pico_weights = getattr(output, "pico_view_weights", None)
            pico_part_source = getattr(output, "pico_part_box_source", None)
            pico_mask_stats = getattr(output, "pico_mask_stats", None)
            if qj2_available:
                legacy = class_probability.pow(float(score_cfg.get("legacy_class_exponent", 0.5))) * qj2.pow(
                    float(score_cfg.get("legacy_quality_exponent", 1.5))
                )
                balanced = class_probability * qj2
            else:
                # This is an explicit V0 ablation score, not an implicit
                # substitution for QJ-2: QJ-only protocols stay unavailable.
                legacy = class_probability
                balanced = class_probability

            offset = 0
            for image_index, item in enumerate(batch):
                count = int(prepared.labels_list[image_index].numel())
                image_ids_seen.append(int(item["image_id"]))
                processed_images += 1
                if count == 0:
                    continue
                current = slice(offset, offset + count)
                post_logits = preprocessor.postprocess_instance_logits(
                    pred_masks[current], prepared.input_sizes[image_index], prepared.original_sizes[image_index]
                )
                # Keep threshold/reductions/boundary morphology on CUDA as one
                # instance batch.  Only the RLE inputs and final scalar rows
                # cross to CPU once per image.
                predicted_binary_gpu = (post_logits > 0.0).float()
                gt_masks_cpu = torch.as_tensor(item["masks"], dtype=torch.float32)
                gt_masks_gpu = gt_masks_cpu.to(predicted_binary_gpu.device, non_blocking=True)
                metric_values = _batch_mask_instance_metrics(predicted_binary_gpu, gt_masks_gpu)
                # All occupancy/FPR/over-segmentation reductions stay batched
                # on GPU.  The CPU receives only N scalar diagnostics.
                boxes_original_gpu = torch.as_tensor(item["boxes"], dtype=torch.float32, device=predicted_binary_gpu.device)
                height, width = prepared.original_sizes[image_index]
                ys = torch.arange(height, device=predicted_binary_gpu.device)[None, :, None]
                xs = torch.arange(width, device=predicted_binary_gpu.device)[None, None, :]
                x1 = boxes_original_gpu[:, 0].floor().clamp(0, width)[:, None, None]
                y1 = boxes_original_gpu[:, 1].floor().clamp(0, height)[:, None, None]
                x2 = boxes_original_gpu[:, 2].ceil().clamp(0, width)[:, None, None]
                y2 = boxes_original_gpu[:, 3].ceil().clamp(0, height)[:, None, None]
                inside_box = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
                gt_bool = gt_masks_gpu > 0.5
                pred_bool = predicted_binary_gpu > 0.5
                box_area = inside_box.sum(dim=(1, 2)).float().clamp_min(1.0)
                occupancy = gt_bool.sum(dim=(1, 2)).float() / box_area
                box_background = inside_box & ~gt_bool
                box_background_fpr = (
                    (pred_bool & box_background).sum(dim=(1, 2)).float()
                    / box_background.sum(dim=(1, 2)).float().clamp_min(1.0)
                )
                oversegmentation = (
                    (pred_bool.sum(dim=(1, 2)).float() - gt_bool.sum(dim=(1, 2)).float()).clamp_min(0.0)
                    / gt_bool.sum(dim=(1, 2)).float().clamp_min(1.0)
                )
                extra_cpu = torch.stack((occupancy, box_background_fpr, oversegmentation), dim=1).detach().cpu().numpy()
                # These compact per-image outputs are consumed immediately by
                # Python/RLE.  Keep the transfer synchronous; async D2H would
                # expose an unfinished buffer.  This is still one transfer per
                # tensor batch rather than the former per-instance transfers.
                metric_cpu = metric_values.detach().cpu().numpy()
                predicted_binary = predicted_binary_gpu.detach().cpu()
                labels = prepared.labels_list[image_index].detach().cpu()
                probs_cpu = probabilities[current].detach().float().cpu().numpy()
                predicted_labels_cpu = predicted_labels[current].detach().cpu().tolist()
                scores = [class_probability[current], native_iou[current], legacy[current], balanced[current]]
                if qj2_available:
                    scores.append(qj2[current])
                score_cpu = torch.stack(scores).detach().cpu().tolist()
                if pico_weights is not None:
                    pico_columns = [pico_weights[current].float()]
                    if pico_part_source is not None:
                        pico_columns.append(pico_part_source[current, None].float())
                    if pico_mask_stats is not None:
                        pico_columns.append(pico_mask_stats[current, :1].float())
                    pico_cpu = torch.cat(pico_columns, dim=1).detach().cpu().numpy()
                else:
                    pico_cpu = None
                probabilities_all.append(probs_cpu)
                labels_all.extend(labels.tolist())

                for local_index in range(count):
                    global_index = offset + local_index
                    true_label = int(labels[local_index])
                    predicted_label = int(predicted_labels_cpu[local_index])
                    true_category = int(dataset.contiguous_to_category[true_label])
                    predicted_category = int(dataset.contiguous_to_category[predicted_label])
                    pred_np = predicted_binary[local_index].numpy().astype(np.uint8)
                    gt_np = gt_masks_cpu[local_index].numpy().astype(np.uint8)
                    pred_rle, pred_bbox, pred_area = _encode_binary_mask(pred_np)
                    gt_rle, gt_bbox, gt_area = _encode_binary_mask(gt_np)
                    iou, dice, completeness, bf1, boundary_precision, boundary_recall = (
                        float(value) for value in metric_cpu[local_index]
                    )
                    mq_quality = 0.5 * iou + 0.3 * bf1 + 0.2 * completeness
                    class_score = float(score_cpu[0][local_index])
                    native_score = float(score_cpu[1][local_index])
                    legacy_score = float(score_cpu[2][local_index])
                    balanced_score = float(score_cpu[3][local_index])
                    qj2_score = float(score_cpu[4][local_index]) if qj2_available else None
                    common_pred = {
                        "image_id": int(item["image_id"]),
                        "segmentation": pred_rle,
                        "bbox": pred_bbox,
                        "area": pred_area,
                    }
                    common_gt = {
                        "image_id": int(item["image_id"]),
                        "segmentation": gt_rle,
                        "bbox": gt_bbox,
                        "area": gt_area,
                    }
                    if qj2_available:
                        protocols["B_qj2"].append({**common_pred, "category_id": true_category, "score": qj2_score})
                    protocols["B_native_iou"].append({**common_pred, "category_id": true_category, "score": native_score})
                    protocols["C_legacy"].append({**common_gt, "category_id": predicted_category, "score": legacy_score})
                    protocols["C_class_only"].append({**common_gt, "category_id": predicted_category, "score": class_score})
                    protocols["D_legacy"].append({**common_pred, "category_id": predicted_category, "score": legacy_score})
                    protocols["D_balanced"].append({**common_pred, "category_id": predicted_category, "score": balanced_score})
                    protocols["D_class_only"].append({**common_pred, "category_id": predicted_category, "score": class_score})
                    if qj2_available:
                        protocols["D_qj2_only"].append({**common_pred, "category_id": predicted_category, "score": qj2_score})
                    annotation_ids = item.get("annotation_ids", [])
                    annotation_id = int(annotation_ids[local_index]) if local_index < len(annotation_ids) else -1
                    is_low_occupancy = bool(extra_cpu[local_index, 0] <= 0.1)
                    if is_low_occupancy:
                        low_occupancy_d_predictions.append(
                            {**common_pred, "category_id": predicted_category, "score": legacy_score}
                        )
                        if annotation_id >= 0:
                            low_occupancy_annotation_ids.add(annotation_id)
                    instance_rows.append(
                        {
                            "image_id": int(item["image_id"]),
                            "annotation_id": annotation_id,
                            "true_label": true_label,
                            "true_class": dataset.class_names[true_label],
                            "predicted_label": predicted_label,
                            "predicted_class": dataset.class_names[predicted_label],
                            "correct": int(true_label == predicted_label),
                            "class_probability": class_score,
                            "true_class_probability": float(probs_cpu[local_index, true_label]),
                            "qj2_quality": qj2_score,
                            "native_iou_prediction": native_score,
                            "d_legacy": legacy_score,
                            "d_balanced": balanced_score,
                            "mask_iou": iou,
                            "mask_dice": dice,
                            "boundary_f1": bf1,
                            "boundary_precision": boundary_precision,
                            "boundary_recall": boundary_recall,
                            "completeness": completeness,
                            "mq_quality": mq_quality,
                            "predicted_mask_area": pred_area,
                            "gt_mask_area": gt_area,
                            "occupancy": float(extra_cpu[local_index, 0]),
                            "box_inner_background_fpr": float(extra_cpu[local_index, 1]),
                            "oversegmentation_ratio": float(extra_cpu[local_index, 2]),
                            "pico_global_gate": float(pico_cpu[local_index, 0]) if pico_cpu is not None else None,
                            "pico_part_gate": float(pico_cpu[local_index, 1]) if pico_cpu is not None else None,
                            "pico_mask_gate": float(pico_cpu[local_index, 2]) if pico_cpu is not None else None,
                            "pico_real_part_box": int(pico_cpu[local_index, 3]) if pico_cpu is not None and pico_part_source is not None else None,
                            "pico_mask_occupancy": float(pico_cpu[local_index, -1]) if pico_cpu is not None and pico_mask_stats is not None else None,
                        }
                    )
                offset += count
            if offset != logits.shape[0]:
                raise RuntimeError(f"evaluation offset mismatch: {offset} vs {logits.shape[0]}")

    if probabilities_all:
        probability_matrix = np.concatenate(probabilities_all, axis=0)
    else:
        probability_matrix = np.zeros((0, len(dataset.class_names)), dtype=np.float32)
    labels_array = np.asarray(labels_all, dtype=np.int64)
    classification = classification_metrics(probability_matrix, labels_array, dataset.class_names, calibration_bins)

    prediction_array = probability_matrix.argmax(axis=1) if probability_matrix.size else np.empty(0, dtype=np.int64)
    max_confidence = probability_matrix.max(axis=1) if probability_matrix.size else np.empty(0, dtype=np.float64)
    true_confidence = (
        probability_matrix[np.arange(labels_array.size), labels_array]
        if probability_matrix.size else np.empty(0, dtype=np.float64)
    )
    confidence_statistics: dict[str, Any] = {
        "global_predicted_class_confidence": _confidence_summary(max_confidence),
        "global_true_class_probability": _confidence_summary(true_confidence),
        "per_true_class": {},
        "per_predicted_class": {},
    }
    for class_index, class_name in enumerate(dataset.class_names):
        confidence_statistics["per_true_class"][class_name] = _confidence_summary(max_confidence[labels_array == class_index])
        confidence_statistics["per_predicted_class"][class_name] = _confidence_summary(max_confidence[prediction_array == class_index])
    confusion_pairs = [
        ("Deformation", "Scratches"), ("Scratch", "Scratches"),
        ("Glass_breakage", "Glass_crack"), ("Dislocation", "Missing"),
    ]
    pair_errors: dict[str, Any] = {}
    class_to_index = {name: index for index, name in enumerate(dataset.class_names)}
    for left, right in confusion_pairs:
        if left not in class_to_index or right not in class_to_index:
            continue
        li, ri = class_to_index[left], class_to_index[right]
        left_support = int((labels_array == li).sum())
        right_support = int((labels_array == ri).sum())
        left_to_right = int(((labels_array == li) & (prediction_array == ri)).sum())
        right_to_left = int(((labels_array == ri) & (prediction_array == li)).sum())
        pair_errors[f"{left}<->{right}"] = {
            f"{left}_to_{right}": {"errors": left_to_right, "rate": float(left_to_right / max(left_support, 1)), "support": left_support},
            f"{right}_to_{left}": {"errors": right_to_left, "rate": float(right_to_left / max(right_support, 1)), "support": right_support},
            "bidirectional_errors": left_to_right + right_to_left,
            "bidirectional_rate": float((left_to_right + right_to_left) / max(left_support + right_support, 1)),
        }
    pico_rows = [row for row in instance_rows if row["pico_global_gate"] is not None]
    if pico_rows:
        gate_distribution: dict[str, Any] = {
            "enabled": True,
            "global": _confidence_summary(np.asarray([row["pico_global_gate"] for row in pico_rows])),
            "part": _confidence_summary(np.asarray([row["pico_part_gate"] for row in pico_rows])),
            "mask": _confidence_summary(np.asarray([row["pico_mask_gate"] for row in pico_rows])),
            "real_part_box_ratio": float(np.mean([row["pico_real_part_box"] for row in pico_rows])),
            "by_true_class": {},
            "by_correctness": {},
        }
        for class_name in dataset.class_names:
            selected = [row for row in pico_rows if row["true_class"] == class_name]
            gate_distribution["by_true_class"][class_name] = {
                name: _confidence_summary(np.asarray([row[f"pico_{name}_gate"] for row in selected]))
                for name in ("global", "part", "mask")
            }
        for correct in (0, 1):
            selected = [row for row in pico_rows if row["correct"] == correct]
            gate_distribution["by_correctness"][str(correct)] = {
                name: _confidence_summary(np.asarray([row[f"pico_{name}_gate"] for row in selected]))
                for name in ("global", "part", "mask")
            }
    else:
        gate_distribution = {"enabled": False}

    def column(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in instance_rows], dtype=np.float64)

    mask_iou = column("mask_iou")
    mask_dice = column("mask_dice")
    prediction_area = column("predicted_mask_area")
    gt_area = column("gt_mask_area")
    area_ratio = prediction_area / np.maximum(gt_area, 1e-6)
    mask_metrics = {
        "count": len(instance_rows),
        "mean_iou": _safe_mean(mask_iou),
        "median_iou": float(np.median(mask_iou)) if mask_iou.size else float("nan"),
        "mean_dice": _safe_mean(mask_dice),
        "median_dice": float(np.median(mask_dice)) if mask_dice.size else float("nan"),
        "iou_ge_0_50": float((mask_iou >= 0.50).mean()) if mask_iou.size else float("nan"),
        "iou_ge_0_75": float((mask_iou >= 0.75).mean()) if mask_iou.size else float("nan"),
        "mean_boundary_f1": _safe_mean(column("boundary_f1")),
        "mean_completeness": _safe_mean(column("completeness")),
        "empty_prediction_rate": float((column("predicted_mask_area") == 0).mean()) if instance_rows else float("nan"),
        "per_class": {},
    }
    for class_index, class_name in enumerate(dataset.class_names):
        selected = np.asarray([int(row["true_label"]) == class_index for row in instance_rows], dtype=bool)
        mask_metrics["per_class"][class_name] = {
            "count": int(selected.sum()),
            "mean_iou": _safe_mean(mask_iou[selected]),
            "mean_dice": _safe_mean(mask_dice[selected]),
            "mean_boundary_f1": _safe_mean(column("boundary_f1")[selected]),
            "mean_completeness": _safe_mean(column("completeness")[selected]),
        }

    area_metrics = {
        "count": len(instance_rows),
        "mean_prediction_to_gt_area_ratio": _safe_mean(area_ratio),
        "median_prediction_to_gt_area_ratio": float(np.median(area_ratio)) if area_ratio.size else float("nan"),
        "p10_prediction_to_gt_area_ratio": float(np.percentile(area_ratio, 10)) if area_ratio.size else float("nan"),
        "p90_prediction_to_gt_area_ratio": float(np.percentile(area_ratio, 90)) if area_ratio.size else float("nan"),
        "fp_expansion_ratio": _safe_mean(np.maximum(prediction_area - gt_area, 0.0) / np.maximum(gt_area, 1e-6)),
        "fn_shrink_ratio": _safe_mean(np.maximum(gt_area - prediction_area, 0.0) / np.maximum(gt_area, 1e-6)),
        "mean_oversegmentation_ratio": _safe_mean(column("oversegmentation_ratio")),
        "box_inner_background_fpr": _safe_mean(column("box_inner_background_fpr")),
    }

    quality_targets = {
        "mask_iou": column("mask_iou"),
        "mq_quality": column("mq_quality"),
    }
    score_columns = {
        "class_probability": column("class_probability"),
        "true_class_probability": column("true_class_probability"),
        "native_iou_prediction": column("native_iou_prediction"),
        "d_legacy": column("d_legacy"),
        "d_balanced": column("d_balanced"),
    }
    if instance_rows and instance_rows[0]["qj2_quality"] is not None:
        score_columns["qj2_quality"] = column("qj2_quality")
    calibration: dict[str, Any] = {}
    for score_name, scores in score_columns.items():
        calibration[score_name] = {}
        for target_name, targets in quality_targets.items():
            calibration[score_name][target_name] = {
                "pearson": pearson(scores, targets),
                "spearman": spearman(scores, targets),
                "brier": float(np.square(scores - targets).mean()) if scores.size else float("nan"),
                "soft_ece": soft_ece(scores, targets, bins=calibration_bins),
            }

    coco_results: dict[str, Any] = {}
    prediction_dir = output_dir / "predictions"
    for protocol_name, predictions in protocols.items():
        qj_only = protocol_name in {"B_qj2", "D_qj2_only"}
        if qj_only and instance_rows and instance_rows[0]["qj2_quality"] is None:
            coco_results[protocol_name] = {
                "available": False,
                "reason": "QJ-2 disabled for this ablation member",
            }
        else:
            coco_results[protocol_name] = run_coco_segm_eval(
                dataset.coco,
                predictions,
                image_ids_seen,
                prediction_dir / f"{split_name}_{protocol_name}.json",
                dataset.class_names,
            )

    low_occupancy = column("occupancy") <= 0.1
    low_occupancy_coco = _coco_with_annotation_ids(dataset, low_occupancy_annotation_ids)
    low_occupancy_d = run_coco_segm_eval(
        low_occupancy_coco, low_occupancy_d_predictions, image_ids_seen,
        prediction_dir / f"{split_name}_D_legacy_occupancy_le_0_1.json", dataset.class_names,
    )
    difficult_subset = {
        "definition": "GT mask area divided by rasterized GT-box area <= 0.1",
        "instance_count": int(low_occupancy.sum()),
        "mean_iou": _safe_mean(mask_iou[low_occupancy]),
        "D_legacy": low_occupancy_d,
    }

    qj2_enabled = bool(instance_rows and instance_rows[0]["qj2_quality"] is not None)
    primary_aliases = (
        {"B": "B_qj2", "C": "C_legacy", "D": "D_legacy"}
        if qj2_enabled
        else {"B": "B_native_iou", "C": "C_class_only", "D": "D_class_only"}
    )

    metrics = {
        "protocol": "oracle GT-box prompts; one selected SAM mask per GT instance; segmentation COCOeval",
        "split": split_name,
        "processed_images": processed_images,
        "processed_instances": len(instance_rows),
        "class_names": dataset.class_names,
        "category_ids": dataset.category_ids,
        "qj2_enabled": qj2_enabled,
        "joint_score_formula": (
            "class_probability**0.5 * qj2_quality**1.5"
            if qj2_enabled else "class_probability"
        ),
        "primary_aliases": primary_aliases,
        "coco_segm": coco_results,
        "classification": classification,
        "classification_confidence_statistics": confidence_statistics,
        "major_confusion_pair_errors": pair_errors,
        "picoplus_gate_distribution": gate_distribution,
        "mask_pixel": mask_metrics,
        "area": area_metrics,
        "difficult_occupancy_le_0_1": difficult_subset,
        "calibration_and_correlation": calibration,
    }
    atomic_json_dump(metrics, output_dir / f"{split_name}_metrics.json")
    if instance_rows:
        fieldnames = list(instance_rows[0].keys())
        with (output_dir / f"{split_name}_instances.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(instance_rows)
    return metrics
