"""One shared detection evaluator for EVERY model (YOLO, Faster R-CNN, ...).

Never use a framework's built-in mAP when comparing models: each computes it
slightly differently. Convert each model's raw predictions to the format below
and score them here, with the same thresholds.

Box format: [x_min, y_min, x_max, y_max] in pixels.

ground_truths: list with one dict per image
    {"boxes": [[x1,y1,x2,y2], ...], "labels": [class_id, ...]}
predictions: list with one dict per image, SAME ORDER as ground_truths
    {"boxes": [[...], ...], "scores": [0.9, ...], "labels": [class_id, ...]}
class_names: list where index == class id (the frozen class order).
"""
from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NAN = float("nan")


def _to_coco(ground_truths, predictions, class_names):
    images, gt_anns, dt_anns = [], [], []
    ann_id = 1
    for img_id, (gt, pr) in enumerate(zip(ground_truths, predictions)):
        images.append({"id": img_id, "width": 0, "height": 0})
        for box, label in zip(gt["boxes"], gt["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            w, h = x2 - x1, y2 - y1
            gt_anns.append({
                "id": ann_id, "image_id": img_id, "category_id": int(label),
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
        for box, score, label in zip(pr["boxes"], pr["scores"], pr["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            dt_anns.append({
                "image_id": img_id, "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
            })
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    return {"images": images, "annotations": gt_anns, "categories": categories}, dt_anns


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _precision_recall_at_threshold(
    ground_truths, predictions, num_classes, conf_thr, iou_thr
) -> Dict[str, np.ndarray]:
    """Greedy matching (highest score first), per class. Returns TP/FP/FN arrays."""
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for gt, pr in zip(ground_truths, predictions):
        g_boxes = np.asarray(gt["boxes"], dtype=float).reshape(-1, 4)
        g_labels = np.asarray(gt["labels"], dtype=int)
        p_boxes = np.asarray(pr["boxes"], dtype=float).reshape(-1, 4)
        p_scores = np.asarray(pr["scores"], dtype=float)
        p_labels = np.asarray(pr["labels"], dtype=int)
        keep = p_scores >= conf_thr
        p_boxes, p_scores, p_labels = p_boxes[keep], p_scores[keep], p_labels[keep]
        for c in range(num_classes):
            gb = g_boxes[g_labels == c]
            pm = p_labels == c
            pb, ps = p_boxes[pm], p_scores[pm]
            order = np.argsort(-ps)
            pb = pb[order]
            ious = _iou_matrix(pb, gb)
            matched = set()
            for i in range(len(pb)):
                best_j, best_iou = -1, iou_thr
                for j in range(len(gb)):
                    if j in matched:
                        continue
                    if ious[i, j] >= best_iou:
                        best_iou, best_j = ious[i, j], j
                if best_j >= 0:
                    matched.add(best_j)
                    tp[c] += 1
                else:
                    fp[c] += 1
            fn[c] += len(gb) - len(matched)
    return {"tp": tp, "fp": fp, "fn": fn}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def evaluate_detection(
    ground_truths: Sequence[dict],
    predictions: Sequence[dict],
    class_names: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Return a flat dict of metrics ready for mlflow (see log_metrics_safe).

    Keys: map_50_95, map_50, map_75, map_small, map_medium, map_large,
          ap50_<class>, precision, recall, f1 (at conf_threshold, iou_threshold),
          precision_<class>, recall_<class>, num_gt_boxes, num_pred_boxes.
    Values that cannot be computed (e.g. no small objects exist) are NaN.
    """
    if len(ground_truths) != len(predictions):
        raise ValueError("ground_truths and predictions must have the same length")
    num_classes = len(class_names)
    n_gt = sum(len(g["boxes"]) for g in ground_truths)
    n_pred = sum(len(p["boxes"]) for p in predictions)

    result: Dict[str, float] = {
        "map_50_95": NAN, "map_50": NAN, "map_75": NAN,
        "map_small": NAN, "map_medium": NAN, "map_large": NAN,
        "num_gt_boxes": float(n_gt), "num_pred_boxes": float(n_pred),
    }
    for name in class_names:
        result[f"ap50_{name}"] = NAN

    if n_gt == 0:
        raise ValueError("No ground-truth boxes found: cannot evaluate detection.")

    if n_pred == 0:
        # Nothing predicted: every AP is 0 (pycocotools cannot load an empty result list).
        for k in ("map_50_95", "map_50", "map_75"):
            result[k] = 0.0
        for name in class_names:
            result[f"ap50_{name}"] = 0.0
    else:
        gt_dict, dt_list = _to_coco(ground_truths, predictions, class_names)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(dt_list)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats

        def clean(x):
            return NAN if x < 0 else float(x)

        result.update({
            "map_50_95": clean(s[0]), "map_50": clean(s[1]), "map_75": clean(s[2]),
            "map_small": clean(s[3]), "map_medium": clean(s[4]), "map_large": clean(s[5]),
        })
        # Per-class AP@0.5: precision array shape [T, R, K, A, M]
        precision = ev.eval["precision"]
        iou_idx = int(np.where(np.isclose(ev.params.iouThrs, 0.5))[0][0])
        for k, name in enumerate(class_names):
            p = precision[iou_idx, :, k, 0, -1]  # area=all, maxDets=100
            p = p[p > -1]
            result[f"ap50_{name}"] = float(p.mean()) if p.size else NAN

    counts = _precision_recall_at_threshold(
        ground_truths, predictions, num_classes, conf_threshold, iou_threshold
    )
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    result["precision"] = _safe_div(tp.sum(), tp.sum() + fp.sum())
    result["recall"] = _safe_div(tp.sum(), tp.sum() + fn.sum())
    result["f1"] = _safe_div(2 * result["precision"] * result["recall"],
                             result["precision"] + result["recall"])
    for c, name in enumerate(class_names):
        pc = _safe_div(tp[c], tp[c] + fp[c])
        rc = _safe_div(tp[c], tp[c] + fn[c])
        result[f"precision_{name}"] = pc
        result[f"recall_{name}"] = rc
    return result
