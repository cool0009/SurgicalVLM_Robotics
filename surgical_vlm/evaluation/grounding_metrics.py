"""
Grounding metrics: IoU, Dice, mAP@0.5, mAP@0.75.
Parses [x1, y1, x2, y2] from generated text.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


class GroundingEvaluator:
    """Evaluator for visual grounding / bounding box prediction."""

    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.all_preds: List[List[Tuple[int, int, int, int]]] = []
        self.all_labels: List[List[Tuple[int, int, int, int]]] = []
        self.all_pred_labels: List[List[str]] = []
        self.all_label_classes: List[List[str]] = []

    def add_batch(
        self,
        predictions: List[List[Dict]],
        labels: List[List[Dict]],
    ) -> None:
        """
        Add a batch.
        Each prediction/label is a list of dicts: {"label": str, "bbox": [x1,y1,x2,y2]}.
        """
        for pred_list, label_list in zip(predictions, labels):
            pred_bboxes = []
            pred_labels = []
            for p in pred_list:
                bbox = tuple(p["bbox"]) if "bbox" in p else (0, 0, 0, 0)
                pred_bboxes.append(bbox)
                pred_labels.append(p.get("label", ""))

            label_bboxes = []
            label_classes = []
            for l in label_list:
                bbox = tuple(l["bbox"]) if "bbox" in l else (0, 0, 0, 0)
                label_bboxes.append(bbox)
                label_classes.append(l.get("label", ""))

            self.all_preds.append(pred_bboxes)
            self.all_labels.append(label_bboxes)
            self.all_pred_labels.append(pred_labels)
            self.all_label_classes.append(label_classes)

    def add_batch_strings(
        self, predictions: List[str], labels: List[List[Dict]]
    ) -> None:
        """Parse bboxes from generated text strings."""
        for pred_str, label_list in zip(predictions, labels):
            pred_bboxes = self._parse_bboxes_from_text(pred_str)
            self.all_preds.append([b for b, _ in pred_bboxes])
            self.all_pred_labels.append([l for _, l in pred_bboxes])

            label_bboxes = []
            label_classes = []
            for l in label_list:
                bbox = tuple(l["bbox"]) if "bbox" in l else (0, 0, 0, 0)
                label_bboxes.append(bbox)
                label_classes.append(l.get("label", ""))
            self.all_labels.append(label_bboxes)
            self.all_label_classes.append(label_classes)

    @staticmethod
    def _parse_bboxes_from_text(text: str) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Parse [x1, y1, x2, y2] from generated text."""
        pattern = r"([\w\s]+?)\s*at\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(pattern, text)
        results = []
        for m in matches:
            label = m[0].strip()
            bbox = (int(m[1]), int(m[2]), int(m[3]), int(m[4]))
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                results.append((bbox, label))
        return results

    @staticmethod
    def compute_iou(
        box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def compute_dice(
        box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute Dice coefficient between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        total = area1 + area2
        if total <= 0:
            return 0.0
        return 2 * intersection / total

    def compute(self) -> Dict:
        """Compute all grounding metrics."""
        if not self.all_labels:
            return {"error": "No samples added"}

        all_ious = []
        all_dices = []
        ap_per_threshold = {t: [] for t in self.iou_thresholds}

        for pred_bboxes, label_bboxes in zip(self.all_preds, self.all_labels):
            if not label_bboxes:
                continue

            # Match each ground truth to best predicted box
            matched_ious = []
            for lb in label_bboxes:
                if not pred_bboxes:
                    matched_ious.append(0.0)
                    continue
                best_iou = max(self.compute_iou(lb, pb) for pb in pred_bboxes)
                matched_ious.append(best_iou)
                all_dices.append(self.compute_dice(lb, pred_bboxes[np.argmax([self.compute_iou(lb, pb) for pb in pred_bboxes])]))

            all_ious.extend(matched_ious)

            for threshold in self.iou_thresholds:
                hits = sum(1 for i in matched_ious if i >= threshold)
                precision = hits / max(1, len(pred_bboxes))
                recall = hits / max(1, len(label_bboxes))
                if precision + recall > 0:
                    ap_per_threshold[threshold].append(
                        2 * precision * recall / (precision + recall)
                    )
                else:
                    ap_per_threshold[threshold].append(0.0)

        result = {
            "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "median_iou": float(np.median(all_ious)) if all_ious else 0.0,
            "mean_dice": float(np.mean(all_dices)) if all_dices else 0.0,
            "num_samples": len(self.all_labels),
            "num_gt_boxes": sum(len(l) for l in self.all_labels),
            "num_pred_boxes": sum(len(p) for p in self.all_preds),
        }

        for threshold in self.iou_thresholds:
            key = f"mAP@{threshold}"
            result[key] = float(np.mean(ap_per_threshold[threshold])) if ap_per_threshold[threshold] else 0.0

        return result

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []
        self.all_pred_labels = []
        self.all_label_classes = []