"""
Phase recognition metrics: accuracy, weighted F1, per-phase F1.
Used by: Cholec80, HeiChole.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, hamming_loss, jaccard_score


class PhaseEvaluator:
    """Evaluator for surgical phase recognition."""

    def __init__(self, phase_names: List[str]):
        self.phase_names = phase_names
        self.num_phases = len(phase_names)
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []

    def add_batch(self, predictions: List[str], labels: List[str]) -> None:
        """Add a batch of predictions and labels (as phase name strings)."""
        name_to_idx = {name: i for i, name in enumerate(self.phase_names)}
        for pred, label in zip(predictions, labels):
            pred_idx = name_to_idx.get(pred, -1)
            label_idx = name_to_idx.get(label, -1)
            if pred_idx >= 0 and label_idx >= 0:
                self.all_preds.append(pred_idx)
                self.all_labels.append(label_idx)

    def add_batch_indices(self, predictions: List[int], labels: List[int]) -> None:
        """Add a batch using integer indices."""
        for p, l in zip(predictions, labels):
            if 0 <= p < self.num_phases and 0 <= l < self.num_phases:
                self.all_preds.append(p)
                self.all_labels.append(l)

    def compute(self) -> Dict:
        """Compute all phase metrics."""
        if not self.all_preds:
            return {"error": "No samples added"}

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        accuracy = accuracy_score(labels, preds)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        per_phase_f1 = f1_score(
            labels, preds, average=None, zero_division=0
        ).tolist()
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_phases)))

        # Transition accuracy: correct phase at phase boundary
        transition_acc = self._compute_transition_accuracy(preds, labels)

        result = {
            "accuracy": float(accuracy),
            "weighted_f1": float(weighted_f1),
            "macro_f1": float(macro_f1),
            "per_phase_f1": {name: float(f) for name, f in zip(self.phase_names, per_phase_f1)},
            "confusion_matrix": cm.tolist(),
            "transition_accuracy": float(transition_acc),
            "num_samples": len(preds),
        }
        return result

    def _compute_transition_accuracy(
        self, preds: np.ndarray, labels: np.ndarray, tolerance: int = 5
    ) -> float:
        """Compute accuracy at phase transition boundaries."""
        transitions = 0
        correct = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                transitions += 1
                # Check if prediction is correct within tolerance window
                window_start = max(0, i - tolerance)
                window_end = min(len(preds), i + tolerance + 1)
                if labels[i] in preds[window_start:window_end]:
                    correct += 1
        return correct / max(1, transitions)

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []


class MultiLabelEvaluator:
    """Evaluator for multi-label instrument/tool detection."""

    def __init__(self, tool_names: List[str]):
        self.tool_names = tool_names
        self.num_tools = len(tool_names)
        self.all_preds: List[List[int]] = []
        self.all_labels: List[List[int]] = []

    def add_batch(self, predictions: List[List[str]], labels: List[List[str]]) -> None:
        """Add a batch of predictions and labels (as lists of tool name strings)."""
        name_to_idx = {name: i for i, name in enumerate(self.tool_names)}
        for pred_tools, label_tools in zip(predictions, labels):
            pred_idx = [name_to_idx.get(t, -1) for t in pred_tools]
            label_idx = [name_to_idx.get(t, -1) for t in label_tools]
            pred_idx = [i for i in pred_idx if i >= 0]
            label_idx = [i for i in label_idx if i >= 0]
            
            # Convert to binary vectors
            pred_vec = np.zeros(self.num_tools)
            label_vec = np.zeros(self.num_tools)
            pred_vec[pred_idx] = 1
            label_vec[label_idx] = 1
            
            self.all_preds.append(pred_vec)
            self.all_labels.append(label_vec)

    def compute(self) -> Dict:
        """Compute multi-label metrics."""
        if not self.all_preds:
            return {"error": "No samples added"}

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # Per-tool metrics
        per_tool_f1 = []
        per_tool_precision = []
        per_tool_recall = []
        
        for i in range(self.num_tools):
            tool_pred = preds[:, i]
            tool_label = labels[:, i]
            
            tp = np.sum((tool_pred == 1) & (tool_label == 1))
            fp = np.sum((tool_pred == 1) & (tool_label == 0))
            fn = np.sum((tool_pred == 0) & (tool_label == 1))
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            
            per_tool_f1.append(f1)
            per_tool_precision.append(precision)
            per_tool_recall.append(recall)

        # Overall metrics (micro-averaged)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        samples_f1 = f1_score(labels, preds, average="samples", zero_division=0)
        
        # Hamming loss (fraction of incorrect labels)
        hamming = hamming_loss(labels, preds)
        
        # Subset accuracy (exact match)
        subset_acc = accuracy_score(labels, preds)

        # mAP@0.5 (approximate using per-tool AP)
        # For multi-label, we compute AP per tool and average
        map_05 = np.mean(per_tool_f1)  # Simplified

        result = {
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "samples_f1": float(samples_f1),
            "subset_accuracy": float(subset_acc),
            "hamming_loss": float(hamming),
            "map_05": float(map_05),
            "per_tool_f1": {name: float(f) for name, f in zip(self.tool_names, per_tool_f1)},
            "per_tool_precision": {name: float(p) for name, p in zip(self.tool_names, per_tool_precision)},
            "per_tool_recall": {name: float(r) for name, r in zip(self.tool_names, per_tool_recall)},
            "num_samples": len(preds),
        }
        return result

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []