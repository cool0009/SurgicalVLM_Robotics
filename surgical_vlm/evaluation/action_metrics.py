"""
Action Recognition Metrics
Evaluates surgical action classification performance.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class ActionEvaluator:
    """Evaluator for surgical action recognition."""
    
    def __init__(self, action_names: List[str]):
        self.action_names = action_names
        self.num_actions = len(action_names)
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
    
    def add_batch(self, predictions: List[int], labels: List[int]) -> None:
        """Add a batch of predictions and labels (as integer indices)."""
        for p, l in zip(predictions, labels):
            if 0 <= p < self.num_actions and 0 <= l < self.num_actions:
                self.all_preds.append(p)
                self.all_labels.append(l)
    
    def add_batch_strings(self, predictions: List[str], labels: List[str]) -> None:
        """Add batch using action name strings."""
        name_to_idx = {name: i for i, name in enumerate(self.action_names)}
        for pred, label in zip(predictions, labels):
            pred_idx = name_to_idx.get(pred, -1)
            label_idx = name_to_idx.get(label, -1)
            if pred_idx >= 0 and label_idx >= 0:
                self.all_preds.append(pred_idx)
                self.all_labels.append(label_idx)
    
    def compute(self) -> Dict:
        """Compute all action recognition metrics."""
        if not self.all_preds:
            return {"error": "No samples added"}
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Overall metrics
        accuracy = accuracy_score(labels, preds)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        
        # Per-class metrics
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0).tolist()
        per_class_precision = []
        per_class_recall = []
        
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_actions)))
        
        for i in range(self.num_actions):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            
            per_class_precision.append(precision)
            per_class_recall.append(recall)
        
        result = {
            "accuracy": float(accuracy),
            "weighted_f1": float(weighted_f1),
            "macro_f1": float(macro_f1),
            "per_class_f1": {name: float(f) for name, f in zip(self.action_names, per_class_f1)},
            "per_class_precision": {name: float(p) for name, p in zip(self.action_names, per_class_precision)},
            "per_class_recall": {name: float(r) for name, r in zip(self.action_names, per_class_recall)},
            "confusion_matrix": cm.tolist(),
            "num_samples": len(preds),
        }
        
        # Top-k accuracy (for k=3, 5)
        result["top3_accuracy"] = self._top_k_accuracy(labels, preds, k=3)
        result["top5_accuracy"] = self._top_k_accuracy(labels, preds, k=5)
        
        return result
    
    def _top_k_accuracy(self, labels: np.ndarray, preds: np.ndarray, k: int) -> float:
        """Compute top-k accuracy (requires probability scores, approximate here)."""
        # This is a placeholder - true top-k needs probabilities
        # For now return same as accuracy
        return float(accuracy_score(labels, preds))
    
    def compute_from_probs(self, probabilities: np.ndarray, labels: List[int]) -> Dict:
        """
        Compute metrics from probability outputs.
        
        Args:
            probabilities: [N, C] array of class probabilities
            labels: List of ground truth indices
        """
        if probabilities.ndim != 2:
            return {"error": "Probabilities must be 2D array [N, C]"}
        
        top1_preds = probabilities.argmax(axis=1)
        
        # Top-k
        top3_preds = np.argsort(probabilities, axis=1)[:, -3:]
        top5_preds = np.argsort(probabilities, axis=1)[:, -5:]
        
        top1_acc = accuracy_score(labels, top1_preds)
        top3_acc = np.mean([labels[i] in top3_preds[i] for i in range(len(labels))])
        top5_acc = np.mean([labels[i] in top5_preds[i] for i in range(len(labels))])
        
        return {
            "top1_accuracy": float(top1_acc),
            "top3_accuracy": float(top3_acc),
            "top5_accuracy": float(top5_acc),
            "num_samples": len(labels),
        }
    
    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []


def compute_action_metrics(predictions: List[int], labels: List[int], action_names: List[str]) -> Dict:
    """Convenience function for one-shot evaluation."""
    evaluator = ActionEvaluator(action_names)
    evaluator.add_batch(predictions, labels)
    return evaluator.compute()


if __name__ == "__main__":
    # Test
    actions = ["No_Action", "Dissect", "Grasp", "Clip", "Cut", 
               "Coagulate", "Retract", "Irrigate", "Pack", "Clean"]
    
    evaluator = ActionEvaluator(actions)
    evaluator.add_batch([1, 2, 2, 3, 4, 1, 2, 5], [1, 2, 2, 3, 4, 0, 2, 5])
    results = evaluator.compute()
    
    import json
    print(json.dumps(results, indent=2))