"""
Triplet metrics for CholecT50: exact-match triplet accuracy, component-level F1.
"""

from typing import Dict, List, Tuple


class TripletEvaluator:
    """Evaluator for <instrument, verb, target> triplet prediction."""

    def __init__(self):
        self.all_preds: List[Tuple[str, str, str]] = []
        self.all_labels: List[Tuple[str, str, str]] = []

    def add_batch(
        self,
        predictions: List[Tuple[str, str, str]],
        labels: List[Tuple[str, str, str]],
    ) -> None:
        for p, l in zip(predictions, labels):
            self.all_preds.append(p)
            self.all_labels.append(l)

    def add_batch_strings(self, predictions: List[str], labels: List[str]) -> None:
        """Parse triplets from strings like 'Hook, Dissect, CysticDuct'."""
        for pred_str, label_str in zip(predictions, labels):
            pred = self._parse_triplet(pred_str)
            label = self._parse_triplet(label_str)
            if pred and label:
                self.all_preds.append(pred)
                self.all_labels.append(label)

    @staticmethod
    def _parse_triplet(s: str) -> Tuple[str, str, str]:
        """Parse a triplet from a comma-separated string."""
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            return (parts[0], parts[1], parts[2])
        elif len(parts) == 2:
            return (parts[0], parts[1], "")
        elif len(parts) == 1:
            return (parts[0], "", "")
        return ("", "", "")

    def compute(self) -> Dict:
        if not self.all_preds:
            return {"error": "No samples added"}

        n = len(self.all_preds)

        # Exact match: all three components correct
        exact_matches = sum(
            1 for p, l in zip(self.all_preds, self.all_labels) if p == l
        )
        exact_accuracy = exact_matches / n

        # Component-level accuracy and F1
        instrument_correct = sum(
            1 for p, l in zip(self.all_preds, self.all_labels) if p[0] == l[0]
        )
        verb_correct = sum(
            1 for p, l in zip(self.all_preds, self.all_labels) if p[1] == l[1]
        )
        target_correct = sum(
            1 for p, l in zip(self.all_preds, self.all_labels) if p[2] == l[2]
        )

        # Component F1 using set-based evaluation
        inst_f1 = self._component_f1(
            [p[0] for p in self.all_preds], [l[0] for l in self.all_labels]
        )
        verb_f1 = self._component_f1(
            [p[1] for p in self.all_preds], [l[1] for l in self.all_labels]
        )
        target_f1 = self._component_f1(
            [p[2] for p in self.all_preds], [l[2] for l in self.all_labels]
        )

        return {
            "exact_match_accuracy": float(exact_accuracy),
            "instrument_accuracy": float(instrument_correct / n),
            "verb_accuracy": float(verb_correct / n),
            "target_accuracy": float(target_correct / n),
            "instrument_f1": float(inst_f1),
            "verb_f1": float(verb_f1),
            "target_f1": float(target_f1),
            "num_samples": n,
        }

    @staticmethod
    def _component_f1(preds: List[str], labels: List[str]) -> float:
        """Compute F1 for a single component."""
        from sklearn.metrics import f1_score

        # Filter out empty strings
        valid_pairs = [(p, l) for p, l in zip(preds, labels) if p or l]
        if not valid_pairs:
            return 0.0
        vp, vl = zip(*valid_pairs)
        return f1_score(list(vl), list(vp), average="weighted", zero_division=0)

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []