"""
Language metrics: BLEU-1/4, ROUGE-L.
Used by: Surg-396K scene description outputs.

Preference order for metric backends:
  1. HuggingFace `evaluate` (sacrebleu / rouge) if the package is installed
     and the metric can be loaded (requires hub access on first use).
  2. Self-contained pure-Python implementations (sacrebleu-style corpus
     BLEU with exp smoothing, and ROUGE-1/2/L F-measures) so that metrics
     still compute on machines without HF hub access (e.g. offline pods).

Both backends expose the same interface used by LanguageEvaluator.compute():
  - bleu:  .compute(predictions=[...], references=[[ref,...], ...]) -> {"score": float}
  - rouge: .compute(predictions=[...], references=[...]) -> {"rouge1","rouge2","rougeL",...}
"""

import re
from typing import Dict, List, Optional

try:
    import evaluate
except Exception:  # pragma: no cover - package not installed
    evaluate = None


# --------------------------------------------------------------------------
# Tokenization (sacrebleu 13a-style approximation)
# --------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase and split into words/punctuation tokens (13a-style)."""
    return re.findall(r"\w+|[^\s\w]", text.lower())


# --------------------------------------------------------------------------
# BLEU fallback (corpus-level, sacrebleu-compatible)
# --------------------------------------------------------------------------

class _BleuFallback:
    """Corpus-level BLEU with exp smoothing, max n-gram order 4."""

    def __init__(self, max_ngram_order: int = 4):
        self.max_ngram_order = max_ngram_order

    @staticmethod
    def _modified_precision(pred_tokens: List[str], ref_tokens_list: List[List[str]],
                            n: int) -> tuple:
        """Return (clipped_count, total_count) for n-grams of order n."""
        if len(pred_tokens) < n:
            return 0, 0

        pred_ngrams = [tuple(pred_tokens[i:i + n])
                       for i in range(len(pred_tokens) - n + 1)]

        # Max reference counts per n-gram.
        max_ref_counts: Dict[tuple, int] = {}
        for ref_tokens in ref_tokens_list:
            ref_ngrams = [tuple(ref_tokens[i:i + n])
                          for i in range(len(ref_tokens) - n + 1)]
            ref_counts: Dict[tuple, int] = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1
            for ng, cnt in ref_counts.items():
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), cnt)

        clipped = sum(min(pred_ngrams.count(ng), max_ref_counts.get(ng, 0))
                      for ng in set(pred_ngrams))
        return clipped, len(pred_ngrams)

    def compute(self, predictions: List[str],
                references: List[List[str]]) -> Dict:
        refs_flat = [[_tokenize(r) for r in refs] for refs in references]
        preds = [_tokenize(p) for p in predictions]

        total = [0] * (self.max_ngram_order + 1)
        clip = [0] * (self.max_ngram_order + 1)

        for pred_tokens, ref_list in zip(preds, refs_flat):
            for n in range(1, self.max_ngram_order + 1):
                c, t = self._modified_precision(pred_tokens, ref_list, n)
                clip[n] += c
                total[n] += t

        precisions = []
        for n in range(1, self.max_ngram_order + 1):
            if total[n] == 0:
                precisions.append(0.0)
            else:
                precisions.append(clip[n] / total[n])

        # Exp smoothing (as in mteval-v13a, sacrebleu default smooth_method="exp").
        smoothed = [max(p, 1.0 / (2.0 ** n)) for n, p in enumerate(precisions, 1)]

        # Brevity penalty.
        pred_len = sum(len(p) for p in preds)
        if preds:
            ref_lens = [len(r) for ref_list in refs_flat for r in ref_list]
            min_ref_len = min(ref_lens) if ref_lens else pred_len
        else:
            min_ref_len = 0

        if pred_len == 0:
            bp = 0.0
        elif pred_len > min_ref_len:
            bp = 1.0
        else:
            bp = 2.718281828459045 ** (1.0 - min_ref_len / pred_len)

        log_sum = sum(0.25 * __import__("math").log(p) if p > 0 else float("-inf")
                      for p in smoothed)
        if log_sum == float("-inf"):
            score = 0.0
        else:
            score = bp * 2.718281828459045 ** log_sum

        return {
            "score": round(score, 2),
            "precisions": [round(p, 6) for p in precisions],
            "bp": round(bp, 6),
        }


# --------------------------------------------------------------------------
# ROUGE fallback (ROUGE-1 / ROUGE-2 / ROUGE-L F-measures)
# --------------------------------------------------------------------------

def _fmeasure(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)


def _rouge_n_f1(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_ngrams = _ngrams(pred_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    pred_counts: Dict[tuple, int] = {}
    for ng in pred_ngrams:
        pred_counts[ng] = pred_counts.get(ng, 0) + 1
    ref_counts: Dict[tuple, int] = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1
    overlap = sum(min(pred_counts.get(ng, 0), ref_counts.get(ng, 0))
                  for ng in set(pred_counts) & set(ref_counts))
    p = overlap / len(pred_ngrams)
    r = overlap / len(ref_ngrams)
    return _fmeasure(p, r)


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_len(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    return _fmeasure(p, r)


class _RougeFallback:
    """Sentence-level ROUGE-1/2/L averaged over the corpus (F-measures)."""

    def compute(self, predictions: List[str],
                references: List[str]) -> Dict:
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        n = len(predictions)
        for pred, ref in zip(predictions, references):
            pt = _tokenize(pred)
            rt = _tokenize(ref)
            rouge1 += _rouge_n_f1(pt, rt, 1)
            rouge2 += _rouge_n_f1(pt, rt, 2)
            rougeL += _rouge_l_f1(pt, rt)
        if n == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        return {
            "rouge1": round(rouge1 / n, 4),
            "rouge2": round(rouge2 / n, 4),
            "rougeL": round(rougeL / n, 4),
            "rougeLsum": round(rougeL / n, 4),
        }


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------

def _load_bleu():
    if evaluate is not None:
        try:
            return evaluate.load("sacrebleu")
        except Exception:
            pass
    return _BleuFallback()


def _load_rouge():
    if evaluate is not None:
        try:
            return evaluate.load("rouge")
        except Exception:
            pass
    return _RougeFallback()


class LanguageEvaluator:
    """Evaluator for natural language generation quality."""

    def __init__(self):
        self.all_preds: List[str] = []
        self.all_labels: List[str] = []
        self.bleu = _load_bleu()
        self.rouge = _load_rouge()

    def add_batch(self, predictions: List[str], labels: List[str]) -> None:
        self.all_preds.extend(predictions)
        self.all_labels.extend(labels)

    def compute(self) -> Dict:
        if not self.all_preds:
            return {"error": "No samples added"}

        result = {"num_samples": len(self.all_preds)}

        try:
            bleu_result = self.bleu.compute(
                predictions=self.all_preds,
                references=[[l] for l in self.all_labels],
            )
            if isinstance(bleu_result, dict) and "score" in bleu_result:
                result["bleu"] = bleu_result.get("score", 0.0)
            else:
                result["bleu"] = bleu_result
        except Exception as e:
            result["bleu_error"] = str(e)

        try:
            rouge_result = self.rouge.compute(
                predictions=self.all_preds,
                references=self.all_labels,
            )
            if isinstance(rouge_result, dict):
                result["rouge_l"] = rouge_result.get("rougeL", 0.0)
                result["rouge_1"] = rouge_result.get("rouge1", 0.0)
                result["rouge_2"] = rouge_result.get("rouge2", 0.0)
            else:
                result["rouge_l"] = rouge_result
        except Exception as e:
            result["rouge_error"] = str(e)

        return result

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []
