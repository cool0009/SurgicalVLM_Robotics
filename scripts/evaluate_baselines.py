"""Phase 7 baselines: Random and Majority-class, for the dissertation's
baseline comparison table. Reuses the exact same evaluator classes as
scripts/evaluate_multi_task.py so numbers are directly comparable — only the
prediction source changes (random / majority-class vote instead of the model).

Runs entirely locally (no GPU, no pod needed) against the same test splits.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from surgical_vlm.data.vlm_dataset import VLMJSONLDataset
from surgical_vlm.data.split_manager import SplitManager
from surgical_vlm.evaluation.phase_metrics import PhaseEvaluator, MultiLabelEvaluator
from surgical_vlm.evaluation.triplet_metrics import TripletEvaluator

random.seed(42)

JSONL_DIR = "data/processed/vlm_jsonl"
IMAGE_ROOT = "data/raw"  # unused for labels-only, but required by the ctor
sm = SplitManager()

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation",
    "GallbladderRetraction",
]
TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
TRIPLET_INSTRUMENTS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator"]
TRIPLET_VERBS = ["No_Action", "Grasp", "Retract", "Dissect", "Coagulate", "Cut", "Clip", "Aspirate", "Irrigate", "Pack"]
TRIPLET_TARGETS = [
    "No_Target", "Gallbladder", "Cystic_Duct", "Cystic_Artery", "Hepatic_Duct",
    "Liver", "Peritoneum", "Fat", "Omentum", "Adhesion", "Clip", "Suture",
    "Needle", "Specimen_Bag", "Other",
]


def load_items(jsonl_name, dataset_name, split, max_samples=None):
    return list(VLMJSONLDataset(
        f"{JSONL_DIR}/{jsonl_name}", dataset_name, split, sm,
        image_root=IMAGE_ROOT, max_samples=max_samples,
    ))


def majority_label(items, field, default):
    vals = [r.get(field) for r in items if r.get(field)]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


def majority_tools(items, threshold=0.5):
    """Per-tool majority vote: predict a tool present if it's in >threshold of train frames."""
    n = len(items)
    counts = Counter()
    for r in items:
        for t in (r.get("tools") or []):
            counts[t] += 1
    return [t for t in TOOL_NAMES if counts.get(t, 0) / max(n, 1) > threshold]


def run_phase_baseline(dataset_name, jsonl_train, jsonl_test, split_name_test="test"):
    train_items = load_items(jsonl_train, dataset_name, "train")
    test_items = load_items(jsonl_test, dataset_name, "test", max_samples=500 if dataset_name == "cholec80" else None)

    maj_phase = majority_label(train_items, "phase", PHASE_NAMES[0])

    rand_eval, maj_eval = PhaseEvaluator(PHASE_NAMES), PhaseEvaluator(PHASE_NAMES)
    for r in test_items:
        gt = r.get("phase")
        if not gt:
            continue
        rand_eval.add_batch([random.choice(PHASE_NAMES)], [gt])
        maj_eval.add_batch([maj_phase], [gt])
    return rand_eval.compute(), maj_eval.compute(), maj_phase


def run_instrument_baseline(dataset_name, jsonl_train, jsonl_test):
    train_items = load_items(jsonl_train, dataset_name, "train")
    test_items = load_items(jsonl_test, dataset_name, "test", max_samples=500 if dataset_name == "cholec80" else None)

    maj_tools = majority_tools(train_items)

    rand_eval, maj_eval = MultiLabelEvaluator(TOOL_NAMES), MultiLabelEvaluator(TOOL_NAMES)
    for r in test_items:
        gt = r.get("tools")
        if gt is None:
            continue
        rand_pred = [t for t in TOOL_NAMES if random.random() < 0.5]
        rand_eval.add_batch([rand_pred], [gt])
        maj_eval.add_batch([maj_tools], [gt])
    return rand_eval.compute(), maj_eval.compute(), maj_tools


def run_triplet_baseline():
    train_items = load_items("cholect50_triplets_vlm_train.jsonl", "cholect50", "train")
    test_items = load_items("cholect50_triplets_vlm_test.jsonl", "cholect50", "test", max_samples=500)

    maj_inst = majority_label(train_items, "triplet_instrument", TRIPLET_INSTRUMENTS[0])
    maj_verb = majority_label(train_items, "triplet_verb", TRIPLET_VERBS[0])
    maj_target = majority_label(train_items, "triplet_target", TRIPLET_TARGETS[0])
    maj_triplet_str = f"{maj_inst}, {maj_verb}, {maj_target}"

    rand_eval, maj_eval = TripletEvaluator(), TripletEvaluator()
    for r in test_items:
        inst, verb, tgt = r.get("triplet_instrument"), r.get("triplet_verb"), r.get("triplet_target")
        if not (inst and verb and tgt):
            continue
        gt_str = f"{inst}, {verb}, {tgt}"
        rand_str = f"{random.choice(TRIPLET_INSTRUMENTS)}, {random.choice(TRIPLET_VERBS)}, {random.choice(TRIPLET_TARGETS)}"
        rand_eval.add_batch_strings([rand_str], [gt_str])
        maj_eval.add_batch_strings([maj_triplet_str], [gt_str])
    return rand_eval.compute(), maj_eval.compute(), maj_triplet_str


def main():
    results = {}

    print("=== Phase (Cholec80) ===")
    r, m, maj_phase = run_phase_baseline("cholec80", "cholec80_phases_vlm_train.jsonl", "cholec80_phases_vlm_test.jsonl")
    print(f"Random: acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")
    print(f"Majority ({maj_phase}): acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    results["phase_cholec80"] = {"random": r, "majority": m, "majority_class": maj_phase}

    print("\n=== Phase (HeiChole) ===")
    r, m, maj_phase = run_phase_baseline("heichole", "heichole_multitask_vlm_train.jsonl", "heichole_multitask_vlm_test.jsonl")
    print(f"Random: acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")
    print(f"Majority ({maj_phase}): acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    results["phase_heichole"] = {"random": r, "majority": m, "majority_class": maj_phase}

    print("\n=== Instruments (Cholec80) ===")
    r, m, maj_tools = run_instrument_baseline("cholec80", "cholec80_phases_vlm_train.jsonl", "cholec80_phases_vlm_test.jsonl")
    print(f"Random: macro_f1={r['macro_f1']:.4f} map@0.5={r['map_05']:.4f}")
    print(f"Majority ({maj_tools}): macro_f1={m['macro_f1']:.4f} map@0.5={m['map_05']:.4f}")
    results["instruments_cholec80"] = {"random": r, "majority": m, "majority_class": maj_tools}

    print("\n=== Triplet (CholecT50) ===")
    r, m, maj_triplet = run_triplet_baseline()
    print(f"Random: exact={r['exact_match_accuracy']:.4f} verb_f1={r['verb_f1']:.4f}")
    print(f"Majority ({maj_triplet}): exact={m['exact_match_accuracy']:.4f} verb_f1={m['verb_f1']:.4f}")
    results["triplet_cholect50"] = {"random": r, "majority": m, "majority_class": maj_triplet}

    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/baseline_results.json")


if __name__ == "__main__":
    main()
