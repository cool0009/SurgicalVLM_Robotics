"""Phase 7 baseline: Zero-shot Qwen2.5-VL-7B-Instruct (no LoRA, no fine-tuning),
same prompts/parsing as the trained-model demo path. Generative only — there is
no trained classification head to fall back on, so phase/instruments both come
from one JSON-prompted generation per frame (matches the demo's approach), and
triplet uses its own JSON-prompted generation. Language reuses the existing
evaluate_language_generation() unchanged (already generative).

Run on the pod: python scripts/evaluate_zeroshot.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

from tqdm import tqdm
from PIL import Image

import scripts.evaluate_multi_task as emt
from scripts.evaluate_multi_task import load_model, evaluate_language_generation
from surgical_vlm.models.surgical_vlm import SURGICAL_JSON_PROMPT
from surgical_vlm.data.vlm_dataset import VLMJSONLDataset
from surgical_vlm.data.split_manager import SplitManager
from surgical_vlm.evaluation.phase_metrics import PhaseEvaluator, MultiLabelEvaluator
from surgical_vlm.evaluation.triplet_metrics import TripletEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

emt._JSONL_DIR = "/workspace/eval_data/processed/vlm_jsonl"
IMAGE_ROOT = "/workspace/eval_data"

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

TRIPLET_PROMPT = """Analyze this surgical frame and output a JSON object with exactly these 3 keys:
{
  "instrument": "one of: Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator",
  "verb": "one of: No_Action, Grasp, Retract, Dissect, Coagulate, Cut, Clip, Aspirate, Irrigate, Pack",
  "target": "one of: No_Target, Gallbladder, Cystic_Duct, Cystic_Artery, Hepatic_Duct, Liver, Peritoneum, Fat, Omentum, Adhesion, Clip, Suture, Needle, Specimen_Bag, Other"
}
Output ONLY valid JSON, no extra text."""


def eval_phase_and_instruments(vlm, dataset_name, jsonl_name, max_samples):
    sm = SplitManager()
    ds = VLMJSONLDataset(
        f"{emt._JSONL_DIR}/{jsonl_name}", dataset_name, "test", sm,
        image_root=IMAGE_ROOT, max_samples=max_samples,
    )
    phase_eval = PhaseEvaluator(PHASE_NAMES)
    tools_eval = MultiLabelEvaluator(TOOL_NAMES)
    for item in tqdm(ds, desc=f"Zero-shot phase+tools {dataset_name}"):
        gt_phase, gt_tools = item.get("phase"), item.get("tools")
        if gt_phase is None:
            continue
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            continue
        pred = vlm.generate_single(img, prompt=SURGICAL_JSON_PROMPT)
        phase_eval.add_batch([pred.get("phase", "Unknown")], [gt_phase])
        tools_eval.add_batch([pred.get("tools", [])], [gt_tools or []])
    return phase_eval.compute(), tools_eval.compute()


def eval_triplet(vlm, max_samples=500):
    sm = SplitManager()
    ds = VLMJSONLDataset(
        f"{emt._JSONL_DIR}/cholect50_triplets_vlm_test.jsonl", "cholect50", "test", sm,
        image_root=IMAGE_ROOT, max_samples=max_samples,
    )
    ev = TripletEvaluator()
    for item in tqdm(ds, desc="Zero-shot triplet cholect50"):
        gt_inst, gt_verb, gt_tgt = item.get("triplet_instrument"), item.get("triplet_verb"), item.get("triplet_target")
        if not (gt_inst and gt_verb and gt_tgt):
            continue
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            continue
        raw = vlm.vlm.generate(image=img, prompt=TRIPLET_PROMPT, max_new_tokens=128, temperature=0.1)
        parsed = vlm._extract_json(raw) or {}
        pred_str = f"{parsed.get('instrument','')}, {parsed.get('verb','')}, {parsed.get('target','')}"
        gt_str = f"{gt_inst}, {gt_verb}, {gt_tgt}"
        ev.add_batch_strings([pred_str], [gt_str])
    return ev.compute()


def main():
    logger.info("Loading BASE Qwen2.5-VL-7B-Instruct (no LoRA checkpoint)...")
    vlm = load_model("configs/training/eval_local_transfer_config.yaml", checkpoint="", device="cuda")

    results = {}

    logger.info("=== Zero-shot phase+instruments (Cholec80) ===")
    phase_r, tools_r = eval_phase_and_instruments(vlm, "cholec80", "cholec80_phases_vlm_test.jsonl", 500)
    logger.info(f"Phase: acc={phase_r.get('accuracy')} macro_f1={phase_r.get('macro_f1')}")
    logger.info(f"Instruments: macro_f1={tools_r.get('macro_f1')} map@0.5={tools_r.get('map_05')}")
    results["phase_cholec80"] = phase_r
    results["instruments_cholec80"] = tools_r

    logger.info("=== Zero-shot phase+instruments (HeiChole) ===")
    phase_r2, tools_r2 = eval_phase_and_instruments(vlm, "heichole", "heichole_multitask_vlm_test.jsonl", None)
    logger.info(f"Phase: acc={phase_r2.get('accuracy')}")
    results["phase_heichole"] = phase_r2
    results["instruments_heichole"] = tools_r2

    logger.info("=== Zero-shot triplet (CholecT50) ===")
    triplet_r = eval_triplet(vlm, 500)
    logger.info(f"Triplet: exact={triplet_r.get('exact_match_accuracy')} verb_f1={triplet_r.get('verb_f1')}")
    results["triplet_cholect50"] = triplet_r

    logger.info("=== Zero-shot language (Surg-396K) ===")
    lang_r = evaluate_language_generation(vlm, "surg396k", "test", max_samples=500, image_root=IMAGE_ROOT)
    logger.info(f"Language: {lang_r}")
    results["language_surg396k"] = lang_r

    with open("results/zeroshot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results/zeroshot_results.json")


if __name__ == "__main__":
    main()
