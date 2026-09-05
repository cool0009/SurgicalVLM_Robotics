"""Heads-only fix-forward fine-tune.

Root cause (verified 2026-08-19): VLMJSONLDataset.__getitem__ never parsed the
gpt-turn JSON answer into phase/tools/action/triplet_* fields, so
MultiTaskSurgicalTrainer._prepare_targets never found real labels and the
output_adapter classification heads (phase/instrument/action/triplet) trained
on zero gradient (only AdamW weight decay moved them) for the whole original
5000-step run. That's now fixed in vlm_dataset.py. The LoRA-adapted language
model itself trained fine (language eval was never broken) so there is no
need to retrain the 7B backbone — only the small classification heads need
fitting, using the SAME frozen-backbone feature extraction path
(`_get_training_matching_features`) that evaluate_multi_task.py uses at
inference, so train/eval are guaranteed consistent.
"""
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace")

import torch
import torch.nn.functional as F
from PIL import Image

import scripts.evaluate_multi_task as emt
from scripts.evaluate_multi_task import load_model
from surgical_vlm.data.vlm_dataset import VLMJSONLDataset
from surgical_vlm.data.split_manager import SplitManager
from surgical_vlm.training.label_mapping import get_label_mapper
from surgical_vlm.training.output_adapter import create_output_adapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

emt._JSONL_DIR = "/workspace/eval_data/processed/vlm_jsonl"
IMAGE_ROOT = "/workspace/eval_data"
DEVICE = "cuda"
# batch_size=1 is deliberate: _get_training_matching_features mean-pools over
# the full padded sequence without masking out padding, so a batch >1 dilutes
# shorter sequences' features by however much padding the batch's longest
# sequence forces on them — but predict_phase() at inference always calls
# this with a single image (no padding at all). Training with batch_size=1
# eliminates that train/inference mismatch entirely.
BATCH_SIZE = 1
# Gradient accumulation (over ACCUM_STEPS single-image forward passes) instead
# of a real batch>1: keeps the forward pass identical to inference (no padding)
# while still giving AdamW a stabler, lower-variance gradient than an update
# from every single sample — batch_size=1 + lr=5e-4 + AdamW is a known-unstable
# combo that can collapse a small linear head to a constant prediction.
ACCUM_STEPS = 8

PHASE_PROMPT = "What is the current surgical phase and what instruments are visible?"
MULTITASK_PROMPT = "Analyze this surgical frame and output the phase, instruments, action, and skill assessment."
TRIPLET_PROMPT = "What is the instrument, action, and target in this frame?"


def load_image(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def make_batches(records, batch_size, shuffle=True):
    idx = list(range(len(records)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [records[j] for j in idx[i : i + batch_size]]


def compute_tool_pos_weight(records, label_mapper, num_tools=7, cap=20.0):
    """neg/pos ratio per tool, capped to avoid unstable gradients for very rare
    tools (Bipolar 4.9%, Clipper 3.4%, Scissors 1.9%, Irrigator 6.2% of rows).
    Common tools (Grasper 65%, Hook 56%) naturally get weight < 1."""
    counts = torch.zeros(num_tools)
    for rec in records:
        for iid in label_mapper.text_to_instrument_ids(", ".join(rec.get("tools", []) or [])):
            if iid < num_tools:
                counts[iid] += 1
    total = float(len(records))
    pos = counts.clamp(min=1.0)
    weight = ((total - counts) / pos).clamp(min=1.0, max=cap)
    logger.info(f"tool pos_weight (counts={counts.tolist()}): {weight.tolist()}")
    return weight.to(DEVICE)


def compute_phase_class_weight(records, label_mapper, num_phases=8, cap=4.0):
    """Inverse-frequency weight per phase class (median-normalised, capped) for
    CrossEntropyLoss's `weight=` arg. Targets ClippingCutting (8.3% of rows,
    the one class stuck at 0 F1 in v5 -- see EVALUATION_RESULTS_REPORT.md
    Sec 2.1) and the three even-rarer classes (Preparation, GallbladderPackaging,
    GallbladderRetraction, each ~4.2-4.6%) without the aggressive 20x cap used
    for instruments -- CE weight scales the per-sample gradient directly (not
    just a pos/neg ratio like BCE), and batch_size=1 + gradient accumulation is
    already a known-unstable combo (see BATCH_SIZE comment above), so a modest
    cap avoids blowing up the loss on the rare classes and destabilising the
    already-working majority classes (CalotTriangleDissection, GallbladderDissection).
    """
    counts = torch.zeros(num_phases)
    for rec in records:
        pid = label_mapper.text_to_phase_id(rec.get("phase", ""))
        if pid < num_phases:
            counts[pid] += 1
    nonzero = counts[counts > 0]
    median = nonzero.median().item() if len(nonzero) else 1.0
    weight = (median / counts.clamp(min=1.0)).clamp(min=1.0, max=cap)
    logger.info(f"phase class_weight (counts={counts.tolist()}): {weight.tolist()}")
    return weight.to(DEVICE)


def main():
    label_mapper = get_label_mapper()

    logger.info("Loading model + checkpoint (frozen backbone)…")
    vlm = load_model(
        "configs/training/eval_local_transfer_config.yaml",
        "checkpoints/multitask/final",
        DEVICE,
    )
    for p in vlm.vlm.model.parameters():
        p.requires_grad = False
    vlm.vlm.model.eval()

    output_adapter_config = {
        "hidden_dim": 3584, "num_phases": 8, "num_instruments": 7, "num_actions": 10,
        "triplet_instruments": 6, "triplet_verbs": 10, "triplet_targets": 15,
        "vocab_size": 152064, "use_temporal": False,
    }
    adapter = create_output_adapter(output_adapter_config).to(DEVICE).to(torch.bfloat16)
    adapter.train()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.01)

    sm = SplitManager()

    # No max_samples cap this run: use every train-split row whose JSONL
    # entry exists (VLMJSONLDataset skips gracefully at __getitem__/load_image
    # time for the ~46% of cholec80/cholect50 videos whose frames were never
    # transferred to this pod). This is a big step up from the previous
    # 3000-sample uniform-stride subset (only ~900-2700 steps/task).
    cholec80 = list(VLMJSONLDataset(
        f"{emt._JSONL_DIR}/cholec80_phases_vlm_train.jsonl", "cholec80", "train", sm,
        image_root=IMAGE_ROOT,
    ))
    heichole = list(VLMJSONLDataset(
        f"{emt._JSONL_DIR}/heichole_multitask_vlm_train.jsonl", "heichole", "train", sm,
        image_root=IMAGE_ROOT,
    ))
    cholect50 = list(VLMJSONLDataset(
        f"{emt._JSONL_DIR}/cholect50_triplets_vlm_train.jsonl", "cholect50", "train", sm,
        image_root=IMAGE_ROOT,
    ))
    logger.info(f"cholec80={len(cholec80)} heichole={len(heichole)} cholect50={len(cholect50)}")

    def run_task(name, records, n_steps, prompt, has_phase, has_tools, has_action, has_triplet, tool_pos_weight=None, phase_class_weight=None):
        from collections import Counter, deque
        step = 0
        micro = 0
        t0 = time.time()
        recent_phase_preds = deque(maxlen=200)
        optimizer.zero_grad()
        while step < n_steps:
            for batch in make_batches(records, BATCH_SIZE):
                if step >= n_steps:
                    break
                imgs, phase_ids, tool_targets, action_ids = [], [], [], []
                triplet_inst_ids, triplet_verb_ids, triplet_tgt_ids = [], [], []
                for rec in batch:
                    img = load_image(rec.get("image_path", ""))
                    if img is None:
                        continue
                    imgs.append(img)
                    if has_phase:
                        phase_ids.append(label_mapper.text_to_phase_id(rec.get("phase", "")))
                    if has_tools:
                        ids = label_mapper.text_to_instrument_ids(
                            ", ".join(rec.get("tools", []) or [])
                        )
                        vec = torch.zeros(7)
                        for iid in ids:
                            if iid < 7:
                                vec[iid] = 1.0
                        tool_targets.append(vec)
                    if has_action:
                        action_ids.append(label_mapper.text_to_action_id(rec.get("action", "")))
                    if has_triplet:
                        ids = label_mapper.text_to_triplet_ids(
                            f"{rec.get('triplet_instrument','')}, {rec.get('triplet_verb','')}, {rec.get('triplet_target','')}"
                        )
                        triplet_inst_ids.append(ids[0])
                        triplet_verb_ids.append(ids[1])
                        triplet_tgt_ids.append(ids[2])
                if not imgs:
                    continue

                pooled = vlm._get_training_matching_features(imgs, prompt).to(torch.bfloat16)
                out = adapter.forward_training(pooled)

                loss = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
                if has_phase:
                    t = torch.tensor(phase_ids, device=DEVICE, dtype=torch.long)
                    loss = loss + F.cross_entropy(
                        out.phase_logits.float(), t, weight=phase_class_weight
                    )
                    recent_phase_preds.append(out.phase_logits.argmax(dim=-1).item())
                if has_tools:
                    t = torch.stack(tool_targets).to(DEVICE)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out.instrument_logits.float(), t, pos_weight=tool_pos_weight
                    )
                if has_action:
                    t = torch.tensor(action_ids, device=DEVICE, dtype=torch.long)
                    loss = loss + F.cross_entropy(out.action_logits.float(), t)
                if has_triplet:
                    ti = torch.tensor(triplet_inst_ids, device=DEVICE, dtype=torch.long)
                    tv = torch.tensor(triplet_verb_ids, device=DEVICE, dtype=torch.long)
                    tt = torch.tensor(triplet_tgt_ids, device=DEVICE, dtype=torch.long)
                    loss = loss + F.cross_entropy(out.triplet_logits["instrument"].float(), ti)
                    loss = loss + F.cross_entropy(out.triplet_logits["verb"].float(), tv)
                    loss = loss + F.cross_entropy(out.triplet_logits["target"].float(), tt)

                (loss / ACCUM_STEPS).backward()
                micro += 1
                if micro % ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                if step % 20 == 0 or step == 1:
                    elapsed = time.time() - t0
                    msg = f"[{name}] step {step}/{n_steps} loss={loss.item():.4f} ({elapsed:.0f}s)"
                    if has_phase and recent_phase_preds:
                        dist = Counter(recent_phase_preds).most_common(3)
                        msg += f" phase_pred_dist(last{len(recent_phase_preds)})={dist}"
                    logger.info(msg)
        # Flush any partial accumulation window at the end of this task's run.
        if micro % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

    # Order matters: heichole's tools labels are heavily skewed (76% empty,
    # only Grasper/Hook of 7 tools ever appear) and its phase labels are ~100%
    # "Preparation" -- both cases where training a shared head on heichole's
    # narrow signal AFTER cholec80's broader, better-balanced signal risks
    # catastrophically overwriting what cholec80 just learned (confirmed for
    # phase_head; see EVALUATION_RESULTS_REPORT.md root-cause notes). Fix:
    # heichole (phase excluded entirely; tools kept but runs FIRST) -> cholec80
    # (runs LAST, its cholec80 phase+tools signal has the final say) -> cholect50
    # (independent triplet heads, order doesn't matter here).
    logger.info("=== Training tools+action head on heichole (phase excluded, runs first) ===")
    run_task("heichole", heichole, 1800, MULTITASK_PROMPT, False, True, True, False)

    logger.info("Computing per-tool pos_weight from cholec80 (up-weights rare tools: "
                "Bipolar/Clipper/Scissors/Irrigator) ...")
    tool_pos_weight = compute_tool_pos_weight(cholec80, label_mapper)

    logger.info("Computing per-phase class_weight from cholec80 (up-weights ClippingCutting "
                "and the other rare phases, capped at 4x to avoid destabilising the majority "
                "classes that already work) ...")
    phase_class_weight = compute_phase_class_weight(cholec80, label_mapper)

    logger.info("=== Training phase+tools head on cholec80 (runs last -> final word on shared heads) ===")
    run_task("cholec80", cholec80, 6000, PHASE_PROMPT, True, True, False, False,
              tool_pos_weight=tool_pos_weight, phase_class_weight=phase_class_weight)

    logger.info("=== Training triplet heads on cholect50 ===")
    run_task("cholect50", cholect50, 6000, TRIPLET_PROMPT, False, False, False, True)

    # Save as a CANDIDATE, not in place -- the current live output_adapter.pt (v5)
    # scores phase=57.2%/macroF1=0.374, instrument macro-F1=41.8% (evaluation_results_v5.json).
    # Evaluate this candidate first; only promote it over v5 if it's actually
    # better (class-weighted CE risks nudging ClippingCutting recall up at the
    # cost of the majority classes' precision, or destabilizing the phase head).
    candidate_path = Path("checkpoints/multitask/final/output_adapter.pt.phase_weighted_candidate")
    torch.save(adapter.state_dict(), candidate_path)
    logger.info(f"Saved candidate (NOT live) to {candidate_path} -- evaluate before promoting")


if __name__ == "__main__":
    main()
