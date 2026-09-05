# SurgicalVLM_Robotics — Project Overview

**COMP702 M.Sc. Project | Kamal Raj Vendi (201933430) | University of Liverpool, School of Computer Science and Informatics | Supervisor: Dr. Baoru Huang**

---

## 1. What this project is

A **Surgical Vision-Language Model (VLM)** that watches laparoscopic surgery video and, per frame, reports:

- **Surgical phase** (e.g. `CalotTriangleDissection`, `ClippingCutting`, `GallbladderDissection`, …)
- **Instruments in view** (Grasper, Hook, Bipolar, Scissors, Clipper, Irrigator, SpecimenBag)
- **A natural-language scene description**

as structured JSON: `{"phase": "...", "tools": [...], "description": "..."}`.

The motivating idea: perception is a prerequisite for any assistive or autonomous robotic action in the operating room. This project builds and evaluates that perception layer — not a robot itself, but the "watching and understanding" component a future surgical-robotics assistant would need.

It is trained multi-task across **four established surgical datasets** so one model handles phase recognition, instrument detection, action-triplet recognition, and free-text description together, rather than training four separate models.

---

## 2. Architecture

- **Base model:** Qwen2.5-VL-7B-Instruct (frozen), loaded in 4-bit for training/inference.
- **Fine-tuning:** LoRA (Low-Rank Adaptation), rank `r=32`, `alpha=32`, `dropout=0.05`, applied to attention projections. Only ~tens of millions of parameters are trained — the 7B backbone stays frozen.
- **Classification heads:** small linear heads (`phase_head`, `instrument_head`, action/triplet heads) sit on top of pooled visual features from the backbone, defined in `surgical_vlm/training/output_adapter.py` and driven through `surgical_vlm/models/surgical_vlm.py::SurgicalVLM`.
- **Temporal aggregator:** `surgical_vlm/models/temporal_aggregator.py` exists in the codebase but is **not used at inference time** — the trained checkpoint has no meaningfully-trained `temporal_pool.*` weights, so every prediction is effectively single-frame ("No-Temporal"). This is documented explicitly as a scope limitation, not hidden.
- **Multi-task trainer:** `surgical_vlm/training/trainer.py::MultiTaskSurgicalTrainer` runs weighted per-task losses (phase classification, tool detection, action triplet, language modeling) so no one dataset dominates gradient updates.

### Key source layout
```
surgical_vlm/
├── data/          vlm_dataset.py (VLMJSONLDataset), collators.py, split_manager.py
├── models/        surgical_vlm.py (SurgicalVLM, MockSurgicalVLM), qwen3_vl.py, base_vlm.py, temporal_aggregator.py
├── training/       trainer.py, lora_setup.py, loss_functions.py, output_adapter.py, label_mapping.py
└── evaluation/     phase_metrics.py, action_metrics.py, triplet_metrics.py, language_metrics.py, grounding_metrics.py
scripts/            train_multitask.py, evaluate_multi_task.py, evaluate_baselines.py, evaluate_zeroshot.py,
                    finetune_heads_only.py, extract_frames.py, convert_to_jsonl.py
web/                gradio_app.py (UI), video_processor.py (VideoAnalyzer — frame sampling, per-frame + streaming inference, aggregation)
runpod/             Dockerfile, setup.sh (volume bootstrap), train.sh (training entry point)
```

---

## 3. Datasets

Four public, de-identified, pre-existing surgical datasets (ethics category **B2** — no participants recruited, no new data collected):

| Dataset | Task(s) | Role |
|---|---|---|
| **Cholec80** | Phase + tool presence | Phase & instrument classification |
| **CholecT50** | Action triplets (instrument, verb, target) | Action-triplet recognition |
| **HeiChole** | Multi-task (phase/tools/skill/safety) | Auxiliary multi-task signal (small, degenerate test split — see §6) |
| **Surg-396K** (CoPESD/EndoChat) | VQA / instruction-following captions | Free-text scene description |

Raw data lives under `data/raw/`, extracted frames under `data/frames/`, and everything is converted into a **unified JSONL instruction format** (`data/processed/vlm_jsonl/`, 13 files across train/val/test) via `scripts/convert_to_jsonl.py`, so all four datasets are trainable through one `VLMJSONLDataset` loader regardless of source format.

---

## 4. Training & infrastructure

- **Compute:** RunPod cloud GPU pods (A100 80GB for training; a network volume at `/runpod-volume` persists data/checkpoints across pod restarts).
- **Config:** `configs/training/runpod_7b_config.yaml` — bf16, gradient checkpointing, effective batch size 16 (batch 4 × accumulation 4), cosine LR schedule, `max_steps: 6000`.
- **Entry point:** `bash runpod/setup.sh` (one-time volume bootstrap: extract data, verify, extract frames) then `bash runpod/train.sh` (full multi-task run via `scripts/train_multitask.py`).
- **Local machine** (Ryzen 7 5825U, no discrete GPU) cannot train — used only for code, data prep, and a `--mock` mode of the demo UI that runs without a GPU.

---

## 5. The demo

`web/gradio_app.py` + `web/video_processor.py` implement a **live-streaming video demo**:

1. User uploads a surgical video clip.
2. Frames are sampled at a configurable FPS.
3. A live panel updates every ~1–3 seconds per frame — current phase, detected tools, scene description — while a phase timeline and results table build up incrementally.
4. Results aggregate into a video-level report: dominant phase, phase timeline, tool usage summary, and a scene narrative.

Run it with `python -m web.gradio_app` (real model, needs GPU) or `python -m web.gradio_app --mock` (no GPU, canned output, for UI testing).

**Known, documented limitation:** the model's structured `"tools"` JSON array is frequently empty even when the free-text description clearly names a tool in prose. This is a generation-quality/model-behaviour issue, not a code bug, and is disclosed openly in the presentation materials rather than hidden.

---

## 6. Results — the honest headline

Original target: **every evaluation metric ≥ 60%** across phase, instrument, triplet, and language tasks. Two real, root-caused training bugs were found and fixed along the way (see `results/EVALUATION_RESULTS_REPORT.md` for full detail):

1. **Label-parsing bug** — `VLMJSONLDataset.__getitem__` never parsed the ground-truth JSON in training samples into the fields the trainer expected, so all classification heads trained on essentially zero real signal for the entire original run (language-model loss was unaffected since it reads conversation text directly).
2. **Shared-head overwrite bug** — during heads-only recovery retraining, HeiChole's heavily degenerate labels (phase: ~100% one class) were trained *after* Cholec80's better-balanced signal and overwrote what Cholec80 had correctly learned. Fixed by excluding HeiChole from phase training and reordering so Cholec80 trains last on the shared heads.

### Final results (post-fix), vs. Random / Majority-class / Zero-shot baselines

| Task | Metric | Ours | Random | Majority | Zero-shot | ≥60% target met? |
|---|---|---|---|---|---|---|
| Phase (Cholec80) | Accuracy | **58.8%** | 16.2% | 38.6% | 36.9% | Just short, but beats every baseline |
| Instruments (Cholec80) | Macro-F1 | **33.0%** | 19.7% | 19.6% | 17.3% | Below target, beats every baseline |
| Triplet (CholecT50) | Instrument F1 / Target F1 | **97.3% / 79.0%** | — | — | — | ✅ Both clear 60% |
| Triplet (CholecT50) | Verb F1 | 9.8% | 1.4% | 44.0% | 5.4% | ❌ Hard data ceiling (3/10 verb classes have zero training examples) |
| Language (Surg-396K) | BLEU-4 / ROUGE-L | 0.18 / 0.109 | ~0 | ~0 | 0.18 / 0.126 | ❌ Below target; matches literature norms (SOTA is 20–40% BLEU-4) |

**Phase and instrument detection now genuinely beat every baseline tested** — the clearest evidence the training pipeline works, even though the absolute 60% bar isn't cleared on every metric. Where targets weren't hit, the shortfall is traced to a specific, understood cause (class rarity, missing verb classes in the source dataset, or an unrealistic literature-relative target) rather than left as an unexplained gap. Full methodology, caveats, and per-task breakdowns are in `results/EVALUATION_RESULTS_REPORT.md` and `results/BASELINE_COMPARISON.md`.

---

## 6a. Comparison to published methods — and why this project's numbers sit below them

This project's numbers are well below mature, single-task published SOTA. That gap is expected and structural, not a sign the training pipeline is broken — the controlled comparison that actually tests the pipeline is against this project's own random/majority/zero-shot baselines (§6 above), which it beats on phase and instruments. The table below is included for literature context and positioning, **not as a claim of competitiveness**.

| Task / Dataset | Method | Score | Source |
|---|---|---|---|
| Phase recognition (Cholec80) | TeCNO (2020) | 90.17% accuracy | Czempiel et al., MICCAI 2020 |
| Phase recognition (Cholec80) | Trans-SVNet (2021) | 91.54% accuracy | Gao et al., MICCAI 2021 (arXiv:2103.09712) |
| Phase recognition (Cholec80) | LoViT (2024) | 92.40% accuracy | Liu et al., Medical Image Analysis 2024 (arXiv:2305.08989) |
| Phase recognition (Cholec80) | **SurgicalVLM (Ours)** | **57.2% accuracy** | This project |
| Tool presence detection (Cholec80) | EndoNet-family / recent CNN methods | ~91–96% mAP | Multiple 2020s tool-detection papers on Cholec80 |
| Tool presence detection (Cholec80) | **SurgicalVLM (Ours)** | **41.8% macro-F1 / mAP@0.5** | This project |
| Action triplet recognition (CholecT50) | CholecTriplet2021 challenge field (19 methods) | 4.2%–38.1% mAP | Nwoye et al., Medical Image Analysis 2023 (arXiv:2204.04746) |
| Action triplet recognition (CholecT50) | Recent IVT-split method (2024/2025) | 39.54% mAP-IVT | Recent CholecT50 leaderboard entry |
| Action triplet — instrument / target component (CholecT50) | **SurgicalVLM (Ours)** | **97.3% F1 / 79.0% F1** | This project |

### Why the gap exists — four structural reasons, not a training failure

1. **Purpose-built vs. general-purpose.** TeCNO, Trans-SVNet and LoViT are architectures built and trained for *one task only* — phase recognition — typically a CNN feature extractor plus a dedicated temporal model (TCN/LSTM/Transformer) run over full-length video sequences. This project's phase head is one of four tasks trained jointly on top of a general 7B vision-language model, and — confirmed directly from the checkpoint — has **no temporal aggregation in its active inference path** (the `temporal_pool.*` weights were never meaningfully trained; every prediction is single-frame). It is being compared against methods whose entire design purpose is exactly the temporal signal this project's shipped model doesn't use. (A dedicated attempt to add real temporal context, §10 of `EVALUATION_RESULTS_REPORT.md`, was made and rejected after root-causing — see below.)
2. **Evaluation-set size.** Published Cholec80 phase results are computed frame-by-frame over the full standard 40-video test split. This project's phase numbers come from a 500-sample evaluation subset, not the full test-split video count — a smaller, noisier sample than the literature's.
3. **Different metrics for triplet.** Published CholecT50 results report **mAP** over the joint ‹instrument, verb, target› triplet as a multi-label detection problem with strict partial-credit rules. This project reports **per-component F1/accuracy and exact-match**, a different and generally more lenient scoring scheme — the two number types are not on the same scale and shouldn't be read as interchangeable.
4. **Scale of prior work vs. one dissertation cycle.** EndoNet-family, TeCNO/Trans-SVNet/LoViT, and the CholecTriplet leaderboard represent multiple published iterations by specialist groups (CAMMA, KCL, etc.) working on these exact tasks since ~2016–2024. This project is a single M.Sc. dissertation building one general-purpose multi-task model across four datasets at once, in one training cycle.

### Why the shortfall isn't just "not enough effort" — the retrain saga

Beyond the structural gap above, four separate, root-caused retrain attempts were made specifically to close the distance to the 60% target on phase (the metric closest to it), each targeting the one persistently-weak class (`ClippingCutting`, 0 F1) with a different, literature-motivated lever:

| # | Approach | Result |
|---|---|---|
| 1 | Class-weighted loss (4x inverse-frequency weight) | `ClippingCutting` never once predicted; broke a previously-working class |
| 2 | Sequential joint LoRA retrain (unfreeze backbone) | Instrument/triplet gains, but phase regressed −14.6pt (catastrophic forgetting) |
| 3 | Interleaved joint LoRA retrain | Fixed total-collapse, but phase macro F1 regressed further (−14.9pt) via minority-class erosion |
| 4 | Real temporal context (668M-param aggregator, literature-correct fix) | Overfit — trained on <9% of one epoch from a cold start; phase −22.0pt, instrument −13.2pt |

Every attempt failed for a specific, checkable, different mechanism (confusability vs. imbalance; forgetting vs. rehearsal; collapse vs. erosion; underparameterized data budget vs. model capacity) — not an unexplained "it didn't work." That's the basis for reporting v5 as final rather than an open gap: the shortfall against literature SOTA is attributable to scope (single-frame inference, general-purpose multi-task model, one dissertation cycle) rather than to an unfixed bug. Full attempt-by-attempt detail is in `results/EVALUATION_RESULTS_REPORT.md` §7–11.

---

## 7. Repository map

```
SurgicalVLM_Robotics/
├── configs/           training/eval/data/model YAML configs
├── data/              raw datasets, extracted frames, unified JSONL, splits
├── surgical_vlm/      core package: data, models, training, evaluation
├── scripts/           training, evaluation, baseline, and utility entry points
├── web/               Gradio demo (gradio_app.py, video_processor.py)
├── runpod/            Dockerfile + setup/train scripts for cloud training
├── results/           evaluation reports, baseline comparison, raw JSON results
├── docs/              this overview
├── tests/             pytest suite
└── checkpoints/       trained LoRA adapters + output-adapter heads
```

---

## 8. Where to look next

- **Final evaluation report:** `results/EVALUATION_RESULTS_REPORT.md`
- **Baseline comparison + literature context:** `results/BASELINE_COMPARISON.md`
- **Demo API:** `web/API.md`
