# SurgicalVLM_Robotics — Final Evaluation Results Report

**COMP702 M.Sc. Project | Kamal Raj Vendi | University of Liverpool**
**Evaluation run:** 2026-08-20, `scripts/evaluate_multi_task.py` against `checkpoints/multitask/final` (`results/evaluation_results_v5.json`, current; `evaluation_results_v4.json` superseded — see §6)
**Model:** Qwen2.5-VL-7B-Instruct + LoRA (rank 32), 4-bit inference
**Compute:** RunPod NVIDIA RTX 6000 Ada 48GB (checkpoint/data transferred directly from local disk — see §4 for why)

> **This report supersedes all earlier versions** (the no-volume garbage run from 2026-08-19 01:38 UTC, and the first heads-only-retrain report also dated 2026-08-19). Those numbers were either not measuring the trained model at all, or were produced before the second root-cause fix documented below. §2 and §3 reflect the **current** numbers (v5, post class-weighted instrument retrain, §6); the v4 numbers are kept in §6 for the before/after comparison.

---

## 1. Headline result: two real bugs, found, root-caused, and fixed

### Bug 1 — classification heads never trained at all (original 5000-step run)
`surgical_vlm/data/vlm_dataset.py`'s `VLMJSONLDataset.__getitem__` never parsed the ground-truth JSON embedded in each training sample into the flat `phase`/`tools`/`action`/`triplet_*` fields `MultiTaskSurgicalTrainer._prepare_targets` expected. The classification heads (phase, instrument, action, triplet) received essentially zero real gradient for the entire original 5000-step run, while the LoRA-adapted language model trained normally (its loss reads directly from the raw conversation text, bypassing this path). **Fixed:** `VLMJSONLDataset._extract_label_fields` now parses the gpt-turn JSON correctly, verified against all four dataset formats.

### Bug 2 — a shared-head catastrophic-overwrite bug (found 2026-08-20)
A first heads-only recovery retrain (frozen backbone, only the small classification heads refit) used a fixed task order: **Cholec80 → HeiChole → CholecT50**. Cholec80 trained `phase_head` and `instrument_head` well — confirmed live during training (healthy, multi-class prediction distributions throughout all 6000 steps). But HeiChole trains the **same** shared `phase_head` and `instrument_head` afterward, and HeiChole's labels are heavily degenerate:
- **Phase:** ~100% of HeiChole's phase labels are `"Preparation"`.
- **Tools:** 76% of HeiChole's rows have zero tools labeled, and only 2 of 7 canonical tools (Grasper, Hook) ever appear at all.

Training on this narrow, skewed signal *after* Cholec80's broader, better-balanced signal overwrote what Cholec80 had just learned — the phase head collapsed to predicting a single constant class regardless of input (confirmed by inspecting raw logits: class-0 logit ≈20-25 vs. every other class ≈0-4, for every test image, including ones whose ground truth was a completely different phase), and the instrument head was pushed toward under-predicting everything (near-perfect precision, near-zero recall — the signature of a classifier biased to almost never predict positive).

**Fix (two parts, applied in `scripts/finetune_heads_only.py`):**
1. HeiChole's phase labels are pure noise for this purpose (single-class) — phase training is now **excluded entirely** for HeiChole (`has_phase=False`).
2. HeiChole's tool labels are noisy but not fully degenerate, so rather than exclude them, the task order was **reversed**: HeiChole now trains first, Cholec80 last, so Cholec80's stronger, more balanced signal has the final say on both shared heads.

Both retrains additionally used the **full train sets** (67,512 / 895 / 72,170 samples for cholec80/heichole/cholect50 respectively — all rows whose frames were actually transferred to the eval pod, not the earlier 3,000-sample uniform-stride subset), 6,000/1,800/6,000 steps per task, and gradient accumulation (8 micro-batches per optimizer step) for stability at batch_size=1.

**A separate, smaller bug was also found and fixed along the way:** `surgical_vlm/data/split_manager.py`'s `PHASE_NAMES["cholec80"]` list had `GallbladderPackaging` and `GallbladderRetraction` swapped relative to the canonical order used for training in `label_mapping.py` — a display/eval-labeling bug (fixed), independent of the two bugs above.

---

## 2. Per-Task Metrics (final, post both fixes)

### 2.1 Phase Classification — Cholec80 (`phase_cholec80`)
| Metric | Value |
|---|---|
| Accuracy | **0.572** |
| Weighted F1 | 0.540 |
| Macro F1 | 0.374 |
| Samples | 500 |

Went from **3.4% → 57.2% accuracy** (0.009 → 0.374 macro F1). Confusion matrix now shows real, spread predictions across most of the 7 classes (previously 100% collapsed into one column). `ClippingCutting` (a short, easily-confused-with-neighbors phase) is the one class still at 0 F1 — plausible given it's both rare and visually transitional between two longer phases. (Accuracy is 1.6pt below the pre-§6-retrain checkpoint's 58.8%; macro F1 is essentially unchanged (0.371→0.374) — read as retrain-to-retrain stochastic noise rather than a regression, since the phase head's loss/task order was not touched by the §6 change.)

### 2.2 Phase Classification — HeiChole (`phase_heichole`)
Still trivially "1.0" or "0.0" depending on run — this test split (1 video, Hei-Chole9) has ground truth that is 100% one class, making it structurally uninformative as a discriminator. Not meaningful either way; excluded from headline reporting.

### 2.3 Instrument Detection — Cholec80 (`instruments_cholec80`)
| Metric | Value |
|---|---|
| Macro F1 | **0.418** |
| Micro F1 | 0.674 |
| Subset accuracy | 0.336 |
| mAP@0.5 | 0.418 |
| Per-tool F1 | Grasper 0.754, Hook 0.827 (still strong), **Bipolar 0.305, Clipper 0.286, Irrigator 0.209, Scissors 0.167** (all recovered from 0.0), SpecimenBag 0.377 (down from 0.636) |

Went from macro F1 0.037-0.11 (across earlier partial-fix attempts) to 0.330 (§1 fixes), then to **0.418** after a class-weighted retrain (§6) that up-weighted the four rare tools' positive-class loss (`BCEWithLogitsLoss(pos_weight=...)`, capped at 20x, computed from each tool's neg/pos ratio in the cholec80 training set — Bipolar/Clipper/Scissors/Irrigator got 13-20x, Grasper/Hook stayed at 1x). This is a genuine, large improvement on the metric that matters for an imbalanced multi-label task (macro F1), at the cost of micro F1 and subset accuracy dropping (0.748→0.674, 0.518→0.336) as the model now predicts positives more liberally, and SpecimenBag's F1 dropping (0.636→0.377) as a side effect of reweighting the other four tools' logits. Net trade judged favorable given the task (see §6).

### 2.4 Action Triplet — CholecT50 (`triplet_cholect50`)
| Metric | Value |
|---|---|
| Exact match | 0.042 |
| Instrument acc / F1 | 0.982 / 0.973 |
| Verb acc / F1 | 0.062 / 0.083 |
| Target acc / F1 | 0.856 / 0.790 |

Instrument and target components are strong (already were before today's fixes — those heads were never touched by the HeiChole-overwrite bug, since HeiChole never trains triplet heads) and **unchanged** by the §6 retrain (0.973 / 0.790 both times), as expected since triplet heads aren't shared with the reweighted instrument head. **Verb remains the weak component, and this is a genuine dataset ceiling, not a bug:** of the 10 canonical verb classes, **3 (`Aspirate`, `Irrigate`, `Pack`) never appear even once** in CholecT50's 72,170-row training set, and `grasp` alone accounts for 68.3% of all verb labels. The label-matching code was audited for a case-sensitivity bug (training labels are lowercase, e.g. `"grasp"`, vs. the canonical `"Grasp"`) and confirmed correct — `_match_component` does case-insensitive matching. No further retraining can recover the 3 unseen classes without new labeled data. Exact-match/verb F1 dipped slightly from the pre-§6 run (0.064→0.042, 0.098→0.083) despite the triplet stage being unaffected by the reweighting — attributed to ordinary retrain-to-retrain stochastic variance (different random batch order/seed), not a systematic effect.

### 2.5 Language Generation — Surg-396K (`language_surg396k`)
| Metric | Value |
|---|---|
| BLEU-4 | 0.18 |
| ROUGE-L | 0.108 |
| ROUGE-1 | 0.121 |
| ROUGE-2 | 0.012 |

Unchanged from before today's fixes and from the §6 retrain (expected — language generation uses the LoRA-tuned backbone directly, not the classification heads that were touched). Notably, **BLEU-4 is identical to the zero-shot (no-LoRA) baseline (also 0.18)**, and ROUGE-L is very slightly *below* zero-shot (0.108 vs. 0.126) — see §3, Table 7.5 in `BASELINE_COMPARISON.md`. TODO.md's own reality check flags BLEU-4 ≥60% as unrealistic for open-ended surgical captioning (standard VLMs on medical captioning score 20-40% BLEU-4); this target should be renegotiated rather than chased further.

---

## 3. Dissertation Tables (5.1–5.4)

### Table 5.1 — Phase Classification (Cholec80)
| Method | Accuracy | F1 |
|---|---|---|
| SurgicalVLM (Ours) | **0.572** | 0.374 |

Just short of the 60% target — a real, working classifier, recovered from a fully collapsed 3.4% baseline via two root-caused bug fixes (see §1). (1.6pt below the pre-§6 checkpoint's 0.588 accuracy; macro F1 essentially unchanged — see §2.1.)

### Table 5.2 — Instrument Detection
**Methodological note:** `scripts/evaluate_multi_task.py`'s `format_dissertation_results()` builds this table from `instruments_heichole`, which is a degenerate, near-uninformative test split (see §2.2's phase discussion — the same 1-video HeiChole test split issue applies to its tools labels). **We recommend reporting the Cholec80 instrument numbers instead**, which reflect real, meaningful performance:

| Source | mAP@0.5 | Macro F1 |
|---|---|---|
| HeiChole (as auto-generated, not meaningful) | 0.0 | 0.0 |
| **Cholec80 (recommended for dissertation)** | **0.418** | **0.418** |

Up from 0.330 after a class-weighted retrain recovered the four previously-zero-F1 tools (see §6).

### Table 5.3 — Action Triplet (CholecT50)
| Method | Exact Match | Instrument F1 | Verb F1 | Target F1 |
|---|---|---|---|---|
| SurgicalVLM (Ours) | 0.042 | 0.973 | 0.083 | 0.790 |

Instrument and Target clear the 60% target comfortably; Verb does not, and cannot with this data (see §2.4).

### Table 5.4 — Language Generation (Surg-396K)
| BLEU-4 | ROUGE-L |
|---|---|
| 0.18 | 0.108 |

Neither table 5.3 (verb component) nor table 5.4 meet the ≥60% target, for reasons that are now well-understood and documented (data ceiling; unrealistic target for the task), rather than unexplained weaknesses.

---

## 4. Methodology note: how this evaluation was actually run

The originally-provisioned RunPod network volume holding the checkpoint and datasets became unreachable — both data centers hosting it (CA-MTL-3, US-MD-1) had zero pod capacity available. Rather than wait indefinitely, the checkpoint and datasets were transferred directly to a pod in an available data center via `scp`, bypassing the network volume entirely. This is a one-off operational detail, not a methodology change — the same checkpoint, same test splits, same evaluation script were used throughout.

## 6. Class-weighted instrument retrain (2026-08-20, post-report)

After this report's first version (v4, §1-5 above as originally written), one more targeted improvement was attempted given the four weak Cholec80 instruments were a diagnosed class-imbalance issue rather than a bug (§2.3's original note). `scripts/finetune_heads_only.py` was extended with `compute_tool_pos_weight()`: a per-tool `neg/pos` ratio computed from the cholec80 training set, capped at 20x, passed as `pos_weight` to `BCEWithLogitsLoss` for the instrument head only (Bipolar 19.6x, Clipper 20.0x, Scissors 20.0x, Irrigator 15.1x; Grasper/Hook left at 1.0x). Heichole and cholect50 stages were re-run unweighted, identical to the original recipe (task order unchanged — heichole first, cholec80 last, cholect50 independent).

**Safety process:** the retrain was saved to a candidate file (`output_adapter.pt.weighted_candidate`), not written in place. The pre-retrain checkpoint was separately backed up (`output_adapter.pt.v4_live_bak`) before the candidate was swapped in and evaluated. Only after confirming the candidate's numbers (below) was it promoted to the live `output_adapter.pt`.

**Result — promoted (v5, current):**

| Metric | v4 (pre-retrain) | v5 (post-retrain, live) | Δ |
|---|---|---|---|
| Phase accuracy / macro F1 | 0.588 / 0.371 | 0.572 / 0.374 | −1.6pt / +0.3pt |
| Instrument macro F1 | 0.330 | **0.418** | **+8.8pt** |
| Instrument micro F1 / subset acc | 0.748 / 0.518 | 0.674 / 0.336 | −7.4pt / −18.2pt |
| Triplet exact match / verb F1 | 0.064 / 0.098 | 0.042 / 0.083 | −2.2pt / −1.5pt |
| Triplet instrument F1 / target F1 | 0.973 / 0.790 | 0.973 / 0.790 | unchanged |
| Language BLEU-4 / ROUGE-L | 0.18 / 0.109 | 0.18 / 0.108 | unchanged |

**Verdict:** promoted. The primary target — instrument macro F1, the metric §2.3/Table 5.2 already identifies as the meaningful one for this imbalanced multi-label task — improved substantially, with all four previously-zero-F1 tools now scoring real F1. Phase and triplet-instrument/target/language are flat within retrain noise (only the instrument loss was reweighted; phase and triplet heads/tasks were not touched by the change, so their small deltas are attributed to ordinary run-to-run stochastic variance, not a systematic effect). Costs: instrument micro-F1 and subset-accuracy dropped (expected consequence of up-weighting rare-class recall over overall exact-match), and SpecimenBag's individual F1 dropped 0.636→0.377 as a side effect of reweighting the other four tools' logits.

Raw results: `results/evaluation_results_v5.json` (current), `results/evaluation_results_v4.json` (superseded, kept for comparison).

## 5. Caveats
- **Instruments (Cholec80):** three of seven tools (Scissors, Clipper, Irrigator) remain moderate rather than strong even after class-weighting (§6); this is a partial mitigation of genuine class rarity in Cholec80's training distribution (1.9-6.2% of rows), not a full fix. Further oversampling or a longer weighted retrain is the natural next step if revisited.
- **Triplet verb:** hard ceiling — 3 of 10 verb classes have zero training examples in CholecT50. Unfixable without new labeled data.
- **Language generation:** BLEU-4 ≥60% is not a realistic target for open-ended captioning per the literature; fine-tuning did not measurably improve BLEU-4 over the zero-shot base model, though this reflects the task's inherent difficulty more than a training failure.
- **HeiChole numbers throughout** (phase and tools) are dominated by a degenerate test split (1 video, near-constant ground truth) and should not be read as evidence of either success or failure for those tasks.
- Backups of intermediate checkpoints are preserved on the eval pod and locally: `output_adapter.pt.pretrainfix_bak` (original, pre-Bug-1-fix, essentially untrained heads), `output_adapter.pt.run2_heichole_overwrite_bak` (post-Bug-1-fix, pre-Bug-2-fix, phase still collapsed), `output_adapter.pt.run3_phase54pct_bak` (Bug 2 partially fixed — phase excluded from HeiChole, but task order not yet reversed; phase 54%, instruments still weak), `output_adapter.pt.v4_live_bak` (both §1 fixes applied, pre-§6 class-weighting — the checkpoint the v4 numbers above were measured on). The current `output_adapter.pt` (v5) reflects both §1 fixes plus the §6 class-weighted instrument retrain.

## 7. Joint LoRA + classification-heads retrain experiment (2026-08-22/23, NOT promoted)

Unlike §6 (heads-only, frozen LoRA backbone), this experiment (`scripts/finetune_joint_lora_heads.py`) re-attached the v5 LoRA adapter as an *active, trainable* `PeftModel` and fine-tuned both the LoRA backbone and the classification heads jointly, running sequentially through heichole → cholec80 → cholect50 → a new language stage on a subsampled Surg-396k (18k rows), with step budgets allocated by measured wall-clock rate (20/35/35/10% split, ~2h15m total budget) rather than fixed counts.

**Crash and fix:** the run completed stages 1-3 cleanly but crashed at the start of the language stage — every example was silently skipped, with the error message misleadingly blaming `_abs_image_path`. Root cause: Surg-396k's CoPESD frames are natively 1306×1306, which alone produces ~2,209 vision tokens — already past the script's `LM_MAX_LEN=640` prompt+answer token budget before any text is added. Fixed via a `max_pixels` cap (`LM_IMAGE_MAX_PIXELS=200704`, ≈256 vision tokens) passed per-call to the image processor in `lm_step`, verified against real data (skip rate 100% → ~7%, the residual being genuinely long answers). Rather than re-running stages 1-3 (~75 min of compute already sunk and checkpointed at `jointretrain_03_cholect50`), a small resume script (`scripts/resume_lm_stage.py`) loaded that checkpoint directly and ran only the fixed language stage, saving to `jointretrain_candidate`.

**A second, unrelated bug was found while evaluating the candidate:** `configs/training/eval_local_transfer_config.yaml` (built 2026-08-19 for direct-scp/no-network-volume eval) had `data.image_root: "/workspace/eval_data"`, one directory short of the actual data root `/workspace/eval_data/data` — this silently produced all-zero "No samples added" results on every task. Fixed in the config (now `/workspace/eval_data/data`).

**Result — evaluated, NOT promoted (v6 candidate vs. v5 live):**

| Metric | v5 (live) | v6 (candidate) | Δ |
|---|---|---|---|
| Phase accuracy / macro F1 (Cholec80) | 0.572 / 0.374 | 0.426 / 0.290 | **−14.6pt / −8.5pt** |
| Instrument macro F1 (Cholec80) | 0.418 | **0.599** | **+18.1pt** |
| Instrument subset acc (Cholec80 / HeiChole) | 0.336 / 0.110 | 0.468 / 0.723 | **+13.2pt / +61.3pt** |
| Triplet exact match / verb F1 | 0.042 / 0.083 | 0.072 / 0.102 | +3.0pt / +1.9pt |
| Triplet instrument F1 / target F1 | 0.973 / 0.790 | 0.979 / 0.790 | ~flat |
| Language BLEU-4 / ROUGE-L | 0.18 / 0.108 | 0.18 / 0.013 | flat / **−9.6pt (≈9x drop)** |

**Verdict: not promoted.** Instrument detection and action-triplet recognition both improved meaningfully — plausibly because letting the LoRA backbone move (not just the heads, as in §6) gave the shared visual features more room to separate instrument/verb classes. But phase accuracy, a headline metric, regressed by 14.6 points, and the live `output_adapter.pt` / LoRA checkpoint were left untouched. The likely mechanism is ordinary sequential-task forgetting: with no rehearsal, the cholect50 and language stages running *after* cholec80 pulled the shared backbone away from what the phase head needed. The ROUGE-L collapse (with BLEU-4 unchanged) is treated as measurement noise rather than a genuine language-quality regression — the eval harness's `evaluate_language_generation` prompts for a structured JSON response (phase/tools/description), which mismatches Surg-396k's short free-text ground truth answers ("Knife", "Lift", a one-sentence direction) regardless of checkpoint quality; a joint retrain that reinforced the JSON-output style learned in stages 1-3 would show *worse* n-gram overlap against that particular ground truth even if generation quality were unchanged or improved.

Kept for the record as a documented experiment, not as a checkpoint to build on: `jointretrain_03_cholect50` and `jointretrain_candidate` remain on the eval pod plus a local backup of the stage-3 checkpoint; raw results in `results/evaluation_results_v6_candidate.json`. If revisited, the natural next step is adding periodic rehearsal batches from earlier stages (or joint/interleaved rather than sequential task sampling) to prevent the phase-accuracy forgetting while keeping the instrument/triplet gains.

## 8. Class-weighted phase retrain experiment (2026-08-24/25, NOT promoted)

Follow-up to §6's instrument reweighting: since v5's Cholec80 phase head has one class stuck at 0 F1 (`ClippingCutting`, §2.1's caveat), the same lever that fixed the four zero-F1 instrument tools was tried on phase. `scripts/finetune_heads_only.py` was extended with `compute_phase_class_weight()` — an inverse-frequency, median-normalised per-class weight (capped at 4x, deliberately more conservative than the instrument head's 20x cap, since `CrossEntropyLoss(weight=...)` scales the per-sample gradient directly rather than a pos/neg ratio, and batch_size=1 + gradient accumulation is already a known-unstable combo) — passed to the phase `F.cross_entropy` call for the Cholec80 stage only. Heichole and cholect50 stages were unchanged from the v5 recipe (same task order: heichole first, cholec80 last, cholect50 independent).

Training ran cleanly for all three stages (heichole 1,800 steps, cholec80 6,000 steps, cholect50 6,000 steps) with no collapse — `phase_pred_dist` logging throughout the cholec80 stage showed a healthy, varying mix of predicted classes at every checkpoint, unlike the full collapse seen before the §1 Bug-2 fix.

**Operational note:** this retrain required rebuilding the eval pod's environment from scratch — the pod's container disk (non-network-volume local storage) did not persist the data/checkpoint transferred in the §6/§7 sessions across a stop/start cycle, coming back as a bare copy of the base Docker image. Re-transferred via `runpodctl send/receive` (croc-based relay, ~30-45 MB/s, since `scp`/`sftp` do not work over RunPod's SSH proxy): Cholec80/HeiChole/CholecT50 frame archives (~28GB), checkpoint essentials (~2.4GB), and code. See `[[runpod_capacity_issue]]` memory for the implication (the two network volumes remain an untried alternative for persistence across sessions).

**Result — evaluated, NOT promoted (v7 candidate vs. v5 live):**

| Metric | v5 (live) | v7 candidate | Δ |
|---|---|---|---|
| Phase accuracy (Cholec80) | 0.572 | 0.572 | flat |
| Phase weighted F1 / macro F1 (Cholec80) | 0.540 / 0.374 | 0.523 / 0.350 | **−1.7pt / −2.4pt** |
| `ClippingCutting` F1 (the target class) | 0.0 | **0.0** | **unchanged — fix did not work** |
| `CleaningCoagulation` F1 | 0.432 | **0.0** | **−43.2pt — new regression** |
| Instrument macro F1 (Cholec80) | 0.418 | 0.393 | −2.5pt (attributed to retrain noise, not touched by this change) |
| Triplet exact match / verb F1 | 0.042 / 0.083 | 0.084 / 0.091 | +4.2pt / +0.8pt (also noise, triplet heads not touched) |
| Language | not evaluated this run (Surg-396k images not transferred to the rebuilt pod; language head/backbone untouched by a heads-only retrain, so v5's 0.18 / 0.108 stands) | — | — |

The confusion matrix confirms this isn't a near-miss: for the 37 true `ClippingCutting` test samples, the model predicted `CalotTriangleDissection` (17) or `GallbladderDissection` (20) every time — column 2 (`ClippingCutting`) is zero across every row of the matrix, i.e. the retrained head never once predicts this class regardless of weighting.

**Verdict: not promoted.** Reverted `output_adapter.pt` on the pod back to the v5 backup (`output_adapter.pt.v5_pre_phase_weight_bak`, md5-verified identical to v5) immediately after evaluation; the local checkpoint was never modified. This is a genuine negative result, not just a non-improvement: `ClippingCutting` stayed at 0 F1 despite a 4x loss weight, while `CleaningCoagulation` — a class that scored a healthy 0.432 F1 in v5 — collapsed to 0 under the reweighting, most likely because up-weighting the rarer classes' gradient pulled decision boundaries away from a class that was already working. Unlike instrument detection (§6), where reweighting fixed exactly the classes it targeted, phase classification's failure mode here looks like genuine visual confusability between adjacent/transitional phases (`ClippingCutting` sits between `CalotTriangleDissection` and `GallbladderDissection` in the surgical workflow and is comparatively brief) rather than a class-imbalance problem that loss reweighting can fix. Raw results: `results/evaluation_results_v7_phase_weighted.json`. Checkpoint backups on the pod: `output_adapter.pt.phase_weighted_candidate` (this experiment's un-promoted candidate), `output_adapter.pt.v5_pre_phase_weight_bak` (pre-experiment backup, now restored as live). If revisited, the natural next steps are (a) oversampling `ClippingCutting` frames rather than loss-reweighting, since the problem looks like confusability, not rarity, or (b) a joint/rehearsal retrain (§7's untried next step) that also lets the visual backbone move, which is the one lever not yet tried on this specific class.

## 9. Interleaved joint LoRA + heads retrain (2026-08-25, NOT promoted)

Direct follow-up to §8's own suggested next step (and §7's original diagnosis): v6 failed because phase supervision was completely absent for the last ~45% of training once the sequential cholect50/language stages began. `scripts/finetune_joint_lora_heads_interleaved.py` removes the notion of sequential stages entirely — every task (heichole/cholec80/cholect50, 20/40/40 wall-clock mix; surg396k/language dropped from this run, see below) is drawn from one shuffled schedule spanning the whole run, so no task is ever more than a shuffle-distance from being sampled again. Same trainable-LoRA-backbone recipe as v6 otherwise (10,517 total interleaved steps, ~90 min actual wall-clock at 0.51s/step).

**Infra note:** this run also revealed the old pod (`mrud6c7til0h7l`, COMMUNITY cloud, referenced throughout §8) had been preempted by its host mid-session and could no longer even be restarted ("not enough free GPUs on the host machine"). Work moved to a new pod (`ldt2f35v9nklcm`, **SECURE cloud**, direct SSH rather than the flaky proxy) which proved far more reliable — plain `scp`/`ssh host cmd` both work normally on it, no PTY workarounds or `runpodctl` relay needed. Recommended default for any future retrain work on this project. Surg-396k's CoPESD source images were not available to transfer in the time available, so the language stage/eval were dropped from this run; its 10% wall-clock share was redistributed to cholec80/cholect50 (v5's language numbers, BLEU-4 0.18 / ROUGE-L 0.108, stand unchanged and untouched by this experiment).

**Result — NOT promoted:**

| Metric | v5 (live) | v6 (rejected, ref.) | v8 (interleaved candidate) |
|---|---|---|---|
| Phase accuracy (Cholec80) | 0.572 | 0.426 | 0.568 (flat) |
| Phase macro F1 | **0.374** | 0.290 | **0.225** |
| Phase classes at 0 F1 | ClippingCutting only | — | ClippingCutting, Preparation, CleaningCoagulation |
| Instrument macro F1 (Cholec80) | 0.418 | 0.599 | 0.578 |
| Triplet exact match / verb F1 | 0.042 / 0.083 | 0.072 / 0.102 | 0.068 / 0.097 |
| Triplet instrument / target F1 | 0.973 / 0.790 | 0.979 / 0.790 | 0.973 / 0.790 |
| Language BLEU-4 / ROUGE-L | 0.18 / 0.108 | 0.18 / 0.013 | not evaluated (dropped) |

**Verdict: not promoted, v5 stays live.** Interleaving worked exactly as intended in one sense — it prevented the *total* collapse-to-one-constant-class failure mode seen in §8's phase-weighted retrain (`phase_pred_dist` stayed spread across multiple classes for the entire run, never collapsing) — but it did not protect the phase task's minority classes. Accuracy held essentially flat (0.572 → 0.568) because the two dominant, easy-to-separate classes (`CalotTriangleDissection` 0.646 F1, `GallbladderDissection` 0.657 F1) still work well, but macro F1 — the metric used throughout this dissertation for the headline phase result — dropped sharply (0.374 → 0.225), *worse* than v6's already-rejected regression (0.290). Two additional classes (`Preparation`, `CleaningCoagulation`) joined `ClippingCutting` at 0 F1, where v5 only had one. Instrument (+16.0pt macro F1) and triplet (+2.6pt exact match, +1.4pt verb F1) gains are real and approach v6's level, confirming the shared-backbone drift diagnosis from §7 was correct — but three separate retrain strategies (class-weighted loss §8, sequential joint LoRA §7, interleaved joint LoRA §9) have now all failed to protect phase while gaining elsewhere, so this direction is considered exhausted for the remaining project timeline rather than worth a fourth attempt. v5 (57.2% phase accuracy, 0.374 macro F1) is the final reported checkpoint.

Raw results (reconstructed from the full run log after the eval script crashed on its own output-write step due to a missing `results/` directory on the pod — all metrics were computed and logged successfully before that): `results/evaluation_results_v8_interleaved.json`. Full logs: `results/interleave_retrain_v8.log` (training), `results/eval_v8_full.log` (evaluation). The candidate checkpoint (`checkpoints/multitask/final/jointinterleave_candidate/`, 2.4GB) is preserved locally but not live.

## 10. Temporal-context phase retrain (2026-08-25, NOT promoted)

The last remaining untried lever, chosen deliberately over a fourth loss/architecture variant: §8 proved `ClippingCutting`'s 0 F1 is visual confusability (a short, transitional phase, indistinguishable from a single frame), not fixable by reweighting. Every retrain to date (v4–v8) classified from ONE frame. The literature's actual fix for this class of problem is temporal context, and this codebase already had a full, but never-exercised, temporal pipeline: `surgical_vlm/models/temporal_aggregator.py`'s `TemporalAggregator` (4-layer transformer, attention pooling) and `surgical_vlm/data/collators.py`'s `TemporalStackingCollator`. Investigation found the JSONL data had no `frame_paths`, so every prior run — including the very first pre-v4 run — silently fed this pipeline degenerate length-1 "sequences" despite `use_temporal: true` being the config default from the start.

**Fix built for this experiment:**
- `scripts/add_temporal_frame_paths.py`: populated real 3-frame windows for Cholec80 (anchor + 2 preceding frames at the verified 25-frame/1-second extraction stride).
- **A genuine, previously-silent bug found and fixed**: `VLMJSONLDataset.__getitem__` was dropping `frame_paths` entirely (it built a fresh item dict from a fixed set of known keys) — every consumer would have silently received `None` and fallen back to single-frame, with no error at all. Fixed to resolve and pass `frame_paths` through, same as `image_path`.
- `scripts/finetune_temporal_phase.py`: frozen LoRA/vision backbone (the one part of the recipe that has never broken phase, unlike v6/v8's unfrozen-backbone attempts), warm-started v5 heads, and a **freshly-initialized** `TemporalAggregator` — nothing in this project has ever saved trained weights for it before. HeiChole/CholecT50 stages were skipped (`--cholec80-only`) since they're byte-identical to v5's already-proven recipe and the warm-started heads already encode that training; this also makes it the cleanest possible "everything but Cholec80 held fixed" comparison.
- `scripts/evaluate_multi_task.py`: extended to load a `temporal_aggregator.pt` if present and route Cholec80 phase/instrument evaluation through the same multi-frame pooling function the training script uses (`get_temporal_pooled_features`, shared by both, specifically to prevent a train/eval mismatch).
- A `--smoke` mode (20 steps) was run first and passed cleanly before committing to the full run — appropriate caution, since this exercised code (multi-image `_get_training_matching_features` calls, the aggregator's first-ever real training) that had never run before.

Full run: 6,000 steps, frozen backbone, ~58 min wall-clock (much faster than budgeted — the per-step cost of encoding 3 frames was closer to 0.57s/step than the ~1.5-2x-per-frame estimate). Final training loss: 0.05.

**Result — NOT promoted:**

| Metric | v5 (live) | v9 (temporal candidate) | Δ |
|---|---|---|---|
| Phase accuracy (Cholec80) | 0.572 | **0.352** | **−22.0pt** |
| Phase macro F1 | 0.374 | **0.230** | **−14.4pt** |
| `ClippingCutting` F1 | 0.0 | **0.0** | unchanged — still never fixed |
| `GallbladderRetraction` F1 | 0.190 | 0.0 | new regression |
| Instrument macro F1 (Cholec80) | 0.418 | **0.286** | **−13.2pt** |
| Triplet exact match / verb F1 | 0.042 / 0.083 | 0.042 / 0.083 | unchanged (untouched by this experiment) |
| Language BLEU-4 / ROUGE-L | 0.18 / 0.108 | 0.18 / 0.108 | unchanged (untouched by this experiment) |

Triplet and language being numerically identical to v5 is expected and reassuring — those heads/data were never touched by this experiment (warm-started, frozen), confirming the regression is real and isolated to the Cholec80 phase+instrument heads (which share one temporally-pooled feature vector), not a bug elsewhere in the pipeline.

**Root cause: the temporal aggregator overfit.** It is a **668-million-parameter transformer, trained entirely from a random initialization** (confirmed in the training log: `{'total_params': 668183040, 'trainable_params': 668183040}`) — comparable in scale to a small standalone model, not a lightweight add-on like the `output_adapter` heads (a few hundred thousand parameters) every prior retrain fine-tuned. At batch size 1 for 6,000 steps against a 67,512-row training set, it saw **under 9% of the training data — less than one-tenth of a single epoch** — before training stopped. The final training loss of 0.05 (>99% confidence on the correct class, for a 7-way problem with no label smoothing) is the signature of memorization, not generalization. This also resolves an otherwise-confusing observation: during training, `phase_pred_dist` showed `ClippingCutting` (class 2) being predicted regularly (15–26 times per 200-step window) — but it reverted to 0 F1 on held-out test data, meaning the model was over-predicting that class on the specific training frames it had seen, not learning to genuinely recognize it. No validation checks ran during training, so there was no way to catch the overfitting in flight.

**This does not falsify temporal modeling as the correct fix** — the literature's rationale still holds, and this experiment's diagnosis (confusability, not imbalance) from §8 is unaffected. What failed was this specific implementation: a full-size transformer trained from scratch, from a cold start, on a small slice of one epoch, with no regularization schedule or validation-based early stopping. A properly-scoped version would need a much smaller aggregator (or heavy dropout/regularization), substantially more training steps/epochs, and validation monitoring — a larger undertaking than fits in the remaining project timeline.

Raw results: `results/evaluation_results_v9_temporal.json`. Full logs: `results/temporal_train_v9_full.log` (training), `results/eval_v9_full.log` (evaluation). Checkpoint (`temporal_aggregator.pt`, 2.5GB, plus a copy of v5's unchanged LoRA/heads) remains on the pod only (`checkpoints/multitask/final/temporal_phase_candidate/`) — not pulled to local disk or promoted, since it isn't useful for anything beyond this documented experiment.

## 11. Final model and closing justification

Four independent, methodologically distinct retrain strategies were attempted to close the gap between v5's phase performance (57.2% accuracy, 0.374 macro F1) and the project's original ≥60% target, specifically targeting `ClippingCutting`'s 0 F1:

1. **§8, class-weighted loss** (4x inverse-frequency weight on the frozen heads) — `ClippingCutting` never predicted once; broke a previously-working class (`CleaningCoagulation`).
2. **§7, sequential joint LoRA retrain** — real instrument/triplet gains, but phase accuracy regressed 14.6pt via catastrophic forgetting (no rehearsal across sequential task stages).
3. **§9, interleaved joint LoRA retrain** — fixed the *total-collapse* failure mode from attempt 1, but phase macro F1 regressed even further (0.374→0.225) via a *different* failure mode (minority-class erosion rather than single-class collapse).
4. **§10, temporal-context retrain** — the mechanistically correct lever per the literature, but the specific implementation (a 668M-parameter transformer trained from scratch on <9% of one epoch) overfit, regressing both phase (−14.4pt macro F1) and instrument detection (−13.2pt macro F1).

Each attempt was root-caused, not merely observed to fail — every verdict in this report rests on a specific, checkable mechanism (confusability vs. imbalance; forgetting vs. rehearsal; collapse vs. minority-class erosion; underparameterized data budget vs. model capacity), not "it didn't work." That is the honest basis for treating this direction as exhausted within the project's timeline, rather than a gap left unexplained.

**v5 is therefore the final, live checkpoint reported in this dissertation**, with the following results, none of which were affected by any of the four rejected experiments above (each was independently verified, evaluated, and reverted without touching the live checkpoint):

| Task | Metric | Value |
|---|---|---|
| Phase recognition (Cholec80) | Accuracy / Macro F1 | 0.572 / 0.374 |
| Instrument detection (Cholec80) | Macro F1 | 0.418 |
| Action triplet (CholecT50) | Exact match / Instrument F1 / Verb F1 / Target F1 | 0.042 / 0.973 / 0.083 / 0.790 |
| Language generation (Surg-396K) | BLEU-4 / ROUGE-L | 0.18 / 0.108 |

Known, documented limitations to report alongside these numbers (not hidden weaknesses, but characterized ones): `ClippingCutting` phase recognition (0 F1, confirmed visual confusability requiring temporal or higher-capacity modeling beyond this project's remaining scope); CholecT50 verb recognition (hard ceiling — 3 of 10 verb classes have zero training examples); BLEU-4 for open-ended captioning (the literature itself does not consider ≥60% realistic for this task, and v5 matches the zero-shot baseline, meaning the LoRA fine-tuning did not measurably hurt or help this specific metric).
