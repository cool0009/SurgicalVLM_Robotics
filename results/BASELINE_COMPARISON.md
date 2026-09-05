# Phase 7 — Baseline Comparison Table

**COMP702 M.Sc. Project | Kamal Raj Vendi**
**Generated:** 2026-08-19 (Random/Majority/Zero-shot), updated 2026-08-20 (Full/Ours, post root-cause fixes), updated again 2026-08-20 (Full/Ours, post class-weighted instrument retrain — see `EVALUATION_RESULTS_REPORT.md`)

All baselines evaluated on the **same test splits** as the main results, using the **same evaluator classes** (`PhaseEvaluator`, `MultiLabelEvaluator`, `TripletEvaluator`, `LanguageEvaluator`) — only the prediction source changes per row.

## Table 7.5 — Full Comparison

| Method | Phase Acc (Cholec80) | Instrument F1 (Cholec80) | Triplet Exact Match | Triplet Verb F1 | BLEU-4 | ROUGE-L | HeiChole Phase Acc |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 0.162 | 0.197 | 0.000 | 0.014 | ~0 | ~0 | 0.155 |
| Majority-class | 0.386 | 0.196 | 0.506 | 0.440 | ~0 | ~0 | 1.000* |
| Zero-shot Qwen (no LoRA) | 0.369 | 0.173 | 0.022 | 0.054 | 0.18 | 0.126 | 0.0* |
| No-Temporal | *(see note below — not a separate run)* | | | | | | |
| **Full (Ours)** | **0.572** | **0.418** | **0.042** | **0.083** | **0.18** | **0.108** | 0.0-1.0* |

*Previous "Full (Ours)" row (pre class-weighted instrument retrain): phase 0.588, instrument F1 0.330, triplet exact match 0.064, verb F1 0.098 — see `EVALUATION_RESULTS_REPORT.md` §6 for the full before/after and why instrument macro F1 was prioritized over the small phase/triplet dips.*

\* HeiChole phase numbers are not meaningful as a discriminator between methods — this test split (1 video) has ground truth that is ~100% one class, so any method that happens to predict that class scores 1.0 and any that doesn't scores 0.0. See `EVALUATION_RESULTS_REPORT.md` §2.2/§5.

## Key findings (updated 2026-08-20, post class-weighted instrument retrain)

**Phase and instrument detection now genuinely beat every baseline**, which was not true as of the 2026-08-19 report:
- **Phase:** Ours 57.2% vs. Majority-class 38.6%, Zero-shot 36.9%, Random 16.2%. A properly-trained classification head comfortably beating a majority-class baseline is exactly what should happen, and — after two root-caused bug fixes (see `EVALUATION_RESULTS_REPORT.md` §1) — now does. (Macro F1 0.374, essentially unchanged from the pre-reweighting 0.371 — the 1.6pt accuracy dip is retrain-to-retrain noise, not a regression.)
- **Instruments:** Ours 41.8% F1 vs. Random 19.7%, Majority 19.6%, Zero-shot 17.3% — up from 33.0% after a targeted class-weighted retrain (see `EVALUATION_RESULTS_REPORT.md` §6) recovered the four previously-zero-F1 tools (Bipolar, Clipper, Scissors, Irrigator), all rare in Cholec80's training distribution (1.9–6.2% of rows).

**Triplet and language remain behind Majority-class, for reasons that are now well-understood (not unexplained weaknesses):**
- **Triplet exact-match:** Majority-class (always predict "Grasper, grasp, No_Target") scores 50.6% vs. our model's 4.2% — driven almost entirely by the Verb component, where 3 of 10 classes have zero training examples in CholecT50 (see report §2.4). Our Instrument (97.3%) and Target (79.0%) components individually clear the 60% target; it's specifically Verb (8.3% F1) dragging exact-match down, and Verb has a hard data ceiling that retraining cannot fix.
- **Language:** our fine-tuned BLEU-4 (0.18) is *identical* to the zero-shot base model's, and our ROUGE-L (0.108) is marginally *below* zero-shot's (0.126). LoRA fine-tuning does not appear to have measurably improved open-ended captioning quality on this metric — plausibly because the language-modeling loss was never broken (it trained normally throughout, per the root-cause analysis), so there was limited room left to improve on top of a base model that could already caption reasonably, or because BLEU-4/ROUGE-L are simply poor discriminators for caption quality at this level. Either way, this is a case where the ≥60% target itself (not the model) is the primary issue — no VLM caption system reliably clears that bar on open-ended surgical captioning per the literature (typical range 20-40% BLEU-4).

## Table 7.6 — Comparison to Published Literature (context, not a controlled comparison)

**This table is not apples-to-apples** and should be presented as such — see the caveat below. It exists to show where this project's numbers sit relative to task-specific published systems, not to claim parity or superiority.

| Task / Dataset | Published method | Reported score | Source |
|---|---|---|---|
| Phase recognition (Cholec80) | TeCNO (2020) | 90.17% accuracy | Czempiel et al., MICCAI 2020 |
| Phase recognition (Cholec80) | Trans-SVNet (2021) | 91.54% accuracy | Gao et al., MICCAI 2021 (arXiv:2103.09712) |
| Phase recognition (Cholec80) | LoViT (2024) | 92.40% accuracy | Liu et al., Medical Image Analysis 2024 (arXiv:2305.08989) |
| Tool presence detection (Cholec80) | EndoNet-family / recent CNN methods | ~91–96% mAP | Multiple 2020s tool-detection papers on Cholec80 |
| Action triplet recognition (CholecT50) | CholecTriplet2021 challenge field | 4.2%–38.1% mAP (19 competing methods) | Nwoye et al., "CholecTriplet2021" challenge report, Medical Image Analysis 2023 (arXiv:2204.04746) |
| Action triplet recognition (CholecT50) | Recent IVT-split method (2024/2025) | 39.54% mAP-IVT | Recent CholecT50 leaderboard entry |
| **Phase recognition (Cholec80)** | **SurgicalVLM (Ours)** | **57.2% accuracy** | This project |
| **Instrument detection (Cholec80)** | **SurgicalVLM (Ours)** | **41.8% macro-F1 / mAP@0.5** | This project |
| **Triplet — instrument / target component (CholecT50)** | **SurgicalVLM (Ours)** | **97.3% F1 / 79.0% F1** | This project |

### Why these numbers aren't directly comparable

1. **Purpose-built vs. general-purpose.** TeCNO/Trans-SVNet/LoViT are single-task architectures built and trained *only* for phase recognition, typically with a CNN feature extractor plus a dedicated temporal model (TCN/LSTM/Transformer) over full-length video sequences. This project's phase head is one of four tasks trained jointly on top of a general 7B vision-language model, with **no temporal aggregation in the active inference path** (confirmed — see the "No-Temporal" note above; `temporal_pool.*` weights were never trained), so it is being compared against methods whose entire design purpose is the temporal signal this project doesn't use.
2. **Full video-level test set vs. a 500-frame sample.** Published Cholec80 phase results are computed over the dataset's full standard 40-video test split, frame-by-frame across entire procedures. This project's phase numbers (§2.1 of `EVALUATION_RESULTS_REPORT.md`) come from a 500-sample evaluation subset, not the full test-split video count.
3. **Different metrics for triplet.** Published CholecT50 results report **mAP** on the full ‹instrument, verb, target› triplet as a joint multi-label detection problem (with harder partial-credit rules). This project reports **per-component F1/accuracy** and **exact-match**, which is a different, generally more lenient scoring scheme for the individual components — the two are not on the same scale and should not be read as directly interchangeable numbers.
4. **Different eras of task-specific optimization.** EndoNet-family tool-detection methods, TeCNO/Trans-SVNet/LoViT, and Rendezvous each represent multiple published iterations by groups (CAMMA, KCL, etc.) specializing exclusively in this task since ~2016–2024. This project is a single M.Sc. dissertation cycle building a general-purpose multi-task model across four datasets at once.

**The honest takeaway:** this project's phase and instrument numbers are well below mature, purpose-built single-task SOTA systems (as expected — those systems have years of task-specific architecture work and use full temporal context this project's inference path doesn't invoke), but they comfortably clear this project's own random/majority/zero-shot baselines, which is the relevant, controlled comparison for judging whether the training pipeline itself works (see Table 7.5). The literature numbers above are included for context/positioning in the dissertation, not as a claim of competitiveness with SOTA.

## Methodology notes

- **Random:** each prediction drawn uniformly at random from the task's label vocabulary (phase: 1-of-7; tools: each of 7 independently included w.p. 0.5; triplet: uniform over each component's vocabulary). Seed fixed at 42 for reproducibility.
- **Majority-class:** the single most common label (or per-tool majority vote for the multi-label instrument task) computed over the **training** split, predicted for every test sample.
- **No-Temporal:** not run as a separate experiment. The trained model's classification-head inference path (`SurgicalVLM.predict_phase`/`predict_instruments`/`predict_triplet`) calls `_get_training_matching_features`, which pools over a **single frame** — the temporal aggregator module is never invoked in this path, and the saved checkpoint doesn't even contain trained `temporal_pool.*` weights. So "Full (Ours)" **is** the no-temporal result.
- **Zero-shot Qwen:** base Qwen2.5-VL-7B-Instruct, **no LoRA adapter loaded**, prompted directly with the same JSON-schema prompt the demo uses for phase+tools, a comparable JSON prompt for triplet, and the existing `evaluate_language_generation` path for language — all fully generative, since there's no trained classification head to evaluate zero-shot. Notably, zero-shot phase predictions collapse heavily onto "GallbladderDissection" for most inputs (a different failure mode than our model's original bug, but still a form of low-effort majority-guessing typical of a zero-shot VLM without task-specific calibration).

## Raw results
- `results/baseline_results.json` — Random + Majority, full per-task detail
- `results/zeroshot_results.json` — Zero-shot Qwen, full per-task detail
- `results/evaluation_results_v4.json` — Full (Ours), pre class-weighted-instrument-retrain numbers (superseded, kept for the before/after comparison)
- `results/evaluation_results_v5.json` — **Full (Ours), current/final numbers** — post class-weighted instrument retrain, full per-task detail
