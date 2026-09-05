# SurgicalVLM_Robotics
**COMP702 M.Sc. Project | Kamal Raj Vendi | University of Liverpool**
Implementation of a **Surgical Vision-Language Model** for video understanding, oriented toward future robotic surgical assistance.

- Frozen **Qwen2.5-VL-7B-Instruct** base with **LoRA fine-tuning** (target rank r=32).
- Image → structured JSON with `phase`, `tools`, and natural-language `description`.
- Multi-task training on a unified JSONL format over **4 datasets** (Surg-396K, Cholec80, CholecT50, HeiChole).
- Trained on RunPod cloud (A100 80GB GPU Pods; shared Network Volume at `/runpod-volume`); interactive demo served from a RunPod pod via Gradio (`share=True`).

---

## Project Structure
```
SurgicalVLM_Robotics/
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data/            # splits.yaml
│   ├── models/
│   ├── training/        # runpod_7b_config, lora_config, eval_local_transfer_config
│   └── eval/
├── data/
│   ├── raw/             # Surg-396K, Cholec80, HeiChole, CholecT50 (raw)
│   ├── frames/          # extracted frames (cholec80/, heichole/, cholect50/)
│   ├── processed/vlm_jsonl/  # unified JSONL instruction files (13 files)
│   └── splits/
├── surgical_vlm/
│   ├── data/            # vlm_dataset, split_manager, collators
│   ├── models/          # surgical_vlm, qwen3_vl, base_vlm, temporal_aggregator
│   ├── training/        # trainer, loss_functions, lora_setup, output_adapter, label_mapping
│   └── evaluation/       # phase, language, triplet, action, grounding metrics
├── scripts/
│   ├── extract_frames.py
│   ├── convert_to_jsonl.py
│   ├── train_multitask.py     # multi-task training entry point
│   ├── finetune_heads_only.py # heads-only recovery/reweighting retrain (produced the final checkpoint)
│   ├── evaluate_multi_task.py
│   ├── evaluate_baselines.py
│   ├── evaluate_zeroshot.py
│   └── check_data_bundle.py   # used by runpod/setup.sh to verify the data bundle
├── web/
│   ├── gradio_app.py        # Gradio dashboard (share=True for public demo)
│   └── video_processor.py   # VideoAnalyzer
├── runpod/
│   ├── Dockerfile           # trains/evals image (Qwen2.5-VL-7B; repo at /workspace)
│   ├── setup.sh             # one-time volume bootstrap (extract + verify + auto frame-extraction)
│   └── train.sh             # full run (default) or HeiChole LR sweep via --lr
├── checkpoints/
├── docs/
├── results/
└── tests/
```

---

## Quick Start

### 1. Install
```bash
git clone <repo> && cd SurgicalVLM_Robotics
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data (local)
```bash
python scripts/extract_frames.py --dataset all --strategy uniform --fps 1 --workers 4
python scripts/convert_to_jsonl.py --data-root data/raw --output-dir data/processed/vlm_jsonl --splits train val test
```

### 3. Train (RunPod A100 GPU Pod)
```bash
# 0. Build & push the image once, attach a Network Volume, upload data.tar.
# 1. On the pod (repo at /workspace), bootstrap the volume (extract + verify):
bash runpod/setup.sh
# 2. Full 7B multitask run:
bash runpod/train.sh
```

### 4. Evaluate
```bash
python scripts/evaluate_multi_task.py --checkpoint /runpod-volume/checkpoints/multitask/final \
    --model-config configs/training/runpod_7b_config.yaml \
    --tasks phase instruments heichole triplet language \
    --output results/evaluation_results.json
```

### 5. Demo (RunPod pod — public link)
```bash
python -m web.gradio_app        # share=True → Gradio public URL
python -m web.gradio_app --mock # no-GPU local demo, simulated output
```

### Run tests
```bash
pytest tests/ -v
```

---
## Configuration

**Primary (7B, RunPod A100 80GB):** `configs/training/runpod_7b_config.yaml`
- `model.type: qwen3_vl`, `load_in_4bit: true`, `use_temporal: false`
- LoRA `r: 32` *(target — confirm it is applied on upload)*, `alpha: 32`
- bf16, grad checkpoint, effective batch 16 (BS 4 × acc 4); `output_dir` on the volume at `/runpod-volume/checkpoints/multitask`

**Free tier (`free_tier_config.yaml`):** 4-bit, fp16, effective BS 16 for opportunistic local runs.

---

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| Phase (Cholec80) | Accuracy, Macro/Weighted F1, Transition Accuracy |
| Instruments | mAP@0.5/.75, per-tool F1/PR |
| Action Triplets (CholecT50) | P/R/F1 (instrument/verb/target) |
| Language | BLEU-4, ROUGE-L, BERTScore |
| Skill / Safety(HeiChole) | (as annotated) |

Baselines for Tables: Random, Majority, No-Temporal, **Full**.

---

## Datasets

| Dataset | Tasks | JSONL status |
|---------|-------|--------------|
| **Cholec80** | Phase + Tool | ✅ train/val/test |
| **CholecT50** | Triplets | ✅ train/val/test |
| **HeiChole** | Multi-task | ✅ train/val/test (small test=155) |
| **Surg-396K** | VQA/Instruction | ✅ train/val/test (largest, 166k train) |
---

## Hardware

| Stage | GPU | Notes |
|-------|-----|-------|
| Training | RunPod A100 80GB GPU Pod | bf16, grad checkpoint; volume persists checkpoints/logs |
| Eval | RunPod (same pod or cheap A100) | LoRA adapter only |
| Demo | RunPod (same pod or cheap GPU) | `share=True` |

Local (Ryzen 7 5825U / iGPU) cannot train; use `--mock` for offline UI.

---

## References
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **LoRA**: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- **EndoChat/Surg-396K**: Wang et al., *Grounded multimodal large language model for endoscopic surgery*, MIAA 2025

---

## License
Academic research (COMP702). Datasets under their respective licenses; no clinical data is distributed.

---

## Author
**Kamal Raj Vendi** (201933430)
- University of Liverpool
- School of Computer Science and Informatics
- Supervisor: Dr. Baoru Huang
