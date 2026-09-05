#!/usr/bin/env python3
"""
Multi-task evaluation script for SurgicalVLM.

Evaluates model on:
- Phase recognition (Cholec80, HeiChole) -> Table 5.1
- Language generation (Surg-396K) -> Table 5.4
- Action triplet recognition (CholecT50) -> Table 5.3
- HeiChole multi-task (Phase + Tool + Action + Skill)
"""

import os
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path (same pattern as train_multitask.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
from tqdm import tqdm
from PIL import Image

from surgical_vlm.models.surgical_vlm import create_surgical_vlm, SURGICAL_JSON_PROMPT
from surgical_vlm.data.vlm_dataset import VLMJSONLDataset
from surgical_vlm.data.split_manager import SplitManager
from surgical_vlm.evaluation.phase_metrics import PhaseEvaluator
from surgical_vlm.evaluation.language_metrics import LanguageEvaluator
from surgical_vlm.evaluation.triplet_metrics import TripletEvaluator
from surgical_vlm.evaluation.action_metrics import ActionEvaluator
from surgical_vlm.training.label_mapping import get_label_mapper
from surgical_vlm.models.temporal_aggregator import create_temporal_aggregator
from surgical_vlm.training.output_adapter import create_output_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default location of the processed VLM JSONL datasets (relative to project
# root). Overridden from the model config's `data.jsonl_dir` in
# run_full_evaluation when that key is present.
_JSONL_DIR = "data/processed/vlm_jsonl"


def load_model(model_config: str, checkpoint: str = "", device: str = "cuda", use_temporal: bool = True):
    """Load SurgicalVLM with optional LoRA checkpoint and temporal aggregator."""
    import yaml
    from pathlib import Path
    
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Support both training config format (model.type) and standalone model config format (model_type)
    if "model" in config and "type" in config["model"]:
        model_type = config["model"]["type"]
        load_in_4bit = config["model"].get("load_in_4bit", True)
        temporal_config = config["model"].get("temporal_config", {})
    else:
        # Standalone model config (e.g., configs/models/qwen2.5_vl_3b.yaml)
        model_type = config.get("model_type", "qwen2.5_vl_3b")
        load_in_4bit = config.get("load_in_4bit", True)
        temporal_config = config.get("temporal_config", {})
    
    vlm = create_surgical_vlm(
        model_type=model_type,
        load_in_4bit=load_in_4bit,
        device=device,
        use_temporal=use_temporal,
        temporal_config=temporal_config if use_temporal else {},
    )
    vlm.load()
    
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        # Check if it's a LoRA adapter (has adapter_config.json)
        if (checkpoint_path / "adapter_config.json").exists():
            # PeftModel.from_pretrained wraps a fresh PeftModel around the base model
            # using the checkpoint's own adapter_config.json, so do NOT call apply_lora
            # here (that would double-wrap via get_peft_model and corrupt the state-dict
            # key prefix). load_lora_checkpoint also merge_and_unloads the adapter.
            vlm.load_lora_checkpoint(checkpoint)
            logger.info(f"Loaded LoRA checkpoint from {checkpoint}")
        else:
            # Full model checkpoint - load state dict directly.
            # Trainer saves model.safetensors; fall back to pytorch_model.bin if present.
            logger.info(f"Loading full model checkpoint from {checkpoint}")
            safetensors_path = checkpoint_path / "model.safetensors"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path), device=device)
            else:
                state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=device)
            vlm.vlm.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded full model checkpoint from {checkpoint}")

        # Load multi-task classification heads (OutputAdapter) saved alongside the checkpoint.
        # The vision encoder runs in bf16 on CUDA, so heads must be cast to bf16 to avoid dtype mismatch.
        adapter_path = checkpoint_path / "output_adapter.pt"
        if adapter_path.exists():
            adapter = create_output_adapter(config.get("output_adapter", {}))
            adapter.load_state_dict(torch.load(adapter_path, map_location=device), strict=False)
            head_map = {
                "_phase_head": adapter.phase_head,
                "_instrument_head": adapter.instrument_head,
                "_action_head": adapter.action_head,
                "_triplet_inst_head": adapter.triplet_inst_head,
                "_triplet_verb_head": adapter.triplet_verb_head,
                "_triplet_target_head": adapter.triplet_target_head,
            }
            for attr, head in head_map.items():
                setattr(vlm, attr, head.to(torch.bfloat16).to(device))
            logger.info(f"Loaded output adapter heads from {adapter_path}")

        # Optional: a trained TemporalAggregator saved alongside the checkpoint
        # (only scripts/finetune_temporal_phase.py produces this, 2026-08-25 --
        # every earlier checkpoint has none, and `vlm.temporal_aggregator`
        # created by create_surgical_vlm() above is randomly initialized).
        # Attached as a NEW attribute rather than overwriting vlm.temporal_aggregator
        # so existing single-frame callers (predict_phase/predict_instruments with
        # 1 frame) are completely unaffected -- only code that explicitly checks
        # for `_trained_temporal_aggregator` uses it.
        temporal_agg_path = checkpoint_path / "temporal_aggregator.pt"
        if temporal_agg_path.exists():
            trained_temporal_aggregator = create_temporal_aggregator({
                "hidden_dim": 3584, "strategy": "attention", "num_heads": 8,
                "num_layers": 4, "dropout": 0.1, "max_frames": 8, "use_pos_encoding": True,
            }).to(device)
            trained_temporal_aggregator.load_state_dict(
                torch.load(temporal_agg_path, map_location=device), strict=True
            )
            trained_temporal_aggregator.eval()
            vlm._trained_temporal_aggregator = trained_temporal_aggregator
            logger.info(f"Loaded trained temporal aggregator from {temporal_agg_path}")

    vlm.to(device)
    vlm.eval_mode()
    return vlm


def get_temporal_pooled_features(vlm, temporal_aggregator, frame_paths, prompt, image_root,
                                  num_frames=3, train_aggregator=False):
    """Shared by scripts/finetune_temporal_phase.py (training) and the
    evaluate_phase_recognition/evaluate_instrument_detection functions below
    (eval), so train and eval feature extraction can never silently drift
    apart -- exactly the kind of train/eval mismatch that caused real bugs
    earlier in this project (see EVALUATION_RESULTS_REPORT.md Sec 1/6).

    For ONE sample: loads its frame_paths (anchor frame first), encodes each
    frame independently via vlm._get_training_matching_features (the same
    per-frame, no_grad, prompt-conditioned feature extractor every
    single-frame retrain in this project has used -- so this is genuinely
    "the same features, now pooled over time" rather than a different
    feature pipeline), stacks into [1, T, D], and pools via
    temporal_aggregator. Returns None if the anchor frame fails to load.

    `train_aggregator`: the per-frame backbone features are ALWAYS extracted
    no_grad (the vision/LM backbone is frozen throughout this project's
    retrains -- see Sec 6-9's "backbone stays frozen" pattern), but the
    temporal_aggregator pooling step itself must run WITH grad when training
    it (finetune_temporal_phase.py passes True) and WITHOUT grad at eval time
    (the default, False) -- getting this backwards would either silently
    stop the aggregator from ever training, or waste memory/compute at eval.
    """
    imgs = []
    for p in (frame_paths or [])[:num_frames]:
        pth = Path(p)
        full = pth if pth.is_absolute() else Path(image_root) / p
        try:
            img = Image.open(full).convert("RGB")
        except Exception:
            img = None
        if img is not None:
            imgs.append(img)
    if not imgs:
        return None

    per_frame = vlm._get_training_matching_features(imgs, prompt)  # [T, D], always no_grad internally
    frame_features = per_frame.unsqueeze(0).to(torch.float32)  # [1, T, D]
    mask = torch.ones(1, frame_features.shape[1], dtype=torch.bool, device=frame_features.device)
    if train_aggregator:
        pooled = temporal_aggregator(frame_features, mask=mask, pool=True)  # [1, D], grad flows
    else:
        with torch.no_grad():
            pooled = temporal_aggregator(frame_features, mask=mask, pool=True)  # [1, D]
    return pooled.to(torch.bfloat16)


def _get_jsonl_path(dataset_name: str, split: str) -> str:
    """Get the JSONL file path for a dataset and split.
    
    Files follow pattern: {dataset}_{task}_vlm_{split}.jsonl
    """
    dataset_file_map = {
        "cholec80": "cholec80_phases_vlm",
        "heichole": "heichole_multitask_vlm",
        "cholect50": "cholect50_triplets_vlm",
        "surg396k": "surg396k_vlm",
    }
    base = dataset_file_map.get(dataset_name)
    if base:
        return f"{_JSONL_DIR}/{base}_{split}.jsonl"
    return ""


def _smooth_labels_1d(labels, window_size: int = 5):
    """Sliding-window majority-vote smoothing of a 1D label sequence.

    Window is centered on each position and clipped at sequence boundaries so
    every window holds exactly ``window_size`` labels. ``window_size`` of 0 or 1
    is a no-op.
    """
    if window_size <= 1:
        return list(labels)
    n = len(labels)
    if n == 0:
        return []
    from collections import Counter
    half = window_size // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + window_size)
        window = labels[lo:hi]
        smoothed.append(Counter(window).most_common(1)[0][0])
    return smoothed


def evaluate_phase_recognition(vlm, dataset_name: str, split: str = "test", max_samples: int = 500, image_root: Optional[str] = None, smooth_window: int = 0):
    """Evaluate surgical phase recognition using classification heads (logits)."""
    logger.info(f"Evaluating phase recognition on {dataset_name} ({split})...")
    
    jsonl_path = _get_jsonl_path(dataset_name, split)
    if not jsonl_path or not Path(jsonl_path).exists():
        logger.warning(f"Dataset {dataset_name} not found at {jsonl_path}")
        return {}
    
    sm = SplitManager()
    dataset = VLMJSONLDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        split_manager=sm,
        max_samples=max_samples,
        image_root=image_root,
    )
    
    phase_names = sm.get_phase_names(dataset_name)
    evaluator = PhaseEvaluator(phase_names=phase_names)
    
    # Use label mapper for consistent indexing
    label_mapper = get_label_mapper()
    
    # Accumulate per-video predictions so temporal smoothing never crosses video boundaries.
    per_video = {}

    temporal_aggregator = getattr(vlm, "_trained_temporal_aggregator", None)

    for item in tqdm(dataset, desc=f"Phase eval {dataset_name}"):
        convs = item["conversations"]
        prompt = convs[0]["value"]
        gt_answer = convs[1]["value"]

        frame_paths = item.get("frame_paths")
        if temporal_aggregator is not None and frame_paths and len(frame_paths) > 1:
            # Temporal path: real multi-frame window, pooled via the trained
            # TemporalAggregator (only populated for Cholec80 -- see
            # scripts/add_temporal_frame_paths.py / finetune_temporal_phase.py).
            pooled = get_temporal_pooled_features(
                vlm, temporal_aggregator, frame_paths, prompt, image_root or "",
            )
            if pooled is None:
                logger.warning(f"Failed to load temporal window for {item.get('image_path')}")
                continue
            phase_logits = vlm._phase_head(pooled)
        else:
            try:
                img = Image.open(item["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image: {e}")
                continue
            # Use the classification head for logits
            phase_logits = vlm.predict_phase([img], prompt=prompt)
        pred_phase_idx = phase_logits.argmax(dim=-1).item()
        
        # Extract ground truth phase
        try:
            gt_json = json.loads(gt_answer)
            gt_phase = gt_json.get("phase", "Unknown")
        except json.JSONDecodeError:
            gt_phase = "Unknown"
        
        video_id = item.get("video_id", "unknown")
        frame_idx = item.get("frame_idx", 0)
        per_video.setdefault(video_id, []).append((frame_idx, pred_phase_idx, gt_phase))
    
    # Per-video temporal smoothing, then batch-add to the evaluator.
    for video_id, samples in per_video.items():
        samples.sort(key=lambda s: s[0])
        pred_idxs = [s[1] for s in samples]
        gt_phases = [s[2] for s in samples]
        if smooth_window > 1:
            pred_idxs = _smooth_labels_1d(pred_idxs, window_size=smooth_window)
        pred_phases = [label_mapper.phase_id_to_text(i) for i in pred_idxs]
        evaluator.add_batch(pred_phases, gt_phases)
    
    results = evaluator.compute()
    logger.info(f"Phase Recognition Results ({dataset_name}): {json.dumps(results, indent=2)}")
    return results


def evaluate_phase_recognition_generative(vlm, dataset_name: str, split: str = "test", max_samples: int = 500, image_root: Optional[str] = None):
    """Evaluate surgical phase recognition using generative text (original method)."""
    logger.info(f"Evaluating phase recognition (generative) on {dataset_name} ({split})...")
    
    jsonl_path = _get_jsonl_path(dataset_name, split)
    if not jsonl_path or not Path(jsonl_path).exists():
        logger.warning(f"Dataset {dataset_name} not found at {jsonl_path}")
        return {}
    
    sm = SplitManager()
    dataset = VLMJSONLDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        split_manager=sm,
        max_samples=max_samples,
        image_root=image_root,
    )
    
    phase_names = sm.get_phase_names(dataset_name)
    evaluator = PhaseEvaluator(phase_names=phase_names)
    
    for item in tqdm(dataset, desc=f"Phase eval (gen) {dataset_name}"):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            continue
        
        convs = item["conversations"]
        prompt = convs[0]["value"]
        gt_answer = convs[1]["value"]
        
        pred_answer = vlm.generate_single(img, SURGICAL_JSON_PROMPT, max_new_tokens=64)
        
        # Extract phase from JSON response (generate_single returns a dict)
        try:
            pred_json = pred_answer if isinstance(pred_answer, dict) else json.loads(pred_answer)
            pred_phase = pred_json.get("phase", "Unknown")
        except (json.JSONDecodeError, TypeError):
            pred_phase = "Unknown"
        
        # Extract ground truth phase
        try:
            gt_json = json.loads(gt_answer)
            gt_phase = gt_json.get("phase", "Unknown")
        except json.JSONDecodeError:
            gt_phase = "Unknown"
        
        evaluator.add_batch([pred_phase], [gt_phase])
    
    results = evaluator.compute()
    logger.info(f"Phase Recognition Results - Generative ({dataset_name}): {json.dumps(results, indent=2)}")
    return results


def evaluate_instrument_detection(vlm, dataset_name: str, split: str = "test", max_samples: int = 500, image_root: Optional[str] = None):
    """Evaluate surgical instrument detection using classification heads (logits)."""
    logger.info(f"Evaluating instrument detection on {dataset_name} ({split})...")
    
    jsonl_path = _get_jsonl_path(dataset_name, split)
    if not jsonl_path or not Path(jsonl_path).exists():
        logger.warning(f"Dataset {dataset_name} not found at {jsonl_path}")
        return {}
    
    sm = SplitManager()
    dataset = VLMJSONLDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        split_manager=sm,
        max_samples=max_samples,
        image_root=image_root,
    )
    
    tool_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    
    from surgical_vlm.evaluation.phase_metrics import MultiLabelEvaluator
    evaluator = MultiLabelEvaluator(tool_names)
    label_mapper = get_label_mapper()
    
    temporal_aggregator = getattr(vlm, "_trained_temporal_aggregator", None)

    for item in tqdm(dataset, desc=f"Instrument eval {dataset_name}"):
        convs = item["conversations"]
        prompt = convs[0]["value"]
        gt_answer = convs[1]["value"]

        frame_paths = item.get("frame_paths")
        if temporal_aggregator is not None and frame_paths and len(frame_paths) > 1:
            pooled = get_temporal_pooled_features(
                vlm, temporal_aggregator, frame_paths, prompt, image_root or "",
            )
            if pooled is None:
                logger.warning(f"Failed to load temporal window for {item.get('image_path')}")
                continue
            inst_logits = vlm._instrument_head(pooled)
        else:
            try:
                img = Image.open(item["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image: {e}")
                continue
            # Use the classification head for logits
            inst_logits = vlm.predict_instruments([img], prompt=prompt)
        pred_probs = torch.sigmoid(inst_logits).squeeze(0)
        pred_tools = [tool_names[i] for i, p in enumerate(pred_probs) if p > 0.5]
        
        # Extract ground truth tools
        try:
            gt_json = json.loads(gt_answer)
            gt_tools = gt_json.get("tools", [])
        except json.JSONDecodeError:
            gt_tools = []
        
        evaluator.add_batch([pred_tools], [gt_tools])
    
    results = evaluator.compute()
    logger.info(f"Instrument Detection Results ({dataset_name}): {json.dumps(results, indent=2)}")
    return results


def evaluate_heiChole_multitask(vlm, split: str = "test", max_samples: int = 500, image_root: Optional[str] = None, smooth_window: int = 0):
    """Evaluate HeiChole multi-task: Phase, Tool, Action, Skill."""
    logger.info(f"Evaluating HeiChole multi-task ({split})...")
    
    jsonl_path = _get_jsonl_path("heichole", split)
    if not jsonl_path or not Path(jsonl_path).exists():
        logger.warning(f"Dataset not found: {jsonl_path}")
        return {}
    
    sm = SplitManager()
    dataset = VLMJSONLDataset(
        jsonl_path=jsonl_path,
        dataset_name="heichole",
        split=split,
        split_manager=sm,
        max_samples=max_samples,
        image_root=image_root,
    )
    
    # Phase evaluator
    phase_names = sm.get_phase_names("heichole")
    phase_evaluator = PhaseEvaluator(phase_names=phase_names)
    
    # Tool evaluator
    tool_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    from surgical_vlm.evaluation.phase_metrics import MultiLabelEvaluator
    tool_evaluator = MultiLabelEvaluator(tool_names)
    
    # Action evaluator
    action_names = ["No_Action", "Dissect", "Grasp", "Clip", "Cut", 
                    "Coagulate", "Retract", "Irrigate", "Pack", "Clean"]
    action_evaluator = ActionEvaluator(action_names)
    
    label_mapper = get_label_mapper()
    
    # Accumulate per-video phase predictions so temporal smoothing never
    # crosses video boundaries.
    phase_samples = {}
    
    for item in tqdm(dataset, desc=f"HeiChole multi-task eval"):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            continue
        
        convs = item["conversations"]
        prompt = convs[0]["value"]
        gt_answer = convs[1]["value"]
        
        # Use classification heads for phase, tools, action
        phase_logits = vlm.predict_phase([img], prompt=prompt)
        pred_phase_idx = phase_logits.argmax(dim=-1).item()
        pred_phase = label_mapper.phase_id_to_text(pred_phase_idx)
        
        inst_logits = vlm.predict_instruments([img], prompt=prompt)
        pred_probs = torch.sigmoid(inst_logits).squeeze(0)
        pred_tools = [tool_names[i] for i, p in enumerate(pred_probs) if p > 0.5]
        
        action_logits = vlm.predict_action([img], prompt=prompt)
        pred_action_idx = action_logits.argmax(dim=-1).item()
        pred_action = label_mapper.action_id_to_text(pred_action_idx)
        
        # Parse ground truth
        try:
            gt_json = json.loads(gt_answer)
        except json.JSONDecodeError:
            continue
        
        gt_phase = gt_json.get("phase", "Unknown")
        gt_tools = gt_json.get("tools", [])
        gt_action = gt_json.get("action", "No_Action")
        
        # Phase (accumulate per-video for temporal smoothing)
        if pred_phase in phase_names and gt_phase in phase_names:
            video_id = item.get("video_id", "unknown")
            frame_idx = item.get("frame_idx", 0)
            phase_samples.setdefault(video_id, []).append((frame_idx, pred_phase, gt_phase))
        
        # Tools (multi-label)
        tool_evaluator.add_batch([pred_tools], [gt_tools])
        
        # Action
        action_evaluator.add_batch_strings([pred_action], [gt_action])
        
        
    
    # Per-video temporal smoothing for phase, then batch-add to the evaluator.
    for video_id, samples in phase_samples.items():
        samples.sort(key=lambda s: s[0])
        pred_phases = [s[1] for s in samples]
        gt_phases = [s[2] for s in samples]
        if smooth_window > 1:
            pred_idxs = [phase_names.index(p) if p in phase_names else -1 for p in pred_phases]
            pred_idxs = _smooth_labels_1d(pred_idxs, window_size=smooth_window)
            pred_phases = [phase_names[i] if 0 <= i < len(phase_names) else pred_phases[k] for k, i in enumerate(pred_idxs)]
        phase_evaluator.add_batch(pred_phases, gt_phases)
    
    results = {
        "phase": phase_evaluator.compute(),
        "tools": tool_evaluator.compute(),
        "action": action_evaluator.compute(),
    }
    
    logger.info(f"HeiChole Multi-task Results: {json.dumps(results, indent=2)}")
    return results

def evaluate_language_generation(vlm, dataset_name: str = "surg396k", split: str = "test", max_samples: int = 200, image_root: Optional[str] = None):
    """Evaluate natural language generation quality."""
    logger.info(f"Evaluating language generation on {dataset_name} ({split})...")
    
    jsonl_path = _get_jsonl_path(dataset_name, split)
    if not jsonl_path or not Path(jsonl_path).exists():
        logger.warning(f"Dataset not found: {jsonl_path}")
        return {}
    
    sm = SplitManager()
    dataset = VLMJSONLDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        split_manager=sm,
        max_samples=max_samples,
        image_root=image_root,
    )
    
    evaluator = LanguageEvaluator()
    
    for item in tqdm(dataset, desc=f"Language eval {dataset_name}"):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            continue
        
        convs = item["conversations"]
        # Use the JSON-instructed surgical prompt so outputs are parseable
        prompt = SURGICAL_JSON_PROMPT
        
        # Ground truth is the assistant response
        gt_answer = ""
        for conv in convs:
            if conv.get("from") in ["gpt", "assistant"]:
                gt_answer = conv.get("value", "")
                break
        
        # generate_single returns {phase, tools, description}; compare description text
        pred = vlm.generate_single(img, prompt, max_new_tokens=256)
        pred_answer = pred.get("description", "") if isinstance(pred, dict) else str(pred)
        
        evaluator.add_batch([pred_answer], [gt_answer])
    
    results = evaluator.compute()
    logger.info(f"Language Generation Results: {json.dumps(results, indent=2)}")
    return results


def evaluate_triplets(vlm, dataset_name: str = "cholect50", split: str = "test", max_samples: int = 200, image_root: Optional[str] = None):
    """Evaluate surgical action triplet recognition (instrument, action, target)."""
    logger.info(f"Evaluating action triplets on {dataset_name} ({split})...")
    
    jsonl_path = _get_jsonl_path(dataset_name, split)
    if not jsonl_path or not Path(jsonl_path).exists():
        logger.warning(f"Dataset not found: {jsonl_path}")
        return {}
    
    sm = SplitManager()
    dataset = VLMJSONLDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        split_manager=sm,
        max_samples=max_samples,
        image_root=image_root,
    )
    
    evaluator = TripletEvaluator()
    
    for item in tqdm(dataset, desc=f"Triplet eval {dataset_name}"):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            continue
        
        convs = item["conversations"]
        prompt = convs[0]["value"]
        gt_answer = convs[1]["value"]
        
        # Use triplet prediction head
        triplet_logits = vlm.predict_triplet([img], prompt=prompt)
        
        # Get predictions from logits
        pred_inst = triplet_logits['instrument'].argmax(dim=-1).item()
        pred_verb = triplet_logits['verb'].argmax(dim=-1).item()
        pred_target = triplet_logits['target'].argmax(dim=-1).item()
        
        label_mapper = get_label_mapper()
        pred_triplet = label_mapper.triplet_ids_to_text(pred_inst, pred_verb, pred_target)
        
        gt_triplet = None
        try:
            gt_json = json.loads(gt_answer)
            gt_triplet = f"{gt_json['instrument']}, {gt_json['verb']}, {gt_json['target']}"
        except Exception:
            gt_triplet = None
        
        if gt_triplet is None:
            continue
        evaluator.add_batch_strings([pred_triplet], [gt_triplet])
    
    results = evaluator.compute()
    logger.info(f"Triplet Results ({dataset_name}): {json.dumps(results, indent=2)}")
    return results

def format_dissertation_results(all_results: dict) -> dict:
    """Format results for dissertation tables 5.1-5.4."""
    formatted = {}
    
    # Table 5.1: Phase recognition accuracy (Cholec80 test)
    if "phase_cholec80" in all_results:
        r = all_results["phase_cholec80"]
        formatted["table_5_1"] = {
            "Method": "SurgicalVLM (Ours)",
            "Accuracy": r.get("accuracy", 0),
            "F1": r.get("macro_f1", r.get("weighted_f1", 0)),
        }
    
    # Table 5.2: Instrument detection mAP (HeiChole test)
    if "instruments_heichole" in all_results:
        r = all_results["instruments_heichole"]
        formatted["table_5_2"] = {
            "Method": "SurgicalVLM (Ours)",
            "mAP@0.5": r.get("map_05", 0),
            "F1": r.get("macro_f1", r.get("weighted_f1", 0)),
        }
    
    # Table 5.3: Action triplet recognition (CholecT50 test)
    if "triplet_cholect50" in all_results:
        r = all_results["triplet_cholect50"]
        formatted["table_5_3"] = {
            "Method": "SurgicalVLM (Ours)",
            "Exact Match": r.get("exact_match_accuracy", 0),
            "Instrument F1": r.get("instrument_f1", 0),
            "Verb F1": r.get("verb_f1", 0),
            "Target F1": r.get("target_f1", 0),
        }
    
    # Table 5.4: Language generation quality (Surg396K test)
    if "language_surg396k" in all_results:
        r = all_results["language_surg396k"]
        formatted["table_5_4"] = {
            "Method": "SurgicalVLM (Ours)",
            "BLEU-4": r.get("bleu", r.get("bleu_4", 0)),
            "ROUGE-L": r.get("rouge_l", 0),
        }
    
    return formatted


def run_full_evaluation(
    model_config: str,
    checkpoint: str = "",
    device: str = "cuda",
    tasks: list = None,
    max_samples: int = 500,
    image_root: str = "/runpod-volume/data",
    output_file: str = None,
    split: str = "test",
    smooth_window: int = 0
):
    """Run comprehensive multi-task evaluation."""
    
    if tasks is None:
        tasks = ["phase", "instruments", "heichole", "language", "triplet"]
    
    logger.info("=" * 60)
    logger.info("SURGICALVLM MULTI-TASK EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Max samples per task: {max_samples}")
    logger.info(f"Split: {split}")
    logger.info(f"Smooth window: {smooth_window}")

    # Resolve the JSONL dataset directory from the model config, falling back
    # to the module-level default when the config has no data.jsonl_dir key.
    global _JSONL_DIR
    try:
        import yaml
        with open(model_config, "r") as f:
            cfg = yaml.safe_load(f)
        configured = cfg.get("data", {}).get("jsonl_dir")
        if configured:
            _JSONL_DIR = configured
            logger.info(f"JSONL dir (from config): {_JSONL_DIR}")
        else:
            logger.info(f"JSONL dir (default): {_JSONL_DIR}")
    except Exception as e:
        logger.warning(f"Could not read jsonl_dir from config ({e}); using default {_JSONL_DIR}")

    # Load model
    vlm = load_model(model_config, checkpoint, device)
    
    all_results = {}
    
    # Phase recognition
    if "phase" in tasks:
        for dataset in ["cholec80", "heichole"]:
            try:
                results = evaluate_phase_recognition(vlm, dataset, split=split, max_samples=max_samples, image_root=image_root, smooth_window=smooth_window)
                all_results[f"phase_{dataset}"] = results
            except Exception as e:
                logger.error(f"Phase evaluation failed for {dataset}: {e}")
    
    # Instrument detection
    if "instruments" in tasks:
        for dataset in ["cholec80", "heichole"]:
            try:
                results = evaluate_instrument_detection(vlm, dataset, split=split, max_samples=max_samples, image_root=image_root)
                all_results[f"instruments_{dataset}"] = results
            except Exception as e:
                logger.error(f"Instrument evaluation failed for {dataset}: {e}")
    
    # HeiChole multi-task
    if "heichole" in tasks:
        try:
            results = evaluate_heiChole_multitask(vlm, split=split, max_samples=max_samples, image_root=image_root, smooth_window=smooth_window)
            all_results["heichole_multitask"] = results
        except Exception as e:
            logger.error(f"HeiChole multi-task evaluation failed: {e}")
    
    # Language generation
    if "language" in tasks:
        try:
            results = evaluate_language_generation(vlm, split=split, max_samples=max_samples, image_root=image_root)
            all_results["language_surg396k"] = results
        except Exception as e:
            logger.error(f"Language evaluation failed: {e}")
    
    # Triplet recognition
    if "triplet" in tasks:
        try:
            results = evaluate_triplets(vlm, split=split, max_samples=max_samples, image_root=image_root)
            all_results["triplet_cholect50"] = results
        except Exception as e:
            logger.error(f"Triplet evaluation failed: {e}")
    
    # Format for dissertation tables
    dissertation_tables = format_dissertation_results(all_results)
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "detailed_results": all_results,
                "dissertation_tables": dissertation_tables
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for task, results in all_results.items():
        if isinstance(results, dict) and "error" not in results:
            if "accuracy" in results:
                logger.info(f"  {task}: Accuracy = {results['accuracy']:.4f}")
            elif "weighted_f1" in results:
                logger.info(f"  {task}: Weighted F1 = {results['weighted_f1']:.4f}")
            elif "bleu" in results:
                logger.info(f"  {task}: BLEU = {results['bleu']:.4f}, ROUGE-L = {results.get('rouge_l', 0):.4f}")
            elif "exact_match_accuracy" in results:
                logger.info(f"  {task}: Exact Match = {results['exact_match_accuracy']:.4f}")
            elif "phase" in results:
                # HeiChole multi-task results
                logger.info(f"  {task}: Phase Acc = {results['phase'].get('accuracy', 0):.4f}, "
                          f"Tool F1 = {results.get('tools', {}).get('macro_f1', 0):.4f}, "
                          f"Action Acc = {results['action'].get('accuracy', 0):.4f}")
    
    # Print dissertation tables
    logger.info("=" * 60)
    logger.info("DISSERTATION TABLES (5.1-5.4)")
    logger.info("=" * 60)
    for table_name, table_data in dissertation_tables.items():
        logger.info(f"  {table_name}: {json.dumps(table_data)}")
    
    return {
        "detailed_results": all_results,
        "dissertation_tables": dissertation_tables
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SurgicalVLM Multi-task Evaluation")
    parser.add_argument("--model-config", default="configs/models/qwen2.5_vl_3b.yaml", help="Model config YAML")
    parser.add_argument("--checkpoint", default="", help="LoRA checkpoint path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--tasks", nargs="+", default=["phase", "instruments", "heichole", "language", "triplet"],
                        choices=["phase", "instruments", "heichole", "language", "triplet", "all"])
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per task")
    parser.add_argument("--image-root", default="/runpod-volume/data", help="Root directory containing frames/ and raw/ image directories")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate on")
    parser.add_argument("--smooth-window", type=int, default=5, help="Temporal smoothing window size (0 disables)")
    
    args = parser.parse_args()
    
    if "all" in args.tasks:
        args.tasks = ["phase", "instruments", "heichole", "language", "triplet"]
    
    run_full_evaluation(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        tasks=args.tasks,
        max_samples=args.max_samples,
        image_root=args.image_root,
        output_file=args.output,
        split=args.split,
        smooth_window=args.smooth_window
    )