"""
Multi-task Training Script for Surgical VLM
Unified training loop across Cholec80, CholecT50, HeiChole using JSONL datasets
"""

import os
import sys
import json
import re
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure accelerate is available for device_map
try:
    import accelerate
    print(f"[OK] Accelerate {accelerate.__version__} loaded in training script")
except ImportError:
    print("[ERROR] Accelerate not available - device_map='auto' will fail")
    raise

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
surgical_vlm_path = os.path.join(project_root, "surgical_vlm")
sys.path.insert(0, project_root)
sys.path.insert(0, surgical_vlm_path)
print(f"Added to path: {project_root}")
print(f"Added to path: {surgical_vlm_path}")
print(f"Path exists: {os.path.exists(project_root)}")
print(f"sys.path[0]: {sys.path[0]}")
print(f"__file__: {__file__}")
print(f"cwd: {os.getcwd()}")

import os
print(f"Dir contents: {os.listdir(sys.path[0])}")

import surgical_vlm
print(f"surgical_vlm module: {surgical_vlm}")
print(f"surgical_vlm.__path__: {surgical_vlm.__path__}")
import surgical_vlm.models
print(f"surgical_vlm.models: {surgical_vlm.models}")

from surgical_vlm.data.vlm_dataset import VLMJSONLDataset
from surgical_vlm.data.split_manager import SplitManager
from surgical_vlm.data.collators import MultiTaskCollator, VQACollator, HeadCollator, TemporalStackingCollator
from surgical_vlm.models.surgical_vlm import create_surgical_vlm, SurgicalVLM
from surgical_vlm.training.trainer import create_surgical_trainer, MultiTaskSurgicalTrainer
from surgical_vlm.training.loss_functions import MultiTaskLoss
from surgical_vlm.training.output_adapter import create_output_adapter, OutputAdapter
from surgical_vlm.models.temporal_aggregator import create_temporal_aggregator, TemporalAggregator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Map non-canonical dataset keys used in configs to the canonical keys defined
# in configs/data/splits.yaml. SplitManager looks these up via
# `splits["datasets"][key]`, so a KeyError is raised unless we alias here.
DATASET_KEY_ALIASES = {
    "cholec_t50": "cholect50",
    "cholec_t_50": "cholect50",
}


def infer_dataset_name(filename: str) -> str:
    """Infer the dataset key from a JSONL filename."""
    if "cholec80" in filename:
        return "cholec80"
    if "heichole" in filename:
        return "heichole"
    if "cholect50" in filename or "cholec_t50" in filename:
        return "cholec_t50"
    if "surg396k" in filename:
        return "surg396k"
    return (
        filename.replace("_vlm_train.jsonl", "")
        .replace("_vlm_val.jsonl", "")
        .replace("_vlm_test.jsonl", "")
        .replace("_vlm.jsonl", "")
    )


class DatasetGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, dataset_weights, steps_per_epoch=None):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.groups = []
        offset = 0
        total = 0
        for dataset_obj, weight in dataset_weights:
            size = len(dataset_obj)
            self.groups.append((offset, offset + size, size * weight))
            offset += size
            total += size
        group_total = sum(g[2] for g in self.groups)
        self.group_weights = [g[2] / group_total for g in self.groups]
        num_samples = steps_per_epoch if steps_per_epoch is not None else total
        self.num_batches = num_samples // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            group_idx = random.choices(range(len(self.groups)), weights=self.group_weights, k=1)[0]
            start, end, _ = self.groups[group_idx]
            yield torch.randint(start, end, (self.batch_size,)).tolist()

    def __len__(self):
        return self.num_batches


class JSONLDataModule:
    """Manages multi-task dataset creation and data loading from JSONL files."""
    
    def __init__(self, config: Dict, tokenizer=None, temporal_aggregator=None):
        self.config = config
        self.tokenizer = tokenizer
        self.temporal_aggregator = temporal_aggregator
        self.datasets = {}
        self.dataloaders = {}
        self.split_manager = SplitManager(config.get("data", {}).get("splits_dir", "data/splits"))

    def build_datasets(self, stage: str = "train") -> Dict[str, Any]:
        """Build all datasets for the given stage from JSONL files."""
        data_config = self.config.get("data", {})
        multitask_config = self.config.get("multitask", {})
        
        dataset_ratios = multitask_config.get("dataset_ratios", {})
        jsonl_dir = data_config.get("jsonl_dir", "data/processed/vlm_jsonl")
        
        # Get file lists from config
        if stage == "train":
            file_lists = data_config.get("train_files", [])
        else:
            file_lists = data_config.get("val_files", [])
        
        # Common dataset config
        common_config = {
            "fps": data_config.get("fps", 1),
            "seq_length": data_config.get("seq_length", 16),
            "split": stage,
            "split_manager": self.split_manager,
            "tokenizer": self.tokenizer,
            "max_text_length": data_config.get("max_text_length", 512),
            "image_root": data_config.get("image_root", "data"),
        }
        
        # Map dataset names to their JSONL files
        dataset_file_map = {}
        for filename in file_lists:
            dataset_name = infer_dataset_name(filename)
            if dataset_name not in dataset_file_map:
                dataset_file_map[dataset_name] = []
            dataset_file_map[dataset_name].append(filename)
        
        # Build datasets
        for dataset_name, filenames in dataset_file_map.items():
            if dataset_name not in dataset_ratios:
                logger.info(f"Skipping {dataset_name} - not in dataset_ratios")
                continue
            
            # Combine multiple files for the same dataset
            combined_samples = []
            split_key = DATASET_KEY_ALIASES.get(dataset_name, dataset_name)
            for filename in filenames:
                jsonl_path = Path(jsonl_dir) / filename
                if not jsonl_path.exists():
                    # Fix 2 (Option A): fall back to the train JSONL for val, filtering
                    # to the val videos via split_manager (splits.yaml) below.
                    if stage == "val":
                        train_filename = filename.replace("_val", "_train")
                        alt_path = Path(jsonl_dir) / train_filename
                        if alt_path.exists():
                            logger.warning(
                                f"Val JSONL not found: {jsonl_path}; "
                                f"falling back to {train_filename} filtered by val split"
                            )
                            jsonl_path = alt_path
                            filename = train_filename
                    if not jsonl_path.exists():
                        logger.warning(f"JSONL file not found: {jsonl_path}")
                        continue
                
                try:
                    dataset = VLMJSONLDataset(
                        jsonl_path=str(jsonl_path),
                        dataset_name=split_key,
                        split=stage,
                        split_manager=self.split_manager,
                        max_samples=None,
                        image_root=data_config.get("image_root", "data"),
                    )
                    logger.info(f"Loaded {len(dataset)} samples from {filename}")
                    combined_samples.extend(dataset._data)
                except Exception as e:
                    logger.error(f"Failed to load {jsonl_path}: {e}")
            
            if combined_samples:
                # Create a combined dataset
                self.datasets[dataset_name] = self._create_combined_dataset(
                    combined_samples, split_key, stage, common_config
                )
            else:
                logger.warning(f"No samples loaded for {dataset_name}")
        
        logger.info(f"Built {len(self.datasets)} datasets for {stage}")
        for name, ds in self.datasets.items():
            logger.info(f"  {name}: {len(ds)} samples")
        return self.datasets
    
    def _create_combined_dataset(self, samples: List[Dict], dataset_name: str, stage: str, common_config: Dict):
        """Create a dataset from combined samples."""
        class CombinedJSONLDataset(torch.utils.data.Dataset):
            def __init__(self, samples, dataset_name, split, split_manager, tokenizer, max_text_length, image_root):
                self.samples = samples
                self.dataset_name = dataset_name
                self.split = split
                self.split_manager = split_manager
                self.tokenizer = tokenizer
                self.max_text_length = max_text_length
                self.image_root = Path(image_root)
                
                # Filter by split if needed
                if split_manager and dataset_name and split:
                    allowed_videos = set()
                    if split == "train":
                        allowed_videos = set(split_manager.get_train(dataset_name))
                    elif split == "val":
                        allowed_videos = set(split_manager.get_val(dataset_name))
                    elif split == "test":
                        allowed_videos = set(split_manager.get_test(dataset_name))
                    
                    if allowed_videos:
                        filtered = []
                        for record in self.samples:
                            video_id = record.get("video_id", "")
                            if video_id in allowed_videos:
                                filtered.append(record)
                        self.samples = filtered
                        logger.info(f"Filtered {dataset_name} {split}: {len(self.samples)} samples after split filtering")
            
            def __len__(self):
                return len(self.samples)
            
            def _frame_paths(self, record, full_image_path, frame_idx):
                # Temporal neighbor frames for Option A (T=3). Neighbor indices
                # step back 25 frames (~1s at 25fps). Any invalid or missing
                # neighbor collapses to the anchor so the patch count stays
                # constant across the batch and datasets (e.g. surg396k has no
                # frame_idx -> defaults to 0 -> all neighbors are the anchor).
                anchor = Path(full_image_path)
                paths = [str(anchor)]
                for k in (1, 2, 3):
                    nidx = frame_idx - 25 * k
                    if nidx < 0:
                        paths.append(str(anchor))
                        continue
                    match = re.search(r"(\d+)", anchor.name)
                    if not match:
                        paths.append(str(anchor))
                        continue
                    digits = match.group(1)
                    new_name = (
                        anchor.name[: match.start()] 
                        + str(nidx).zfill(len(digits)) 
                        + anchor.name[match.end():]
                    )
                    candidate = anchor.with_name(new_name)
                    paths.append(str(candidate) if candidate.exists() else str(anchor))
                return paths
            
            def __getitem__(self, idx):
                record = self.samples[idx]
                image_path = record.get("image", "")
                full_image_path = self.image_root / image_path
                if not full_image_path.exists():
                    full_image_path = Path(image_path)
                
                # Tokenize if tokenizer available
                input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
                attention_mask = torch.zeros(self.max_text_length, dtype=torch.long)
                
                if self.tokenizer:
                    prompt = record.get("conversations", [{}])[0].get("value", "")
                    encoded = self.tokenizer(
                        prompt,
                        max_length=self.max_text_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    input_ids = encoded['input_ids'].squeeze(0)
                    attention_mask = encoded['attention_mask'].squeeze(0)
                
                # Parse ground-truth labels from the gpt answer JSON (the answer
                # carries the canonical phase / instruments / actions). These are
                # emitted as `phase` / `tools` / `action` so that the trainer's
                # _prepare_targets maps them to class IDs via the label mapper.
                gpt_value = ""
                conversations = record.get("conversations", [])
                for turn in conversations:
                    if isinstance(turn, dict) and turn.get("from") == "gpt":
                        gpt_value = turn.get("value", "")
                        break

                phase = ""
                tools = ""
                action = ""
                triplet_instrument = ""
                triplet_verb = ""
                triplet_target = ""
                if gpt_value:
                    try:
                        parsed = json.loads(gpt_value)
                        if isinstance(parsed, dict):
                            phase = parsed.get("phase", "")
                            raw_tools = parsed.get("tools", [])
                            tool_list = raw_tools if isinstance(raw_tools, list) else ([str(raw_tools)] if raw_tools else [])
                            tools = ", ".join(str(t) for t in tool_list if str(t).strip())
                            # cholect50 stores the action under "verb"; heichole uses "action"
                            action = parsed.get("action", parsed.get("verb", ""))
                            triplet_instrument = parsed.get("instrument", "")
                            triplet_verb = parsed.get("verb", "")
                            triplet_target = parsed.get("target", "")
                    except (json.JSONDecodeError, TypeError):
                        # Non-JSON answer: fall back to the record's label field
                        if isinstance(gpt_value, str):
                            phase = gpt_value.strip()
                        if not phase:
                            phase = ""

                result = {
                    "image_path": str(full_image_path),
                    "frame_paths": self._frame_paths(record, full_image_path, record.get("frame_idx", 0)),
                    "conversations": conversations,
                    "video_id": record.get("video_id", ""),
                    "frame_idx": record.get("frame_idx", 0),
                    "dataset": record.get("dataset", self.dataset_name),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "task": "vqa" if self.dataset_name == "surg396k" else record.get("task", ""),
                    "label": record.get("label", ""),
                    "phase": phase,
                    "tools": tools,
                    "action": action,
                    "triplet_instrument": triplet_instrument,
                    "triplet_verb": triplet_verb,
                    "triplet_target": triplet_target,
                }

                return result
        
        return CombinedJSONLDataset(
            samples, dataset_name, stage, 
            common_config["split_manager"], 
            common_config["tokenizer"],
            common_config["max_text_length"],
            common_config.get("image_root", "data")
        )

    def build_dataloaders(self, batch_size: int, num_workers: int = 4, pin_memory: bool = True):
        """Build dataloaders for all datasets."""
        for name, dataset in self.datasets.items():
            if dataset is None or len(dataset) == 0:
                logger.warning(f"Dataset {name} is empty, skipping")
                continue
            
            self.dataloaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                persistent_workers=num_workers > 0,
            )
        
        return self.dataloaders
    
    def get_combined_dataloader(self, batch_size: int, ratios: Dict[str, float], num_workers: int = 4, steps_per_epoch: Optional[int] = None, collate_fn=None):
        """Create a combined dataloader with task-homogeneous micro-batches."""
        pairs = []
        for name, dataset in self.datasets.items():
            if dataset is not None and len(dataset) > 0:
                pairs.append((dataset, ratios.get(name, 1.0)))
        
        if not pairs:
            return None
        
        combined = ConcatDataset([p[0] for p in pairs])
        
        num_samples = steps_per_epoch if steps_per_epoch is not None else len(combined)
        
        batch_sampler = DatasetGroupedBatchSampler(
            combined,
            batch_size,
            pairs,
            steps_per_epoch=num_samples,
        )
        
        return DataLoader(
            combined,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def get_validation_loaders(self, batch_size: int, num_workers: int = 2):
        """Build validation dataloaders (separate per dataset)."""
        val_loaders = {}
        for name, dataset in self.datasets.items():
            if dataset is not None and len(dataset) > 0:
                val_loaders[name] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                logger.info(f"Val {name}: {len(dataset)} samples")
        return val_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task Surgical VLM Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    return parser.parse_args()


def _coerce_numeric(value: Any) -> Any:
    """Return numeric-typed scalars from numeric-looking string leaves.

    YAML files occasionally carry quoted numerics (e.g. 'learning_rate: "1e-4"')
    which load as str; torch.optim then dies with
    "'<=' not supported between instances of 'float' and 'str'".
    """
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _normalize_config(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _normalize_config(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_config(v) for v in node]
    return _coerce_numeric(node)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return _normalize_config(config)


def setup_distributed(local_rank: int):
    """Initialize distributed training."""
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        logger.info(f"Initialized DDP on rank {dist.get_rank()}")


def seed_everything(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set global seed to {seed}")


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("training", {}).get("seed", 42))
    
    # Setup distributed
    setup_distributed(args.local_rank)
    
    # Device
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Create temporal aggregator
    model_config = config.get("model", {})
    temporal_config = model_config.get("temporal_config", {})
    use_temporal = model_config.get("use_temporal", True)
    
    temporal_aggregator = None
    if use_temporal:
        temporal_aggregator = create_temporal_aggregator(temporal_config)
        temporal_aggregator.to(device)
        logger.info(f"Created temporal aggregator: {temporal_aggregator.strategy}")
    
    # Create model
    vlm = create_surgical_vlm(
        model_type=model_config.get("type", "qwen2.5_vl_3b"),
        load_in_4bit=model_config.get("load_in_4bit", True),
        device=str(device),
        use_temporal=use_temporal,
        temporal_config=temporal_config,
        max_pixels=model_config.get("max_pixels", 12845056),
    )
    vlm.load()
    
    # Enable gradient checkpointing to reduce VRAM usage (~6-8GB saved)
    if hasattr(vlm, "vlm") and hasattr(vlm.vlm, "model"):
        vlm.vlm.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Apply LoRA if configured
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        vlm.apply_lora(lora_config.get("config_path", "configs/training/lora_config.yaml"))
    
    # Get tokenizer
    tokenizer = vlm.vlm.processor.tokenizer if hasattr(vlm.vlm, "processor") else None
    processor = vlm.vlm.processor if hasattr(vlm.vlm, "processor") else None

    # Multi-task collator: route VQA rows through the VLM processor (chat
    # template + image encoding), all other head-task rows through HeadCollator
    # (pre-tokenized ids + single-frame pixel_values + aggregated labels).
    multitask_config = config.get("multitask", {})
    collate_max_length = multitask_config.get("collator_max_length", 2048)
    image_root = config.get("data", {}).get("image_root", "data")
    pad_token_id = 0
    if processor is not None and getattr(processor, "tokenizer", None) is not None:
        _ptid = processor.tokenizer.pad_token_id
        pad_token_id = _ptid if _ptid is not None else processor.tokenizer.eos_token_id
    # Temporal Option-A: when use_temporal is on, expand each sample into T
    # per-frame rows (anchor + neighbors) and mean-pool frame features in the
    # trainer. Single-frame HeadCollator otherwise.
    if use_temporal:
        default_collator = TemporalStackingCollator(
            processor=processor,
            image_root=image_root,
            max_length=collate_max_length,
            num_frames=temporal_config.get("num_frames", 3),
        )
        logger.info(
            f"Temporal stacking collator enabled (num_frames={default_collator.num_frames})"
        )
    else:
        default_collator = HeadCollator(processor=processor, image_root=image_root, max_length=collate_max_length)
    multitask_collator = MultiTaskCollator(
        task_collators={
            "vqa": VQACollator(processor=processor, max_length=collate_max_length, image_root=image_root)
        },
        task_key="task",
        default_collator=default_collator,
        pad_token_id=pad_token_id,
    )
    
    # Build data module - USE JSONL DATASETS
    data_module = JSONLDataModule(config, tokenizer=tokenizer, temporal_aggregator=temporal_aggregator)
    train_datasets = data_module.build_datasets("train")
    val_datasets = data_module.build_datasets("val")
    
    # Get dataset ratios for proportional sampling
    dataset_ratios = config.get("multitask", {}).get("dataset_ratios", {})
    
    # Build combined training loader with proportional sampling
    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("training", {}).get("num_workers", 4)
    
    train_loader = data_module.get_combined_dataloader(
        batch_size,
        dataset_ratios,
        num_workers,
        steps_per_epoch=config.get("training", {}).get("steps_per_epoch", None),
        collate_fn=multitask_collator,
    )
    
    # Build validation loaders (separate per dataset)
    val_loaders = {}
    for name, dataset in val_datasets.items():
        if dataset is not None and len(dataset) > 0:
            val_loaders[name] = DataLoader(
                dataset,
                batch_size=config.get("training", {}).get("eval_batch_size", 4),
                shuffle=False,
                num_workers=config.get("training", {}).get("num_workers", 2),
                pin_memory=True,
                drop_last=False,
            )
            logger.info(f"Val {name}: {len(dataset)} samples")
    
    # Create output adapter
    output_adapter_config = config.get("output_adapter", {})
    output_adapter = create_output_adapter(output_adapter_config)
    output_adapter.to(device)
    
    # Loss function
    loss_config = config.get("multitask", {}).get("loss_weights", {})
    criterion = MultiTaskLoss(
        phase_weight=loss_config.get("phase_classification", 1.0),
        instrument_weight=loss_config.get("tool_detection", 1.0),
        action_weight=loss_config.get("action_triplet", 1.5),
        language_weight=loss_config.get("language_modeling", 1.0),
        triplet_weight=loss_config.get("triplet", 1.5),
        grounding_weight=loss_config.get("grounding", 0.5),
    )
    
    # Optimizer - include temporal aggregator and output adapter parameters
    train_config = config.get("training", {})
    trainable_params = []
    for p in vlm.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    if temporal_aggregator:
        for p in temporal_aggregator.parameters():
            if p.requires_grad:
                trainable_params.append(p)
    for p in output_adapter.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.get("learning_rate", 2e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Scheduler
    scheduler = None
    if train_config.get("scheduler", "cosine") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        steps_per_epoch = len(train_loader) if train_loader else 100
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config.get("epochs", 10) * steps_per_epoch,
            eta_min=train_config.get("min_lr", 1e-6),
        )
    
    # Create trainer using factory
    from surgical_vlm.training.label_mapping import get_label_mapper
    label_mapper = get_label_mapper()
    
    trainer = create_surgical_trainer(
        model=vlm,
        config=config,
        train_loaders={"combined": train_loader} if train_loader else {},
        val_loaders=val_loaders,
    )
    # Override with our combined loader and custom components
    trainer.combined_loader = train_loader
    trainer.output_adapter = output_adapter
    trainer.temporal_aggregator = temporal_aggregator
    trainer.criterion = criterion
    trainer.label_mapper = label_mapper
    # The factory builds its own optimizer/scheduler over its internal adapter
    # params; override them so the custom optimizer (which covers the real
    # output_adapter/temporal_aggregator/vlm params) is actually used.
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    epochs = train_config.get("epochs", 10)
    trainer.fit(
        train_loaders={"combined": train_loader} if train_loader else {},
        val_loaders=val_loaders,
        epochs=epochs,
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()