"""
Loads processed/vlm_jsonl/ files for instruction fine-tuning.
Reads split_manager to exclude leakage videos.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .split_manager import SplitManager


class VLMJSONLDataset(Dataset):
    """Dataset that loads VLM conversation data from JSONL files."""

    def __init__(
        self,
        jsonl_path: str,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        split_manager: Optional[SplitManager] = None,
        max_samples: Optional[int] = None,
        image_root: Optional[str] = "data/raw",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.dataset_name = dataset_name
        self.split = split
        self.split_manager = split_manager
        self.image_root = Path(image_root)
        self.max_samples = max_samples

        self._data: List[Dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and optionally filter JSONL data."""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._data.append(record)

        # Filter by split if both dataset_name and split are provided
        if self.dataset_name and self.split and self.split_manager:
            allowed_videos = set()
            if self.split == "train":
                allowed_videos = set(self.split_manager.get_train(self.dataset_name))
            elif self.split == "val":
                allowed_videos = set(self.split_manager.get_val(self.dataset_name))
            elif self.split == "test":
                allowed_videos = set(self.split_manager.get_test(self.dataset_name))

            if allowed_videos:
                filtered = []
                for record in self._data:
                    video_id = record.get("video_id", "")
                    if video_id in allowed_videos:
                        filtered.append(record)
                self._data = filtered

        if self.max_samples and len(self._data) > self.max_samples:
            step = len(self._data) / self.max_samples
            indices = [int(i * step) for i in range(self.max_samples)]
            self._data = [self._data[i] for i in indices]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict:
        record = self._data[idx]

        image_path = record.get("image", "")
        full_image_path = self.image_root / image_path
        if not full_image_path.exists():
            full_image_path = Path(image_path)  # Try absolute path

        # Resolve frame_paths (added by scripts/add_temporal_frame_paths.py for
        # temporal-window training/eval) the same way image_path is resolved
        # above, so downstream code (get_temporal_pooled_features) can treat
        # every path uniformly as ready-to-open. Previously this field was
        # silently dropped here -- __getitem__ only copied a fixed set of
        # known keys -- so every frame_paths-consuming caller always saw None
        # and silently fell back to single-frame with no error at all.
        raw_frame_paths = record.get("frame_paths") or []
        resolved_frame_paths = []
        for p in raw_frame_paths:
            fp = self.image_root / p
            if not fp.exists():
                fp = Path(p)
            resolved_frame_paths.append(str(fp))

        item = {
            "image_path": str(full_image_path),
            "frame_paths": resolved_frame_paths,
            "conversations": record.get("conversations", []),
            "video_id": record.get("video_id", ""),
            "frame_idx": record.get("frame_idx", 0),
            "dataset": record.get("dataset", self.dataset_name or ""),
        }
        item.update(self._extract_label_fields(item["conversations"]))
        return item

    @staticmethod
    def _extract_label_fields(conversations: List[Dict]) -> Dict:
        """Parse the gpt-turn JSON answer into the flat phase/tools/action/
        triplet_* fields HeadCollator and MultiTaskSurgicalTrainer expect.

        The gpt answer is a JSON object for classification tasks (phase,
        heichole multitask, cholect50 triplet) but plain text for surg396k
        VQA rows — json.loads fails there and this returns {} (surg396k only
        needs the language-modeling path, not these fields).
        """
        gpt_answer = None
        for conv in conversations:
            if conv.get("from") == "gpt":
                gpt_answer = conv.get("value")
                break
        if not gpt_answer:
            return {}

        try:
            parsed = json.loads(gpt_answer)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(parsed, dict):
            return {}

        fields: Dict = {}
        if "phase" in parsed:
            fields["phase"] = parsed.get("phase")
        if "tools" in parsed:
            fields["tools"] = parsed.get("tools")
        if "action" in parsed:
            fields["action"] = parsed.get("action")
        if "instrument" in parsed and "verb" in parsed and "target" in parsed:
            fields["triplet_instrument"] = parsed.get("instrument")
            fields["triplet_verb"] = parsed.get("verb")
            fields["triplet_target"] = parsed.get("target")
        return fields