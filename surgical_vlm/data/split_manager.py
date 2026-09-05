"""
Single source of truth for all train/val/test splits.
Reads configs/data/splits.yaml → data/splits/*.json
Applies global_exclusions.json at load time.
Raises error if same video appears in train AND test.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import yaml



class SplitLeakageError(Exception):
    """Raised when a video ID appears in both train and test splits."""

    pass


class SplitManager:
    """Manages dataset splits with leakage prevention."""

    PHASE_NAMES = {
        "cholec80": [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
        ],
        "heichole": [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
        ],
    }

    def __init__(self, config_path: str = "configs/data/splits.yaml"):
        # Resolve config relative to repo root (independent of current working directory).
        self.repo_root = Path(__file__).resolve().parents[2]  # SurgicalVLM_Robotics/


        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = self.repo_root / cfg_path


        # Also handle config paths provided with mixed separators.
        if not cfg_path.exists():
            alt = self.repo_root / Path(config_path.replace('\\', '/'))
            if alt.exists():
                cfg_path = alt


        if not cfg_path.exists():
            # Tests only need phase names; allow running without splits config.
            self.config = {}
            self.splits_dir = Path("data/splits")
            self._global_exclusions = set()
            self._splits_cache = {}
            return

        with open(cfg_path, "r") as f:
            self.config = yaml.safe_load(f)



        # Resolve splits_dir relative to repo root
        self.splits_dir = (self.repo_root / self.config["splits_dir"]).resolve()


        self._global_exclusions: Set[str] = set()
        self._load_global_exclusions()
        self._splits_cache: Dict[str, Dict[str, List[str]]] = {}

    def _load_global_exclusions(self) -> None:
        """Load videos that must be excluded from test sets (e.g., Surg-396K overlap)."""
        excl_path = Path(self.config["global_exclusions"])
        if excl_path.exists():
            with open(excl_path, "r") as f:
                data = json.load(f)
            # Support both:
            #  1) exact video IDs (e.g., "v4") - list format
            #  2) dict with "excluded_from_test" key - dict format
            #  3) dataset-qualified paths (e.g., "test_ds/v4")
            if isinstance(data, dict):
                raw_exclusions = data.get("excluded_from_test", [])
            else:
                raw_exclusions = data
            self._global_exclusions = set(raw_exclusions)


    def _load_split_file(self, dataset: str) -> Dict[str, List[str]]:
        """Load and validate a single dataset split file."""
        if dataset in self._splits_cache:
            return self._splits_cache[dataset]

        split_rel = Path(self.config["datasets"][dataset])
        split_path = (self.repo_root / split_rel).resolve() if not split_rel.is_absolute() else split_rel.resolve()
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. Resolved from {split_rel} using repo_root={self.repo_root}"
            )



        with open(split_path, "r") as f:
            splits = json.load(f)

        # Validate arrays exist
        train_list = splits.get("train", [])
        val_list = splits.get("val", [])
        test_list = splits.get("test", [])

        if not isinstance(train_list, list) or not isinstance(val_list, list) or not isinstance(test_list, list):
            raise ValueError(
                f"Invalid split schema in {split_path} for dataset={dataset}. Expected JSON keys: train/val/test as lists."
            )

        # Validate: no overlap between train and test
        train_set = set(train_list)
        test_set = set(test_list)
        overlap = train_set & test_set
        if overlap:
            raise SplitLeakageError(
                f"Video-level leakage in {dataset}: {overlap} appear in both train and test"
            )

        # Validate: no overlap between train and val
        val_set = set(val_list)
        train_val_overlap = train_set & val_set
        if train_val_overlap:
            raise SplitLeakageError(
                f"Video-level leakage in {dataset}: {train_val_overlap} appear in both train and val"
            )

        # Fail-fast: detect empty split files (prevents silent 'no data' runs)
        if len(train_list) == 0 and len(val_list) == 0 and len(test_list) == 0:
            raise ValueError(
                f"Empty split file loaded: {split_path} (dataset={dataset}). "
                "train/val/test are all empty. Populate the split JSON or fix the generator/source for splits." 
            )


        # Apply global exclusions to test set
        original_test = splits.get("test", [])

        def is_excluded(video_id: str) -> bool:
            if video_id in self._global_exclusions:
                return True
            # Also handle qualified ids like "dataset/video_id"
            qualified = f"{dataset}/{video_id}"
            return qualified in self._global_exclusions

        filtered_test = [v for v in original_test if not is_excluded(v)]

        if len(filtered_test) != len(original_test):
            excluded = set(original_test) - set(filtered_test)
            splits["test"] = filtered_test
            splits["_excluded_from_test"] = list(excluded)

        self._splits_cache[dataset] = splits
        return splits

    def get_train(self, dataset: str) -> List[str]:
        """Return list of training video IDs for a dataset."""
        return self._load_split_file(dataset).get("train", [])

    def get_val(self, dataset: str) -> List[str]:
        """Return list of validation video IDs for a dataset."""
        return self._load_split_file(dataset).get("val", [])

    def get_test(self, dataset: str) -> List[str]:
        """Return list of test video IDs (with global exclusions applied)."""
        return self._load_split_file(dataset).get("test", [])

    def get_all_splits(self, dataset: str) -> Dict[str, List[str]]:
        """Return full split dict for a dataset."""
        return self._load_split_file(dataset)

    def is_in_split(self, dataset: str, video_id: str, split: str) -> bool:
        """Check if a video ID belongs to a specific split."""
        splits = self._load_split_file(dataset)
        return video_id in splits.get(split, [])

    def get_phase_names(self, dataset: str) -> List[str]:
        """Return phase name list for a dataset."""
        return self.PHASE_NAMES.get(dataset, [])