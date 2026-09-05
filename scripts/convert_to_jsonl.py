"""
Convert Annotations to Unified VLM JSONL Format
Converts all dataset annotations to a single unified JSONL format for training.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
from tqdm import tqdm

# Add surgical_vlm to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
surgical_vlm_path = os.path.join(project_root, "surgical_vlm")
sys.path.insert(0, project_root)
sys.path.insert(0, surgical_vlm_path)

from surgical_vlm.data.split_manager import SplitManager


def load_split_manager(splits_dir: str = "data/splits"):
    """Load split manager for consistent splits."""
    # SplitManager expects path to config YAML, not the splits directory
    return SplitManager("configs/data/splits.yaml")


# ============================================================
# Cholec80 Conversion
# ============================================================

def convert_cholec80(
    root_path: str,
    output_path: str,
    split: str = "train",
    fps: int = 1,
    max_frames_per_video: Optional[int] = None,
) -> int:
    """Convert Cholec80 phase annotations to JSONL."""
    root = Path(root_path)
    videos_dir = root / "videos"
    phase_dir = root / "phase_annotations"
    tool_dir = root / "tool_annotations"
    
    split_manager = load_split_manager()
    video_names = split_manager.get_train("cholec80") if split == "train" else (
        split_manager.get_val("cholec80") if split == "val" else split_manager.get_test("cholec80")
    )
    video_names = sorted(video_names)
    
    PHASE_NAMES = [
        "Preparation", "CalotTriangleDissection", "ClippingCutting",
        "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation",
        "GallbladderRetraction", "Unknown"
    ]
    
    TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    
    frame_interval = 25 // fps
    count = 0
    
    with open(output_path, 'w') as f:
        for video_name in tqdm(video_names, desc=f"Cholec80 {split}"):
            video_path = videos_dir / f"{video_name}.mp4"
            if not video_path.exists():
                continue
            
            # Load phase annotations
            phase_file = phase_dir / f"{video_name}-phase.txt"
            phase_labels = {}
            if phase_file.exists():
                with open(phase_file, 'r') as pf:
                    for line in pf:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            try:
                                frame_idx = int(parts[0])
                                phase_name = parts[1].strip().replace(' ', '').replace('-', '')
                                phase_idx = PHASE_NAMES.index(phase_name) if phase_name in PHASE_NAMES else 7
                                phase_labels[frame_idx] = phase_idx
                            except (ValueError, IndexError):
                                continue
            
            # Load tool annotations
            tool_file = tool_dir / f"{video_name}-tool.txt"
            tool_labels = {}
            if tool_file.exists():
                with open(tool_file, 'r') as tf:
                    lines = tf.readlines()
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 8:
                            try:
                                frame_idx = int(parts[0])
                                tools = [int(parts[i]) for i in range(1, 8)]
                                tool_labels[frame_idx] = tools
                            except (ValueError, IndexError):
                                continue
            
            # Get video info
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Sample frames
            max_start = total_frames - 1
            if max_frames_per_video:
                step = max(1, max_start // max_frames_per_video)
            else:
                step = frame_interval
            
            for frame_idx in range(0, max_start, step):
                # Get phase at this frame
                phase_idx = 7  # Unknown
                for ts, p in sorted(phase_labels.items()):
                    if ts <= frame_idx:
                        phase_idx = p
                    else:
                        break
                
                # Get tools at this frame
                tools = [0] * 7
                for ts, t in sorted(tool_labels.items()):
                    if ts <= frame_idx:
                        tools = t
                    else:
                        break
                
                # Build JSONL entry
                phase_name = PHASE_NAMES[phase_idx]
                tool_names = [TOOL_NAMES[i] for i, v in enumerate(tools) if v > 0]
                
                # Frame image path (will be extracted later)
                rel_frame_path = f"frames/cholec80/{video_name}/frame_{frame_idx:06d}.jpg"
                
                # Build the response JSON
                response_json = json.dumps({
                    "phase": phase_name,
                    "tools": tool_names,
                    "description": f"Surgical phase: {phase_name}. Instruments visible: {', '.join(tool_names) if tool_names else 'None'}"
                })
                
                entry = {
                    "image": rel_frame_path,
                    "conversations": [
                        {"from": "human", "value": "What is the current surgical phase and what instruments are visible?"},
                        {"from": "gpt", "value": response_json}
                    ],
                    "task": "phase",
                    "dataset": "cholec80",
                    "video_id": video_name,
                    "frame_idx": frame_idx,
                }
                
                f.write(json.dumps(entry) + '\n')
                count += 1
    
    return count


# ============================================================
# HeiChole Conversion
# ============================================================

def convert_heichole(
    root_path: str,
    output_path: str,
    split: str = "train",
    fps: int = 1,
    tasks: List[str] = None,
) -> int:
    """Convert HeiChole CSV annotations to JSONL."""
    root = Path(root_path)
    
    tasks = tasks or ["phase", "instrument", "action", "skill"]
    
    split_manager = load_split_manager()
    video_indices = split_manager.get_train("heichole") if split == "train" else (
        split_manager.get_val("heichole") if split == "val" else split_manager.get_test("heichole")
    )
    
    # Phase names (from EvalPhase.py)
    PHASE_NAMES = [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ]
    
    # Instrument names (from EvalInstrument.py)
    INSTRUMENT_NAMES = [
        "Grasper",
        "Bipolar",
        "Hook",
        "Scissors",
        "Clipper",
        "Irrigator",
        "SpecimenBag",
    ]
    
    # Action names (from EvalAction.py)
    ACTION_NAMES = [
        "No_Action",
        "Grasp",
        "Retract",
        "Dissect",
        "Coagulate",
        "Cut",
        "Clip",
        "Aspirate",
        "Irrigate",
        "Pack",
    ]
    
    # Skill categories (from EvalSkill.py)
    SKILL_CATEGORIES = [
        "TissueHandling",
        "InstrumentHandling",
        "EconomyOfMotion",
        "FlowOfOperation",
        "OverallPerformance",
    ]
    
    frame_interval = 25 // fps if fps > 0 else 25
    count = 0
    
    with open(output_path, 'w') as f:
        for video_name in tqdm(video_indices, desc=f"HeiChole {split}"):
            # video_name is already like "Hei-Chole1" from split file
            # Check if video exists
            video_path = root / f"{video_name}_dissection.mp4"
            if not video_path.exists():
                video_path = root / f"{video_name}_calot.mp4"
            if not video_path.exists():
                video_path = root / f"{video_name}.mp4"
            if not video_path.exists():
                continue
            
            # Load all annotations
            phase_annotations = {}
            instrument_annotations = {}
            action_annotations = {}
            skill_annotations = {}
            
            # Phase annotations
            if "phase" in tasks:
                phase_file = root / f"{video_name}_Annotation_Phase.csv"
                if phase_file.exists():
                    phase_annotations = _parse_heichole_phase_csv(phase_file)
            
            # Instrument annotations
            if "instrument" in tasks:
                inst_file = root / f"{video_name}_Annotation_Instrument.csv"
                if inst_file.exists():
                    instrument_annotations = _parse_heichole_instrument_csv(inst_file)
            
            # Action annotations
            if "action" in tasks:
                action_file = root / "action_annotations" / f"{video_name}_Annotation_Action.csv"
                if not action_file.exists():
                    action_file = root / "annotations" / "action" / f"{video_name}_action.txt"
                if action_file.exists():
                    action_annotations = _parse_heichole_action_csv(action_file)
            
            # Skill annotations
            if "skill" in tasks:
                skill_file = root / "skill_annotations" / f"{video_name}_Skill.csv"
                if not skill_file.exists():
                    skill_file = root / "annotations" / "skill" / f"{video_name}_skill.json"
                if skill_file.exists():
                    skill_annotations = _parse_heichole_skill_csv(skill_file)
            
            # Get video info
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Sample frames
            for frame_idx in range(0, total_frames, frame_interval):
                # Get phase at this frame
                phase_idx = 0
                for ts, p in sorted(phase_annotations.items()):
                    if ts <= frame_idx:
                        phase_idx = p
                    else:
                        break
                
                # Get instruments at this frame
                instruments = [0] * 7
                for ts, inst in sorted(instrument_annotations.items()):
                    if ts <= frame_idx:
                        instruments = inst
                    else:
                        break
                
                # Get action at this frame
                action_idx = 0
                for ts, act in sorted(action_annotations.items()):
                    if ts <= frame_idx:
                        action_idx = act
                    else:
                        break
                
                phase_name = PHASE_NAMES[phase_idx] if phase_idx < len(PHASE_NAMES) else "Unknown"
                tool_names = [INSTRUMENT_NAMES[i] for i, v in enumerate(instruments) if v > 0]
                action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else "No_Action"
                
                rel_frame_path = f"frames/heichole/{video_name}/frame_{frame_idx:06d}.jpg"
                
                # Build response JSON
                response_json = json.dumps({
                    "phase": phase_name,
                    "tools": tool_names,
                    "action": action_name,
                    "skill": skill_annotations if skill_annotations else {},
                    "description": f"Surgical phase: {phase_name}. Instruments: {', '.join(tool_names) if tool_names else 'None'}. Action: {action_name}."
                })
                
                entry = {
                    "image": rel_frame_path,
                    "conversations": [
                        {"from": "human", "value": "Analyze this surgical frame and output the phase, instruments, action, and skill assessment."},
                        {"from": "gpt", "value": response_json}
                    ],
                    "task": "multitask",
                    "dataset": "heichole",
                    "video_id": video_name,
                    "frame_idx": frame_idx,
                }
                
                f.write(json.dumps(entry) + '\n')
                count += 1
    
    return count


def _parse_heichole_phase_csv(file_path: Path) -> Dict[int, int]:
    """Parse HeiChole phase CSV file."""
    annotations = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    frame_idx = int(parts[0])
                    phase_name = parts[1].strip().replace(' ', '').replace('-', '')
                    phase_map = {
                        "Preparation": 0,
                        "CalotTriangleDissection": 1,
                        "ClippingCutting": 2,
                        "GallbladderDissection": 3,
                        "GallbladderPackaging": 4,
                        "CleaningCoagulation": 5,
                        "GallbladderRetraction": 6,
                    }
                    phase_idx = phase_map.get(phase_name, 0)
                    annotations[frame_idx] = phase_idx
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return annotations


def _parse_heichole_instrument_csv(file_path: Path) -> Dict[int, List[float]]:
    """Parse HeiChole instrument CSV file (1 FPS, 7 binary values)."""
    annotations = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 8:
                try:
                    frame_idx = int(parts[0])
                    instruments = [float(parts[i]) for i in range(1, 8)]
                    annotations[frame_idx] = instruments
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return annotations


def _parse_heichole_action_csv(file_path: Path) -> Dict[int, int]:
    """Parse HeiChole action CSV file."""
    annotations = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    frame_idx = int(parts[0])
                    action_name = parts[1].strip()
                    action_map = {
                        "No_Action": 0,
                        "Grasp": 1,
                        "Retract": 2,
                        "Dissect": 3,
                        "Coagulate": 4,
                        "Cut": 5,
                        "Clip": 6,
                        "Aspirate": 7,
                        "Irrigate": 8,
                        "Pack": 9,
                    }
                    action_idx = action_map.get(action_name, 0)
                    annotations[frame_idx] = action_idx
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return annotations


def _parse_heichole_skill_csv(file_path: Path) -> Dict[str, float]:
    """Parse HeiChole skill CSV/JSON file."""
    try:
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix == '.csv':
            skills = {}
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    skills[parts[0]] = float(parts[1])
            return skills
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return {}


# ============================================================
# CholecT50 Conversion
# ============================================================

def convert_cholect50(
    root_path: str,
    output_path: str,
    split: str = "train",
) -> int:
    """Convert CholecT50 triplet annotations to JSONL."""
    root = Path(root_path)
    videos_dir = root / "videos"
    labels_dir = root / "labels"
    mapping_file = root / "label_mapping.txt"
    
    split_manager = load_split_manager()
    video_names = split_manager.get_train("cholect50") if split == "train" else (
        split_manager.get_val("cholect50") if split == "val" else split_manager.get_test("cholect50")
    )
    
    # Load triplet mapping
    triplet_to_id = {}
    id_to_triplet = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    triplet_to_id[f"{parts[1]},{parts[2]},{parts[3]}"] = int(parts[0])
                    id_to_triplet[int(parts[0])] = {
                        'instrument': parts[1],
                        'verb': parts[2],
                        'target': parts[3],
                    }
    
    count = 0
    with open(output_path, 'w') as f:
        for video_name in tqdm(video_names, desc=f"CholecT50 {split}"):
            label_file = labels_dir / f"{video_name}.json"
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as lf:
                data = json.load(lf)
            
            # Get annotations from the 'annotations' key
            annotations = data.get('annotations', {})
            
            # Get category mappings
            categories = data.get('categories', {})
            instrument_map = categories.get('instrument', {})
            verb_map = categories.get('verb', {})
            target_map = categories.get('target', {})
            
            for frame_idx_str, triplet_list in annotations.items():
                # Skip non-numeric keys
                try:
                    frame_idx = int(frame_idx_str)
                except ValueError:
                    continue
                
                # triplet_list is a list of lists, take the first one
                if not triplet_list or not isinstance(triplet_list[0], list):
                    continue
                triplet = triplet_list[0]
                
                # Map indices to names
                inst_idx = triplet[0] if len(triplet) > 0 else 0
                verb_idx = triplet[1] if len(triplet) > 1 else 0
                target_idx = triplet[2] if len(triplet) > 2 else 0
                
                inst = instrument_map.get(str(inst_idx), 'Grasper')
                verb = verb_map.get(str(verb_idx), 'No_Action')
                target = target_map.get(str(target_idx), 'No_Target')
                
                rel_frame_path = f"frames/cholect50/{video_name}/frame_{frame_idx:06d}.jpg"
                
                # Build response JSON
                response_json = json.dumps({
                    "instrument": inst,
                    "verb": verb,
                    "target": target,
                    "description": f"Instrument: {inst}, Action: {verb}, Target: {target}"
                })
                
                entry = {
                    "image": rel_frame_path,
                    "conversations": [
                        {"from": "human", "value": "What is the instrument, action, and target in this frame?"},
                        {"from": "gpt", "value": response_json}
                    ],
                    "task": "triplet",
                    "dataset": "cholect50",
                    "video_id": video_name,
                    "frame_idx": frame_idx,
                }
                
                f.write(json.dumps(entry) + '\n')
                count += 1
    
    return count


# ============================================================
# Surg-396K Conversion (Already in conversation format)
# ============================================================

def convert_surg396k(
    root_path: str,
    output_path: str,
    split: str = "train",
) -> int:
    """Convert Surg-396K total_train.json to unified JSONL."""
    root = Path(root_path)
    annotations_file = root / "total_train.json"
    
    if not annotations_file.exists():
        return 0
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Filter by video-level splits. SplitManager has no get_split_indices;
    # the splits map to video IDs (e.g. "000937"), derived from the first
    # path segment after "CoPESD/" in each image path.
    split_manager = load_split_manager()
    if split == "train":
        allowed_video_ids = set(split_manager.get_train("surg396k"))
    elif split == "val":
        allowed_video_ids = set(split_manager.get_val("surg396k"))
    else:
        allowed_video_ids = set(split_manager.get_test("surg396k"))
    
    count = 0
    with open(output_path, 'w') as f:
        for item in tqdm(data, desc=f"Surg396K {split}"):
            image_path = item.get('image', '')
            conversations = item.get('conversations', [])
            
            if not image_path or len(conversations) < 2:
                continue
            
            # Derive video ID the same way prepare_splits.py does:
            # e.g. ".../CoPESD/000937/0001.jpg" -> "000937"
            if 'CoPESD/' in image_path:
                video_id = image_path.split('CoPESD/')[-1].split('/')[0]
            else:
                video_id = "unknown"
            
            # Skip items that do not belong to the requested split
            if video_id not in allowed_video_ids:
                continue
            
            # Extract instruction and response
            instruction = ""
            response = ""
            for conv in conversations:
                if conv.get('from') == 'human':
                    instruction = conv.get('value', '')
                elif conv.get('from') in ['gpt', 'assistant']:
                    response = conv.get('value', '')
            
            if not instruction or not response:
                continue
            
            # Determine task type from tags
            main_tag = item.get('main_tag', '')
            sub_tag = item.get('sub_tag', '')
            
            task_type = "vqa"
            if 'phase' in sub_tag.lower():
                task_type = "phase"
            elif 'instrument' in sub_tag.lower() or 'tool' in sub_tag.lower():
                task_type = "instrument"
            elif 'action' in sub_tag.lower():
                task_type = "action"
            
            # Clean image path. With a universal image_root of "data", the
            # path must be "raw/Surg-396k/CoPESD/CoPESD/000937/0001.jpg" so it
            # resolves to "data/raw/Surg-396k/CoPESD/CoPESD/000937/0001.jpg".
            norm = image_path.replace("\\", "/")
            marker = "CoPESD/"
            if marker in norm:
                rel_image = f"raw/Surg-396k/CoPESD/CoPESD/{norm.split(marker)[-1]}"
            else:
                # No CoPESD marker: preserve whatever directory structure
                # remains after the dataset root rather than flattening to a
                # basename (which would never resolve on disk).
                tail = norm
                if "Surg-396k/" in norm:
                    tail = norm.split("Surg-396k/")[-1].lstrip("/")
                rel_image = f"raw/Surg-396k/CoPESD/CoPESD/{tail.lstrip('/')}"
            
            # Use the original conversations format
            entry = {
                "image": rel_image,
                "conversations": conversations,
                "task": task_type,
                "dataset": "surg396k",
                "video_id": video_id,
                "main_tag": main_tag,
                "sub_tag": sub_tag,
            }
            
            f.write(json.dumps(entry) + '\n')
            count += 1
    
    return count


# ============================================================
# Main Conversion Pipeline
# ============================================================

def convert_all_datasets(
    data_root: str = "data/raw",
    output_dir: str = "data/processed/vlm_jsonl",
    splits: List[str] = None,
) -> Dict[str, int]:
    """Run conversion for all datasets."""
    splits = splits or ["train", "val", "test"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for split in splits:
        print(f"\n=== Converting {split} split ===")
        
        # Cholec80
        out_file = output_path / f"cholec80_phases_vlm_{split}.jsonl"
        count = convert_cholec80(
            root_path=f"{data_root}/cholec80",
            output_path=str(out_file),
            split=split,
        )
        results[f"cholec80_{split}"] = count
        print(f"  Cholec80 {split}: {count} samples")
        
        # CholecT50
        out_file = output_path / f"cholect50_triplets_vlm_{split}.jsonl"
        count = convert_cholect50(
            root_path=f"{data_root}/CholecT50",
            output_path=str(out_file),
            split=split,
        )
        results[f"cholect50_{split}"] = count
        print(f"  CholecT50 {split}: {count} samples")
        
        # HeiChole
        out_file = output_path / f"heichole_multitask_vlm_{split}.jsonl"
        count = convert_heichole(
            root_path=f"{data_root}/HeiChole",
            output_path=str(out_file),
            split=split,
        )
        results[f"heichole_{split}"] = count
        print(f"  HeiChole {split}: {count} samples")
        
        # Surg-396K
        out_file = output_path / f"surg396k_vlm_{split}.jsonl"
        count = convert_surg396k(
            root_path=f"{data_root}/Surg-396k",
            output_path=str(out_file),
            split=split,
        )
        results[f"surg396k_{split}"] = count
        print(f"  Surg-396K {split}: {count} samples")
    
    # Save summary
    summary_file = output_path / "conversion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Summary saved to {summary_file}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert surgical datasets to unified JSONL format")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Root data directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/vlm_jsonl", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to convert")
    parser.add_argument("--dataset", type=str, default="all", help="Specific dataset to convert")
    
    args = parser.parse_args()
    
    if args.dataset != "all":
        # Convert only specific dataset
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split in args.splits:
            print(f"\n=== Converting {split} split for {args.dataset} ===")
            if args.dataset == "cholec80":
                out_file = output_path / f"cholec80_phases_vlm_{split}.jsonl"
                count = convert_cholec80(
                    root_path=f"{args.data_root}/cholec80",
                    output_path=str(out_file),
                    split=split,
                )
            elif args.dataset == "cholect50":
                out_file = output_path / f"cholect50_triplets_vlm_{split}.jsonl"
                count = convert_cholect50(
                    root_path=f"{args.data_root}/CholecT50",
                    output_path=str(out_file),
                    split=split,
                )
            elif args.dataset == "heichole":
                out_file = output_path / f"heichole_multitask_vlm_{split}.jsonl"
                count = convert_heichole(
                    root_path=f"{args.data_root}/hei-chole",
                    output_path=str(out_file),
                    split=split,
                )
            elif args.dataset == "surg396k":
                out_file = output_path / f"surg396k_vlm_{split}.jsonl"
                count = convert_surg396k(
                    root_path=f"{args.data_root}/Surg-396k",
                    output_path=str(out_file),
                    split=split,
                )
            else:
                print(f"Unknown dataset: {args.dataset}")
                continue
            print(f"  {args.dataset} {split}: {count} samples")
    else:
        convert_all_datasets(
            data_root=args.data_root,
            output_dir=args.output_dir,
            splits=args.splits,
        )