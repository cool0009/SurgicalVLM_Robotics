#!/usr/bin/env python3
"""
Extract Frames from Surgical Videos
Batch frame extraction for all datasets at specified FPS.
Supports uniform sampling and keyframe extraction.
"""

import cv2
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def extract_frames_uniform(
    video_path: Path,
    output_dir: Path,
    fps: int = 1,
    max_frames: Optional[int] = None,
    quality: int = 95,
) -> List[dict]:
    """
    Extract frames at uniform FPS interval.
    
    Returns list of frame metadata dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path} — file may be missing, corrupt, "
            "or OpenCV was built without the required video backend (ffmpeg/gstreamer)."
        )
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))
    
    frame_metadata = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_name
            
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            frame_metadata.append({
                "frame_name": frame_name,
                "frame_idx": frame_count,
                "timestamp": frame_count / video_fps,
                "video_fps": video_fps,
            })
            
            saved_count += 1
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frame_metadata


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.3,
    max_frames: Optional[int] = None,
    quality: int = 95,
) -> List[dict]:
    """
    Extract keyframes using scene change detection (histogram difference).
    
    Returns list of frame metadata dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path} — file may be missing, corrupt, "
            "or OpenCV lacks the video backend (ffmpeg/gstreamer)."
        )
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    
    frame_metadata = []
    saved_count = 0
    prev_hist = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Compute color histogram
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        is_keyframe = False
        if prev_hist is not None:
            # Correlation distance (1.0 = identical, -1.0 = opposite)
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < threshold:
                is_keyframe = True
        else:
            is_keyframe = True  # First frame is always a keyframe
        
        if is_keyframe:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = output_dir / frame_name
            
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            frame_metadata.append({
                "frame_name": frame_name,
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps,
                "video_fps": video_fps,
                "is_keyframe": True,
                "hist_correlation": float(correlation) if prev_hist is not None else 1.0,
            })
            
            saved_count += 1
            if max_frames and saved_count >= max_frames:
                break
        
        prev_hist = hist
    
    cap.release()
    return frame_metadata


def extract_sliding_window(
    video_path: Path,
    output_dir: Path,
    window_size: int = 8,
    stride: int = 4,
    fps: int = 1,
    quality: int = 95,
) -> List[dict]:
    """
    Extract frames for sliding window temporal modeling.
    Returns overlapping windows of frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path} — file may be missing, corrupt, "
            "or OpenCV lacks the ffmpeg backend."
        )
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))
    
    frame_metadata = []
    frame_count = 0
    
    # First, extract all frames at target FPS
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            all_frames.append({
                "frame_name": frame_name,
                "frame_idx": frame_count,
                "timestamp": frame_count / video_fps,
            })
        
        frame_count += 1
    
    cap.release()
    
    # Create sliding windows
    for i in range(0, len(all_frames) - window_size + 1, stride):
        window = all_frames[i:i + window_size]
        for j, frame_info in enumerate(window):
            frame_metadata.append({
                **frame_info,
                "window_idx": i,
                "position_in_window": j,
            })
    
    return frame_metadata


def process_video(
    video_path: Path,
    output_root: Path,
    dataset_name: str,
    video_id: str,
    strategy: str = "uniform",
    fps: int = 1,
    max_frames: Optional[int] = None,
    quality: int = 95,
    keyframe_threshold: float = 0.3,
) -> dict:
    """Process a single video and return metadata."""
    
    output_dir = output_root / dataset_name / video_id
    
    if strategy == "uniform":
        frames = extract_frames_uniform(video_path, output_dir, fps, max_frames, quality)
    elif strategy == "keyframe":
        frames = extract_keyframes(video_path, output_dir, keyframe_threshold, max_frames, quality)
    elif strategy == "sliding":
        frames = extract_sliding_window(video_path, output_dir, fps=fps, quality=quality)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Save metadata
    metadata = {
        "video_id": video_id,
        "dataset": dataset_name,
        "video_path": str(video_path),
        "strategy": strategy,
        "fps": fps,
        "num_frames": len(frames),
        "frames": frames,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract frames from surgical videos")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Root data directory")
    parser.add_argument("--output-root", type=str, default="data/frames", help="Output directory for frames")
    parser.add_argument("--dataset", type=str, 
                        choices=["cholec80", "heichole", "cholect50", "all"],
                        default="all", help="Dataset to process")
    parser.add_argument("--strategy", type=str, 
                        choices=["uniform", "keyframe", "sliding"],
                        default="uniform", help="Frame extraction strategy")
    parser.add_argument("--fps", type=int, default=1, help="Target FPS for uniform extraction")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--keyframe-threshold", type=float, default=0.3, help="Keyframe detection threshold")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Splits directory")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    splits_dir = Path(args.splits_dir)
    
    # Define dataset paths
    datasets = {
        "cholec80": data_root / "cholec80" / "videos",
        "heichole": data_root / "hei-chole",  # videos directly in folder
        "cholect50": data_root / "CholecT50" / "videos",
    }
    
    datasets = {k: v for k, v in datasets.items() if v.exists()}
    
    if not datasets:
        sys.exit(
            f"ERROR: no datasets matched under data_root={args.data_root}. "
            "Check paths in the 'datasets' dict."
        )
    
    if args.dataset != "all":
        datasets = {args.dataset: datasets.get(args.dataset)}
        if not datasets[args.dataset]:
            sys.exit(
                f"ERROR: dataset '{args.dataset}' not found at expected path. "
                f"Looked under data_root={args.data_root}."
            )
    
    all_metadata = {}
    
    for dataset_name, videos_dir in datasets.items():
        print(f"\n=== Processing {dataset_name} ===")
        
        # Get video list
        if dataset_name == "cholect50":
            # These datasets have folders of pre-extracted frames
            video_dirs = [d for d in videos_dir.iterdir() if d.is_dir()]
            
            # Filter by split if specified
            if args.split != "all" and (splits_dir / f"{dataset_name}_splits.json").exists():
                with open(splits_dir / f"{dataset_name}_splits.json", "r") as f:
                    splits = json.load(f)
                split_videos = set(splits.get(args.split, []))
                video_dirs = [d for d in video_dirs if d.name in split_videos]
            
            print(f"Found {len(video_dirs)} video folders for split '{args.split}'")
            if not video_dirs:
                raise RuntimeError(
                    f"{dataset_name}: 0 video folders matched under {videos_dir} "
                    f"(split='{args.split}'). Aborting so we don't emit empty metadata."
                )
            # For these datasets, frames already exist - just create metadata
            for video_dir in video_dirs:
                frames = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
                if frames:
                    metadata = {
                        "video_id": video_dir.name,
                        "dataset": dataset_name,
                        "strategy": "pre_extracted",
                        "num_frames": len(frames),
                        "frames": [{"frame_name": f.name, "frame_idx": i} for i, f in enumerate(frames)],
                    }
                    output_dir = output_root / dataset_name / video_dir.name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    all_metadata[f"{dataset_name}/{video_dir.name}"] = metadata
            continue
        
        # For video-based datasets, load splits if available
        if args.split != "all" and (splits_dir / f"{dataset_name}_splits.json").exists():
            with open(splits_dir / f"{dataset_name}_splits.json", "r") as f:
                splits = json.load(f)
            video_names = splits.get(args.split, [])
        else:
            # Get all videos
            video_names = [f.stem for f in videos_dir.glob("*.mp4")]
        
        # Process videos
        video_paths = []
        for video_name in video_names:
            # Handle HeiChole naming: video files are like Hei-Chole1_dissection.mp4
            # Match the video_name as a whole token. The naive glob "Hei-Chole1*.mp4"
            # also matches Hei-Chole10_dissection.mp4 (ASCII '0' < '_'), which would
            # silently extract the wrong video's frames. Only accept exact stems or
            # stems starting with "<video_name>_".
            matching_videos = [
                v
                for v in videos_dir.glob(f"{video_name}*.mp4")
                if v.stem == video_name or v.stem.startswith(f"{video_name}_")
            ]
            if matching_videos:
                # Prefer _dissection over _calot if both exist
                dissection_videos = [v for v in matching_videos if "_dissection" in v.stem]
                video_path = dissection_videos[0] if dissection_videos else matching_videos[0]
                video_paths.append((video_path, video_name))
            else:
                # Fallback: exact match
                video_path = videos_dir / f"{video_name}.mp4"
                if video_path.exists():
                    video_paths.append((video_path, video_name))
        
        print(f"Processing {len(video_paths)} videos with {args.workers} workers...")
        if not video_paths:
            raise RuntimeError(
                f"{dataset_name}: 0 videos matched in {videos_dir} "
                f"(split='{args.split}'). Aborting so we don't emit empty metadata."
            )
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for video_path, video_name in video_paths:
                future = executor.submit(
                    process_video,
                    video_path,
                    output_root,
                    dataset_name,
                    video_name,
                    args.strategy,
                    args.fps,
                    args.max_frames,
                    args.quality,
                    args.keyframe_threshold,
                )
                futures[future] = video_name
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
                video_name = futures[future]
                try:
                    metadata = future.result()
                    all_metadata[f"{dataset_name}/{video_name}"] = metadata
                except Exception as e:
                    print(f"Error processing {video_name}: {e}")
    
    # Save global metadata
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "extraction_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    total_frames = sum(m.get("num_frames", 0) for m in all_metadata.values())
    print(f"\n=== Extraction Complete ===")
    print(f"Total videos processed: {len(all_metadata)}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Metadata saved to: {output_root / 'extraction_metadata.json'}")
    
    if len(all_metadata) == 0:
        sys.exit("ERROR: no videos were processed — extraction produced zero videos.")
    if total_frames == 0:
        sys.exit("ERROR: extraction produced zero frames — check video backends/sources.")


if __name__ == "__main__":
    main()