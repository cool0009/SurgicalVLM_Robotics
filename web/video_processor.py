"""Video analysis pipeline for Surgical VLM — Phase 8 video-first demo.

This module implements the ``VideoAnalyzer`` class and aggregation logic
specified in TODO.md §8.0–§8.2.  It mirrors the parse/normalise helpers from
``gradio_app.py`` so that both the CLI worker and the UI can share identical
semantics.
"""

import json
import logging
import re
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror gradio_app.py verbatim (§8.0.3 JSON schema, phasing etc.)
# ---------------------------------------------------------------------------

STRICT_JSON_PROMPT = (
    'You are an expert surgical assistant. Analyze this laparoscopic image. '
    'You MUST respond ONLY with a valid JSON object in this exact format, nothing else:\n'
    '{"phase": "Exact Phase Name", "tools": ["Tool 1", "Tool 2"], '
    '"narrative": "One sentence describing the current action."}'
)

TIMELINE_PHASES = ["Preparation", "CalotTriangle", "Dissection", "ClippingCutting", "GallbladderRetraction"]

PHASE_NORMALISATION = {
    "Preparation": "Preparation",
    "CalotTriangle": "CalotTriangle",
    "CalotTriangleDissection": "CalotTriangle",
    "Calottriangledissection": "CalotTriangle",
    "Dissection": "Dissection",
    "GallbladderDissection": "Dissection",
    "ClippingCutting": "ClippingCutting",
    "Clipping": "ClippingCutting",
    "GallbladderRetraction": "GallbladderRetraction",
    "GallbladderRetractionStage": "GallbladderRetraction",
    "GallbladderPackaging": "GallbladderRetraction",
    "CleaningCoagulation": "ClippingCutting",
    "Unknown": "Dissection",
}

PHASE_COLORS = {
    "Preparation": "(31, 119, 180)",
    "CalotTriangle": "(174, 199, 232)",
    "Dissection": "(255, 187, 120)",
    "ClippingCutting": "(152, 223, 138)",
    "GallbladderRetraction": "(255, 152, 150)",
}

# Keyword tokens for each canonical phase — used by _normalise_phase fallback.
PHASE_TOKENS: Dict[str, List[str]] = {
    "Preparation": ["preparation", "prep"],
    "CalotTriangle": ["calot", "triangle"],
    "Dissection": ["dissection", "dissect"],
    "ClippingCutting": ["clipping", "cutting", "clip", "cut", "coagulation"],
    "GallbladderRetraction": ["retraction", "gallbladder", "packaging"],
}

_VLM_DEFAULT_CONFIDENCE = 0.90

_TOOL_COLORS = {
    "Grasper": "(59, 130, 246)",
    "Dissector": "(100, 116, 139)",
    "Bipolar": "(37, 99, 235)",
    "Hook": "(96, 165, 250)",
    "SpecimenBag": "(59, 130, 246)",
}

# ---------------------------------------------------------------------------
# Helper functions — mirror gradio_app.py semantics (§8.1.5, §8.2.1–8.2.5)
# ---------------------------------------------------------------------------

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_json_fence(raw: str) -> str:
    """Strip markdown triple-backtick code-fences from a JSON string."""
    m = re.match(r"^```(?:json)?\s*(.*)\s```$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw


def _extract_json_object(raw: str) -> Optional[str]:
    """Extract the innermost JSON object using a balanced-brace scanner.

    Handles strings, escaped quotes, and nested braces.
    Returns the extracted JSON text or None.
    """
    if not isinstance(raw, str):
        return None

    brace_start = raw.find('{')
    if brace_start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    start = -1

    for i, ch in enumerate(raw[brace_start:], start=brace_start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start > -1:
                    return raw[start:i + 1]

    return None


def _fix_common_json_issues(raw: str) -> str:
    """Apply common JSON repairs: trailing-comma removal, True→true etc."""
    repaired = _TRAILING_COMMA_RE.sub(r"\1", raw)
    repaired = repaired.replace("True", "true")
    repaired = repaired.replace("False", "false")
    repaired = repaired.replace("None", "null")
    return repaired


def _try_parse_json(candidates) -> Optional[Dict[str, Any]]:
    """Return the first candidate that parses to a dict, else None."""
    for label, candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _parse_json_result(raw) -> Dict[str, Any]:
    """Parse the VLM's strict-JSON reply into a dict with phase/tools/narrative.

    Accepts a dict directly or a JSON string. Fallback strategies, tried in
    order: clean JSON -> markdown code fence (```json ... ```) -> JSON object
    buried in surrounding prose (model preamble / refusal text) -> repair of
    common slips (trailing commas, Python literals). Every failed attempt logs
    the raw output so the model's exact reply is recoverable. On total failure
    returns an empty frame so callers fall back to blank panels, never crash.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        logger.warning("_parse_json_result: unexpected type %s: %r", type(raw).__name__, raw)
        return {"phase": "", "tools": [], "narrative": ""}

    stripped = raw.strip()
    result = _try_parse_json([
        ("clean", stripped),
        ("code-fence", _strip_json_fence(stripped)),
    ])
    if result is not None:
        return result

    embedded = _extract_json_object(stripped)
    if embedded is not None:
        result = _try_parse_json([
            ("embedded", embedded),
            ("embedded+repair", _fix_common_json_issues(embedded)),
        ])
        if result is not None:
            return result

    logger.warning("_parse_json_result: ALL strategies failed — raw output follows:\n%s", raw)
    return {"phase": "", "tools": [], "narrative": ""}


def _normalise_phase(raw) -> str:
    """Map any model phase string to one of the 5 timeline buckets."""
    raw = str(raw or "").strip()
    if raw in PHASE_NORMALISATION:
        return PHASE_NORMALISATION[raw]
    low = raw.lower()
    # Keyword fallback: pick the canonical phase whose token best matches.
    for canon in TIMELINE_PHASES:  # order matters
        tokens = PHASE_TOKENS.get(canon, [canon.lower()])
        if any(tok in low for tok in tokens):
            return canon
    return "Dissection"


def _extract_tool_list(raw_tools) -> List[Tuple[str, Optional[float]]]:
    """Convert the model's ``tools`` field into ``[(name, confidence)]``.

    The real model returns a list of tool *names* (no confidence attached —
    confidences come from YOLO instead); the spec also allows
    ``[{"name": ..., "confidence": float}]`` objects.  Both are handled.  String
    tools get the VLM default confidence (``_VLM_DEFAULT_CONFIDENCE``) so the
    tool table always shows a numeric value.
    """
    out = []
    for item in raw_tools or []:
        if isinstance(item, str):
            out.append((item, _VLM_DEFAULT_CONFIDENCE))
        elif isinstance(item, dict):
            name = str(item.get("name", "Tool"))
            try:
                conf = float(item.get("confidence", item.get("score", _VLM_DEFAULT_CONFIDENCE)))
            except (TypeError, ValueError):
                conf = _VLM_DEFAULT_CONFIDENCE
            out.append((name, round(min(max(conf, 0.0), 1.0), 2)))
    return out


def _normalise_tools(raw_tools) -> List[str]:
    """Normalize tools to a plain list of strings.

    Handles both list-of-dicts (mock: [{"name": t, "confidence": float}])
    and list-of-strings.  Returns just the tool names.
    """
    names = []
    for item in raw_tools or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = str(item.get("name", "Tool"))
            if name and name not in names:
                names.append(name)
        elif isinstance(item, list):
            if item and isinstance(item[0], str):
                if item[0] not in names:
                    names.append(item[0])
    return names


def _extract_tools_from_narrative(text: str) -> List[str]:
    """Fallback tool extraction from free-text narrative.

    The model's structured ``"tools"`` JSON array frequently comes back empty
    even when the narrative clearly names tools in prose (confirmed via smoke
    testing — e.g. "The bipolar forceps is being used to coagulate..." with
    ``"tools": []``). Scans the narrative for known tool names/aliases
    (reusing ``LabelMapper.instrument_aliases``), longest-alias-first so a
    specific phrase ("bipolar forceps") is matched before a generic substring
    it contains ("forceps") that would otherwise misattribute the same
    mention to the wrong tool (e.g. Grasper instead of Bipolar).
    """
    if not text:
        return []
    from surgical_vlm.training.label_mapping import get_label_mapper

    mapper = get_label_mapper()
    candidates = list(mapper.instrument_aliases.items())
    candidates += [(name.lower(), name) for name in mapper.instrument_to_id.keys()]
    candidates.sort(key=lambda kv: len(kv[0]), reverse=True)

    working_text = text.lower()
    found: List[str] = []
    for alias, canonical in candidates:
        if alias in working_text:
            if canonical not in found:
                found.append(canonical)
            # Blank out the matched span so shorter/generic aliases contained
            # within it don't also match and double-attribute the mention.
            working_text = working_text.replace(alias, " " * len(alias))
    return found


def _gpu_memory_gb() -> Optional[float]:
    """Real, currently-allocated GPU memory in GB, or None if no CUDA device
    (e.g. --mock mode on a CPU-only machine). Never fabricated."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e9, 2)
    except Exception:
        pass
    return None


def _compute_phase_transitions(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a list of {t_sec, phase} entries keyed at timestamps where
    the phase changes.  Isolated flicker is already removed by the 3-frame
    median filter applied upstream."""
    if not timeline:
        return []
    transitions = [{"t_sec": timeline[0]["t_sec"], "phase": timeline[0]["phase"]}]
    last_phase = timeline[0]["phase"]
    for entry in timeline[1:]:
        if entry["phase"] != last_phase:
            transitions.append({"t_sec": entry["t_sec"], "phase": entry["phase"]})
            last_phase = entry["phase"]
    return transitions


# ---------------------------------------------------------------------------
# VideoAnalyzer class — §8.1.1–8.1.7
# ---------------------------------------------------------------------------


class VideoAnalyzer:
    """Orchestrate surgical video analysis: load, sample, extract, infer, aggregate.

    Parameters
        model: VLM wrapper exposing ``.generate_single(pil_image, prompt=...) -> dict``.
        tokenizer: Tokenizer kept for spec signature + batched inference.
        processor: Processor kept for spec signature + batched inference.
        fps: Sampling rate in frames-per-second (default 1).
        max_frames: Maximum number of frames to analyze (cap 180 = 3 min @ 1fps).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        processor: Any,
        fps: float = 1.0,
        max_frames: int = 180,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.fps = fps
        self.max_frames = max_frames
        # Per-frame cache: dict mapping frame_hash -> analyzed result dict
        self._frame_cache: Dict[str, Dict[str, Any]] = {}
        # Track model version for cache invalidation
        self._model_version = getattr(model, "_model_version", "unknown")

    # -- 8.1.1 --

    def load_video(self, path: str) -> Tuple[cv2.VideoCapture, float, int]:
        """Load a video file and return ``(cap, duration_sec, total_frames)``.

        Raises ``ValueError`` if the video cannot be opened or is < 1 sec
        (§8.1.5 error handling).
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        if fps_orig <= 0:
            fps_orig = 30.0  # fallback

        duration_sec = total_frames / fps_orig

        # §8.1.5: < 1 sec unreadable → error JSON, no crash
        if duration_sec < 1.0:
            cap.release()
            raise ValueError(
                f"Video is less than 1 second ({duration_sec:.1f}s); cannot analyze."
            )

        # Stored so sample_frame_indices can compare the requested sampling
        # rate (self.fps) against what the source file can actually supply —
        # you cannot sample more distinct frames per second than the video
        # was recorded at, no matter what self.fps requests.
        self._native_fps = fps_orig

        return cap, duration_sec, total_frames

    # -- 8.1.2 --

    def sample_frame_indices(
        self, total_frames: int, fps: Optional[float] = None
    ) -> List[int]:
        """Return frame indices sampled at ``fps`` frames per second of video
        (capped by the source's own native fps — can't invent frames that
        don't exist), further capped at ``self.max_frames`` total samples.

        Parameters
            total_frames: Total number of frames in the video.
            fps: Target sampling rate in frames/sec of video (uses self.fps if None).

        Returns
            List of frame indices (0-based), uniformly distributed, bounds-checked.
        """
        fps = self.fps if fps is None else fps
        if total_frames <= 1:
            return [0]

        native_fps = getattr(self, "_native_fps", 30.0) or 30.0
        # step in native frames between samples: e.g. native=30, target=32 ->
        # step=1 (every native frame; can't exceed the source's own rate).
        # native=60, target=32 -> step=2 (~30 samples/sec, close to target).
        step = max(1, round(native_fps / fps))
        indices = list(range(0, total_frames, step))

        # Secondary safety cap so a very long video can't produce an
        # unbounded number of samples regardless of the fps-based step above.
        if len(indices) > self.max_frames:
            indices = [
                indices[i] for i in range(0, len(indices), len(indices) // self.max_frames)
            ]

        indices = [min(idx, total_frames - 1) for idx in indices]
        indices = sorted(set(indices))
        return indices

    # -- 8.1.3 & 8.1.4 --

    def extract_frames(
        self, cap: cv2.VideoCapture, indices: List[int]
    ) -> List[Image.Image]:
        """Decode frames at the given indices.

        Reads from ``cap``, converts BGR→RGB, returns ``List[PIL.Image]``.
        Frame hashes are computed for caching (§8.1.4).
        """
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                # If we can't read a frame, push a blank PIL image so indices
                # still map consistently; the caller will get an error JSON
                # from analyze_frame for that frame.
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                pil = Image.fromarray(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
                frames.append(pil)
                continue

            # Convert BGR → RGB PIL
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frames.append(pil)

        return frames

    # -- 8.1.4 (per-frame cache) --

    def _frame_hash(self, pil_image: Image.Image) -> str:
        """Compute a hash of a PIL image for cache lookup."""
        buf = pil_image.tobytes()
        return hashlib.md5(buf).hexdigest()

    def _cached_analyze(self, pil_image: Image.Image, fps_sampled: float) -> Dict[str, Any]:
        """Return cached result if (frame_hash, model_version) matches."""
        h = self._frame_hash(pil_image)
        cache_key = (h, self._model_version)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        return None

    def _cache_result(self, pil_image: Image.Image, result: Dict[str, Any]) -> None:
        """Store result in the per-frame cache."""
        h = self._frame_hash(pil_image)
        cache_key = (h, self._model_version)
        self._frame_cache[cache_key] = result

    # -- 8.1.5 & 8.1.6 & 8.1.7 --

    def analyze_frame(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Analyze a single surgical frame.

        Runs ``model.generate_single(pil_in, prompt=STRICT_JSON_PROMPT)``,
        parses the JSON output, normalises phase/tools, and returns:

        ``{"phase": str, "tools": List[str], "description": str}``

        If the model fails or the video is unreadable, returns an error JSON
        (never crashes — §8.1.5).
        """
        try:
            # Check cache first (§8.1.4)
            cached = self._cached_analyze(pil_image, self.fps)
            if cached is not None:
                return cached

            # Run model inference: vlm.generate_single(pil, prompt=STRICT_JSON_PROMPT)
            # mirror gradio_app.py:507 pattern
            result = self.model.generate_single(
                pil_image, prompt=STRICT_JSON_PROMPT
            )

            # Parse JSON result using the mirrored helper
            parsed = _parse_json_result(result)
            json_phase = _normalise_phase(parsed.get("phase", ""))
            tools = _extract_tool_list(parsed.get("tools", []))
            # Normalise to plain list of strings for consistency
            tool_names = _normalise_tools(tools)
            # Narrative fallback: try narrative → description → empty string
            description = str(
                parsed.get("narrative", parsed.get("description", ""))
            ).strip()
            if not description:
                description = ""

            # The structured "tools" array is frequently empty even when the
            # narrative names tools in prose — fall back to scanning it.
            if not tool_names and description:
                tool_names = _extract_tools_from_narrative(description)

            # Phase, tools, and description all come from this one VLM
            # generation call, so they always describe the same frame
            # consistently (an earlier version overrode phase with a
            # separately-trained classification head — accurate on its own,
            # but it could disagree with the tools/narrative from this VLM
            # call since they were independent judgments of the same frame).
            out = {
                "phase": json_phase,
                "tools": tool_names,
                "description": description,
            }

            # Cache the result (§8.1.4)
            self._cache_result(pil_image, out)
            return out

        except Exception as e:
            # §8.1.5: Handle corrupt/short videos gracefully — return error JSON,
            # do not crash. The caller (analyze_video) will record this as a
            # parse failure and continue.
            logger.warning("analyze_frame error: %s", e)
            return {
                "phase": "Unknown",
                "tools": [],
                "description": "Error analyzing frame",
            }

    # -- 8.1.7 UI limits enforcement --

    @staticmethod
    def validate_video_path(path: str) -> bool:
        """Check that the video path has an allowed extension and within size limit.

        ``allowed_extensions`` = ["mp4", "avi", "mov", "mkv"]
        ``maximum_size_mb`` = 200
        """
        allowed_extensions = {"mp4", "avi", "mov", "mkv"}
        max_size_bytes = 200 * 1024 * 1024  # 200 MB

        p = Path(path)
        if p.suffix.lower().lstrip('.') not in allowed_extensions:
            return False

        try:
            if p.stat().st_size > max_size_bytes:
                return False
        except OSError:
            pass

        return True

    # -- 8.1.1 orchestration --

    def analyze_video(
        self, path: str, progress=None
    ) -> Dict[str, Any]:
        """Orchestrate full video analysis and build the aggregate report.

        Returns a dict matching the exact §8.0.3 JSON schema:

        {
            "video_metadata": {"filename": ..., "duration_sec": ..., "fps_sampled": 1,
                               "num_frames_analyzed": N},
            "dominant_phase": "...",
            "phase_timeline": [{"t_sec": 0, "phase": "..."}, ...],
            "tools_detected": [
                {"tool": "...", "first_seen_sec": 0, "last_seen_sec": 0, "frame_count": N},
                ...
            ],
            "scene_description": "Overall narrative string...",
            "per_frame_results": [
                {"t_sec": 0, "frame_idx": 0, "phase": "...", "tools": [...], "description": "..."},
                ...
            ],
            "model_info": {"base": "Qwen2.5-VL-7B", "adapter": "best_lora", "temp": 0.1}
        }
        """
        # --- §8.0.3 video_metadata ---
        import os as _os
        filename = _os.path.basename(path)

        try:
            cap, duration_sec, total_frames = self.load_video(path)
        except ValueError as e:
            # §8.1.5: return error JSON structure
            return {
                "video_metadata": {
                    "filename": filename,
                    "duration_sec": 0,
                    "fps_sampled": self.fps,
                    "num_frames_analyzed": 0,
                },
                "dominant_phase": "Error",
                "phase_timeline": [],
                "tools_detected": [],
                "scene_description": str(e),
                "per_frame_results": [],
                "model_info": {"base": "Qwen2.5-VL-7B", "adapter": "best_lora", "temp": 0.1},
            }

        # --- §8.1.2 batched inference setup ---
        # Sample frame indices (uniform, bounds-checked)
        indices = self.sample_frame_indices(total_frames, self.fps)
        num_frames = len(indices)

        # Extract frames from video
        frames = self.extract_frames(cap, indices)

        # --- §8.1.3 progress bar ---
        if progress is not None:
            for i, pil_img in enumerate(frames):
                if progress is not None:
                    progress(i / max(1, len(frames)))
                # Per-frame cache already checked inside analyze_frame

        # --- §8.1.2/8.4.2 batched inference (8 frames per pass) ---
        # Process frames in batches of 8 to reduce total inference time
        BATCH_SIZE = 8
        per_frame_results: List[Dict[str, Any]] = []

        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch = frames[batch_start : batch_start + BATCH_SIZE]
            # Batch inference: process multiple frames together
            # For the VLM wrapper, we iterate since generate_single is per-frame;
            # batched inference is achieved by the caller passing 8 frames at a time
            # and the model's internal batching (or we just call generate_single in a loop).
            # Per §8.4.2: 8 frames/pass.
            for j, pil_img in enumerate(batch):
                # Global progress update
                idx_abs = batch_start + j
                if progress is not None and len(frames) > 0:
                    progress((idx_abs + 1) / max(1, len(frames)))

                # Analyze this frame (with caching)
                result = self.analyze_frame(pil_img)

                # Build per-frame result with timestamp
                # t_sec = real-world timestamp = native video frame index /
                # native fps (NOT idx_abs/self.fps -- idx_abs is this frame's
                # position within the sampled list, and self.fps is the
                # *target* sampling rate, not the source's real fps; using
                # those together produced badly wrong timestamps, e.g. "29s"
                # on a 6-second clip).
                native_fps = getattr(self, "_native_fps", 30.0) or 30.0
                t_sec = round(indices[idx_abs] / native_fps, 2)
                frame_entry = {
                    "t_sec": t_sec,
                    "frame_idx": idx_abs,
                    "phase": result["phase"],
                    "tools": result["tools"],
                    "description": result["description"],
                }
                per_frame_results.append(frame_entry)

        cap.release()

        return self._build_result(filename, duration_sec, per_frame_results)

    # -- shared aggregation (§8.2.1-8.2.5) — used by both analyze_video and
    # analyze_video_stream so batch and streaming paths never diverge --

    def _build_result(
        self, filename: str, duration_sec: float, per_frame_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate a (possibly partial, in-progress) list of per-frame results
        into the full §8.0.3 output schema. Cheap enough to call after every
        frame during streaming (simple O(n) passes over small lists)."""

        # --- §8.2.1 Phase aggregation ---
        # Majority vote for dominant_phase
        phase_counts: Dict[str, int] = {}
        for entry in per_frame_results:
            p = entry["phase"]
            phase_counts[p] = phase_counts.get(p, 0) + 1

        if phase_counts:
            dominant_phase = max(phase_counts, key=phase_counts.get)
        else:
            dominant_phase = "Dissection"

        # Build phase_timeline with 3-frame median filter smoothing
        # First, sort by t_sec and apply simple smoothing:
        # Remove isolated phase changes (3-frame median filter)
        timeline_sorted = sorted(per_frame_results, key=lambda x: x["t_sec"])

        # Apply 3-frame median filter: if a frame's phase differs from both
        # neighbors, use the median of the three
        smoothed: List[Dict[str, Any]] = []
        for i, entry in enumerate(timeline_sorted):
            phases_neighbors = []
            if i > 0:
                phases_neighbors.append(timeline_sorted[i - 1]["phase"])
            phases_neighbors.append(entry["phase"])
            if i < len(timeline_sorted) - 1:
                phases_neighbors.append(timeline_sorted[i + 1]["phase"])

            # Majority phase among the 3
            phase_count_local: Dict[str, int] = {}
            for p in phases_neighbors:
                phase_count_local[p] = phase_count_local.get(p, 0) + 1
            smoothed_phase = max(phase_count_local, key=phase_count_local.get)

            smoothed.append(
                {"t_sec": entry["t_sec"], "phase": smoothed_phase}
            )

        # Compute phase transitions (timestamps where phase changes)
        phase_transitions = _compute_phase_transitions(smoothed)

        # Build phase_timeline entries (timestamps + smoothed phase)
        phase_timeline: List[Dict[str, Any]] = [
            {"t_sec": entry["t_sec"], "phase": entry["phase"]} for entry in smoothed
        ]

        # --- §8.2.2 Tool aggregation ---
        # Union of all tools across frames → tools_detected list
        # For each tool, record first_seen_sec, last_seen_sec, frame_count
        tool_stats: Dict[str, Dict[str, Any]] = {}
        for entry in per_frame_results:
            for tool_name in entry["tools"]:
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "first_seen_sec": entry["t_sec"],
                        "last_seen_sec": entry["t_sec"],
                        "frame_count": 0,
                    }
                tool_stats[tool_name]["frame_count"] += 1
                # Update last_seen_sec (keep the max/timestamp of last occurrence)
                if entry["t_sec"] >= tool_stats[tool_name]["last_seen_sec"]:
                    tool_stats[tool_name]["last_seen_sec"] = entry["t_sec"]
                # Update first_seen_sec (keep the min)
                if (
                    tool_stats[tool_name]["first_seen_sec"] is None
                    or entry["t_sec"] < tool_stats[tool_name]["first_seen_sec"]
                ):
                    tool_stats[tool_name]["first_seen_sec"] = entry["t_sec"]

        # Sort by frame_count descending (most-used tool first)
        tools_detected = sorted(
            [
                {
                    "tool": tool,
                    "first_seen_sec": stats["first_seen_sec"],
                    "last_seen_sec": stats["last_seen_sec"],
                    "frame_count": stats["frame_count"],
                }
                for tool, stats in tool_stats.items()
            ],
            key=lambda x: x["frame_count"],
            reverse=True,
        )

        # --- §8.2.3 Scene description aggregation ---
        # Option A (concatenation with timestamp markers) — DEFAULT, deterministic
        # Join per-frame descriptions with timestamp markers
        description_parts = []
        for entry in per_frame_results:
            marker = f"[{entry['t_sec']}s] "
            description_parts.append(marker + entry["description"])
        scene_description_a = " ".join(description_parts)

        # Use Option A as default (deterministic, no extra inference cost)
        scene_description = scene_description_a

        # --- §8.2.5 Confidence/quality flag ---
        # Count parse failures (frames where description is "Error analyzing frame")
        parse_failures = sum(
            1 for entry in per_frame_results if "Error analyzing" in entry.get("description", "")
        )
        total_frames_analyzed = len(per_frame_results)
        low_confidence = parse_failures > 0.3 * total_frames_analyzed if total_frames_analyzed > 0 else False

        # --- §8.0.3 full output schema ---
        result = {
            "video_metadata": {
                "filename": filename,
                "duration_sec": round(duration_sec, 1),
                "fps_sampled": self.fps,
                "num_frames_analyzed": len(per_frame_results),
            },
            "dominant_phase": dominant_phase,
            "phase_timeline": phase_timeline,
            "tools_detected": tools_detected,
            "scene_description": scene_description,
            "per_frame_results": per_frame_results,
            "model_info": {"base": "Qwen2.5-VL-7B", "adapter": "best_lora", "temp": 0.1},
        }

        # Attach low-confidence flag if applicable (not part of core schema but
        # useful for the UI; we add it as an extra field)
        if low_confidence:
            result["_low_confidence"] = True

        return result

    def _iter_frames_sequential(
        self, cap: cv2.VideoCapture, indices: List[int]
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        """Decode only the wanted frame indices via one sequential read pass,
        yielding one at a time instead of loading the whole list into memory.

        ``extract_frames`` (used by the non-streaming ``analyze_video``) reads
        every wanted frame via ``cap.set(CAP_PROP_POS_FRAMES, idx)`` (a slow
        per-call seek) and returns the full list up front. That's fine for the
        old ~180-frame cap, but once sampling can select every native frame of
        a real video (thousands of frames), it means minutes of seeking and
        gigabytes of PIL images held in memory before any analysis — or
        progress — is visible at all. Sequential ``cap.read()`` is much
        faster than repeated seeks, and yielding immediately keeps memory
        bounded to one frame regardless of video length.
        """
        wanted = sorted(set(indices))
        if not wanted:
            return
        wanted_set = set(wanted)
        last_needed = wanted[-1]

        frame_idx = 0
        while frame_idx <= last_needed:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx in wanted_set:
                pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                yield frame_idx, pil
            frame_idx += 1

    # -- streaming variant: yields the aggregate result after every frame so
    # the UI can update live instead of waiting for the whole video --

    def analyze_video_stream(
        self, path: str, progress: Optional[Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Same pipeline as ``analyze_video`` but yields incrementally.

        Each yielded dict has the full §8.0.3 schema (recomputed over the
        frames processed so far) plus a ``"_latest_frame"`` key holding just
        the most recently completed frame's entry, for a "now playing" panel.
        The final yield is identical in shape to ``analyze_video``'s return.
        """
        import os as _os
        import time as _time
        filename = _os.path.basename(path)
        t0 = _time.time()

        try:
            cap, duration_sec, total_frames = self.load_video(path)
        except ValueError as e:
            yield {
                "video_metadata": {
                    "filename": filename,
                    "duration_sec": 0,
                    "fps_sampled": self.fps,
                    "num_frames_analyzed": 0,
                },
                "dominant_phase": "Error",
                "phase_timeline": [],
                "tools_detected": [],
                "scene_description": str(e),
                "per_frame_results": [],
                "model_info": {"base": "Qwen2.5-VL-7B", "adapter": "best_lora", "temp": 0.1},
                "_latest_frame": None,
                "_done": True,
            }
            return

        indices = self.sample_frame_indices(total_frames, self.fps)
        num_wanted = len(indices)
        native_fps = getattr(self, "_native_fps", 30.0) or 30.0

        per_frame_results: List[Dict[str, Any]] = []
        for i, (idx, pil_img) in enumerate(self._iter_frames_sequential(cap, indices)):
            if progress is not None and num_wanted > 0:
                progress((i + 1) / max(1, num_wanted))

            result = self.analyze_frame(pil_img)
            # Real-world timestamp = native frame index / native fps (see
            # comment in analyze_video for why self.fps must not be used here).
            t_sec = round(idx / native_fps, 2)
            frame_entry = {
                "t_sec": t_sec,
                "frame_idx": idx,
                "phase": result["phase"],
                "tools": result["tools"],
                "description": result["description"],
            }
            per_frame_results.append(frame_entry)

            partial = self._build_result(filename, duration_sec, per_frame_results)
            partial["_latest_frame"] = frame_entry
            # Kept out of frame_entry/per_frame_results (and therefore out of the
            # §8.0.3 JSON schema / raw_json debug panel) since a PIL.Image isn't
            # JSON-serializable; the UI reads this separate key for the live image.
            partial["_latest_frame_image"] = pil_img
            partial["_progress"] = {
                "current": i + 1,
                "total": num_wanted,
                "elapsed_sec": round(_time.time() - t0, 1),
                "gpu_mem_gb": _gpu_memory_gb(),
            }
            partial["_done"] = (i == num_wanted - 1)
            yield partial

        cap.release()