# API Reference — SurgicalVLM_Robotics Video Demo (Phase 8)

All endpoints are served via the Gradio app at `http://localhost:7860` (or the public share URL). The app runs on `gr.Blocks` with queue-enabled processing.

---

## POST /analyze_video

Analyze a laparoscopic video and return phase/tools/narrative metadata.

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_path` | `str` | Yes | Path to the input MP4/AVI/MKV video file. Must be ≤ 200 MB and one of `["mp4", "avi", "mov", "mkv"]`. |
| `fps` | `float` | No | Frame extraction rate (frames per second). Default: `1.0`. |
| `max_frames` | `int` | No | Maximum number of frames to extract. Default: `180`. |
| `narrative` | `bool` | No | Whether to include the narrative description. Default: `true`. |
| `breakdown` | `bool` | No | Whether to include per-frame phase/tool breakdown. Default: `true`. |

### Response (JSON)

The response follows the **§8.0.3 JSON schema**:

```json
{
  "phase": "Preparation",
  "dominant_phase": "Preparation",
  "phase_confidence": 0.92,
  "phases": ["Preparation", "Dissection"],
  "phase_timestamps": {"Preparation": [0.0, 12.5], "Dissection": [12.5, 45.0]},
  "tools_detected": [
    {"name": "Grasper", "first_seen_sec": 0.0, "last_seen_sec": 52.3, "frame_count": 47},
    {"name": "Bipolar", "first_seen_sec": 3.2, "last_seen_sec": 48.1, "frame_count": 41}
  ],
  "tools_union_sorted": ["Grasper", "Bipolar"],
  "narrative": "The surgeon uses a grasper to hold the tissue while bipolar coagulates the vessels.",
  "low_confidence": false,
  "scene_description_option_a": "Preparation phase: Grasper used to retract tissue. ClippingCutting phase: Bipolar coagulates bleeding vessels. Dissection phase: Calot triangle dissection proceeding.",
  "scene_description_option_b": "",
  "scene_description_option_c": ""
}
```

### Response Fields Detail

| Field | Type | Description |
|-------|------|-------------|
| `phase` | `str` | The dominant/diagnosed phase for the **entire video** (one of: `Preparation`, `CalotTriangle`, `Dissection`, `ClippingCutting`, `GallbladderRetraction`). |
| `dominant_phase` | `str` | Alias for `phase`. |
| `phase_confidence` | `float` | Confidence score in [0, 1] from the majority‑vote / median‑filter aggregation. |
| `phases` | `List[str]` | All phases detected in temporal order (may contain 1–5 entries). |
| `phase_timestamps` | `Dict[str, List[float]]` | `{phase_name: [start_sec, end_sec]}` for each detected phase. |
| `tools_detected` | `List[Dict]` | List of tools found, each with `name`, `first_seen_sec`, `last_seen_sec`, `frame_count`. Sorted by `frame_count` descending. |
| `tools_union_sorted` | `List[str]` | Union of all tool names, sorted by total `frame_count` descending. |
| `narrative` | `str` | One‑sentence free‑form description of the action. If `narrative` flag was `false`, returns `""`. |
| `low_confidence` | `bool` | `true` if **>30%** of parsed frames failed JSON validation (indicates model output issues). |
| `scene_description_option_a` | `str` | Concatenated narrative with timestamp markers (default Option A). |
| `scene_description_option_b` | `str` | Empty by default; populated if a second prompting strategy is enabled. |
| `scene_description_option_c` | `str` | Empty by default; populated if a third prompting strategy is enabled. |

### Example Requests

#### cURL (video file)

```bash
curl -X POST "http://localhost:7860/api/analyze_video" \
  -F "video_path=@/path/to/surgery.mp4" \
  -F "fps=1" \
  -F "max_frames=180" \
  -F "narrative=true" \
  -F "breakdown=true"
```

#### Python

```python
import requests

files = {"video_path": open("/path/to/surgery.mp4", "rb")}
params = {"fps": 1, "max_frames": 180, "narrative": True, "breakdown": True}
resp = requests.post("http://localhost:7860/api/analyze_video", files=files, params=params)
data = resp.json()
print(data["phase"], data["narrative"])
```

---

## POST /health

Health check endpoint to verify the service is running and optionally whether the VLM model is loaded.

### Request

No parameters required.

### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "mock_mode": false
}
```

- If the app was launched with `--mock`, `mock_mode` will be `true` and `model_loaded` will be `false` (a MockSurgicalVLM is used instead).
- If the queue is backed up, the response still returns `healthy` but latency may be higher.

---

## UI Integration

The Gradio app (`web/gradio_app.py`) exposes a visual interface at `http://localhost:7860` with the following components:

- **Video input**: `gr.Video` (primary), accepts MP4/AVI/MKV up to 200 MB.
- **Legacy image tab**: `gr.Image` (secondary, for backward compatibility with image‑only mode).
- **Controls**: FPS slider, max frames checkbox, narrative/breakdown toggles.
- **Output layout**:
  - **Video Metadata Accordion**: dominant phase, phase confidence, phase timeline plot.
  - **Tools Dataframe**: table of tools detected (name, first_seen, last_seen, frame_count).
  - **Scene Description Textbox**: Option A / Option B / Option C narrative.
  - **Per‑Frame Breakdown Dataframe**: 8‑frame batched inference results.
  - **Raw JSON Accordion**: the full parsed JSON output.
  - **Thumbnails**: optional frame thumbnails (enabled via checkbox).
- **Clear button**: resets all outputs and re‑enables the upload control.
- **Examples**: three demo videos located in `web/examples/`:
  1. `cholec80_30s.mp4` — 30‑second Cholec80 clip (Preparation → Dissection).
  2. `hei_chole_clip.mp4` — HeiChole procedure clip.
  3. `desk_scene.mp4` — Non‑surgical desk scene (all Preparation).

---

## Development / Testing

### Local pytest suite

Run the existing test suite to verify the video processor logic:

```bash
pytest tests/ -v
```

All 6 tests in `tests/test_video_processor.py` cover JSON parsing, phase normalisation, and tool list extraction.

### Mock mode

Launch the app in mock mode (no VLM required) for UI testing:

```bash
python -m web.gradio_app --mock
```

This sets `_MOCK_MODE = True` and every frame receives a deterministic mock response, allowing the full UI to function without a loaded VLM.

---

## Version

API version `1.0.0` — corresponds to the video‑first live-streaming demo implementation.