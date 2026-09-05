"""Gradio Video Demo UI — Phase 8 video-first demo (TODO.md §8.3).

Launches the surgical video analysis UI. All heavy logic lives in
``web.video_processor``; this module only handles Gradio wiring and launch.

Usage:
    python -m web.gradio_app --mock          # local test, no GPU needed
    python -m web.gradio_app --share         # RunPod public URL
    python -m web.gradio_app --enable-queue  # concurrent uploads, OOM-safe
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from web.video_processor import (
    VideoAnalyzer,
    TIMELINE_PHASES,
    PHASE_COLORS,
    PHASE_NORMALISATION,
    PHASE_TOKENS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

DEMO_CSS = """
.demo-header { font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem; }
.control-row { margin-bottom: 0.5rem; }
/* !important + targeting all descendants: Gradio's theme (esp. dark mode)
   applies its own color to headings/bold/em inside rendered Markdown at a
   more specific selector than a plain class rule, which was silently
   overriding this and making the text unreadable regardless of formatting. */
.teal-white, .teal-white * { color: #2c3e50 !important; background-color: #ecfdf9 !important; }
.tool-table { font-size: 0.85rem; }
"""

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_MOCK_MODE: bool = False
_model_holder: Dict[str, Any] = {}


def _set_mock_mode(value: bool) -> None:
    global _MOCK_MODE
    _MOCK_MODE = value


def _model_holder_get_vlm() -> Optional[Any]:
    return _model_holder.get("vlm")


# ---------------------------------------------------------------------------
# Mock support (§8.3.6)
# ---------------------------------------------------------------------------


def _mock_video_analyzer() -> VideoAnalyzer:
    """Return a VideoAnalyzer backed by MockSurgicalVLM for UI testing."""
    from surgical_vlm.models.surgical_vlm import MockSurgicalVLM

    class _DummyTok:
        pass

    class _DummyProc:
        pass

    return VideoAnalyzer(
        model=MockSurgicalVLM(),
        tokenizer=_DummyTok(),
        processor=_DummyProc(),
        fps=1.0,
        max_frames=180,
    )


def _wrap_live_panel(inner_html: str) -> str:
    """Wrap content in a div with inline styles (not a CSS class) so text
    color/background survive regardless of Gradio's theme/shadow-DOM CSS."""
    return (
        '<div style="color:#ffffff; background-color:#111111; padding:0.75rem 1rem; '
        'border-radius:8px; font-family:inherit;">' + inner_html + "</div>"
    )


# ---------------------------------------------------------------------------
# Phase timeline plot helper
# ---------------------------------------------------------------------------


def _build_phase_plot(phase_timeline: List[Dict[str, Any]]):
    """Return a matplotlib Figure showing phase strips over time, or None."""
    if not phase_timeline:
        return None

    try:
        phase_names = [p["phase"] for p in phase_timeline]
        t_secs = [p["t_sec"] for p in phase_timeline]
        total_dur = max(t_secs) if t_secs else 1.0
        # Uniform bar width so strips don't overlap or have gaps
        bar_w = total_dur / max(len(t_secs), 1)

        fig, ax = plt.subplots(figsize=(10, 0.8))
        for t, ph in zip(t_secs, phase_names):
            color_str = PHASE_COLORS.get(ph, "(120, 120, 120)")
            rgb = tuple(int(v) / 255 for v in color_str.strip("()").split(","))
            ax.barh(0, width=bar_w, left=t, color=rgb, height=0.6)

        ax.set_xlim(0, total_dur * 1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_title("Phase Timeline")

        # Legend
        seen = {}
        for ph in phase_names:
            if ph not in seen:
                color_str = PHASE_COLORS.get(ph, "(120, 120, 120)")
                rgb = tuple(int(v) / 255 for v in color_str.strip("()").split(","))
                seen[ph] = plt.Rectangle((0, 0), 1, 1, fc=rgb)
        if seen:
            ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7)

        fig.tight_layout()
        return fig  # do NOT plt.close() here — Gradio needs the live figure
    except Exception as exc:
        logger.warning("Phase plot failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Gradio UI (§8.3)
# ---------------------------------------------------------------------------


def _build_video_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks layout for the video demo."""

    with gr.Blocks(title="Surgical VLM Video Demo", css=DEMO_CSS) as demo:
        gr.Markdown("# Surgical VLM — Video Analytics Demo")
        gr.Markdown(
            "*Real-time surgical video analysis using Qwen2.5-VL-7B with LoRA fine-tuning*"
        )
        gr.Markdown(
            "**Model:** SurgicalVLM based on Qwen2.5-VL-7B, LoRA fine-tuned on "
            "Cholec80 + CholecT50 + HeiChole + Surg-396K"
        )

        with gr.Row():
            # --- Left: input + controls + summary panels ---
            with gr.Column(scale=1, min_width=380):
                video_input = gr.Video(
                    label="Upload Surgical Video (.mp4 / .avi / .mov / .mkv, ≤ 200 MB)",
                    sources=["upload"],
                    height=220,
                )

                with gr.Row():
                    gen_narrative_cb = gr.Checkbox(
                        label="Generate narrative summary", value=False,
                    )
                    show_breakdown_cb = gr.Checkbox(
                        label="Show per-frame breakdown", value=True,
                    )

                with gr.Row():
                    analyze_btn = gr.Button("Analyze Video", variant="primary")
                    clear_btn = gr.Button("Clear")

                error_msg = gr.Markdown(visible=False)

                with gr.Accordion("Phase Timeline", open=True):
                    phase_plot = gr.Plot(label="Phase Timeline (smoothed)")

                with gr.Accordion("Tools Detected", open=True):
                    tools_df = gr.Dataframe(
                        headers=["Tool", "First Seen (s)", "Last Seen (s)", "Frame Count"],
                        datatype=("str", "number", "number", "number"),
                        label="Tools Summary",
                        interactive=False,
                        row_count=4,
                        column_count=4,
                    )

            # --- Right: live view + details ---
            with gr.Column(scale=1, min_width=380):
                # gr.HTML + inline styles (not gr.Markdown + a CSS class) —
                # external stylesheets, even with !important, were not
                # reaching this text (Gradio's Markdown likely renders inside
                # a shadow-DOM-style boundary the injected CSS can't cross).
                # Inline style="" attributes are on the element itself and
                # always win regardless of that.
                now_playing_md = gr.HTML(
                    _wrap_live_panel("<i>Upload a video and click Analyze to start live detection.</i>")
                )

                current_frame_img = gr.Image(
                    label="Current Frame (analyzed live, paced by model speed)",
                    height=200,
                    interactive=False,
                )
                snapshot_download = gr.DownloadButton(
                    "📸 Download Current Frame", visible=False
                )

                with gr.Accordion("Scene Description", open=True):
                    scene_txt = gr.Textbox(
                        label="Narrative Summary", lines=3, interactive=False
                    )

                with gr.Accordion("Per-Frame Breakdown", open=False):
                    per_frame_df = gr.Dataframe(
                        headers=["t_sec (s)", "Phase", "Tools", "Description"],
                        datatype=("number", "str", "str", "str"),
                        label="Per-Frame Results",
                        interactive=False,
                        row_count=6,
                        column_count=4,
                    )

        # ----------------------------------------------------------------
        # Event handlers
        # ----------------------------------------------------------------

        def _now_playing_text(
            latest: Optional[Dict[str, Any]], progress_info: Optional[Dict[str, Any]]
        ) -> str:
            import html as _html
            if not latest:
                return _wrap_live_panel("<i>Analyzing…</i>")
            tools = _html.escape(
                ", ".join(latest["tools"]) if latest["tools"] else "none visible"
            )
            status_line = ""
            if progress_info:
                cur, tot = progress_info.get("current", 0), progress_info.get("total", 0)
                pct = round(100 * cur / tot) if tot else 0
                gpu = progress_info.get("gpu_mem_gb")
                gpu_str = f" | GPU: {gpu} GB" if gpu is not None else ""
                status_line = (
                    f"<i>Frame {cur}/{tot} ({pct}%) — Elapsed: "
                    f"{progress_info.get('elapsed_sec', 0)}s{gpu_str}</i><br><br>"
                )
            body = (
                f"{status_line}"
                f"<h3>🔴 Live — t={latest['t_sec']}s</h3>"
                f"<b>Phase:</b> {_html.escape(str(latest['phase']))}<br>"
                f"<b>Tools:</b> {tools}<br>"
                f"<b>Scene:</b> {_html.escape(latest['description'] or '—')}"
            )
            return _wrap_live_panel(body)

        def _rows_from_result(result: Dict[str, Any]) -> Tuple:
            phase_fig = _build_phase_plot(result.get("phase_timeline", []))

            tools_data = [
                [t["tool"], t["first_seen_sec"], t["last_seen_sec"], t["frame_count"]]
                for t in result.get("tools_detected", [])
            ] or [["No tools detected yet", None, None, None]]

            scene_desc = result.get("scene_description", "")

            per_frame_data = [
                [r["t_sec"], r["phase"], ", ".join(r["tools"]) if r["tools"] else "", r["description"]]
                for r in result.get("per_frame_results", [])
            ] or [["—", "—", "—", "—"]]

            low_conf = result.get("_low_confidence", False)
            warn = gr.update(
                visible=low_conf,
                value="⚠ Low confidence: >30% of frames had parse failures." if low_conf else "",
            )
            now_playing = _now_playing_text(result.get("_latest_frame"), result.get("_progress"))
            current_frame = result.get("_latest_frame_image")

            snapshot_path = None
            if current_frame is not None:
                try:
                    import tempfile
                    snapshot_path = str(Path(tempfile.gettempdir()) / "svlm_snapshot.png")
                    current_frame.save(snapshot_path)
                except Exception as exc:
                    logger.warning("Snapshot save failed: %s", exc)

            return (
                phase_fig, tools_data, scene_desc, per_frame_data, warn,
                now_playing, current_frame,
                gr.update(value=snapshot_path, visible=snapshot_path is not None),
            )

        # Fixed defaults now that the Sampling FPS / Max Frames sliders were
        # removed from the UI (they controlled the VLM's frame-sampling rate
        # only — unrelated to any object-detection pipeline).
        # _DEFAULT_FPS=32 requests 32 analyzed frames per second of video;
        # sample_frame_indices caps this at the source's own native fps (you
        # cannot sample more distinct frames/sec than the file contains), so
        # for typical ≤30fps surgical footage this analyzes every native
        # frame -- the densest possible sampling for that source.
        # _DEFAULT_MAX_FRAMES set far above any realistic video's frame count
        # as a safety cap only; the fps target above is what actually drives
        # sampling now. This means total processing time scales directly with
        # the video's frame count (~2-4s/frame model speed) -- only practical
        # for short clips.
        _DEFAULT_FPS = 32.0
        _DEFAULT_MAX_FRAMES = 100_000

        def run_analysis(
            video_path: str,
            gen_narrative: bool,
            show_breakdown: bool,
            progress=gr.Progress(),
        ):
            """Stream VideoAnalyzer results into the UI live, frame by frame."""
            if not video_path:
                yield (
                    None,
                    [["No video uploaded", None, None, None]],
                    "", [["—", "—", "—", "—"]],
                    gr.update(visible=True, value="Please upload a video first."),
                    _wrap_live_panel("<i>Upload a video and click Analyze to start live detection.</i>"),
                    None, gr.update(value=None, visible=False),
                )
                return

            if _MOCK_MODE:
                analyzer = _mock_video_analyzer()
            else:
                vlm = _model_holder_get_vlm()
                if vlm is None:
                    yield (
                        None,
                        [["No model loaded", None, None, None]],
                        "", [["—", "—", "—", "—"]],
                        gr.update(
                            visible=True,
                            value="No VLM model loaded. Run with `--mock` or load a checkpoint.",
                        ),
                        _wrap_live_panel("<i>No model loaded.</i>"),
                        None, gr.update(value=None, visible=False),
                    )
                    return
                analyzer = VideoAnalyzer(
                    model=vlm,
                    tokenizer=getattr(vlm, "tokenizer", None) or object(),
                    processor=getattr(vlm, "processor", None) or object(),
                    fps=_DEFAULT_FPS,
                    max_frames=_DEFAULT_MAX_FRAMES,
                )

            try:
                for partial in analyzer.analyze_video_stream(video_path, progress=progress):
                    yield _rows_from_result(partial)
            except Exception as exc:
                logger.exception("analyze_video_stream failed")
                yield (
                    None,
                    [["Error", None, None, None]],
                    "", [["—", "—", "—", "—"]],
                    gr.update(visible=True, value=f"Analysis failed: {exc}"),
                    _wrap_live_panel(f"<i>Analysis failed: {exc}</i>"),
                    None, gr.update(value=None, visible=False),
                )

        _OUTPUTS = [
            phase_plot, tools_df, scene_txt, per_frame_df, error_msg,
            now_playing_md, current_frame_img, snapshot_download,
        ]

        analyze_btn.click(
            fn=run_analysis,
            inputs=[video_input, gen_narrative_cb, show_breakdown_cb],
            outputs=_OUTPUTS,
            # "full" (Gradio's default) overlays a progress bar across the whole
            # output region for the duration of the click event, which visually
            # buries the live per-frame yields underneath it. Hide that overlay
            # so the streamed outputs (current frame, live text, tables) are
            # what's actually visible while analysis runs.
            show_progress="hidden",
        )

        def _clear():
            plt.close("all")
            return (
                None,
                [["No tools detected", None, None, None]],
                "", [["—", "—", "—", "—"]],
                gr.update(visible=False, value=""),
                _wrap_live_panel("<i>Upload a video and click Analyze to start live detection.</i>"),
                None, gr.update(value=None, visible=False),
            )

        clear_btn.click(fn=_clear, inputs=[], outputs=_OUTPUTS, queue=False)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Surgical VLM Video Demo UI")
    parser.add_argument("--mock", action="store_true",
                        help="Run without a loaded VLM (mock responses)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public sharing link")
    parser.add_argument("--enable-queue", action="store_true",
                        help="Enable Gradio queue (max_size=5) for concurrent uploads")
    parser.add_argument("--checkpoint", default="",
                        help="Path to LoRA checkpoint to load before launching")
    parser.add_argument("--model-config",
                        default="configs/training/runpod_7b_config.yaml",
                        help="Model config YAML (used when --checkpoint is set)")
    args = parser.parse_args()

    if args.mock:
        _set_mock_mode(True)
        logger.info("Running in --mock mode (no VLM required)")
    elif args.checkpoint:
        logger.info("Loading checkpoint %s …", args.checkpoint)
        from scripts.evaluate_multi_task import load_model
        _model_holder["vlm"] = load_model(args.model_config, args.checkpoint)
        logger.info("Checkpoint loaded.")

    demo = _build_video_ui()

    # Queuing is required for the live-streaming analyze handler (a generator)
    # to push incremental updates to the browser, so it's always on; --enable-queue
    # just caps concurrent uploads to bound GPU memory use.
    demo.queue(max_size=5 if args.enable_queue else None)

    logger.info("Launching demo%s …", " (mock)" if args.mock else "")
    demo.launch(share=args.share, debug=False)
