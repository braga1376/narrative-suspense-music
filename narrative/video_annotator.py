"""
video_annotator.py
──────────────────
Burns Gemini extraction metadata and computed suspense values directly onto
video frames for visual validation.

Layout
──────
┌─────────────────────────────────────────────┐
│ [EVENT PANEL - top left]  [SUSPENSE METER]  │
│  event id / time range                right │
│  description (wrapped)                side  │
│  threads + importance bars                  │
│  characters present                         │
│  visual activity                            │
│  ⚡ TURNING POINT (if applicable)           │
│                                             │
│ ─────── video content ────────────────────  │
│                                             │
│ [TIMELINE BAR - bottom]                     │
│  event blocks coloured by visual_activity   │
│  red tick marks at turning points           │
└─────────────────────────────────────────────┘

Usage
──────
    from video_annotator import annotate_video

    annotate_video(
        video_path="wing_it.mp4",
        narrative_data=your_dict,   # the dict Gemini returns
        output_path="wing_it_annotated.mp4",
    )
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Optional moviepy for audio passthrough ─────────────────────────────────
try:
    from moviepy.editor import VideoFileClip, VideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


# ── Try to import project suspense module (graceful fallback) ───────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from suspense import compute_suspense
    SUSPENSE_AVAILABLE = True
except ImportError:
    SUSPENSE_AVAILABLE = False
    print("[video_annotator] suspense.py not found — suspense meter disabled.")


# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

# BGR tuples for OpenCV
C_WHITE        = (255, 255, 255)
C_BLACK        = (0,   0,   0)
C_RED          = (0,   0,   220)
C_ORANGE       = (0,   140, 255)
C_YELLOW       = (0,   220, 220)
C_GREEN        = (80,  200, 80)
C_BLUE         = (220, 140, 30)
C_CYAN         = (220, 220, 0)
C_GREY_DARK    = (40,  40,  40)
C_GREY_MED     = (90,  90,  90)
C_PANEL_BG     = (20,  20,  20)          # near-black for panels

# Visual activity → colour mapping (BGR)
ACTIVITY_COLOURS = {
    "static":   (100, 100, 100),
    "low":      (180, 200, 100),
    "moderate": (60,  180, 255),
    "high":     (30,  120, 255),
    "frantic":  (0,   0,   220),
}

# Thread importance sign → bar colour
IMPORTANCE_POS = (200, 120, 40)   # warm blue — positive importance
IMPORTANCE_NEG = (60,  60,  200)  # red       — negative / antagonist

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _alpha_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: tuple,
    alpha: float = 0.55,
) -> None:
    """Draw a semi-transparent filled rectangle in-place."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _text_lines(
    frame: np.ndarray,
    lines: list[tuple],          # (text, scale, colour, bold?)
    x: int, y: int,
    line_height: int = 18,
) -> int:
    """
    Draw multiple text lines starting at (x, y).
    Returns the y position after the last line.
    """
    for text, scale, colour, bold in lines:
        thickness = 2 if bold else 1
        cv2.putText(frame, text, (x, y), FONT, scale, C_BLACK, thickness + 1,
                    cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness,
                    cv2.LINE_AA)
        y += line_height
    return y


def _wrap(text: str, width: int = 42) -> list[str]:
    """Word-wrap text to `width` characters."""
    return textwrap.wrap(text, width=width) or [""]


def _importance_bar(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    value: float,                 # –10 … +10
    label: str,
) -> None:
    """Draw a small horizontal importance bar with label."""
    # Background track
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GREY_DARK, -1)
    # Fill
    norm = (value + 10) / 20.0             # map –10…+10 → 0…1
    fill_w = max(1, int(norm * w))
    colour = IMPORTANCE_POS if value >= 0 else IMPORTANCE_NEG
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), colour, -1)
    # Centre zero marker
    mid_x = x + w // 2
    cv2.line(frame, (mid_x, y), (mid_x, y + h), C_WHITE, 1)
    # Label
    cv2.putText(frame, label, (x + w + 4, y + h - 1),
                FONT, 0.32, C_WHITE, 1, cv2.LINE_AA)


def _h_bar(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    norm: float,                  # 0.0 … 1.0
    colour: tuple,
) -> None:
    """Draw a filled horizontal bar normalised to [0, 1]."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GREY_DARK, -1)
    fill_w = max(1, int(norm * w))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), colour, -1)


# ═══════════════════════════════════════════════════════════════════════════
#  TIMELINE BAR
# ═══════════════════════════════════════════════════════════════════════════

def _draw_timeline(
    frame: np.ndarray,
    events: list[dict],
    current_time: float,
    total_duration: float,
    x: int, y: int, w: int, h: int,
    turning_point_times: list[float],
) -> None:
    """Draw a colour-coded timeline scrubber at the bottom."""
    _alpha_rect(frame, x, y, x + w, y + h, C_PANEL_BG, alpha=0.70)

    scale = w / max(total_duration, 1.0)

    for ev in events:
        x1 = x + int(ev["start_time"] * scale)
        x2 = x + int(ev["end_time"]   * scale)
        activity = ev.get("visual_activity", "low")
        # Normalise enum value string
        if hasattr(activity, "value"):
            activity = activity.value
        col = ACTIVITY_COLOURS.get(str(activity), C_GREY_MED)
        cv2.rectangle(frame, (x1, y + 2), (x2 - 1, y + h - 2), col, -1)
        # Event boundary tick
        cv2.line(frame, (x1, y), (x1, y + h), C_GREY_DARK, 1)

    # Turning point markers
    for tp_time in turning_point_times:
        tx = x + int(tp_time * scale)
        cv2.line(frame, (tx, y), (tx, y + h), C_RED, 2)
        cv2.putText(frame, "|", (tx - 4, y + h - 2),
                    FONT, 0.3, C_RED, 1, cv2.LINE_AA)

    # Playhead
    px = x + int(current_time * scale)
    cv2.line(frame, (px, y - 2), (px, y + h + 2), C_WHITE, 2)

    # Time label
    mins = int(current_time) // 60
    secs = int(current_time) % 60
    cv2.putText(frame, f"{mins:02d}:{secs:02d}",
                (x + 4, y + h - 3),
                FONT, 0.35, C_WHITE, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  EVENT PANEL  (includes suspense section at the bottom)
# ═══════════════════════════════════════════════════════════════════════════

# Spacing constants — tweak here if you need a different density
_PAD        = 8    # inner horizontal padding
_LH_HEADER  = 20  # line height: header row
_LH_BADGE   = 20  # line height: turning-point badge
_LH_DESC    = 17  # line height: description text lines
_LH_TH_LBL  = 10  # line height: thread label
_BAR_H      = 8   # height of importance / suspense bars
_LH_TH_BAR  = 15  # line height: thread bar row (bar + gap below)
_LH_SMALL   = 20  # line height: characters / activity rows
_SEC_GAP    = 15  # vertical gap between logical sections
_DIVIDER_H  = 9   # space consumed by a divider line


def _divider(frame: np.ndarray, x: int, cy: int, w: int) -> int:
    """Draw a subtle horizontal rule; returns cy after the rule."""
    mid = cy + _DIVIDER_H // 2
    cv2.line(frame, (x + _PAD, mid), (x + w - _PAD, mid), C_GREY_MED, 1)
    return cy + _DIVIDER_H


def _estimate_panel_h(
    event: dict,
    suspense_result: Optional[dict],
) -> int:
    """Deterministically estimate total panel height before drawing."""
    desc_lines  = _wrap(event["description"], width=44)
    n_threads   = len(event.get("thread_ids", []))
    has_tp      = event.get("is_turning_point", False)
    has_chars   = bool(event.get("characters_present", []))
    has_suspense = suspense_result is not None
    n_ts_threads = len(suspense_result["thread_suspense"]) if has_suspense else 0

    h  = _PAD                           # top padding
    h += _LH_HEADER                     # event header
    h += _SEC_GAP
    if has_tp:
        h += _LH_BADGE + _SEC_GAP
    h += len(desc_lines) * _LH_DESC + _SEC_GAP
    h += n_threads * (_LH_TH_LBL + _LH_TH_BAR) + _SEC_GAP
    if has_chars:
        h += _LH_SMALL
    h += _LH_SMALL                      # activity
    if has_suspense:
        h += _SEC_GAP + _DIVIDER_H
        h += _LH_HEADER + 25               # "SUSPENSE" label row
        h += _LH_TH_BAR + _SEC_GAP     # global bar
        h += n_ts_threads * (_LH_TH_LBL + _LH_TH_BAR)
    h += _PAD                           # bottom padding
    return h


def _draw_event_panel(
    frame: np.ndarray,
    event: dict,
    suspense_result: Optional[dict],
    max_suspense: float,
    characters: dict,
    threads: dict,
    x: int, y: int, w: int,
) -> None:
    """
    Draw the unified annotation panel (top-left).
    Sections, top to bottom:
      1. Event header (id + time range)
      2. ⚡ Turning-point badge  [if applicable]
      3. Description (word-wrapped)
      4. Active threads + importance bars
      5. Characters present
      6. Visual activity level
      ── divider ──
      7. Suspense section (global bar + per-thread bars)  [if available]
    """
    panel_h = _estimate_panel_h(event, suspense_result)
    _alpha_rect(frame, x, y, x + w, y + panel_h, C_PANEL_BG, alpha=0.72)

    cx = x + _PAD        # left content edge
    bar_w = w - _PAD * 4  # usable bar width
    cy = y + _PAD + 12   # y cursor (OpenCV text baseline)

    # ── 1. Event header ────────────────────────────────────────────────
    is_tp = event.get("is_turning_point", False)
    header_col = C_ORANGE if is_tp else C_CYAN
    header_text = (
        f"Event {event['event_id']}   "
        f"{event['start_time']}s : {event['end_time']}s"
    )
    cv2.putText(frame, header_text, (cx, cy), FONT, 0.42, C_BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, header_text, (cx, cy), FONT, 0.42, header_col, 1, cv2.LINE_AA)
    cy += _LH_HEADER + _SEC_GAP

    # ── 2. Turning-point badge ─────────────────────────────────────────
    if is_tp:
        badge = " TURNING POINT "
        (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.4, 1)
        cv2.rectangle(frame, (cx, cy - bh - 2), (cx + bw, cy + 3), C_RED, -1)
        cv2.putText(frame, badge, (cx, cy), FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)
        cy += _LH_BADGE + _SEC_GAP

    # ── 3. Description ─────────────────────────────────────────────────
    for line in _wrap(event["description"], width=44):
        cv2.putText(frame, line, (cx, cy), FONT, 0.36, C_BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, line, (cx, cy), FONT, 0.36, C_WHITE, 1, cv2.LINE_AA)
        cy += _LH_DESC
    cy += _SEC_GAP

    # ── 4. Active threads + importance bars ────────────────────────────
    for tid in event.get("thread_ids", []):
        t = threads.get(tid)
        if not t:
            continue
        imp = t["importance"]
        label = f"T{tid} ({imp:+.1f})  {t['description'][:36]}"
        cv2.putText(frame, label[:52], (cx, cy),
                    FONT, 0.30, C_WHITE, 1, cv2.LINE_AA)
        cy += _LH_TH_LBL
        _importance_bar(frame, cx, cy, bar_w, _BAR_H, imp, "")
        cy += _LH_TH_BAR
    cy += _SEC_GAP

    # ── 5. Characters present ──────────────────────────────────────────
    char_names = [
        characters.get(cid, {}).get("name", f"#{cid}")
        for cid in event.get("characters_present", [])
    ]
    if char_names:
        chars_text = "Characters: " + ", ".join(char_names)
        cv2.putText(frame, chars_text[:56], (cx, cy),
                    FONT, 0.32, C_WHITE, 1, cv2.LINE_AA)
        cy += _LH_SMALL

    # ── 6. Visual activity ─────────────────────────────────────────────
    activity = event.get("visual_activity", "")
    if hasattr(activity, "value"):
        activity = activity.value
    activity_str = str(activity).upper()
    act_col = ACTIVITY_COLOURS.get(str(activity).lower(), C_WHITE)
    cv2.putText(frame, f"Activity: {activity_str}", (cx, cy),
                FONT, 0.35, act_col, 1, cv2.LINE_AA)
    cy += _LH_SMALL

    # ── 7. Suspense section ────────────────────────────────────────────
    if suspense_result is None:
        return

    cy += _SEC_GAP//2
    cy = _divider(frame, x, cy, w)
    cy += _SEC_GAP//2

    # "SUSPENSE" label + numeric value on the same row
    gs = suspense_result["global_suspense"]
    norm_gs = min(1.0, abs(gs) / max(max_suspense, 0.01))
    r_c = int(255 * norm_gs)
    g_c = int(255 * (1 - norm_gs))
    gs_col = (0, g_c, r_c)              # green → red

    cv2.putText(frame, "SUSPENSE", (cx, cy),
                FONT, 0.38, C_CYAN, 1, cv2.LINE_AA)
    val_text = f"{gs:+.2f}"
    (vw, _), _ = cv2.getTextSize(val_text, FONT, 0.38, 1)
    cv2.putText(frame, val_text, (x + w - _PAD - vw, cy),
                FONT, 0.38, gs_col, 1, cv2.LINE_AA)
    cy += _LH_HEADER

    # Global suspense bar
    _h_bar(frame, cx, cy, bar_w, _BAR_H + 2, norm_gs, gs_col)
    cy += _LH_TH_BAR + _SEC_GAP + 20

    # Per-thread suspense bars
    ts = suspense_result.get("thread_suspense", {})
    for tid, ts_data in sorted(ts.items()):
        imp  = threads.get(tid, {}).get("importance", 0)
        val  = ts_data.get("suspense", 0.0)
        norm_t = min(1.0, abs(val) / max(max_suspense, 0.01))
        col  = IMPORTANCE_POS if imp >= 0 else IMPORTANCE_NEG
        label = f"T{tid} ({val:+.2f})"
        cv2.putText(frame, label, (cx, cy),
                    FONT, 0.30, C_GREY_MED, 1, cv2.LINE_AA)
        cy += _LH_TH_LBL
        _h_bar(frame, cx, cy, bar_w, _BAR_H, norm_t, col)
        cy += _LH_TH_BAR + 5


# ═══════════════════════════════════════════════════════════════════════════
#  CORE ANNOTATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def _find_event(events: list[dict], t: float) -> dict:
    """Return the active event dict for time t (seconds)."""
    for ev in events:
        if ev["start_time"] <= t < ev["end_time"]:
            return ev
    # Beyond last event — return last one
    return events[-1]


def _build_lookup(narrative_data: dict) -> tuple:
    """Pre-build fast lookup structures from narrative_data."""
    characters = {c["character_id"]: c for c in narrative_data["characters"]}
    threads    = {t["thread_id"]:    t for t in narrative_data["threads"]}
    events     = narrative_data["events"]

    # Normalise VisualActivity enums to plain strings
    for ev in events:
        va = ev.get("visual_activity", "low")
        if hasattr(va, "value"):
            ev["visual_activity"] = va.value

    # Compute suspense once
    suspense_by_event_id: dict = {}
    max_suspense = 1.0
    if SUSPENSE_AVAILABLE:
        try:
            results = compute_suspense(narrative_data)
            for r in results:
                suspense_by_event_id[r["event_id"]] = r
            vals = [abs(r["global_suspense"]) for r in results]
            max_suspense = max(vals) if vals else 1.0
        except Exception as e:
            print(f"[video_annotator] Suspense computation failed: {e}")

    # Turning point mid-times for timeline
    tp_times = [
        (ev["start_time"] + ev["end_time"]) / 2
        for ev in events if ev.get("is_turning_point", False)
    ]

    return characters, threads, events, suspense_by_event_id, max_suspense, tp_times


def annotate_frame(
    frame: np.ndarray,
    current_time: float,
    total_duration: float,
    events: list[dict],
    characters: dict,
    threads: dict,
    suspense_by_event_id: dict,
    max_suspense: float,
    tp_times: list[float],
    panel_width: int  = 320,
    timeline_h: int   = 18,
) -> np.ndarray:
    """
    Annotate a single frame (numpy BGR array).
    Returns the annotated frame (modifies in-place).
    """
    h, w = frame.shape[:2]

    event           = _find_event(events, current_time)
    suspense_result = suspense_by_event_id.get(event["event_id"])

    # ── Unified event + suspense panel (top-left) ─────────────────────
    _draw_event_panel(
        frame, event, suspense_result, max_suspense,
        characters, threads,
        x=4, y=4, w=panel_width,
    )

    # ── Timeline bar (bottom) ──────────────────────────────────────────
    _draw_timeline(
        frame, events, current_time, total_duration,
        x=0, y=h - timeline_h, w=w, h=timeline_h,
        turning_point_times=tp_times,
    )

    return frame


def annotate_video(
    video_path: str,
    narrative_data: dict,
    output_path: str,
    panel_width: int = 320,
    timeline_h:  int = 18,
    codec:       str = "mp4v",
) -> None:
    """
    Burn annotation overlays onto a video and write to output_path.

    Parameters
    ──────────
    video_path      : path to the source video
    narrative_data  : the dict returned by Gemini extraction
    output_path     : path for the annotated video (mp4)
    panel_width     : width of the left info panel in pixels
    timeline_h      : height of the bottom timeline bar in pixels
    codec           : FourCC codec string (default: mp4v)

    Notes
    ─────
    - If moviepy is installed, audio is preserved in the output.
    - If not, a silent annotated video is written via OpenCV only.
    """
    print(f"[video_annotator] Loading: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps            = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps if fps > 0 else 1.0

    print(f"[video_annotator] {frame_w}×{frame_h} @ {fps:.2f} fps  "
          f"({total_frames} frames, {total_duration:.1f}s)")

    # Pre-build lookup tables and compute suspense once
    (characters, threads, events,
     suspense_by_event_id, max_suspense,
     tp_times) = _build_lookup(narrative_data)

    # Temporary output path (silent) — we'll mux audio afterwards
    tmp_path = str(Path(output_path).with_suffix(".tmp.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        annotate_frame(
            frame, t, total_duration,
            events, characters, threads,
            suspense_by_event_id, max_suspense, tp_times,
            panel_width, timeline_h,
        )
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 60 == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total_frames}", end="\r")

    cap.release()
    writer.release()
    print(f"\n[video_annotator] Frames written to: {tmp_path}")

    # ── Mux audio if moviepy available ────────────────────────────────
    if MOVIEPY_AVAILABLE:
        print("[video_annotator] Muxing audio with moviepy …")
        try:
            original = VideoFileClip(str(video_path))
            annotated = VideoFileClip(tmp_path)

            if original.audio is not None:
                annotated = annotated.set_audio(original.audio)

            annotated.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            original.close()
            annotated.close()
            Path(tmp_path).unlink(missing_ok=True)
            print(f"[video_annotator] ✓ Saved with audio: {output_path}")
        except Exception as e:
            print(f"[video_annotator] Audio mux failed ({e}), "
                  f"keeping silent file: {tmp_path}")
            Path(tmp_path).rename(output_path)
    else:
        Path(tmp_path).rename(output_path)
        print(f"[video_annotator] ✓ Saved (silent, install moviepy for audio): "
              f"{output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Annotate a video with Gemini extraction results."
    )
    parser.add_argument("video",   help="Input video path")
    parser.add_argument("json",    help="Narrative data JSON file")
    parser.add_argument("output",  help="Output video path")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    annotate_video(args.video, data, args.output)