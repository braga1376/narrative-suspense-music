"""
excerpt_selection.py

Algorithmic excerpt selection for evaluation surveys.

Selects video excerpts that maximise suspense variation (direction
changes) within a target duration window.  The selection criterion
is defined entirely from the suspense model output — the system's
own intermediate representation — not from the generated music,
eliminating manual selection bias.

Algorithm:
    1. Slide a window of [min_dur, max_dur] seconds across events
    2. For each valid window, compute:
       a) inflection_count — number of direction changes in the
          suspense curve (sign changes in first differences)
       b) suspense_range — max(suspense) - min(suspense) in window
    3. Filter out windows below a minimum range threshold
       (default: 20% of the film's full suspense range)
    4. Rank by inflection_count (descending), break ties by range
    5. Return the top-k non-overlapping excerpts

Usage:
    from suspense import compute_suspense
    from excerpt_selection import select_excerpts, plot_excerpt_selection

    suspense_results = compute_suspense(narrative_data)
    excerpts = select_excerpts(suspense_results, min_duration=75, max_duration=100)
    plot_excerpt_selection(suspense_results, excerpts, narrative_data)

Reference (for paper):
    "Excerpts were selected algorithmically to ensure representative
    suspense variation.  For each film, we computed the suspense curve
    and applied a sliding window of 75-100 seconds.  We selected the
    window maximising the number of suspense direction changes, subject
    to a minimum range constraint of 20% of the film's global suspense
    range.  This ensures excerpts contain multiple tension inflection
    points without manual selection bias."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import datetime

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExcerptCandidate:
    """One candidate excerpt window."""
    start_event_idx: int          # index into suspense_results
    end_event_idx: int            # inclusive
    start_time: float             # seconds
    end_time: float               # seconds
    duration: float               # seconds
    inflection_count: int         # number of suspense direction changes
    suspense_range: float         # max - min suspense in window
    suspense_values: List[float]  # suspense per event in window
    turning_point_count: int      # number of turning points in window
    boundary_extended: bool = False  # True if window was extended beyond
                                     # max_duration to complete a cycle

    @property
    def score(self) -> Tuple[int, float]:
        """Primary sort key: (inflection_count DESC, range DESC)."""
        return (self.inflection_count, self.suspense_range)

    def __repr__(self) -> str:
        ext = " [EXTENDED]" if self.boundary_extended else ""
        return (
            f"Excerpt({self.start_time:.1f}s–{self.end_time:.1f}s, "
            f"dur={self.duration:.1f}s, "
            f"inflections={self.inflection_count}, "
            f"range={self.suspense_range:.3f}, "
            f"turning_points={self.turning_point_count}, "
            f"events={self.start_event_idx}–{self.end_event_idx}{ext})"
        )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _count_inflections(
    values: List[float],
    tolerance_fraction: float = 0.05,
) -> int:
    """Count direction changes in a sequence.

    An inflection occurs when the sign of the first difference
    changes (rise → fall or fall → rise).  Small movements below
    a proportional tolerance are ignored — they don't count as
    either direction, so the previous direction carries forward.

    The tolerance is computed as a fraction of the window's value
    range, so noise-level oscillations in flat regions don't
    generate spurious inflections while real suspense swings are
    always detected.

    Args:
        values:             sequence of suspense values
        tolerance_fraction: minimum |diff| as a fraction of the
                            window's value range to count as a
                            real direction change (default 5%)

    Returns:
        Number of direction changes (inflection points)
    """
    if len(values) < 3:
        return 0

    value_range = max(values) - min(values)
    tolerance = tolerance_fraction * value_range

    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

    # Find the first non-flat difference to establish initial direction
    prev_sign = 0
    inflections = 0

    for d in diffs:
        if abs(d) < tolerance:
            continue  # below threshold — skip, keep previous direction

        current_sign = 1 if d > 0 else -1

        # if prev_sign != 0 and current_sign != prev_sign:
        #     inflections += 1
        if prev_sign == 1 and current_sign == -1:
            inflections += 1

        prev_sign = current_sign

    return inflections


def _windows_overlap(a: ExcerptCandidate, b: ExcerptCandidate) -> bool:
    """Check whether two excerpts share any events."""
    return not (a.end_event_idx < b.start_event_idx or
                b.end_event_idx < a.start_event_idx)


def _direction_at_boundary(values: List[float], tolerance_fraction: float = 0.05) -> Optional[int]:
    """Return the direction of movement at the end of a value sequence.

    Returns +1 (rising), -1 (falling), or None (flat / too few values).
    Used to detect whether a window boundary cuts off a suspense cycle
    that is still in progress.
    """
    if len(values) < 2:
        return None
    value_range = max(values) - min(values)
    tolerance = tolerance_fraction * value_range
    # Walk backwards to find the last significant movement
    for i in range(len(values) - 1, 0, -1):
        d = values[i] - values[i - 1]
        if abs(d) >= tolerance:
            return 1 if d > 0 else -1
    return None


def _try_boundary_extension(
    suspense_results: List[dict],
    i: int,
    j: int,
    max_extend_events: int,
    tolerance_fraction: float = 0.05,
) -> Optional[int]:
    """Check if extending the window by 1-2 events completes a direction change.

    If the window ends while suspense is still moving in one direction,
    and the next event(s) reverse that direction, return the new end
    index.  Otherwise return None.

    This ensures excerpts contain complete suspense cycles rather than
    cutting off at peaks or troughs.
    """
    n = len(suspense_results)
    if j + 1 >= n:
        return None  # already at end of film

    window_suspense = [
        suspense_results[k]["global_suspense"] for k in range(i, j + 1)
    ]
    boundary_dir = _direction_at_boundary(window_suspense, tolerance_fraction)
    if boundary_dir is None:
        return None  # flat at boundary, nothing to complete

    value_range = max(window_suspense) - min(window_suspense)
    tolerance = tolerance_fraction * value_range

    # Try extending by 1, then 2, ... up to max_extend_events
    for extra in range(1, max_extend_events + 1):
        new_j = j + extra
        if new_j >= n:
            break

        d = (suspense_results[new_j]["global_suspense"]
             - suspense_results[new_j - 1]["global_suspense"])

        if abs(d) < tolerance:
            continue  # flat — keep looking

        new_dir = 1 if d > 0 else -1
        if new_dir != boundary_dir:
            # Direction reversed — this extension completes the cycle
            return new_j

    return None


def select_excerpts(
    suspense_results: List[dict],
    min_duration: float = 75.0,
    max_duration: float = 100.0,
    min_range_fraction: float = 0.20,
    min_events: int = 4,
    top_k: int = 3,
    non_overlapping: bool = True,
    boundary_extend_events: int = 2,
) -> List[ExcerptCandidate]:
    """Select the best excerpt(s) from a film's suspense curve.

    Args:
        suspense_results: output of compute_suspense() — list of dicts
                          with keys: event_id, start_time, end_time,
                          global_suspense, is_turning_point
        min_duration:     minimum excerpt length in seconds
        max_duration:     maximum excerpt length in seconds
        min_range_fraction: minimum suspense range as a fraction of the
                            film's total range (filters out flat windows)
        min_events:       minimum number of events in a valid window
                          (need ≥4 to have meaningful inflections)
        top_k:            number of excerpts to return
        non_overlapping:  if True, returned excerpts share no events
        boundary_extend_events: max events to extend beyond max_duration
                                to complete a suspense cycle (0 = strict)

    Returns:
        List of ExcerptCandidate, sorted best-first
    """
    n = len(suspense_results)
    if n < min_events:
        raise ValueError(
            f"Need at least {min_events} events, got {n}"
        )

    # Film-level suspense range for threshold
    all_suspense = [r["global_suspense"] for r in suspense_results]
    film_range = max(all_suspense) - min(all_suspense)
    min_range = min_range_fraction * film_range

    if film_range < 1e-9:
        raise ValueError("Suspense curve is flat — no variation to select from")

    # --- Helper to build a candidate from index range ---
    def _make_candidate(start_idx: int, end_idx: int) -> Optional[ExcerptCandidate]:
        start_t = suspense_results[start_idx]["start_time"]
        end_t = suspense_results[end_idx]["end_time"]
        duration = end_t - start_t

        window_suspense = [
            suspense_results[k]["global_suspense"]
            for k in range(start_idx, end_idx + 1)
        ]
        s_range = max(window_suspense) - min(window_suspense)
        if s_range < min_range:
            return None

        inflections = _count_inflections(window_suspense)
        tp_count = sum(
            1 for k in range(start_idx, end_idx + 1)
            if suspense_results[k]["is_turning_point"]
        )

        return ExcerptCandidate(
            start_event_idx=start_idx,
            end_event_idx=end_idx,
            start_time=start_t,
            end_time=end_t,
            duration=duration,
            inflection_count=inflections,
            suspense_range=s_range,
            suspense_values=window_suspense,
            turning_point_count=tp_count,
        )

    # --- Enumerate all valid windows ---
    candidates: List[ExcerptCandidate] = []
    seen_ranges = set()  # avoid duplicate (i, j) from extension

    for i in range(n):
        for j in range(i + min_events - 1, n):
            start_t = suspense_results[i]["start_time"]
            end_t = suspense_results[j]["end_time"]
            duration = end_t - start_t

            if duration < min_duration:
                continue
            if duration > max_duration:
                # --- Boundary extension check ---
                # The window [i, j-1] was the last one within max_duration.
                # Check if extending j-1 by a few events completes a cycle.
                if boundary_extend_events > 0 and j > i + min_events - 1:
                    prev_j = j - 1
                    extended_j = _try_boundary_extension(
                        suspense_results, i, prev_j,
                        boundary_extend_events,
                    )
                    if extended_j is not None and (i, extended_j) not in seen_ranges:
                        cand = _make_candidate(i, extended_j)
                        if cand is not None:
                            # Only add if the extension actually gained
                            # an inflection vs the unextended version
                            base_cand = _make_candidate(i, prev_j)
                            if (base_cand is None or
                                    cand.inflection_count > base_cand.inflection_count):
                                cand.boundary_extended = True
                                seen_ranges.add((i, extended_j))
                                candidates.append(cand)
                break  # events are chronological, no shorter window ahead

            # Standard candidate within duration bounds
            key = (i, j)
            if key not in seen_ranges:
                cand = _make_candidate(i, j)
                if cand is not None:
                    seen_ranges.add(key)
                    candidates.append(cand)

    if not candidates:
        raise ValueError(
            f"No valid windows found with duration [{min_duration}, "
            f"{max_duration}]s and range ≥ {min_range:.3f}. "
            f"Try relaxing constraints (shorter duration or lower "
            f"min_range_fraction)."
        )

    # --- Rank candidates ---
    # Primary: inflection count (descending)
    # Secondary: suspense range (descending)
    candidates.sort(key=lambda c: c.score, reverse=True)

    # --- Select top-k, optionally non-overlapping ---
    selected: List[ExcerptCandidate] = []
    for candidate in candidates:
        if non_overlapping:
            if any(_windows_overlap(candidate, s) for s in selected):
                continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return selected


# ---------------------------------------------------------------------------
# Diagnostics and reporting
# ---------------------------------------------------------------------------

def print_selection_report(
    suspense_results: List[dict],
    excerpts: List[ExcerptCandidate],
    film_name: str = "Film",
) -> None:
    """Print a human-readable report of the selection process."""
    all_suspense = [r["global_suspense"] for r in suspense_results]
    film_duration = (
        suspense_results[-1]["end_time"] - suspense_results[0]["start_time"]
    )

    print(f"\n{'=' * 65}")
    print(f"  EXCERPT SELECTION REPORT — {film_name}")
    print(f"{'=' * 65}")
    print(f"  Film duration:    {film_duration:.1f}s")
    print(f"  Total events:     {len(suspense_results)}")
    print(f"  Suspense range:   [{min(all_suspense):.3f}, "
          f"{max(all_suspense):.3f}]")
    print(f"  Film inflections: "
          f"{_count_inflections(all_suspense)}")
    print()

    for rank, exc in enumerate(excerpts, 1):
        print(f"  --- Rank {rank} ---")
        print(f"  Time window:      {str(datetime.timedelta(seconds=exc.start_time))}s – "
              f"{str(datetime.timedelta(seconds=exc.end_time))}s  ({exc.duration:.1f}s)")
        print(f"  Time window:      {exc.start_time:.1f}s – "
              f"{exc.end_time:.1f}s  ({exc.duration:.1f}s)")
        print(f"  Events:           {exc.start_event_idx} – "
              f"{exc.end_event_idx}  "
              f"({exc.end_event_idx - exc.start_event_idx + 1} events)")
        print(f"  Inflections:      {exc.inflection_count}")
        print(f"  Suspense range:   {exc.suspense_range:.3f}")
        print(f"  Turning points:   {exc.turning_point_count}")
        print(f"  Suspense curve:   {' → '.join(f'{v:.2f}' for v in exc.suspense_values)}")
        if exc.boundary_extended:
            print(f"  ⚠ BOUNDARY EXTENDED beyond max_duration to complete suspense cycle")
        print()

    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_excerpt_selection(
    suspense_results: List[dict],
    excerpts: List[ExcerptCandidate],
    narrative_data: Optional[dict] = None,
    output_path: str = "excerpt_selection.png",
    dpi: int = 150,
) -> None:
    """Visualise the suspense curve with selected excerpt(s) highlighted.

    Produces two panels:
        Top:    Full suspense curve with excerpt windows shaded
        Bottom: Zoomed view of the selected excerpt with inflection
                points marked

    Args:
        suspense_results: output of compute_suspense()
        excerpts:         output of select_excerpts()
        narrative_data:   optional, for title/description
        output_path:      where to save the figure
        dpi:              output resolution
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_excerpts = len(excerpts)
    fig, axes = plt.subplots(
        1 + n_excerpts, 1,
        figsize=(14, 4 + 3.5 * n_excerpts),
        gridspec_kw={"height_ratios": [1.2] + [1.0] * n_excerpts},
    )
    if n_excerpts == 0:
        axes = [axes]

    title = "Excerpt Selection"
    if narrative_data:
        desc = narrative_data.get("description", "")[:70]
        title = f"Excerpt Selection — {desc}"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Colours for up to 5 excerpts
    excerpt_colours = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    # ── Panel 1: Full suspense curve ──────────────────────────────────
    ax = axes[0]
    ax.set_title("Full Suspense Curve with Selected Excerpt Windows",
                 fontsize=10)

    # Draw suspense as step function
    for r in suspense_results:
        ax.fill_between(
            [r["start_time"], r["end_time"]],
            0, r["global_suspense"],
            alpha=0.15, color="steelblue",
        )
        ax.hlines(
            r["global_suspense"],
            r["start_time"], r["end_time"],
            colors="steelblue", linewidth=2,
        )

    # Connect steps
    for i in range(len(suspense_results) - 1):
        ax.vlines(
            suspense_results[i]["end_time"],
            suspense_results[i]["global_suspense"],
            suspense_results[i + 1]["global_suspense"],
            colors="gray", linewidth=0.8, linestyle="--", alpha=0.4,
        )

    # Mark turning points
    for r in suspense_results:
        if r["is_turning_point"]:
            mid = (r["start_time"] + r["end_time"]) / 2
            ax.axvline(mid, color="red", linewidth=0.8, linestyle=":",
                       alpha=0.6)

    # Shade selected excerpts
    y_max = max(r["global_suspense"] for r in suspense_results) * 1.15
    legend_patches = []
    for idx, exc in enumerate(excerpts):
        colour = excerpt_colours[idx % len(excerpt_colours)]
        ax.axvspan(
            exc.start_time, exc.end_time,
            alpha=0.15, color=colour, zorder=0,
        )
        ax.axvline(exc.start_time, color=colour, linewidth=1.5,
                   linestyle="-", alpha=0.7)
        ax.axvline(exc.end_time, color=colour, linewidth=1.5,
                   linestyle="-", alpha=0.7)
        legend_patches.append(mpatches.Patch(
            color=colour, alpha=0.3,
            label=(f"Excerpt {idx+1}: {exc.start_time:.0f}–"
                   f"{exc.end_time:.0f}s  "
                   f"(inflections={exc.inflection_count}, "
                   f"range={exc.suspense_range:.2f})"),
        ))

    ax.set_ylabel("Global Suspense")
    ax.set_xlabel("Time (seconds)")
    ax.set_xlim(suspense_results[0]["start_time"],
                suspense_results[-1]["end_time"])
    ax.set_ylim(bottom=0)
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Panels 2+: Zoomed excerpt views ──────────────────────────────
    for idx, exc in enumerate(excerpts):
        ax = axes[1 + idx]
        colour = excerpt_colours[idx % len(excerpt_colours)]

        ax.set_title(
            f"Excerpt {idx+1}: {exc.start_time:.1f}s – {exc.end_time:.1f}s  "
            f"({exc.duration:.0f}s, {len(exc.suspense_values)} events, "
            f"{exc.inflection_count} inflections)",
            fontsize=10,
        )

        # Draw events in the excerpt window
        event_mids = []
        for k in range(exc.start_event_idx, exc.end_event_idx + 1):
            r = suspense_results[k]
            mid = (r["start_time"] + r["end_time"]) / 2
            event_mids.append(mid)

            ax.fill_between(
                [r["start_time"], r["end_time"]],
                0, r["global_suspense"],
                alpha=0.2, color=colour,
            )
            ax.hlines(
                r["global_suspense"],
                r["start_time"], r["end_time"],
                colors=colour, linewidth=2.5,
            )

            # Event index label
            ax.text(
                mid, r["global_suspense"] + 0.02 * exc.suspense_range,
                f"E{r['event_id']}", fontsize=7, ha="center",
                color="gray",
            )

            # Mark turning points
            if r["is_turning_point"]:
                ax.annotate(
                    "▲TP", xy=(mid, r["global_suspense"]),
                    xytext=(mid, r["global_suspense"] + 0.15 * exc.suspense_range),
                    fontsize=8, color="red", ha="center",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                )

        # Connect steps
        for k in range(exc.start_event_idx, exc.end_event_idx):
            ax.vlines(
                suspense_results[k]["end_time"],
                suspense_results[k]["global_suspense"],
                suspense_results[k + 1]["global_suspense"],
                colors="gray", linewidth=0.8, linestyle="--", alpha=0.5,
            )

        # Mark inflection points
        vals = exc.suspense_values
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        prev_sign = 0
        for i, d in enumerate(diffs):
            if abs(d) < 1e-6:
                continue
            current_sign = 1 if d > 0 else -1
            if prev_sign != 0 and current_sign != prev_sign:
                # Inflection at event i+1 (relative to window)
                k = exc.start_event_idx + i + 1
                r = suspense_results[k]
                mid = (r["start_time"] + r["end_time"]) / 2
                ax.plot(mid, r["global_suspense"], "o",
                        color="darkred", markersize=8, zorder=5)
                ax.annotate(
                    "inflection", xy=(mid, r["global_suspense"]),
                    xytext=(mid + 2, r["global_suspense"] - 0.1 * exc.suspense_range),
                    fontsize=7, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.6),
                )
            prev_sign = current_sign

        ax.set_ylabel("Global Suspense")
        ax.set_xlabel("Time (seconds)")
        ax.set_xlim(exc.start_time - 2, exc.end_time + 2)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved excerpt selection plot to {output_path}")


# ---------------------------------------------------------------------------
# Standalone usage / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo with synthetic data to verify the algorithm works
    print("=" * 65)
    print("  DEMO 1: Varied suspense patterns")
    print("=" * 65)

    # Simulate a ~4 minute film with 30 events
    np.random.seed(42)
    n_events = 30
    event_duration = 8.0  # ~8 seconds per event

    # Create a suspense curve with clear patterns:
    # - slow build (events 0-8)
    # - sharp spike and drop (events 9-14)  ← should be selected
    # - flat middle (events 15-20)
    # - oscillating tension (events 21-28)  ← also good
    # - resolution (event 29)

    base_curve = np.concatenate([
        np.linspace(0.1, 0.4, 9),        # slow build
        [0.5, 0.8, 1.2, 0.6, 0.3, 0.7],  # spike-drop-spike
        np.full(6, 0.35),                  # flat
        [0.5, 0.9, 0.4, 0.8, 0.3, 0.7, 0.5, 0.2],  # oscillating
        [0.1],                             # resolution
    ])

    # Add small noise
    suspense_values = base_curve + np.random.normal(0, 0.02, n_events)
    suspense_values = np.maximum(suspense_values, 0)

    # Build mock suspense_results
    mock_results = []
    for i in range(n_events):
        mock_results.append({
            "event_id": i + 1,
            "start_time": i * event_duration,
            "end_time": (i + 1) * event_duration,
            "global_suspense": float(suspense_values[i]),
            "is_turning_point": i in [10, 13, 22, 25],
            "thread_suspense": {},
        })

    # Run selection
    excerpts = select_excerpts(
        mock_results,
        min_duration=50,
        max_duration=90,
        min_range_fraction=0.15,
        top_k=2,
    )

    print_selection_report(mock_results, excerpts, film_name="Synthetic Demo")

    # Generate plot
    plot_excerpt_selection(
        mock_results, excerpts,
        narrative_data={"description": "Synthetic demo film with varied suspense patterns"},
        output_path="excerpt_selection_demo.png",
    )

    # ===================================================================
    # DEMO 2: Long build with peak at boundary (tests extension)
    # ===================================================================
    print("\n" + "=" * 65)
    print("  DEMO 2: Long build peaking at window boundary")
    print("  (tests boundary extension)")
    print("=" * 65)

    # Real Gemini extractions have VARIABLE-duration events.
    # Build-up scenes tend to be long (~12-15s each) while dramatic
    # peaks get subdivided into rapid short events (~4-6s).
    # This creates the boundary problem: the build fills the window,
    # and the short peak events barely extend beyond max_duration.
    #
    # To isolate the extension effect, we use uniform 15s events:
    # a 100s window fits exactly 6 events (90s) but NOT 7 (105s).
    # The build spans events 0-6, peak at event 7, drop at event 8.
    # Any strict 6-event window ending at the peak (events 2-7, 90s)
    # is monotonic.  Extension adds event 8 (the drop) → 1 inflection.

    event_data = [
        # (duration_s, suspense, is_tp)
        (15, 0.10, False),   # 0: 0-15s    quiet opening
        (15, 0.20, False),   # 1: 15-30s   build
        (15, 0.30, False),   # 2: 30-45s   build
        (15, 0.42, False),   # 3: 45-60s   build
        (15, 0.55, False),   # 4: 60-75s   build
        (15, 0.70, False),   # 5: 75-90s   build
        (15, 0.85, True),    # 6: 90-105s  build (turning point)
        (15, 1.10, False),   # 7: 105-120s PEAK
        (15, 0.35, True),    # 8: 120-135s SHARP DROP
        (15, 0.25, False),   # 9: 135-150s aftermath
        (15, 0.40, False),   # 10: 150-165s recovery
        (15, 0.20, False),   # 11: 165-180s resolution
    ]

    mock_results_2 = []
    t = 0.0
    for i, (dur, susp, tp) in enumerate(event_data):
        mock_results_2.append({
            "event_id": i + 1,
            "start_time": t,
            "end_time": t + dur,
            "global_suspense": susp,
            "is_turning_point": tp,
            "thread_suspense": {},
        })
        t += dur

    print(f"\n  Film duration: {t:.0f}s, {len(event_data)} events")
    print(f"  All events 15s → 100s window fits 6 events (90s), not 7 (105s)")
    print(f"  Build: events 0-7, peak at event 7 (105-120s)")
    print(f"  Drop at event 8 (120-135s) — just beyond strict window reach")

    # Without boundary extension
    print("\n  --- Without boundary extension (boundary_extend_events=0) ---")
    excerpts_strict = select_excerpts(
        mock_results_2,
        min_duration=75,
        max_duration=100,
        min_range_fraction=0.15,
        top_k=1,
        boundary_extend_events=0,
    )
    print_selection_report(mock_results_2, excerpts_strict,
                           film_name="Metropolis-like (strict)")
    plot_excerpt_selection(
        mock_results_2, excerpts_strict,
        narrative_data={"description": "Metropolis-like: STRICT — no boundary extension"},
        output_path="excerpt_selection_strict.png",
    )

    # With boundary extension
    print("\n  --- With boundary extension (boundary_extend_events=2) ---")
    excerpts_extended = select_excerpts(
        mock_results_2,
        min_duration=75,
        max_duration=100,
        min_range_fraction=0.15,
        top_k=1,
        boundary_extend_events=2,
    )
    print_selection_report(mock_results_2, excerpts_extended,
                           film_name="Metropolis-like (extended)")

    plot_excerpt_selection(
        mock_results_2, excerpts_extended,
        narrative_data={"description": "Metropolis-like: boundary extension captures peak+drop"},
        output_path="excerpt_selection_boundary_demo.png",
    )