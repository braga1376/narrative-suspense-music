from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import RHO, PHI, BETA, TURNING_POINT_BOOST, TURNING_POINT_BOOST_THRESHOLD, PLOT_OUTPUT_PATH, PLOT_DPI

def compute_suspense(narrative_data: dict) -> List[dict]:
    threads = {t["thread_id"]: t for t in narrative_data["threads"]}
    events = narrative_data["events"]

    # --- Precomputation ---
    remaining_at_step = _precompute_remaining_events(threads, events)

    # For each (step, thread_id): distance to next disallowing event
    interruption_at_step = _precompute_interruption_distances(threads, events)

    # --- Thread state ---
    thread_state = {
        tid: {
            "active": False,
            "events_told": 0,
            "disallow_attempts": 0,
            "last_active_step": -1,
        }
        for tid in threads
    }

    results = []
    rho = RHO
    phi = PHI
    beta = BETA

    for step, event in enumerate(events):
        active_thread_ids = set(event["thread_ids"])

        # 1. Activate threads appearing for first time, update told events and last active step
        for tid in active_thread_ids:
            thread_state[tid]["active"] = True
            thread_state[tid]["events_told"] += 1
            thread_state[tid]["last_active_step"] = step

        # 2. Compute suspense
        thread_suspense = {}

        for tid, thread in threads.items():
            state = thread_state[tid]
            if not state["active"]:
                continue

            importance = thread["importance"]
            P = state["events_told"]
            Q = state["disallow_attempts"]

            # --- Imminence ---
            remaining = remaining_at_step[step].get(tid, 0)
            total = P + remaining
            completion_imminence = P / total if total > 0 else 1.0

            # Added turning point to original model
            if event["is_turning_point"] and completion_imminence < TURNING_POINT_BOOST_THRESHOLD:
                completion_imminence = min(1.0, completion_imminence * TURNING_POINT_BOOST)

            R = interruption_at_step[step].get(tid)
            if R is not None and R > 0:
                imminence = (rho * completion_imminence +
                            (1 - rho) / R)
            else:
                imminence = completion_imminence

            # --- Foregroundedness ---
            if tid in active_thread_ids:
                foregroundedness = 1.0
            else:
                foregroundedness = beta ** (step - state["last_active_step"])

            # --- Confidence ---
            confidence = 0.5 if P == 0 else 1 / (1 + phi * Q / P)

            # --- Suspense ---
            suspense = imminence * importance * foregroundedness * confidence
            thread_suspense[tid] = {
                "suspense": suspense,
                "imminence": imminence,
                "importance": importance,
                "foregroundedness": foregroundedness,
                "confidence": confidence,
            }

        # 3. Update disallow attempts
        for tid in event["disallows_thread_ids"]:
            if thread_state[tid]["active"]:
                thread_state[tid]["disallow_attempts"] += 1

        # 4. Deactivate explicitly disallowed threads
        for tid in event["disallows_thread_ids"]:
            thread_state[tid]["active"] = False

        # 5. Deactivate naturally completed threads —
        #    active threads with no remaining events and not in current event
        for tid in list(thread_state.keys()):
            state = thread_state[tid]
            if (state["active"] and
                tid not in active_thread_ids and
                remaining_at_step[step].get(tid, 0) == 0 and
                state["events_told"] > 0):
                thread_state[tid]["active"] = False

        # 6. Global suspense
        global_suspense = (
            max(abs(v["suspense"]) for v in thread_suspense.values())
            if thread_suspense else 0.0
        )

        results.append({
            "event_id": event["event_id"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "global_suspense": global_suspense,
            "thread_suspense": thread_suspense,
            "is_turning_point": event["is_turning_point"],
        })

    return results


def _precompute_remaining_events(
        threads: dict,
        events: List[dict]
    ) -> List[dict]:
    """
    Precompute remaining event counts for each thread at each step.
    Returns list of dicts: remaining_at_step[step][tid] = count
    
    Built using a suffix sum approach:
    scan backwards, incrementing counts when thread appears.
    """
    n = len(events)
    # Initialize all steps with zero counts
    remaining = [{tid: 0 for tid in threads} for _ in range(n)]

    # Scan backwards — at each step, remaining = next step's remaining
    # plus whether current step's next event contains this thread
    for step in range(n - 2, -1, -1):
        for tid in threads:
            remaining[step][tid] = remaining[step + 1][tid]
        for tid in events[step + 1]["thread_ids"]:
            if tid in threads:
                remaining[step][tid] += 1

    return remaining


def _precompute_interruption_distances(
        threads: dict,
        events: List[dict]
    ) -> List[dict]:
    """
    Precompute interruption distances for each thread at each step.
    Returns list of dicts: interruption_at_step[step][tid] = R or None
    
    Built in O(n·t) by scanning backwards:
    if a future event disallows tid, distance is 1 + distance from next step.
    """
    n = len(events)
    # None means no interruption found
    interruption = [{tid: None for tid in threads} for _ in range(n)]

    # Scan backwards
    for step in range(n - 2, -1, -1):
        next_event = events[step + 1]

        for tid in threads:
            if tid in next_event["disallows_thread_ids"]:
                # Next event directly disallows this thread
                interruption[step][tid] = 1
            elif interruption[step + 1][tid] is not None:
                # Interruption exists further ahead — add 1
                interruption[step][tid] = interruption[step + 1][tid] + 1
            else:
                interruption[step][tid] = None

    return interruption

def plot_suspense(suspense_results: list, narrative_data: dict, output_path: str) -> None:
    """
    Visualize suspense metrics from Doust & Piwek computation.
    
    Produces three panels:
    1. Global suspense curve over time with turning points marked
    2. Per-thread suspense heatmap over time
    3. Per-thread component breakdown (imminence, foregroundedness, confidence)
    """
    threads = {t["thread_id"]: t for t in narrative_data["threads"]}
    thread_ids = sorted(threads.keys())
    
    # --- Extract time axis (use midpoint of each event) ---
    times = [
        (r["start_time"] + r["end_time"]) / 2 
        for r in suspense_results
    ]
    start_times = [r["start_time"] for r in suspense_results]
    end_times = [r["end_time"] for r in suspense_results]
    turning_points = [r for r in suspense_results if r["is_turning_point"]]

    # --- Color palette per thread ---
    colors = plt.cm.Set2(np.linspace(0, 1, len(thread_ids)))
    thread_colors = {tid: colors[i] for i, tid in enumerate(thread_ids)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Suspense Analysis — {narrative_data['description'][:80]}...",
        fontsize=11, fontweight='bold', wrap=True
    )

    # ================================================================
    # PANEL 1: Global suspense curve
    # ================================================================
    ax1 = axes[0]
    ax1.set_title("Global Suspense Over Time", fontweight='bold')

    # Draw suspense as step function matching event duration
    for i, result in enumerate(suspense_results):
        dominant_sign = (
            max(result["thread_suspense"].values(),
                key=lambda x: abs(x["suspense"]))["suspense"]
            if result["thread_suspense"] else 0.0
        )
        ax1.fill_between(
            [result["start_time"], result["end_time"]],
            0, result["global_suspense"],
            alpha=0.3,
            color = 'steelblue' if dominant_sign >= 0 else 'tomato',
        )
        ax1.hlines(
            result["global_suspense"],
            result["start_time"], result["end_time"],
            color = 'steelblue' if dominant_sign >= 0 else 'tomato',
            linewidth=2
        )

    # Connect steps with vertical lines
    for i in range(len(suspense_results) - 1):
        ax1.vlines(
            suspense_results[i]["end_time"],
            suspense_results[i]["global_suspense"],
            suspense_results[i + 1]["global_suspense"],
            colors='gray', linewidth=1, linestyle='--', alpha=0.5
        )

    # Mark turning points
    for tp in turning_points:
        mid = (tp["start_time"] + tp["end_time"]) / 2
        ax1.axvline(mid, color='red', linewidth=1, linestyle=':', alpha=0.7)
        ax1.annotate(
            '▲TP',
            xy=(mid, tp["global_suspense"]),
            xytext=(mid + 3, tp["global_suspense"] + 0.3),
            fontsize=7, color='red'
        )

    ax1.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax1.set_ylabel("Suspense Value")
    ax1.set_xlim(start_times[0], end_times[-1])
    ax1.grid(axis='y', alpha=0.3)

    # ================================================================
    # PANEL 2: Per-thread suspense heatmap
    # ================================================================
    ax2 = axes[1]
    ax2.set_title("Per-Thread Suspense Over Time", fontweight='bold')

    for tid in thread_ids:
        thread_suspense_values = []
        for result in suspense_results:
            ts = result["thread_suspense"]
            val = ts[tid]["suspense"] if tid in ts else 0.0
            thread_suspense_values.append(val)

        label = (
            f"T{tid}: {threads[tid]['description'][:40]}... "
            f"(imp={threads[tid]['importance']})"
        )

        # Draw as step function
        for i, result in enumerate(suspense_results):
            ax2.hlines(
                thread_suspense_values[i],
                result["start_time"], result["end_time"],
                colors=[thread_colors[tid]],
                linewidth=2.5,
                alpha=0.85
            )

        # Add line for legend
        ax2.plot([], [], color=thread_colors[tid], linewidth=2.5, label=label)

    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel("Thread Suspense")
    ax2.set_xlim(start_times[0], end_times[-1])
    ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.15),
               ncol=1, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)

    # ================================================================
    # PANEL 3: Component breakdown for dominant thread per event
    # ================================================================
    ax3 = axes[2]
    ax3.set_title(
        "Suspense Components — Dominant Thread per Event",
        fontweight='bold'
    )

    component_keys = ["imminence", "foregroundedness", "confidence"]
    component_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bar_width = [(e - s) * 0.25 for s, e in zip(start_times, end_times)]

    for i, result in enumerate(suspense_results):
        ts = result["thread_suspense"]
        if not ts:
            continue

        # Find dominant thread (highest abs suspense)
        dominant_tid = max(ts.keys(), key=lambda t: abs(ts[t]["suspense"]))
        dominant = ts[dominant_tid]
        mid = times[i]
        w = bar_width[i]

        for j, (comp, col) in enumerate(
            zip(component_keys, component_colors)
        ):
            offset = (j - 1) * w
            ax3.bar(
                mid + offset, dominant[comp],
                width=w, color=col, alpha=0.75,
                label=comp if i == 0 else ""
            )

        # Label dominant thread
        ax3.annotate(
            f"T{dominant_tid}",
            xy=(mid, 0.05),
            fontsize=6, ha='center', color='gray'
        )

    ax3.set_ylabel("Component Value (0-1)")
    ax3.set_xlim(start_times[0], end_times[-1])
    ax3.set_ylim(0, 1.1)
    ax3.legend(
        handles=[
            mpatches.Patch(color=c, label=k)
            for k, c in zip(component_keys, component_colors)
        ],
        fontsize=8, loc='upper right'
    )
    ax3.grid(axis='y', alpha=0.3)

    # --- Shared x-axis formatting ---
    for ax in axes:
        ax.set_xlabel("Time (seconds)")
        # Add event boundary markers
        for s in start_times:
            ax.axvline(s, color='gray', linewidth=0.4, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    else:
        plt.savefig(PLOT_OUTPUT_PATH, dpi=PLOT_DPI, bbox_inches='tight')

    plt.show()
    print("Saved to suspense_analysis.png")