import bisect
import math
import warnings
from fractions import Fraction
from config import TEMPO_MIN_BPM, TEMPO_MAX_BPM
from narrative.schema import Event
from music.sections import Section

def find_tempo(
    duration: float,
    bpm_center: int,
    bpm_std: float,
    time_signature: tuple[int, int] = (4, 4),
    bpm_min: int = TEMPO_MIN_BPM,
    bpm_max: int = TEMPO_MAX_BPM,
) -> dict:
    """
    Find the closest integer BPM to bpm_center such that the number of
    complete bars fitting in `duration` seconds is a multiple of 2 (and >= 2),
    and the total bar duration never exceeds `duration`.

    Parameters
    ----------
    duration       : float  - target duration in seconds
    bpm_center     : int    - reference / preferred tempo in BPM
    bpm_std        : float  - standard deviation of the acceptable BPM range
    time_signature : tuple  - (numerator, denominator), e.g. (4, 4) or (3, 4)
    bpm_min        : int    - absolute minimum BPM to consider
    bpm_max        : int    - absolute maximum BPM to consider

    Returns
    -------
    dict with keys:
        bpm           - chosen tempo (int)
        bars          - number of complete bars (int, multiple of 2)
        bar_duration  - duration of one bar in seconds (float)
        used_duration - total duration of all bars in seconds (float)
        remaining     - leftover seconds after the last bar (float)
        in_range      - True if result is within n_std * bpm_std of bpm_center
        time_signature- the time signature used
    """

    numerator, denominator = time_signature

    def bar_duration_seconds(bpm: int) -> float:
        """Duration of one bar in seconds for a given BPM and time signature."""
        # One beat = 60 / bpm seconds
        # One bar  = numerator beats × (denominator / 4) scaling
        # Standard: one beat is a quarter note (denominator=4 reference)
        beat_duration = 60.0 / bpm
        return numerator * beat_duration * (4.0 / denominator)

    def is_valid(bpm: int) -> bool:
        """Return True if this BPM yields an even number of bars >= 2."""
        if bpm < bpm_min or bpm > bpm_max:
            return False
        bar_dur = bar_duration_seconds(bpm)
        bars = math.floor(duration / bar_dur)
        return bars >= 2 and bars % 2 == 0

    def is_valid_flex(bpm: int) -> bool:
        """Return True if this BPM yields an even number of bars >= 2."""
        if bpm < bpm_min or bpm > bpm_max:
            return False
        return True

    def build_result(bpm: int, in_range: bool) -> dict:
        bar_dur = bar_duration_seconds(bpm)
        bars = math.floor(duration / bar_dur)
        used = bars * bar_dur
        remaining = duration - used
        return {
            "bpm": bpm,
            "bars": bars,
            "bar_duration": round(bar_dur, 6),
            "used_duration": round(used, 6),
            "remaining": round(remaining, 6),
            "in_range": in_range,
            "time_signature": time_signature,
        }

    bpm_lo = round(bpm_center - bpm_std)
    bpm_hi = round(bpm_center + bpm_std)

    # ------------------------------------------------------------------ #
    # Phase 1: search outward from bpm_center within ±n_std * bpm_std    #
    # ------------------------------------------------------------------ #
    offset = 0
    flexible_candidate = None
    while True:
        for candidate in _unique_offsets(bpm_center, offset):
            if bpm_lo <= candidate <= bpm_hi:
                if is_valid(candidate):
                    return build_result(candidate, in_range=True)
                elif flexible_candidate == None and is_valid_flex(candidate):
                    flexible_candidate = candidate
        # Once the search window has expanded beyond both bounds, stop phase 1
        if bpm_center - offset < bpm_lo and bpm_center + offset > bpm_hi:
            if flexible_candidate != None:
                warnings.warn(
                    f"No valid BPM found within [{bpm_lo}, {bpm_hi}] "
                    f"(bpm_center={bpm_center}, bpm_std={bpm_std}) "
                    f"for duration={duration}s in {time_signature[0]}/{time_signature[1]}. "
                    f"but found valid BPM without constraints {flexible_candidate}.",
                    UserWarning,
                    stacklevel=2,
                )
                return build_result(flexible_candidate, in_range=True)
            break
        offset += 1

    # ------------------------------------------------------------------ #
    # Phase 2: no valid BPM found in the preferred window → warn and     #
    #          search the full [bpm_min, bpm_max] range                  #
    # ------------------------------------------------------------------ #
    warnings.warn(
        f"No valid BPM found within [{bpm_lo}, {bpm_hi}] "
        f"(bpm_center={bpm_center}, bpm_std={bpm_std}) "
        f"for duration={duration}s in {time_signature[0]}/{time_signature[1]}. "
        f"Falling back to full search in [{bpm_min}, {bpm_max}].",
        UserWarning,
        stacklevel=2,
    )

    offset = 0
    while True:
        for candidate in _unique_offsets(bpm_center, offset):
            if bpm_min <= candidate <= bpm_max:
                if is_valid(candidate):
                    return build_result(candidate, in_range=False)
        if bpm_center - offset < bpm_min and bpm_center + offset > bpm_max:
            break
        offset += 1

    # ------------------------------------------------------------------ #
    # Should never reach here given a reasonable bpm_min/bpm_max range,  #
    # but raise explicitly if it does                                     #
    # ------------------------------------------------------------------ #
    raise ValueError(
        f"Could not find any valid BPM in [{bpm_min}, {bpm_max}] "
        f"for duration={duration}s."
    )


def _unique_offsets(center: int, offset: int):
    """
    Yield the 1 or 2 BPM candidates for a given offset step.
    offset=0 → [center]
    offset>0 → [center+offset, center-offset]
    """
    yield center + offset
    if offset > 0:
        yield center - offset
        

def assign_bars_to_events(
    section: Section,
    tempo_result: dict,
) -> list[dict]:
    """
    Attribute each bar to an event based on the bar's midpoint.

    A bar belongs to the event whose range [start_time, end_time) contains
    the bar's midpoint. Events must be contiguous and non-overlapping and
    collectively cover the section.

    Parameters
    ----------
    section       : Section  - with start_time, end_time, duration
    events        : list     - sorted by start_time, each with start_time,
                               end_time, and index
    tempo_result  : dict     - output from find_tempo() (needs bar_duration, bars)

    Returns
    -------
    List of dicts, one per event, each with:
        event_index  - the event's index
        start_time   - event start time
        end_time     - event end time
        bars         - list of bar indices assigned to this event
        n_bars       - number of bars assigned
    """
    bar_duration = tempo_result["bar_duration"]
    n_bars = tempo_result["bars"]

    events = section.events

    start_times = [e["start_time"] for e in events]

    def event_index_for_midpoint(midpoint: float) -> int:
        """
        Binary search: find the last event whose start_time <= midpoint.
        Clamp to valid range to handle any floating-point overshoot.
        """
        idx = bisect.bisect_right(start_times, midpoint) - 1
        return max(0, min(idx, len(events) - 1))

    # Accumulate bar indices per event
    bar_lists: list[list[int]] = [[] for _ in events]

    for i in range(n_bars):
        midpoint = section.start_time + (i + 0.5) * bar_duration
        idx = event_index_for_midpoint(midpoint)
        bar_lists[idx].append(i)

    return [
        {
            "event_id": e["event_id"],
            "start_time": e["start_time"],
            "end_time": e["end_time"],
            "bars": bar_lists[j],
            "n_bars": len(bar_lists[j]),
        }
        for j, e in enumerate(events)
    ]
