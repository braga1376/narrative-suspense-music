"""
music/objectives.py

NSGA-II objective functions.  All three are minimised (return [0, 1]):

    1. tonal_incoherence    — melody-chord compatibility, beat-weighted
    2. motif_recognition    — 1 - best motif match (contour + rhythm + magnitude)
    3. tension_misalignment — (1 - Pearson(musical_tension, suspense)) / 2

Each function takes the rendered Arrangements and returns a float.

References:
    Farbood, M. (2012). A parametric, temporal model of musical tension.
    Chew, E. (2014). Mathematical and Computational Modeling of Tonality.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr

from .voicing import Arrangement, HarmonicChord, VoicedNote, get_chord_pitches
from .structures import Motif
from .tension_ribbons import calculate_tension_fast

from config import (
    TEMPO_MAX_BPM, TEMPO_MIN_BPM, TENSION_WEIGHT_STRAIN, TENSION_WEIGHT_DIAMETER, TENSION_WEIGHT_MOMENTUM,
    TENSION_FARBOOD_LOUDNESS, TENSION_FARBOOD_PITCH,
    TENSION_FARBOOD_TEMPO, TENSION_FARBOOD_ONSET, TENSION_FARBOOD_TONAL,
    TENSION_TREND_BETA, TENSION_MIN_FEATURE_STD, TENSION_ABSOLUTE_WEIGHT,
    TONAL_INCOHERENCE_SCALE_PENALTY, TONAL_INCOHERENCE_OUTSIDE_PENALTY,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_rest(note: VoicedNote) -> bool:
    """A rest is a note with velocity 0 (produced by REST_PITCH in genome)."""
    return note.velocity == 0


def _sounding_notes(notes: List[VoicedNote]) -> List[VoicedNote]:
    """Filter out rest notes — returns only sounding (velocity > 0) notes."""
    return [n for n in notes if not _is_rest(n)]

def _melody_per_chord(
    melody_notes: List[VoicedNote],
    harmonic_chords: List[HarmonicChord],
) -> List[List[VoicedNote]]:
    """Assign melody notes to the HarmonicChord whose time window they
    start in.  Returns one list per chord."""
    assigned: List[List[VoicedNote]] = [[] for _ in harmonic_chords]
    if not harmonic_chords or not melody_notes:
        return assigned

    # Build windows
    windows: List[Tuple[float, float]] = []
    beat = 0.0
    for hc in harmonic_chords:
        windows.append((beat, beat + hc.chord.duration))
        beat += hc.chord.duration

    note_beat = 0.0
    chord_idx = 0
    n = len(harmonic_chords)

    for note in melody_notes:
        while chord_idx < n - 1 and note_beat >= windows[chord_idx][1]:
            chord_idx += 1
        assigned[chord_idx].append(note)
        note_beat += note.duration

    return assigned


def _event_suspense_map(narrative_data: dict) -> Dict[int, float]:
    """Return {event_id: suspense} from narrative data."""
    return {
        e["event_id"]: float(e.get("global_suspense"))
        for e in narrative_data["events"]
    }


# ---------------------------------------------------------------------------
# Objective 1: tonal_incoherence
# ---------------------------------------------------------------------------

# Harmonic repetition penalty settings
REPETITION_FREE_LIMIT = 2     # consecutive same chords before penalty starts
REPETITION_PENALTY_WEIGHT = 0.3  # blend weight (0.7 melody + 0.3 harmony)


def _harmonic_repetition_penalty(
    harmonic_chords: List[HarmonicChord],
) -> float:
    """Penalty for consecutive repeated chords.

    2 in a row: no penalty (common in music)
    3 in a row: mild penalty
    4+: increasing penalty

    Returns value in [0.0, 1.0], lower is better.
    """
    if len(harmonic_chords) < 2:
        return 0.0

    total_penalty = 0.0
    total_chords  = len(harmonic_chords)
    run_length    = 1

    for i in range(1, total_chords):
        curr_rn = harmonic_chords[i].chord.roman_numeral
        prev_rn = harmonic_chords[i - 1].chord.roman_numeral

        if curr_rn == prev_rn:
            run_length += 1
        else:
            # Score the completed run
            excess = max(0, run_length - REPETITION_FREE_LIMIT)
            total_penalty += excess * excess  # quadratic: 1, 4, 9, ...
            run_length = 1

    # Score the final run
    excess = max(0, run_length - REPETITION_FREE_LIMIT)
    total_penalty += excess * excess

    # Normalise: maximum possible penalty is when all chords are the same
    max_excess = max(0, total_chords - REPETITION_FREE_LIMIT)
    max_penalty = max_excess * max_excess if max_excess > 0 else 1.0

    return min(1.0, total_penalty / max_penalty) if max_penalty > 0 else 0.0


def tonal_incoherence(
    arrangements: Dict[int, Arrangement],
    narrative_data: dict,
    motifs: List[Motif],
) -> float:
    """Combined melody-chord compatibility and harmonic diversity.

    Two components blended:
        melody_penalty (weight 0.7): beat-weighted penalty for melody
            notes outside their chord (chord→0, scale→0.5, outside→1.0)
        harmony_penalty (weight 0.3): quadratic penalty for consecutive
            repeated chords (2 free, then escalating)

    Returns [0.0, 1.0], lower is better.
    """
    melody_penalty_sum = 0.0
    melody_beats       = 0.0
    harmony_penalties: List[float] = []

    for arr in arrangements.values():
        if not arr.melody_notes or not arr.harmonic_chords:
            continue

        # Melody component
        mpc = _melody_per_chord(arr.melody_notes, arr.harmonic_chords)
        key_pcs = arr.key.pitch_classes

        for hc, mel_notes in zip(arr.harmonic_chords, mpc):
            chord_pcs = set(get_chord_pitches(hc.chord))

            for note in mel_notes:
                if _is_rest(note):
                    continue  # rests don't affect tonal coherence

                pc  = note.pitch % 12
                dur = note.duration

                if pc in chord_pcs:
                    penalty = 0.0
                elif pc in key_pcs:
                    penalty = TONAL_INCOHERENCE_SCALE_PENALTY
                else:
                    penalty = TONAL_INCOHERENCE_OUTSIDE_PENALTY

                melody_penalty_sum += penalty * dur
                melody_beats       += dur

        # Harmony component
        harmony_penalties.append(
            _harmonic_repetition_penalty(arr.harmonic_chords))

    mel_score = melody_penalty_sum / melody_beats if melody_beats > 0 else 0.0
    har_score = (sum(harmony_penalties) / len(harmony_penalties)
                 if harmony_penalties else 0.0)

    return (1.0 - REPETITION_PENALTY_WEIGHT) * mel_score + REPETITION_PENALTY_WEIGHT * har_score


# ---------------------------------------------------------------------------
# Objective 2: motif_recognition
# ---------------------------------------------------------------------------

# Weights for the three recognition components
RECOGNITION_W_CONTOUR   = 0.35
RECOGNITION_W_RHYTHM    = 0.35
RECOGNITION_W_MAGNITUDE = 0.30

# Max semitone difference to count as "close" for magnitude matching
MAGNITUDE_TOLERANCE = 2


def _interval_contour(notes: List[VoicedNote]) -> List[int]:
    """Direction signs: +1 / 0 / -1 between consecutive notes."""
    return [
        (1 if notes[i+1].pitch > notes[i].pitch
         else (-1 if notes[i+1].pitch < notes[i].pitch else 0))
        for i in range(len(notes) - 1)
    ]


def _interval_magnitudes(notes: List[VoicedNote]) -> List[int]:
    """Absolute interval sizes in semitones."""
    return [abs(notes[i+1].pitch - notes[i].pitch) for i in range(len(notes) - 1)]


def _rhythm_pattern(notes: List[VoicedNote]) -> List[float]:
    """Normalised duration ratios."""
    total = sum(n.duration for n in notes)
    if total == 0:
        return [1.0 / max(1, len(notes))] * len(notes)
    return [n.duration / total for n in notes]


def _window_similarity(
    window: List[VoicedNote],
    motif_contour: List[int],
    motif_magnitudes: List[int],
    motif_rhythm: List[float],
) -> float:
    """Three-component similarity between a melody window and a motif.

    1. Contour:   fraction of direction signs that match
    2. Rhythm:    1 - mean clamped ratio deviation
    3. Magnitude: fraction of intervals within ±MAGNITUDE_TOLERANCE
    """
    n = len(motif_contour)
    if len(window) < n + 1:
        return 0.0

    # Contour
    win_contour = _interval_contour(window[:n+1])
    contour_sim = sum(w == m for w, m in zip(win_contour, motif_contour)) / n if n else 1.0

    # Rhythm
    win_rhythm = _rhythm_pattern(window[:n+1])
    mot_rhythm = motif_rhythm[:n+1]
    devs = [min(abs(wr / mr - 1.0), 1.0) if mr > 0 else 0.0
            for wr, mr in zip(win_rhythm, mot_rhythm)]
    rhythm_sim = 1.0 - (sum(devs) / len(devs)) if devs else 1.0

    # Magnitude
    win_mag = _interval_magnitudes(window[:n+1])
    mag_matches = sum(1 for wm, mm in zip(win_mag, motif_magnitudes)
                      if abs(wm - mm) <= MAGNITUDE_TOLERANCE)
    magnitude_sim = mag_matches / n if n else 1.0

    return (RECOGNITION_W_CONTOUR   * contour_sim
            + RECOGNITION_W_RHYTHM   * rhythm_sim
            + RECOGNITION_W_MAGNITUDE * magnitude_sim)


# Cache for motif features (they never change during evolution)
_motif_feature_cache: Dict[int, Tuple] = {}


def _get_motif_features(motif: Motif) -> Tuple:
    """Get or compute cached motif features."""
    cid = motif.character_id
    if cid not in _motif_feature_cache:
        motif_voiced = [
            VoicedNote(n.pitch, n.duration, n.velocity, "melody")
            for n in motif.notes
        ]
        _motif_feature_cache[cid] = (
            _interval_contour(motif_voiced),
            _interval_magnitudes(motif_voiced),
            _rhythm_pattern(motif_voiced),
            len(motif.notes),
        )
    return _motif_feature_cache[cid]


def motif_recognition(
    arrangements: Dict[int, Arrangement],
    narrative_data: dict,
    motifs: List[Motif],
) -> float:
    """1 - mean best sliding-window motif match across all characters.

    Uses three-component scoring (contour + rhythm + magnitude) so
    development cells that use the motif's interval vocabulary receive
    partial recognition credit.

    Returns float in [0.0, 1.0], lower is better.
    """
    if not motifs:
        return 1.0

    # Collect full-piece melody (sounding notes only)
    all_melody: List[VoicedNote] = []
    for s_idx in sorted(arrangements.keys()):
        all_melody.extend(_sounding_notes(arrangements[s_idx].melody_notes))

    if not all_melody:
        return 1.0

    char_scores: List[float] = []

    for motif in motifs:
        if len(motif.notes) < 2:
            continue

        # Use cached features
        m_contour, m_magnitudes, m_rhythm, win_len = _get_motif_features(motif)

        best_sim = 0.0
        for i in range(len(all_melody) - win_len + 1):
            sim = _window_similarity(
                all_melody[i:i + win_len],
                m_contour, m_magnitudes, m_rhythm,
            )
            if sim > best_sim:
                best_sim = sim

        char_scores.append(best_sim)

    if not char_scores:
        return 1.0

    return 1.0 - (sum(char_scores) / len(char_scores))


# ---------------------------------------------------------------------------
# Objective 3: tension_misalignment
# ---------------------------------------------------------------------------

BEATS_PER_BAR = 4


# --- Fixed bar grid construction ---

@dataclass
class BarWindow:
    """One fixed-size bar window with all sounding content."""
    bar_idx: int
    start_beat: float
    end_beat: float
    event_id: int
    # Collected content (duration-weighted)
    melody_pitches: List[int]       # pitches of melody notes overlapping this bar
    melody_velocities: List[int]    # velocities of melody notes overlapping this bar
    melody_durations: List[float]   # overlap duration of each melody note in this bar
    melody_onsets: int              # number of melody notes that START in this bar
    harmonic_onsets: int            # number of chord onsets (×3 voices) in this bar
    harmonic_pitches: List[int]     # A/T/B pitches (from chords overlapping this bar)
    harmonic_velocities: List[int]  # A/T/B velocities
    all_pitches: List[int]          # combined pitches for spiral array


def _build_bar_grid(arr: Arrangement) -> List[BarWindow]:
    """Build a fixed bar grid from an Arrangement.

    Each bar is exactly BEATS_PER_BAR beats wide. For each bar:
    - Collect melody notes that overlap the bar window
    - Collect harmonic voice pitches from the chord active at bar midpoint
    - Determine event_id from the chord at bar midpoint
    """
    total_beats = sum(hc.chord.duration for hc in arr.harmonic_chords)
    n_bars = max(1, int(round(total_beats / BEATS_PER_BAR)))

    # Build chord timeline: [(start_beat, end_beat, HarmonicChord), ...]
    chord_timeline: List[Tuple[float, float, HarmonicChord]] = []
    beat = 0.0
    for hc in arr.harmonic_chords:
        chord_timeline.append((beat, beat + hc.chord.duration, hc))
        beat += hc.chord.duration

    # Build melody timeline: [(start_beat, end_beat, VoicedNote), ...]
    melody_timeline: List[Tuple[float, float, VoicedNote]] = []
    beat = 0.0
    for note in arr.melody_notes:
        melody_timeline.append((beat, beat + note.duration, note))
        beat += note.duration

    bars: List[BarWindow] = []

    for bar_idx in range(n_bars):
        bar_start = bar_idx * BEATS_PER_BAR
        bar_end = bar_start + BEATS_PER_BAR

        # Find chord at bar midpoint → determines event_id and harmonic content
        bar_mid = bar_start + BEATS_PER_BAR / 2
        active_hc = None
        for c_start, c_end, hc in chord_timeline:
            if c_start <= bar_mid < c_end:
                active_hc = hc
                break
        if active_hc is None and chord_timeline:
            active_hc = chord_timeline[-1][2]

        event_id = active_hc.event_id if active_hc else 0

        # Harmonic voices from active chord
        if active_hc:
            h_pitches = [active_hc.alto.pitch, active_hc.tenor.pitch, active_hc.bass.pitch]
            h_vels = [active_hc.alto.velocity, active_hc.tenor.velocity, active_hc.bass.velocity]
        else:
            h_pitches = []
            h_vels = []

        # Melody notes overlapping this bar
        m_pitches = []        # sounding pitches only (for pitch_height + tonal)
        m_vels = []           # all velocities (rests have vel=0, correctly reduce loudness)
        m_durs = []           # overlap durations (for all notes)
        m_onsets = 0          # sounding onsets only

        for m_start, m_end, note in melody_timeline:
            if m_start >= bar_end or m_end <= bar_start:
                continue  # no overlap
            overlap = min(m_end, bar_end) - max(m_start, bar_start)
            if overlap > 0.001:
                m_vels.append(note.velocity)
                m_durs.append(overlap)
                # Only include sounding notes in pitch calculations
                if note.velocity > 0:
                    m_pitches.append(note.pitch)
                # Count onset only if sounding note starts in this bar
                if bar_start <= m_start < bar_end and note.velocity > 0:
                    m_onsets += 1

        # Harmonic onsets: count chord boundaries that fall in this bar
        # Each chord change produces 3 onsets (alto, tenor, bass)
        h_onsets = 0
        for c_start, c_end, hc in chord_timeline:
            if bar_start <= c_start < bar_end:
                h_onsets += 3

        # Pitches for spiral array: only sounding notes
        all_pitches = h_pitches + m_pitches
        if not all_pitches:
            all_pitches = [60 + arr.key.root]  # fallback: key root

        bars.append(BarWindow(
            bar_idx=bar_idx,
            start_beat=bar_start,
            end_beat=bar_end,
            event_id=event_id,
            melody_pitches=m_pitches,
            melody_velocities=m_vels,
            melody_durations=m_durs,
            melody_onsets=m_onsets,
            harmonic_onsets=h_onsets,
            harmonic_pitches=h_pitches,
            harmonic_velocities=h_vels,
            all_pitches=all_pitches,
        ))

    return bars


# --- Per-bar feature computation ---

def _grid_loudness(bars: List[BarWindow]) -> np.ndarray:
    """Duration-weighted mean velocity per bar, normalised to [0, 1]."""
    result = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        # Harmonic voices: each sounds for the full bar
        h_total = sum(bar.harmonic_velocities) * BEATS_PER_BAR
        h_dur = len(bar.harmonic_velocities) * BEATS_PER_BAR

        # Melody: duration-weighted
        m_total = sum(v * d for v, d in zip(bar.melody_velocities, bar.melody_durations))
        m_dur = sum(bar.melody_durations)

        total_weighted = h_total + m_total
        total_dur = h_dur + m_dur

        result[i] = (total_weighted / total_dur / 127.0) if total_dur > 0 else 0.0

    return np.clip(result, 0.0, 1.0)


def _grid_pitch_height(bars: List[BarWindow]) -> np.ndarray:
    """Duration-weighted mean pitch per bar, normalised to [0, 1].

    Uses only sounding melody pitches (rests excluded in bar grid).
    Harmonic voices always sound — they contribute to every bar.
    """
    MIDI_MIN, MIDI_MAX = 21.0, 108.0
    result = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        # Harmonic voices: each sounds for full bar
        h_total = sum(bar.harmonic_pitches) * BEATS_PER_BAR
        h_dur = len(bar.harmonic_pitches) * BEATS_PER_BAR

        # Melody: only sounding pitches (already filtered in bar grid)
        # Need to pair pitches with their durations — pitches list may be
        # shorter than durs list because rests were excluded from pitches
        # but kept in durs/vels. Use a separate accumulator.
        m_total = 0.0
        m_dur = 0.0
        for p in bar.melody_pitches:
            # Each sounding pitch contributes equally (simplified)
            m_total += p
            m_dur += 1.0

        if m_dur > 0:
            total_weighted = h_total + m_total * (BEATS_PER_BAR / max(1, len(bar.melody_velocities)))
            total_dur = h_dur + m_dur * (BEATS_PER_BAR / max(1, len(bar.melody_velocities)))
        else:
            total_weighted = h_total
            total_dur = h_dur

        mean_pitch = (total_weighted / total_dur) if total_dur > 0 else 65.0
        result[i] = (mean_pitch - MIDI_MIN) / (MIDI_MAX - MIDI_MIN)

    return np.clip(result, 0.0, 1.0)


def _grid_onset_frequency(bars: List[BarWindow]) -> np.ndarray:
    """Total note onsets per beat per bar, normalised to [0, 1].

    Counts all note attacks: melody onsets + harmonic voice onsets
    (3 per chord change within the bar).
    """
    MAX_ONSETS_PER_BEAT = 4.0
    result = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        n_onsets = bar.harmonic_onsets + bar.melody_onsets
        result[i] = n_onsets / (BEATS_PER_BAR * MAX_ONSETS_PER_BEAT)
    return np.clip(result, 0.0, 1.0)


def _grid_tonal_tension(
    bars: List[BarWindow],
    key_root: int,
    is_major: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spiral array tension on the fixed bar grid.

    Returns (cloud_diameter, cloud_momentum, tensile_strain) per bar.
    """
    pitch_lists = [bar.all_pitches for bar in bars]
    return calculate_tension_fast(pitch_lists, key_root, is_major)


def _feature_tension_curve(
    values: np.ndarray,
    beta: float = TENSION_TREND_BETA,
) -> np.ndarray:
    """Build a tension curve from absolute feature values using Farbood's
    slope-based model with trend amplification.

    1. Compute bar-to-bar slopes (first differences)
    2. If the slope continues in the same direction as the previous bar,
       multiply by beta (sustained trends amplify perceived tension)
    3. Cumulatively integrate to produce a tension curve

    Args:
        values: absolute feature values per bar, shape (n_bars,)
        beta:   trend amplification factor (Farbood: 5, our default: 3)

    Returns:
        Tension curve, shape (n_bars,)
    """
    n = len(values)
    if n < 2:
        return np.zeros(n)

    # Step 1: slopes (first differences)
    slopes = np.diff(values)  # length n-1

    # Step 2: trend amplification
    amplified = np.zeros(len(slopes))
    amplified[0] = slopes[0]
    for i in range(1, len(slopes)):
        if slopes[i] != 0 and np.sign(slopes[i]) == np.sign(slopes[i - 1]):
            amplified[i] = slopes[i] * beta
        else:
            amplified[i] = slopes[i]

    # Step 3: cumulative integration → tension curve
    # Prepend 0 so the curve has the same length as input
    tension = np.zeros(n)
    tension[1:] = np.cumsum(amplified)

    return tension


def _compute_bar_features(
    bars: List[BarWindow],
    arr: Arrangement,
) -> Dict[str, np.ndarray]:
    """Compute all Farbood features as absolute values per bar.

    Returns a dict of feature name → values array.
    """
    n_bars = len(bars)

    loudness     = _grid_loudness(bars)
    pitch_height = _grid_pitch_height(bars)
    onset_freq   = _grid_onset_frequency(bars)

    tempo_norm = np.full(n_bars, np.clip(
        (arr.tempo_bpm - TEMPO_MIN_BPM) / (TEMPO_MAX_BPM - TEMPO_MIN_BPM), 0.0, 1.0))

    cd, cm, ts = _grid_tonal_tension(bars, arr.key.root, arr.key.is_major)
    tonal = (TENSION_WEIGHT_STRAIN   * ts
             + TENSION_WEIGHT_DIAMETER * cd
             + TENSION_WEIGHT_MOMENTUM * cm)

    return {
        "loudness": loudness,
        "pitch":    pitch_height,
        "tempo":    tempo_norm,
        "onset":    onset_freq,
        "tonal":    tonal,
    }


# Feature weights from Farbood (2012) Experiment 1
_FEATURE_WEIGHTS = {
    "loudness": TENSION_FARBOOD_LOUDNESS,
    "pitch":    TENSION_FARBOOD_PITCH,
    "tempo":    TENSION_FARBOOD_TEMPO,
    "onset":    TENSION_FARBOOD_ONSET,
    "tonal":    TENSION_FARBOOD_TONAL,
}


def _compute_bar_tension(
    bars: List[BarWindow],
    arr: Arrangement,
) -> np.ndarray:
    """Farbood (2012) composite tension on fixed bar grid.

    Uses slope-based model: computes per-feature tension curves from
    slopes with trend amplification, then combines with Farbood weights.

    This function is kept for backward compatibility (notebook plots).
    """
    features = _compute_bar_features(bars, arr)

    # Normalise weights to sum to 1
    total_w = sum(_FEATURE_WEIGHTS.values())

    composite = np.zeros(len(bars))
    for name, values in features.items():
        w = _FEATURE_WEIGHTS.get(name, 1.0) / total_w
        curve = _feature_tension_curve(values)
        composite += w * curve

    return composite


def _bars_to_event_tension(
    tension: np.ndarray,
    bars: List[BarWindow],
) -> Dict[int, float]:
    """Aggregate bar tension to event level by equal-weight averaging."""
    event_sums: Dict[int, float] = {}
    event_counts: Dict[int, int] = {}

    for i, bar in enumerate(bars):
        eid = bar.event_id
        event_sums[eid] = event_sums.get(eid, 0.0) + tension[i]
        event_counts[eid] = event_counts.get(eid, 0) + 1

    return {
        eid: event_sums[eid] / event_counts[eid]
        for eid in event_sums if event_counts[eid] > 0
    }


def _slope_correlation(
    tension_vals: np.ndarray,
    suspense_vals: np.ndarray,
    beta: float = TENSION_TREND_BETA,
) -> float:
    """Compute correlation between event-to-event slopes.

    For both tension and suspense, computes consecutive differences
    (how much each changes from one event to the next). If the slope
    continues in the same direction as the previous step, amplifies
    by beta — rewarding sustained aligned transitions.

    Returns (1 - r) / 2 where r is Pearson on the amplified slopes.
    """
    if len(tension_vals) < 3:
        return 0.5

    # Event-to-event differences
    t_slopes = np.diff(tension_vals)
    s_slopes = np.diff(suspense_vals)

    # Trend amplification on tension slopes
    amplified = np.zeros(len(t_slopes))
    amplified[0] = t_slopes[0]
    for i in range(1, len(t_slopes)):
        if t_slopes[i] != 0 and np.sign(t_slopes[i]) == np.sign(t_slopes[i - 1]):
            amplified[i] = t_slopes[i] * beta
        else:
            amplified[i] = t_slopes[i]

    # Same for suspense slopes
    s_amplified = np.zeros(len(s_slopes))
    s_amplified[0] = s_slopes[0]
    for i in range(1, len(s_slopes)):
        if s_slopes[i] != 0 and np.sign(s_slopes[i]) == np.sign(s_slopes[i - 1]):
            s_amplified[i] = s_slopes[i] * beta
        else:
            s_amplified[i] = s_slopes[i]

    if amplified.std() < 1e-9 or s_amplified.std() < 1e-9:
        return 0.5

    r, _ = pearsonr(amplified, s_amplified)
    r = float(np.clip(r, -1.0, 1.0))
    return (1.0 - r) / 2.0


def tension_misalignment(
    arrangements: Dict[int, Arrangement],
    narrative_data: dict,
    motifs: List[Motif],
) -> float:
    """Combined absolute + slope tension-suspense misalignment.

    For each Farbood feature independently:
        1. Compute absolute values per bar
        2. Average per event → one tension level per event
        3. Absolute score: correlate levels with suspense
        4. Slope score: correlate event-to-event transitions with
           suspense transitions (with trend amplification)
        5. Combined: α × absolute + (1-α) × slope

    Features whose event-level values have std < TENSION_MIN_FEATURE_STD
    are skipped (too flat for a reliable correlation).

    Returns the mean of combined scores across included features.
    Lower is better (0 = perfect alignment).
    """
    alpha = TENSION_ABSOLUTE_WEIGHT
    suspense_map = _event_suspense_map(narrative_data)

    # Collect per-feature absolute values across all sections
    feature_event_values: Dict[str, Dict[int, List[float]]] = {}

    for arr in arrangements.values():
        if not arr.harmonic_chords:
            continue

        bars = _build_bar_grid(arr)
        if not bars:
            continue

        features = _compute_bar_features(bars, arr)

        for name, values in features.items():
            event_avgs = _bars_to_event_tension(values, bars)

            if name not in feature_event_values:
                feature_event_values[name] = {}
            for eid, v in event_avgs.items():
                feature_event_values[name].setdefault(eid, []).append(v)

    if not feature_event_values:
        return 0.5

    # Average across sections for events that span multiple sections
    feature_event_means: Dict[str, Dict[int, float]] = {}
    for name, eid_dict in feature_event_values.items():
        feature_event_means[name] = {
            eid: np.mean(vals) for eid, vals in eid_dict.items()
        }

    # Compute per-feature combined scores
    feature_scores: List[float] = []

    for name in _FEATURE_WEIGHTS:
        event_means = feature_event_means.get(name, {})

        # Pair with suspense (sorted by event_id for consistent ordering)
        eids = sorted(set(event_means.keys()) & set(suspense_map.keys()))
        if len(eids) < 3:
            continue

        t = np.array([event_means[eid] for eid in eids])
        s = np.array([suspense_map[eid] for eid in eids])

        # Variance safety check
        if t.std() < TENSION_MIN_FEATURE_STD or s.std() < 1e-9:
            continue

        # Absolute correlation
        r_abs, _ = pearsonr(t, s)
        r_abs = float(np.clip(r_abs, -1.0, 1.0))
        abs_score = (1.0 - r_abs) / 2.0

        # Slope correlation (needs at least 4 events for 3 slopes)
        if len(eids) >= 4:
            slope_score = _slope_correlation(t, s)
        else:
            slope_score = abs_score  # fallback: use absolute only

        combined = alpha * abs_score + (1.0 - alpha) * slope_score
        feature_scores.append(combined)

    if not feature_scores:
        return 0.5

    return sum(feature_scores) / len(feature_scores)