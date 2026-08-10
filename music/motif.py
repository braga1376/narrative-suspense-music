"""
motif.py

Musical motif generation for narrative characters.

Generates a two-bar MIDI motif per character whose pitch interval
structure is derived from narrative valence and whose rhythmic
character is derived from average visual activity.

Valence  → Gaussian distribution over dissonance-ordered intervals
Activity → Predominant note value (sparse ↔ dense)

Two generation modes:
    chromatic: intervals sampled freely from all 12 semitones
    tonal:     at each step only intervals landing on scale tones
               are available, weighted by the same Gaussian

All motifs are generated in C (root = MIDI 60) and transposed
to the appropriate key during composition.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

from .structures import Note, Motif


# ---------------------------------------------------------------------------
# Interval set ordered by increasing dissonance
# Standard music theory ordering (consonant → dissonant)
# ---------------------------------------------------------------------------
INTERVALS_BY_DISSONANCE = [
    0,   # P1  — unison
    7,   # P5  — perfect fifth
    5,   # P4  — perfect fourth
    4,   # M3  — major third
    9,   # M6  — major sixth
    3,   # m3  — minor third
    8,   # m6  — minor sixth
    2,   # M2  — major second
    10,  # m7  — minor seventh
    1,   # m2  — minor second
    11,  # M7  — major seventh
    6,   # TT  — tritone
]

# ---------------------------------------------------------------------------
# Common scales as pitch class sets (relative to C = 0)
# ---------------------------------------------------------------------------
SCALES = {
    "major":            {0, 2, 4, 5, 7, 9, 11},
    "natural_minor":    {0, 2, 3, 5, 7, 8, 10},
    "harmonic_minor":   {0, 2, 3, 5, 7, 8, 11},
    "dorian":           {0, 2, 3, 5, 7, 9, 10},
    "phrygian":         {0, 1, 3, 5, 7, 8, 10},
    "pentatonic_major": {0, 2, 4, 7, 9},
    "pentatonic_minor": {0, 3, 5, 7, 10},
}

DEFAULT_SCALE = SCALES["major"]

# ---------------------------------------------------------------------------
# Rhythm: note durations in beats mapped from activity level
# ---------------------------------------------------------------------------
ACTIVITY_RHYTHM_POOLS = {
    0.1: [(2.0, 0.3), (1.0, 0.5), (0.5, 0.2)],
    0.3: [(1.0, 0.5), (0.5, 0.35), (2.0, 0.15)],
    0.5: [(1.0, 0.35), (0.5, 0.45), (0.25, 0.2)],
    0.7: [(0.5, 0.45), (0.25, 0.35), (1.0, 0.2)],
    0.9: [(0.25, 0.5), (0.5, 0.35), (0.125, 0.15)],
}

MOTIF_DURATION_BEATS = 8.0
MOTIF_ROOT = 60
DEFAULT_VELOCITY = 80

# ---------------------------------------------------------------------------
# Weight precomputation
# ---------------------------------------------------------------------------

def _compute_interval_weights(valence: float, sigma: float) -> np.ndarray:
    """
    Precompute Gaussian weights over INTERVALS_BY_DISSONANCE.

    Called once per motif generation — weights are fixed for a given
    valence and sigma, so there is no need to recompute per note.

    Args:
        valence: character valence (-10 to +10)
        sigma:   distribution spread

    Returns:
        normalised weight array of length len(INTERVALS_BY_DISSONANCE)
    """
    n = len(INTERVALS_BY_DISSONANCE)
    valence_norm = np.clip(valence / 10.0, -1.0, 1.0)
    center = (1 - valence_norm) / 2 * (n - 1)

    weights = np.exp(-0.5 * ((np.arange(n) - center) / sigma) ** 2)
    weights /= weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Rhythm helpers
# ---------------------------------------------------------------------------

def _get_rhythm_pool(avg_activity: float) -> list:
    available = list(ACTIVITY_RHYTHM_POOLS.keys())
    closest = min(available, key=lambda x: abs(x - avg_activity))
    return ACTIVITY_RHYTHM_POOLS[closest]


def _sample_duration(rhythm_pool: list) -> float:
    durations, weights = zip(*rhythm_pool)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return float(np.random.choice(durations, p=weights))


def _generate_rhythm(avg_activity: float) -> List[float]:
    """
    Generate note durations filling exactly MOTIF_DURATION_BEATS beats.
    """
    rhythm_pool = _get_rhythm_pool(avg_activity)
    durations = []
    total = 0.0

    while total < MOTIF_DURATION_BEATS:
        remaining = MOTIF_DURATION_BEATS - total
        duration = _sample_duration(rhythm_pool)

        if duration > remaining:
            durations.append(remaining)
            break

        durations.append(duration)
        total += duration

    return durations


# ---------------------------------------------------------------------------
# Chromatic pitch generation
# ---------------------------------------------------------------------------

def _generate_pitches(
    n_notes: int,
    interval_weights: np.ndarray,
    root: int = MOTIF_ROOT,
) -> List[int]:
    """
    Generate pitches by sampling intervals freely from all 12 semitones.

    Args:
        n_notes:          number of notes to generate
        interval_weights: precomputed Gaussian weights over
                          INTERVALS_BY_DISSONANCE — call
                          _compute_interval_weights() once before
                          generating pitches
        root:             starting MIDI pitch

    Returns:
        list of MIDI pitch values
    """
    pitches = []
    current = root
    lower_bound = root - 12
    upper_bound = root + 12

    for _ in range(n_notes):
        idx = np.random.choice(len(INTERVALS_BY_DISSONANCE),
                               p=interval_weights)
        interval = INTERVALS_BY_DISSONANCE[idx]

        # Bias direction by position within range
        position = (current - lower_bound) / (upper_bound - lower_bound)
        direction = 1 if random.random() < (1.0 - position) else -1

        current = int(np.clip(current + direction * interval,
                               lower_bound, upper_bound))
        pitches.append(current)

    return pitches


# ---------------------------------------------------------------------------
# Tonal pitch generation
# ---------------------------------------------------------------------------

def _generate_pitches_tonal(
    n_notes: int,
    interval_weights: np.ndarray,
    scale_pitches: Set[int],
    root: int = MOTIF_ROOT,
) -> List[int]:
    """
    Generate pitches where at each step only intervals landing on
    scale tones are available, weighted by precomputed Gaussian weights.

    Args:
        n_notes:          number of notes to generate
        interval_weights: precomputed Gaussian weights over
                          INTERVALS_BY_DISSONANCE — call
                          _compute_interval_weights() once before
                          generating pitches
        scale_pitches:    set of pitch classes (0-11) in the target scale
        root:             starting MIDI pitch (should be in scale)

    Returns:
        list of MIDI pitch values
    """
    pitches = []
    current = root
    lower_bound = root - 12
    upper_bound = root + 12

    for _ in range(n_notes):
        # Collect all (candidate_pitch, interval_index) pairs that
        # land on a scale tone within range
        candidates = [
            (current + direction * interval, idx)
            for idx, interval in enumerate(INTERVALS_BY_DISSONANCE)
            for direction in [1, -1]
            if (lower_bound
                <= current + direction * interval
                <= upper_bound
                and (current + direction * interval) % 12 in scale_pitches)
        ]

        if not candidates:
            pitches.append(current)
            continue

        # Reuse precomputed weights — look up by interval index
        weights = np.array([interval_weights[idx] for _, idx in candidates])
        weights /= weights.sum()

        chosen_pitch = candidates[
            np.random.choice(len(candidates), p=weights)
        ][0]
        current = chosen_pitch
        pitches.append(current)

    return pitches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_motif(
    character_id: int,
    character_name: str,
    valence: float,
    avg_activity: float,
    sigma: float = 3.0,
    tonal: bool = True,
    scale_pitches: Optional[Set[int]] = None,
    seed: Optional[int] = None,
) -> Motif:
    """
    Generate a two-bar MIDI motif for a narrative character.

    Gaussian weights over the dissonance-ordered interval set are
    computed once from valence and sigma, then reused for every
    note in the motif.

    Args:
        character_id:    character_id from narrative data
        character_name:  character name for reference
        valence:         narrative valence from compute_character_valence()
        avg_activity:    median activity from compute_avg_activity()
        sigma:           interval distribution spread (default 3.0)
        tonal:           if True, restrict pitches to scale tones
        scale_pitches:   pitch class set for tonal mode
                         defaults to C major if tonal=True and not provided
        seed:            random seed for reproducibility (optional)

    Returns:
        Motif instance with MIDI notes
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Precompute weights once for this motif
    interval_weights = _compute_interval_weights(valence, sigma)

    # Generate rhythm first to know how many notes we need
    durations = _generate_rhythm(avg_activity)
    n_notes = len(durations)

    # Generate pitches
    if tonal:
        scale = scale_pitches if scale_pitches is not None else DEFAULT_SCALE
        pitches = _generate_pitches_tonal(n_notes, interval_weights, scale)
    else:
        pitches = _generate_pitches(n_notes, interval_weights)

    notes = [
        Note(pitch=pitch, duration=duration, velocity=DEFAULT_VELOCITY)
        for pitch, duration in zip(pitches, durations)
    ]

    return Motif(
        notes=notes,
        character_id=character_id,
        character_name=character_name,
        valence=valence,
        avg_activity=avg_activity,
        tonal=tonal,
        key=MOTIF_ROOT,
    )


def generate_all_motifs(
    narrative_data: dict,
    character_profiles: list,
    sigma: float = 3.0,
    tonal: bool = True,
    scale_pitches: Optional[Set[int]] = None,
    seed: Optional[int] = None,
) -> List[Motif]:
    """
    Generate motifs for all characters in narrative_data.

    Args:
        narrative_data:     full narrative dict from extract_narrative()
        character_profiles: list of dicts from compute_character_profiles()
        sigma:              interval distribution spread
        tonal:              if True, restrict pitches to scale tones
        scale_pitches:      pitch class set for tonal mode
        seed:               random seed for reproducibility (optional)

    Returns:
        list of Motif instances, one per character
    """
    motifs = []

    for i, profile in enumerate(character_profiles):
        character_seed = (seed + i) if seed is not None else None

        motif = generate_motif(
            character_id=profile["character_id"],
            character_name=profile["name"],
            valence=profile["valence"],
            avg_activity=profile["avg_activity"],
            sigma=sigma,
            tonal=tonal,
            scale_pitches=scale_pitches,
            seed=character_seed,
        )
        motifs.append(motif)

    return motifs