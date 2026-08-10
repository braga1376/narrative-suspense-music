"""
music/operators.py

Genetic operators for NSGA-II melody evolution.

Mutation operators (modify one individual):
    mutate_pitch      — shift one note's pitch by ±1-4 semitones
    mutate_rhythm     — redistribute duration between two adjacent notes
    mutate_velocity   — shift all notes in one event by the same amount
    mutate_cell_type  — change one bar's rendering interpretation
    mutate_density    — split/merge notes across all bars in one event

Crossover operators (combine two individuals):
    crossover_bars    — swap bars between two individuals
    crossover_notes   — swap note tails within one bar

All operators maintain the invariant:
    len(pitches) == len(durations) == len(velocities)
    sum(durations) == bar beat budget (preserved exactly)
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional

from .genome import BarGenes, MelodyGenes, CELL_TYPES, REST_PITCH
from .structures import Key
from .voicing import VOICE_RANGES, Voice

MELODY_LOW, MELODY_HIGH = VOICE_RANGES[Voice.MELODY]
MIN_NOTE_DURATION = 0.25
MIN_VELOCITY = 50
MAX_VELOCITY = 120

# Standard rhythmic subdivisions (in beats)
RHYTHMIC_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def _quantize_duration(dur: float) -> float:
    """Snap a duration to the nearest standard rhythmic value."""
    return min(RHYTHMIC_GRID, key=lambda g: abs(g - dur))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap_to_scale(pitch: int, key: Key) -> int:
    """Snap a pitch to the nearest scale tone."""
    pcs = set(key.get_scale_degree().tolist())
    if pitch % 12 in pcs:
        return pitch
    for offset in [1, -1, 2, -2]:
        c = pitch + offset
        if c % 12 in pcs and MELODY_LOW <= c <= MELODY_HIGH:
            return c
    return pitch


def _clamp_pitch(pitch: int) -> int:
    return max(MELODY_LOW, min(MELODY_HIGH, pitch))


def _clamp_velocity(velocity: int) -> int:
    return max(MIN_VELOCITY, min(MAX_VELOCITY, velocity))


def _random_bar(genes: List[MelodyGenes]) -> Optional[Tuple[int, int]]:
    """Pick a random (event_idx, bar_idx) from a list of MelodyGenes.
    Returns None if no bars exist."""
    options = [
        (e_idx, b_idx)
        for e_idx, mg in enumerate(genes)
        for b_idx in range(mg.n_bars)
    ]
    return random.choice(options) if options else None


def _random_note(bar: BarGenes) -> Optional[int]:
    """Pick a random note index from a BarGenes.
    Returns None if bar is empty."""
    if bar.n_notes == 0:
        return None
    return random.randint(0, bar.n_notes - 1)


# ---------------------------------------------------------------------------
# Mutation: pitch
# ---------------------------------------------------------------------------

def mutate_pitch(
    genes: List[MelodyGenes],
    key: Key,
) -> List[MelodyGenes]:
    """Shift one note's pitch by ±1 to ±4 semitones, snapped to scale.

    This is the primary creative operator — it changes what the
    listener hears at one specific moment.
    """
    loc = _random_bar(genes)
    if loc is None:
        return genes

    e_idx, b_idx = loc
    bar = genes[e_idx].bar_genes[b_idx]
    n_idx = _random_note(bar)
    if n_idx is None:
        return genes

    shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    new_pitch = _clamp_pitch(bar.pitches[n_idx] + shift)
    new_pitch = _snap_to_scale(new_pitch, key)
    bar.pitches[n_idx] = new_pitch

    return genes


# ---------------------------------------------------------------------------
# Mutation: rhythm
# ---------------------------------------------------------------------------

def mutate_rhythm(
    genes: List[MelodyGenes],
) -> List[MelodyGenes]:
    """Redistribute duration between adjacent notes across all bars in one event.

    For each bar (50% chance), picks two adjacent notes and swaps their
    durations to a different grid-aligned pair that sums to the same total.
    """
    if not genes:
        return genes

    e_idx = random.randint(0, len(genes) - 1)
    mg = genes[e_idx]
    if not mg.bar_genes:
        return genes

    for bg in mg.bar_genes:
        if random.random() >= 0.5:
            continue
        if bg.n_notes < 2:
            continue

        # Pick first of two adjacent notes
        i = random.randint(0, bg.n_notes - 2)
        j = i + 1

        total = bg.durations[i] + bg.durations[j]

        # Find all grid pairs that sum to the same total
        candidates = []
        for g1 in RHYTHMIC_GRID:
            g2 = total - g1
            if g2 >= MIN_NOTE_DURATION:
                g2_snap = _quantize_duration(g2)
                if abs(g2_snap - g2) < 0.01:
                    candidates.append((g1, g2_snap))

        if not candidates:
            continue

        new_a, new_b = random.choice(candidates)
        bg.durations[i] = new_a
        bg.durations[j] = new_b

    return genes


# ---------------------------------------------------------------------------
# Mutation: velocity
# ---------------------------------------------------------------------------

def mutate_velocity(
    genes: List[MelodyGenes],
) -> List[MelodyGenes]:
    """Shift all sounding notes in one event by the same velocity amount.

    Preserves the internal dynamic shape (crescendos, accents) while
    raising or lowering the overall level. This lets the GA match
    event dynamics to narrative suspense without creating jagged
    per-note jumps.
    """
    if not genes:
        return genes

    e_idx = random.randint(0, len(genes) - 1)
    mg = genes[e_idx]
    if not mg.bar_genes:
        return genes

    nudge = random.choice([-10, -5, -3, 3, 5, 10])

    for bg in mg.bar_genes:
        for i in range(len(bg.velocities)):
            if bg.pitches[i] != REST_PITCH:
                bg.velocities[i] = _clamp_velocity(bg.velocities[i] + nudge)

    return genes


def mutate_register(
    genes: List[MelodyGenes],
    key: Key,
) -> List[MelodyGenes]:
    """Shift all sounding notes in one event by the same pitch interval.

    Preserves the internal melodic shape (intervals between notes)
    while raising or lowering the overall register. Each note is
    snapped to the nearest scale tone after shifting.
    """
    if not genes:
        return genes

    e_idx = random.randint(0, len(genes) - 1)
    mg = genes[e_idx]
    if not mg.bar_genes:
        return genes

    shift = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])

    for bg in mg.bar_genes:
        for i in range(len(bg.pitches)):
            if bg.pitches[i] != REST_PITCH:
                new_p = _clamp_pitch(bg.pitches[i] + shift)
                new_p = _snap_to_scale(new_p, key)
                bg.pitches[i] = new_p

    return genes


# ---------------------------------------------------------------------------
# Mutation: cell type
# ---------------------------------------------------------------------------

def mutate_cell_type(
    genes: List[MelodyGenes],
) -> List[MelodyGenes]:
    """Change one bar's cell_type to a different type."""
    loc = _random_bar(genes)
    if loc is None:
        return genes

    e_idx, b_idx = loc
    bar = genes[e_idx].bar_genes[b_idx]
    current = bar.cell_type
    alternatives = [ct for ct in CELL_TYPES if ct != current]
    bar.cell_type = random.choice(alternatives)

    return genes


# ---------------------------------------------------------------------------
# Mutation: split note
# ---------------------------------------------------------------------------

def mutate_split(
    genes: List[MelodyGenes],
) -> List[MelodyGenes]:
    """Split one note into two notes at the same pitch.

    The original duration is divided (roughly in half, with some
    randomness). Creates rhythmic variety and gives future mutations
    more material to work with.

    Only splits notes longer than 2 * MIN_NOTE_DURATION.
    """
    loc = _random_bar(genes)
    if loc is None:
        return genes

    e_idx, b_idx = loc
    bar = genes[e_idx].bar_genes[b_idx]

    # Find notes long enough to split
    splittable = [i for i in range(bar.n_notes)
                  if bar.durations[i] >= 2 * MIN_NOTE_DURATION]
    if not splittable:
        return genes

    i = random.choice(splittable)
    original_dur = bar.durations[i]

    # Find grid pairs that sum to original duration
    candidates = []
    for g1 in RHYTHMIC_GRID:
        if g1 >= original_dur:
            continue
        g2 = original_dur - g1
        if g2 >= MIN_NOTE_DURATION:
            g2_snap = _quantize_duration(g2)
            if abs(g2_snap - g2) < 0.01 and g1 >= MIN_NOTE_DURATION:
                candidates.append((g1, g2_snap))

    if not candidates:
        return genes

    dur_a, dur_b = random.choice(candidates)

    # Ensure both parts are valid
    if dur_a < MIN_NOTE_DURATION or dur_b < MIN_NOTE_DURATION:
        return genes

    # Insert the split
    bar.pitches.insert(i + 1, bar.pitches[i])
    bar.durations[i] = dur_a
    bar.durations.insert(i + 1, dur_b)
    bar.velocities.insert(i + 1, bar.velocities[i])

    return genes


# ---------------------------------------------------------------------------
# Mutation: rest (toggle note ↔ rest)
# ---------------------------------------------------------------------------

def mutate_rest(
    genes: List[MelodyGenes],
    key: Key,
) -> List[MelodyGenes]:
    """Toggle one note between sounding and rest.

    If the selected note is sounding (pitch > 0), convert to rest (pitch = 0).
    If it's already a rest, convert to a nearby scale tone.

    Rests allow the melody to breathe — essential for musical phrasing.
    """
    loc = _random_bar(genes)
    if loc is None:
        return genes

    e_idx, b_idx = loc
    bar = genes[e_idx].bar_genes[b_idx]
    n_idx = _random_note(bar)
    if n_idx is None:
        return genes

    if bar.pitches[n_idx] == REST_PITCH:
        # Rest → sounding: pick a scale tone near the surrounding pitches
        neighbors = []
        if n_idx > 0 and bar.pitches[n_idx - 1] != REST_PITCH:
            neighbors.append(bar.pitches[n_idx - 1])
        if n_idx < bar.n_notes - 1 and bar.pitches[n_idx + 1] != REST_PITCH:
            neighbors.append(bar.pitches[n_idx + 1])
        if neighbors:
            base = random.choice(neighbors)
        else:
            base = 65  # fallback: F4
        bar.pitches[n_idx] = _snap_to_scale(_clamp_pitch(base + random.choice([-2,-1,0,1,2])), key)
    else:
        # Sounding → rest
        bar.pitches[n_idx] = REST_PITCH

    return genes


# ---------------------------------------------------------------------------
# Mutation: merge notes
# ---------------------------------------------------------------------------

def mutate_merge(
    genes: List[MelodyGenes],
) -> List[MelodyGenes]:
    """Merge two adjacent notes into one.

    The merged note takes the pitch of the first note and the
    combined duration of both.  Velocity is the average.

    Only merges if the bar has at least 2 notes.
    """
    loc = _random_bar(genes)
    if loc is None:
        return genes

    e_idx, b_idx = loc
    bar = genes[e_idx].bar_genes[b_idx]
    if bar.n_notes < 2:
        return genes

    i = random.randint(0, bar.n_notes - 2)

    # Merge: keep first pitch, sum durations, average velocity
    merged_dur = bar.durations[i] + bar.durations[i + 1]
    merged_dur = _quantize_duration(merged_dur)
    merged_vel = round((bar.velocities[i] + bar.velocities[i + 1]) / 2)

    bar.durations[i] = merged_dur
    bar.velocities[i] = _clamp_velocity(merged_vel)

    # Remove the second note
    bar.pitches.pop(i + 1)
    bar.durations.pop(i + 1)
    bar.velocities.pop(i + 1)

    return genes


# ---------------------------------------------------------------------------
# Crossover: bar-level
# ---------------------------------------------------------------------------

def crossover_bars(
    parent_a: List[MelodyGenes],
    parent_b: List[MelodyGenes],
) -> Tuple[List[MelodyGenes], List[MelodyGenes]]:
    """Swap bars between two individuals at a random crossover point.

    Operates across the flat list of all bars in the section.
    Bars before the crossover point come from parent A,
    bars after come from parent B (and vice versa for child B).

    Both parents must have the same number of events with the same
    number of bars per event. If they don't match, returns copies
    of the parents unchanged.
    """
    # Verify structure matches
    if len(parent_a) != len(parent_b):
        return _copy_genes(parent_a), _copy_genes(parent_b)

    for mg_a, mg_b in zip(parent_a, parent_b):
        if mg_a.n_bars != mg_b.n_bars:
            return _copy_genes(parent_a), _copy_genes(parent_b)

    # Build flat bar index list: [(event_idx, bar_idx), ...]
    flat = [(e, b) for e, mg in enumerate(parent_a) for b in range(mg.n_bars)]
    if len(flat) <= 1:
        return _copy_genes(parent_a), _copy_genes(parent_b)

    # Pick crossover point
    cut = random.randint(1, len(flat) - 1)

    child_a = _copy_genes(parent_a)
    child_b = _copy_genes(parent_b)

    # Swap bars after the cut point
    for e_idx, b_idx in flat[cut:]:
        child_a[e_idx].bar_genes[b_idx] = parent_b[e_idx].bar_genes[b_idx].copy()
        child_b[e_idx].bar_genes[b_idx] = parent_a[e_idx].bar_genes[b_idx].copy()

    return child_a, child_b


# ---------------------------------------------------------------------------
# Crossover: within-bar (note-level)
# ---------------------------------------------------------------------------

def crossover_notes(
    parent_a: List[MelodyGenes],
    parent_b: List[MelodyGenes],
) -> Tuple[List[MelodyGenes], List[MelodyGenes]]:
    """Single-point crossover within one randomly selected bar.

    Picks a bar that exists in both parents with the same number of
    notes, splits the pitch/duration/velocity lists at a random point,
    and swaps the tails.

    Duration sums may differ slightly after the swap, so the last note's
    duration is adjusted to preserve the bar's total.
    """
    if len(parent_a) != len(parent_b):
        return _copy_genes(parent_a), _copy_genes(parent_b)

    # Find bars with matching note counts
    candidates = []
    for e_idx in range(len(parent_a)):
        mg_a, mg_b = parent_a[e_idx], parent_b[e_idx]
        for b_idx in range(min(mg_a.n_bars, mg_b.n_bars)):
            ba = mg_a.bar_genes[b_idx]
            bb = mg_b.bar_genes[b_idx]
            if ba.n_notes == bb.n_notes and ba.n_notes >= 2:
                candidates.append((e_idx, b_idx))

    if not candidates:
        return _copy_genes(parent_a), _copy_genes(parent_b)

    e_idx, b_idx = random.choice(candidates)
    child_a = _copy_genes(parent_a)
    child_b = _copy_genes(parent_b)

    ba = child_a[e_idx].bar_genes[b_idx]
    bb = child_b[e_idx].bar_genes[b_idx]
    n = ba.n_notes

    # Crossover point
    cut = random.randint(1, n - 1)

    # Save original totals
    total_a = ba.total_duration
    total_b = bb.total_duration

    # Swap tails
    ba.pitches[cut:], bb.pitches[cut:] = bb.pitches[cut:], ba.pitches[cut:]
    ba.durations[cut:], bb.durations[cut:] = bb.durations[cut:], ba.durations[cut:]
    ba.velocities[cut:], bb.velocities[cut:] = bb.velocities[cut:], ba.velocities[cut:]

    # Fix durations: adjust last note to preserve bar total
    _fix_bar_duration(ba, total_a)
    _fix_bar_duration(bb, total_b)

    return child_a, child_b


# ---------------------------------------------------------------------------
# Duration fix helper
# ---------------------------------------------------------------------------

def _fix_bar_duration(bar: BarGenes, target_total: float) -> None:
    """Adjust the last note's duration so the bar sums to target_total."""
    if bar.n_notes == 0:
        return
    current = sum(bar.durations[:-1])
    last_dur = target_total - current
    bar.durations[-1] = max(MIN_NOTE_DURATION, round(last_dur, 6))


# ---------------------------------------------------------------------------
# Copy helper
# ---------------------------------------------------------------------------

def _copy_genes(genes: List[MelodyGenes]) -> List[MelodyGenes]:
    """Deep copy a list of MelodyGenes."""
    return [mg.copy() for mg in genes]


# ---------------------------------------------------------------------------
# Mutation dispatch
# ---------------------------------------------------------------------------

# Per-note probabilities (applied independently to each note in selected bars)
NOTE_PITCH_PROB    = 0.15   # chance each note gets pitch-shifted
NOTE_VELOCITY_PROB = 0.0   # chance each note gets velocity-nudged (small texture)
NOTE_REST_PROB     = 0.0   # chance each note gets toggled to/from rest

# Bar-level structural operation probabilities
BAR_OP_PROBS = {
    "rhythm":          0.25,
    # "cell_type":       0.10,
    "event_velocity":  0.25,   # shift all notes in event by same velocity
    "event_density":   0.25,   # split/merge across all bars in event
    "event_register":  0.25,   # shift all notes in event by same interval
    # "none":            0.10,   # no structural change this call
}


def _split_longest_note(bar: BarGenes, key: Key) -> bool:
    """Split the longest note in a bar into two grid-aligned halves.

    Returns True if a split was performed.
    """
    # Find longest splittable note (>= 1.0 beat, not a rest)
    best_i = -1
    best_dur = 0.0
    for i in range(bar.n_notes):
        if bar.pitches[i] != REST_PITCH and bar.durations[i] >= 1.0:
            if bar.durations[i] > best_dur:
                best_dur = bar.durations[i]
                best_i = i

    if best_i < 0:
        return False

    original_dur = bar.durations[best_i]

    # Find grid-aligned split
    candidates = []
    for g1 in RHYTHMIC_GRID:
        if g1 >= original_dur:
            continue
        g2 = original_dur - g1
        g2_snap = _quantize_duration(g2)
        if abs(g2_snap - g2) < 0.01 and g1 >= MIN_NOTE_DURATION and g2_snap >= MIN_NOTE_DURATION:
            candidates.append((g1, g2_snap))

    if not candidates:
        return False

    dur_a, dur_b = random.choice(candidates)

    # New note: neighbor pitch or same pitch
    original_pitch = bar.pitches[best_i]
    if random.random() < 0.5:
        new_pitch = _snap_to_scale(
            _clamp_pitch(original_pitch + random.choice([-2, -1, 1, 2])), key)
    else:
        new_pitch = original_pitch

    bar.pitches.insert(best_i + 1, new_pitch)
    bar.durations[best_i] = dur_a
    bar.durations.insert(best_i + 1, dur_b)
    bar.velocities.insert(best_i + 1, bar.velocities[best_i])

    return True


def _merge_shortest_pair(bar: BarGenes) -> bool:
    """Merge the two shortest adjacent sounding notes in a bar.

    Returns True if a merge was performed.
    """
    if bar.n_notes < 3:
        return False

    # Find adjacent pair of sounding notes with smallest combined duration
    best_i = -1
    best_combined = float('inf')
    for i in range(bar.n_notes - 1):
        if bar.pitches[i] == REST_PITCH or bar.pitches[i + 1] == REST_PITCH:
            continue
        combined = bar.durations[i] + bar.durations[i + 1]
        if combined < best_combined:
            best_combined = combined
            best_i = i

    if best_i < 0:
        return False

    # Keep exact sum — no quantization here to preserve bar total
    merged_dur = bar.durations[best_i] + bar.durations[best_i + 1]
    merged_vel = round((bar.velocities[best_i] + bar.velocities[best_i + 1]) / 2)

    bar.durations[best_i] = merged_dur
    bar.velocities[best_i] = _clamp_velocity(merged_vel)

    bar.pitches.pop(best_i + 1)
    bar.durations.pop(best_i + 1)
    bar.velocities.pop(best_i + 1)

    return True


def mutate_density(
    genes: List[MelodyGenes],
    key: Key,
) -> List[MelodyGenes]:
    """Change note density across all bars in one event.

    Randomly picks 'denser' or 'sparser', then applies split or merge
    to each bar with 50% probability. This creates a meaningful change
    in onset frequency at the event level.
    """
    if not genes:
        return genes

    e_idx = random.randint(0, len(genes) - 1)
    mg = genes[e_idx]
    if not mg.bar_genes:
        return genes

    direction = random.choice(["denser", "sparser"])

    for bg in mg.bar_genes:
        if random.random() < 0.5:
            continue  # skip this bar

        if direction == "denser":
            _split_longest_note(bg, key)
        else:
            _merge_shortest_pair(bg)

    return genes


def _mutate_bar_notes(bar: BarGenes, key: Key) -> None:
    """Apply per-note mutations across all notes in one bar.

    Each note independently has a chance of pitch, velocity, or rest
    mutation. Modifies bar in place.
    """
    for i in range(bar.n_notes):
        # Pitch mutation
        if random.random() < NOTE_PITCH_PROB:
            if bar.pitches[i] == REST_PITCH:
                continue  # don't pitch-shift a rest
            shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            new_p = _clamp_pitch(bar.pitches[i] + shift)
            new_p = _snap_to_scale(new_p, key)
            bar.pitches[i] = new_p

        # Velocity mutation
        # if random.random() < NOTE_VELOCITY_PROB:
        #     if bar.pitches[i] == REST_PITCH:
        #         continue  # rests stay at velocity 0
        #     nudge = random.choice([-5, -3, 3, 5])
        #     bar.velocities[i] = _clamp_velocity(bar.velocities[i] + nudge)

        # Rest toggle
        if random.random() < NOTE_REST_PROB:
            if bar.pitches[i] == REST_PITCH:
                # Rest → sounding: pick nearby scale tone
                neighbors = []
                if i > 0 and bar.pitches[i-1] != REST_PITCH:
                    neighbors.append(bar.pitches[i-1])
                if i < bar.n_notes - 1 and bar.pitches[i+1] != REST_PITCH:
                    neighbors.append(bar.pitches[i+1])
                base = random.choice(neighbors) if neighbors else 65
                bar.pitches[i] = _snap_to_scale(
                    _clamp_pitch(base + random.choice([-2,-1,0,1,2])), key)
                bar.velocities[i] = _clamp_velocity(
                    bar.velocities[max(0, i-1)] if i > 0 else 80)
            else:
                bar.pitches[i] = REST_PITCH


def mutate(
    genes: List[MelodyGenes],
    key: Key,
) -> List[MelodyGenes]:
    """Apply a meaningful batch of mutations.

    Each call does two things:
      1. Note-level: pick 1-3 random bars, apply per-note mutations
         (pitch, velocity, rest) independently to each note
      2. Bar-level: pick one bar, apply one structural change
         (rhythm, split, merge, cell_type)

    This ensures each mutation event changes ~3-8 genes, making
    the 20% mutation rate in the NSGA-II loop meaningful.
    """
    if not genes:
        return genes

    # --- Note-level mutations across 1-3 bars ---
    n_bars_to_mutate = random.randint(1, 3)
    for _ in range(n_bars_to_mutate):
        loc = _random_bar(genes)
        if loc is None:
            continue
        e_idx, b_idx = loc
        _mutate_bar_notes(genes[e_idx].bar_genes[b_idx], key)

    # --- One bar-level structural operation ---
    op = random.choices(
        list(BAR_OP_PROBS.keys()),
        list(BAR_OP_PROBS.values()),
    )[0]

    if   op == "rhythm":          mutate_rhythm(genes)
    elif op == "cell_type":       mutate_cell_type(genes)
    elif op == "event_velocity":  mutate_velocity(genes)
    elif op == "event_density":   mutate_density(genes, key)
    elif op == "event_register":  mutate_register(genes, key)
    # "none" → no structural change

    return genes