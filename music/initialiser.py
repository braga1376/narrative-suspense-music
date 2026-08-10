"""
music/initialiser.py

Population seeding for NSGA-II.

Uses the cell renderers from melody.py to generate musically valid
melodies, then captures the result into BarGenes/MelodyGenes from
genome.py.  Each individual in the initial population gets a different
random seed, producing diverse starting points.

Flow:
    generate_bar_genes    — render one bar, capture into BarGenes
    generate_event_genes  — render all bars in an event → MelodyGenes
    generate_section_genes — render all events in a section → List[MelodyGenes]
"""

from __future__ import annotations

import math
import random
import copy
from typing import List, Optional, Tuple

import numpy as np

from .genome import BarGenes, MelodyGenes, CELL_TYPES, REST_PITCH
from .structures import Key, Chord, ChordProgression, Motif, Note, group_chords_by_event
from .voicing import VoicedNote, Voice, VOICE_RANGES, get_chord_pitches

# Import melody.py cell rendering infrastructure
from .melody import (
    MotifData, CellGenes,
    extract_motif_data,
    build_skeleton,
    _render_cell,
    _group_chords_into_bars, _bar_budget, _bar_exit,
    random_cell_genes,
    MELODY_LOW, MELODY_HIGH, BEATS_PER_BAR,
    CONTOUR_DIRECTIONS, REGISTER_CENTERS,
    # We need the old MelodyGenes for skeleton building
    MelodyGenes as OldMelodyGenes,
)


# ---------------------------------------------------------------------------
# Cell type selection (biased seeding)
# ---------------------------------------------------------------------------

def _pick_cell_type(bar_idx: int, n_bars: int, has_character: bool) -> str:
    """Pick a cell type for one bar with musically sensible biases.

    When a character is present:
        first bar  → mostly statement (introduce the motif)
        middle bars → mix of development, sequence, statement, passage
        last bar   → mostly liquidation or passage (transition out)

    When no character is present:
        passage-dominated with some development
    """
    if has_character:
        if bar_idx == 0:
            return random.choices(
                ["statement", "development"], weights=[0.8, 0.2])[0]
        elif bar_idx == n_bars - 1 and n_bars > 1:
            return random.choices(
                ["liquidation", "passage", "development"],
                weights=[0.4, 0.3, 0.3])[0]
        else:
            return random.choices(
                ["development", "sequence", "statement", "passage"],
                weights=[0.30, 0.30, 0.20, 0.20])[0]
    else:
        return random.choices(
            ["passage", "development", "liquidation"],
            weights=[0.60, 0.25, 0.15])[0]


# ---------------------------------------------------------------------------
# Velocity patterns
# ---------------------------------------------------------------------------

def _apply_velocity_pattern(bar_genes: List[BarGenes], base_velocity: int) -> None:
    """Apply a musical velocity pattern across an event's bars.

    Modifies bar_genes in place.  Patterns:
        flat       — constant velocity
        crescendo  — gradually louder
        diminuendo — gradually softer
        accent     — first beat louder, rest softer
    """
    if not bar_genes:
        return

    pattern = random.choice(["flat", "crescendo", "diminuendo", "accent"])

    for bar_idx, bg in enumerate(bar_genes):
        n = len(bg.velocities)
        if n == 0:
            continue

        t = bar_idx / max(1, len(bar_genes) - 1)  # 0 to 1 across event

        if pattern == "flat":
            bg.velocities = [base_velocity] * n

        elif pattern == "crescendo":
            v_start = max(40, base_velocity - 15)
            v_end = min(110, base_velocity + 15)
            for i in range(n):
                note_t = (bar_idx + i / max(1, n)) / max(1, len(bar_genes))
                bg.velocities[i] = round(v_start + note_t * (v_end - v_start))

        elif pattern == "diminuendo":
            v_start = min(110, base_velocity + 15)
            v_end = max(40, base_velocity - 15)
            for i in range(n):
                note_t = (bar_idx + i / max(1, n)) / max(1, len(bar_genes))
                bg.velocities[i] = round(v_start + note_t * (v_end - v_start))

        elif pattern == "accent":
            for i in range(n):
                if i == 0 and bar_idx == 0:
                    bg.velocities[i] = min(110, base_velocity + 15)
                else:
                    bg.velocities[i] = max(40, base_velocity - 5)


# ---------------------------------------------------------------------------
# Bar-level generation
# ---------------------------------------------------------------------------

def generate_bar_genes(
    cell_type: str,
    start_pitch: int,
    bar_chords: list,
    beat_budget: float,
    motif: Optional[Motif],
    motif_data: Optional[MotifData],
    velocity: int,
    key: Key,
) -> Tuple[BarGenes, int]:
    """Generate one bar of melody and capture into BarGenes.

    Uses melody.py's _render_cell to produce notes, then extracts
    pitches, durations, and velocities.

    Args:
        cell_type:   rendering type for this bar
        start_pitch: last pitch of previous bar (continuity)
        bar_chords:  List[BarChord] — chord context for this bar
        beat_budget: total beats in this bar
        motif:       character's transposed motif (or None)
        motif_data:  cached motif properties (or None)
        velocity:    base velocity for this bar
        key:         section tonal context

    Returns:
        (BarGenes, last_pitch) — the captured genome and exit pitch
    """
    # Create random CellGenes to drive the renderer
    cell_genes = random_cell_genes(cell_type, motif_data)

    # Render using melody.py
    notes, last_pitch = _render_cell(
        cell_type, start_pitch, bar_chords, beat_budget,
        motif, motif_data, cell_genes, velocity, key,
    )

    # Capture into BarGenes
    if not notes:
        return BarGenes(
            cell_type=cell_type,
            pitches=[start_pitch],
            durations=[beat_budget],
            velocities=[velocity],
        ), start_pitch

    pitches = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    velocities = [n.velocity for n in notes]

    # Guard: if all pitches are identical and there are multiple notes,
    # introduce scale-tone variation around the base pitch
    if len(set(pitches)) <= 1 and len(pitches) > 1:
        base = pitches[0]
        scale_pcs = set(key.get_scale_degree().tolist())
        # Build nearby scale tones
        neighbors = []
        for offset in range(-4, 5):
            candidate = base + offset
            if (candidate % 12 in scale_pcs
                    and MELODY_LOW <= candidate <= MELODY_HIGH
                    and candidate != base):
                neighbors.append(candidate)
        if neighbors:
            for i in range(1, len(pitches)):
                # Alternate: base, neighbor, base, different neighbor...
                pitches[i] = neighbors[i % len(neighbors)]
            last_pitch = pitches[-1]

    # --- Longer note injection for non-motif bars ---
    # Occasionally merge two adjacent notes into one longer note
    # This creates held notes and reduces the quarter-note dominance
    if cell_type != "statement" and len(pitches) >= 4 and random.random() < 0.3:
        i = random.randint(0, len(pitches) - 2)
        merged_dur = durations[i] + durations[i + 1]
        merged_vel = round((velocities[i] + velocities[i + 1]) / 2)
        pitches.pop(i + 1)
        durations[i] = merged_dur
        durations.pop(i + 1)
        velocities[i] = merged_vel
        velocities.pop(i + 1)

    # --- Rest seeding: introduce rests at phrase boundaries ---
    # With some probability, convert the last note to a rest (phrase ending)
    # or insert a rest before the first note (pickup breath)
    rest_chance = 0.25  # 25% chance per bar
    if len(pitches) >= 3 and random.random() < rest_chance:
        # Convert last note to rest
        pitches[-1] = REST_PITCH
    elif len(pitches) >= 4 and random.random() < rest_chance * 0.5:
        # Convert first note to rest (breath before phrase)
        pitches[0] = REST_PITCH

    return BarGenes(
        cell_type=cell_type,
        pitches=pitches,
        durations=durations,
        velocities=velocities,
    ), last_pitch


# ---------------------------------------------------------------------------
# Phrase planning
# ---------------------------------------------------------------------------

# Roles a bar can play within a phrase structure
BAR_ROLE_MOTIF    = "motif"       # part of a multi-bar motif statement
BAR_ROLE_ACTIVE   = "active"      # normal melodic content
BAR_ROLE_HELD     = "held"        # one or two long held notes (breathing)
BAR_ROLE_REST     = "rest"        # mostly or entirely silence
BAR_ROLE_CADENCE  = "cadence"     # ending pattern (long note → rest)


def _assign_motif_characters(
    n_motif_bars: int,
    dominant_id: Optional[int],
    other_ids: List[int],
) -> List[int]:
    """Assign character IDs to motif bars.

    Dominant character gets ~50% of motif bars (at least 1).
    Remaining bars are split equally among other characters.
    Each character's bars are grouped together so their motif flows.

    Returns:
        List of character IDs, one per motif bar.
    """
    if n_motif_bars == 0:
        return []

    all_ids = [dominant_id] if dominant_id is not None else []
    all_ids.extend(other_ids)

    if not all_ids:
        return []

    if len(all_ids) == 1:
        return all_ids * n_motif_bars

    # Dominant gets ~50%, at least 1 bar
    dom_bars = max(1, round(n_motif_bars * 0.5))
    remaining_bars = n_motif_bars - dom_bars

    assignment = []

    # Dominant character first
    if dominant_id is not None:
        assignment.extend([dominant_id] * dom_bars)
    else:
        # No dominant — give first character the dom share
        assignment.extend([all_ids[0]] * dom_bars)

    # Split remaining equally among other characters
    if other_ids and remaining_bars > 0:
        bars_each = max(1, remaining_bars // len(other_ids))
        for cid in other_ids:
            n = min(bars_each, remaining_bars - (len(assignment) - dom_bars))
            if n <= 0:
                break
            assignment.extend([cid] * n)

    # Fill any leftover with dominant
    while len(assignment) < n_motif_bars:
        assignment.append(dominant_id if dominant_id is not None else all_ids[0])

    return assignment[:n_motif_bars]


def _plan_phrase_structure(
    n_bars: int,
    has_character: bool,
    motif_bars_needed: int,
    is_last_event: bool,
) -> List[str]:
    """Plan bar roles for one event.

    Creates a phrase structure that gives the melody shape:
    motif statements, active development, held notes for breathing,
    and a proper cadence at the end.

    Args:
        n_bars:             total bars in this event
        has_character:      whether any character motif is available
        motif_bars_needed:  total bars to allocate for motif statements
        is_last_event:      whether this is the last event in the section
    """
    if n_bars <= 1:
        return [BAR_ROLE_ACTIVE]

    roles = [BAR_ROLE_ACTIVE] * n_bars

    # --- Ending: last bar gets a cadential pattern ---
    if is_last_event and n_bars >= 2:
        roles[-1] = BAR_ROLE_CADENCE
    elif n_bars >= 3:
        if random.random() < 0.4:
            roles[-1] = BAR_ROLE_HELD

    # --- Motif statements ---
    if has_character and motif_bars_needed >= 1:
        bars_for_motif = min(motif_bars_needed, n_bars - 1)  # leave room for ending
        for i in range(bars_for_motif):
            if random.random() < 0.9 - i * 0.15:
                roles[i] = BAR_ROLE_MOTIF
            else:
                break

    # --- Breathing: insert held/rest bars in the middle of long events ---
    # Find first non-motif bar for breathing position
    first_active = next((i for i in range(n_bars) if roles[i] == BAR_ROLE_ACTIVE), n_bars)

    if n_bars >= 6:
        breath_pos = max(first_active, 1)
        if breath_pos < n_bars - 1 and roles[breath_pos] == BAR_ROLE_ACTIVE:
            roles[breath_pos] = random.choice([BAR_ROLE_HELD, BAR_ROLE_REST])

    if n_bars >= 8:
        mid = n_bars * 2 // 3
        if roles[mid] == BAR_ROLE_ACTIVE:
            roles[mid] = random.choice([BAR_ROLE_HELD, BAR_ROLE_REST])

    return roles


def _generate_held_bar(
    pitch: int,
    beat_budget: float,
    velocity: int,
    key: Key,
) -> BarGenes:
    """Generate a bar with 1-2 long held notes.

    Creates breathing space in the melody.
    """
    p = _snap_to_scale_init(pitch, key)

    if random.random() < 0.5:
        # Single held note for the full bar
        return BarGenes(
            cell_type="passage",
            pitches=[p],
            durations=[beat_budget],
            velocities=[velocity],
        )
    else:
        # Two notes: one long, one short rest — both must be grid-aligned
        # Pick rest so that hold = budget - rest is also on standard grid
        valid_rests = [r for r in [0.5, 1.0, 1.5, 2.0]
                       if (beat_budget - r) in {0.5, 1.0, 1.5, 2.0, 3.0, 4.0}
                       and beat_budget - r >= 0.5]
        if not valid_rests:
            return BarGenes(
                cell_type="passage",
                pitches=[p],
                durations=[beat_budget],
                velocities=[velocity],
            )
        rest_dur = random.choice(valid_rests)
        hold_dur = beat_budget - rest_dur
        return BarGenes(
            cell_type="passage",
            pitches=[p, REST_PITCH],
            durations=[hold_dur, rest_dur],
            velocities=[velocity, 0],
        )


def _generate_rest_bar(beat_budget: float) -> BarGenes:
    """Generate a bar that is entirely (or mostly) silent."""
    return BarGenes(
        cell_type="passage",
        pitches=[REST_PITCH],
        durations=[beat_budget],
        velocities=[0],
    )


def _generate_cadence_bar(
    pitch: int,
    bar_chords: list,
    beat_budget: float,
    velocity: int,
    key: Key,
) -> BarGenes:
    """Generate a cadential ending bar.

    Pattern: approach note → long resolution note → optional rest.
    The resolution pitch is a chord tone of the last chord.
    """
    from .voicing import get_chord_pitches

    # Resolution: chord tone of the last chord in this bar
    last_chord = bar_chords[-1][0]
    chord_pcs = get_chord_pitches(last_chord)
    # Find nearest chord tone to current pitch
    candidates = []
    for pc in chord_pcs:
        p = pc
        while p < MELODY_LOW:
            p += 12
        while p <= MELODY_HIGH:
            candidates.append(p)
            p += 12
    resolution = min(candidates, key=lambda p: abs(p - pitch)) if candidates else pitch

    # Approach note: one step away from resolution
    approach = _snap_to_scale_init(resolution + random.choice([-2, -1, 1, 2]), key)

    # Pattern: approach (short) → resolution (long) → rest
    # Use grid-aligned values
    if beat_budget >= 4.0:
        approach_dur = 1.0
        rest_dur = 1.0
    elif beat_budget >= 3.0:
        approach_dur = 0.5
        rest_dur = 0.5
    else:
        approach_dur = 0.5
        rest_dur = 0.5
    resolution_dur = beat_budget - approach_dur - rest_dur

    if resolution_dur < 0.5:
        # Not enough room, just do resolution + rest
        return BarGenes(
            cell_type="passage",
            pitches=[resolution, REST_PITCH],
            durations=[beat_budget * 0.75, beat_budget * 0.25],
            velocities=[velocity, 0],
        )

    return BarGenes(
        cell_type="passage",
        pitches=[approach, resolution, REST_PITCH],
        durations=[approach_dur, resolution_dur, rest_dur],
        velocities=[velocity, velocity - 5, 0],
    )


def _generate_motif_bars(
    motif: Motif,
    start_pitch: int,
    n_bars: int,
    beat_budget_per_bar: float,
    velocity: int,
    key: Key,
) -> List[BarGenes]:
    """Generate multi-bar motif statement.

    Takes consecutive notes from the motif and distributes them across
    n_bars bars, each with beat_budget_per_bar beats.
    """
    if not motif or not motif.notes:
        return []

    offset = start_pitch - motif.notes[0].pitch
    bars: List[BarGenes] = []
    note_idx = 0

    MIN_DUR = 0.25  # minimum note duration in beats

    for bar_i in range(n_bars):
        pitches = []
        durations = []
        velocities = []
        bar_remaining = beat_budget_per_bar

        while bar_remaining >= MIN_DUR and note_idx < len(motif.notes):
            note = motif.notes[note_idx]
            dur = min(note.duration, bar_remaining)

            # Skip notes that would be too short
            if dur < MIN_DUR:
                # If there's a previous note, extend it instead
                if durations:
                    durations[-1] += dur
                    bar_remaining -= dur
                    if dur >= note.duration - 0.001:
                        note_idx += 1
                    else:
                        motif.notes[note_idx] = Note(
                            note.pitch,
                            round(note.duration - dur, 6),
                            note.velocity,
                        )
                break

            p = note.pitch + offset
            p = max(MELODY_LOW, min(MELODY_HIGH, p))
            p = _snap_to_scale_init(p, key)

            pitches.append(p)
            durations.append(round(dur, 6))
            velocities.append(max(30, min(120, note.velocity)))

            bar_remaining -= dur

            # If the note was fully consumed, move to next
            if dur >= note.duration - 0.001:
                note_idx += 1
            else:
                # Note was split at bar boundary — the remainder goes to next bar
                # Adjust the remaining duration for next bar's first note
                motif.notes[note_idx] = Note(
                    note.pitch,
                    round(note.duration - dur, 6),
                    note.velocity,
                )
                break

        if not pitches:
            pitches = [start_pitch]
            durations = [beat_budget_per_bar]
            velocities = [velocity]

        bars.append(BarGenes(
            cell_type="statement",
            pitches=pitches,
            durations=durations,
            velocities=velocities,
        ))

    return bars


def _snap_to_scale_init(pitch: int, key: Key) -> int:
    """Snap pitch to nearest scale tone (for initialiser use)."""
    pcs = set(key.get_scale_degree().tolist())
    p = max(MELODY_LOW, min(MELODY_HIGH, pitch))
    if p % 12 in pcs:
        return p
    for offset in [1, -1, 2, -2]:
        c = p + offset
        if c % 12 in pcs and MELODY_LOW <= c <= MELODY_HIGH:
            return c
    return p


# ---------------------------------------------------------------------------
# Event-level generation (rewritten with phrase planning)
# ---------------------------------------------------------------------------

def generate_event_genes(
    event_chords: List[Chord],
    skeleton: List[int],
    chord_start_idx: int,
    event_motifs: dict,
    event_motif_data: dict,
    dominant_id: Optional[int],
    other_ids: List[int],
    base_velocity: int,
    key: Key,
    is_last_event: bool = False,
    beats_per_bar: int = BEATS_PER_BAR,
) -> Tuple[MelodyGenes, int]:
    """Generate melody genes for one narrative event.

    Supports multiple character motifs. The dominant character's motif
    gets ~50% of statement bars; other characters split the rest equally.

    Args:
        event_chords:    chords belonging to this event
        skeleton:        full-section skeleton pitches
        chord_start_idx: offset into skeleton for first chord
        event_motifs:    dict {character_id: transposed Motif}
        event_motif_data: dict {character_id: MotifData}
        dominant_id:     character_id of the dominant character (or None)
        other_ids:       character_ids of other characters present
        base_velocity:   base MIDI velocity
        key:             section tonal context
        is_last_event:   whether this is the last event in the section
        beats_per_bar:   time signature

    Returns:
        (MelodyGenes, last_pitch) — event genome and exit pitch
    """
    bars = _group_chords_into_bars(
        event_chords, skeleton, chord_start_idx, beats_per_bar)

    n_bars = len(bars)
    if n_bars == 0:
        return MelodyGenes(bar_genes=[]), chord_start_idx

    has_character = len(event_motifs) > 0

    # Determine the dominant motif (used for non-statement bars' duration pool)
    dom_motif = event_motifs.get(dominant_id)
    dom_motif_data = event_motif_data.get(dominant_id)
    if dom_motif is None and event_motifs:
        # Fallback: first available motif
        first_id = next(iter(event_motifs))
        dom_motif = event_motifs[first_id]
        dom_motif_data = event_motif_data.get(first_id)

    # Calculate total motif bars needed across all characters
    total_motif_bars = 0
    for cid, m in event_motifs.items():
        motif_beats = sum(n.duration for n in m.notes)
        total_motif_bars += max(1, int(np.ceil(motif_beats / beats_per_bar)))

    # Plan phrase structure
    roles = _plan_phrase_structure(n_bars, has_character, total_motif_bars,
                                   is_last_event)

    # Assign characters to motif bars
    n_motif_bars = sum(1 for r in roles if r == BAR_ROLE_MOTIF)
    char_assignment = _assign_motif_characters(n_motif_bars, dominant_id, other_ids)

    # Pre-generate motif bars per character
    # Each character's motif bars are generated as a batch
    motif_bar_caches: dict = {}  # character_id → List[BarGenes]
    motif_bar_indices: dict = {}  # character_id → next index to use

    bar_genes_list: List[BarGenes] = []
    prev_pitch: Optional[int] = None
    motif_slot = 0  # tracks which slot in char_assignment we're at

    for bar_idx, bar_chords in enumerate(bars):
        budget = _bar_budget(bar_chords)
        role = roles[bar_idx]

        if prev_pitch is None:
            prev_pitch = _bar_exit(bar_chords)

        if role == BAR_ROLE_MOTIF and motif_slot < len(char_assignment):
            cid = char_assignment[motif_slot]
            motif_slot += 1

            # Generate this character's motif bars on first encounter
            if cid not in motif_bar_caches:
                char_motif = event_motifs.get(cid)
                if char_motif is not None:
                    motif_copy = copy.deepcopy(char_motif)
                    # Count how many bars this character gets
                    char_bar_count = sum(1 for c in char_assignment if c == cid)
                    motif_bar_caches[cid] = _generate_motif_bars(
                        motif_copy, prev_pitch, char_bar_count, budget,
                        base_velocity, key)
                    motif_bar_indices[cid] = 0
                else:
                    motif_bar_caches[cid] = []
                    motif_bar_indices[cid] = 0

            cache = motif_bar_caches.get(cid, [])
            idx = motif_bar_indices.get(cid, 0)

            if cache and idx < len(cache):
                bg = cache[idx]
                motif_bar_indices[cid] = idx + 1
            else:
                # Fallback to normal generation
                bg, _ = generate_bar_genes(
                    "statement", prev_pitch, bar_chords, budget,
                    dom_motif, dom_motif_data, base_velocity, key)

        elif role == BAR_ROLE_HELD:
            bg = _generate_held_bar(prev_pitch, budget, base_velocity, key)

        elif role == BAR_ROLE_REST:
            bg = _generate_rest_bar(budget)

        elif role == BAR_ROLE_CADENCE:
            bg = _generate_cadence_bar(
                prev_pitch, bar_chords, budget, base_velocity, key)

        else:  # BAR_ROLE_ACTIVE
            cell_type = _pick_cell_type(bar_idx, n_bars, has_character)
            bg, _ = generate_bar_genes(
                cell_type, prev_pitch, bar_chords, budget,
                dom_motif, dom_motif_data, base_velocity, key)

        bar_genes_list.append(bg)

        # Track exit pitch
        sounding = [p for p in bg.pitches if p != REST_PITCH]
        prev_pitch = sounding[-1] if sounding else prev_pitch

    # Apply velocity pattern across the event
    _apply_velocity_pattern(bar_genes_list, base_velocity)

    return MelodyGenes(bar_genes=bar_genes_list), prev_pitch


# ---------------------------------------------------------------------------
# Section-level generation
# ---------------------------------------------------------------------------

def generate_section_genes(
    progression: ChordProgression,
    narrative_data: dict,
    motifs: List[Motif],
    section_key: Key,
    base_velocity: int = 80,
    beats_per_bar: int = BEATS_PER_BAR,
) -> List[MelodyGenes]:
    """Generate melody genes for all events in a section.

    Builds the skeleton, resolves motifs, then generates per-event
    MelodyGenes.  Returns one MelodyGenes per event.

    Args:
        progression:    section-level ChordProgression
        narrative_data: full narrative dict
        motifs:         list of Motif (in C, will be transposed)
        section_key:    section tonal context
        base_velocity:  base MIDI velocity
        beats_per_bar:  time signature

    Returns:
        List[MelodyGenes], one per event in the section
    """
    if not progression.chords:
        return []

    events_by_id = {e["event_id"]: e for e in narrative_data["events"]}
    motifs_by_char = {m.character_id: m for m in motifs}

    # Transpose motifs to section key
    transposition = section_key.root
    transposed = {
        cid: m.transpose(transposition)
        for cid, m in motifs_by_char.items()
    }

    # Cache motif data
    motif_data_cache = {
        cid: extract_motif_data(m) for cid, m in transposed.items()
    }

    # Build skeleton — needs the old MelodyGenes format for contour/register.
    # Use random contour and register for diversity in population seeding.
    groups = group_chords_by_event(progression.chords)
    old_genes = [
        OldMelodyGenes(
            target_register=random.randint(0, 2),
            contour_direction=random.choice(CONTOUR_DIRECTIONS),
        )
        for _ in groups
    ]
    skeleton = build_skeleton(progression, old_genes, events_by_id)

    # Generate per-event genes
    all_genes: List[MelodyGenes] = []
    chord_idx = 0
    prev_pitch: Optional[int] = None

    for group_idx, (event_id, event_chords) in enumerate(groups):
        event = events_by_id.get(event_id, {})

        # Resolve all characters present in this event
        chars_present = event.get("characters_present", [])
        # Coerce IDs to int
        chars_present = []
        for c in event.get("characters_present", []):
            try:
                chars_present.append(int(c))
            except (ValueError, TypeError):
                pass

        # Resolve dominant character
        dominant_id = event.get("dominant_character_id")
        if dominant_id is not None:
            try:
                dominant_id = int(dominant_id)
            except (ValueError, TypeError):
                dominant_id = None

        # If no dominant specified, use first character present
        if dominant_id is None and chars_present:
            dominant_id = chars_present[0]

        # If no characters at all, use first available motif
        if dominant_id is None and transposed:
            dominant_id = next(iter(transposed))
            chars_present = [dominant_id]

        # Build per-event motif dicts (only characters we have motifs for)
        other_ids = [c for c in chars_present
                     if c != dominant_id and c in transposed]
        event_motifs = {}
        event_motif_data_map = {}
        for cid in ([dominant_id] if dominant_id else []) + other_ids:
            if cid in transposed:
                event_motifs[cid] = transposed[cid]
                event_motif_data_map[cid] = motif_data_cache.get(cid)

        is_last = (group_idx == len(groups) - 1)

        event_genes, prev_pitch = generate_event_genes(
            event_chords=event_chords,
            skeleton=skeleton,
            chord_start_idx=chord_idx,
            event_motifs=event_motifs,
            event_motif_data=event_motif_data_map,
            dominant_id=dominant_id,
            other_ids=other_ids,
            base_velocity=base_velocity,
            key=section_key,
            is_last_event=is_last,
            beats_per_bar=beats_per_bar,
        )
        all_genes.append(event_genes)
        chord_idx += len(event_chords)

    return all_genes