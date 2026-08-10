"""
music/renderer.py

Renders genome content (BarGenes/MelodyGenes) into VoicedNotes.

The renderer is deliberately simple — the musical content lives in
the genome (pitches, durations, velocities).  The renderer's job is:

    1. Convert BarGenes → List[VoicedNote]
    2. For development cells: snap to chord tones on strong beats
    3. Ensure durations sum correctly
    4. Optionally apply elaboration (passing/neighbor tones on long notes)

The cell_type determines interpretation:
    statement / sequence:  play exactly as encoded
    development:           snap to nearest chord tone on strong beats
    liquidation / passage: play exactly as encoded
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .genome import BarGenes, MelodyGenes
from .structures import Key, Chord, ChordProgression, group_chords_by_event
from .voicing import (
    VoicedNote, Voice, VOICE_RANGES, VOICE_VELOCITY_OFFSETS,
    get_chord_pitches,
)

MELODY_LOW, MELODY_HIGH = VOICE_RANGES[Voice.MELODY]
BEATS_PER_BAR = 4
MIN_NOTE_DURATION = 0.25
ELABORATION_MIN_NOTE_DUR = 2.0
ORNAMENT_PROBABILITY = 0.1

# Pitch value 0 means "rest" — the renderer emits a note with velocity 0
REST_PITCH = 0


# ---------------------------------------------------------------------------
# Chord context helpers (reused from melody.py pattern)
# ---------------------------------------------------------------------------

# A chord slice within a bar: (Chord, duration_in_this_bar, skeleton_pitch)
BarChord = Tuple[Chord, float, int]


def _group_chords_into_bars(
    chords: List[Chord],
    beats_per_bar: int = BEATS_PER_BAR,
) -> List[List[BarChord]]:
    """Group chords into bars. Simplified version — no skeleton needed,
    just chord and duration per slice."""
    bars: List[List[BarChord]] = []
    cur: List[BarChord] = []
    bar_beats = 0.0

    for chord in chords:
        rem = chord.duration
        while rem > MIN_NOTE_DURATION:
            space = beats_per_bar - bar_beats
            take = min(rem, space)
            if take < MIN_NOTE_DURATION:
                break
            cur.append((chord, take, 0))
            bar_beats += take
            rem -= take
            if bar_beats >= beats_per_bar - 0.001:
                bars.append(cur)
                cur = []
                bar_beats = 0.0

    if cur:
        bars.append(cur)
    return bars


def _chord_at_beat(bar_chords: List[BarChord], beat: float) -> Chord:
    """Return the Chord active at a given beat offset within the bar."""
    acc = 0.0
    for chord, dur, _ in bar_chords:
        acc += dur
        if beat < acc - 0.001:
            return chord
    return bar_chords[-1][0]


def _chord_tones_in_range(chord: Chord) -> List[int]:
    """All MIDI pitches of the chord triad within melody range."""
    pcs = get_chord_pitches(chord)
    pitches: List[int] = []
    for pc in pcs:
        p = pc
        while p < MELODY_LOW:
            p += 12
        while p <= MELODY_HIGH:
            pitches.append(p)
            p += 12
    return sorted(set(pitches)) if pitches else [67]


def _closest_chord_tone(pitch: int, chord: Chord) -> int:
    """Snap a pitch to the nearest chord tone within melody range."""
    candidates = _chord_tones_in_range(chord)
    if not candidates:
        return pitch
    return min(candidates, key=lambda p: abs(p - pitch))


# ---------------------------------------------------------------------------
# Bar-level rendering
# ---------------------------------------------------------------------------

def render_bar(
    bar_genes: BarGenes,
    bar_chords: List[BarChord],
    key: Key,
) -> List[VoicedNote]:
    """Render one bar from its genome into VoicedNotes.

    Args:
        bar_genes:   BarGenes with pitches, durations, velocities
        bar_chords:  chord context for this bar (for development snapping)
        key:         section tonal context (for elaboration)

    Returns:
        List[VoicedNote], durations sum to bar_genes.total_duration
    """
    if not bar_genes.pitches:
        return []

    notes: List[VoicedNote] = []
    beat_acc = 0.0

    for i in range(bar_genes.n_notes):
        pitch = bar_genes.pitches[i]
        duration = bar_genes.durations[i]
        velocity = bar_genes.velocities[i]

        # Rest: pitch 0 → emit with velocity 0 (silence)
        if pitch == REST_PITCH:
            notes.append(VoicedNote(
                pitch=60,  # arbitrary, won't sound
                duration=max(MIN_NOTE_DURATION, round(duration, 6)),
                velocity=0,
                voice=Voice.MELODY,
                is_chord_tone=False,
            ))
            beat_acc += duration
            continue

        # # Development cells: snap to chord tone on strong beats
        # if bar_genes.cell_type == "development" and bar_chords:
        #     is_strong_beat = (beat_acc % 2.0) < 0.01
        #     if is_strong_beat:
        #         active_chord = _chord_at_beat(bar_chords, beat_acc)
        #         pitch = _closest_chord_tone(pitch, active_chord)

        # # Clamp to range
        # pitch = max(MELODY_LOW, min(MELODY_HIGH, pitch))
        # velocity = max(0, min(127, velocity))
        # duration = max(MIN_NOTE_DURATION, duration)

        notes.append(VoicedNote(
            pitch=pitch,
            duration=round(duration, 6),
            velocity=velocity,
            voice=Voice.MELODY,
            is_chord_tone=False,
        ))
        beat_acc += duration

    return notes


# ---------------------------------------------------------------------------
# Elaboration (post-processing)
# ---------------------------------------------------------------------------

def _next_scale_tone(pitch: int, direction: int, key: Key) -> int:
    """Find the next diatonic scale tone above (+1) or below (-1)."""
    pcs = set(key.get_scale_degree().tolist())
    for step in range(1, 13):
        c = pitch + direction * step
        if c % 12 in pcs and MELODY_LOW <= c <= MELODY_HIGH:
            return c
    return max(MELODY_LOW, min(MELODY_HIGH, pitch + direction * 2))


def elaborate(
    notes: List[VoicedNote],
    key: Key,
    probability: float = ORNAMENT_PROBABILITY,
) -> List[VoicedNote]:
    """Add passing/neighbor tones to long notes.

    Only affects notes >= ELABORATION_MIN_NOTE_DUR beats.
    Preserves total duration exactly.
    """
    if probability <= 0 or not notes:
        return notes

    result: List[VoicedNote] = []

    for i, note in enumerate(notes):
        if note.duration < ELABORATION_MIN_NOTE_DUR or random.random() > probability:
            result.append(note)
            continue

        ornament = random.choice(["passing", "neighbor"])
        vel = note.velocity
        dur = note.duration

        if ornament == "passing" and i + 1 < len(notes):
            direction = 1 if notes[i + 1].pitch > note.pitch else -1
            passing = _next_scale_tone(note.pitch, direction, key)
            d1 = round(dur * 0.5, 6)
            d2 = round(dur * 0.25, 6)
            d3 = round(dur - d1 - d2, 6)
            result.append(VoicedNote(note.pitch, d1, vel, Voice.MELODY, False))
            result.append(VoicedNote(passing, d2, max(0, vel - 5), Voice.MELODY, False))
            result.append(VoicedNote(note.pitch, d3, vel, Voice.MELODY, False))
        elif ornament == "neighbor":
            neighbor = _next_scale_tone(note.pitch, random.choice([1, -1]), key)
            d1 = round(dur * 0.5, 6)
            d2 = round(dur * 0.25, 6)
            d3 = round(dur - d1 - d2, 6)
            result.append(VoicedNote(note.pitch, d1, vel, Voice.MELODY, False))
            result.append(VoicedNote(neighbor, d2, max(0, vel - 5), Voice.MELODY, False))
            result.append(VoicedNote(note.pitch, d3, vel, Voice.MELODY, False))
        else:
            result.append(note)

    return result


# ---------------------------------------------------------------------------
# Section-level rendering
# ---------------------------------------------------------------------------

def render_melody(
    progression: ChordProgression,
    melody_genes: List[MelodyGenes],
    section_key: Key,
    beats_per_bar: int = BEATS_PER_BAR,
    apply_elaboration: bool = False,
) -> List[VoicedNote]:
    """Render a full section's melody from genome into VoicedNotes.

    Args:
        progression:       section ChordProgression (chords carry event_id)
        melody_genes:      one MelodyGenes per event (ordered)
        section_key:       tonal context
        beats_per_bar:     time signature
        apply_elaboration: whether to add passing/neighbor tones

    Returns:
        List[VoicedNote], total duration == progression.total_duration
    """
    if not progression.chords:
        return []

    groups = group_chords_by_event(progression.chords)
    all_notes: List[VoicedNote] = []
    chord_idx = 0

    for group_idx, (event_id, event_chords) in enumerate(groups):
        genes = (melody_genes[group_idx]
                 if group_idx < len(melody_genes)
                 else MelodyGenes())

        # Group this event's chords into bars
        bars = _group_chords_into_bars(event_chords, beats_per_bar)

        event_notes: List[VoicedNote] = []

        for bar_idx, bar_chords in enumerate(bars):
            # Get bar genes (fallback to empty bar)
            bg = (genes.bar_genes[bar_idx]
                  if bar_idx < len(genes.bar_genes)
                  else BarGenes())

            bar_notes = render_bar(bg, bar_chords, section_key)
            event_notes.extend(bar_notes)

        # Elaboration
        if apply_elaboration:
            event_notes = elaborate(event_notes, section_key)

        all_notes.extend(event_notes)
        chord_idx += len(event_chords)

    return all_notes