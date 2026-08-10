"""
music/voicing.py

Harmonic voicing layer — Alto, Tenor, Bass voice assignment.

Receives a ChordProgression and produces voiced chords (HarmonicChord)
with proper voice leading.  This module is a pure harmonic renderer:
given chords and velocities, the output is deterministic.

The melody (Soprano) is handled separately by melody.py and combined
in compose.py.

Voice assignment:
    Alto    — chord tone, closest to previous alto (smooth voice leading)
    Tenor   — chord tone, below alto
    Bass    — melodic bass, root preference with stepwise voice leading

Hard constraints:
    - Notes within voice ranges
    - No voice crossing (Alto > Tenor)
    - No parallel fifths or octaves

Soft constraints (penalty scores for NSGA-II):
    - Stepwise motion preference
    - Correct doubling
    - Large leaps penalised
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple
import warnings

from .structures import (
    Key, Chord, ChordProgression, Motif, Note,
    SCALE_DEGREES_INDEX,
)


# ---------------------------------------------------------------------------
# Voice definitions and ranges
# ---------------------------------------------------------------------------

class Voice(str, Enum):
    MELODY = "melody"
    ALTO   = "alto"
    TENOR  = "tenor"
    BASS   = "bass"


VOICE_RANGES: Dict[Voice, Tuple[int, int]] = {
    Voice.MELODY: (55, 90),   # G3–
    Voice.ALTO:   (55, 72),   # G3–C5
    Voice.TENOR:  (48, 67),   # C3–G4
    Voice.BASS:   (40, 60),   # E2–C4
}

HARMONIC_VOICES = [Voice.ALTO, Voice.TENOR, Voice.BASS]

VOICE_VELOCITY_OFFSETS: Dict[Voice, int] = {
    Voice.MELODY:  0,
    Voice.ALTO:   -10,
    Voice.TENOR:  -15,
    Voice.BASS:   -12,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VoicedNote:
    """A single note assigned to a specific voice."""
    pitch: int
    duration: float
    velocity: int
    voice: Voice
    is_chord_tone: bool = True


@dataclass
class HarmonicChord:
    """A single chord with harmonic voices (Alto, Tenor, Bass) assigned."""
    chord: Chord
    alto: VoicedNote
    tenor: VoicedNote
    bass: VoicedNote
    base_velocity: int
    event_id: int

    @property
    def harmonic_voices(self) -> Dict[Voice, VoicedNote]:
        return {
            Voice.ALTO:  self.alto,
            Voice.TENOR: self.tenor,
            Voice.BASS:  self.bass,
        }

    @property
    def pitches(self) -> Dict[Voice, int]:
        return {v: n.pitch for v, n in self.harmonic_voices.items()}


@dataclass
class Arrangement:
    """Full arrangement for a music section.

    harmonic_chords: voiced A/T/B at harmonic rhythm
    melody_notes:    soprano VoicedNotes at melodic rhythm
    section_idx:     source section index
    key:             tonal context
    tempo_bpm:       section tempo
    """
    harmonic_chords: List[HarmonicChord]
    melody_notes: List[VoicedNote]
    section_idx: int
    key: Key
    tempo_bpm: float

    @property
    def duration_beats(self) -> float:
        return sum(hc.chord.duration for hc in self.harmonic_chords)


# ---------------------------------------------------------------------------
# Chord tone utilities
# ---------------------------------------------------------------------------

def get_chord_pitches(chord: Chord) -> List[int]:
    """Return pitch classes (0-11) for the chord's triad."""
    rn = chord.roman_numeral
    degree = SCALE_DEGREES_INDEX.get(rn, 0)
    return chord.key.get_chord(degree).tolist()


def closest_chord_tone(
    chord_pcs: List[int],
    target_midi: int,
    voice: Voice,
) -> int:
    """Find the MIDI pitch closest to target that belongs to the chord,
    within the voice's range."""
    low, high = VOICE_RANGES[voice]
    candidates = []
    for pc in chord_pcs:
        base = pc
        while base < low:
            base += 12
        while base <= high:
            candidates.append(base)
            base += 12
    if not candidates:
        return max(low, min(high, target_midi))
    return min(candidates, key=lambda p: abs(p - target_midi))


# ---------------------------------------------------------------------------
# Voice assignment
# ---------------------------------------------------------------------------

def _assign_bass(
    chord: Chord,
    chord_pcs: List[int],
    prev_bass: Optional[int],
    base_velocity: int,
) -> VoicedNote:
    """Assign Bass — root preference with stepwise voice leading."""
    low, high = VOICE_RANGES[Voice.BASS]
    velocity  = max(0, min(127, base_velocity + VOICE_VELOCITY_OFFSETS[Voice.BASS]))
    target    = prev_bass if prev_bass is not None else low + 7

    root_pitch = closest_chord_tone([chord_pcs[0]], target, Voice.BASS)
    best_pitch = root_pitch
    best_leap  = abs(root_pitch - target) if prev_bass is not None else 999

    if prev_bass is not None:
        for pc in chord_pcs[1:]:
            candidate = closest_chord_tone([pc], target, Voice.BASS)
            leap = abs(candidate - prev_bass)
            if leap <= 2 and best_leap > 4 and low <= candidate <= high:
                best_pitch = candidate
                best_leap  = leap

    return VoicedNote(pitch=best_pitch, duration=chord.duration,
                      velocity=velocity, voice=Voice.BASS,
                      is_chord_tone=True)


def _pitches_in_range(chord_pcs: List[int], voice: Voice) -> List[int]:
    """Return all MIDI pitches from chord_pcs within a voice's range."""
    low, high = VOICE_RANGES[voice]
    pitches: List[int] = []
    for pc in chord_pcs:
        p = pc
        while p < low:
            p += 12
        while p <= high:
            pitches.append(p)
            p += 12
    return sorted(set(pitches))


def _assign_upper_voices(
    chord: Chord,
    chord_pcs: List[int],
    bass_pitch: int,
    prev_alto: Optional[int],
    prev_tenor: Optional[int],
    base_velocity: int,
) -> Tuple[VoicedNote, VoicedNote]:
    """Assign Alto and Tenor together, maximising chord-tone coverage.

    Enumerates all valid (alto, tenor) pairs from chord tones within
    range, then selects the pair that:
        1. Covers the most distinct pitch classes (with bass) — primary
        2. Has the best voice-leading from previous positions — secondary

    Args:
        chord:       current Chord
        chord_pcs:   chord pitch classes [root, third, fifth]
        bass_pitch:  already-assigned bass MIDI pitch
        prev_alto:   previous alto MIDI pitch (None if first chord)
        prev_tenor:  previous tenor MIDI pitch (None if first chord)
        base_velocity: base MIDI velocity

    Returns:
        (alto_note, tenor_note) — VoicedNote pair
    """
    alto_vel  = max(0, min(127, base_velocity + VOICE_VELOCITY_OFFSETS[Voice.ALTO]))
    tenor_vel = max(0, min(127, base_velocity + VOICE_VELOCITY_OFFSETS[Voice.TENOR]))

    alto_candidates  = _pitches_in_range(chord_pcs, Voice.ALTO)
    tenor_candidates = _pitches_in_range(chord_pcs, Voice.TENOR)

    if not alto_candidates or not tenor_candidates:
        #
        warnings.warn(
            f"No candidates for alto in chord: {chord}] "
            f"Assigning silence) ",
            UserWarning,
            stacklevel=2,
        )
        return (
            VoicedNote(0, chord.duration, alto_vel, Voice.ALTO, False),
            VoicedNote(0, chord.duration, tenor_vel, Voice.TENOR, False),
        )

    bass_pc = bass_pitch % 12
    pc_set  = set(chord_pcs)

    # Default targets for voice leading
    alto_target  = prev_alto  if prev_alto  is not None else (VOICE_RANGES[Voice.ALTO][0] + VOICE_RANGES[Voice.ALTO][1]) // 2
    tenor_target = prev_tenor if prev_tenor is not None else (VOICE_RANGES[Voice.TENOR][0] + VOICE_RANGES[Voice.TENOR][1]) // 2

    best_pair     = None
    best_coverage = 0
    best_vl_cost  = float('inf')

    for a in alto_candidates:
        for t in tenor_candidates:
            # Hard constraint: alto must be above tenor
            if a <= t:
                continue

            # Count distinct pitch classes across all three voices
            covered = len({bass_pc, a % 12, t % 12} & pc_set)

            # Voice-leading cost: sum of distances from previous positions
            vl_cost = abs(a - alto_target) + abs(t - tenor_target)

            # Primary: maximise coverage. Secondary: minimise vl_cost.
            if (covered > best_coverage or
                    (covered == best_coverage and vl_cost < best_vl_cost)):
                best_pair     = (a, t)
                best_coverage = covered
                best_vl_cost  = vl_cost

    if best_pair == None:
        warnings.warn(f"Could not find best pair for chord: {chord}] "
                      f"Assigning lowest tenor and highest alto ",
                      UserWarning,
                      stacklevel=2,
                      )
        best_pair = (alto_candidates[-1], tenor_candidates[0])

    a_pitch, t_pitch = best_pair
    return (
        VoicedNote(a_pitch, chord.duration, alto_vel, Voice.ALTO, True),
        VoicedNote(t_pitch, chord.duration, tenor_vel, Voice.TENOR, True),
    )


# ---------------------------------------------------------------------------
# Hard constraint checking
# ---------------------------------------------------------------------------

def check_ranges(hc: HarmonicChord) -> List[str]:
    """Check harmonic voices are within defined pitch ranges."""
    violations = []
    for voice, vn in hc.harmonic_voices.items():
        low, high = VOICE_RANGES[voice]
        if not (low <= vn.pitch <= high):
            violations.append(
                f"{voice.value} pitch {vn.pitch} out of range [{low},{high}]")
    return violations


def check_voice_crossing(hc: HarmonicChord) -> List[str]:
    """Check Alto stays above Tenor."""
    violations = []
    if hc.alto.pitch <= hc.tenor.pitch:
        violations.append("Voice crossing: Alto <= Tenor")
    return violations


def check_parallel_motion(
    prev: HarmonicChord, curr: HarmonicChord,
) -> List[str]:
    """Check for parallel fifths and octaves between harmonic voices."""
    violations = []
    voices = HARMONIC_VOICES

    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            v1, v2 = voices[i], voices[j]
            prev_interval = abs(prev.pitches[v1] - prev.pitches[v2]) % 12
            curr_interval = abs(curr.pitches[v1] - curr.pitches[v2]) % 12
            d1 = curr.pitches[v1] - prev.pitches[v1]
            d2 = curr.pitches[v2] - prev.pitches[v2]
            parallel = (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0)
            if parallel:
                if prev_interval == 7 and curr_interval == 7:
                    violations.append(f"Parallel fifths: {v1.value}-{v2.value}")
                if prev_interval == 0 and curr_interval == 0:
                    violations.append(f"Parallel octaves: {v1.value}-{v2.value}")
    return violations


def is_valid(
    hc: HarmonicChord, prev_hc: Optional[HarmonicChord] = None,
) -> bool:
    """Return True if all hard constraints are satisfied."""
    if check_ranges(hc):        return False
    if check_voice_crossing(hc): return False
    if prev_hc and check_parallel_motion(prev_hc, hc): return False
    return True


# ---------------------------------------------------------------------------
# Soft constraint scoring
# ---------------------------------------------------------------------------

def score_voice_leading(prev: HarmonicChord, curr: HarmonicChord) -> float:
    """Penalise large leaps in harmonic voices. Lower is better."""
    penalty = 0.0
    weights = {Voice.ALTO: 1.5, Voice.TENOR: 1.5, Voice.BASS: 0.8}
    for voice in HARMONIC_VOICES:
        leap = abs(curr.pitches[voice] - prev.pitches[voice])
        if leap > 7:    penalty += weights[voice] * (leap - 7)
        elif leap > 4:  penalty += weights[voice] * 0.5
    return penalty


def score_doubling(hc: HarmonicChord) -> float:
    """Penalise incorrect doubling. Lower is better."""
    penalty   = 0.0
    chord_pcs = get_chord_pitches(hc.chord)
    if not chord_pcs:
        return penalty

    pcs = [n.pitch % 12 for n in hc.harmonic_voices.values()]
    pc_counts = {pc: pcs.count(pc) for pc in set(pcs)}

    scale        = hc.chord.key.get_scale_degree()
    leading_tone = int(scale[6]) if len(scale) > 6 else None

    if leading_tone is not None and pc_counts.get(leading_tone, 0) > 1:
        penalty += 2.0

    third_pc = chord_pcs[1] if len(chord_pcs) > 1 else None
    root_pc  = chord_pcs[0]
    bass_pc  = hc.bass.pitch % 12

    if bass_pc == root_pc and third_pc is not None:
        if pc_counts.get(third_pc, 0) > 1:
            penalty += 1.0

    return penalty


def score_leaps(prev: HarmonicChord, curr: HarmonicChord) -> float:
    """Penalise unresolved large leaps. Lower is better."""
    penalty = 0.0
    for voice in HARMONIC_VOICES:
        leap = curr.pitches[voice] - prev.pitches[voice]
        if abs(leap) > 4:
            penalty += abs(leap) * 0.3
    return penalty


def total_soft_penalty(
    hc: HarmonicChord, prev_hc: Optional[HarmonicChord] = None,
) -> float:
    """Combined soft constraint penalty."""
    penalty = score_doubling(hc)
    if prev_hc is not None:
        penalty += score_voice_leading(prev_hc, hc)
        penalty += score_leaps(prev_hc, hc)
    return penalty


# ---------------------------------------------------------------------------
# Main voicing function
# ---------------------------------------------------------------------------

def voice_chord(
    chord: Chord,
    event_id: int,
    base_velocity: int,
    prev_hc: Optional[HarmonicChord] = None,
) -> HarmonicChord:
    """Voice a single chord for Alto, Tenor, Bass.

    Uses coordinated assignment: Bass picks first (root preference),
    then Alto and Tenor are assigned together to maximise chord-tone
    coverage while maintaining smooth voice leading.

    Hard constraints (range, crossing, parallels) are checked on the
    result.  If parallel motion is detected with prev_hc, a second
    attempt is made with the tenor shifted by one chord tone.

    Args:
        chord:          Chord to voice
        event_id:       narrative event id
        base_velocity:  base MIDI velocity
        prev_hc:        previous HarmonicChord (for voice leading)

    Returns:
        HarmonicChord with maximised chord-tone coverage
    """
    chord_pcs    = get_chord_pitches(chord)
    prev_pitches = prev_hc.pitches if prev_hc else {}

    # Step 1: Bass
    bass = _assign_bass(chord, chord_pcs,
                         prev_pitches.get(Voice.BASS),
                         base_velocity)

    # Step 2: Alto + Tenor (coordinated)
    alto, tenor = _assign_upper_voices(
        chord, chord_pcs, bass.pitch,
        prev_pitches.get(Voice.ALTO),
        prev_pitches.get(Voice.TENOR),
        base_velocity,
    )

    hc = HarmonicChord(chord=chord, alto=alto, tenor=tenor, bass=bass,
                        base_velocity=base_velocity, event_id=event_id)

    # Validate — if parallel motion detected, try shifting tenor
    if prev_hc and not is_valid(hc, prev_hc):
        tenor_candidates = _pitches_in_range(chord_pcs, Voice.TENOR)
        for t_alt in tenor_candidates:
            if t_alt >= alto.pitch:
                continue
            tenor_alt = VoicedNote(t_alt, chord.duration,
                                   tenor.velocity, Voice.TENOR, True)
            hc_alt = HarmonicChord(chord=chord, alto=alto, tenor=tenor_alt,
                                    bass=bass, base_velocity=base_velocity,
                                    event_id=event_id)
            if is_valid(hc_alt, prev_hc):
                return hc_alt

    return hc


# ---------------------------------------------------------------------------
# Section voicing
# ---------------------------------------------------------------------------

def voice_progression(
    progression: ChordProgression,
    base_velocity: int = 80,
) -> List[HarmonicChord]:
    """Voice an entire chord progression.

    Iterates through chords in order, maintaining voice leading state.

    Args:
        progression:   ChordProgression (each chord carries event_id)
        base_velocity: base MIDI velocity

    Returns:
        ordered list of HarmonicChord
    """
    harmonic_chords: List[HarmonicChord] = []
    prev_hc: Optional[HarmonicChord] = None

    for chord in progression.chords:
        hc = voice_chord(
            chord=chord,
            event_id=chord.event_id,
            base_velocity=base_velocity,
            prev_hc=prev_hc,
        )
        harmonic_chords.append(hc)
        prev_hc = hc

    return harmonic_chords