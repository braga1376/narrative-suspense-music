"""
music/models.py

Shared musical data classes and constants used across the music module.

Contains:
    - Conversion constants (pitch, duration, note names)
    - Key: tonal center representation with scale and chord utilities
    - Note: single MIDI note
    - Motif: character leitmotif (with transposition)
    - Chord: single chord with duration
    - ChordProgression: sequence of chords for a narrative event
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Conversion constants
# ---------------------------------------------------------------------------

PITCH_CONVERSION     = [60, 62, 64, 65, 67, 69, 71,
                         72, 74, 76, 77, 79, 81, 83, 84]
DURATION_CONVERSION  = [4, 2, 1, 0.5, 0.25]
NOTE_VALUES          = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

ROOT_TO_NOTE = {
    0: 'C',  1: 'C#', 2: 'D',  3: 'D#',
    4: 'E',  5: 'F',  6: 'F#', 7: 'G',
    8: 'G#', 9: 'A', 10: 'A#', 11: 'B',
}
NOTE_TO_ROOT = {v: k for k, v in ROOT_TO_NOTE.items()}

SCALE_DEGREES_INDEX = {
    'I': 0, 'II': 1, 'III': 2, 'IV': 3,
    'V': 4, 'VI': 5, 'VII': 6,
}

SCALE_DEGREES_KIND_MAJOR = {
    'I': ' ', 'II': 'm', 'III': 'm', 'IV': ' ',
    'V': ' ', 'VI': 'm', 'VII': 'dim',
}

SCALE_DEGREES_KIND_MINOR = {
    'I': 'm', 'II': 'dim', 'III': ' ', 'IV': 'm',
    'V': 'm', 'VI': ' ', 'VII': ' ',
}

MAX_PITCH = 108
MIN_PITCH = 36

CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

KEYS_MAJOR = {
    't':   (0,  True),
    's':   (5,  True),
    'd':   (7,  True),
    'tp':  (9,  False),
    'sp':  (2,  False),
    'tcp': (4,  False),
}
KEYS_MINOR = {
    't':   (0,  False),
    's':   (5,  False),
    'd':   (7,  False),
    'tp':  (3,  True),
    'dp':  (10, True),
    'sp':  (8,  True),
    'tcp': (8,  True),
}


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------

@dataclass
class Key:
    """
    Represents a tonal center.

    root:     pitch class of the tonic (0=C, 1=C#, ..., 11=B)
    is_major: True for major, False for minor
    """
    root: int
    is_major: bool

    @property
    def is_minor(self) -> bool:
        return not self.is_major

    @property
    def name(self) -> str:
        mode = "major" if self.is_major else "minor"
        return f"{ROOT_TO_NOTE[self.root % 12]} {mode}"

    def get_scale_degree(self) -> np.ndarray:
        """Return pitch classes of all seven scale degrees."""
        intervals = (np.array([0, 2, 4, 5, 7, 9, 11]) if self.is_major
                     else np.array([0, 2, 3, 5, 7, 8, 10]))
        return (intervals + self.root) % 12

    def get_chord(self, degree: int) -> np.ndarray:
        """Return pitch classes of the triad on the given scale degree."""
        scale = self.get_scale_degree()
        return scale[[degree, (degree + 2) % 7, (degree + 4) % 7]]

    @property
    def pitch_classes(self) -> Set[int]:
        """Scale pitch classes as a set — for tonal motif generation."""
        return set(self.get_scale_degree().tolist())

    def relative_key(self, symbol_name: str) -> 'Key':
        """Return the key for a grammar terminal symbol relative to this key."""
        table = KEYS_MAJOR if self.is_major else KEYS_MINOR
        if symbol_name not in table:
            return self
        interval, is_major = table[symbol_name]
        return Key(root=(self.root + interval) % 12, is_major=is_major)

    def fifths_distance(self, other: 'Key') -> int:
        """Minimum circle-of-fifths steps between two key roots (0-6)."""
        i = CIRCLE_OF_FIFTHS.index(self.root % 12)
        j = CIRCLE_OF_FIFTHS.index(other.root % 12)
        clockwise = (j - i) % 12
        counter   = (i - j) % 12
        return min(clockwise, counter)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.root, self.is_major))

    def __eq__(self, other):
        if not isinstance(other, Key):
            return False
        return self.root == other.root and self.is_major == other.is_major


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------

@dataclass
class Note:
    """Single MIDI note."""
    pitch: int       # MIDI note number (0-127)
    duration: float  # duration in beats (1.0 = quarter note)
    velocity: int    # MIDI velocity (0-127)


# ---------------------------------------------------------------------------
# Motif
# ---------------------------------------------------------------------------

@dataclass
class Motif:
    """
    Character leitmotif — a two-bar melodic idea derived from
    the character's narrative properties.

    Generated in C (root = MIDI 60). Call transpose(semitones)
    to move to the section key before arrangement.
    """
    notes: List[Note]
    character_id: int
    character_name: str
    valence: float        # narrative valence (-10 to +10)
    avg_activity: float   # median visual activity (0.0 to 1.0)
    tonal: bool           # whether tonal constraints were applied
    key: int = field(default=60)  # root MIDI note (default C = 60)

    def transpose(self, semitones: int) -> 'Motif':
        """
        Return a new Motif transposed by the given number of semitones.

        The original motif is unchanged — this returns a copy with all
        note pitches shifted and the key root updated accordingly.

        Args:
            semitones: semitones to transpose (positive = up, negative = down)

        Returns:
            new Motif transposed by semitones
        """
        return Motif(
            notes=[
                Note(
                    pitch=n.pitch + semitones,
                    duration=n.duration,
                    velocity=n.velocity,
                )
                for n in self.notes
            ],
            character_id=self.character_id,
            character_name=self.character_name,
            valence=self.valence,
            avg_activity=self.avg_activity,
            tonal=self.tonal,
            key=self.key + semitones,
        )


# ---------------------------------------------------------------------------
# Chord
# ---------------------------------------------------------------------------

@dataclass
class Chord:
    """
    Single chord in a harmonic progression.

    symbol:        grammar terminal (Symbol enum from harmony)
    roman_numeral: scale degree string (e.g. 'I', 'V', 'II')
    duration:      duration in beats
    key:           tonal context for this chord
    event_id:      narrative event this chord belongs to (optional —
                   set per-chord when one ChordProgression spans a
                   full section with multiple events)
    """
    symbol: object    # Symbol — typed as object to avoid circular import
    roman_numeral: str
    duration: float
    key: Key
    event_id: Optional[int] = None


# ---------------------------------------------------------------------------
# ChordProgression
# ---------------------------------------------------------------------------

@dataclass
class ChordProgression:
    """
    Sequence of chords for a section or event.

    In the new architecture one ChordProgression spans a full section.
    Each Chord carries its own event_id so the arrangement layer can
    still apply per-event logic (motif placement, alto voice, etc.).

    chords:   ordered list of Chord instances (each with event_id)
    key:      tonal context for the progression
    event_id: set only when the progression covers a single event
              (kept for backwards compatibility)
    suspense: set only when the progression covers a single event
    """
    chords: List[Chord]
    key: Key
    event_id: Optional[int] = None
    suspense: Optional[float] = None

    @property
    def total_duration(self) -> float:
        return sum(c.duration for c in self.chords)

    @property
    def roman_numerals(self) -> List[str]:
        return [c.roman_numeral for c in self.chords]

    def __str__(self) -> str:
        return " → ".join(self.roman_numerals)
    
def group_chords_by_event(
    chords: List['Chord'],
) -> List[Tuple[int, List['Chord']]]:
    """Group consecutive chords sharing an event_id.
 
    Returns a list of (event_id, chords) pairs in timeline order.
    Used by melody.py, compose.py, and objectives.py.
    """
    groups: List[Tuple[int, List['Chord']]] = []
    for chord in chords:
        eid = chord.event_id
        if groups and groups[-1][0] == eid:
            groups[-1][1].append(chord)
        else:
            groups.append((eid, [chord]))
    return groups
 