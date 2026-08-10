"""
music/genome.py

Genome data structures for NSGA-II melody evolution.

The genome encodes actual musical content — MIDI pitches, durations,
and velocities — not parameters for generation algorithms.

Structure:
    MelodyGenes        — one per narrative event
        bar_genes      — one BarGenes per bar in the event
            cell_type  — rendering interpretation
            pitches    — MIDI pitches (one per note)
            durations  — beat durations (sum = bar beat budget)
            velocities — per-note MIDI velocity

The cell_type determines how literally the pitches are rendered:
    statement:   play as-is (preserves motif identity)
    development: snap to chord tones on strong beats
    passage:     stepwise interpolation between pitches
    liquidation: literal at start, simplify toward end
    sequence:    play as-is with transposed repetition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Valid cell types
CELL_TYPES = ["statement", "development", "sequence", "liquidation", "passage"]

# Pitch value 0 means "rest" — the renderer emits silence
REST_PITCH = 0


@dataclass
class BarGenes:
    """Genome for one bar of melody.

    pitches, durations, and velocities must all have the same length.
    sum(durations) should equal the bar's beat budget.
    """
    cell_type: str         = "passage"
    pitches: List[int]     = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    velocities: List[int]  = field(default_factory=list)

    @property
    def n_notes(self) -> int:
        return len(self.pitches)

    @property
    def total_duration(self) -> float:
        return sum(self.durations)

    def copy(self) -> BarGenes:
        return BarGenes(
            cell_type=self.cell_type,
            pitches=self.pitches.copy(),
            durations=self.durations.copy(),
            velocities=self.velocities.copy(),
        )


@dataclass
class MelodyGenes:
    """Genome for one narrative event's melody.

    bar_genes has one BarGenes per bar in the event.
    """
    bar_genes: List[BarGenes] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return len(self.bar_genes)

    def copy(self) -> MelodyGenes:
        return MelodyGenes(
            bar_genes=[bg.copy() for bg in self.bar_genes],
        )