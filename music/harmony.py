"""
music/harmony.py

Harmony grammar and chord progression generation.

Based on Rohrmeier's generative syntax of tonal harmony, implemented
as a context-free grammar over tonal regions (Tonic, Dominant,
Subdominant). Progressions are generated randomly from the grammar
for use as the initial population in NSGA-II optimisation.

Per-section generation pipeline:
    1. Compute median visual activity across section events
    2. Sample tempo from activity-appropriate normal distribution
    3. Compute number of bars (rounded to nearest multiple of 2)
    4. Compute harmonic rhythm from mean suspense
    5. Generate one chord progression per event

References:
    Rohrmeier, M. (2011). Towards a generative syntax of tonal harmony.
    Journal of Mathematics and Music, 5(1), 35-53.
"""

import random
import numpy as np
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass

from .tempo import assign_bars_to_events, find_tempo

from .structures import Key, Chord, ChordProgression
from narrative.schema import ACTIVITY_VALUES
from config import (
    HARMONIC_RHYTHM_MIN_BARS,
    HARMONIC_RHYTHM_MAX_BARS,
    ACTIVITY_TEMPO,
    TEMPO_MIN_BPM,
    TEMPO_MAX_BPM,
)


# ---------------------------------------------------------------------------
# Grammar symbols
# ---------------------------------------------------------------------------

class Symbol(Enum):
    # Non-terminals
    TR  = "TR"   # Tonic Region
    DR  = "DR"   # Dominant Region
    SR  = "SR"   # Subdominant Region

    # Terminals
    t   = "t"    # tonic chord
    d   = "d"    # dominant chord
    s   = "s"    # subdominant chord
    tp  = "tp"   # tonic parallel
    dp  = "dp"   # dominant parallel
    sp  = "sp"   # subdominant parallel
    tcp = "tcp"  # tonic cadential parallel

    def is_terminal(self) -> bool:
        return self in {
            Symbol.t, Symbol.d, Symbol.s,
            Symbol.tp, Symbol.dp, Symbol.sp, Symbol.tcp,
        }

    def is_non_terminal(self) -> bool:
        return self in {Symbol.TR, Symbol.DR, Symbol.SR}

    def to_roman_numeral(self, key: Key) -> str:
        """Map terminal symbol to Roman numeral for the given key."""
        if self == Symbol.t:
            return "I"
        elif self == Symbol.s:
            return "IV"
        elif self == Symbol.d:
            return "V"
        elif self == Symbol.tp:
            return "VI" if key.is_major else "III"
        elif self == Symbol.dp:
            return "VII"
        elif self == Symbol.sp:
            return "II" if key.is_major else "VI"
        elif self == Symbol.tcp:
            return "III" if key.is_major else "VI"
        return ""


# ---------------------------------------------------------------------------
# Grammar rules and parse tree
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    left: Symbol
    right: List[Symbol]

    def __hash__(self):
        return hash((self.left, *self.right))

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return (self.left == other.left and
                all(s1 == s2 for s1, s2 in zip(self.right, other.right)))


@dataclass
class TreeNode:
    symbol: Symbol
    children: List['TreeNode']
    rule_used: Optional[Rule] = None

    def get_terminals(self) -> List[Symbol]:
        """Return terminal symbols in left-to-right order."""
        if self.symbol.is_terminal():
            return [self.symbol]
        terminals = []
        for child in self.children:
            terminals.extend(child.get_terminals())
        return terminals

    def to_tuple(self) -> tuple:
        if not self.children:
            return (self.symbol.value,)
        return (self.symbol.value,
                tuple(child.to_tuple() for child in self.children))

    def copy(self) -> 'TreeNode':
        return TreeNode(
            self.symbol,
            [child.copy() for child in self.children],
            self.rule_used,
        )

    def print_tree(self, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.symbol.value)
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child.print_tree(child_prefix, i == len(self.children) - 1)

    def get_terminals_nodes_list(self) -> List['TreeNode']:
        """Return all terminal TreeNode instances in left-to-right order."""
        if self.symbol.is_terminal():
            return [self]
        nodes = []
        for child in self.children:
            nodes.extend(child.get_terminals_nodes_list())
        return nodes

    def to_structural_key(self) -> tuple:
        if self.symbol.is_terminal():
            return (self.symbol,)
        return (self.symbol, self.rule_used,
                tuple(child.to_structural_key() for child in self.children))


# ---------------------------------------------------------------------------
# Tempo computation
# ---------------------------------------------------------------------------

def compute_section_tempo(
    section,
    narrative_data: dict,
) -> dict:
    """
    Compute tempo for a music section in BPM.

    Derives the median visual activity across all events in the section,
    maps it to an activity-appropriate normal distribution, and samples
    a tempo value. The median is used because activity is discrete and
    ordinal — robust to outlier events.

    Args:
        section:        Section instance (from sections.py)
        narrative_data: full narrative dict from extract_narrative()
        seed:           random seed for reproducibility

    Returns:
        tempo in BPM, clamped to [TEMPO_MIN_BPM, TEMPO_MAX_BPM]
    """

    events_by_id = {e["event_id"]: e for e in narrative_data["events"]}

    # Collect activity values for all section events
    activity_values = sorted([
        ACTIVITY_VALUES[events_by_id[e["event_id"]]["visual_activity"]]
        for e in section.events
        if e["event_id"] in events_by_id
    ])

    if not activity_values:
        median_activity = 0.5
    else:
        n = len(activity_values)
        mid = n // 2
        if n % 2 == 0:
            median_activity = (activity_values[mid - 1] +
                               activity_values[mid]) / 2
        else:
            median_activity = activity_values[mid]

    # Find closest activity level in ACTIVITY_TEMPO
    closest = min(ACTIVITY_TEMPO.keys(),
                  key=lambda x: abs(x - median_activity))
    bpm_center, bpm_std = ACTIVITY_TEMPO[closest]

    # Sample from normal distribution
    # tempo = np.random.normal(bpm_center, bpm_std)
    tempo_info = find_tempo(section.duration, bpm_center, bpm_std)
    return tempo_info

# ---------------------------------------------------------------------------
# Harmonic rhythm
# ---------------------------------------------------------------------------

def compute_harmonic_rhythm_activity(
    activity: str,
    time_signature: int = 4,
) -> float:
    """
    Compute chord duration in beats from an event's visual activity.

    Maps activity to harmonic rhythm and snaps to a musical grid so
    that chord boundaries always fall on musically sensible subdivisions.

        low activity  → slow harmonic rhythm (longer chords)
        high activity → fast harmonic rhythm (shorter chords)

    Args:
        activity:        visual_activity string (e.g. "static", "frantic")
        time_signature:  beats per bar (default 4)

    Returns:
        chord duration in beats (grid-snapped)
    """
    bars_per_chord = (
        HARMONIC_RHYTHM_MAX_BARS -
        ACTIVITY_VALUES[activity] * (HARMONIC_RHYTHM_MAX_BARS - HARMONIC_RHYTHM_MIN_BARS)
    )

    # Snap to musical grid (in bars) 
    grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    bars_per_chord = min(grid, key=lambda x: abs(x - bars_per_chord))

    return bars_per_chord * time_signature

# ---------------------------------------------------------------------------
# Harmony grammar
# ---------------------------------------------------------------------------

class HarmonyGrammar:
    """
    Context-free grammar for tonal harmony generation.

    Based on Rohrmeier (2011). Generates chord progressions as
    derivation trees over tonal regions (TR, DR, SR).

    Parallel chord variants (tp, dp, sp, tcp) are not produced
    by the grammar directly — they are available as mutation
    operators in NSGA-II optimisation.
    """

    def __init__(self):
        self.rules = [
            Rule(Symbol.TR, [Symbol.DR, Symbol.t]),   # TR → DR t
            Rule(Symbol.DR, [Symbol.SR, Symbol.d]),   # DR → SR d
            Rule(Symbol.TR, [Symbol.TR, Symbol.DR]),  # TR → TR DR
            Rule(Symbol.TR, [Symbol.t]),              # TR → t
            Rule(Symbol.DR, [Symbol.d]),              # DR → d
            Rule(Symbol.SR, [Symbol.s]),              # SR → s
            # XR → XR XR handled dynamically in generate_random_tree
        ]

    def generate_random_tree(
        self,
        length: int,
        start_symbol: Symbol = Symbol.TR,
        max_attempts: int = 100,
    ) -> Optional[TreeNode]:
        """
        Generate a random derivation tree with exactly `length` terminals.

        Uses breadth-first expansion with random rule selection.
        Returns None if unable to reach the exact length within
        max_attempts.

        Args:
            length:       exact number of terminal symbols required
            start_symbol: starting non-terminal (default TR)
            max_attempts: number of retries before giving up

        Returns:
            TreeNode with exactly `length` terminals, or None
        """
        def count_leaves(node: TreeNode) -> int:
            if not node.children:
                return 1
            return sum(count_leaves(c) for c in node.children)

        def convert_to_terminals(node: TreeNode) -> TreeNode:
            # Each unexpanded non-terminal can become its base terminal
            # or a parallel variant — giving the grammar access to all
            # 7 diatonic chords from the start, not just I/IV/V.
            terminal_options = {
                Symbol.TR: [(Symbol.t, 0.55), (Symbol.tp, 0.25), (Symbol.tcp, 0.20)],
                Symbol.DR: [(Symbol.d, 0.65), (Symbol.dp, 0.35)],
                Symbol.SR: [(Symbol.s, 0.55), (Symbol.sp, 0.45)],
            }
            if not node.children:
                if node.symbol in terminal_options:
                    options = terminal_options[node.symbol]
                    symbols, weights = zip(*options)
                    node.symbol = random.choices(symbols, weights=weights)[0]
                return node
            node.children = [convert_to_terminals(c) for c in node.children]
            return node

        for _ in range(max_attempts):
            root = TreeNode(start_symbol, [])
            current_leaves = 1

            while current_leaves < length:
                # Collect expandable leaf nodes
                expandable = []
                stack = [(root, [])]
                while stack:
                    node, path = stack.pop()
                    if not node.children and node.symbol.is_non_terminal():
                        expandable.append((node, path))
                    for i, child in enumerate(node.children):
                        stack.append((child, path + [i]))

                if not expandable:
                    break

                node, path = random.choice(expandable)
                applicable_rules = [r for r in self.rules
                                     if r.left == node.symbol]
                # Duplication rule (XR → XR XR) with reduced probability.
                # Give it 1 weight vs 2 for each grammar rule, so it's
                # chosen less often than structural expansions.
                dup_rule = Rule(node.symbol, [node.symbol, node.symbol])
                all_rules = applicable_rules + [dup_rule]
                weights = [2.0] * len(applicable_rules) + [1.0]
                rule = random.choices(all_rules, weights=weights)[0]

                new_leaves = current_leaves - 1 + len(rule.right)
                if new_leaves > length:
                    continue

                target = root
                for idx in path:
                    target = target.children[idx]
                target.children = [TreeNode(sym, []) for sym in rule.right]
                target.rule_used = rule
                current_leaves = count_leaves(root)

            if current_leaves == length:
                return convert_to_terminals(root)

        return None

    def generate_progression(
        self,
        n_chords: int,
        chord_duration: float,
        key: Key,
        event_id: int,
        suspense: float,
    ) -> Optional[ChordProgression]:
        """
        Generate a chord progression for a single event.

        Args:
            n_chords:       number of chords
            chord_duration: duration per chord in beats
            key:            tonal context
            event_id:       narrative event id
            suspense:       global suspense value at this event

        Returns:
            ChordProgression, or None if generation failed
        """
        tree = self.generate_random_tree(n_chords)
        if tree is None:
            return None

        chords = [
            Chord(
                symbol=sym,
                roman_numeral=sym.to_roman_numeral(key),
                duration=chord_duration,
                key=key,
            )
            for sym in tree.get_terminals()
        ]

        return ChordProgression(
            chords=chords,
            event_id=event_id,
            suspense=suspense,
            key=key,
        )

    def generate_progression_for_section(
        self,
        section,
        narrative_data: dict,
        time_signature: int = 4,
    ) -> tuple:
        """
        Generate chord progression for all events in a music section.

        Args:
            section:        Section instance (from sections.py)
            narrative_data: full narrative dict from extract_narrative()
            time_signature: beats per bar (default 4)
            tempo_seed:     random seed for tempo sampling

        Returns:
            tuple of (tempo_bpm, progression) where progressions is
            a ChordProgression
        """

        tempo_info = compute_section_tempo(section, narrative_data)
        events_bars = assign_bars_to_events(section, tempo_info)
        events_by_id = {e["event_id"]: e for e in narrative_data["events"]}
        
        total_n_chords = 0
        durations = []
        event_ids = []   # parallel list: which event_id each chord belongs to
        for event_bar in events_bars:
            event = events_by_id.get(event_bar["event_id"], {})
            activity = event.get("visual_activity", "moderate")
            chord_duration_beats = compute_harmonic_rhythm_activity(activity, time_signature)
            event_beats = event_bar["n_bars"] * time_signature

            # Round to nearest integer — bars_per_chord of 2 or 4 with an odd
            # bar count produces a fractional n_chords (e.g. 3 bars at 2 bars/
            # chord = 1.5 chords).  We round and let the last chord absorb the
            # difference so all other chords stay on the musical grid.
            n_chords = max(1, round(event_beats / chord_duration_beats))
            total_n_chords += n_chords

            for j in range(n_chords - 1):
                durations.append(chord_duration_beats)
                event_ids.append(event_bar["event_id"])
            # Last chord fills whatever beats remain
            last_dur = event_beats - (n_chords - 1) * chord_duration_beats
            durations.append(max(chord_duration_beats * 0.5, last_dur))
            event_ids.append(event_bar["event_id"])

        tree = self.generate_random_tree(total_n_chords)
        if tree is None:
            return None

        chords = [
            Chord(
                symbol=sym,
                roman_numeral=sym.to_roman_numeral(section.key),
                duration=durations[i],
                key=section.key,
                event_id=event_ids[i],
            )
            for i, sym in enumerate(tree.get_terminals())
        ]

        progression = ChordProgression(
            chords=chords,
            key=section.key,
        )

        return tempo_info, progression