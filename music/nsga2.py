"""
music/nsga2.py

NSGA-II optimisation for narrative music composition.

Evolves melody content (pitches, durations, velocities per note) and
harmony trees (chord grammar) simultaneously to optimise three objectives:
    1. tonal_incoherence
    2. motif_recognition
    3. tension_misalignment

Rendering pipeline:
    harmony_tree → ChordProgression → melody (renderer.py) →
    velocity matching → voicing (voicing.py) → Arrangement

References:
    Deb, K. et al. (2002). NSGA-II.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable

import numpy as np

from .genome import BarGenes, MelodyGenes
from .structures import Key, Motif, Chord, ChordProgression, group_chords_by_event
from .voicing import (
    Arrangement, VoicedNote, HarmonicChord, Voice,
    voice_chord, voice_progression,
)
from .renderer import render_melody
from .initialiser import generate_section_genes
from .harmony import (
    HarmonyGrammar, TreeNode, Symbol,
    compute_section_tempo, compute_harmonic_rhythm_activity,
)
from .tempo import assign_bars_to_events

from .operators import (
    mutate as mutate_melody,
    _copy_genes,
)

from config import (
    NSGA_POPULATION_SIZE, NSGA_N_GENERATIONS,
    NSGA_CROSSOVER_RATE, NSGA_MUTATION_RATE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SectionGenome:
    """One section's worth of genetic material."""
    section_idx: int
    harmony_trees: List[List[TreeNode]]  # per event: list of phrase-level trees
    melody_genes: List[MelodyGenes]      # one per event

    def copy(self) -> SectionGenome:
        return SectionGenome(
            section_idx=self.section_idx,
            harmony_trees=[[t.copy() for t in event_trees]
                           for event_trees in self.harmony_trees],
            melody_genes=[mg.copy() for mg in self.melody_genes],
        )


class Individual:
    """One complete musical piece — all sections."""

    def __init__(self, sections: List[SectionGenome]):
        self.sections = sections
        self.objectives: List[float] = []
        self.rank: float = float('inf')
        self.crowding_distance: float = 0.0

    def dominates(self, other: Individual) -> bool:
        """All objectives minimised."""
        better = False
        for a, b in zip(self.objectives, other.objectives):
            if a > b:
                return False
            if a < b:
                better = True
        return better

    def copy(self) -> Individual:
        ind = Individual([s.copy() for s in self.sections])
        ind.objectives = self.objectives.copy()
        ind.rank = self.rank
        ind.crowding_distance = self.crowding_distance
        return ind


# ---------------------------------------------------------------------------
# Harmony trees → ChordProgression (per-event)
# ---------------------------------------------------------------------------

# Maximum terminals per tree — keeps progressions varied
MAX_TREE_SIZE = 6


def trees_to_progression(
    event_trees: List[List[TreeNode]],
    section,
    narrative_data: dict,
    tempo_info: dict,
    time_signature: int = 4,
) -> ChordProgression:
    """Convert per-event phrase-level trees into a section ChordProgression.

    Each event has a list of small trees (≤ MAX_TREE_SIZE terminals each).
    Their terminals are concatenated to fill the event's chord slots.
    No cycling — if more chords are needed, the terminals are used once
    and truncated.
    """
    events_by_id = {e["event_id"]: e for e in narrative_data["events"]}

    if not section.events:
        return ChordProgression(chords=[], key=section.key)

    events_bars = assign_bars_to_events(section, tempo_info)

    all_chords: List[Chord] = []

    for ev_idx, eb in enumerate(events_bars):
        event = events_by_id.get(eb["event_id"], {})
        activity = event.get("visual_activity", "moderate")
        chord_dur = compute_harmonic_rhythm_activity(activity, time_signature)
        event_beats = eb["n_bars"] * time_signature
        n_chords = max(1, round(event_beats / chord_dur))

        # Get phrase trees for this event
        phrase_trees = event_trees[ev_idx] if ev_idx < len(event_trees) else []

        # Concatenate all terminals from the phrase trees
        terms = []
        for tree in phrase_trees:
            if tree is not None:
                terms.extend(tree.get_terminals())

        if not terms:
            # Fallback: single tonic chord
            terms = [Symbol.t]

        for j in range(n_chords):
            sym = terms[j] if j < len(terms) else terms[-1]
            rn = sym.to_roman_numeral(section.key)
            if j == n_chords - 1:
                used = (n_chords - 1) * chord_dur
                dur = max(chord_dur * 0.5, event_beats - used)
            else:
                dur = chord_dur
            all_chords.append(Chord(
                symbol=sym, roman_numeral=rn,
                duration=dur, key=section.key, event_id=eb["event_id"],
            ))

    return ChordProgression(chords=all_chords, key=section.key)


# ---------------------------------------------------------------------------
# Velocity matching: melody → chord velocities
# ---------------------------------------------------------------------------

def _melody_velocity_per_chord(
    melody_notes: List[VoicedNote],
    chords: List[Chord],
) -> List[int]:
    """Extract per-chord velocity from melody notes.

    For each chord's time window, average the velocities of melody notes
    that start within it. If no melody notes fall in a window, use the
    previous chord's velocity.

    Works directly with Chord list (no HarmonicChord needed).
    """
    if not chords:
        return []

    # Build chord windows from durations
    windows: List[Tuple[float, float]] = []
    beat = 0.0
    for chord in chords:
        windows.append((beat, beat + chord.duration))
        beat += chord.duration

    # Assign melody notes to windows (sounding notes only)
    chord_vels: List[List[int]] = [[] for _ in chords]
    note_beat = 0.0
    ci = 0

    for note in melody_notes:
        while ci < len(chords) - 1 and note_beat >= windows[ci][1]:
            ci += 1
        # Only include sounding notes — rests (velocity=0) should not
        # drag down chord velocity to silence
        if note.velocity > 0:
            chord_vels[ci].append(note.velocity)
        note_beat += note.duration

    # Compute per-chord velocity
    result: List[int] = []
    prev_vel = 80

    for vels in chord_vels:
        if vels:
            v = round(sum(vels) / len(vels))
            result.append(v)
            prev_vel = v
        else:
            result.append(prev_vel)

    return result


# ---------------------------------------------------------------------------
# Rendering pipeline
# ---------------------------------------------------------------------------

def render_individual(
    individual: Individual,
    sections: list,
    narrative_data: dict,
    motifs: List[Motif],
    section_tempo_infos: List[dict],
    apply_elaboration: bool = False,
) -> Dict[int, Arrangement]:
    """Render all sections of an individual into Arrangements.

    Pipeline per section:
        1. harmony_trees → ChordProgression
        2. MelodyGenes + ChordProgression → melody VoicedNotes
        3. Extract per-chord velocity from melody (no voicing needed)
        4. Voice chords ONCE with matched velocities
        5. Combine → Arrangement

    Set apply_elaboration=True only for final output (adds passing/
    neighbor tones but slows evaluation).
    """
    arrangements: Dict[int, Arrangement] = {}

    for sg in individual.sections:
        section = sections[sg.section_idx]
        tempo_info = section_tempo_infos[sg.section_idx]

        # Step 1: per-event trees → progression
        progression = trees_to_progression(
            sg.harmony_trees, section, narrative_data, tempo_info)

        if not progression.chords:
            continue

        # Step 2: melody
        melody_notes = render_melody(
            progression=progression,
            melody_genes=sg.melody_genes,
            section_key=section.key,
            apply_elaboration=apply_elaboration,
        )

        # Step 3: extract per-chord velocity from melody
        chord_velocities = _melody_velocity_per_chord(
            melody_notes, progression.chords)

        # Step 4: voice chords ONCE with matched velocities
        harmonic_chords: List[HarmonicChord] = []
        prev_hc = None
        for i, chord in enumerate(progression.chords):
            vel = chord_velocities[i] if i < len(chord_velocities) else 80
            hc = voice_chord(chord, chord.event_id, vel, prev_hc)
            harmonic_chords.append(hc)
            prev_hc = hc

        # Step 5: combine
        arrangements[sg.section_idx] = Arrangement(
            harmonic_chords=harmonic_chords,
            melody_notes=melody_notes,
            section_idx=sg.section_idx,
            key=section.key,
            tempo_bpm=tempo_info["bpm"],
        )

    return arrangements


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def create_initial_population(
    sections: list,
    narrative_data: dict,
    motifs: List[Motif],
    section_tempo_infos: List[dict],
    grammar: HarmonyGrammar,
    population_size: int,
) -> List[Individual]:
    """Generate diverse initial population.

    Each event gets one or more small harmony trees (≤ MAX_TREE_SIZE
    terminals each). Long events that need many chords get multiple
    phrase-level trees, ensuring harmonic variety.
    """
    population: List[Individual] = []

    for pop_idx in range(population_size):
        section_genomes: List[SectionGenome] = []

        for s_idx, section in enumerate(sections):
            if section.is_silence:
                continue

            tempo_info = section_tempo_infos[s_idx]
            events_bars = assign_bars_to_events(section, tempo_info)
            events_by_id = {e["event_id"]: e for e in narrative_data["events"]}

            # Generate phrase-level trees for each event
            all_event_trees: List[List[TreeNode]] = []
            for eb in events_bars:
                event = events_by_id.get(eb["event_id"], {})
                activity = event.get("visual_activity", "moderate")
                chord_dur = compute_harmonic_rhythm_activity(activity)
                event_beats = eb["n_bars"] * 4
                n_chords = max(1, round(event_beats / chord_dur))

                # Split into phrase-sized chunks
                phrase_trees: List[TreeNode] = []
                remaining = n_chords
                while remaining > 0:
                    chunk = min(remaining, MAX_TREE_SIZE)
                    tree = grammar.generate_random_tree(max(2, chunk))
                    if tree is None:
                        tree = grammar.generate_random_tree(2)
                    if tree is None:
                        tree = TreeNode(Symbol.t, [])
                    phrase_trees.append(tree)
                    remaining -= len(tree.get_terminals())

                all_event_trees.append(phrase_trees)

            # Build progression from phrase trees
            progression = trees_to_progression(
                all_event_trees, section, narrative_data, tempo_info)

            if not progression.chords:
                continue

            # Generate melody genes against this exact progression
            melody_genes = generate_section_genes(
                progression=progression,
                narrative_data=narrative_data,
                motifs=motifs,
                section_key=section.key,
                base_velocity=random.randint(60, 100),
            )

            section_genomes.append(SectionGenome(
                section_idx=s_idx,
                harmony_trees=all_event_trees,
                melody_genes=melody_genes,
            ))

        population.append(Individual(section_genomes))

    return population


# ---------------------------------------------------------------------------
# Harmony mutations
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Combined mutation dispatch
# ---------------------------------------------------------------------------

# Probability of choosing each mutation type
MUTATION_TYPE_PROBS = {
    "melody":               0.55,
    "harmony_parallel":     0.25,
    "harmony_regeneration": 0.20,
}

# Per-event application probability
EVENT_MUTATION_PROB = 0.5


def _mutate_harmony_regeneration(
    individual: Individual,
    grammar: HarmonyGrammar,
) -> Individual:
    """Replace one or more event's phrase trees with freshly generated ones.

    For each event (50/50 chance), regenerates all phrase trees,
    preserving the total terminal count.
    """
    new_ind = individual.copy()
    if not new_ind.sections:
        return new_ind

    sg = random.choice(new_ind.sections)
    for i in range(len(sg.harmony_trees)):
        if random.random() < EVENT_MUTATION_PROB:
            # Count total terminals across all phrase trees in this event
            total_terms = sum(
                len(t.get_terminals()) for t in sg.harmony_trees[i])
            # Regenerate as phrase-sized chunks
            new_phrases: List[TreeNode] = []
            remaining = max(2, total_terms)
            while remaining > 0:
                chunk = min(remaining, MAX_TREE_SIZE)
                tree = grammar.generate_random_tree(max(2, chunk))
                if tree is None:
                    tree = TreeNode(Symbol.t, [])
                new_phrases.append(tree)
                remaining -= len(tree.get_terminals())
            sg.harmony_trees[i] = new_phrases

    return new_ind


def mutate(
    individual: Individual,
    sections: list,
    grammar: HarmonyGrammar = None,
) -> Individual:
    """Apply mutations per-event with 50/50 probability.

    Chooses a mutation type, then for each event in a random section,
    independently applies it with EVENT_MUTATION_PROB chance.
    """
    op = random.choices(
        list(MUTATION_TYPE_PROBS.keys()),
        list(MUTATION_TYPE_PROBS.values()),
    )[0]

    if op == "melody":
        new_ind = individual.copy()
        if not new_ind.sections:
            return new_ind
        sg = random.choice(new_ind.sections)
        section = sections[sg.section_idx]
        # Apply melody mutation per-event with 50/50
        for i in range(len(sg.melody_genes)):
            if random.random() < EVENT_MUTATION_PROB:
                # mutate_melody operates on the full list but picks random bars
                # Call it once per selected event by passing a single-event list
                single = [sg.melody_genes[i]]
                mutate_melody(single, section.key)
                sg.melody_genes[i] = single[0]
        return new_ind

    elif op == "harmony_parallel":
        new_ind = individual.copy()
        if not new_ind.sections:
            return new_ind
        sg = random.choice(new_ind.sections)
        # Apply parallel substitution per-event with 50/50
        for i in range(len(sg.harmony_trees)):
            if random.random() < EVENT_MUTATION_PROB:
                # Each terminal has 30% chance of substitution
                for phrase_tree in sg.harmony_trees[i]:
                    for node in phrase_tree.get_terminals_nodes_list():
                        if random.random() < 0.3:
                            if node.symbol == Symbol.t:
                                node.symbol = random.choices(
                                    [Symbol.tp, Symbol.tcp], weights=[0.7, 0.3])[0]
                            elif node.symbol == Symbol.d:
                                node.symbol = Symbol.dp
                            elif node.symbol == Symbol.s:
                                node.symbol = Symbol.sp
        return new_ind

    elif op == "harmony_regeneration":
        if grammar is None:
            grammar = HarmonyGrammar()
        return _mutate_harmony_regeneration(individual, grammar)

    return individual


# ---------------------------------------------------------------------------
# Combined crossover dispatch (uniform crossover per event)
# ---------------------------------------------------------------------------

CROSSOVER_PROBS = {
    "melody":   0.35,    # swap melody_genes per event, 50/50
    "harmony":  0.30,    # swap harmony_trees per event, 50/50
    "coupled":  0.35,    # swap both together per event, 50/50
}


def _uniform_crossover_section(
    sg1: SectionGenome,
    sg2: SectionGenome,
    swap_melody: bool,
    swap_harmony: bool,
) -> None:
    """Apply uniform crossover to one section's events.

    For each event index, independently flip a coin. If heads,
    swap the specified components between sg1 and sg2.
    Modifies sg1 and sg2 in place.
    """
    n_events = min(len(sg1.melody_genes), len(sg2.melody_genes))
    n_trees = min(len(sg1.harmony_trees), len(sg2.harmony_trees))

    for i in range(max(n_events, n_trees)):
        if random.random() < 0.5:
            # Swap this event's components
            if swap_melody and i < n_events:
                sg1.melody_genes[i], sg2.melody_genes[i] = (
                    sg2.melody_genes[i].copy(),
                    sg1.melody_genes[i].copy(),
                )
            if swap_harmony and i < n_trees:
                sg1.harmony_trees[i], sg2.harmony_trees[i] = (
                    [t.copy() for t in sg2.harmony_trees[i]],
                    [t.copy() for t in sg1.harmony_trees[i]],
                )


def crossover(
    p1: Individual, p2: Individual,
    probs: dict = None,
) -> Tuple[Individual, Individual]:
    """Apply uniform crossover — melody, harmony, or both.

    For each event position, independently flip a coin to decide
    which parent contributes. Children are always complementary.
    """
    if probs is None:
        probs = CROSSOVER_PROBS

    op = random.choices(list(probs.keys()), list(probs.values()))[0]

    c1, c2 = p1.copy(), p2.copy()
    n = min(len(c1.sections), len(c2.sections))
    if n == 0:
        return c1, c2

    for s_idx in range(n):
        if op == "melody":
            _uniform_crossover_section(
                c1.sections[s_idx], c2.sections[s_idx],
                swap_melody=True, swap_harmony=False)
        elif op == "harmony":
            _uniform_crossover_section(
                c1.sections[s_idx], c2.sections[s_idx],
                swap_melody=False, swap_harmony=True)
        elif op == "coupled":
            _uniform_crossover_section(
                c1.sections[s_idx], c2.sections[s_idx],
                swap_melody=True, swap_harmony=True)

    return c1, c2


# ---------------------------------------------------------------------------
# NSGA-II algorithm
# ---------------------------------------------------------------------------

class NSGAII:
    """NSGA-II optimisation for narrative music composition.

    Operates on Individuals containing SectionGenomes with harmony trees
    and melody genes.  Three objectives are minimised simultaneously.
    """

    def __init__(
        self,
        sections: list,
        narrative_data: dict,
        motifs: List[Motif],
        section_tempo_infos: List[dict],
        objectives: List[Callable],
        population_size: int = NSGA_POPULATION_SIZE,
        n_generations: int   = NSGA_N_GENERATIONS,
        crossover_rate: float = NSGA_CROSSOVER_RATE,
        mutation_rate: float  = NSGA_MUTATION_RATE,
    ):
        self.sections = sections
        self.narrative_data = narrative_data
        self.motifs = motifs
        self.section_tempo_infos = section_tempo_infos
        self.objectives = objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.grammar = HarmonyGrammar()

    def evaluate(self, individual: Individual) -> None:
        """Render and evaluate all objectives."""
        arrangements = render_individual(
            individual, self.sections, self.narrative_data,
            self.motifs, self.section_tempo_infos,
        )
        individual.objectives = [
            obj(arrangements, self.narrative_data, self.motifs)
            for obj in self.objectives
        ]

    def fast_non_dominated_sort(
        self, population: List[Individual],
    ) -> List[List[Individual]]:
        """Sort into Pareto fronts. O(n²) per objective."""
        n = len(population)
        dom_counts = [0] * n
        dominated_by: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]  # store indices, not objects

        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_by[i].append(j)
                    dom_counts[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_by[j].append(i)
                    dom_counts[i] += 1

            if dom_counts[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)

        fi = 0
        while fronts[fi]:
            next_front: List[int] = []
            for p_idx in fronts[fi]:
                for q_idx in dominated_by[p_idx]:
                    dom_counts[q_idx] -= 1
                    if dom_counts[q_idx] == 0:
                        population[q_idx].rank = fi + 1
                        next_front.append(q_idx)
            fi += 1
            fronts.append(next_front)

        # Convert index fronts to object fronts
        return [[population[i] for i in front] for front in fronts[:-1]]

    def crowding_distance_sort(
        self, front: List[Individual],
    ) -> List[Individual]:
        """Calculate crowding distance and sort descending."""
        if len(front) <= 2:
            return front

        for ind in front:
            ind.crowding_distance = 0.0

        n_obj = len(self.objectives)
        for oi in range(n_obj):
            front.sort(key=lambda x: x.objectives[oi])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            obj_range = front[-1].objectives[oi] - front[0].objectives[oi]
            if obj_range == 0:
                continue

            for k in range(1, len(front) - 1):
                front[k].crowding_distance += (
                    front[k+1].objectives[oi] - front[k-1].objectives[oi]
                ) / obj_range

        return sorted(front, key=lambda x: x.crowding_distance, reverse=True)

    def tournament_selection(
        self, population: List[Individual],
    ) -> List[Individual]:
        """Binary tournament selection."""
        selected: List[Individual] = []
        while len(selected) < self.population_size:
            i1, i2 = random.sample(range(len(population)), 2)
            p1, p2 = population[i1], population[i2]
            if p1.rank < p2.rank:
                selected.append(p1)
            elif p2.rank < p1.rank:
                selected.append(p2)
            elif p1.crowding_distance > p2.crowding_distance:
                selected.append(p1)
            else:
                selected.append(p2)
        return selected

    def evolve(
        self,
        initial_population: Optional[List[Individual]] = None,
        targets: Optional[List[float]] = None
    ) -> Tuple[List[Individual], Dict]:
        """Main NSGA-II evolution loop.

        Returns (final_population, tracking_data).
        """
        tracking = {
            "objective_values": [],
            "pareto_front_sizes": [],
        }

        # Initialise
        if initial_population is None:
            population = create_initial_population(
                self.sections, self.narrative_data, self.motifs,
                self.section_tempo_infos, self.grammar,
                self.population_size,
            )
        else:
            population = initial_population

        # Evaluate initial population
        for ind in population:
            self.evaluate(ind)

        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.crowding_distance_sort(front)

        # Evolution
        for gen in range(self.n_generations):
            if gen % 10 == 0 or gen == self.n_generations - 1:
                print(f"Generation {gen + 1}/{self.n_generations}", end="")

            parents = self.tournament_selection(population)
            offspring: List[Individual] = []

            for i in range(0, len(parents), 2):
                if i + 1 >= len(parents):
                    break

                p1, p2 = parents[i], parents[i + 1]

                if random.random() < self.crossover_rate:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if random.random() < self.mutation_rate:
                    c1 = mutate(c1, self.sections, self.grammar)
                if random.random() < self.mutation_rate:
                    c2 = mutate(c2, self.sections, self.grammar)

                offspring.extend([c1, c2])

            # Evaluate offspring
            for ind in offspring:
                self.evaluate(ind)

            # Survivor selection
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)

            new_population: List[Individual] = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    remaining = self.population_size - len(new_population)
                    sorted_front = self.crowding_distance_sort(front)
                    new_population.extend(sorted_front[:remaining])
                    break

            population = new_population
            self._update_tracking(population, tracking)

            # Print generation summary
            if population and population[0].objectives:
                obj_arr = np.array([ind.objectives for ind in population])
                avgs = np.mean(obj_arr, axis=0)
                if gen % 10 == 0:
                    if len(self.objectives) == 1:
                        print(f"  avg: [{avgs[0]:.3f}]")
                        if targets != None and targets[0] >= avgs[0]:
                            return population, tracking
                    elif len(self.objectives) == 2:
                        print(f"  avg: [{avgs[0]:.3f}, {avgs[1]:.3f}]")
                        if targets != None and targets[0] >= avgs[0] and targets[1] >= avgs[1]:
                            return population, tracking
                    elif len(self.objectives) == 3:
                        print(f"  avg: [{avgs[0]:.3f}, {avgs[1]:.3f}, {avgs[2]:.3f}]")
                        if targets != None and targets[0] >= avgs[0] and targets[1] >= avgs[1] and targets[2] >= avgs[2]:
                            return population, tracking

            else:
                if gen % 10 == 0:
                    print()

        if population and population[0].objectives: 
            if gen == self.n_generations - 1:    
                if len(self.objectives) == 1:
                    print(f"  avg: [{avgs[0]:.3f}]")
                elif len(self.objectives) == 2:
                    print(f"  avg: [{avgs[0]:.3f}, {avgs[1]:.3f}]")
                elif len(self.objectives) == 3:
                    print(f"  avg: [{avgs[0]:.3f}, {avgs[1]:.3f}, {avgs[2]:.3f}]")
                        
        return population, tracking

    def _update_tracking(
        self, population: List[Individual], tracking: Dict,
    ) -> None:
        obj_arr = np.array([ind.objectives for ind in population
                            if ind.objectives])
        if len(obj_arr) == 0:
            return

        tracking["objective_values"].append({
            "min": np.min(obj_arr, axis=0).tolist(),
            "max": np.max(obj_arr, axis=0).tolist(),
            "avg": np.mean(obj_arr, axis=0).tolist(),
            "std": np.std(obj_arr, axis=0).tolist(),
        })

        fronts = self.fast_non_dominated_sort(population)
        tracking["pareto_front_sizes"].append([len(f) for f in fronts])

    def get_pareto_front(
        self, population: List[Individual],
    ) -> List[Individual]:
        """Return the first Pareto front."""
        fronts = self.fast_non_dominated_sort(population)
        return fronts[0] if fronts else []


# ---------------------------------------------------------------------------
# Selection utilities
# ---------------------------------------------------------------------------

def select_by_weights(
    front: List[Individual],
    weights: List[float],
) -> Individual:
    """Select the solution closest to a weighted ideal."""
    w = np.array(weights) / sum(weights)
    scores = [sum(wi * obj for wi, obj in zip(w, ind.objectives))
              for ind in front]
    return front[int(np.argmin(scores))]


def select_knee_point(front: List[Individual]) -> Individual:
    """Select the solution with greatest marginal improvement (knee)."""
    obj_values = np.array([ind.objectives for ind in front])
    nadir = np.max(obj_values, axis=0)
    distances = np.linalg.norm(nadir - obj_values, axis=1)
    return front[int(np.argmax(distances))]