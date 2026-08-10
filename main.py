"""
Generate a score for a film excerpt from its extracted narrative structure.

Runs the NSGA-II optimisation twice per film: once with the tension alignment
objective (full system) and once without it (ablation), matching the two
conditions compared in the listening study.

    python main.py --film output/spring
    python main.py --all
"""

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import music21
from music21 import stream, note, meter
from music21 import tempo as m21_tempo

import config
from music.harmony import HarmonyGrammar, compute_section_tempo
from music.motif import generate_motif
from music.nsga2 import (
    NSGAII,
    create_initial_population,
    render_individual,
    select_knee_point,
)
from music.objectives import (
    motif_recognition,
    tension_misalignment,
    tonal_incoherence,
)
from music.sections import assign_keys, detect_music_sections
from music.tempo import assign_bars_to_events
from narrative.character import compute_character_profiles

FILMS = ["output/spring", "output/easy_street", "output/metropolis"]

CONDITIONS = {
    "full": [tonal_incoherence, motif_recognition, tension_misalignment],
    "ablation": [tonal_incoherence, motif_recognition],
}

SEED = 42


def load_narrative(film_dir):
    """Load the extracted narrative structure and its suspense curve."""
    with open(os.path.join(film_dir, "narrative.json")) as f:
        narrative_data = json.load(f)
    with open(os.path.join(film_dir, "suspense.json")) as f:
        suspense_results = json.load(f)
    print(
        f"{len(narrative_data['events'])} events, "
        f"{len(narrative_data['characters'])} characters, "
        f"suspense for {len(suspense_results)} events"
    )
    return narrative_data, suspense_results


def build_sections(narrative_data, suspense_results, initial_key=None):
    """Group events into music and silence sections and assign keys."""
    sections = detect_music_sections(suspense_results, narrative_data)
    sections = assign_keys(sections, narrative_data, initial_key=initial_key)
    n_music = sum(1 for s in sections if s.is_music)
    print(f"{len(sections)} sections ({n_music} music, {len(sections) - n_music} silence)")
    return sections


def build_motifs(narrative_data, sigma=3, tonal=True, seed=None):
    """Generate one motif per character from their narrative profile.

    Motifs are built once per film and shared by both conditions, so any
    difference between conditions cannot come from motif material.
    """
    motifs = []
    for i, profile in enumerate(compute_character_profiles(narrative_data)):
        motifs.append(
            generate_motif(
                character_id=profile["character_id"],
                character_name=profile["name"],
                valence=profile["valence"],
                avg_activity=profile["avg_activity"],
                sigma=sigma,
                tonal=tonal,
                scale_pitches=None,
                seed=(seed + i) if seed is not None else None,
            )
        )
    return motifs


def build_tempo_infos(sections, narrative_data):
    """Compute tempo and bar allocation for every section."""
    infos = []
    for section in sections:
        if section.is_silence:
            infos.append(
                {
                    "bpm": 120,
                    "bars": 0,
                    "bar_duration": 0.5,
                    "used_duration": 0.0,
                    "remaining": 0.0,
                    "time_signature": (4, 4),
                    "in_range": True,
                }
            )
        else:
            info = compute_section_tempo(section, narrative_data)
            infos.append(info)
            assign_bars_to_events(section, info)
    return infos


def arrangements_to_score(arrangements, sections, tempo_infos):
    """Lay the per-section arrangements onto a single four-part timeline.

    Everything is placed at an absolute beat offset so that section tempo
    changes do not accumulate rounding drift over a long piece.
    """
    parts = {
        name: stream.Part(id=name) for name in ("Melody", "Alto", "Tenor", "Bass")
    }
    melody = parts["Melody"]
    for part in parts.values():
        part.insert(0, meter.TimeSignature("4/4"))

    prev_tempo = 120.0
    offset = 0.0

    for idx, section in enumerate(sections):
        arrangement = arrangements.get(idx) if section.is_music else None

        if arrangement is None:
            # Silence section, or a music section that produced no arrangement.
            duration = section.end_time - section.start_time
            beats = duration * (prev_tempo / 60.0)
            melody.insert(offset, m21_tempo.MetronomeMark(number=prev_tempo))
            for part in parts.values():
                rest = note.Rest()
                rest.quarterLength = max(beats, 0.001)
                part.insert(offset, rest)
            offset += beats
            continue

        prev_tempo = arrangement.tempo_bpm
        section_beats = arrangement.duration_beats
        remaining_beats = tempo_infos[idx]["remaining"] * arrangement.tempo_bpm / 60.0
        melody.insert(offset, m21_tempo.MetronomeMark(number=arrangement.tempo_bpm))

        beat = offset
        for mn in arrangement.melody_notes:
            if mn.velocity == 0:
                element = note.Rest()
            else:
                element = note.Note(mn.pitch)
                element.volume.velocity = mn.velocity
            element.quarterLength = mn.duration
            melody.insert(beat, element)
            beat += mn.duration

        # Pad the melody out to the harmonic boundary.
        if beat < offset + section_beats - 1e-6:
            rest = note.Rest()
            rest.quarterLength = (offset + section_beats) - beat
            melody.insert(beat, rest)

        beat = offset
        for chord in arrangement.harmonic_chords:
            for part_name, voiced in (
                ("Alto", chord.alto),
                ("Tenor", chord.tenor),
                ("Bass", chord.bass),
            ):
                element = note.Note(voiced.pitch)
                element.quarterLength = chord.chord.duration
                element.volume.velocity = voiced.velocity
                parts[part_name].insert(beat, element)
            beat += chord.chord.duration

        # Sub-bar gap up to the next section boundary.
        if remaining_beats > 1e-6:
            for part in parts.values():
                rest = note.Rest()
                rest.quarterLength = remaining_beats
                part.insert(offset + section_beats, rest)

        offset += section_beats + remaining_beats

    score = stream.Score(list(parts.values()))
    score.metadata = music21.metadata.Metadata()
    score.metadata.title = "Narrative Music"
    return score


def run_condition(name, objectives, sections, narrative_data, motifs, tempo_infos, out_dir):
    """Evolve a population under one objective set and write the result to MIDI."""
    random.seed(SEED)
    np.random.seed(SEED)

    population = create_initial_population(
        sections,
        narrative_data,
        motifs,
        tempo_infos,
        HarmonyGrammar(),
        population_size=config.NSGA_POPULATION_SIZE,
    )

    nsga = NSGAII(
        sections=sections,
        narrative_data=narrative_data,
        motifs=motifs,
        section_tempo_infos=tempo_infos,
        objectives=objectives,
        population_size=config.NSGA_POPULATION_SIZE,
        n_generations=config.NSGA_N_GENERATIONS,
        crossover_rate=config.NSGA_CROSSOVER_RATE,
        mutation_rate=config.NSGA_MUTATION_RATE,
    )

    start = time.perf_counter()
    final_population, _tracking = nsga.evolve(population)
    elapsed = time.perf_counter() - start

    pareto = nsga.get_pareto_front(final_population)
    best = select_knee_point(pareto)

    labels = ["tonal", "motif", "tension"]
    scores = "  ".join(
        f"{label}={value:.4f}" for label, value in zip(labels, best.objectives)
    )
    print(
        f"  [{name}] {elapsed:.0f}s, "
        f"{len(pareto)} on the Pareto front, knee point: {scores}"
    )

    arrangements = render_individual(
        best, sections, narrative_data, motifs, tempo_infos, apply_elaboration=False
    )
    score = arrangements_to_score(arrangements, sections, tempo_infos)

    midi_dir = os.path.join(out_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(midi_dir, f"{name}_{stamp}.mid")
    score.write("midi", fp=path)
    print(f"  [{name}] wrote {path}")
    return path


def process_film(film_dir, conditions):
    print(f"\n=== {film_dir} ===")
    narrative_data, suspense_results = load_narrative(film_dir)
    sections = build_sections(narrative_data, suspense_results)
    motifs = build_motifs(narrative_data)
    tempo_infos = build_tempo_infos(sections, narrative_data)

    for name in conditions:
        run_condition(
            name,
            CONDITIONS[name],
            sections,
            narrative_data,
            motifs,
            tempo_infos,
            film_dir,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--film", help="directory holding narrative.json and suspense.json")
    group.add_argument("--all", action="store_true", help="run all three study films")
    parser.add_argument(
        "--condition",
        choices=["full", "ablation", "both"],
        default="both",
        help="which objective set to run (default: both)",
    )
    args = parser.parse_args()

    conditions = ["full", "ablation"] if args.condition == "both" else [args.condition]
    for film_dir in (FILMS if args.all else [args.film]):
        process_film(film_dir, conditions)


if __name__ == "__main__":
    main()