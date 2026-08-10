from .motif import (
    generate_motif,
    generate_all_motifs,
)
from .structures import Key, Note, Motif, Chord, ChordProgression, SCALE_DEGREES_KIND_MAJOR, SCALE_DEGREES_KIND_MINOR, group_chords_by_event
from .motif import generate_motif, generate_all_motifs, SCALES
from .harmony import HarmonyGrammar, Symbol, compute_section_tempo
from .nsga2 import NSGAII, create_initial_population, select_knee_point, select_by_weights
from .objectives import tonal_incoherence, motif_recognition, tension_misalignment
from .sections import detect_music_sections, assign_keys, print_sections, Section