# ============================================================
# config.py
# Non-secret configuration — safe to commit to git
# API keys go in api.py (gitignored)
# ============================================================

# --- Gemini model ---
GEMINI_MODEL = "gemini-3-flash-preview"

# --- Video upload ---
# Frame rate for video upload to Gemini (frames per second)
# Higher = better temporal resolution but slower processing
# Current version is always 1 fps
# EXTRACTION_FPS = 1

# --- Event extraction ---
EVENT_MIN_DURATION = 3   # seconds
EVENT_MAX_DURATION = 15  # seconds

# --- Doust & Piwek suspense model parameters ---
# Weight of completion imminence vs interruption imminence
# Original paper: rho = 0.7
RHO = 0.7

# Confidence formula scaling factor
# Original paper: phi = 1.5
PHI = 1.5

# Foregroundedness decay per step
# Original paper: beta = 0.88
BETA = 0.88

# --- Turning point boost (our adaptation, not in original model) ---
# Multiplicative boost applied to completion imminence at turning points
TURNING_POINT_BOOST = 1.3

# Only apply boost when imminence is below this threshold
# Prevents artificially hitting ceiling when thread is already near completion
TURNING_POINT_BOOST_THRESHOLD = 0.8

# --- Visualization ---
PLOT_OUTPUT_PATH = "suspense_analysis.png"
PLOT_DPI = 150

# --- Harmonic rhythm ---
# Chord duration range in bars (mapped from suspense)
# low suspense  → HARMONIC_RHYTHM_MAX_BARS per chord
# high suspense → HARMONIC_RHYTHM_MIN_BARS per chord
HARMONIC_RHYTHM_MIN_BARS = 0.125
HARMONIC_RHYTHM_MAX_BARS = 1.0

# --- Tempo ---
# Mapping from visual activity values to (bpm_center, bpm_std_dev)
# Activity values: 0.1=static, 0.3=low, 0.5=moderate, 0.7=high, 0.9=frantic
ACTIVITY_TEMPO = {
    0.1: (52,  8),   # static:   Largo
    0.3: (72,  8),   # low:      Adagio/Andante
    0.5: (96,  10),  # moderate: Andante/Moderato
    0.7: (120, 12),  # high:     Allegro
    0.9: (152, 15),  # frantic:  Vivace/Presto
}

# Minimum and maximum allowable tempo in BPM
TEMPO_MIN_BPM = 40
TEMPO_MAX_BPM = 200

# --- Section transitions ---
# Duration of silence between music sections in seconds
SECTION_TRANSITION_SECONDS = 2.0

# ---------------------------------------------------------------------------
# NSGA-II optimisation
# ---------------------------------------------------------------------------

NSGA_POPULATION_SIZE = 100
NSGA_N_GENERATIONS   = 100
NSGA_CROSSOVER_RATE  = 0.9

# Crossover type probabilities (must sum to 1.0)
NSGA_CROSSOVER_TYPE_PROBS = {
    "melody":         1/3,
    "harmony":        1/3,
    "melody_harmony": 1/3,
}

# Overall mutation probability per individual per generation
NSGA_MUTATION_RATE = 0.2

# Mutation type probabilities (must sum to 1.0)
NSGA_MUTATION_MELODY_PARAM_PROB    = 0.30
NSGA_MUTATION_MELODY_TRANSFORM_PROB = 0.10
NSGA_MUTATION_HARMONY_PROB          = 0.60   # split equally between parallel and modulation
NSGA_MUTATION_VELOCITY_LEVEL_PROB   = 0.15   # nudge one event's velocity
NSGA_MUTATION_VELOCITY_CONTOUR_PROB = 0.10   # ramp velocity across a whole section

# Velocity bounds and step size
VELOCITY_MIN  = 60    # pp — minimum rendered velocity
VELOCITY_MAX  = 110   # ff — maximum rendered velocity
VELOCITY_STEP = 2    # nudge amount for level mutation

# ---------------------------------------------------------------------------
# Objective function parameters
# ---------------------------------------------------------------------------

# tonal_incoherence: penalty weights for melody notes outside chord/scale
TONAL_INCOHERENCE_SCALE_PENALTY   = 0.5   # note in key but not in chord triad
TONAL_INCOHERENCE_OUTSIDE_PENALTY = 1.0   # note outside key entirely

# motif_recognition: sliding window similarity weights
MOTIF_RECOGNITION_INTERVAL_WEIGHT = 0.5   # interval contour sign match fraction
MOTIF_RECOGNITION_RHYTHM_WEIGHT   = 0.5   # normalised duration ratio similarity

# tension_misalignment: five-feature composite (Farbood 2012, Table 1)
# Weights represent the relative contribution of each perceptual feature
# to perceived musical tension.  Loudness and pitch height are the strongest
# predictors; tempo and onset frequency are moderate; tonal tension (spiral
# array) is the weakest individual predictor but combines with all others.
TENSION_FARBOOD_LOUDNESS = 3   # mean bar velocity → perceived loudness
TENSION_FARBOOD_PITCH    = 3   # mean melody pitch height
TENSION_FARBOOD_TEMPO    = 2   # section tempo (fixed per section)
TENSION_FARBOOD_ONSET    = 2   # note onsets per beat → rhythmic density
TENSION_FARBOOD_TONAL    = 1   # tonal tension from spiral array

# Trend amplification: when slope continues in same direction as
# previous bar, multiply by this factor (Farbood used 5 at 250ms;
# we use 3 at bar-level resolution).
TENSION_TREND_BETA = 3.0

# Minimum event-level std for a feature to be included in correlation.
# Below this threshold, the feature is too flat to produce reliable
# Pearson values, so we skip it rather than penalise.
TENSION_MIN_FEATURE_STD = 0.01

# Balance between absolute level correlation and slope (transition)
# correlation. 1.0 = pure absolute, 0.0 = pure slope.
TENSION_ABSOLUTE_WEIGHT = 1.0

# Intra-harmonic weights for the spiral array sub-components.
# Equal weights (1/3 each) — no empirical decomposition available in
# the literature separating strain, diameter, and momentum contributions.
TENSION_WEIGHT_STRAIN   = 1 / 3   # tensile_strain   (distance from key centre)
TENSION_WEIGHT_DIAMETER = 1 / 3   # cloud_diameter   (dissonance within bar)
TENSION_WEIGHT_MOMENTUM = 1 / 3   # cloud_momentum   (harmonic change rate)

# ---------------------------------------------------------------------------
# Melody v2 — skeleton
# ---------------------------------------------------------------------------
SKELETON_BASE_STEP       = 2      # semitones — default inter-chord step
SKELETON_SUSPENSE_WEIGHT = 3.0    # suspense delta amplifier for pitch contour
SKELETON_REGISTER_PULL   = 0.1    # gravity toward register center (0.0–1.0)

# ---------------------------------------------------------------------------
# Melody v2 — cell rendering
# ---------------------------------------------------------------------------
CELL_FOREIGN_INTERVAL_MAX  = 0.3  # max foreign_interval_prob gene value
CELL_ANCHOR_FREQ_MIN       = 2    # development cell anchoring bounds
CELL_ANCHOR_FREQ_MAX       = 6
CELL_STATEMENT_MIN_LENGTH  = 3    # motif notes in a statement cell
CELL_STATEMENT_MAX_LENGTH  = 8
CELL_SEQUENCE_MAX_INTERVAL = 4    # max semitones between sequence repetitions

# ---------------------------------------------------------------------------
# Melody v2 — elaboration
# ---------------------------------------------------------------------------
ELABORATION_MIN_NOTE_DUR = 2.0    # only elaborate notes >= this (beats)

# ---------------------------------------------------------------------------
# Melody v2 — updated motif recognition weights
# ---------------------------------------------------------------------------
MOTIF_RECOGNITION_CONTOUR_WEIGHT   = 0.35
MOTIF_RECOGNITION_RHYTHM_WEIGHT    = 0.35
MOTIF_RECOGNITION_MAGNITUDE_WEIGHT = 0.30

# ---------------------------------------------------------------------------
# Melody v2 — NSGA-II mutation probabilities
# ---------------------------------------------------------------------------
NSGA_MUT_CELL_TYPE_PROB       = 0.20
NSGA_MUT_CONTOUR_PROB         = 0.10
NSGA_MUT_REGISTER_PROB        = 0.08
NSGA_MUT_DEV_PARAM_PROB       = 0.12
NSGA_MUT_STATEMENT_PARAM_PROB = 0.08
NSGA_MUT_SEQUENCE_PARAM_PROB  = 0.07
NSGA_MUT_ORNAMENT_PROB        = 0.05
# Harmony and velocity probs remain as-is (0.20 total harmony, 0.10 velocity)