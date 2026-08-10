# Narrative Suspense as Musical Tension: Composing Film Music From Video

Code and supplementary material for the ACM Multimedia 2026 paper.

**[Generated scores and demo →](https://USER.github.io/REPO)** · [Paper](https://doi.org/10.1145/3767308.3835878)

Francisco Braga, Nuno Correia, Roger B. Dannenberg, Gilberto Bernardes

---

## What this does

Most systems that generate music for video bridge the two through emotion, usually valence and arousal. Those representations describe how a scene feels but discard the temporal and causal structure of the narrative.

This system uses **narrative suspense** and **musical tension** as the bridge instead. It extracts narrative structure from a film — characters, story threads, events — using a multimodal language model, computes a per-event suspense curve with an adaptation of the Doust and Piwek model, composes symbolic music from that structure, and aligns the music's tension with the suspense curve through multi-objective evolutionary optimisation (NSGA-II).

A within-subjects listening study with 83 participants found that tension-aligned scores were rated significantly higher on music–narrative fit than an ablation with the alignment objective removed (OR = 2.26–2.45, all *p* < 0.0001), with no difference in musical pleasantness.

## Layout

```
config.py              all model and search parameters
main.py                entry point — generates a score from a narrative structure

narrative/
  extraction.py        two-pass narrative extraction from video
  prompts.py           extraction prompts
  schema.py            structured output schema
  suspense.py          adapted Doust & Piwek suspense model
  character.py         character valence and activity profiles
  video_annotator.py   visual activity annotation
  export_selection.py  excerpt selection

music/
  sections.py          event grouping, key assignment
  motif.py             character motif generation
  harmony.py           Rohrmeier grammar, section tempo
  tempo.py             activity-to-tempo mapping, bar allocation
  melody.py            melody cell rendering
  initialiser.py       initial population
  genome.py            genome representation
  operators.py         mutation and crossover
  objectives.py        the three NSGA-II objectives
  tension_ribbons.py   Spiral Array tonal tension
  voicing.py           four-part SATB voicing
  renderer.py          genome to notes
  nsga2.py             evolutionary loop

evaluation/
  analysis.py          study analysis: CLMM, Wilcoxon, Cliff's delta
  eval_*.csv           anonymised study responses
  analysis_output/     generated figures

output/
  spring/  easy_street/  metropolis/
    narrative.json     extracted narrative structure
    suspense.json      per-event suspense curve
    midi/              generated scores, full system and ablation
  global_suspense.png  suspense curves for all three films (Figure 2)

docs/                  demo site
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Rendering MIDI to audio needs a piano sample library. The study used Spitfire Audio LABS Intimate Grand Piano.

## Generating a score

Narrative extraction depends on a proprietary multimodal model, so that stage is not reproducible from source alone. The extracted structures are committed for all three films, which means **everything downstream of extraction runs without an API key**:

```bash
python main.py --film output/metropolis     # both conditions
python main.py --all                        # all three study films
python main.py --film output/spring --condition full
```

Each run evolves a population under two objective sets — the full system (tonal coherence, motif recognition, tension alignment) and the ablation (tonal coherence and motif recognition only) — and writes the knee-point solution of each to `output/<film>/midi/`.

Motifs, sections, keys, tempo and harmonic rhythm are computed once per film and shared by both conditions, so any difference between them comes from the tension alignment objective alone. Both runs are seeded (`SEED = 42`) and reproduce exactly.

Search parameters live in `config.py`. The study used a population of 200 over 200 generations, with crossover 0.9 and mutation 0.3.

## Running extraction on a new video

Extraction needs a Gemini API key in `api.py`, which is gitignored:

```python
# api.py
GEMINI_API_KEY = "..."
```

The two-pass process identifies characters and narrative threads, then extracts a chronological event sequence with timestamps, thread activity, turning points and visual activity levels. Extracted structures should be checked before use — the paper reports that all three were verified by the authors for factual accuracy, temporal consistency and thread coherence.

## Study data and analysis

`evaluation/` holds the anonymised responses and the analysis that produces the results in the paper: cumulative link mixed models with condition, film, their interaction and form as fixed effects and participant as a random intercept, Wilcoxon signed-rank tests on per-participant medians, Krippendorff's α, and per-film Cliff's delta.

```bash
python evaluation/analysis.py
```

## Citation

```bibtex
@inproceedings{braga2026narrative,
  title     = {Narrative Suspense as Musical Tension: Composing Film Music From Video},
  author    = {Braga, Francisco and Correia, Nuno and Dannenberg, Roger B. and Bernardes, Gilberto},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3767308.3835878}
}
```

## License

Code released under the MIT License.

*Spring* © Blender Foundation, released under CC BY 4.0. *Easy Street* (1917) is in the public domain. Film excerpts are reproduced for research demonstration.