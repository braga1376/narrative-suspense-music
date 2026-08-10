"""
music/melody.py

Motif-derived continuous melody construction (bar-level cells).

Three-layer architecture:
    Layer 1: Skeleton      — chord-tone backbone (one target per chord)
    Layer 2: Cell Rendering — motif-derived content (one cell per BAR)
    Layer 3: Elaboration    — subdivision and ornamentation

Bar-level cell granularity ensures every cell gets a full metric unit
(~4 beats in 4/4) regardless of harmonic rhythm.  Fast harmonic rhythm
(2+ chords/bar) gives the cell multiple harmonic contexts to react to.
Slow harmonic rhythm (chord spanning 2+ bars) gives multiple cells over
one harmonic context, enabling variety.

Cell types:
    statement    — recognisable motif fragment (high recognition)
    development  — new material from motif interval vocabulary (medium)
    sequence     — motif fragment restated at different pitch levels (medium-high)
    liquidation  — motif material progressively simplified (transitional)
    passage      — pure voice-leading between chord tones (low recognition)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set

import numpy as np

from .structures import Key, Chord, ChordProgression, Motif, Note, group_chords_by_event
from .voicing import (
    VoicedNote, Voice, VOICE_RANGES, VOICE_VELOCITY_OFFSETS,
    get_chord_pitches,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKELETON_BASE_STEP       = 2
SKELETON_SUSPENSE_WEIGHT = 3.0
# SKELETON_SUSPENSE_WEIGHT = 0.0
SKELETON_REGISTER_PULL   = 0.1
REGISTER_CENTERS         = {0: 58, 1: 67, 2: 76}

CELL_FOREIGN_INTERVAL_MAX  = 0.3
CELL_ANCHOR_FREQ_MIN       = 2
CELL_ANCHOR_FREQ_MAX       = 6
CELL_STATEMENT_MIN_LENGTH  = 3
CELL_STATEMENT_MAX_LENGTH  = 8
CELL_SEQUENCE_MAX_INTERVAL = 4
MAX_INTERVAL_INDICES       = 12

ELABORATION_MIN_NOTE_DUR = 2.0
MIN_NOTE_DURATION = 0.25
BEATS_PER_BAR     = 4

CELL_TYPES          = ["statement", "development", "sequence", "liquidation", "passage"]
CONTOUR_DIRECTIONS  = ["ascending", "descending", "arch", "valley"]
APPROACH_TYPES      = ["stepwise", "neighbor", "arpeggio"]
RHYTHM_VARIANTS     = [0, 1, 2, 3]
STRIP_ORDERS        = ["rhythm_first", "pitch_first"]
MOTIF_TRANSFORMS    = ["invert", "retrograde", "augment", "diminish", None]

MELODY_LOW, MELODY_HIGH = VOICE_RANGES[Voice.MELODY]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellGenes:
    """Genes for one BAR's melodic cell.  All fields have defaults so
    dormant genes are always present and valid."""

    cell_type: str = "passage"

    # Statement
    motif_start_note: int        = 0
    statement_length: int        = 4
    motif_transform: Optional[str] = None

    # Development
    interval_indices: List[int]  = field(default_factory=lambda: [0] * MAX_INTERVAL_INDICES)
    rhythm_variant: int          = 0
    anchor_frequency: int        = 4
    foreign_interval_prob: float = 0.1

    # Sequence
    sequence_interval: int       = 2
    is_tonal_sequence: bool      = True

    # Liquidation
    simplification_rate: int     = 4
    strip_order: str             = "pitch_first"

    # Passage
    approach_type: str           = "stepwise"
    rhythmic_density: float      = 0.5

    # Elaboration
    ornament_probability: float  = 0.3

    def copy(self) -> CellGenes:
        return CellGenes(
            cell_type=self.cell_type,
            motif_start_note=self.motif_start_note,
            statement_length=self.statement_length,
            motif_transform=self.motif_transform,
            interval_indices=self.interval_indices.copy(),
            rhythm_variant=self.rhythm_variant,
            anchor_frequency=self.anchor_frequency,
            foreign_interval_prob=self.foreign_interval_prob,
            sequence_interval=self.sequence_interval,
            is_tonal_sequence=self.is_tonal_sequence,
            simplification_rate=self.simplification_rate,
            strip_order=self.strip_order,
            approach_type=self.approach_type,
            rhythmic_density=self.rhythmic_density,
            ornament_probability=self.ornament_probability,
        )


@dataclass
class MelodyGenes:
    """All melody genes for one narrative event.
    cell_genes has one CellGenes per BAR in the event."""

    target_register: int     = 1
    contour_direction: str   = "arch"
    cell_genes: List[CellGenes] = field(default_factory=list)
    event_velocity: int      = 80

    def copy(self) -> MelodyGenes:
        return MelodyGenes(
            target_register=self.target_register,
            contour_direction=self.contour_direction,
            cell_genes=[cg.copy() for cg in self.cell_genes],
            event_velocity=self.event_velocity,
        )


@dataclass
class MotifData:
    """Cached properties extracted from a Motif."""
    intervals: List[int]
    interval_pool: List[int]
    durations: List[float]
    notes: List[Note]


# A chord slice within a bar: (Chord, duration_in_this_bar, skeleton_target)
BarChord = Tuple[Chord, float, int]


# ---------------------------------------------------------------------------
# Motif helpers
# ---------------------------------------------------------------------------

def extract_motif_data(motif: Motif) -> MotifData:
    notes = motif.notes
    intervals = [notes[i+1].pitch - notes[i].pitch for i in range(len(notes)-1)]
    seen: Set[int] = set()
    pool: List[int] = []
    for iv in intervals:
        if iv not in seen:
            seen.add(iv)
            pool.append(iv)
    pool.sort(key=lambda x: (abs(x), x))
    return MotifData(
        intervals=intervals or [0],
        interval_pool=pool or [0],
        durations=[n.duration for n in notes] or [1.0],
        notes=notes,
    )


def apply_motif_transform(motif: Motif, transform: Optional[str]) -> Motif:
    if transform is None or not motif.notes:
        return motif
    notes = motif.notes
    if transform == "invert":
        axis = notes[0].pitch
        nn = [Note(max(MELODY_LOW, min(MELODY_HIGH, axis+(axis-n.pitch))),
                   n.duration, n.velocity) for n in notes]
    elif transform == "retrograde":
        nn = [Note(n.pitch, n.duration, n.velocity) for n in reversed(notes)]
    elif transform == "augment":
        total = sum(n.duration for n in notes)
        nn, used = [], 0.0
        for n in notes:
            if used >= total: break
            nn.append(Note(n.pitch, min(n.duration*2, total-used), n.velocity))
            used += nn[-1].duration
    elif transform == "diminish":
        total = sum(n.duration for n in notes)
        nn, used, idx = [], 0.0, 0
        while used < total:
            n = notes[idx % len(notes)]
            d = min(n.duration*0.5, total-used)
            if d < MIN_NOTE_DURATION: break
            nn.append(Note(n.pitch, d, n.velocity)); used += d; idx += 1
    else:
        return motif
    return Motif(notes=nn, character_id=motif.character_id,
                 character_name=motif.character_name, valence=motif.valence,
                 avg_activity=motif.avg_activity, tonal=motif.tonal, key=motif.key)


# ---------------------------------------------------------------------------
# Pitch / note utilities
# ---------------------------------------------------------------------------

def _chord_tones_in_range(chord: Chord) -> List[int]:
    pcs = get_chord_pitches(chord)
    pitches: List[int] = []
    for pc in pcs:
        p = pc
        while p < MELODY_LOW: p += 12
        while p <= MELODY_HIGH: pitches.append(p); p += 12
    return sorted(set(pitches)) or [REGISTER_CENTERS[1]]


def _closest_to(target: int, candidates: List[int]) -> int:
    return min(candidates, key=lambda p: abs(p-target)) if candidates else max(MELODY_LOW, min(MELODY_HIGH, target))


def _clamp(pitch: int) -> int:
    return max(MELODY_LOW, min(MELODY_HIGH, pitch))


def _next_scale_tone(pitch: int, direction: int, key: Key) -> int:
    pcs = set(key.get_scale_degree().tolist())
    for s in range(1, 13):
        c = pitch + direction * s
        if c % 12 in pcs and MELODY_LOW <= c <= MELODY_HIGH:
            return c
    return _clamp(pitch + direction * 2)


def _snap_to_scale(pitch: int, key: Key) -> int:
    pcs = set(key.get_scale_degree().tolist())
    if pitch % 12 in pcs: return pitch
    for off in [1, -1, 2, -2]:
        c = pitch + off
        if c % 12 in pcs and MELODY_LOW <= c <= MELODY_HIGH: return c
    return pitch


def _mn(pitch: int, duration: float, velocity: int) -> VoicedNote:
    return VoicedNote(pitch=_clamp(pitch),
                      duration=max(MIN_NOTE_DURATION, round(duration, 6)),
                      velocity=max(0, min(127, velocity)),
                      voice=Voice.MELODY, is_chord_tone=False)


def _enforce_duration(notes: List[VoicedNote], budget: float,
                      fallback_pitch: int, vel: int) -> List[VoicedNote]:
    if budget <= 0: return []
    if not notes: return [_mn(fallback_pitch, budget, vel)]
    total = sum(n.duration for n in notes)
    if abs(total - budget) < 0.001: return notes
    if total > budget:
        out: List[VoicedNote] = []
        acc = 0.0
        for n in notes:
            rem = budget - acc
            if rem < MIN_NOTE_DURATION: break
            out.append(_mn(n.pitch, min(n.duration, rem), n.velocity))
            acc += out[-1].duration
        if not out: return [_mn(fallback_pitch, budget, vel)]
        gap = budget - sum(n.duration for n in out)
        if gap > 0.001:
            out[-1] = _mn(out[-1].pitch, out[-1].duration + gap, out[-1].velocity)
        return out
    notes[-1] = _mn(notes[-1].pitch, notes[-1].duration + budget - total, notes[-1].velocity)
    return notes


def _build_durations_from_pool(pool: List[float], variant: int,
                               budget: float) -> List[float]:
    if not pool: pool = [1.0]
    if   variant == 1: o = sorted(pool)
    elif variant == 2: o = sorted(pool, reverse=True)
    elif variant == 3: o = list(pool); random.shuffle(o)
    else:              o = list(pool)
    ds: List[float] = []
    t, i = 0.0, 0
    while t < budget - MIN_NOTE_DURATION:
        d = min(o[i % len(o)], budget - t)
        if d < MIN_NOTE_DURATION: break
        ds.append(d); t += d; i += 1
    gap = budget - sum(ds)
    if gap >= MIN_NOTE_DURATION: ds.append(gap)
    elif ds and gap > 0: ds[-1] += gap
    return ds


# ---------------------------------------------------------------------------
# Bar grouping
# ---------------------------------------------------------------------------

def _group_chords_into_bars(
    chords: List[Chord],
    skeleton: List[int],
    chord_start_idx: int,
    beats_per_bar: int = BEATS_PER_BAR,
) -> List[List[BarChord]]:
    """Group chords into bars.  A chord spanning a bar boundary is split."""
    bars: List[List[BarChord]] = []
    cur: List[BarChord] = []
    bar_beats = 0.0

    for i, chord in enumerate(chords):
        si = chord_start_idx + i
        st = skeleton[si] if si < len(skeleton) else 67
        rem = chord.duration

        while rem > MIN_NOTE_DURATION:
            space = beats_per_bar - bar_beats
            take = min(rem, space)
            if take < MIN_NOTE_DURATION: break
            cur.append((chord, take, st))
            bar_beats += take
            rem -= take
            if bar_beats >= beats_per_bar - 0.001:
                bars.append(cur); cur = []; bar_beats = 0.0

    if cur: bars.append(cur)
    return bars


def _chord_at_beat(bar_chords: List[BarChord], beat: float) -> Chord:
    acc = 0.0
    for chord, dur, _ in bar_chords:
        acc += dur
        if beat < acc - 0.001: return chord
    return bar_chords[-1][0]


def _bar_budget(bc: List[BarChord]) -> float:
    return sum(d for _, d, _ in bc)


def _bar_exit(bc: List[BarChord]) -> int:
    return bc[-1][2] if bc else 67


# ---------------------------------------------------------------------------
# Layer 1 — Skeleton (per-chord)
# ---------------------------------------------------------------------------

def build_skeleton(
    prog: ChordProgression,
    melody_genes: List[MelodyGenes],
    events_by_id: dict,
) -> List[int]:
    if not prog.chords: return []
    groups = group_chords_by_event(prog.chords)
    susp = {eid: float(e.get("global_suspense"))
            for eid, e in events_by_id.items()}
    skel: List[int] = []
    prev: Optional[int] = None
    prev_s = 0.5

    for gi, (eid, echords) in enumerate(groups):
        g = melody_genes[gi] if gi < len(melody_genes) else MelodyGenes()
        rc = REGISTER_CENTERS.get(g.target_register, REGISTER_CENTERS[1])
        cont = g.contour_direction
        es = susp.get(eid)
        sd = es - prev_s
        nc = len(echords)

        for cp, chord in enumerate(echords):
            ct = _chord_tones_in_range(chord)
            if prev is None:
                tgt = _closest_to(rc, ct)
            else:
                t = cp / (nc - 1) if nc > 1 else 0.5
                if   cont == "ascending":  d =  1
                elif cont == "descending": d = -1
                elif cont == "arch":       d =  1 if t < 0.5 else -1
                elif cont == "valley":     d = -1 if t < 0.5 else  1
                else:                      d =  1
                step = SKELETON_BASE_STEP + (SKELETON_SUSPENSE_WEIGHT * abs(sd) if cp == 0 else 0)
                ideal = prev + d * step + SKELETON_REGISTER_PULL * (rc - prev)
                tgt = _closest_to(int(round(ideal)), ct)
            skel.append(tgt)
            prev = tgt
        prev_s = es
    return skel


# ---------------------------------------------------------------------------
# Layer 2 — Cell Renderers (bar-level)
# ---------------------------------------------------------------------------

def _render_statement(sp, bc, budget, motif, genes, vel, key):
    tgt = _bar_exit(bc)

    def _take_notes(m, start_idx):
        """Take motif notes from start_idx until budget is filled."""
        mn = m.notes
        if not mn:
            return []
        notes = []
        off = sp - mn[start_idx % len(mn)].pitch
        total = 0.0
        idx = start_idx
        while total < budget - MIN_NOTE_DURATION:
            n = mn[idx % len(mn)]
            dur = min(n.duration, budget - total)
            if dur < MIN_NOTE_DURATION:
                break
            notes.append(_mn(_snap_to_scale(_clamp(n.pitch + off), key), dur, vel))
            total += dur
            idx += 1
            if idx - start_idx >= len(mn):
                break
        return notes

    start = genes.motif_start_note % len(motif.notes) if motif.notes else 0

    # Try with transform first
    transformed = apply_motif_transform(motif, genes.motif_transform)
    notes = _take_notes(transformed, start)

    # Guard: if all pitches are the same or too few notes, retry without transform
    unique = len(set(n.pitch for n in notes)) if notes else 0
    if unique <= 1 and len(notes) > 1:
        notes = _take_notes(motif, start)

    # Guard: if still only 1-2 notes, take from the beginning of the motif
    if len(notes) <= 2 and len(motif.notes) >= 3:
        notes = _take_notes(motif, 0)

    notes = _enforce_duration(notes, budget, tgt, vel)
    return notes, notes[-1].pitch if notes else tgt


def _render_development(sp, bc, budget, md, genes, vel, key):
    tgt = _bar_exit(bc)
    pool = md.interval_pool
    anc = max(CELL_ANCHOR_FREQ_MIN, min(CELL_ANCHOR_FREQ_MAX, genes.anchor_frequency))
    durs = _build_durations_from_pool(md.durations, genes.rhythm_variant, budget)
    cur = sp
    notes: List[VoicedNote] = []
    ba = 0.0

    for i, dur in enumerate(durs):
        ac = _chord_at_beat(bc, ba)
        ct = _chord_tones_in_range(ac)
        if i > 0 and i % anc == 0:
            cur = _closest_to(cur, ct)
        else:
            if random.random() < genes.foreign_interval_prob:
                iv = random.choice([-7,-5,-4,-3,-2,-1,1,2,3,4,5,7])
            else:
                idx = genes.interval_indices[i % len(genes.interval_indices)]
                iv = pool[idx % len(pool)]
            c2 = cur + iv
            if abs(c2 - tgt) > abs(cur - tgt) + 7: c2 = cur - iv
            cur = _snap_to_scale(_clamp(c2), key)
        notes.append(_mn(cur, dur, vel)); ba += dur

    notes = _enforce_duration(notes, budget, tgt, vel)
    return notes, notes[-1].pitch if notes else tgt


def _render_sequence(sp, bc, budget, motif, genes, vel, key):
    tgt = _bar_exit(bc)
    mn = motif.notes
    if not mn: return [_mn(tgt, budget, vel)], tgt
    fs = genes.motif_start_note % len(mn)
    fl = min(max(3, genes.statement_length), len(mn))
    frag = [mn[(fs+i) % len(mn)] for i in range(fl)]
    si = max(-CELL_SEQUENCE_MAX_INTERVAL, min(CELL_SEQUENCE_MAX_INTERVAL, genes.sequence_interval))
    if si == 0: si = 2
    notes: List[VoicedNote] = []
    bo = sp - frag[0].pitch
    tot = 0.0
    for rep in range(4):
        tr = bo + rep * si
        for fn in frag:
            if tot >= budget - MIN_NOTE_DURATION: break
            p = _snap_to_scale(_clamp(fn.pitch + tr), key)
            d = min(fn.duration, budget - tot)
            if d < MIN_NOTE_DURATION: break
            notes.append(_mn(p, d, vel)); tot += d
        if tot >= budget - MIN_NOTE_DURATION: break
    rem = budget - sum(n.duration for n in notes)
    if rem >= MIN_NOTE_DURATION and notes:
        c = notes[-1].pitch
        dr = 1 if tgt > c else -1
        while rem >= MIN_NOTE_DURATION:
            c = _next_scale_tone(c, dr, key)
            d = min(0.5, rem)
            notes.append(_mn(c, d, vel)); rem -= d
            if abs(c - tgt) <= 2: break
    notes = _enforce_duration(notes, budget, tgt, vel)
    return notes, notes[-1].pitch if notes else tgt


def _render_liquidation(sp, bc, budget, md, genes, vel, key):
    tgt = _bar_exit(bc)
    ivs = list(md.intervals)
    sr = max(2, min(6, genes.simplification_rate))
    phase, nip = 0, 0
    cur = sp
    ds = _build_durations_from_pool(md.durations, 0, budget)
    notes: List[VoicedNote] = []
    for i, dur in enumerate(ds):
        if nip >= sr and phase < 2: phase += 1; nip = 0
        if phase == 0:
            iv = ivs[i % len(ivs)]
        elif phase == 1:
            if genes.strip_order == "pitch_first":
                raw = ivs[i % len(ivs)]
                iv = 2 * (1 if raw > 0 else -1) if abs(raw) > 2 else raw
            else:
                iv = ivs[i % len(ivs)]
        else:
            # Stepwise toward target, alternating direction when at target
            if cur == tgt:
                iv = random.choice([1, -1, 2, -2])
            else:
                dr = 1 if tgt > cur else -1
                iv = dr * random.choice([1, 2])
        c2 = cur + iv
        if phase >= 1 and abs(c2-tgt) > abs(cur-tgt)+4: c2 = cur - iv
        cur = _snap_to_scale(_clamp(c2), key)
        notes.append(_mn(cur, dur, vel)); nip += 1
    notes = _enforce_duration(notes, budget, tgt, vel)
    return notes, notes[-1].pitch if notes else tgt


def _render_passage(sp, bc, budget, md, genes, vel, key):
    tgt = _bar_exit(bc)
    sd = sorted(md.durations)
    dn = max(0.0, min(1.0, genes.rhythmic_density))
    mx = sd[-1] - dn * (sd[-1] - sd[0])
    pl = [d for d in sd if d <= mx + 0.01] or [sd[0]]
    durs = _build_durations_from_pool(pl, 0, budget)
    cur = sp
    notes: List[VoicedNote] = []
    ba = 0.0
    ap = genes.approach_type
    arrived = False   # have we reached the target?

    # Pre-compute neighbors of target for oscillation
    upper_nb = _next_scale_tone(tgt, 1, key)
    lower_nb = _next_scale_tone(tgt, -1, key)

    # Arpeggio direction: randomly chosen per bar
    arp_dir = random.choice(["up", "down", "alt"])

    for i, dur in enumerate(durs):
        ac = _chord_at_beat(bc, ba)
        ct = _chord_tones_in_range(ac)

        if ap == "stepwise":
            if abs(cur - tgt) <= 1:
                arrived = True
            if not arrived:
                cur = _next_scale_tone(cur, 1 if tgt > cur else -1, key)
            else:
                # Oscillate: target, upper, target, lower, target...
                cycle = [tgt, upper_nb, tgt, lower_nb]
                cur = cycle[i % len(cycle)]

        elif ap == "neighbor":
            if abs(cur - tgt) <= 2:
                arrived = True
            if not arrived:
                cur = _next_scale_tone(cur, 1 if tgt > cur else -1, key)
            else:
                # Neighbor oscillation
                if i % 3 == 0: cur = tgt
                elif i % 3 == 1: cur = upper_nb
                else: cur = lower_nb

        elif ap == "arpeggio":
            if not ct:
                cur = tgt
            else:
                if arp_dir == "down":
                    cur = ct[-(i % len(ct)) - 1]
                elif arp_dir == "alt":
                    # Bounce: 0,1,2,...,n-1,n-2,...,1,0,1,...
                    cycle_len = max(1, 2 * len(ct) - 2)
                    pos = i % cycle_len
                    if pos < len(ct):
                        cur = ct[pos]
                    else:
                        cur = ct[cycle_len - pos]
                else:  # "up"
                    cur = ct[i % len(ct)]

        notes.append(_mn(cur, dur, vel)); ba += dur

    notes = _enforce_duration(notes, budget, tgt, vel)
    return notes, notes[-1].pitch if notes else tgt


def _render_cell(ct, sp, bc, budget, motif, md, genes, vel, key):
    if motif is None or md is None:
        ct = "passage"
        md = md or MotifData([0], [0], [1.0, 0.5], [])
    if   ct == "statement":   return _render_statement(sp, bc, budget, motif, genes, vel, key)
    elif ct == "development": return _render_development(sp, bc, budget, md, genes, vel, key)
    elif ct == "sequence":    return _render_sequence(sp, bc, budget, motif, genes, vel, key)
    elif ct == "liquidation": return _render_liquidation(sp, bc, budget, md, genes, vel, key)
    else:                     return _render_passage(sp, bc, budget, md, genes, vel, key)


# ---------------------------------------------------------------------------
# Layer 3 — Elaboration
# ---------------------------------------------------------------------------

def _elaborate(notes, orn_prob, key):
    if orn_prob <= 0 or not notes: return notes
    out: List[VoicedNote] = []
    for i, n in enumerate(notes):
        if n.duration < ELABORATION_MIN_NOTE_DUR or random.random() > orn_prob:
            out.append(n); continue
        ot = random.choice(["passing", "neighbor"])
        v, d = n.velocity, n.duration
        if ot == "passing" and i+1 < len(notes):
            dr = 1 if notes[i+1].pitch > n.pitch else -1
            pt = _next_scale_tone(n.pitch, dr, key)
            d1, d2 = round(d*0.5, 6), round(d*0.25, 6)
            d3 = round(d - d1 - d2, 6)
            out.extend([_mn(n.pitch,d1,v), _mn(pt,d2,v-5), _mn(n.pitch,d3,v)])
        elif ot == "neighbor":
            nb = _next_scale_tone(n.pitch, random.choice([1,-1]), key)
            d1, d2 = round(d*0.5, 6), round(d*0.25, 6)
            d3 = round(d - d1 - d2, 6)
            out.extend([_mn(n.pitch,d1,v), _mn(nb,d2,v-5), _mn(n.pitch,d3,v)])
        else:
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_melody_v2(
    progression: ChordProgression,
    events_by_id: dict,
    transposed_motifs: dict,
    melody_genes: List[MelodyGenes],
    base_velocity: int,
    section_key: Key,
    beats_per_bar: int = BEATS_PER_BAR,
) -> List[VoicedNote]:
    """Build melody: skeleton(per-chord) → cells(per-bar) → elaboration."""
    if not progression.chords: return []

    skeleton = build_skeleton(progression, melody_genes, events_by_id)

    md_cache: dict = {}
    for cid, m in transposed_motifs.items():
        md_cache[cid] = extract_motif_data(m)

    groups = group_chords_by_event(progression.chords)
    all_notes: List[VoicedNote] = []
    cgi = 0
    prev_p: Optional[int] = None
    last_m: Optional[Motif] = None
    last_md: Optional[MotifData] = None

    for gi, (eid, echords) in enumerate(groups):
        ev = events_by_id.get(eid, {})
        dom = ev.get("dominant_character_id")
        g = melody_genes[gi] if gi < len(melody_genes) else MelodyGenes()
        vel = max(0, min(127, g.event_velocity + VOICE_VELOCITY_OFFSETS[Voice.MELODY]))

        motif = transposed_motifs.get(dom) if dom else None
        md = md_cache.get(dom) if dom else None
        if motif is not None: last_m, last_md = motif, md
        else: motif, md = last_m, last_md

        bars = _group_chords_into_bars(echords, skeleton, cgi, beats_per_bar)
        ev_notes: List[VoicedNote] = []

        for bi, bc in enumerate(bars):
            budget = _bar_budget(bc)
            sp = prev_p if prev_p is not None else _bar_exit(bc)
            cg = g.cell_genes[bi] if bi < len(g.cell_genes) else CellGenes()
            cn, lp = _render_cell(cg.cell_type, sp, bc, budget, motif, md,
                                   cg, vel, section_key)
            ev_notes.extend(cn)
            prev_p = lp

        cgi += len(echords)

        op = max((c.ornament_probability for c in g.cell_genes[:len(bars)]), default=0.0)
        ev_notes = _elaborate(ev_notes, op, section_key)
        all_notes.extend(ev_notes)

    return all_notes


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------

def random_cell_genes(ct: str, md: Optional[MotifData] = None) -> CellGenes:
    ps = len(md.interval_pool) if md else 3
    return CellGenes(
        cell_type=ct,
        motif_start_note=random.randint(0, 7),
        statement_length=random.randint(CELL_STATEMENT_MIN_LENGTH, CELL_STATEMENT_MAX_LENGTH),
        motif_transform=random.choice(MOTIF_TRANSFORMS),
        interval_indices=[random.randint(0, max(0, ps-1)) for _ in range(MAX_INTERVAL_INDICES)],
        rhythm_variant=random.choice(RHYTHM_VARIANTS),
        anchor_frequency=random.randint(CELL_ANCHOR_FREQ_MIN, CELL_ANCHOR_FREQ_MAX),
        foreign_interval_prob=random.uniform(0, CELL_FOREIGN_INTERVAL_MAX),
        sequence_interval=random.choice([i for i in range(-CELL_SEQUENCE_MAX_INTERVAL,
                                                           CELL_SEQUENCE_MAX_INTERVAL+1) if i!=0]),
        is_tonal_sequence=random.choice([True, False]),
        simplification_rate=random.randint(2, 6),
        strip_order=random.choice(STRIP_ORDERS),
        approach_type=random.choices(APPROACH_TYPES, weights=[0.5, 0.4, 0.1])[0],
        rhythmic_density=random.uniform(0.0, 1.0),
        ornament_probability=random.uniform(0.0, 0.5),
    )


def random_melody_genes_for_event(
    n_bars: int,
    has_character: bool,
    motif_data: Optional[MotifData] = None,
    velocity_min: int = 40,
    velocity_max: int = 110,
) -> MelodyGenes:
    """Create MelodyGenes — one CellGenes per BAR."""
    cgs: List[CellGenes] = []
    for i in range(n_bars):
        if has_character:
            if i == 0:
                ct = random.choices(["statement","development"], weights=[0.8,0.2])[0]
            elif i == n_bars-1 and n_bars > 1:
                ct = random.choices(["liquidation","passage","development"],
                                     weights=[0.4,0.3,0.3])[0]
            else:
                ct = random.choices(["development","sequence","statement","passage"],
                                     weights=[0.30,0.30,0.20,0.20])[0]
        else:
            ct = random.choices(["passage","development","liquidation"],
                                 weights=[0.60,0.25,0.15])[0]
        cgs.append(random_cell_genes(ct, motif_data))
    return MelodyGenes(
        target_register=random.randint(0, 2),
        contour_direction=random.choice(CONTOUR_DIRECTIONS),
        cell_genes=cgs,
        event_velocity=random.randint(velocity_min, velocity_max),
    )