"""
narrative/sections.py

Music section detection and key assignment.

Segments narrative events into music sections based on thread
continuity, then assigns a tonal key to each section derived
from narrative properties:

    Mode:   dominant thread importance > 0 → Major
            dominant thread importance < 0 → Minor

    Root:   character overlap between consecutive sections
            determines distance on the circle of fifths —
            more characters in common → closer keys

Section boundaries occur when:
    - Active threads have no overlap with the previous event's threads
    - There are no active threads and no characters present (silence)
"""

import random
from typing import List, Optional

from music.structures import Key, CIRCLE_OF_FIFTHS, ROOT_TO_NOTE


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------

class SectionType:
    MUSIC   = "music"
    SILENCE = "silence"


class Section:
    """
    A contiguous segment of narrative events sharing at least one
    active thread, or a silence segment with no narrative activity.

    Attributes:
        type:           "music" or "silence"
        events:         list of suspense result dicts in this section
        thread_ids:     union of all thread_ids across events
        character_ids:  union of all characters_present across events
        key:            assigned tonal key (None for silence sections)
    """

    def __init__(self, section_type: str):
        self.type = section_type
        self.events: list = []
        self.thread_ids: set = set()
        self.character_ids: set = set()
        self.key: Optional[Key] = None

    @property
    def start_time(self) -> float:
        return self.events[0]["start_time"] if self.events else 0.0

    @property
    def end_time(self) -> float:
        return self.events[-1]["end_time"] if self.events else 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_music(self) -> bool:
        return self.type == SectionType.MUSIC

    @property
    def is_silence(self) -> bool:
        return self.type == SectionType.SILENCE

    @property
    def peak_suspense(self) -> float:
        if not self.events:
            return 0.0
        return max(e["global_suspense"] for e in self.events)

    @property
    def mean_suspense(self) -> float:
        if not self.events:
            return 0.0
        return sum(e["global_suspense"] for e in self.events) / len(self.events)

    def __repr__(self) -> str:
        if self.is_silence:
            return (f"Section(silence, "
                    f"{self.start_time:.1f}s-{self.end_time:.1f}s)")
        return (f"Section(music, "
                f"{self.start_time:.1f}s-{self.end_time:.1f}s, "
                f"threads={self.thread_ids}, "
                f"key={self.key})")


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def detect_music_sections(
    suspense_results: list,
    narrative_data: dict,
) -> List[Section]:
    """
    Segment narrative events into music sections.

    A new section begins when:
    - The current event has no thread overlap with the previous event
    - The current event has no active threads or characters (silence)

    Args:
        suspense_results: output of compute_suspense()
        narrative_data:   full narrative dict from extract_narrative()

    Returns:
        ordered list of Section instances covering the full narrative
    """
    # Build event lookup by event_id
    events_by_id = {
        e["event_id"]: e for e in narrative_data["events"]
    }

    # Write computed suspense values back into narrative_data["events"]
    # so they are accessible everywhere downstream (objectives, harmony, etc.)
    for result in suspense_results:
        eid = result["event_id"]
        if eid in events_by_id:
            events_by_id[eid]["global_suspense"] = result["global_suspense"]

    sections = []
    current_section: Optional[Section] = None

    for result in suspense_results:
        event = events_by_id[result["event_id"]]
        active_threads = set(event["thread_ids"])
        active_characters = set(event["characters_present"])

        # Attach timing from suspense result to event for convenience
        result_with_timing = {**result,
                              "start_time": result["start_time"],
                              "end_time": result["end_time"]}

        is_silent = (
            len(active_threads) == 0 and
            len(active_characters) == 0
        )

        if is_silent:
            # Close current music section if open
            if current_section is not None:
                sections.append(current_section)
                current_section = None

            # Add silence section
            silence = Section(SectionType.SILENCE)
            silence.events.append(result_with_timing)
            sections.append(silence)

        elif current_section is None:
            # Start first music section
            current_section = Section(SectionType.MUSIC)
            current_section.events.append(result_with_timing)
            current_section.thread_ids |= active_threads
            current_section.character_ids |= active_characters

        else:
            # Check thread overlap with current section's last event
            last_event = events_by_id[current_section.events[-1]["event_id"]]
            last_threads = set(last_event["thread_ids"])
            thread_overlap = active_threads & last_threads

            if thread_overlap:
                # Continue current section
                current_section.events.append(result_with_timing)
                current_section.thread_ids |= active_threads
                current_section.character_ids |= active_characters
            else:
                # No overlap — close current, start new section
                sections.append(current_section)
                current_section = Section(SectionType.MUSIC)
                current_section.events.append(result_with_timing)
                current_section.thread_ids |= active_threads
                current_section.character_ids |= active_characters

    # Close any remaining open section
    if current_section is not None:
        sections.append(current_section)

    return sections


# ---------------------------------------------------------------------------
# Key assignment
# ---------------------------------------------------------------------------

def _dominant_thread_importance(
    section: Section,
    narrative_data: dict,
) -> float:
    """
    Return the importance of the thread with the highest absolute
    importance among threads active in this section.
    """
    threads = {t["thread_id"]: t for t in narrative_data["threads"]}
    importances = [
        threads[tid]["importance"]
        for tid in section.thread_ids
        if tid in threads
    ]
    if not importances:
        return 0.0
    return max(importances, key=abs)


def _character_overlap_ratio(
    section_a: Section,
    section_b: Section,
) -> float:
    """
    Compute character overlap ratio between two sections.

        overlap / total_unique_characters

    Returns 0.0 if neither section has characters.
    """
    union = section_a.character_ids | section_b.character_ids
    if not union:
        return 0.0
    intersection = section_a.character_ids & section_b.character_ids
    return len(intersection) / len(union)


def _fifths_steps_from_overlap(overlap_ratio: float) -> int:
    """
    Map character overlap ratio to distance in circle of fifths steps.

        1.0        → 0 steps (same root)
        [0.6, 1.0) → 1 step
        [0.35,0.6) → 2 steps
        [0.1, 0.35)→ 3 steps
        [0.0, 0.1) → 4-6 steps (random, maximise contrast)
    """
    if overlap_ratio >= 1.0:
        return 0
    elif overlap_ratio >= 0.6:
        return 1
    elif overlap_ratio >= 0.35:
        return 2
    elif overlap_ratio >= 0.1:
        return 3
    else:
        return random.randint(4, 6)


def _key_from_previous(
    previous_key: Key,
    steps: int,
    is_major: bool,
) -> Key:
    """
    Compute a new key root by moving `steps` steps on the circle
    of fifths from the previous key's root.

    Direction is chosen to minimise enharmonic complexity —
    always move in the direction that keeps the root pitch class
    within the simpler half of the circle.

    Args:
        previous_key: key of the preceding section
        steps:        number of fifths steps to move
        is_major:     mode of the new key

    Returns:
        new Key
    """
    if steps == 0:
        return Key(root=previous_key.root, is_major=is_major)

    i = CIRCLE_OF_FIFTHS.index(previous_key.root % 12)
    n = len(CIRCLE_OF_FIFTHS)

    # Clockwise and counterclockwise candidates
    clockwise_root = CIRCLE_OF_FIFTHS[(i + steps) % n]
    counter_root = CIRCLE_OF_FIFTHS[(i - steps) % n]

    # Prefer the direction that avoids sharps/flats beyond 4
    # (i.e. stays closer to C on the circle)
    clockwise_pos = (i + steps) % n
    counter_pos = (i - steps) % n

    # Position closer to 0 (C) = simpler key signature
    if clockwise_pos <= n // 2:
        new_root = clockwise_root
    elif counter_pos <= n // 2:
        new_root = counter_root
    else:
        new_root = clockwise_root  # default to clockwise

    return Key(root=new_root, is_major=is_major)


def assign_keys(
    sections: List[Section],
    narrative_data: dict,
    initial_key: Optional[Key] = None,
    seed: Optional[int] = None,
) -> List[Section]:
    """
    Assign a tonal key to each music section.

    Mode is derived from the dominant thread's importance sign.
    Root is derived from character overlap with the previous
    section via the circle of fifths.

    Silence sections receive no key (key remains None).

    Args:
        sections:     output of detect_music_sections()
        narrative_data: full narrative dict from extract_narrative()
        initial_key:  key for the first music section — if None,
                      derived from the first section's dominant thread
        seed:         random seed for reproducible key assignment

    Returns:
        sections list with key assigned to all music sections
    """
    if seed is not None:
        random.seed(seed)

    music_sections = [s for s in sections if s.is_music]

    if not music_sections:
        return sections

    # --- Assign first section ---
    first = music_sections[0]
    if initial_key is not None:
        first.key = initial_key
    else:
        importance = _dominant_thread_importance(first, narrative_data)
        is_major = importance >= 0
        # First section defaults to C major or A minor
        root = 0 if is_major else 9
        first.key = Key(root=root, is_major=is_major)

    # --- Assign subsequent sections ---
    for i in range(1, len(music_sections)):
        section = music_sections[i]
        previous = music_sections[i - 1]

        importance = _dominant_thread_importance(section, narrative_data)
        is_major = importance >= 0

        overlap = _character_overlap_ratio(previous, section)
        steps = _fifths_steps_from_overlap(overlap)

        section.key = _key_from_previous(previous.key, steps, is_major)

    return sections


# ---------------------------------------------------------------------------
# Summary utility
# ---------------------------------------------------------------------------

def print_sections(sections: List[Section]) -> None:
    """Print a readable summary of detected sections and their keys."""
    print(f"{'#':<4} {'Type':<8} {'Start':>6} {'End':>6} "
          f"{'Duration':>9} {'Key':<15} {'Threads'}")
    print("-" * 70)
    for i, section in enumerate(sections):
        key_str = str(section.key) if section.key else "-"
        threads_str = str(sorted(section.thread_ids)) if section.thread_ids else "-"
        print(
            f"{i:<4} {section.type:<8} "
            f"{section.start_time:>5.1f}s "
            f"{section.end_time:>5.1f}s "
            f"{section.duration:>8.1f}s "
            f"{key_str:<15} "
            f"{threads_str}"
        )