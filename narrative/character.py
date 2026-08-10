"""
character.py

Narrative character analysis functions.
Computes character-level properties from narrative data
for use in musical motif generation.
"""

from typing import Optional

from .schema import ACTIVITY_VALUES

def compute_character_valence(
    character_id: int,
    narrative_data: dict,
) -> float:
    """
    Compute narrative valence for a character.

    For each thread the character appears in:
        contribution = importance × (events_present / thread_total_events)

    Valence = sum of contributions / number of threads character appears in

    Returns value in range (-10, +10) matching importance scale.
    Returns 0.0 if character appears in no threads.

    Args:
        character_id:   character_id from narrative data
        narrative_data: full narrative dict from extract_narrative()
    """
    threads = {t["thread_id"]: t for t in narrative_data["threads"]}
    events = narrative_data["events"]

    # Count total events per thread and events per thread where
    # this character is present
    thread_event_counts = {tid: 0 for tid in threads}
    thread_character_counts = {tid: 0 for tid in threads}

    for event in events:
        for tid in event["thread_ids"]:
            if tid in threads:
                thread_event_counts[tid] += 1
                if character_id in event["characters_present"]:
                    thread_character_counts[tid] += 1

    # Only consider threads where character appears at least once
    character_threads = [
        tid for tid in threads
        if thread_character_counts[tid] > 0
    ]

    if not character_threads:
        return 0.0

    total = sum(
        threads[tid]["importance"] *
        (thread_character_counts[tid] / thread_event_counts[tid])
        for tid in character_threads
        if thread_event_counts[tid] > 0
    )

    return total / len(character_threads)


def compute_avg_activity(
    character_id: int,
    narrative_data: dict,
) -> float:
    """
    Compute median visual activity across events where a character
    is present.
    ...
    """
    events = narrative_data["events"]

    character_events = [
        e for e in events
        if character_id in e["characters_present"]
    ]

    if not character_events:
        return 0.5

    activity_values = sorted(
        ACTIVITY_VALUES[e["visual_activity"]]
        for e in character_events
    )

    n = len(activity_values)
    mid = n // 2

    if n % 2 == 0:
        return (activity_values[mid - 1] + activity_values[mid]) / 2
    else:
        return activity_values[mid]
    

def compute_character_profiles(narrative_data: dict) -> list:
    """
    Compute valence and activity profiles for all characters
    in the narrative.

    Returns a list of dicts with keys:
        character_id, name, valence, avg_activity

    Convenience function to process all characters at once.

    Args:
        narrative_data: full narrative dict from extract_narrative()
    """
    profiles = []

    for character in narrative_data["characters"]:
        cid = character["character_id"]
        profiles.append({
            "character_id": cid,
            "name": character["name"],
            "valence": compute_character_valence(cid, narrative_data),
            "avg_activity": compute_avg_activity(cid, narrative_data),
        })

    return profiles

def compute_dominant_character(event, narrative_data) -> Optional[int]:
    threads_by_id = {t["thread_id"]: t for t in narrative_data["threads"]}

    active_threads = [
        threads_by_id[tid]
        for tid in event["thread_ids"]
        if tid in threads_by_id
    ]
    if not active_threads:
        return None

    dominant_thread = max(active_threads, key=lambda t: abs(t["importance"]))

    # Count appearances per character across all events in the dominant thread
    char_counts = {}
    for e in narrative_data["events"]:
        if dominant_thread["thread_id"] in e["thread_ids"]:
            for cid in e.get("characters_present", []):
                if cid in event["characters_present"]:
                    char_counts[cid] = char_counts.get(cid, 0) + 1

    if not char_counts:
        return None

    return max(char_counts, key=char_counts.get)