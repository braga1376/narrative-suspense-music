import time
import json
from google.genai import types

from .schema import Narrative, EventList
from .prompts import PASS_1_PROMPT, PASS_2_PROMPT

def _is_youtube_url(source: str) -> bool:
    """
    Detect YouTube URLs in all common formats:
    - https://www.youtube.com/watch?v=...
    - https://youtu.be/...
    - https://youtube.com/watch?v=...
    - https://www.youtube.com/shorts/...
    """
    return isinstance(source, str) and any(
        domain in source
        for domain in ("youtube.com", "youtu.be")
    )

def _build_contents(video_source: str, prompt: str) -> list:
    """
    Build the contents list for a Gemini request.
    Handles both uploaded file objects and YouTube URLs.
    """
    if _is_youtube_url(video_source):
        video_part = types.Part(
            file_data=types.FileData(file_uri=video_source)
        )
    else:
        # Uploaded file object
        video_part = video_source

    return [video_part, prompt]


def _upload_video(client, video_file_path: str):
    """Upload a local video file and wait for processing."""
    print("Uploading video...")
    video_file = client.files.upload(file=video_file_path)

    while video_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    print()  # newline after dots
    return video_file


def _mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS timestamp to seconds.

    Handles formats: "MM:SS", "M:SS", "HH:MM:SS", or plain number.
    """
    if isinstance(mmss, (int, float)):
        return float(mmss)

    s = str(mmss).strip()

    # Already a plain number?
    try:
        return float(s)
    except ValueError:
        pass

    parts = s.split(":")
    if len(parts) == 2:
        # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Cannot parse timestamp: {mmss!r}")


def extract_narrative(client, video_source: str, model: str) -> dict:
    """
    Extract narrative structure from a video.

    Args:
        client:       Gemini client instance
        video_source: Either a local file path or a YouTube URL
        model:        Gemini model name from config

    Returns:
        dict with keys: description, characters, threads, events
    """
    # Resolve video source
    if _is_youtube_url(video_source):
        print(f"Using YouTube URL: {video_source}")
        resolved_source = video_source
    else:
        resolved_source = _upload_video(client, video_source)

    # --- Pass 1: Characters and threads ---
    pass_1_response = client.models.generate_content(
        model=model,
        contents=_build_contents(resolved_source, PASS_1_PROMPT),
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Narrative.model_json_schema(),
        }
    )

    narrative = Narrative.model_validate_json(pass_1_response.text)

    characters_context = json.dumps(
        [c.model_dump() for c in narrative.characters], indent=2
    )
    threads_context = json.dumps(
        [t.model_dump() for t in narrative.threads], indent=2
    )

    # --- Pass 2: Events ---
    pass_2_prompt = PASS_2_PROMPT.format(
        characters=characters_context,
        threads=threads_context
    )

    pass_2_response = client.models.generate_content(
        model=model,
        contents=_build_contents(resolved_source, pass_2_prompt),
        config={
            "response_mime_type": "application/json",
            "response_json_schema": EventList.model_json_schema(),
        }
    )

    event_list = EventList.model_validate_json(pass_2_response.text)

    # Convert MM:SS timestamps to seconds for the rest of the system
    events = []
    for e in event_list.events:
        d = e.model_dump()
        d["start_time"] = _mmss_to_seconds(d["start_time"])
        d["end_time"] = _mmss_to_seconds(d["end_time"])
        events.append(d)

    # Make events contiguous: each event starts where the previous ended.
    # Gemini's 1fps resolution can leave 1-second gaps between events;
    # closing them ensures the music timeline has no dead spots.
    if events:
        events.sort(key=lambda e: e["start_time"])
        for i in range(len(events)-1):
            events[i]["end_time"] = events[i + 1]["start_time"]

    return {
        "description": narrative.description,
        "characters": [c.model_dump() for c in narrative.characters],
        "threads": [t.model_dump() for t in narrative.threads],
        "events": events,
    }