PASS_1_PROMPT = """
You are analyzing an animated short film to extract its narrative structure.
Watch the entire video carefully before responding.

Your task is to identify:
1. All characters that appear in the video
2. All narrative threads — a narrative thread is a sequence of 
   causally related events that together form a mini-story with a potential 
   outcome

For each CHARACTER provide:
- character_id: integer starting from 1
- name: their name if known, otherwise a clear visual description 
  (e.g. "red_rabbit", "old_man_with_hat")
- description: physical appearance details that allow tracking them 
  across scenes

For each THREAD (narrative thread) provide:
- thread_id: integer starting from 1
- description: a clear one-sentence description of what this thread is 
  about and what its potential outcome is
- importance: a float between -10.0 and 10.0 representing the viewer's
  emotional investment in the thread's outcome, from the perspective of 
  how the story is told.

  To determine importance, follow these steps:
  
  STEP 1 - Identify the narrative perspective:
  Who does the story follow most closely? Whose actions and reactions 
  are shown in most detail? This is likely the character the viewer is 
  meant to root for, regardless of their moral standing.
  
  STEP 2 - Identify the tone:
  Is this a comedy, thriller, drama, adventure? In a comedy, the viewer 
  often roots for the protagonist to succeed even in morally ambiguous 
  situations. In a thriller, the viewer may root for the hero against a 
  villain. Tone shapes which outcomes feel satisfying.
  
  STEP 3 - Assign importance from the viewer's emotional perspective:
  A thread is POSITIVE (+) if the viewer would feel satisfied, relieved 
  or happy if this thread succeeds, given the narrative perspective and 
  tone identified above.
  A thread is NEGATIVE (-) if the viewer would feel disappointed, 
  anxious or unsatisfied if this thread succeeds.
  
  Examples by genre and perspective:
  - Comedy protagonist trying to cover up an accident: +7.0
    (viewer roots for the protagonist to get away with it)
  - Villain trying to kill the hero: -9.0
    (viewer does not want this to succeed)
  - Hero trying to reach safety: +8.0
    (viewer wants the hero to succeed)
  - Police investigating a likeable protagonist: -4.0
    (from protagonist's perspective, police succeeding is bad)
  - Police investigating a villain: +6.0
    (from hero's perspective, justice succeeding is good)
  - A neutral informational thread with no emotional stakes: 0.0

  Use values close to 0 for threads where the outcome is genuinely 
  ambiguous or the story presents both sides equally.
  Use values above +7 or below -7 only for threads with very high 
  emotional stakes.

IMPORTANT RULES:
- A thread must have a clear potential outcome that could either 
  succeed or fail
- Every character that plays a meaningful role must be listed
- Do not invent narrative threads that are not visually supported 
  by the video
- Always determine importance from the VIEWER'S emotional perspective 
  as shaped by the story's tone and narrative focus, NOT from a neutral 
  moral standpoint

Respond ONLY with a valid JSON object matching this exact structure, 
with no additional text:
{
    "description": "overall plot description in 2-3 sentences",
    "characters": [...],
    "threads": [...]
}
"""

PASS_2_PROMPT = """
You are analyzing an animated short film to extract its narrative events.
You have already identified the following narrative structure:

CHARACTERS:
{characters}

NARRATIVE THREADS:
{threads}

Now watch the video carefully and extract all narrative events.

An event is a discrete narrative unit where something meaningful happens 
that advances one or more narrative threads. Events should be:
- Meaningful: they change the state of at least one narrative thread
- Non-overlapping: they should cover the video sequentially
- Complete: together they should cover the entire video duration

For each EVENT provide:
- event_id: integer starting from 1
- start_time: start time in MM:SS format (e.g. "01:23" for 1 minute 23 seconds)
- end_time: end time in MM:SS format (e.g. "02:05" for 2 minutes 5 seconds)
- thread_ids: list of narrative thread IDs that this event is part of — an event 
  can belong to multiple threads if it advances more than one thread.
  A thread should only appear here when it is actively present in this event,
  not merely ongoing in the background.
  A thread ends naturally when it no longer appears in thread_ids — 
  this does not require any special action.
- description: a clear one-sentence description of what happens
- characters_present: list of character_ids of characters visible and 
  active in this event
- is_turning_point: true if this event significantly changes the state 
  of one or more narrative threads (e.g. a new threat appears, a plan succeeds 
  or fails, a character's situation changes dramatically). 
  Use this sparingly — a 5 minute video should have at most 4-6 
  turning points.
- disallows_thread_ids: list of narrative thread IDs that this event 
  makes impossible or contradicts — specifically when one thread's event 
  directly prevents another thread from continuing.
  
  This is NOT used when a thread simply ends naturally. A thread ends 
  naturally by no longer appearing in thread_ids.
  
  Use disallows_thread_ids ONLY when there is a direct causal contradiction 
  between threads. Examples:
  - A bomb being defused disallows the thread where the bomb explodes
  - A character escaping disallows the thread where they are captured
  - Two mutually exclusive outcomes where one happening prevents the other
  
  Leave empty [] in most cases.

- visual_activity: how much is visually changing on screen during 
  this event. This is a purely perceptual judgment about motion 
  and visual change, completely independent of narrative importance 
  or tension. Choose ONE of these five levels:
  - "static":   minimal movement, character thinking or sitting still,
                very slow deliberate actions. Example: a character 
                sitting alone reading or thinking without any background movement.
  - "low":      slow deliberate movement, calm conversation with 
                moderate gestures, simple single-character tasks.
                Example: a character walking slowly or performing 
                a calm task.
  - "moderate": normal activity level, characters moving with purpose,
                conversation with physical involvement.
                Example: a character getting dressed or packing a bag.
  - "high":     fast or multiple characters moving simultaneously,
                urgent physical actions, quick scene transitions.
                Example: a character being chased or confronted.
  - "frantic":  rapid cutting, chaotic movement, multiple simultaneous
                actions, very fast pace. Example: a chase sequence 
                with multiple characters in conflict.
  
  Important: visual_activity is independent of narrative suspense or tension.
  A character sitting still while in great danger = "static".
  A character cheerfully running = "high".

IMPORTANT RULES:
- Use ONLY the character_ids and thread_ids defined above
- Events must be chronological and non-overlapping
- Aim for events of roughly similar duration (around 00:03 to 00:15 each)
- Every part of the video must be covered by at least one event
- disallows_thread_ids should be empty [] in most cases — only use it 
  when one thread's event directly makes another thread impossible

Respond ONLY with a valid JSON object matching this exact structure,
with no additional text:
{{
    "events": [...]
}}
"""