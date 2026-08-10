from pydantic import BaseModel, field_validator

from typing import List
from enum import Enum

class Character(BaseModel):
    character_id: int
    name: str
    description: str

class Thread(BaseModel):
    thread_id: int
    description: str
    importance: float
    
    @field_validator('importance')
    def importance_must_be_in_range(cls, v):
        if not -10.0 <= v <= 10.0:
            raise ValueError('importance must be between -10.0 and 10.0')
        return round(v, 2)

class Narrative(BaseModel):
    description: str
    characters: List[Character]
    threads: List[Thread]

class VisualActivity(str, Enum):
    STATIC   = "static"
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    FRANTIC  = "frantic"

class Event(BaseModel):
    event_id: int
    start_time: str   # MM:SS format from Gemini
    end_time: str     # MM:SS format from Gemini
    thread_ids: List[int]
    description: str
    characters_present: List[int]
    is_turning_point: bool
    disallows_thread_ids: List[int]
    visual_activity: VisualActivity

class EventList(BaseModel):
    events: List[Event]

ACTIVITY_VALUES = {
    VisualActivity.STATIC:   0.1,
    VisualActivity.LOW:      0.3,
    VisualActivity.MODERATE: 0.5,
    VisualActivity.HIGH:     0.7,
    VisualActivity.FRANTIC:  0.9,
}