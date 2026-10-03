from enum import StrEnum
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UNSPECIFIED = "Unspecified"  # what the UI and exports show for None

_EMPTY_MARKERS = {"", "unspecified", "unknown", "n/a", "na", "none", "null", "not stated"}

# Models our own code controls: strict. Models an LLM fills in: lenient on extras.
_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)
_LLM_OUTPUT = ConfigDict(str_strip_whitespace=True)


# ---------- Enums ----------

class DecisionStatus(StrEnum):
    DECIDED = "decided"        # explicitly agreed
    PROPOSED = "proposed"      # suggested, no agreement
    REJECTED = "rejected"      # suggested, then turned down
    UNRESOLVED = "unresolved"  # discussed, left open


class TaskStatus(StrEnum):
    CONFIRMED = "confirmed"    # someone clearly took it or was assigned it
    TENTATIVE = "tentative"    # "we should probably..."


class EditStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"  # touches a number, name or negation
    REJECTED = "rejected"          # guard rejected it; raw wording kept


# ---------- Stage 1 and 2: transcripts ----------

class Segment(BaseModel):
    """A piece of transcribed speech with timestamps in seconds. Immutable."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1)            # stable, e.g. "seg_001"
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)
    text: str = Field(min_length=1)
    avg_logprob: Optional[float] = Field(default=None, allow_inf_nan=False)
    no_speech_prob: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def check_time_order(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class Transcript(BaseModel):
    model_config = _STRICT

    segments: list[Segment]
    stt_model: str = ""
    language: str = "en"
    duration_s: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_unique_ids(self) -> Self:
        ids = [s.id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        return self

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    def by_id(self) -> dict[str, Segment]:
        return {s.id: s for s in self.segments}


class Correction(BaseModel):
    model_config = _STRICT

    segment_id: str = Field(min_length=1)
    original: str
    replacement: str
    status: EditStatus
    flags: list[str] = Field(default_factory=list)  # e.g. "number_changed"
    reason: Optional[str] = None


# ---------- Stage 3: meeting record (filled in by an LLM) ----------

class Evidence(BaseModel):
    """One quote from one segment."""

    model_config = _LLM_OUTPUT

    segment_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)   # verbatim text from that segment


class MinutesTopic(BaseModel):
    model_config = _LLM_OUTPUT

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Decision(BaseModel):
    model_config = _LLM_OUTPUT

    text: str = Field(min_length=1)
    status: DecisionStatus
    evidence: list[Evidence] = Field(min_length=1)


class ActionItem(BaseModel):
    model_config = _LLM_OUTPUT

    description: str = Field(min_length=1)
    status: TaskStatus                  # required: no silent "confirmed"
    owner: Optional[str] = None
    deadline: Optional[str] = None      # kept as spoken, e.g. "Friday"
    evidence: list[Evidence] = Field(min_length=1)
    owner_evidence: list[Evidence] = Field(default_factory=list)
    deadline_evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def blank_means_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in _EMPTY_MARKERS:
            return None
        return v

    @model_validator(mode="after")
    def check_assignment_evidence(self) -> Self:
        if self.owner is not None and not self.owner_evidence:
            raise ValueError("owner given without owner_evidence")
        if self.deadline is not None and not self.deadline_evidence:
            raise ValueError("deadline given without deadline_evidence")
        if self.owner is None and self.owner_evidence:
            raise ValueError("owner_evidence given without owner")
        if self.deadline is None and self.deadline_evidence:
            raise ValueError("deadline_evidence given without deadline")
        return self


class OpenQuestion(BaseModel):
    model_config = _LLM_OUTPUT

    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class MeetingRecord(BaseModel):
    model_config = _LLM_OUTPUT

    summary: str = Field(min_length=1)
    minutes: list[MinutesTopic]         # may be empty, but must be returned
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[OpenQuestion]


# ---------- Run bookkeeping ----------

class RunMetadata(BaseModel):
    model_config = _STRICT

    stt_model: str = ""
    refine_model: str = ""
    minutes_model: str = ""
    provider_used: dict[str, str] = Field(default_factory=dict)  # stage -> provider
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)