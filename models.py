from __future__ import annotations
 
from enum import Enum
from typing import Any, Optional
 
from pydantic import BaseModel, Field, field_validator
 
class ActionType(str, Enum):
    EXEC   = "exec"    # execute a Python snippet against the live DataFrame
    SUBMIT = "submit"  # finalise the episode and trigger full grading
 
 
class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"
 
 
class TaskID(str, Enum):
    ECOMMERCE_EASY         = "ecommerce_easy"
    PATIENT_RECORDS_MEDIUM = "patient_records_medium"
    FINANCIAL_AUDIT_HARD   = "financial_audit_hard"
 
class Action(BaseModel):
    """
    What the agent sends on each turn.
 
    - type="exec":   run `code` in the sandboxed REPL; `df` is in scope.
    - type="submit": end the episode and score the current DataFrame.
 
    Constraints enforced server-side:
      • Max 50 lines of code per step.
      • Imports restricted to: pandas, numpy, re, datetime, difflib,
        unicodedata, collections, itertools, math, string.
      • No file I/O, no network access, no subprocess.
    """
 
    type: ActionType
    code: Optional[str] = Field(
        default=None,
        description="Python snippet to execute. Required when type='exec'.",
    )
 
    @field_validator("code")
    @classmethod
    def code_line_limit(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.splitlines()) > 50:
            raise ValueError("code must be ≤ 50 lines per step")
        return v

class Observation(BaseModel):
    """
    Everything the agent sees after each step (or after reset).
 
    df_preview      — first 10 rows rendered as a markdown table.
    df_info         — dtypes, non-null counts, shape (like df.info()).
    df_stats        — df.describe() output as a string.
    task_spec       — plain-English objective + constraints for the task.
    exec_result     — stdout / stderr from the last code execution (empty on reset).
    step_count      — how many exec steps have been used so far.
    partial_score   — lightweight grader snapshot (0.0–1.0); updated after every exec.
    done            — True once the episode has ended.
    error           — non-empty if the last exec raised an unhandled exception.
    """
 
    df_preview:    str   = Field(description="First 10 rows as a markdown table")
    df_info:       str   = Field(description="dtypes + null counts + shape")
    df_stats:      str   = Field(description="df.describe() as string")
    task_spec:     str   = Field(description="Objective and constraints in plain English")
    exec_result:   str   = Field(default="", description="stdout/stderr of last exec")
    step_count:    int   = Field(default=0, ge=0)
    partial_score: float = Field(default=0.0, ge=0.0, le=1.0)
    done:          bool  = Field(default=False)
    error:         str   = Field(default="", description="Exception message if exec failed")

class Reward(BaseModel):
    """
    Decomposed reward returned at the end of an episode (type='submit' or
    step-limit exceeded).
 
    total           — weighted sum of all components (0.0–1.0).
    column_quality  — per-column dtype + null + value-range score (weight 0.50).
    schema_compliance — output columns match expected schema exactly (weight 0.20).
    row_preservation  — penalise unnecessary row drops (weight 0.15).
    efficiency        — bonus for finishing under 10 steps; penalty above 15 (weight 0.10).
    no_crash_bonus    — 0.05 if no unhandled exception was raised during episode.
    breakdown         — per-column quality detail for diagnostics.
    """
 
    total:              float = Field(ge=0.0, le=1.0)
    column_quality:     float = Field(ge=0.0, le=1.0)
    schema_compliance:  float = Field(ge=0.0, le=1.0)
    row_preservation:   float = Field(ge=0.0, le=1.0)
    efficiency:         float = Field(ge=0.0, le=1.0)
    no_crash_bonus:     float = Field(ge=0.0, le=0.05)
    breakdown:          dict[str, Any] = Field(default_factory=dict)
 
    @classmethod
    def zero(cls) -> "Reward":
        return cls(
            total=0.0,
            column_quality=0.0,
            schema_compliance=0.0,
            row_preservation=0.0,
            efficiency=0.0,
            no_crash_bonus=0.0,
        )
    
class State(BaseModel):
    """
    Full serialisable state of one episode.  The server stores this; the
    client never touches it directly — use Observation for training.
 
    df_parquet_b64 — base64-encoded parquet bytes of the live DataFrame.
    task_id        — which task is running.
    seed           — RNG seed used to generate corruptions (ensures reproducibility).
    step_count     — steps consumed.
    done           — episode finished flag.
    had_crash      — True if any exec step raised an unhandled exception.
    """
 
    df_parquet_b64: str
    task_id:        TaskID
    seed:           int
    step_count:     int  = 0
    done:           bool = False
    had_crash:      bool = False