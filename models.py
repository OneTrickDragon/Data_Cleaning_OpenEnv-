"""
models.py — Type-safe contracts for the Data Cleaning OpenEnv environment.

Uses dataclasses inheriting from openenv.core.env_server.{Action,Observation,State}
as required by the OpenEnv framework spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    @dataclass
    class Action: pass         # type: ignore[no-redef]
    @dataclass
    class Observation: pass    # type: ignore[no-redef]
    @dataclass
    class State: pass  

# Enums
class ActionType(str, Enum):
    EXEC   = "exec"    # execute a Python snippet against the live DataFrame
    SUBMIT = "submit"  # finalise the episode and trigger full grading


class TaskID(str, Enum):
    ECOMMERCE_EASY         = "ecommerce_easy"
    PATIENT_RECORDS_MEDIUM = "patient_records_medium"
    FINANCIAL_AUDIT_HARD   = "financial_audit_hard"

# Action
class DataCleaningAction(Action):
    """
    What the agent sends on each turn.
 
    type    — "exec" to run code, "submit" to end the episode.
    code    — Python snippet (≤50 lines). df in scope. Required when type="exec".
    task_id — which task to run (used only on reset).
    seed    — RNG seed for reproducibility (used only on reset).
    """
    type:    str           = "exec"
    code:    Optional[str] = None
    task_id: str           = "ecommerce_easy"
    seed:    int           = 42

# Observation
class DataCleaningObservation(Observation):
    """
    Everything the agent sees after each step or reset.
 
    df_preview      — first 10 rows as a markdown table.
    df_info         — dtypes, non-null counts, shape.
    df_stats        — df.describe() output.
    task_spec       — plain-English objective and constraints.
    exec_result     — stdout/stderr from the last code execution.
    step_count      — how many exec steps have been used.
    partial_score   — lightweight grader snapshot (0.0–1.0), updated every step.
    error           — non-empty if the last exec raised an exception.
    """
    df_preview:    str   = ""
    df_info:       str   = ""
    df_stats:      str   = ""
    task_spec:     str   = ""
    exec_result:   str   = ""
    step_count:    int   = 0
    partial_score: float = 0.0
    error:         str   = ""

# State  (server-side canonical truth)
class DataCleaningState(State):
    """
    Full serialisable state of one episode.
 
    episode_id    — unique ID per reset() call.
    df_state_b64  — base64-encoded pickle bytes of the live DataFrame.
    gold_b64      — base64-encoded pickle bytes of the gold DataFrame.
    task_id       — which task is running.
    seed          — RNG seed used to generate corruptions.
    step_count    — steps consumed so far.
    done          — episode finished flag.
    had_crash     — True if any exec step raised an unhandled exception.
    """
    episode_id:   str  = ""
    df_state_b64: str  = ""
    gold_b64:     str  = ""
    task_id:      str  = "ecommerce_easy"
    seed:         int  = 42
    step_count:   int  = 0
    done:         bool = False
    had_crash:    bool = False