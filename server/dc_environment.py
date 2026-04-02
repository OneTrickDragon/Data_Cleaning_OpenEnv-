"""
server/dc_environment.py — DataCleaningEnvironment subclassing the real
openenv.core.env_server.Environment base class.

The simulation logic (dataset generation, sandbox, grader) lives in
environment.py and is unchanged. This file is the thin OpenEnv-compliant
wrapper around it.
"""

from __future__ import annotations

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    ActionType,
    TaskID,
)
from server.dc_environment import (
    MAX_STEPS,
    TASK_SPECS,
    build_observation,
    df_to_b64,
    b64_to_df,
    generate_datasets,
    grade,
    run_in_sandbox,
)


class DataCleaningEnvironment(Environment):
    """
    OpenEnv-compliant environment for tabular data cleaning.

    An agent receives a messy pandas DataFrame via text observations and must
    clean it using a sandboxed Python REPL. Three tasks of increasing difficulty:

      ecommerce_easy         — 500 rows, type/null/format fixes
      patient_records_medium — 1 200 rows, fuzzy dedup + normalisation
      financial_audit_hard   — 5 000 rows, 10 named business rules

    Rewards are dense: partial_score updates after every exec step so the agent
    gets gradient signal throughout the trajectory, not only at episode end.
    """
    is_concurrent_safe = True

    def __init__(self) -> None:
        super().__init__()
        self._state = DataCleaningState()

    # Core OpenEnv interface
    def reset(self, action: DataCleaningAction | None = None) -> DataCleaningObservation:
        """
        Start a new episode.

        Reads task_id and seed from the action (if provided), generates a fresh
        messy dataset, and returns the initial observation.
        """
        task_id_str = (action.task_id if action else None) or "ecommerce_easy"
        seed        = (action.seed    if action else None) or 42

        try:
            task_id = TaskID(task_id_str)
        except ValueError:
            task_id = TaskID.ECOMMERCE_EASY

        dirty_df, gold_df = generate_datasets(task_id, seed)

        self._state = DataCleaningState(
            episode_id   = str(uuid.uuid4()),
            df_state_b64 = df_to_b64(dirty_df),
            gold_b64     = df_to_b64(gold_df),
            task_id      = task_id.value,
            seed         = seed,
            step_count   = 0,
            done         = False,
            had_crash    = False,
        )

        return build_observation(
            df         = dirty_df,
            task_id    = task_id,
            gold_df    = gold_df,
            step_count = 0,
            done       = False,
        )

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        """
        Execute one action.

        action.type = "exec":   run code in the sandbox, return updated obs.
        action.type = "submit": finalise the episode, return full reward.

        Episodes also auto-terminate after MAX_STEPS exec steps.
        """
        if self._state.done:
            # Return a terminal observation without changing state
            df   = b64_to_df(self._state.df_state_b64)
            gold = b64_to_df(self._state.gold_b64)
            task_id = TaskID(self._state.task_id)
            return build_observation(df, task_id, gold, self._state.step_count, True,
                                     error="Episode already finished. Call reset() to start a new one.")

        df      = b64_to_df(self._state.df_state_b64)
        gold_df = b64_to_df(self._state.gold_b64)
        task_id = TaskID(self._state.task_id)

        exec_result = ""
        error_msg   = ""
        final_reward = 0.0

        if action.type == ActionType.SUBMIT or self._state.step_count >= MAX_STEPS:
            # Finalise episode
            self._state.done = True
            reward_obj = grade(df, gold_df, task_id,
                               self._state.step_count, self._state.had_crash)
            final_reward = reward_obj.total

        else:
            # Execute code in sandbox
            code = action.code or "pass"
            result = run_in_sandbox(code, df)
            exec_result = (result.stdout + result.stderr).strip()
            error_msg   = result.error

            if result.success:
                df = result.df
            else:
                self._state.had_crash = True

            self._state.step_count   += 1
            self._state.df_state_b64  = df_to_b64(df)

            # Auto-terminate at step limit
            if self._state.step_count >= MAX_STEPS:
                self._state.done = True
                reward_obj = grade(df, gold_df, task_id,
                                   self._state.step_count, self._state.had_crash)
                final_reward = reward_obj.total

        obs = build_observation(
            df          = df,
            task_id     = task_id,
            gold_df     = gold_df,
            step_count  = self._state.step_count,
            done        = self._state.done,
            exec_result = exec_result,
            error       = error_msg,
        )
        obs.reward = final_reward
        return obs

    @property
    def state(self) -> DataCleaningState:
        """Return the current episode state (episode_id, step_count, done, etc.)."""
        return self._state

import server.dc_environment as _env_module
from server.dc_environment import (
    TASK_SPECS as _TASK_SPECS,
    partial_grade as _partial_grade,
)
import io as _io
import pandas as _pd


def _build_observation(
    df, task_id, gold_df, step_count, done,
    exec_result="", error=""
) -> DataCleaningObservation:
    buf = _io.StringIO()
    df.info(buf=buf)
    preview = df.head(10).to_markdown(index=False)
    info    = buf.getvalue()
    stats   = df.describe(include="all").to_string()
    pscore  = _partial_grade(df, gold_df, task_id)

    return DataCleaningObservation(
        df_preview    = preview,
        df_info       = info,
        df_stats      = stats,
        task_spec     = _TASK_SPECS[task_id],
        exec_result   = exec_result,
        step_count    = step_count,
        partial_score = pscore,
        done          = done,
        error         = error,
        reward        = 0.0,
    )

_env_module.build_observation = _build_observation

from server.dc_environment import build_observation  # noqa: F811, E402