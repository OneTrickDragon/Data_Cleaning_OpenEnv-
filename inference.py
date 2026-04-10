"""
inference.py — OpenEnv submission script for Data Cleaning OpenEnv.

Mandatory stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    HF_TOKEN          Your HF token (used as API key). Required.
    API_BASE_URL      LLM endpoint. Default: https://router.huggingface.co/v1
    MODEL_NAME        Model identifier. Default: Qwen/Qwen2.5-72B-Instruct
    LOCAL_IMAGE_NAME  Docker image from Space registry (for from_docker_image mode).
    DC_ENV_URL        Live Space URL (used if LOCAL_IMAGE_NAME not set).
                      Default: https://onetrickdragon-data-cleaning-openenv.hf.space
    DC_TASK           Task name. Default: ecommerce_easy
    DC_SEED           RNG seed. Default: 42
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
# Use the validator-injected API_KEY and API_BASE_URL strictly — no fallbacks
API_KEY      = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("DC_TASK",      "ecommerce_easy")
BENCHMARK    = "data-cleaning-openenv"
DC_SEED      = int(os.getenv("DC_SEED",  "42"))
ENV_URL      = os.getenv("DC_ENV_URL",   "https://onetrickdragon-data-cleaning-openenv.hf.space")

MAX_STEPS               = 8
TEMPERATURE             = 0.3
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# OpenAI client (configured via env vars as required by submission spec)
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Import environment client — handle path issues gracefully
# ---------------------------------------------------------------------------

def _load_env_client():
    """
    Import DataCleaningEnv and DataCleaningAction.
    Tries local directory first, then falls back to the open-env GenericEnvClient
    so the script works even when run from /tmp/workspace/ by the validator.
    """
    # Add the script's own directory to path so local imports work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from client import DataCleaningEnv
        from models import DataCleaningAction
        return DataCleaningEnv, DataCleaningAction
    except ImportError:
        pass

    # Fallback: use the open-env GenericEnvClient with raw dicts
    try:
        from openenv.core.env_client import GenericEnvClient, GenericAction
        return GenericEnvClient, GenericAction
    except ImportError:
        pass

    raise ImportError(
        "Cannot import environment client. "
        "Ensure client.py and models.py are in the same directory as inference.py, "
        "or install open-env: pip install open-env"
    )

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ").replace("\r", " ")[:200]
    error_val   = error if error else "null"
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a data engineering expert working in a Python REPL.
    You have a pandas DataFrame called `df` and must clean it according to the task spec.

    Rules:
    - Write Python code that modifies `df` in place.
    - Available: pd, np, re, math, datetime (all pre-injected — no import needed).
    - No file I/O. No network calls. Max 20 lines per response.
    - Write ONLY the Python code — no explanation, no markdown fences.
    - When you have finished all cleaning steps, respond with exactly: SUBMIT
""").strip()


def build_prompt(obs, step: int, prev_result: str = "") -> str:
    parts = [
        f"TASK:\n{obs.task_spec}",
        f"DATAFRAME (first 10 rows):\n{obs.df_preview}",
        f"DTYPES:\n{obs.df_info}",
        f"PARTIAL SCORE: {obs.partial_score:.3f}  |  STEP: {step}/{MAX_STEPS}",
    ]
    if prev_result:
        parts.append(f"LAST OUTPUT:\n{prev_result[:300]}")
    if obs.error:
        parts.append(f"ERROR: {obs.error[:200]}")
    parts.append("Your code (or SUBMIT):")
    return "\n\n".join(parts)


def get_model_action(messages: list) -> str:
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        return (completion.choices[0].message.content or "").strip() or "SUBMIT"
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return "SUBMIT"


def parse_action(text: str) -> tuple:
    t = text.strip()
    if t.upper().startswith("SUBMIT") or not t:
        return "", True
    t = t.replace("```python", "").replace("```", "").strip()
    return t, False

# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

async def run_episode() -> None:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    obs                      = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        DataCleaningEnv, DataCleaningAction = _load_env_client()

        # Connect: docker image takes priority, then live URL
        if IMAGE_NAME:
            env = await DataCleaningEnv.from_docker_image(IMAGE_NAME)
        else:
            env = DataCleaningEnv(base_url=ENV_URL)

        async with env:
            result  = await env.reset(task_id=TASK_NAME, seed=DC_SEED)
            obs     = result.observation
            prev_result = ""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step=0)},
            ]

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                response_text = get_model_action(messages)
                messages.append({"role": "assistant", "content": response_text})

                code, submit = parse_action(response_text)

                try:
                    if submit or step == MAX_STEPS:
                        action = DataCleaningAction(type="submit")
                    else:
                        action = DataCleaningAction(type="exec", code=code or "pass")
                    result = await env.step(action)
                except Exception as step_exc:
                    print(f"[DEBUG] step error: {step_exc}", flush=True)
                    log_step(step=step, action=response_text, reward=0.0,
                             done=True, error=str(step_exc)[:200])
                    steps_taken = step
                    rewards.append(0.0)
                    break

                obs     = result.observation
                reward  = float(result.reward or 0.0)
                done    = result.done
                error   = obs.error if obs.error else None

                rewards.append(reward)
                steps_taken = step
                prev_result = obs.exec_result or ""

                log_step(step=step, action=response_text, reward=reward,
                         done=done, error=error)

                if done:
                    score = float(obs.reward)
                    break

                messages.append({
                    "role":    "user",
                    "content": build_prompt(obs, step, prev_result),
                })

        if rewards:
            score = rewards[-1]
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        if steps_taken == 0:
            log_step(step=1, action="(error)", reward=0.0, done=True,
                     error=str(exc)[:200])
            steps_taken = 1
            rewards     = [0.0]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards or [0.0])


async def main() -> None:
    await run_episode()


if __name__ == "__main__":
    asyncio.run(main())