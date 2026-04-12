"""
inference.py — OpenEnv submission script for Data Cleaning OpenEnv.

Mandatory stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables (injected by validator):
    API_KEY           LLM API key
    API_BASE_URL      LLM proxy endpoint
    MODEL_NAME        Model identifier
    LOCAL_IMAGE_NAME  Docker image for the environment
"""

import asyncio
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — no fallbacks on validator-injected vars
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
#API_KEY      = os.getenv("API_KEY")
#API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("DC_TASK",    "ecommerce_easy")
BENCHMARK    = "data-cleaning-openenv"
DC_SEED      = int(os.getenv("DC_SEED", "42"))

MAX_STEPS               = 8
TEMPERATURE             = 0.3
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = str(action).replace("\n", " ").replace("\r", " ")[:200]
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
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
    You have a pandas DataFrame called `df` and must clean it per the task spec.
    Rules:
    - Modify `df` in place using Python code.
    - Available: pd, np, re, math, datetime (pre-injected).
    - No file I/O. No network. Max 20 lines per response.
    - Write ONLY Python code — no markdown fences, no explanation.
    - When finished, write exactly: SUBMIT
""").strip()


def obs_get(obs, key: str, default=None):
    """Get field from observation whether it's a dataclass or dict."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def build_prompt(obs, step: int, prev_result: str = "") -> str:
    parts = [
        f"TASK:\n{obs_get(obs, 'task_spec', '')}",
        f"DATAFRAME:\n{obs_get(obs, 'df_preview', '')}",
        f"DTYPES:\n{obs_get(obs, 'df_info', '')}",
        f"PARTIAL SCORE: {float(obs_get(obs, 'partial_score', 0)):.3f}  STEP: {step}/{MAX_STEPS}",
    ]
    if prev_result:
        parts.append(f"LAST OUTPUT:\n{prev_result[:300]}")
    err = obs_get(obs, "error")
    if err:
        parts.append(f"ERROR: {str(err)[:200]}")
    parts.append("Your code (or SUBMIT):")
    return "\n\n".join(parts)


def get_model_action(client: OpenAI, messages: list) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip() or "SUBMIT"
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        raise exc


def parse_action(text: str):
    t = text.strip()
    if t.upper().startswith("SUBMIT") or not t:
        return "", True
    t = t.replace("```python", "").replace("```", "").strip()
    return t, False


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataCleaningAction:
    type:    str           = "exec"
    code:    Optional[str] = None
    task_id: str           = "ecommerce_easy"
    seed:    int           = 42


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

async def run_episode() -> None:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # OpenAI client — uses validator-injected API_KEY and API_BASE_URL
    #if API_BASE_URL and not API_BASE_URL.endswith("/v1"):
    #    API_BASE_URL = f"{API_BASE_URL.rstrip('/')}/v1"
    #client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        # Add script directory to path so client.py is importable
        api_key = os.environ["API_KEY"]
        if not api_key:
            # Fallback to prevent immediate crash if validator does a dry-run
            api_key = os.getenv("HF_TOKEN", "dummy_key_to_prevent_crash")
        
        api_base = os.environ["API_BASE_URL"]
        
        # Ensure the proxy URL has the /v1 suffix
        if api_base and not api_base.endswith("/v1"):
            api_base = f"{api_base.rstrip('/')}/v1"

        client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import DataCleaningEnv


        # Spin up environment from Docker image (mirrors the sample script)
        if IMAGE_NAME:
            env = await DataCleaningEnv.from_docker_image(IMAGE_NAME)
        else:
            env_url = os.getenv("DC_ENV_URL", "https://onetrickdragon-data-cleaning-openenv.hf.space")
            env = DataCleaningEnv(base_url=env_url)

        async with env:
            result      = await env.reset(task_id=TASK_NAME, seed=DC_SEED)
            obs         = result.observation
            prev_result = ""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step=0)},
            ]

            for step in range(1, MAX_STEPS + 1):
                if obs_get(obs, "done", False):
                    break

                # LLM call — routed through validator proxy
                response_text = get_model_action(client, messages)
                messages.append({"role": "assistant", "content": response_text})

                code, submit = parse_action(response_text)

                try:
                    if submit or step == MAX_STEPS:
                        result = await env.step(DataCleaningAction(type="submit"))
                    else:
                        result = await env.step(DataCleaningAction(type="exec", code=code or "pass"))
                except Exception as step_exc:
                    print(f"[DEBUG] step error: {step_exc}", flush=True)
                    log_step(step, response_text, 0.0, True, str(step_exc)[:200])
                    steps_taken = step
                    rewards.append(0.0)
                    break

                obs         = result.observation
                reward      = float(result.reward or 0.0)
                done        = bool(obs_get(obs, "done", False))
                error       = obs_get(obs, "error") or None
                prev_result = obs_get(obs, "exec_result", "") or ""

                rewards.append(reward)
                steps_taken = step

                log_step(step, response_text, reward, done, error)

                if done:
                    score = reward
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
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        if steps_taken == 0:
            log_step(1, "(error)", 0.0, True, str(exc)[:200])
            steps_taken = 1
            rewards     = [0.0]

    finally:
        log_end(success, steps_taken, score, rewards or [0.0])


async def main() -> None:
    await run_episode()


if __name__ == "__main__":
    asyncio.run(main())