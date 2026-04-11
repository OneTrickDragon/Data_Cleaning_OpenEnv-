"""
inference.py — OpenEnv submission script for Data Cleaning OpenEnv.

Mandatory stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables (injected by validator):
    API_KEY           LLM API key (injected by validator proxy)
    API_BASE_URL      LLM proxy endpoint (injected by validator)
    MODEL_NAME        Model to use. Default: Qwen/Qwen2.5-72B-Instruct
    LOCAL_IMAGE_NAME  Docker image for the environment (optional)
    DC_ENV_URL        Live Space URL. Default: the deployed HF Space
    DC_TASK           Task name. Default: ecommerce_easy
    DC_SEED           RNG seed. Default: 42
"""

import asyncio
import json
import os
import sys
import textwrap
import traceback
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read at call time inside functions, not at import time
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASK_NAME  = os.getenv("DC_TASK",    "ecommerce_easy")
BENCHMARK  = "data-cleaning-openenv"
DC_SEED    = int(os.getenv("DC_SEED", "42"))
ENV_URL    = os.getenv("DC_ENV_URL", "https://onetrickdragon-data-cleaning-openenv.hf.space")
HF_TOKEN  = os.getenv("HF_TOKEN")

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


def build_prompt(obs_dict: dict, step: int, prev_result: str = "") -> str:
    parts = [
        f"TASK:\n{obs_dict.get('task_spec', '')}",
        f"DATAFRAME:\n{obs_dict.get('df_preview', '')}",
        f"DTYPES:\n{obs_dict.get('df_info', '')}",
        f"PARTIAL SCORE: {obs_dict.get('partial_score', 0):.3f}  STEP: {step}/{MAX_STEPS}",
    ]
    if prev_result:
        parts.append(f"LAST OUTPUT:\n{prev_result[:300]}")
    if obs_dict.get("error"):
        parts.append(f"ERROR: {obs_dict['error'][:200]}")
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
        return "SUBMIT"


def parse_action(text: str):
    t = text.strip()
    if t.upper().startswith("SUBMIT") or not t:
        return "", True
    t = t.replace("```python", "").replace("```", "").strip()
    return t, False

# ---------------------------------------------------------------------------
# HTTP env client (self-contained, no local imports needed)
# ---------------------------------------------------------------------------

async def http_reset(session, base_url: str, task_id: str, seed: int) -> dict:
    import aiohttp
    payload = {"action": {"type": "exec", "task_id": task_id, "seed": seed}}
    async with session.post(f"{base_url}/reset", json=payload) as r:
        return await r.json()


async def http_step(session, base_url: str, action_type: str, code: str = None) -> dict:
    import aiohttp
    action = {"type": action_type}
    if code:
        action["code"] = code
    async with session.post(f"{base_url}/step", json={"action": action}) as r:
        return await r.json()


def extract_obs(response: dict) -> dict:
    """Normalise response to a flat obs dict regardless of server format."""
    # Try nested observation key first
    obs = response.get("observation", response)
    if not isinstance(obs, dict):
        obs = response
    return obs


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

async def run_episode() -> None:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Create OpenAI client here — env vars guaranteed to be set by now
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=HF_TOKEN,
    )

    base_url = ENV_URL.rstrip("/")

    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Reset
            resp    = await http_reset(session, base_url, TASK_NAME, DC_SEED)
            obs     = extract_obs(resp)
            prev_result = ""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step=0)},
            ]

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                # LLM call — goes through the validator's proxy
                response_text = get_model_action(client, messages)
                messages.append({"role": "assistant", "content": response_text})

                code, submit = parse_action(response_text)

                try:
                    if submit or step == MAX_STEPS:
                        resp = await http_step(session, base_url, "submit")
                    else:
                        resp = await http_step(session, base_url, "exec", code or "pass")
                except Exception as step_exc:
                    print(f"[DEBUG] step error: {step_exc}", flush=True)
                    log_step(step, response_text, 0.0, True, str(step_exc)[:200])
                    steps_taken = step
                    rewards.append(0.0)
                    break

                obs     = extract_obs(resp)
                reward  = float(obs.get("reward", resp.get("reward", 0.0)) or 0.0)
                done    = bool(obs.get("done", resp.get("done", False)))
                error   = obs.get("error") or None

                rewards.append(reward)
                steps_taken = step
                prev_result = obs.get("exec_result", "")

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