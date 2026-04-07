"""
inference.py — OpenEnv submission inference script for Data Cleaning OpenEnv.

Mandatory stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL        LLM endpoint (default: HuggingFace router)
    MODEL_NAME          Model identifier
    HF_TOKEN / API_KEY  Auth token
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image)
    DC_TASK             Task to run: ecommerce_easy | patient_records_medium | financial_audit_hard
    DC_SEED             RNG seed (default: 42)
    DC_ENV_URL          Space URL (default: uses docker image)
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import DataCleaningEnv
from models import DataCleaningAction

# Configuration
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("DC_TASK",      "ecommerce_easy")
BENCHMARK    = "data-cleaning-openenv"
DC_SEED      = int(os.getenv("DC_SEED", "42"))
ENV_URL      = os.getenv("DC_ENV_URL",   "")

MAX_STEPS             = 8
TEMPERATURE           = 0.3
MAX_TOKENS            = 512
SUCCESS_SCORE_THRESHOLD = 0.5   # normalized score in [0, 1]

# Prompts
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


def build_user_prompt(obs, step: int, prev_result: str = "") -> str:
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

# Logging helpers (exact format required by submission spec)
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Sanitize action: no newlines allowed in a single [STEP] line
    action_safe = action.replace("\n", " ").replace("\r", " ")[:200]
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# LLM call
def get_model_action(
    client: OpenAI,
    messages: list,
) -> str:
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "SUBMIT"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SUBMIT"


def parse_action(text: str) -> tuple[str, bool]:
    """Returns (code, should_submit)."""
    t = text.strip()
    if t.upper().startswith("SUBMIT") or not t:
        return "", True
    # Strip accidental markdown fences
    t = t.replace("```python", "").replace("```", "").strip()
    return t, False


# Main episode loop
async def run_episode(client: OpenAI) -> None:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Connect to environment
    if IMAGE_NAME:
        env = await DataCleaningEnv.from_docker_image(IMAGE_NAME)
    elif ENV_URL:
        env = DataCleaningEnv(base_url=ENV_URL)
    else:
        raise RuntimeError(
            "Set LOCAL_IMAGE_NAME or DC_ENV_URL to specify the environment.\n"
            "  LOCAL_IMAGE_NAME=registry.hf.space/onetrickdragon-data-cleaning-openenv:latest\n"
            "  DC_ENV_URL=https://onetrickdragon-data-cleaning-openenv.hf.space"
        )

    try:
        async with env:
            # Reset
            result  = await env.reset(task_id=TASK_NAME, seed=DC_SEED)
            obs     = result.observation
            prev_result = ""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, step=0)},
            ]

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                # Get action from model
                response_text = get_model_action(client, messages)
                messages.append({"role": "assistant", "content": response_text})

                code, submit = parse_action(response_text)

                if submit or step == MAX_STEPS:
                    result = await env.step(DataCleaningAction(type="submit"))
                else:
                    result = await env.step(DataCleaningAction(type="exec", code=code))

                obs     = result.observation
                reward  = float(result.reward or 0.0)
                done    = result.done
                error   = obs.error if obs.error else None

                rewards.append(reward)
                steps_taken = step
                prev_result = obs.exec_result or ""

                log_step(
                    step   = step,
                    action = response_text,
                    reward = reward,
                    done   = done,
                    error  = error,
                )

                if done:
                    score = float(obs.reward)
                    break

                # Feed observation back to model
                messages.append({
                    "role":    "user",
                    "content": build_user_prompt(obs, step, prev_result),
                })

        # Score is the final reward (already 0–1 from our grader)
        if rewards:
            score = rewards[-1]   # last reward is the terminal graded score
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        error_msg = str(exc)
        if steps_taken == 0:
            log_step(step=1, action="(error)", reward=0.0, done=True, error=error_msg[:200])
            steps_taken = 1
            rewards = [0.0]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    await run_episode(client)


if __name__ == "__main__":
    asyncio.run(main())