from __future__ import annotations
 
import argparse
import json
import os
import sys
import textwrap
import time
from typing import Optional
 
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from client import DataCleaningEnvClient
from models import ActionType, TaskID

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert data engineer working inside a Python REPL environment.
 
    You will be given a messy pandas DataFrame called `df` and a task
    specification.  Your goal is to clean the DataFrame according to the
    specification and then submit it.
 
    Rules:
    - You interact with the environment by executing Python code.
    - `df` is already in scope.  Modify it in place or reassign.
    - Allowed imports: pandas (pd), numpy (np), re, datetime, difflib,
      unicodedata, collections, itertools, math, string.
    - No file I/O.  No network calls.  Max 50 lines per step.
    - After each step you will receive the updated df preview and a
      partial_score (0.0–1.0).  Use this signal to improve.
    - When satisfied, respond with exactly: ACTION: submit
 
    Respond with EITHER:
      ACTION: exec
      ```python
      <your code here>
      ```
 
    OR when finished:
      ACTION: submit
""").strip()
 
 
def parse_agent_response(response_text: str) -> tuple[ActionType, Optional[str]]:
    """Extract action type and optional code from the model's response."""
    text = response_text.strip()
 
    if "ACTION: submit" in text:
        return ActionType.SUBMIT, None
 
    # Extract code block
    import re
    code_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_match:
        return ActionType.EXEC, code_match.group(1).strip()
 
    # Fallback: treat everything after ACTION: exec as code
    if "ACTION: exec" in text:
        code = text.split("ACTION: exec", 1)[-1].strip()
        return ActionType.EXEC, code or "pass"
 
    # If nothing parseable, submit
    return ActionType.SUBMIT, None
 
 
def run_episode(
    env: DataCleaningEnvClient,
    llm: OpenAI,
    model: str,
    task_id: str,
    seed: int,
    verbose: bool = True,
) -> dict:
    """Run one full episode and return a results dict."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id}   Seed: {seed}   Model: {model}")
        print(f"{'='*60}")
 
    obs = env.reset(task_id=task_id, seed=seed)
 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"TASK SPECIFICATION:\n{obs.task_spec}\n\n"
            f"CURRENT DATAFRAME:\n{obs.df_preview}\n\n"
            f"DTYPES / INFO:\n{obs.df_info}\n\n"
            f"STATS:\n{obs.df_stats}\n\n"
            "Begin cleaning. Remember to submit when done."
        )},
    ]
 
    final_reward = None
    episode_start = time.time()
 
    for step_num in range(1, 22):
        # --- Call the model ---
        completion = llm.chat.completions.create(
            model       = model,
            messages    = messages,
            temperature = 0.0,
            max_tokens  = 1024,
        )
        response_text = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": response_text})
 
        action_type, code = parse_agent_response(response_text)
 
        if verbose:
            print(f"\nStep {step_num} → {action_type.value}")
            if code:
                print(textwrap.indent(code[:300] + ("..." if len(code) > 300 else ""), "    "))
 
        # --- Send action to environment ---
        from models import Action
        action = Action(type=action_type, code=code)
        obs, reward, done, info = env.step(action)
 
        if verbose:
            print(f"  partial_score={obs.partial_score:.3f}  done={done}")
            if obs.error:
                print(f"  ERROR: {obs.error[:200]}")
 
        if done:
            final_reward = reward
            break
 
        # Feed observation back to the model
        user_feedback = (
            f"EXEC RESULT:\n{obs.exec_result or '(no output)'}\n"
            + (f"ERROR: {obs.error}\n" if obs.error else "")
            + f"\nUPDATED DATAFRAME (first 10 rows):\n{obs.df_preview}\n"
            f"\nDTYPES:\n{obs.df_info}\n"
            f"\nPartial score so far: {obs.partial_score:.3f} / 1.0\n"
            f"Steps used: {obs.step_count} / 20\n\n"
            "Continue cleaning or submit."
        )
        messages.append({"role": "user", "content": user_feedback})
 
    elapsed = time.time() - episode_start
 
    result = {
        "task_id":    task_id,
        "seed":       seed,
        "model":      model,
        "steps":      obs.step_count,
        "elapsed_s":  round(elapsed, 1),
        "score":      final_reward.total             if final_reward else 0.0,
        "col_quality":final_reward.column_quality    if final_reward else 0.0,
        "schema":     final_reward.schema_compliance if final_reward else 0.0,
        "rows":       final_reward.row_preservation  if final_reward else 0.0,
        "efficiency": final_reward.efficiency        if final_reward else 0.0,
        "no_crash":   final_reward.no_crash_bonus    if final_reward else 0.0,
    }
 
    if verbose:
        print(f"\nFINAL SCORE: {result['score']:.4f}")
        print(json.dumps({k: v for k, v in result.items() if k not in ("task_id","seed","model")}, indent=2))
 
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for Data Cleaning OpenEnv")
    parser.add_argument("--model",    default="gpt-4o",               help="OpenAI model name")
    parser.add_argument("--env-url",  default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--task",     default=None,                    help="Single task ID to run (default: all)")
    parser.add_argument("--seed",     type=int, default=42,            help="RNG seed")
    parser.add_argument("--quiet",    action="store_true",             help="Suppress per-step output")
    args = parser.parse_args()
 
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
 
    llm = OpenAI(api_key=api_key)
    env = DataCleaningEnvClient(base_url=args.env_url)
 
    # Verify server is up
    try:
        h = env.health()
        print(f"Server healthy: {h}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment server at {args.env_url}\n  {e}")
        sys.exit(1)
 
    tasks_to_run = (
        [args.task] if args.task
        else ["ecommerce_easy", "patient_records_medium", "financial_audit_hard"]
    )
 
    all_results = []
    for task_id in tasks_to_run:
        result = run_episode(
            env     = env,
            llm     = llm,
            model   = args.model,
            task_id = task_id,
            seed    = args.seed,
            verbose = not args.quiet,
        )
        all_results.append(result)
 
    # Summary table
    print("\n" + "="*60)
    print("  BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Task':<30} {'Score':>7} {'Steps':>6} {'Time':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*8}")
    for r in all_results:
        print(f"  {r['task_id']:<30} {r['score']:>7.4f} {r['steps']:>6} {r['elapsed_s']:>7.1f}s")
 
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Average score: {avg:.4f}")
 
    # Save results
    out_path = "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")
 
 
if __name__ == "__main__":
    main()