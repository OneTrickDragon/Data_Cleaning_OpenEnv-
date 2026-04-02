"""
train_grpo.py — GRPO training for the Data Cleaning OpenEnv.

Uses the real OpenEnv sync() client interface. The environment is driven
through DataCleaningEnv(...).sync() which wraps the async WebSocket client.

Usage:
    # Quick smoke test (no GPU needed, 5 episodes)
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --smoke-test

    # Full training run
    python train_grpo.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --episodes 500 \\
        --group-size 8 \\
        --env-url http://localhost:8000

    # Against a HF Space
    python train_grpo.py \\
        --env-url https://YOUR-USERNAME-data-cleaning-openenv.hf.space \\
        --episodes 500

    # Curriculum (easy → medium → hard)
    python train_grpo.py --curriculum --episodes 1000

Requirements:
    pip install torch transformers accelerate tqdm "openenv-core[core]>=0.2.1"
    # Environment server must be running:
    uvicorn server.app:app --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Stub openenv-core for syntax checks without the package ──────────────────
try:
    from openenv.core.env_client import EnvClient  # noqa
except ImportError:
    import types as _t
    for _n in ["openenv","openenv.core","openenv.core.env_server",
               "openenv.core.env_client","openenv.core.client_types"]:
        sys.modules.setdefault(_n, _t.ModuleType(_n))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from client import DataCleaningEnv
from models import DataCleaningAction

# Config
@dataclass
class GRPOConfig:
    model_name:      str   = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens:  int   = 512
    temperature:     float = 0.8
    top_p:           float = 0.95

    group_size:      int   = 8        # G: completions sampled per prompt seed
    clip_eps:        float = 0.2      # PPO-clip epsilon
    entropy_bonus:   float = 0.001

    episodes:        int   = 500
    steps_per_ep:    int   = 5        # max exec steps before auto-submit
    lr:              float = 5e-6
    grad_clip:       float = 1.0
    warmup_steps:    int   = 20
    update_every:    int   = 4        # accumulate N episodes before gradient step

    step_penalty:    float = 0.02     # per-step cost → encourages efficiency
    crash_penalty:   float = 0.10     # penalty for sandbox exceptions

    task_id:         str   = "ecommerce_easy"
    curriculum:      bool  = False
    seed_range:      int   = 1000

    env_url:         str   = "http://localhost:8000"
    save_dir:        str   = "./checkpoints"
    log_every:       int   = 10
    eval_every:      int   = 50
    eval_episodes:   int   = 20
    smoke_test:      bool  = False
    device:          str   = "auto"


# Prompt helpers
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a data engineering expert inside a Python REPL.
    Clean the pandas DataFrame called `df` according to the task spec.
    Available: pd, np, re, math, datetime, difflib (all pre-injected).
    Write ONLY Python code — no markdown, no explanation.
    When done, write exactly: SUBMIT
""").strip()


def build_prompt(tokenizer, obs, step: int, prev_result: str = "") -> str:
    parts = [
        f"TASK:\n{obs.task_spec}",
        f"DF (first 10 rows):\n{obs.df_preview}",
        f"PARTIAL SCORE: {obs.partial_score:.3f}  STEP: {step}",
    ]
    if prev_result:
        parts.append(f"LAST OUTPUT:\n{prev_result[:200]}")
    if obs.error:
        parts.append(f"ERROR: {obs.error[:150]}")
    parts.append("Code (or SUBMIT):")
    user_msg = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_completion(text: str) -> tuple[str, bool]:
    text = text.strip()
    if text.upper().startswith("SUBMIT") or not text:
        return "", True
    text = re.sub(r"```(?:python)?\n?", "", text).strip("`").strip()
    return text, False

# Curriculum
CURRICULUM = [
    ("ecommerce_easy",         0,   200),
    ("patient_records_medium", 200, 600),
    ("financial_audit_hard",   600, 9999),
]

def get_task(episode: int, curriculum: bool, fixed_task: str) -> str:
    if not curriculum:
        return fixed_task
    for task, start, end in CURRICULUM:
        if start <= episode < end:
            return task
    return "financial_audit_hard"

# Episode rollout (uses real OpenEnv sync client)
@dataclass
class StepRecord:
    prompt:     str
    completion: str
    reward:     float


@dataclass
class EpisodeRecord:
    steps:       list[StepRecord]
    final_score: float
    task_id:     str
    seed:        int


def rollout_group(
    policy_model,
    tokenizer,
    cfg: GRPOConfig,
    task_id: str,
    seed: int,
) -> list[EpisodeRecord]:
    """
    Run cfg.group_size independent episodes against the OpenEnv server.
    Each episode uses the real DataCleaningEnv sync client.
    Returns one EpisodeRecord per group member.
    """
    records = []

    for _ in range(cfg.group_size):
        step_records = []
        prev_result  = ""
        prev_score   = 0.0

        with DataCleaningEnv(base_url=cfg.env_url).sync() as env:
            result = env.reset(task_id=task_id, seed=seed)
            obs    = result.observation
            prev_score = obs.partial_score

            for step in range(cfg.steps_per_ep):
                prompt = build_prompt(tokenizer, obs, step, prev_result)

                # Sample one completion from policy
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=2048).to(cfg.device)
                with torch.no_grad():
                    output = policy_model.generate(
                        **inputs,
                        max_new_tokens    = cfg.max_new_tokens,
                        do_sample         = True,
                        temperature       = cfg.temperature,
                        top_p             = cfg.top_p,
                        pad_token_id      = tokenizer.pad_token_id,
                    )
                completion = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                code, submit = parse_completion(completion)

                if submit or step == cfg.steps_per_ep - 1:
                    result = env.step(DataCleaningAction(type="submit"))
                    step_reward = result.observation.reward - prev_score
                else:
                    result = env.step(DataCleaningAction(
                        type="exec", code=code if code else "pass"
                    ))
                    obs = result.observation
                    step_reward = (
                        (obs.partial_score - prev_score)
                        - cfg.step_penalty
                        - (cfg.crash_penalty if obs.error else 0.0)
                    )
                    prev_score  = obs.partial_score
                    prev_result = (obs.exec_result or obs.error)[:200]

                step_records.append(StepRecord(prompt, completion, step_reward))

                if result.observation.done:
                    break

            final_score = result.observation.reward if result.observation.done else result.observation.partial_score

        records.append(EpisodeRecord(
            steps=step_records, final_score=final_score,
            task_id=task_id, seed=seed,
        ))

    return records

# GRPO loss
def grpo_loss(
    policy_model,
    tokenizer,
    episodes: list[EpisodeRecord],
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO objective (DeepSeek-R1 style, no critic):
      1. Group episodes by seed.
      2. Compute returns per episode.
      3. Normalise within group: advantage_i = (R_i - mean) / (std + eps).
      4. Policy gradient: loss = -advantage * log_prob(completion | prompt).
      5. PPO-clip for stability.
    """
    total_loss    = torch.zeros(1, device=cfg.device, requires_grad=True)
    n_steps_total = 0
    metrics = {"policy_loss": 0.0, "mean_return": 0.0}

    groups: dict[int, list[EpisodeRecord]] = {}
    for ep in episodes:
        groups.setdefault(ep.seed, []).append(ep)

    for seed, group in groups.items():
        returns = torch.tensor(
            [sum(s.reward for s in ep.steps) + ep.final_score for ep in group],
            dtype=torch.float32, device=cfg.device,
        )
        metrics["mean_return"] += returns.mean().item()

        if len(group) > 1 and returns.std().item() > 1e-6:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantages = torch.zeros_like(returns)

        adv_list = [advantages[i] for i in range(len(group))]

        for ep_idx, ep in enumerate(group):
            adv = adv_list[ep_idx]
            for step in ep.steps:
                if not step.completion.strip():
                    continue
                full = step.prompt + step.completion
                enc  = tokenizer(full, return_tensors="pt", truncation=True,
                                 max_length=2048).to(cfg.device)
                enc_p = tokenizer(step.prompt, return_tensors="pt",
                                  truncation=True, max_length=2048).to(cfg.device)
                plen  = enc_p["input_ids"].shape[1]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = policy_model(**enc).logits

                shift_logits = logits[0, :-1]
                shift_labels = enc["input_ids"][0, 1:]
                lp = F.log_softmax(shift_logits, dim=-1)
                token_lp = lp[range(len(shift_labels)), shift_labels]
                completion_lp = token_lp[plen - 1:].sum()

                ratio   = torch.exp(completion_lp - completion_lp.detach())
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                step_loss = torch.min(ratio * adv, clipped * adv) * torch.tensor(-1.0, device=cfg.device)

                total_loss = total_loss + step_loss
                n_steps_total += 1

    if n_steps_total > 0:
        total_loss = total_loss / n_steps_total
        metrics["policy_loss"] = total_loss.item()
        metrics["mean_return"] /= max(1, len(groups))

    return total_loss, metrics

# Evaluation
def evaluate(policy_model, tokenizer, cfg: GRPOConfig,
             task_id: str, n_episodes: int, rng: random.Random) -> dict:
    old_temp = cfg.temperature
    cfg.temperature = 0.1
    scores = []
    for _ in range(n_episodes):
        seed = rng.randint(cfg.seed_range, cfg.seed_range * 2)
        eps  = rollout_group(policy_model, tokenizer, cfg, task_id, seed)
        scores.append(eps[0].final_score)
    cfg.temperature = old_temp
    mean = sum(scores) / len(scores)
    std  = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return {"mean": mean, "std": std, "min": min(scores), "max": max(scores)}

# Training loop
def train(cfg: GRPOConfig) -> None:
    print(f"\n{'='*60}\n  GRPO Training — Data Cleaning OpenEnv\n{'='*60}")
    print(f"  Model:      {cfg.model_name}")
    print(f"  Env:        {cfg.env_url}")
    print(f"  Task:       {cfg.task_id}" + (" (curriculum)" if cfg.curriculum else ""))
    print(f"  Episodes:   {cfg.episodes}  |  Group size: {cfg.group_size}")

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    ) if cfg.device == "auto" else cfg.device
    cfg.device = device

    print(f"  Device:     {device}\n")

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, s / max(1, cfg.warmup_steps))
    )

    rng         = random.Random(42)
    save_dir    = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_history:  list[dict] = []
    eval_history: list[dict] = []
    ep_buffer:    list[EpisodeRecord] = []
    update_step   = 0
    best_mean     = 0.0

    # Initial eval
    task0 = get_task(0, cfg.curriculum, cfg.task_id)
    print("  Initial evaluation...")
    init_eval = evaluate(model, tokenizer, cfg, task0, min(3, cfg.eval_episodes), rng)
    print(f"  Initial score: {init_eval['mean']:.4f} ± {init_eval['std']:.4f}\n")
    eval_history.append({"episode": 0, "task": task0, **init_eval})

    pbar = tqdm(range(cfg.episodes), desc="Training", unit="ep")
    for episode in pbar:
        if cfg.smoke_test and episode >= 5:
            print("\n  Smoke test done."); break

        task_id = get_task(episode, cfg.curriculum, cfg.task_id)
        seed    = rng.randint(0, cfg.seed_range)

        model.eval()
        group = rollout_group(model, tokenizer, cfg, task_id, seed)
        ep_buffer.extend(group)
        mean_score = sum(e.final_score for e in group) / len(group)

        # Policy update
        if (episode + 1) % cfg.update_every == 0 and ep_buffer:
            model.train()
            optimizer.zero_grad()
            loss, metrics = grpo_loss(model, tokenizer, ep_buffer, cfg)
            if torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                update_step += 1
            ep_buffer.clear()
            model.eval()
            pbar.set_postfix(score=f"{mean_score:.3f}", loss=f"{metrics['policy_loss']:.4f}", upd=update_step)

        # Logging
        if (episode + 1) % cfg.log_every == 0:
            log_history.append({"episode": episode+1, "task": task_id, "mean_score": mean_score})
            (save_dir / "train_log.json").write_text(json.dumps(log_history, indent=2))

        # Evaluation
        if (episode + 1) % cfg.eval_every == 0:
            eval_task = get_task(episode, cfg.curriculum, cfg.task_id)
            res = evaluate(model, tokenizer, cfg, eval_task, cfg.eval_episodes, rng)
            res.update({"episode": episode+1, "task": eval_task})
            eval_history.append(res)
            (save_dir / "eval_log.json").write_text(json.dumps(eval_history, indent=2))
            tqdm.write(f"\n  [Eval ep={episode+1}] {eval_task}  mean={res['mean']:.4f} ± {res['std']:.4f}")
            if res["mean"] > best_mean:
                best_mean = res["mean"]
                model.save_pretrained(save_dir / "best")
                tokenizer.save_pretrained(save_dir / "best")
                tqdm.write(f"  ✓ New best: {best_mean:.4f}")

    # Final eval + save
    final_task = get_task(cfg.episodes, cfg.curriculum, cfg.task_id)
    final_eval = evaluate(model, tokenizer, cfg, final_task, cfg.eval_episodes, rng)
    print(f"\n  Final score:   {final_eval['mean']:.4f} ± {final_eval['std']:.4f}")
    print(f"  Initial score: {init_eval['mean']:.4f}")
    print(f"  Improvement:   {final_eval['mean'] - init_eval['mean']:+.4f}")

    model.save_pretrained(save_dir / "final")
    tokenizer.save_pretrained(save_dir / "final")
    summary = {
        "model": cfg.model_name, "task": cfg.task_id,
        "initial_score": init_eval["mean"], "final_score": final_eval["mean"],
        "improvement": final_eval["mean"] - init_eval["mean"],
        "update_steps": update_step,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved to: {save_dir}")


# CLI
def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",        default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--episodes",     type=int,   default=500)
    p.add_argument("--group-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-6)
    p.add_argument("--task",         default="ecommerce_easy",
                   choices=["ecommerce_easy","patient_records_medium","financial_audit_hard"])
    p.add_argument("--curriculum",   action="store_true")
    p.add_argument("--env-url",      default="http://localhost:8000")
    p.add_argument("--save-dir",     default="./checkpoints")
    p.add_argument("--eval-every",   type=int,   default=50)
    p.add_argument("--eval-episodes",type=int,   default=20)
    p.add_argument("--device",       default="auto")
    p.add_argument("--smoke-test",   action="store_true")
    args = p.parse_args()

    cfg = GRPOConfig(
        model_name    = args.model,
        episodes      = args.episodes,
        group_size    = args.group_size,
        lr            = args.lr,
        task_id       = args.task,
        curriculum    = args.curriculum,
        env_url       = args.env_url,
        save_dir      = args.save_dir,
        eval_every    = args.eval_every,
        eval_episodes = args.eval_episodes,
        device        = args.device,
        smoke_test    = args.smoke_test,
    )
    train(cfg)


if __name__ == "__main__":
    main()