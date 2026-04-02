"""
tests/test_units.py — Unit tests for the Data Cleaning OpenEnv environment.

Tests the simulation core (dataset generation, sandbox, grader, serialisation)
and the OpenEnv model layer (dataclasses, environment interface).

Run:
    python tests/test_units.py
"""

from __future__ import annotations

import sys
import textwrap
import types
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

#Stub openenv-core so tests run without the package installed 
def _stub_openenv():
    oe      = types.ModuleType("openenv")
    oe_core = types.ModuleType("openenv.core")
    oe_srv  = types.ModuleType("openenv.core.env_server")
    oe_cli  = types.ModuleType("openenv.core.env_client")
    oe_ct   = types.ModuleType("openenv.core.client_types")

    @dataclass
    class Action:    pass
    @dataclass
    class Observation: pass
    @dataclass
    class State:     pass

    class Environment:
        is_concurrent_safe = False
        def __init__(self): pass

    class _GenericAlias:
        def __init__(self, origin, args): pass
    class EnvClient:
        def __class_getitem__(cls, item): return cls
        def __init__(self, base_url=""): self.base_url = base_url
        def _step_payload(self, a): raise NotImplementedError
        def _parse_result(self, p): raise NotImplementedError
        def _parse_state(self, p): raise NotImplementedError
        def sync(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass

    @dataclass
    class StepResult:
        observation: object = None
        reward: float = 0.0
        done: bool = False

    def create_fastapi_app(env, ac=None, oc=None):
        from fastapi import FastAPI
        return FastAPI()

    oe_srv.Action = Action
    oe_srv.Observation = Observation
    oe_srv.State = State
    oe_srv.Environment = Environment
    oe_srv.create_fastapi_app = create_fastapi_app
    oe_cli.EnvClient = EnvClient
    oe_ct.StepResult = StepResult

    for name, mod in [
        ("openenv", oe), ("openenv.core", oe_core),
        ("openenv.core.env_server", oe_srv),
        ("openenv.core.env_client", oe_cli),
        ("openenv.core.client_types", oe_ct),
    ]:
        sys.modules[name] = mod

    # pydantic stub (needed by environment.py)
    pm = types.ModuleType("pydantic")
    class BM:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
        def model_dump(self): return self.__dict__
    pm.BaseModel = BM
    pm.Field = lambda d=None, **k: d
    pm.field_validator = lambda *a, **k: (lambda f: f)
    sys.modules["pydantic"] = pm

try:
    import openenv.core.env_server  # noqa
except ImportError:
    _stub_openenv()

import pandas as pd
import numpy as np

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
results: list[tuple[str, bool, str]] = []


def expect(name: str, condition: bool, detail: str = "") -> None:
    icon = PASS if condition else FAIL
    print(f"  {icon} {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, condition, detail))


def is_string_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


# Models
print("\n── Models ───────────────────────────────────────────────")

from models import (
    DataCleaningAction, DataCleaningObservation,
    DataCleaningState, TaskID, ActionType,
)

a = DataCleaningAction(type="exec", code="df.dropna()", task_id="ecommerce_easy", seed=42)
expect("Action fields",        a.type == "exec" and a.code == "df.dropna()")
expect("Action defaults",      DataCleaningAction().type == "exec")

o = DataCleaningObservation(df_preview="| x |", step_count=3, partial_score=0.7, done=False)
expect("Observation fields",   o.partial_score == 0.7 and o.step_count == 3)
expect("Observation done=False", not o.done)

s = DataCleaningState(episode_id="abc", task_id="financial_audit_hard", step_count=5)
expect("State fields",         s.task_id == "financial_audit_hard" and s.step_count == 5)
expect("State had_crash=False", not s.had_crash)

expect("TaskID enum",          TaskID.ECOMMERCE_EASY.value == "ecommerce_easy")
expect("ActionType enum",      ActionType.SUBMIT.value == "submit")


# Dataset generation
print("\n── Dataset generation ───────────────────────────────────")

from server.environment import generate_datasets, TaskID as EnvTaskID

dirty_e, gold_e = generate_datasets(EnvTaskID.ECOMMERCE_EASY, seed=42)
expect("ecommerce shape",             dirty_e.shape == (500, 8),              str(dirty_e.shape))
expect("order_date is string",        is_string_dtype(dirty_e["order_date"]), str(dirty_e["order_date"].dtype))
expect("price has nulls",             dirty_e["price"].isna().sum() > 0,      str(dirty_e["price"].isna().sum()))
expect("quantity has negatives",      (dirty_e["quantity"] < 0).sum() > 0)
expect("revenue is string",           is_string_dtype(dirty_e["revenue"]),    str(dirty_e["revenue"].dtype))
expect("gold revenue is float64",     pd.api.types.is_float_dtype(gold_e["revenue"]))

dirty_p, gold_p = generate_datasets(EnvTaskID.PATIENT_RECORDS_MEDIUM, seed=42)
expect("patient dirty > gold rows",   dirty_p.shape[0] > gold_p.shape[0],
       f"{dirty_p.shape[0]} > {gold_p.shape[0]}")
expect("patient mixed dob formats",   not dirty_p["dob"].str.match(r"^\d{4}-\d{2}-\d{2}$").all())

dirty_f, gold_f = generate_datasets(EnvTaskID.FINANCIAL_AUDIT_HARD, seed=42)
expect("financial rows",              dirty_f.shape[0] == 5000,              str(dirty_f.shape[0]))
bad_refunds = dirty_f[(dirty_f["txn_type"] == "REFUND") & (dirty_f["amount"] > 0)]
expect("financial seeded violations", len(bad_refunds) > 0,                  f"{len(bad_refunds)} bad refunds")

d1, _ = generate_datasets(EnvTaskID.ECOMMERCE_EASY, seed=99)
d2, _ = generate_datasets(EnvTaskID.ECOMMERCE_EASY, seed=99)
d3, _ = generate_datasets(EnvTaskID.ECOMMERCE_EASY, seed=100)
expect("same seed → same data",       d1["order_date"].tolist() == d2["order_date"].tolist())
expect("diff seed → diff data",       d1["order_date"].tolist() != d3["order_date"].tolist())

# Sandbox
print("\n── Sandbox ──────────────────────────────────────────────")

from server.environment import run_in_sandbox

_df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, None, 6.0]})

r = run_in_sandbox("df['z'] = df['x'] * 2", _df)
expect("basic exec succeeds",         r.success)
expect("new column created",          "z" in r.df.columns)
expect("stdout captured",             "hello" in run_in_sandbox("print('hello')", _df).stdout)
expect("dropna works",                len(run_in_sandbox("df.dropna(inplace=True)", _df).df) == 2)

for mod in ["os", "subprocess", "socket", "pathlib", "shutil"]:
    r = run_in_sandbox(f"import {mod}", _df)
    expect(f"blocks import {mod}",    not r.success and "ImportError" in r.error)

expect("blocks open()",               not run_in_sandbox("open('/etc/passwd')", _df).success)
expect("syntax error handled",        not run_in_sandbox("def broken(\n  pass", _df).success)
expect("non-df assignment → kept",    isinstance(run_in_sandbox("df='str'", _df).df, pd.DataFrame))
expect("math pre-injected",           run_in_sandbox("df['r'] = math.sqrt(4)", _df).success)
expect("explicit import re works",    run_in_sandbox("import re\ndf['p']=re.sub(r'\\d','N','x9')", _df).success)

# Grader — ecommerce
print("\n── Grader: ecommerce ────────────────────────────────────")

from server.environment import (
    grade, partial_grade,
    _col_quality_ecommerce, _col_quality_patient, _col_quality_financial,
)

dirty, gold = generate_datasets(EnvTaskID.ECOMMERCE_EASY, seed=42)
base_score, _ = _col_quality_ecommerce(dirty, gold)
expect("uncleaned score < 0.9",       base_score < 0.9,              f"{base_score:.3f}")

r1 = run_in_sandbox("df['price'] = df['price'].fillna(df['price'].median())", dirty)
s1, _ = _col_quality_ecommerce(r1.df, gold)
expect("fixing price improves score", s1 > base_score,               f"{base_score:.3f}→{s1:.3f}")

full_code = textwrap.dedent("""\
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=False, errors='coerce')
    df['price'] = df['price'].fillna(df['price'].median())
    df['quantity'] = df['quantity'].clip(lower=0)
    def _parse_rev(v):
        s = str(v).replace(',', '.').replace('USD', '').strip().lstrip('$')
        try:
            return float(s)
        except ValueError:
            return float('nan')
    df['revenue'] = df['revenue'].apply(_parse_rev).astype(float)
    df['customer_id'] = df['customer_id'].str.strip()
    df['status'] = df['status'].str.lower().str.strip()
    df.loc[df['status'] == 'complete', 'status'] = 'delivered'
    valid = {'pending', 'shipped', 'delivered', 'cancelled'}
    df = df[df['status'].isin(valid)]
""")

r_full = run_in_sandbox(full_code, dirty)
expect("full clean exec ok",          r_full.success,                r_full.error[:60] if not r_full.success else "")
s_full, detail = _col_quality_ecommerce(r_full.df, gold)
expect("full clean score > 0.95",     s_full > 0.95,                f"{s_full:.3f}")
expect("order_date is datetime",      detail.get("order_date_dtype", 0) == 1.0)
expect("price nulls cleared",         detail.get("price_nulls", 0) == 1.0)
expect("revenue is float",            detail.get("revenue_dtype", 0) == 1.0)

reward = grade(r_full.df, gold, EnvTaskID.ECOMMERCE_EASY, step_count=6, had_crash=False)
expect("full reward > 0.85",          reward.total > 0.85,           f"{reward.total:.4f}")
expect("no_crash bonus = 0.05",       reward.no_crash_bonus == 0.05)
expect("efficiency = 1.0 at 6 steps", reward.efficiency == 1.0)
expect("efficiency drops at 18 steps",
       grade(r_full.df, gold, EnvTaskID.ECOMMERCE_EASY, 18, False).efficiency < 1.0)
expect("crash → no_crash_bonus = 0",
       grade(r_full.df, gold, EnvTaskID.ECOMMERCE_EASY, 6, True).no_crash_bonus == 0.0)


# Grader — patient records
print("\n── Grader: patient records ──────────────────────────────")

dirty_p, gold_p = generate_datasets(EnvTaskID.PATIENT_RECORDS_MEDIUM, seed=42)

r_dedup = run_in_sandbox(
    "df = df.drop_duplicates(subset=['patient_id'], keep='first').reset_index(drop=True)",
    dirty_p,
)
s_before, _ = _col_quality_patient(dirty_p, gold_p)
s_after,  _ = _col_quality_patient(r_dedup.df, gold_p)
expect("dedup improves score",        s_after > s_before,            f"{s_before:.3f}→{s_after:.3f}")
expect("dedup reduces row count",     len(r_dedup.df) < len(dirty_p),
       f"{len(dirty_p)}→{len(r_dedup.df)}")

# Grader — financial
print("\n── Grader: financial ────────────────────────────────────")

dirty_f, gold_f = generate_datasets(EnvTaskID.FINANCIAL_AUDIT_HARD, seed=42)

fin_code = textwrap.dedent("""\
    df.dropna(subset=['txn_id', 'account_id', 'amount', 'transaction_date'], inplace=True)
    df['violation'] = ''
    df['duplicate'] = False
    FX = {'USD': 1.0, 'EUR': 1.085, 'GBP': 1.265, 'JPY': 0.0067, 'CAD': 0.735}
    mask = df['currency'].isin(FX) & (df['currency'] != 'USD')
    df.loc[mask, 'usd_amount'] = (df.loc[mask,'amount'] * df.loc[mask,'currency'].map(FX)).round(2)
""")
r_fin = run_in_sandbox(fin_code, dirty_f)
expect("financial exec ok",           r_fin.success,                 r_fin.error[:60] if not r_fin.success else "")
s_fin, d_fin = _col_quality_financial(r_fin.df, gold_f)
expect("financial score > 0.6",       s_fin > 0.6,                  f"{s_fin:.3f}")
expect("violation col present",       d_fin.get("violation_col", 0) == 1.0)
expect("FX reconciled",               d_fin.get("r4_fx", 0) > 0.9,  f"{d_fin.get('r4_fx',0):.3f}")


# Dense partial reward
print("\n── Dense partial reward ─────────────────────────────────")

dirty, gold = generate_datasets(EnvTaskID.ECOMMERCE_EASY, seed=42)
s0 = partial_grade(dirty, gold, EnvTaskID.ECOMMERCE_EASY)
s1 = partial_grade(run_in_sandbox("df['price']=df['price'].fillna(df['price'].median())", dirty).df, gold, EnvTaskID.ECOMMERCE_EASY)
s2 = partial_grade(run_in_sandbox(full_code, dirty).df, gold, EnvTaskID.ECOMMERCE_EASY)
expect("score step 0→1 non-decreasing", s1 >= s0, f"{s0:.3f}→{s1:.3f}")
expect("score step 1→2 non-decreasing", s2 >= s1, f"{s1:.3f}→{s2:.3f}")
expect("partial score bounded [0,1]",   0.0 <= s2 <= 1.0)

# Serialisation
print("\n── Serialisation ────────────────────────────────────────")

from server.environment import df_to_b64, b64_to_df

df_orig = pd.DataFrame({"a": [1, 2, 3], "b": pd.to_datetime(["2023-01-01","2023-06-15","2024-03-22"]), "c": [1.5, None, 3.5]})
b64 = df_to_b64(df_orig)
rt  = b64_to_df(b64)
expect("shape preserved",             rt.shape == df_orig.shape,     str(rt.shape))
expect("null preserved",              rt["c"].isna().sum() == 1)
expect("b64 is non-empty string",     isinstance(b64, str) and len(b64) > 0)


# OpenEnv Environment interface
print("\n── Environment interface ────────────────────────────────")

from server.dc_environment import DataCleaningEnvironment

env = DataCleaningEnvironment()
expect("is_concurrent_safe",          env.is_concurrent_safe)

obs = env.reset()
expect("reset returns Observation",   isinstance(obs, DataCleaningObservation))
expect("reset step_count=0",          obs.step_count == 0)
expect("reset has task_spec",         "TASK" in obs.task_spec)
expect("reset partial_score in [0,1]",0.0 <= obs.partial_score <= 1.0)

obs2 = env.step(DataCleaningAction(type="exec", code="df['price']=df['price'].fillna(df['price'].median())"))
expect("step returns Observation",    isinstance(obs2, DataCleaningObservation))
expect("step increments step_count",  obs2.step_count == 1)
expect("step not done yet",           not obs2.done)
expect("step improves partial_score", obs2.partial_score >= obs.partial_score,
       f"{obs.partial_score:.3f}→{obs2.partial_score:.3f}")

obs3 = env.step(DataCleaningAction(type="submit"))
expect("submit is done",              obs3.done)
expect("submit reward > 0",           obs3.reward > 0.0, f"{obs3.reward:.4f}")

st = env.state
expect("state returns DataCleaningState", isinstance(st, DataCleaningState))
expect("state.done matches obs",      st.done)
expect("state.step_count correct",    st.step_count == 1)
expect("state.episode_id non-empty",  len(st.episode_id) > 0)

# reset idempotency
a = env.reset(DataCleaningAction(task_id="ecommerce_easy", seed=7))
b = env.reset(DataCleaningAction(task_id="ecommerce_easy", seed=7))
c = env.reset(DataCleaningAction(task_id="ecommerce_easy", seed=8))
expect("same seed → same preview",    a.df_preview == b.df_preview)
expect("diff seed → diff preview",    a.df_preview != c.df_preview)

# step-limit auto-terminates
env2 = DataCleaningEnvironment()
env2.reset()
obs_last = None
for _ in range(22):
    obs_last, _, done, _ = (None, None, False, None)
    obs_last = env2.step(DataCleaningAction(type="exec", code="pass"))
    if obs_last.done:
        break
expect("episode ends by step 20",     obs_last.done and env2.state.step_count <= 20,
       f"ended at step {env2.state.step_count}")


# =============================================================================
# Client serialisation methods
# =============================================================================
print("\n── Client serialisation ─────────────────────────────────")

from client import DataCleaningEnv

cl = DataCleaningEnv(base_url="http://localhost:8000")
p = cl._step_payload(DataCleaningAction(type="exec", code="df.dropna()", task_id="ecommerce_easy", seed=42))
expect("_step_payload keys",          set(p.keys()) == {"type","code","task_id","seed"})
expect("_step_payload values",        p["type"]=="exec" and p["code"]=="df.dropna()")

result = cl._parse_result({
    "observation": {"partial_score": 0.6, "step_count": 2, "done": False, "df_preview":"| a |"},
    "reward": 0.0, "done": False
})
expect("_parse_result observation",   isinstance(result.observation, DataCleaningObservation))
expect("_parse_result partial_score", result.observation.partial_score == 0.6)

st2 = cl._parse_state({"episode_id":"xyz","task_id":"financial_audit_hard","step_count":3,"done":True,"had_crash":False,"df_state_b64":"","gold_b64":"","seed":42})
expect("_parse_state task_id",        st2.task_id == "financial_audit_hard")
expect("_parse_state done",           st2.done)


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 55)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"  {passed}/{total} unit tests passed", end="")
if failed == 0:
    print("  \033[32m— all good!\033[0m\n")
    sys.exit(0)
else:
    print(f"  \033[31m— {failed} failed\033[0m\n")
    for name, ok, detail in results:
        if not ok:
            print(f"    \033[31m✗\033[0m {name}: {detail}")
    print()
    sys.exit(1)