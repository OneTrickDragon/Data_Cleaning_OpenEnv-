"""
tests/test_units.py — Unit tests for environment internals.
 
Tests grader accuracy, sandbox security, serialisation, and
dataset generation properties without any HTTP or FastAPI dependency.
 
Run:
    python tests/test_units.py
    python tests/test_units.py -v
"""

from __future__ import annotations
import sys
import textwrap
from pathlib import Path
 
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa
except ImportError:
    import types as _t
    _pm = _t.ModuleType("pydantic")
    class _BM:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
        def model_dump(self): return self.__dict__
    _pm.BaseModel = _BM
    _pm.Field = lambda default=None, **kw: default
    _pm.field_validator = lambda *a, **kw: (lambda f: f)
    sys.modules["pydantic"] = _pm
 
import pandas as pd
import numpy as np
 
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
results: list[tuple[str, bool, str]] = []

def expect(name: str, condition: bool, detail: str = "") -> None:
    icon = PASS if condition else FAIL
    print(f"  {icon} {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, condition, detail))
 
 
def is_string_dtype(series: pd.Series) -> bool:
    """True for both old object dtype and pandas 3.x StringDtype."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

print("\n── Dataset generation ──────────────────────────────────")
 
from server.environment import generate_datasets, TaskID
 
dirty_e, gold_e = generate_datasets(TaskID.ECOMMERCE_EASY, seed=42)
expect("ecommerce shape",             dirty_e.shape == (500, 8),              str(dirty_e.shape))
expect("ecommerce has order_date",    "order_date" in dirty_e.columns)
expect("order_date is string type",   is_string_dtype(dirty_e["order_date"]), str(dirty_e["order_date"].dtype))
expect("price has nulls",             dirty_e["price"].isna().sum() > 0,      str(dirty_e["price"].isna().sum()))
expect("quantity has negatives",      (dirty_e["quantity"] < 0).sum() > 0)
expect("revenue is string type",      is_string_dtype(dirty_e["revenue"]),    str(dirty_e["revenue"].dtype))
expect("gold revenue is float64",     pd.api.types.is_float_dtype(gold_e["revenue"]))
 
dirty_p, gold_p = generate_datasets(TaskID.PATIENT_RECORDS_MEDIUM, seed=42)
expect("patient dirty > gold rows",   dirty_p.shape[0] > gold_p.shape[0],
       f"{dirty_p.shape[0]} > {gold_p.shape[0]}")
expect("patient has dob column",      "dob" in dirty_p.columns)
expect("patient has mixed dob fmts",  not dirty_p["dob"].str.match(r"^\d{4}-\d{2}-\d{2}$").all())
 
dirty_f, gold_f = generate_datasets(TaskID.FINANCIAL_AUDIT_HARD, seed=42)
expect("financial shape rows",        dirty_f.shape[0] == 5000,              str(dirty_f.shape[0]))
expect("financial has txn_type",      "txn_type" in dirty_f.columns)
bad_refunds = dirty_f[(dirty_f["txn_type"] == "REFUND") & (dirty_f["amount"] > 0)]
expect("financial seeded violations", len(bad_refunds) > 0,                  f"{len(bad_refunds)} bad refunds")

d1, _ = generate_datasets(TaskID.ECOMMERCE_EASY, seed=99)
d2, _ = generate_datasets(TaskID.ECOMMERCE_EASY, seed=99)
d3, _ = generate_datasets(TaskID.ECOMMERCE_EASY, seed=100)
expect("same seed → same data",       d1["order_date"].tolist() == d2["order_date"].tolist())
expect("diff seed → diff data",       d1["order_date"].tolist() != d3["order_date"].tolist())

print("\n── Sandbox ──────────────────────────────────────────────")
 
from server.environment import run_in_sandbox
 
_df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, None, 6.0]})
 
r = run_in_sandbox("df['z'] = df['x'] * 2", _df)
expect("basic exec succeeds",         r.success)
expect("new column created",          "z" in r.df.columns)
expect("column value correct",        list(r.df["z"]) == [2, 4, 6])
 
r = run_in_sandbox("print('hello sandbox')", _df)
expect("stdout captured",             "hello sandbox" in r.stdout)
 
r = run_in_sandbox("df.dropna(inplace=True)", _df)
expect("dropna reduces rows",         len(r.df) == 2)
 
for mod in ["os", "subprocess", "socket", "pathlib", "shutil"]:
    r = run_in_sandbox(f"import {mod}", _df)
    expect(f"blocks import {mod}",    not r.success and "ImportError" in r.error)
 
r = run_in_sandbox("open('/etc/passwd', 'r')", _df)
expect("blocks open()",               not r.success)
 
r = run_in_sandbox("def broken(\n  pass", _df)
expect("syntax error handled",        not r.success and bool(r.error))
 
r = run_in_sandbox("df = 'not a dataframe'", _df)
expect("non-df assignment → original kept", isinstance(r.df, pd.DataFrame))

r = run_in_sandbox("df['root'] = df['x'].apply(lambda v: math.sqrt(v))", _df)
expect("math pre-injected",           r.success and "root" in r.df.columns)
 
r = run_in_sandbox("df['pat'] = re.sub(r'\\d', 'N', 'abc123')", _df)
expect("re pre-injected",             r.success)
 
r = run_in_sandbox("import re\ndf['pat2'] = re.sub(r'\\d','N','test9')", _df)
expect("explicit import re works",    r.success)

print("\n── Grader: ecommerce ────────────────────────────────────")
 
from server.environment import grade, partial_grade, _col_quality_ecommerce
 
dirty, gold = generate_datasets(TaskID.ECOMMERCE_EASY, seed=42)
 
base_score, _ = _col_quality_ecommerce(dirty, gold)
expect("uncleaned score < 0.9",       base_score < 0.9,             f"{base_score:.3f}")
 
r1 = run_in_sandbox("df['price'] = df['price'].fillna(df['price'].median())", dirty)
s1, _ = _col_quality_ecommerce(r1.df, gold)
expect("fixing price improves score", s1 > base_score,              f"{base_score:.3f}→{s1:.3f}")

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
expect("full clean exec succeeds",    r_full.success,               r_full.error[:80] if not r_full.success else "")
 
s_full, detail = _col_quality_ecommerce(r_full.df, gold)
expect("full clean score > 0.95",     s_full > 0.95,               f"{s_full:.3f}")
expect("order_date is datetime",      detail.get("order_date_dtype", 0) == 1.0)
expect("price nulls cleared",         detail.get("price_nulls", 0) == 1.0)
expect("status all valid",            detail.get("status_valid", 0) > 0.95)
expect("revenue is float",            detail.get("revenue_dtype", 0) == 1.0)
 
reward_full = grade(r_full.df, gold, TaskID.ECOMMERCE_EASY, step_count=6, had_crash=False)
expect("full reward total > 0.85",    reward_full.total > 0.85,    f"{reward_full.total:.4f}")
expect("no_crash bonus = 0.05",       reward_full.no_crash_bonus == 0.05)
expect("efficiency = 1.0 at 6 steps", reward_full.efficiency == 1.0)
 
reward_slow = grade(r_full.df, gold, TaskID.ECOMMERCE_EASY, step_count=18, had_crash=False)
expect("efficiency drops at 18 steps",reward_slow.efficiency < reward_full.efficiency,
       f"{reward_full.efficiency:.2f}→{reward_slow.efficiency:.2f}")
 
reward_crash = grade(r_full.df, gold, TaskID.ECOMMERCE_EASY, step_count=6, had_crash=True)
expect("crash → no_crash_bonus = 0",  reward_crash.no_crash_bonus == 0.0)
 
expect("breakdown is a dict",         isinstance(reward_full.breakdown, dict))

print("\n── Grader: patient records ──────────────────────────────")
 
from server.environment import _col_quality_patient
 
dirty_p, gold_p = generate_datasets(TaskID.PATIENT_RECORDS_MEDIUM, seed=42)
 
r_dedup = run_in_sandbox(
    "df = df.drop_duplicates(subset=['patient_id'], keep='first').reset_index(drop=True)",
    dirty_p,
)
s_before, _ = _col_quality_patient(dirty_p, gold_p)
s_after,  _ = _col_quality_patient(r_dedup.df, gold_p)
expect("dedup improves score",        s_after > s_before,          f"{s_before:.3f}→{s_after:.3f}")
expect("dedup reduces row count",     len(r_dedup.df) < len(dirty_p),
       f"{len(dirty_p)}→{len(r_dedup.df)}")
 
dob_code = textwrap.dedent("""\
    def _norm_dob(s):
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y']:
            try:
                return pd.to_datetime(str(s), format=fmt).strftime('%Y-%m-%d')
            except Exception:
                pass
        try:
            return pd.to_datetime(str(s), dayfirst=False).strftime('%Y-%m-%d')
        except Exception:
            return str(s)
    df['dob'] = df['dob'].apply(_norm_dob)
""")
r_dob = run_in_sandbox(dob_code, dirty_p)
iso_rate = r_dob.df["dob"].str.match(r"^\d{4}-\d{2}-\d{2}$").mean()
expect("DOB normalised to ISO-8601",  iso_rate > 0.80,             f"iso_rate={iso_rate:.2f}")

print("\n── Grader: financial audit ──────────────────────────────")
 
from server.environment import _col_quality_financial
 
dirty_f, gold_f = generate_datasets(TaskID.FINANCIAL_AUDIT_HARD, seed=42)
 
fin_code = textwrap.dedent("""\
    df.dropna(subset=['txn_id', 'account_id', 'amount', 'transaction_date'], inplace=True)
    df['violation'] = ''
    df['duplicate'] = False
    FX = {'USD': 1.0, 'EUR': 1.085, 'GBP': 1.265, 'JPY': 0.0067, 'CAD': 0.735}
    mask = df['currency'].isin(FX) & (df['currency'] != 'USD')
    df.loc[mask, 'usd_amount'] = (
        df.loc[mask, 'amount'] * df.loc[mask, 'currency'].map(FX)
    ).round(2)
""")
r_fin = run_in_sandbox(fin_code, dirty_f)
expect("financial exec succeeds",     r_fin.success,                r_fin.error[:80] if not r_fin.success else "")
 
s_fin, d_fin = _col_quality_financial(r_fin.df, gold_f)
expect("financial score > 0.6",       s_fin > 0.6,                 f"{s_fin:.3f}")
expect("violation col present",       d_fin.get("violation_col", 0) == 1.0)
expect("duplicate col is bool",       d_fin.get("duplicate_col", 0) == 1.0)
expect("FX reconciled",               d_fin.get("r4_fx", 0) > 0.9, f"{d_fin.get('r4_fx', 0):.3f}")
expect("R7 nulls cleared",            d_fin.get("r7_no_nulls", 0) >= 0.5)

print("\n── Dense partial reward ─────────────────────────────────")
 
dirty, gold = generate_datasets(TaskID.ECOMMERCE_EASY, seed=42)
s0 = partial_grade(dirty, gold, TaskID.ECOMMERCE_EASY)
 
r_s1 = run_in_sandbox("df['price'] = df['price'].fillna(df['price'].median())", dirty)
s1 = partial_grade(r_s1.df, gold, TaskID.ECOMMERCE_EASY)
 
r_s2 = run_in_sandbox(full_code, dirty)
s2 = partial_grade(r_s2.df, gold, TaskID.ECOMMERCE_EASY)
 
expect("partial score increases step 0→1", s1 >= s0, f"{s0:.3f}→{s1:.3f}")
expect("partial score increases step 1→2", s2 >= s1, f"{s1:.3f}→{s2:.3f}")
expect("partial score bounded [0,1]",      0.0 <= s2 <= 1.0)

print("\n── Serialisation ────────────────────────────────────────")
 
from server.environment import df_to_b64, b64_to_df
 
df_orig = pd.DataFrame({
    "a": [1, 2, 3],
    "b": pd.to_datetime(["2023-01-01", "2023-06-15", "2024-03-22"]),
    "c": [1.5, None, 3.5],
})
b64    = df_to_b64(df_orig)
df_rt  = b64_to_df(b64)
expect("shape preserved",             df_rt.shape == df_orig.shape, str(df_rt.shape))
expect("null preserved",              df_rt["c"].isna().sum() == 1)
expect("b64 is non-empty string",     isinstance(b64, str) and len(b64) > 0)
 
# Round-trip a large DataFrame
dirty2, _ = generate_datasets(TaskID.FINANCIAL_AUDIT_HARD, seed=1)
b64_large = df_to_b64(dirty2)
rt_large  = b64_to_df(b64_large)
expect("large df roundtrip shape",    rt_large.shape == dirty2.shape, str(dirty2.shape))

print("\n── build_observation ────────────────────────────────────")
 
from server.environment import build_observation
 
dirty, gold = generate_datasets(TaskID.ECOMMERCE_EASY, seed=42)
obs = build_observation(dirty, TaskID.ECOMMERCE_EASY, gold, 0, False)
expect("df_preview has table chars",  "|" in obs.df_preview)
expect("df_info has dtype info",      "dtype" in obs.df_info.lower() or "Dtype" in obs.df_info)
expect("task_spec has TASK header",   "TASK" in obs.task_spec)
expect("partial_score is float",      isinstance(obs.partial_score, float))
expect("step_count == 0",             obs.step_count == 0)
expect("done == False",               not obs.done)
 
obs2 = build_observation(dirty, TaskID.ECOMMERCE_EASY, gold, 5, True, "some output", "some error")
expect("exec_result propagated",      obs2.exec_result == "some output")
expect("error propagated",            obs2.error == "some error")
expect("done propagated",             obs2.done is True)
expect("step_count propagated",       obs2.step_count == 5)

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
