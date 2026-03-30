from __future__ import annotations
 
import base64
import contextlib
import io
import math
import random
import re
import sys
import textwrap
import traceback
from typing import Any
 
import numpy as np
import pandas as pd
 
from models import (
    Action, ActionType, Observation, Reward, State, TaskID,
)


MAX_STEPS       = 20
STEP_EXEC_LIMIT = 50   # max lines of code per exec action
 
ALLOWED_IMPORTS = {
    "pandas", "numpy", "re", "datetime", "difflib",
    "unicodedata", "collections", "itertools", "math", "string",
}
 
TASK_SPECS: dict[TaskID, str] = {
    TaskID.ECOMMERCE_EASY: textwrap.dedent("""
        TASK: E-commerce order cleanup  [EASY]
 
        You have a DataFrame called `df` with 500 e-commerce orders.
 
        KNOWN ISSUES to fix:
        1. `order_date` is stored as a plain string — convert to datetime64[ns].
        2. `price` has ~15% nulls — impute with the column median.
        3. `quantity` has some negative values — clip to 0.
        4. `revenue` contains mixed formats like "$12.50", "12,50 USD", "12.50"
           — strip symbols, convert to float (USD).
        5. `customer_id` has leading/trailing whitespace — strip it.
        6. `status` values should be one of: pending, shipped, delivered, cancelled
           — lowercase and strip any others; map "complete" → "delivered".
 
        SCHEMA (expected output dtypes):
          order_id      object
          order_date    datetime64[ns]
          customer_id   object
          product_id    object
          quantity      int64
          price         float64
          revenue       float64
          status        object
 
        Submit your cleaned DataFrame with action type='submit'.
        You have at most 20 exec steps before automatic submission.
    """).strip(),
 
    TaskID.PATIENT_RECORDS_MEDIUM: textwrap.dedent("""
        TASK: Patient records deduplication  [MEDIUM]
 
        You have a DataFrame called `df` with ~1200 patient records.
 
        KNOWN ISSUES to fix:
        1. DUPLICATES: ~20% of records are fuzzy duplicates (same patient,
           slightly different name formatting: "John Smith" / "J. Smith" / "SMITH, John").
           Keep the most-complete record (fewest nulls). Use patient_id as the
           canonical key — records sharing a patient_id are duplicates.
        2. `dob` appears in three formats: "1985-03-22", "03/22/1985", "22-Mar-1985"
           — normalise all to ISO-8601 (YYYY-MM-DD) string.
        3. `phone` uses mixed delimiters: "(555) 123-4567", "555.123.4567", "5551234567"
           — normalise to E.164-ish format: "+15551234567".
        4. `icd10_notes` is a free-text field containing ICD-10 codes embedded
           in sentences — extract the first valid ICD-10 code (pattern: letter +
           2 digits + optional dot + up to 4 alphanumerics, e.g. "J45.9") into a
           new column `icd10_code`. Set to None if no code found.
        5. `email` — lowercase all email addresses.
 
        SCHEMA (expected output):
          patient_id    object
          first_name    object
          last_name     object
          dob           object  (YYYY-MM-DD string)
          phone         object  (+1XXXXXXXXXX)
          email         object
          icd10_code    object  (nullable)
          [icd10_notes may be dropped]
 
        Submit with action type='submit'.
    """).strip(),
 
    TaskID.FINANCIAL_AUDIT_HARD: textwrap.dedent("""
        TASK: Financial transaction audit  [HARD]
 
        You have a DataFrame called `df` with 5000 financial transactions.
        You must apply the following named BUSINESS RULES. Do not simply
        delete bad rows — flag them with a `violation` column (comma-separated
        rule names, or empty string if clean).
 
        BUSINESS RULES:
        R1  REFUND_SIGN:    rows where txn_type='REFUND' must have amount < 0.
        R2  DEBIT_POSITIVE: rows where txn_type='DEBIT'  must have amount > 0.
        R3  DATE_ORDER:     transaction_date must be >= account_open_date for
                            that account_id.
        R4  FX_RECONCILE:   for non-USD rows, usd_amount must equal
                            amount * fx_rate (within 0.01 tolerance).
                            Correct usd_amount where it is wrong.
        R5  PARENT_REF:     REFUND rows must have a parent_txn_id that exists
                            in the txn_id column. Flag those that don't.
        R6  DUPE_TXN:       exact duplicate (txn_id, account_id, amount, date)
                            should have duplicate=True in a new bool column.
        R7  NULL_REQUIRED:  account_id, txn_id, amount, transaction_date must
                            never be null. Drop rows where they are.
        R8  AMOUNT_OUTLIER: flag rows where abs(amount) > 3 standard deviations
                            from the per-account mean with rule name OUTLIER.
        R9  CURRENCY_VALID: currency must be one of USD, EUR, GBP, JPY, CAD.
                            Flag others as INVALID_CCY.
        R10 ZERO_AMOUNT:    amount == 0 is never valid. Flag as ZERO_AMOUNT.
 
        SCHEMA (expected output adds these columns to original):
          violation   object  (comma-separated rule names, "" if clean)
          duplicate   bool
          usd_amount  float64 (corrected per R4)
 
        FX rates for this dataset (treat as ground truth):
          EUR→USD: 1.085,  GBP→USD: 1.265,  JPY→USD: 0.0067,  CAD→USD: 0.735
 
        Submit with action type='submit'.
    """).strip(),
}

def _make_ecommerce(rng: random.Random, np_rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (dirty_df, gold_df) for the easy task."""
    n = 500
    statuses_clean = rng.choices(["pending", "shipped", "delivered", "cancelled"], k=n)
    dates = pd.date_range("2023-01-01", periods=n, freq="12h")
 
    gold = pd.DataFrame({
        "order_id":    [f"ORD-{i:05d}" for i in range(n)],
        "order_date":  dates,
        "customer_id": [f"CUST-{rng.randint(1, 200):04d}" for _ in range(n)],
        "product_id":  [f"PROD-{rng.randint(1, 50):03d}" for _ in range(n)],
        "quantity":    np_rng.integers(1, 20, size=n).astype("int64"),
        "price":       np_rng.uniform(1.5, 299.99, size=n).round(2),
        "revenue":     None,
        "status":      statuses_clean,
    })
    gold["revenue"] = (gold["quantity"] * gold["price"]).round(2)
    gold = gold.astype({"quantity": "int64", "price": "float64", "revenue": "float64"})
 
    dirty = gold.copy()

    fmt_choices = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]
    dirty["order_date"] = [
        d.strftime(rng.choice(fmt_choices)) for d in dirty["order_date"]
    ]

    null_idx = rng.sample(range(n), k=int(n * 0.15))
    dirty.loc[null_idx, "price"] = np.nan
 
    # 3. negative quantities
    neg_idx = rng.sample(range(n), k=int(n * 0.05))
    dirty.loc[neg_idx, "quantity"] = np_rng.integers(-10, 0, size=len(neg_idx))
 
    def fmt_rev(v: float) -> str:
        choice = rng.randint(0, 2)
        if choice == 0:   return f"${v:.2f}"
        elif choice == 1: return f"{v:.2f} USD"
        else:             return f"{str(v).replace('.', ',')}"
    dirty["revenue"] = dirty["revenue"].apply(fmt_rev)

    ws_idx = rng.sample(range(n), k=int(n * 0.2))
    dirty.loc[ws_idx, "customer_id"] = dirty.loc[ws_idx, "customer_id"].apply(
        lambda x: f"  {x}  " if rng.random() > 0.5 else f"\t{x}"
    )

    def corrupt_status(s: str) -> str:
        r = rng.random()
        if r < 0.1:  return s.upper()
        if r < 0.15: return "complete" if s == "delivered" else s
        return s
    dirty["status"] = dirty["status"].apply(corrupt_status)

    gold["revenue"] = (gold["quantity"].clip(lower=0) * gold["price"].fillna(gold["price"].median())).round(2)
 
    return dirty, gold

def _make_patient_records(rng: random.Random, np_rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (dirty_df, gold_df) for the medium task."""
    n_unique = 800
    first_names = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
                   "William","Barbara","David","Susan","Richard","Jessica","Joseph","Sarah"]
    last_names  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
                   "Wilson","Anderson","Taylor","Thomas","Jackson","White","Harris","Martin"]
    icd10_pool  = ["J45.9","I10","E11.9","Z00.00","M54.5","F32.1","K21.0","N39.0","J06.9","R51"]
 
    records = []
    for i in range(n_unique):
        pid   = f"PAT-{i:05d}"
        fname = rng.choice(first_names)
        lname = rng.choice(last_names)
        dob   = pd.Timestamp("1950-01-01") + pd.Timedelta(days=rng.randint(0, 25000))
        phone = f"+1{''.join([str(rng.randint(0,9)) for _ in range(10)])}"
        email = f"{fname.lower()}.{lname.lower()}{rng.randint(1,99)}@example.com"
        icd   = rng.choice(icd10_pool) if rng.random() > 0.2 else None
        records.append({
            "patient_id": pid, "first_name": fname, "last_name": lname,
            "dob": dob.strftime("%Y-%m-%d"), "phone": phone, "email": email,
            "icd10_code": icd, "icd10_notes": f"Patient presents with condition {icd}. Follow-up required." if icd else "No code.",
        })
 
    gold = pd.DataFrame(records)
 
    # Build dirty by duplicating ~20% with name mutations
    dirty_rows = list(records)
    dup_indices = rng.sample(range(n_unique), k=int(n_unique * 0.2))
    for idx in dup_indices:
        dup = dict(records[idx])
        # Mutate name
        mutation = rng.randint(0, 2)
        if mutation == 0:
            dup["first_name"] = dup["first_name"][0] + "."
        elif mutation == 1:
            dup["last_name"], dup["first_name"] = f"{dup['last_name'].upper()}, {dup['first_name']}", ""
        else:
            dup["first_name"] = dup["first_name"].lower()
        # Vary dob format
        dob_dt = pd.Timestamp(dup["dob"])
        fmt = rng.choice(["%m/%d/%Y", "%d-%b-%Y"])
        dup["dob"] = dob_dt.strftime(fmt)
        # Vary phone format
        digits = re.sub(r"\D", "", dup["phone"])[-10:]
        fmt_p = rng.randint(0, 2)
        if fmt_p == 0:   dup["phone"] = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif fmt_p == 1: dup["phone"] = f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
        else:            dup["phone"] = digits
        # Uppercase email sometimes
        if rng.random() > 0.5:
            dup["email"] = dup["email"].upper()
        # Null out a field to make "less complete"
        dup[rng.choice(["icd10_code", "icd10_notes"])] = None
        dirty_rows.append(dup)
 
    dirty = pd.DataFrame(dirty_rows).sample(frac=1, random_state=rng.randint(0, 9999)).reset_index(drop=True)
    return dirty, gold


def _make_financial_audit(rng: random.Random, np_rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (dirty_df, gold_df) for the hard task."""
    n = 5000
    currencies   = ["USD", "EUR", "GBP", "JPY", "CAD"]
    fx            = {"USD": 1.0, "EUR": 1.085, "GBP": 1.265, "JPY": 0.0067, "CAD": 0.735}
    txn_types    = ["DEBIT", "CREDIT", "REFUND", "TRANSFER"]
    account_ids  = [f"ACC-{i:04d}" for i in range(1, 101)]
    open_dates   = {a: pd.Timestamp("2018-01-01") + pd.Timedelta(days=rng.randint(0, 1500)) for a in account_ids}
 
    rows = []
    txn_ids_used = set()
    for i in range(n):
        txn_id  = f"TXN-{i:06d}"
        txn_ids_used.add(txn_id)
        acct    = rng.choice(account_ids)
        txn_t   = rng.choice(txn_types)
        ccy     = rng.choice(currencies)
        amount  = np_rng.uniform(10, 5000) * (1 if txn_t != "REFUND" else -1)
        amount  = round(amount, 2)
        open_d  = open_dates[acct]
        txn_d   = open_d + pd.Timedelta(days=rng.randint(1, 2000))
        usd_amt = round(amount * fx[ccy], 2)
        parent  = None
        if txn_t == "REFUND":
            # valid parent (will exist in dataset)
            parent = f"TXN-{rng.randint(0, max(0, i-1)):06d}" if i > 0 else None
        rows.append({
            "txn_id": txn_id, "account_id": acct, "txn_type": txn_t,
            "currency": ccy, "amount": amount, "usd_amount": usd_amt,
            "fx_rate": fx[ccy], "transaction_date": txn_d.strftime("%Y-%m-%d"),
            "account_open_date": open_d.strftime("%Y-%m-%d"),
            "parent_txn_id": parent,
        })
 
    gold_base = pd.DataFrame(rows)
    gold_base["violation"] = ""
    gold_base["duplicate"] = False
 
    #Corrupt data
    dirty = gold_base.copy().drop(columns=["violation", "duplicate"])
 
    def corrupt_pct(col, pct, fn):
        idx = rng.sample(range(n), k=int(n * pct))
        for i in idx:
            dirty.at[i, col] = fn(dirty.at[i, col])
 
    # R1: flip sign on some refunds
    refund_idx = dirty[dirty["txn_type"] == "REFUND"].index.tolist()
    for i in rng.sample(refund_idx, k=max(1, len(refund_idx)//10)):
        dirty.at[i, "amount"] = abs(dirty.at[i, "amount"])
 
    # R4: introduce FX errors in ~5% of non-USD
    non_usd = dirty[dirty["currency"] != "USD"].index.tolist()
    for i in rng.sample(non_usd, k=max(1, len(non_usd)//20)):
        dirty.at[i, "usd_amount"] = round(dirty.at[i, "usd_amount"] * 1.1, 2)
 
    # R5: break parent_txn_id for ~10% of refunds
    for i in rng.sample(refund_idx, k=max(1, len(refund_idx)//10)):
        dirty.at[i, "parent_txn_id"] = "TXN-INVALID-999"
 
    # R7: introduce nulls in required columns
    for col in ["account_id", "amount"]:
        for i in rng.sample(range(n), k=10):
            dirty.at[i, col] = None
 
    # R9: introduce invalid currencies
    for i in rng.sample(range(n), k=20):
        dirty.at[i, "currency"] = rng.choice(["XYZ", "BTC", "FOO"])
 
    # R10: zero amounts
    for i in rng.sample(range(n), k=15):
        dirty.at[i, "amount"] = 0.0
 
    return dirty, gold_base

_GENERATORS = {
    TaskID.ECOMMERCE_EASY:         _make_ecommerce,
    TaskID.PATIENT_RECORDS_MEDIUM: _make_patient_records,
    TaskID.FINANCIAL_AUDIT_HARD:   _make_financial_audit,
}
 
def generate_datasets(task_id: TaskID, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dirty, gold) DataFrames for the given task and seed."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    return _GENERATORS[task_id](rng, np_rng)

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.MULTILINE)
BLOCKED_BUILTINS = {"open", "compile", "breakpoint", "input"}

def _check_imports(code: str) -> list[str]:
    """Return list of disallowed top-level import names found in code."""
    bad = []
    for m in _IMPORT_RE.finditer(code):
        root = m.group(1).split(".")[0]
        if root not in ALLOWED_IMPORTS:
            bad.append(root)
    return bad
 
class SandboxResult:
    def __init__(self, df: pd.DataFrame, stdout: str, stderr: str, error: str):
        self.df     = df
        self.stdout = stdout
        self.stderr = stderr
        self.error  = error
 
    @property
    def success(self) -> bool:
        return not self.error
    
def run_in_sandbox(code: str, df: pd.DataFrame) -> SandboxResult:
    """Execute `code` with `df` in scope inside a restricted namespace."""
    bad_imports = _check_imports(code)
    if bad_imports:
        return SandboxResult(df, "", "", f"ImportError: disallowed module(s): {bad_imports}")
 
    # Restricted builtins
    safe_builtins = {k: v for k, v in __builtins__.items()
                     if k not in BLOCKED_BUILTINS} if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k) for k in dir(__builtins__)
        if k not in BLOCKED_BUILTINS and not k.startswith("__")
    }
 
    import collections, itertools, math, string, datetime, difflib, unicodedata
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "re": re,
        "datetime": datetime,
        "difflib": difflib,
        "unicodedata": unicodedata,
        "collections": collections,
        "itertools": itertools,
        "math": math,
        "string": string,
    }
 
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    error_msg  = ""
 
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, "<agent>", "exec"), namespace)  # noqa: S102
        result_df = namespace.get("df", df)
        if not isinstance(result_df, pd.DataFrame):
            error_msg = "TypeError: `df` must remain a pandas DataFrame"
            result_df = df
    except Exception:
        error_msg = traceback.format_exc(limit=5)
        result_df = df
 
    return SandboxResult(
        df     = result_df,
        stdout = stdout_buf.getvalue()[:2000],
        stderr = stderr_buf.getvalue()[:500],
        error  = error_msg,
    )

def _col_quality_ecommerce(output: pd.DataFrame, gold: pd.DataFrame) -> tuple[float, dict]:
    detail: dict[str, float] = {}
 
    # order_date dtype
    detail["order_date_dtype"] = 1.0 if pd.api.types.is_datetime64_any_dtype(output.get("order_date", pd.Series(dtype=object))) else 0.0
 
    # price nulls
    if "price" in output.columns:
        null_rate = output["price"].isna().mean()
        detail["price_nulls"] = max(0.0, 1.0 - null_rate * 5)
    else:
        detail["price_nulls"] = 0.0
 
    # quantity no negatives
    if "quantity" in output.columns:
        neg_rate = (output["quantity"] < 0).mean()
        detail["quantity_nonneg"] = 1.0 - float(neg_rate)
    else:
        detail["quantity_nonneg"] = 0.0
 
    # revenue is float
    detail["revenue_dtype"] = 1.0 if pd.api.types.is_float_dtype(output.get("revenue", pd.Series(dtype=object))) else 0.0
 
    # customer_id no whitespace
    if "customer_id" in output.columns:
        ws_rate = output["customer_id"].str.contains(r"^\s|\s$", na=False).mean()
        detail["customer_id_stripped"] = 1.0 - float(ws_rate)
    else:
        detail["customer_id_stripped"] = 0.0
 
    # status valid values
    valid = {"pending", "shipped", "delivered", "cancelled"}
    if "status" in output.columns:
        valid_rate = output["status"].isin(valid).mean()
        detail["status_valid"] = float(valid_rate)
    else:
        detail["status_valid"] = 0.0
 
    score = sum(detail.values()) / len(detail)
    return score, detail

def _col_quality_patient(output: pd.DataFrame, gold: pd.DataFrame) -> tuple[float, dict]:
    detail: dict[str, float] = {}
 
    # dedup: row count close to gold
    detail["dedup"] = min(1.0, gold.shape[0] / max(1, output.shape[0]))
 
    # dob ISO format
    if "dob" in output.columns:
        iso_rate = output["dob"].dropna().str.match(r"^\d{4}-\d{2}-\d{2}$").mean()
        detail["dob_iso"] = float(iso_rate) if len(output["dob"].dropna()) else 0.0
    else:
        detail["dob_iso"] = 0.0
 
    # phone E.164-ish
    if "phone" in output.columns:
        e164 = output["phone"].dropna().str.match(r"^\+1\d{10}$").mean()
        detail["phone_e164"] = float(e164) if len(output["phone"].dropna()) else 0.0
    else:
        detail["phone_e164"] = 0.0
 
    # icd10_code column exists
    detail["icd10_extracted"] = 1.0 if "icd10_code" in output.columns else 0.0
 
    # email lowercase
    if "email" in output.columns:
        lc_rate = output["email"].dropna().str.match(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]+$").mean()
        detail["email_lower"] = float(lc_rate) if len(output["email"].dropna()) else 0.0
    else:
        detail["email_lower"] = 0.0
 
    score = sum(detail.values()) / len(detail)
    return score, detail

def _col_quality_financial(output: pd.DataFrame, gold: pd.DataFrame) -> tuple[float, dict]:
    FX = {"USD": 1.0, "EUR": 1.085, "GBP": 1.265, "JPY": 0.0067, "CAD": 0.735}
    detail: dict[str, float] = {}
 
    # violation column exists
    detail["violation_col"] = 1.0 if "violation" in output.columns else 0.0
 
    # duplicate col exists + is bool
    detail["duplicate_col"] = (
        1.0 if "duplicate" in output.columns and pd.api.types.is_bool_dtype(output["duplicate"])
        else 0.0
    )
 
    # R7: no nulls in required cols
    if all(c in output.columns for c in ["account_id", "txn_id", "amount", "transaction_date"]):
        null_ok = all(output[c].notna().all() for c in ["account_id", "txn_id", "amount", "transaction_date"])
        detail["r7_no_nulls"] = 1.0 if null_ok else 0.5
    else:
        detail["r7_no_nulls"] = 0.0
 
    # R4: usd_amount corrected (check non-USD rows)
    if all(c in output.columns for c in ["currency", "amount", "usd_amount"]):
        non_usd = output[output["currency"].isin(FX) & (output["currency"] != "USD")]
        if len(non_usd):
            expected = non_usd["amount"] * non_usd["currency"].map(FX)
            correct  = (non_usd["usd_amount"] - expected).abs() < 0.02
            detail["r4_fx"] = float(correct.mean())
        else:
            detail["r4_fx"] = 1.0
    else:
        detail["r4_fx"] = 0.0
 
    # R1: refund sign violations flagged
    if "violation" in output.columns and "txn_type" in output.columns and "amount" in output.columns:
        refunds     = output[output["txn_type"] == "REFUND"]
        bad_refunds = refunds[refunds["amount"] > 0]
        if len(bad_refunds):
            flagged = bad_refunds["violation"].str.contains("REFUND_SIGN", na=False).mean()
            detail["r1_refund_flagged"] = float(flagged)
        else:
            detail["r1_refund_flagged"] = 1.0
    else:
        detail["r1_refund_flagged"] = 0.0
 
    score = sum(detail.values()) / len(detail)
    return score, detail
 
 
_COL_GRADERS = {
    TaskID.ECOMMERCE_EASY:         _col_quality_ecommerce,
    TaskID.PATIENT_RECORDS_MEDIUM: _col_quality_patient,
    TaskID.FINANCIAL_AUDIT_HARD:   _col_quality_financial,
}
 
EXPECTED_COLUMNS: dict[TaskID, list[str]] = {
    TaskID.ECOMMERCE_EASY:         ["order_id","order_date","customer_id","product_id","quantity","price","revenue","status"],
    TaskID.PATIENT_RECORDS_MEDIUM: ["patient_id","first_name","last_name","dob","phone","email","icd10_code"],
    TaskID.FINANCIAL_AUDIT_HARD:   ["txn_id","account_id","txn_type","currency","amount","usd_amount","fx_rate",
                                    "transaction_date","account_open_date","parent_txn_id","violation","duplicate"],
}

