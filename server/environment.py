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