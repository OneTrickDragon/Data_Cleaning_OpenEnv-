from __future__ import annotations
 
import argparse
import sys
import threading
import time
from pathlib import Path

try:
    import pydantic  # noqa
except ImportError:
    import types as _types
    _pm = _types.ModuleType("pydantic")
    class _BM:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
        def model_dump(self): return self.__dict__
    _pm.BaseModel = _BM
    _pm.Field = lambda default=None, **kw: default
    _pm.field_validator = lambda *a, **kw: (lambda f: f)
    sys.modules["pydantic"] = _pm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SERVER_URL = "http://127.0.0.1:18765"
_server_thread = None
 
 
def _try_start_server() -> bool:
    """Attempt to start the uvicorn server. Returns True if successful."""
    try:
        import uvicorn
        import httpx
    except ImportError:
        return False
 
    def _run():
        uvicorn.run("server.app:app", host="127.0.0.1", port=18765, log_level="error")
 
    global _server_thread
    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()
    # Wait up to 3s for the server to come up
    import httpx
    for _ in range(15):
        time.sleep(0.2)
        try:
            httpx.get(f"{SERVER_URL}/health", timeout=1)
            return True
        except Exception:
            pass
    return False

class DirectClient:
    """
    Exercises the environment logic directly without HTTP.
    Mirrors the HTTP client interface for test compatibility.
    """
 
    def __init__(self):
        from server.environment import (
            generate_datasets, build_observation, run_in_sandbox,
            grade, partial_grade, df_to_b64, b64_to_df, MAX_STEPS, TaskID
        )
        self._gen   = generate_datasets
        self._build = build_observation
        self._exec  = run_in_sandbox
        self._grade = grade
        self._b64   = df_to_b64
        self._unb64 = b64_to_df
        self._MAX   = MAX_STEPS
        self._TaskID = TaskID
 
        self._df_b64    = None
        self._gold_b64  = None
        self._task_id   = None
        self._steps     = 0
        self._done      = False
        self._had_crash = False
 
    def reset(self, task_id: str, seed: int = 42):
        from server.environment import TaskID
        tid = TaskID(task_id)
        dirty, gold = self._gen(tid, seed)
        self._df_b64   = self._b64(dirty)
        self._gold_b64 = self._b64(gold)
        self._task_id  = tid
        self._steps    = 0
        self._done     = False
        self._had_crash= False
        return self._build(dirty, tid, gold, 0, False)
 
    def step_exec(self, code: str):
        return self._step("exec", code)
 
    def step_submit(self):
        return self._step("submit", None)
 
    def _step(self, action_type: str, code):
        df   = self._unb64(self._df_b64)
        gold = self._unb64(self._gold_b64)
        reward = None
        exec_result = error = ""
 
        if action_type == "exec":
            result = self._exec(code, df)
            exec_result = (result.stdout + result.stderr).strip()
            error = result.error
            if result.success:
                df = result.df
            else:
                self._had_crash = True
            self._steps += 1
            self._df_b64 = self._b64(df)
            if self._steps >= self._MAX:
                self._done = True
                reward = self._grade(df, gold, self._task_id, self._steps, self._had_crash)
        else:
            self._done = True
            reward = self._grade(df, gold, self._task_id, self._steps, self._had_crash)
 
        obs = self._build(df, self._task_id, gold, self._steps, self._done, exec_result, error)
        return obs, reward, self._done, {"step_count": self._steps}
    

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
 
 
class TestRunner:
    def __init__(self, client, verbose: bool = False):
        self.client  = client
        self.verbose = verbose
        self.results: list[tuple[str, bool, str]] = []
 
    def check(self, name: str, condition: bool, detail: str = ""):
        icon = PASS if condition else FAIL
        if self.verbose or not condition:
            print(f"    {icon} {name}" + (f": {detail}" if detail else ""))
        else:
            print(f"    {icon} {name}")
        self.results.append((name, condition, detail))
        return condition
    
    def test_reset_fields(self, task_id: str):
        print(f"\n  [reset] {task_id}")
        obs = self.client.reset(task_id=task_id, seed=42)
        self.check("df_preview non-empty",    bool(obs.df_preview))
        self.check("df_info non-empty",       bool(obs.df_info))
        self.check("task_spec non-empty",     bool(obs.task_spec))
        self.check("step_count == 0",         obs.step_count == 0)
        self.check("done == False",           not obs.done)
        self.check("partial_score in [0,1]",  0.0 <= obs.partial_score <= 1.0,
                   f"{obs.partial_score:.3f}")
        return obs
 
    def test_exec_step(self, task_id: str):
        print(f"\n  [step exec] {task_id}")
        obs0 = self.client.reset(task_id=task_id, seed=42)
        s0   = obs0.partial_score
 
        obs1, reward, done, info = self.client.step_exec(
            "df['price'] = df['price'].fillna(df['price'].median())" if "ecommerce" in task_id
            else "df.dropna(subset=['patient_id'], inplace=True)" if "patient" in task_id
            else "df.dropna(subset=['txn_id','account_id','amount'], inplace=True)"
        )
        self.check("done == False after exec",  not done)
        self.check("step_count == 1",           obs1.step_count == 1)
        self.check("partial_score in [0,1]",    0.0 <= obs1.partial_score <= 1.0,
                   f"{s0:.3f} → {obs1.partial_score:.3f}")
        self.check("reward is None mid-episode", reward is None)
 
    def test_submit_gives_reward(self, task_id: str, good_code: str):
        print(f"\n  [submit] {task_id}")
        self.client.reset(task_id=task_id, seed=42)
        self.client.step_exec(good_code)
        obs, reward, done, info = self.client.step_submit()
 
        self.check("done == True",              done)
        self.check("reward returned",           reward is not None)
        if reward:
            self.check("total in [0,1]",        0.0 <= reward.total <= 1.0, f"{reward.total:.4f}")
            self.check("col_quality in [0,1]",  0.0 <= reward.column_quality <= 1.0)
            self.check("schema in [0,1]",       0.0 <= reward.schema_compliance <= 1.0)
            self.check("rows in [0,1]",         0.0 <= reward.row_preservation <= 1.0)
            self.check("efficiency in [0,1]",   0.0 <= reward.efficiency <= 1.0)
            self.check("no_crash in [0,.05]",   0.0 <= reward.no_crash_bonus <= 0.05)
            self.check("total > 0",             reward.total > 0.0)
            self.check("breakdown is dict",     isinstance(reward.breakdown, dict))
        return reward
 
    def test_perfect_clean_easy(self):
        print(f"\n  [perfect clean] ecommerce_easy")
        self.client.reset(task_id="ecommerce_easy", seed=42)
        # Apply all required fixes
        steps = [
            "df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=False, errors='coerce')",
            "df['price'] = df['price'].fillna(df['price'].median())",
            "df['quantity'] = df['quantity'].clip(lower=0)",
            (
                "import re\n"
                "def parse_rev(v):\n"
                "    s = str(v).replace(',','.').replace('$','').replace('USD','').strip()\n"
                "    try: return float(s)\n"
                "    except: return float('nan')\n"
                "df['revenue'] = df['revenue'].apply(parse_rev)"
            ),
            "df['customer_id'] = df['customer_id'].str.strip()",
            (
                "df['status'] = df['status'].str.lower().str.strip()\n"
                "df.loc[df['status']=='complete','status'] = 'delivered'\n"
                "valid = {'pending','shipped','delivered','cancelled'}\n"
                "df = df[df['status'].isin(valid)]"
            ),
        ]
        for code in steps:
            self.client.step_exec(code)
        _, reward, _, _ = self.client.step_submit()
        if reward:
            self.check("score > 0.80 after full clean", reward.total > 0.80,
                       f"total={reward.total:.4f}")
        return reward
 
    def test_idempotent_seed(self, task_id: str):
        print(f"\n  [idempotent seed] {task_id}")
        obs_a = self.client.reset(task_id=task_id, seed=77)
        obs_b = self.client.reset(task_id=task_id, seed=77)
        obs_c = self.client.reset(task_id=task_id, seed=88)
        self.check("same seed → same preview",      obs_a.df_preview == obs_b.df_preview)
        self.check("diff seed → diff preview",      obs_a.df_preview != obs_c.df_preview)
 
    def test_step_limit(self, task_id: str):
        print(f"\n  [step limit] {task_id}")
        self.client.reset(task_id=task_id, seed=42)
        obs = reward = done = None
        for i in range(22):  # exceed MAX_STEPS=20
            obs, reward, done, _ = self.client.step_exec("pass")
            if done:
                break
        self.check("episode ends by step 20", done and obs.step_count <= 20,
                   f"ended at step {obs.step_count if obs else '?'}")
        self.check("reward returned at limit", reward is not None)

CLEANING_CODE = {
    "ecommerce_easy": (
        "df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')\n"
        "df['price'] = df['price'].fillna(df['price'].median())\n"
        "df['quantity'] = df['quantity'].clip(lower=0)"
    ),
    "patient_records_medium": (
        "df = df.drop_duplicates(subset=['patient_id'], keep='first')\n"
        "df['email'] = df['email'].str.lower()"
    ),
    "financial_audit_hard": (
        "df.dropna(subset=['txn_id','account_id','amount','transaction_date'], inplace=True)\n"
        "df['violation'] = ''\n"
        "df['duplicate'] = False\n"
        "FX = {'USD':1.0,'EUR':1.085,'GBP':1.265,'JPY':0.0067,'CAD':0.735}\n"
        "mask = df['currency'].isin(FX) & (df['currency'] != 'USD')\n"
        "df.loc[mask,'usd_amount'] = (df.loc[mask,'amount'] * df.loc[mask,'currency'].map(FX)).round(2)"
    ),
}
 
 
def main():
    parser = argparse.ArgumentParser(description="Integration tests for data-cleaning-env")
    parser.add_argument("--task", choices=["easy","medium","hard","all"], default="all")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
 
    print("\n  data-cleaning-env  —  integration tests")
    print("  " + "="*50)
 
    # Try HTTP server, fall back to direct
    use_http = _try_start_server()
    if use_http:
        try:
            from client import DataCleaningEnvClient
            client = DataCleaningEnvClient(base_url=SERVER_URL)
            print(f"  Mode: HTTP  ({SERVER_URL})")
        except ImportError:
            use_http = False
 
    if not use_http:
        client = DirectClient()
        print("  Mode: direct (in-process, no HTTP)")
 
    runner = TestRunner(client, verbose=args.verbose)
 
    tasks = {
        "easy":   "ecommerce_easy",
        "medium": "patient_records_medium",
        "hard":   "financial_audit_hard",
    }
 
    run_tasks = (
        list(tasks.values()) if args.task == "all"
        else [tasks[args.task]]
    )
 
    for task_id in run_tasks:
        runner.test_reset_fields(task_id)
        runner.test_exec_step(task_id)
        runner.test_submit_gives_reward(task_id, CLEANING_CODE[task_id])
        runner.test_idempotent_seed(task_id)
        runner.test_step_limit(task_id)
 
    # Full clean test on easy task only (deterministic high score)
    if "ecommerce_easy" in run_tasks:
        runner.test_perfect_clean_easy()
 
    # Summary
    total  = len(runner.results)
    passed = sum(1 for _, ok, _ in runner.results if ok)
    failed = total - passed
 
    print(f"\n  {'='*50}")
    print(f"  {passed}/{total} assertions passed", end="")
    if failed == 0:
        print("  \033[32m— all good!\033[0m\n")
        sys.exit(0)
    else:
        print(f"  \033[31m— {failed} failed\033[0m")
        print("\n  Failed checks:")
        for name, ok, detail in runner.results:
            if not ok:
                print(f"    \033[31m✗\033[0m {name}: {detail}")
        print()
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()