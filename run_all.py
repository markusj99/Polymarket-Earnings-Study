#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py

Project orchestrator for the Polymarket-Earnings-Study pipeline.

Where it lives
--------------
Place this file in:
  Polymarket-Earnings-Study/run_all.py

Expected script location
------------------------
All other Python scripts are expected in:
  Polymarket-Earnings-Study/python/

Behavior
--------
- Runs the pipeline scripts one-by-one (00 -> 07) via subprocess.
- Uses sys.executable to ensure the same interpreter/venv is used.
- Ensures scripts run with cwd set to the project root so relative paths work.

App key handling (important)
----------------------------
Different scripts in this repo expect the app key in different ways:

1) Steps 00 and 01:
   - Do NOT accept a CLI argument '--app-key' (argparse "unrecognized arguments")
   - They should read the key from environment variables instead.

2) Steps 02 and 03 (per earlier error message):
   - Expect '--app-key' with NO value as a *flag* meaning:
     "read EIKON_APP_KEY from the environment".

3) Step 04 (04_fetch_stock_prices.py):
   - Requires '--app-key <value>' (argparse: expected one argument).

4) Step 05 (05_heckman_selection_fetch_universe.py):
   - In your logs it failed with rc=2 when run_all passed flag-only '--app-key'.
   - In practice, this step typically needs the *value* form as well.
   - Therefore we pass '--app-key <value>' for step 05.

Therefore, when --app-key is provided to run_all.py:
- It is always injected into the environment as EIKON_APP_KEY and APP_KEY.
- For steps that require the *flag-only* form, run_all passes '--app-key' with NO value.
- For steps that require the *value* form, run_all passes '--app-key <value>'.
- For steps that do not need it, nothing is passed.

Skip logic
----------
- If --app-key is NOT provided:
    * ONLY run steps that do not require the key (per ScriptSpec.needs_app_key)
    * SKIP steps marked needs_app_key=True
- Prints a summary at the end (success/failed/skipped/missing + timings).

This file is importable: call main([...]) from another script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Pipeline specification
# ----------------------------

@dataclass(frozen=True)
class ScriptSpec:
    """
    Defines a pipeline step.

    Attributes
    ----------
    code : str
        Human-friendly step code (e.g., "00", "01").
    filename : str
        Script filename under Polymarket-Earnings-Study/python/
    needs_app_key : bool
        If True, this step is skipped unless --app-key is provided to run_all.py.
    app_key_cli_mode : str
        How to pass key info on the CLI to this script (if at all):
          - "none": do not pass any --app-key argument (script reads env if needed)
          - "flag": pass '--app-key' with NO value (script uses env var for actual key)
          - "value": pass '--app-key <value>' (only if a script truly supports it)
    """
    code: str
    filename: str
    needs_app_key: bool = False
    app_key_cli_mode: str = "none"  # "none" | "flag" | "value"


PIPELINE: List[ScriptSpec] = [
    # 00/01: DO NOT accept '--app-key' on CLI; use env injection only.
    ScriptSpec("00", "00_fetch_closed_markets.py", needs_app_key=True, app_key_cli_mode="none"),
    ScriptSpec("01", "01_fetch_poly_prices.py", needs_app_key=True, app_key_cli_mode="none"),

    # 02-03: require '--app-key' flag-only to signal "read from env".
    ScriptSpec("02", "02_check_consistency.py", needs_app_key=False, app_key_cli_mode="flag"),
    ScriptSpec("03", "03_fetch_corp_info.py", needs_app_key=False, app_key_cli_mode="flag"),

    # 04-05: require '--app-key <value>' (04 explicitly does; 05 failed with flag-only).
    ScriptSpec("04", "04_fetch_stock_prices.py", needs_app_key=False, app_key_cli_mode="value"),
    ScriptSpec("05", "05_heckman_selection_fetch_universe.py", needs_app_key=False, app_key_cli_mode="value"),

    # 06/07: do not pass app key args by default (they should rely on produced files).
    ScriptSpec("06", "06_brier_scores.py", needs_app_key=False, app_key_cli_mode="none"),
    ScriptSpec("07", "07_create_dataset.py", needs_app_key=False, app_key_cli_mode="none"),
]


# ----------------------------
# Utilities
# ----------------------------

def _project_root() -> Path:
    """Return the directory containing this run_all.py (Polymarket-Earnings-Study/)."""
    return Path(__file__).resolve().parent


def _scripts_dir() -> Path:
    """Return Polymarket-Earnings-Study/python/."""
    return _project_root() / "python"


def _now_utc_compact() -> str:
    """Simple timestamp for logs (no external deps)."""
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    return f"{m}m{r:04.1f}s"


def _run_subprocess(
    script_path: Path,
    args: List[str],
    env: Dict[str, str],
) -> Tuple[int, float]:
    """
    Run a script as:
      <sys.executable> <script_path> <args...>

    Returns (returncode, elapsed_seconds).
    """
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script_path), *args],
        env=env,
        cwd=str(_project_root()),  # ensure relative paths resolve from project root
    )
    dt = time.time() - t0
    return int(proc.returncode), float(dt)


def _build_script_args(spec: ScriptSpec, app_key: Optional[str]) -> List[str]:
    """
    Build CLI args passed to each script based on its configured app_key_cli_mode.
    """
    if not app_key:
        return []

    if spec.app_key_cli_mode == "none":
        return []
    if spec.app_key_cli_mode == "flag":
        return ["--app-key"]
    if spec.app_key_cli_mode == "value":
        return ["--app-key", app_key]

    # Defensive fallback: do not pass anything if misconfigured
    return []


# ----------------------------
# Main
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all Polymarket-Earnings-Study pipeline scripts sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--app-key",
        default=None,
        help="App key used by scripts that require it. Injected into env as EIKON_APP_KEY/APP_KEY.",
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately if any step fails (non-zero exit code).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, but do not execute anything.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    root = _project_root()
    py_dir = _scripts_dir()

    print(f"[run_all] {_now_utc_compact()}  Starting pipeline")
    print(f"[run_all] Project root: {root}")
    print(f"[run_all] Scripts dir : {py_dir}")

    if args.app_key is None:
        print("[run_all] NOTE: --app-key not provided. Will SKIP steps that require it.")
    else:
        print("[run_all] NOTE: --app-key provided. Will inject it into env and pass appropriate CLI args per step.")

    # Base environment inherited from current process
    env = dict(os.environ)

    # Inject key into env for scripts that read env vars (harmless for others)
    if args.app_key:
        env["EIKON_APP_KEY"] = args.app_key
        env["APP_KEY"] = args.app_key

    results = []
    pipeline_t0 = time.time()
    n_total = len(PIPELINE)

    for i, spec in enumerate(PIPELINE, start=1):
        script_path = py_dir / spec.filename

        # Missing file => skip (but record it)
        if not script_path.exists():
            print(f"[run_all] ({i}/{n_total}) MISSING {spec.code} {spec.filename} (expected at {script_path})")
            results.append({"code": spec.code, "name": spec.filename, "status": "MISSING", "rc": None, "seconds": 0.0})
            continue

        # Gate: steps that require --app-key
        if spec.needs_app_key and not args.app_key:
            print(f"[run_all] ({i}/{n_total}) SKIP    {spec.code} {spec.filename} (needs --app-key)")
            results.append(
                {"code": spec.code, "name": spec.filename, "status": "SKIPPED_NO_APP_KEY", "rc": None, "seconds": 0.0}
            )
            continue

        # Build args for subprocess (per-script mode)
        sub_args: List[str] = _build_script_args(spec, args.app_key)

        print(f"[run_all] ({i}/{n_total}) RUN     {spec.code} {spec.filename}")

        if args.dry_run:
            results.append({"code": spec.code, "name": spec.filename, "status": "DRY_RUN", "rc": 0, "seconds": 0.0})
            continue

        rc, dt = _run_subprocess(script_path, sub_args, env)
        if rc == 0:
            print(f"[run_all] ({i}/{n_total}) OK      {spec.code} {spec.filename}  ({_fmt_secs(dt)})")
            results.append({"code": spec.code, "name": spec.filename, "status": "OK", "rc": rc, "seconds": dt})
        else:
            print(f"[run_all] ({i}/{n_total}) FAIL    {spec.code} {spec.filename}  (rc={rc}, {_fmt_secs(dt)})")
            results.append({"code": spec.code, "name": spec.filename, "status": "FAIL", "rc": rc, "seconds": dt})
            if args.stop_on_failure:
                print("[run_all] stop-on-failure enabled -> exiting now.")
                break

    pipeline_dt = time.time() - pipeline_t0

    # ----------------------------
    # Summary
    # ----------------------------
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    ok_n = counts.get("OK", 0)
    fail_n = counts.get("FAIL", 0)
    skipped_n = counts.get("SKIPPED_NO_APP_KEY", 0)
    missing_n = counts.get("MISSING", 0)
    dry_n = counts.get("DRY_RUN", 0)

    print("\n[run_all] ===================== SUMMARY =====================")
    print(f"[run_all] Total steps listed : {n_total}")
    print(f"[run_all] Executed OK        : {ok_n}")
    print(f"[run_all] Executed FAIL      : {fail_n}")
    print(f"[run_all] Skipped (no key)   : {skipped_n}")
    print(f"[run_all] Missing files      : {missing_n}")
    print(f"[run_all] Dry-run steps      : {dry_n}")
    print(f"[run_all] Total elapsed      : {_fmt_secs(pipeline_dt)}")
    print("[run_all] ----------------------------------------------------")

    for r in results:
        code = r["code"]
        name = r["name"]
        status = r["status"]
        sec = r["seconds"]
        rc = r["rc"]
        rc_s = "-" if rc is None else str(rc)
        time_s = "-" if sec <= 0 else _fmt_secs(sec)
        print(f"[run_all] {code}  {status:<18} rc={rc_s:<3} time={time_s:<8}  {name}")

    print("[run_all] ====================================================\n")

    # Exit code: 0 if no failures, else 1
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
