#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py

Project orchestrator for the Corporate_Earnings pipeline.

Where it lives
--------------
Place this file in:
  Corporate_Earnings/run_all.py

Expected script location
------------------------
All other Python scripts are expected in:
  Corporate_Earnings/python/

Behavior (per your requirements)
--------------------------------
- Runs the pipeline scripts one-by-one (00 -> 07) via subprocess.
- Prints a small amount of orchestration logging (the scripts themselves print more).
- If --app-key is NOT provided:
    * DO NOT run scripts: 02, 03, 04, 06 (Eikon-required)
- Prints a summary at the end (success/failed/skipped/missing + timings).

Notes
-----
- Uses sys.executable to ensure the same virtualenv/interpreter is used.
- When --app-key is provided, it is:
    * passed as --app-key to the Eikon scripts, and
    * injected into env as EIKON_APP_KEY and APP_KEY for compatibility.
- This file is importable: call main([...]) from another script.
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
        Script filename under Corporate_Earnings/python/
    needs_eikon : bool
        If True, this step is skipped unless --app-key is provided.
    """
    code: str
    filename: str
    needs_eikon: bool = False


PIPELINE: List[ScriptSpec] = [
    ScriptSpec("00", "00_fetch_closed_markets.py", needs_eikon=False),
    ScriptSpec("01", "01_fetch_poly_prices.py", needs_eikon=False),
    ScriptSpec("02", "02_check_consistency.py", needs_eikon=True),
    ScriptSpec("03", "03_fetch_corp_info.py", needs_eikon=True),
    ScriptSpec("04", "04_fetch_stock_prices.py", needs_eikon=True),
    ScriptSpec("05", "05_descriptive_statistics.py", needs_eikon=False),
    ScriptSpec("06", "06_heckman_selection_fetch_universe.py", needs_eikon=True),
    ScriptSpec("07", "07_brier_scores.py", needs_eikon=False),
]


# ----------------------------
# Utilities
# ----------------------------

def _project_root() -> Path:
    """Return the directory containing this run_all.py (Corporate_Earnings/)."""
    return Path(__file__).resolve().parent


def _scripts_dir() -> Path:
    """Return Corporate_Earnings/python/."""
    return _project_root() / "python"


def _now_utc_compact() -> str:
    """Simple timestamp for logs (no external deps)."""
    # Using time.gmtime keeps it dependency-free and stable across machines.
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
        cwd=str(_project_root()),   # ensure relative paths in scripts resolve from project root
    )
    dt = time.time() - t0
    return int(proc.returncode), float(dt)


# ----------------------------
# Main
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all Corporate_Earnings pipeline scripts sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--app-key",
        default=None,
        help="Refinitiv Eikon/Workspace App Key. If omitted, steps 02/03/04/06 will be skipped.",
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

    print(f"[run_all] { _now_utc_compact() }  Starting pipeline")
    print(f"[run_all] Project root: {root}")
    print(f"[run_all] Scripts dir : {py_dir}")

    if args.app_key is None:
        print("[run_all] NOTE: --app-key not provided. Will SKIP Eikon-required steps: 02, 03, 04, 06.")
        print("[run_all] NOTE: Downstream steps (e.g., 05/07) may fail if their inputs were never generated.")

    # Base environment inherited from current process
    env = dict(os.environ)

    # Inject key into env for scripts that read env vars
    if args.app_key:
        env["EIKON_APP_KEY"] = args.app_key
        env["APP_KEY"] = args.app_key

    results = []  # list of dicts for summary
    pipeline_t0 = time.time()

    n_total = len(PIPELINE)
    for i, spec in enumerate(PIPELINE, start=1):
        script_path = py_dir / spec.filename

        # Missing file => skip (but record it)
        if not script_path.exists():
            msg = f"[run_all] ({i}/{n_total}) MISSING {spec.code} {spec.filename} (expected at {script_path})"
            print(msg)
            results.append(
                {
                    "code": spec.code,
                    "name": spec.filename,
                    "status": "MISSING",
                    "rc": None,
                    "seconds": 0.0,
                }
            )
            continue

        # Eikon gate
        if spec.needs_eikon and not args.app_key:
            print(f"[run_all] ({i}/{n_total}) SKIP    {spec.code} {spec.filename} (needs --app-key)")
            results.append(
                {
                    "code": spec.code,
                    "name": spec.filename,
                    "status": "SKIPPED_NO_APP_KEY",
                    "rc": None,
                    "seconds": 0.0,
                }
            )
            continue

        # Build args for the subprocess call
        sub_args: List[str] = []
        if spec.needs_eikon and args.app_key:
            # Standardize on passing the key explicitly, even though env is also set.
            sub_args += ["--app-key", args.app_key]

        print(f"[run_all] ({i}/{n_total}) RUN     {spec.code} {spec.filename}")

        if args.dry_run:
            results.append(
                {
                    "code": spec.code,
                    "name": spec.filename,
                    "status": "DRY_RUN",
                    "rc": 0,
                    "seconds": 0.0,
                }
            )
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

    # Minimal per-step line items
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

    # Exit code: 0 if no failures, else 1 (even if some were skipped/missing)
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
