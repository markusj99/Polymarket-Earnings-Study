#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py

Portable orchestrator for the Polymarket-Earnings-Study project.

Run order
---------
1) Python scripts, in data-pipeline order:
   00, 01, 01b, 02, 03, 04, 05, 06, 07

2) R scripts, in statistics order:
   00, 01, 02, 03, 04

API-key behavior
----------------
The script first looks for an Eikon/Refinitiv app key in this order:
  1) --app-key
  2) EIKON_APP_KEY or APP_KEY environment variable
  3) an interactive CLI prompt

If no key is provided, steps that require the Eikon/Refinitiv API are skipped.
Steps that do not require an API key still run.

Portability
-----------
- Python steps use sys.executable, so they run with the same Python interpreter
  or virtual environment used to launch run_all.py.
- R steps use Rscript from PATH, or --rscript if provided.
- Scripts are resolved relative to the project root, not hard-coded absolute paths.
- Common project layouts are supported, including scripts under python/, R/,
  R/scripts/, and R/statistics/event_study/.

Usage
-----
  python run_all.py
  python run_all.py --app-key YOUR_KEY
  python run_all.py --dry-run
  python run_all.py --keep-going
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Pipeline specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScriptSpec:
    code: str
    language: str  # "python" or "r"
    candidates: Tuple[str, ...]
    needs_api_key: bool = False
    api_key_mode: str = "none"  # "none", "env_flag", "env_only", "value"
    r_function: Optional[str] = None
    notes: str = ""
    resolved_path: Optional[Path] = field(default=None, compare=False)


PYTHON_PIPELINE: List[ScriptSpec] = [
    ScriptSpec(
        code="PY-00",
        language="python",
        candidates=("python/00_fetch_closed_markets.py", "00_fetch_closed_markets.py"),
        needs_api_key=False,
        notes="Fetch closed Polymarket markets from public Polymarket/Gamma endpoints.",
    ),
    ScriptSpec(
        code="PY-01",
        language="python",
        candidates=("python/01_check_consistency.py", "01_check_consistency.py"),
        needs_api_key=True,
        api_key_mode="env_flag",
        notes="Uses Eikon EPS actual/estimate data.",
    ),
    ScriptSpec(
        code="PY-01b",
        language="python",
        candidates=("python/01b_retry_unmatched_earnings.py", "01b_retry_unmatched_earnings.py"),
        needs_api_key=True,
        api_key_mode="env_flag",
        notes="Retries unmatched Eikon earnings events.",
    ),
    ScriptSpec(
        code="PY-02",
        language="python",
        candidates=("python/02_fetch_corp_info.py", "02_fetch_corp_info.py"),
        needs_api_key=True,
        api_key_mode="env_flag",
        notes="Uses Eikon corporate information and event-time fields.",
    ),
    ScriptSpec(
        code="PY-03",
        language="python",
        candidates=("python/03_fetch_stock_prices.py", "03_fetch_stock_prices.py"),
        needs_api_key=True,
        api_key_mode="env_only",
        notes="Uses Eikon daily stock-price data; reads key from env or --app-key value.",
    ),
    ScriptSpec(
        code="PY-04",
        language="python",
        candidates=("python/04_fetch_poly_prices.py", "04_fetch_poly_prices.py"),
        needs_api_key=False,
        notes="Fetches Polymarket historical prices from public Polymarket/CLOB endpoints.",
    ),
    ScriptSpec(
        code="PY-05",
        language="python",
        candidates=("python/05_brier_scores.py", "05_brier_scores.py"),
        needs_api_key=False,
        notes="Computes Brier scores from local pipeline outputs.",
    ),
    ScriptSpec(
        code="PY-06",
        language="python",
        candidates=("python/06_create_dataset.py", "06_create_dataset.py"),
        needs_api_key=False,
        notes="Builds complete long/wide datasets from local outputs.",
    ),
    ScriptSpec(
        code="PY-07",
        language="python",
        candidates=("python/07_heckman_selection_fetch_universe.py", "07_heckman_selection_fetch_universe.py"),
        needs_api_key=True,
        api_key_mode="env_flag",
        notes="Uses Eikon to fetch the broader Heckman selection universe.",
    ),
]

R_PIPELINE: List[ScriptSpec] = [
    ScriptSpec(
        code="R-00",
        language="r",
        candidates=("R/00_descriptive_statistics.R", "00_descriptive_statistics.R"),
        needs_api_key=False,
        notes="Descriptive statistics from local datasets.",
    ),
    ScriptSpec(
        code="R-01",
        language="r",
        candidates=("R/01_BSS.R", "R/scripts/01_BSS.R", "01_BSS.R"),
        needs_api_key=False,
        notes="Brier skill score analysis from local datasets.",
    ),
    ScriptSpec(
        code="R-02",
        language="r",
        candidates=("R/02_factor_analysis.R", "R/factor_analysis.R", "02_factor_analysis.R"),
        needs_api_key=False,
        notes="Factor-analysis regressions from local datasets.",
    ),
    ScriptSpec(
        code="R-03",
        language="r",
        candidates=("R/03_heckman_selection_robustness.R", "03_heckman_selection_robustness.R"),
        needs_api_key=False,
        r_function="run_heckman_selection_robustness",
        notes="Heckman robustness analysis from local universe files.",
    ),
    ScriptSpec(
        code="R-04",
        language="r",
        candidates=(
            "R/04_event_study.R",
            "R/statistics/event_study/run_polymarket_price_event_study.R",
            "04_event_study.R",
        ),
        needs_api_key=False,
        r_function="run_polymarket_event_study",
        notes="Event study from local datasets. Called via function to avoid hard-coded local paths.",
    ),
]

PIPELINE: List[ScriptSpec] = [*PYTHON_PIPELINE, *R_PIPELINE]


# ---------------------------------------------------------------------------
# Path and command helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parent


def normalize_copy_suffix(name: str) -> str:
    """Remove upload/download copy suffixes such as '(2)' before the extension."""
    return re.sub(r"\(\d+\)(?=\.[^.]+$)", "", name)


def possible_search_dirs(root: Path) -> List[Path]:
    dirs = [
        root,
        root / "python",
        root / "R",
        root / "R" / "scripts",
        root / "R" / "statistics" / "event_study",
    ]
    return [d for d in dirs if d.exists()]


def resolve_script(spec: ScriptSpec, root: Path) -> Optional[Path]:
    """Resolve a script path using exact candidates first, then normalized-name fallback."""
    for rel in spec.candidates:
        p = root / rel
        if p.exists():
            return p.resolve()

    wanted_names = {Path(rel).name for rel in spec.candidates}
    normalized_wanted = {normalize_copy_suffix(name) for name in wanted_names}

    suffix = ".py" if spec.language == "python" else ".R"
    for d in possible_search_dirs(root):
        for p in d.glob(f"*{suffix}"):
            if normalize_copy_suffix(p.name) in normalized_wanted:
                return p.resolve()

    return None


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds - 60 * minutes
    return f"{minutes}m{rest:04.1f}s"


def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def r_string(value: str) -> str:
    """Return a simple R string literal for paths/values."""
    return '"' + value.replace("\\", "/").replace('"', '\\"') + '"'


def find_rscript(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return shutil.which("Rscript") or shutil.which("Rscript.exe")


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

def prompt_for_api_key() -> Optional[str]:
    if not sys.stdin.isatty():
        return None

    prompt = (
        "Enter Eikon/Refinitiv app key, or press Enter to skip "
        "API-key-required steps: "
    )
    try:
        key = getpass.getpass(prompt)
    except Exception:
        key = input(prompt)

    key = key.strip()
    return key or None


def resolve_api_key(args: argparse.Namespace) -> Optional[str]:
    if args.app_key is not None:
        key = str(args.app_key).strip()
        return key or None

    env_key = (os.getenv("EIKON_APP_KEY") or os.getenv("APP_KEY") or "").strip()
    if env_key:
        return env_key

    if args.no_key_prompt:
        return None

    return prompt_for_api_key()


def add_api_key_to_env(env: Dict[str, str], api_key: Optional[str]) -> Dict[str, str]:
    out = dict(env)
    if api_key:
        out["EIKON_APP_KEY"] = api_key
        out["APP_KEY"] = api_key
    return out


def build_python_args(spec: ScriptSpec, api_key: Optional[str]) -> List[str]:
    if not spec.needs_api_key or not api_key:
        return []

    if spec.api_key_mode == "env_flag":
        return ["--app-key"]
    if spec.api_key_mode == "env_only":
        return []
    if spec.api_key_mode == "value":
        return ["--app-key", api_key]
    return []


# ---------------------------------------------------------------------------
# Process runners
# ---------------------------------------------------------------------------

def run_python_script(spec: ScriptSpec, script_path: Path, root: Path, env: Dict[str, str], api_key: Optional[str]) -> Tuple[int, float]:
    cmd = [sys.executable, str(script_path), *build_python_args(spec, api_key)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    return int(proc.returncode), time.time() - t0


def build_r_command(spec: ScriptSpec, script_path: Path, root: Path, rscript: str) -> List[str]:
    if spec.r_function:
        # Source the R file and call its exported function with the detected root.
        # This avoids hard-coded paths in script-level entry points.
        expr = (
            f"ROOT <- normalizePath({r_string(str(root))}, winslash='/', mustWork=TRUE); "
            "setwd(ROOT); "
            "options(polymarket.autorun=FALSE); "
            f"source({r_string(str(script_path))}); "
            f"{spec.r_function}(root=ROOT)"
        )
        return [rscript, "--vanilla", "-e", expr]

    return [rscript, "--vanilla", str(script_path)]


def run_r_script(spec: ScriptSpec, script_path: Path, root: Path, env: Dict[str, str], rscript: str) -> Tuple[int, float]:
    cmd = build_r_command(spec, script_path, root, rscript)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    return int(proc.returncode), time.time() - t0


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Polymarket-Earnings-Study pipeline in the correct order.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--app-key",
        default=None,
        help="Eikon/Refinitiv app key. If omitted, EIKON_APP_KEY/APP_KEY is checked, then an interactive prompt is shown.",
    )
    parser.add_argument(
        "--no-key-prompt",
        action="store_true",
        help="Do not prompt for an API key. Useful for CI/non-interactive runs.",
    )
    parser.add_argument(
        "--rscript",
        default=None,
        help="Path to Rscript/Rscript.exe. If omitted, Rscript is resolved from PATH.",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Run only Python pipeline steps.",
    )
    parser.add_argument(
        "--r-only",
        action="store_true",
        help="Run only R pipeline steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing scripts.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed or missing step. By default, the run stops on the first actual failure/missing script.",
    )
    return parser


def select_pipeline(args: argparse.Namespace) -> List[ScriptSpec]:
    if args.python_only and args.r_only:
        raise SystemExit("Use only one of --python-only or --r-only.")
    if args.python_only:
        return PYTHON_PIPELINE
    if args.r_only:
        return R_PIPELINE
    return PIPELINE


def print_plan(specs: Sequence[ScriptSpec], root: Path, api_key: Optional[str], rscript: Optional[str]) -> List[ScriptSpec]:
    resolved: List[ScriptSpec] = []
    print(f"[run_all] {now_utc()}  Project root: {root}")
    print(f"[run_all] API key: {'provided' if api_key else 'not provided; key-required steps will be skipped'}")
    print(f"[run_all] Rscript: {rscript or 'not found'}")
    print("[run_all] Planned run order:")

    for spec in specs:
        p = resolve_script(spec, root)
        object.__setattr__(spec, "resolved_path", p)
        key_note = "requires API key" if spec.needs_api_key else "no API key"
        path_note = str(p.relative_to(root)) if p and p.is_relative_to(root) else (str(p) if p else "MISSING")
        print(f"[run_all]   {spec.code:<6} {spec.language.upper():<6} {key_note:<17} {path_note}")
        resolved.append(spec)
    return resolved


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = project_root()
    api_key = resolve_api_key(args)
    env = add_api_key_to_env(os.environ, api_key)
    specs = select_pipeline(args)
    rscript = find_rscript(args.rscript)

    specs = print_plan(specs, root, api_key, rscript)

    results: List[Dict[str, object]] = []
    start = time.time()
    total = len(specs)

    for idx, spec in enumerate(specs, start=1):
        script_path = spec.resolved_path

        if script_path is None:
            print(f"[run_all] ({idx}/{total}) MISSING {spec.code}  Expected one of: {', '.join(spec.candidates)}")
            results.append({"code": spec.code, "status": "MISSING", "rc": None, "seconds": 0.0, "path": " | ".join(spec.candidates)})
            if not args.keep_going:
                break
            continue

        if spec.needs_api_key and not api_key:
            print(f"[run_all] ({idx}/{total}) SKIP    {spec.code}  {script_path.name} (needs API key)")
            results.append({"code": spec.code, "status": "SKIPPED_NO_API_KEY", "rc": None, "seconds": 0.0, "path": str(script_path)})
            continue

        print(f"[run_all] ({idx}/{total}) RUN     {spec.code}  {script_path.name}")

        if args.dry_run:
            results.append({"code": spec.code, "status": "DRY_RUN", "rc": 0, "seconds": 0.0, "path": str(script_path)})
            continue

        if spec.language == "r" and not rscript:
            print(f"[run_all] ({idx}/{total}) MISSING {spec.code}  Rscript not found on PATH; install R or pass --rscript")
            results.append({"code": spec.code, "status": "MISSING_RSCRIPT", "rc": None, "seconds": 0.0, "path": str(script_path)})
            if not args.keep_going:
                break
            continue

        if spec.language == "python":
            rc, seconds = run_python_script(spec, script_path, root, env, api_key)
        else:
            assert rscript is not None
            rc, seconds = run_r_script(spec, script_path, root, env, rscript)

        if rc == 0:
            print(f"[run_all] ({idx}/{total}) OK      {spec.code}  ({fmt_seconds(seconds)})")
            results.append({"code": spec.code, "status": "OK", "rc": rc, "seconds": seconds, "path": str(script_path)})
        else:
            print(f"[run_all] ({idx}/{total}) FAIL    {spec.code}  rc={rc} ({fmt_seconds(seconds)})")
            results.append({"code": spec.code, "status": "FAIL", "rc": rc, "seconds": seconds, "path": str(script_path)})
            if not args.keep_going:
                break

    elapsed = time.time() - start
    counts: Dict[str, int] = {}
    for row in results:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1

    print("\n[run_all] ===================== SUMMARY =====================")
    print(f"[run_all] Steps listed       : {total}")
    print(f"[run_all] Steps attempted    : {len(results)}")
    for status in sorted(counts):
        print(f"[run_all] {status:<18}: {counts[status]}")
    print(f"[run_all] Total elapsed      : {fmt_seconds(elapsed)}")
    print("[run_all] ----------------------------------------------------")
    for row in results:
        rc = "-" if row["rc"] is None else str(row["rc"])
        seconds = float(row["seconds"])
        time_s = "-" if seconds <= 0 else fmt_seconds(seconds)
        print(f"[run_all] {row['code']:<6} {str(row['status']):<20} rc={rc:<3} time={time_s:<8} {row['path']}")
    print("[run_all] ====================================================\n")

    hard_failures = sum(counts.get(s, 0) for s in ("FAIL", "MISSING", "MISSING_RSCRIPT"))
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
