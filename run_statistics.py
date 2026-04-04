#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all project R statistics scripts in the correct order.

Behavior
--------
- Assumes this Python file is placed in the project root.
- Runs each R script from the project root so relative paths in the R code work.
- Streams all R output directly to the console.
- Stops immediately if any script fails.
- Works on Windows, macOS, and Linux.
- If Rscript cannot be found automatically, asks the user to enter the path
  in the console.

Environment override
--------------------
Set RSCRIPT_BIN to the full path of the Rscript executable if needed.
Examples:

Windows PowerShell:
    $env:RSCRIPT_BIN = 'C:\\Program Files\\R\\R-4.5.0\\bin\\Rscript.exe'

macOS/Linux:
    export RSCRIPT_BIN='/Library/Frameworks/R.framework/Resources/bin/Rscript'
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _existing_candidates(paths: List[Path]) -> List[Path]:
    """Return only paths that exist, preserving order."""
    return [path for path in paths if path.is_file()]


def _normalize_user_path(text: str) -> str:
    """Trim whitespace and surrounding quotes from console input."""
    return text.strip().strip('"').strip("'")


def _resolve_user_supplied_rscript(user_value: str) -> Optional[str]:
    """Resolve a user-supplied Rscript path or command.

    Accepts either:
    - the full path to Rscript / Rscript.exe
    - a directory containing Rscript
    - a command name that can be found on PATH
    """
    cleaned = _normalize_user_path(user_value)
    if not cleaned:
        return None

    which_value = shutil.which(cleaned)
    if which_value:
        return which_value

    candidate = Path(cleaned).expanduser()

    if candidate.is_dir():
        names = ["Rscript.exe", "Rscript"] if platform.system() == "Windows" else ["Rscript", "Rscript.exe"]
        for name in names:
            possible = candidate / name
            if possible.is_file():
                return str(possible.resolve())
        return None

    if candidate.is_file():
        return str(candidate.resolve())

    return None


def prompt_for_rscript() -> str:
    """Ask the user to enter the Rscript path in the console."""
    if not sys.stdin or not sys.stdin.isatty():
        raise FileNotFoundError(
            "Could not find 'Rscript' automatically, and no interactive console "
            "is available to ask for the path. Install R, add Rscript to PATH, "
            "or set RSCRIPT_BIN to the full path of the executable."
        )

    system_name = platform.system()
    if system_name == "Windows":
        example = r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe"
    elif system_name == "Darwin":
        example = "/Library/Frameworks/R.framework/Resources/bin/Rscript"
    else:
        example = "/usr/bin/Rscript"

    print("\nRscript was not found automatically.")
    print("Enter the full path to Rscript and press Enter.")
    print("You can also enter a directory that contains Rscript.")
    print("Type 'q' to quit.")
    print(f"Example: {example}\n")
    sys.stdout.flush()

    while True:
        try:
            entered = input("Path to Rscript: ")
        except EOFError as exc:
            raise FileNotFoundError(
                "Could not read a path for 'Rscript' from the console."
            ) from exc

        cleaned = _normalize_user_path(entered)
        if cleaned.lower() in {"q", "quit", "exit"}:
            raise FileNotFoundError("User cancelled Rscript path entry.")

        resolved = _resolve_user_supplied_rscript(cleaned)
        if resolved:
            return resolved

        print("That path could not be used.")
        print("Please enter the full path to Rscript, a directory containing it, or 'q' to quit.\n")
        sys.stdout.flush()


def find_rscript() -> str:
    """Return a usable Rscript executable path.

    Resolution order:
    1. RSCRIPT_BIN environment variable
    2. PATH
    3. Common OS-specific installation directories
    4. Interactive console prompt
    """
    env_value = os.environ.get("RSCRIPT_BIN", "").strip().strip('"')
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_file():
            return str(env_path)
        print(
            "WARNING: RSCRIPT_BIN is set, but the file does not exist: "
            f"{env_value}",
            file=sys.stderr,
        )

    path_value = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if path_value:
        return path_value

    system_name = platform.system()
    candidate_paths: List[Path] = []

    if system_name == "Windows":
        candidate_dirs: List[Path] = []
        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if base:
                candidate_dirs.append(Path(base))

        for base in candidate_dirs:
            r_root = base / "R"
            if r_root.is_dir():
                for version_dir in sorted(r_root.glob("R-*"), reverse=True):
                    candidate_paths.append(version_dir / "bin" / "Rscript.exe")
                    candidate_paths.append(version_dir / "bin" / "x64" / "Rscript.exe")

    elif system_name == "Darwin":
        candidate_paths.extend(
            [
                Path("/Library/Frameworks/R.framework/Resources/bin/Rscript"),
                Path("/opt/homebrew/bin/Rscript"),
                Path("/usr/local/bin/Rscript"),
                Path("/usr/bin/Rscript"),
                Path.home() / "homebrew" / "bin" / "Rscript",
            ]
        )

    else:
        candidate_paths.extend(
            [
                Path("/usr/local/bin/Rscript"),
                Path("/usr/bin/Rscript"),
                Path("/snap/bin/Rscript"),
            ]
        )

    matches = _existing_candidates(candidate_paths)
    if matches:
        return str(matches[0])

    return prompt_for_rscript()


def r_string(path: Path) -> str:
    """Return a path safely quoted for use inside an R string literal."""
    text = path.resolve().as_posix()
    return '"' + text.replace('"', '\\"') + '"'


def run_command(cmd: List[str], cwd: Path, label: str) -> None:
    """Run a command, stream output to console, and fail on nonzero exit."""
    print("\n" + "=" * 80)
    print(f"Running: {label}")
    print("=" * 80)
    print("Command:", " ".join(cmd))
    print("Working directory:", str(cwd))
    print()
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def main() -> int:
    """Run all R analysis scripts in the required order."""
    project_root = Path(__file__).resolve().parent
    r_dir = project_root / "R"

    if not r_dir.is_dir():
        print(
            "ERROR: Could not find the R directory at: "
            f"{r_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        rscript_bin = find_rscript()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Using Rscript: {rscript_bin}")
    sys.stdout.flush()

    direct_scripts = [
        r_dir / "00_descriptive_statistics.R",
        r_dir / "01_BSS.R",
        r_dir / "02_factor_analysis.R",
    ]

    heckman_script = r_dir / "03_heckman_selection_robustness.R"
    event_study_script = r_dir / "04_event_study.R"

    for script_path in direct_scripts + [heckman_script, event_study_script]:
        if not script_path.is_file():
            print(f"ERROR: Missing script: {script_path}", file=sys.stderr)
            return 1

    try:
        for script_path in direct_scripts:
            run_command(
                [rscript_bin, str(script_path)],
                cwd=project_root,
                label=script_path.relative_to(project_root).as_posix(),
            )

        heckman_expr = (
            f"options(polymarket.autorun = FALSE); "
            f"source({r_string(heckman_script)}); "
            f"run_heckman_selection_robustness(root = getwd())"
        )
        run_command(
            [rscript_bin, "-e", heckman_expr],
            cwd=project_root,
            label="R/03_heckman_selection_robustness.R",
        )

        event_expr = (
            f"source({r_string(event_study_script)}); "
            f"run_polymarket_event_study(root = getwd())"
        )
        run_command(
            [rscript_bin, "-e", event_expr],
            cwd=project_root,
            label="R/04_event_study.R",
        )

    except subprocess.CalledProcessError as exc:
        print("\nERROR: One of the R scripts failed.", file=sys.stderr)
        print(f"Exit code: {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    print("\nAll R scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
