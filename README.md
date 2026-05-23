# Polymarket Earnings Study

## Questions or contact

If you have any questions about this project, contact **Markus Johansson**:

- LinkedIn: [www.linkedin.com/in/markus-johansson](https://www.linkedin.com/in/markus-johansson)
- Email: [markusj.99@outlook.com](mailto:markusj.99@outlook.com)

---

## What this project does

This project builds a dataset and statistical analysis for Polymarket earnings markets. It collects closed Polymarket markets, validates earnings outcomes using Refinitiv/Eikon data, fetches company and stock-price data, fetches Polymarket historical prices, computes Brier scores, creates complete analysis datasets, and then runs R-based statistical analyses.

The easiest way to run the project is with:

```bash
python run_all.py
```

The full pipeline requires both **Python** and **R**. Some data-collection steps also require a valid **Refinitiv Workspace/Eikon account and API key**. This project was tested with Python 3.13.13. Use Python 3.13 to increase probability of successfully executing the scripts..

---

## Important before you start

There are two types of scripts in this project:

1. **Python scripts**: used mainly for data collection, validation, merging, and dataset construction.
2. **R scripts**: used mainly for statistics, plots, tables, and final analysis outputs.

The file `run_all.py` runs everything in the correct order:

1. Python scripts `00` through `07`
2. R scripts `00` through `04`

If you do not provide a Refinitiv/Eikon API key, `run_all.py` will skip the scripts that require that key. However, later scripts may fail if they need files that would normally be created by the skipped API-dependent steps. For a complete fresh run, you should use a valid Refinitiv/Eikon API key. Without an API key, the statistics cannot be calculated.

---

## Table of contents

1. [Polymarket Earnings Study](#polymarket-earnings-study)
2. [Questions or contact](#questions-or-contact)
3. [What this project does](#what-this-project-does)
4. [Important before you start](#important-before-you-start)
5. [Required software](#required-software)
   - [Python](#1-python)
   - [R](#2-r)
   - [Refinitiv Workspace or Eikon Desktop](#3-refinitiv-workspace-or-eikon-desktop)
6. [How to set up Refinitiv/Eikon](#how-to-set-up-refinitiveikon)
   - [Open Workspace or Eikon Desktop](#step-1-open-workspace-or-eikon-desktop)
   - [Create or retrieve an API key](#step-2-create-or-retrieve-an-api-key)
   - [Provide the API key when running the project](#step-3-provide-the-api-key-when-running-the-project)
7. [Recommended folder structure](#recommended-folder-structure)
8. [First-time setup for users with no coding experience](#first-time-setup-for-users-with-no-coding-experience)
   - [Open a terminal](#step-1-open-a-terminal)
   - [Move into the project folder](#step-2-move-into-the-project-folder)
   - [Create a Python virtual environment](#step-3-create-a-python-virtual-environment)
   - [Upgrade Python packaging tools](#step-4-upgrade-python-packaging-tools)
   - [Install Python packages](#step-5-install-python-packages)
   - [Install R packages](#step-6-install-r-packages)
   - [Test what will run](#step-7-test-what-will-run)
9. [How to run the full project](#how-to-run-the-full-project)
   - [Full run with API key prompt](#full-run-with-api-key-prompt)
   - [Full run with environment variable](#full-run-with-environment-variable)
   - [Full run with key passed directly](#full-run-with-key-passed-directly)
   - [Run without an API key](#run-without-an-api-key)
   - [Continue even if one step fails](#continue-even-if-one-step-fails)
   - [Run only Python scripts](#run-only-python-scripts)
   - [Run only R scripts](#run-only-r-scripts)
10. [How to run individual scripts](#how-to-run-individual-scripts)
    - [Run one Python script](#run-one-python-script)
    - [Run one R script](#run-one-r-script)
11. [Script run order](#script-run-order)
    - [Python pipeline](#python-pipeline)
    - [R statistics pipeline](#r-statistics-pipeline)
12. [Folder structure and outputs](#folder-structure-and-outputs)
    - [`data/`](#data)
    - [`statistics/`](#statistics)
13. [Expected outputs after a successful full run](#expected-outputs-after-a-successful-full-run)
14. [Troubleshooting](#troubleshooting)
    - [`python` is not recognized](#python-is-not-recognized)
    - [`Rscript` is not recognized or not found](#rscript-is-not-recognized-or-not-found)
    - [Python package installation fails](#python-package-installation-fails)
    - [R package installation fails](#r-package-installation-fails)
    - [Refinitiv/Eikon scripts fail with network or proxy errors](#refinitiveikon-scripts-fail-with-network-or-proxy-errors)
    - [`Input file not found`](#input-file-not-found)
    - [The R scripts cannot find the project root](#the-r-scripts-cannot-find-the-project-root)
    - [The pipeline stops after one failed script](#the-pipeline-stops-after-one-failed-script)
    - [The project folder is in a synced or protected folder](#the-project-folder-is-in-a-synced-or-protected-folder)
15. [Reproducibility notes](#reproducibility-notes)
    - [Relative paths](#relative-paths)
    - [API data may change over time](#api-data-may-change-over-time)
    - [Package versions](#package-versions)
    - [API keys and credentials](#api-keys-and-credentials)
    - [Recommended reproducibility workflow](#recommended-reproducibility-workflow)
16. [Quick command summary](#quick-command-summary)

---

## Required software

### 1. Python

Install Python before running the Python scripts.

Recommended version: **Python 3.12 or 3.13**.

#### Check whether Python is installed

Open a terminal and run:

```bash
python --version
```

On some macOS or Linux computers, the command may be:

```bash
python3 --version
```

If Python is installed, you should see something like:

```text
Python 3.11.8
```

If Python is not installed, download it from:

- [https://www.python.org/downloads/](https://www.python.org/downloads/)

On Windows, make sure to tick **Add Python to PATH** during installation.

---

### 2. R

Install R before running the R scripts.

Recommended version: **R 4.2 or newer**.

#### Check whether R is installed

Open a terminal and run:

```bash
Rscript --version
```

If R is installed, you should see an R version number.

If R is not installed, download it from:

- [https://cran.r-project.org/](https://cran.r-project.org/)

Optional but recommended:

- Install RStudio from [https://posit.co/download/rstudio-desktop/](https://posit.co/download/rstudio-desktop/)

RStudio is not required to run the scripts, but it can make it easier to inspect data and troubleshoot R code.

---

### 3. Refinitiv Workspace or Eikon Desktop

Some scripts use the Refinitiv/Eikon Data API. For those scripts, you need:

- Refinitiv Workspace or Eikon Desktop installed.
- A valid Refinitiv/Eikon user account.
- Access rights for the required data fields.
- An Eikon Data API app key.
- Workspace/Eikon open and logged in on the **same computer** that runs the code.

The API works through a local desktop connection. This means that logging into Workspace/Eikon on another computer is not enough. The Workspace/Eikon application must be running on the computer where you run the Python scripts.

Official LSEG/Refinitiv setup references:

- [Eikon Data API Quick Start](https://developers.lseg.com/en/api-catalog/eikon/eikon-data-api/quick-start)
- [Eikon Data API documentation](https://developers.lseg.com/en/api-catalog/eikon/eikon-data-api/documentation)
- [Eikon Data API troubleshooting](https://developers.lseg.com/en/article-catalog/article/eikon-data-api-python-troubleshooting-refinitiv)

---

## How to set up Refinitiv/Eikon

### Step 1: Open Workspace or Eikon Desktop

Open Refinitiv Workspace or Eikon Desktop on the same computer where you will run the code.

Log in with your Refinitiv/Eikon account.

Leave Workspace/Eikon open while the code is running.

---

### Step 2: Create or retrieve an API key

In Workspace or Eikon Desktop:

1. Use the search bar.
2. Search for **App Key** or **AppKey Generator**.
3. Open the **App Key Generator** app.
4. Create a new app key for this project.
5. Select the Eikon Data API option if the app asks which API type to enable.
6. Copy the generated app key.

Keep the key private. Do not upload it to GitHub, do not email it publicly, and do not save it inside scripts.

---

### Step 3: Provide the API key when running the project

You can provide the key in three ways.

#### Option A: Let `run_all.py` ask for it

Run:

```bash
python run_all.py
```

When prompted, paste the API key and press Enter.

If you press Enter without typing a key, the API-key-required scripts will be skipped.

---

#### Option B: Pass the key directly

```bash
python run_all.py --app-key YOUR_API_KEY_HERE
```

This is simple, but less private because the key may be visible in your terminal history.

---

#### Option C: Set an environment variable

This is usually the best option.

On macOS or Linux:

```bash
export EIKON_APP_KEY="YOUR_API_KEY_HERE"
python run_all.py
```

On Windows PowerShell:

```powershell
$env:EIKON_APP_KEY="YOUR_API_KEY_HERE"
python run_all.py
```

On Windows Command Prompt:

```cmd
set EIKON_APP_KEY=YOUR_API_KEY_HERE
python run_all.py
```

---

## Recommended folder structure

The project folder should look like this:

```text
Polymarket-Earnings-Study/
├── run_all.py
├── requirements.txt
├── install_r_packages.R
├── README.md
├── python/
│   ├── 00_fetch_closed_markets.py
│   ├── 01_check_consistency.py
│   ├── 01b_retry_unmatched_earnings.py
│   ├── 02_fetch_corp_info.py
│   ├── 03_fetch_stock_prices.py
│   ├── 04_fetch_poly_prices.py
│   ├── 05_brier_scores.py
│   ├── 06_create_dataset.py
│   └── 07_heckman_selection_fetch_universe.py
├── R/
│   ├── 00_descriptive_statistics.R
│   ├── 01_BSS.R
│   ├── 02_factor_analysis.R
│   ├── 03_heckman_selection_robustness.R
│   ├── 04_event_study.R
│   └── utils/
│       ├── load_data.R
│       └── pm_common.R
├── data/
│   └── generated files appear here
└── statistics/
    └── generated statistics, tables, and figures appear here
```

The script `run_all.py` can also handle some common alternative layouts, but the structure above is the recommended layout.

Important: several R scripts search for the project root using markers such as `renv.lock` or an `.Rproj` file. If your project includes one of those files, keep it in the project root.

---

## First-time setup for users with no coding experience

The steps below assume you have downloaded and unzipped the project folder.

### Step 1: Open a terminal

#### Windows

Use **PowerShell**.

You can open it by searching for `PowerShell` in the Start menu.

#### macOS

Use **Terminal**.

You can open it from Applications > Utilities > Terminal.

#### Linux

Use your normal terminal application.

---

### Step 2: Move into the project folder

The command is `cd`, which means “change directory”.

Example:

```bash
cd path/to/Polymarket-Earnings-Study
```

Examples by operating system:

Windows PowerShell:

```powershell
cd "$HOME\Downloads\Polymarket-Earnings-Study"
```

macOS or Linux:

```bash
cd ~/Downloads/Polymarket-Earnings-Study
```

You are in the correct folder if this command shows `run_all.py`:

```bash
ls
```

On Windows Command Prompt, use:

```cmd
dir
```

---

### Step 3: Create a Python virtual environment

A virtual environment keeps this project’s Python packages separate from the rest of your computer.

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal line will usually show `(.venv)` at the beginning.

---

### Step 4: Upgrade Python packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

### Step 5: Install Python packages

```bash
python -m pip install -r requirements.txt
```

This installs the Python packages needed by the pipeline, including:

- `requests`
- `tqdm`
- `pandas`
- `numpy`
- `eikon`

---

### Step 6: Install R packages

```bash
Rscript install_r_packages.R
```

This installs the R packages used by the statistical scripts, including packages for data handling, plotting, regression models, HTML tables, and the Heckman selection model.

If this step fails on Windows because R needs compilation tools, install Rtools from:

- [https://cran.r-project.org/bin/windows/Rtools/](https://cran.r-project.org/bin/windows/Rtools/)

Then run the command again:

```bash
Rscript install_r_packages.R
```

---

### Step 7: Test what will run

Before running the full pipeline, run a dry run:

```bash
python run_all.py --dry-run
```

This prints the planned run order without actually running the scripts.

If `Rscript` is missing, the dry run will show that R cannot be found. Install R or provide the path manually.

Example with a manual Rscript path on Windows:

```powershell
python run_all.py --rscript "C:\Program Files\R\R-4.4.0\bin\Rscript.exe" --dry-run
```

Change `R-4.4.0` to the R version installed on your computer.

---

## How to run the full project

### Full run with API key prompt

```bash
python run_all.py
```

`run_all.py` will ask for your Eikon/Refinitiv API key. Paste it and press Enter.

---

### Full run with environment variable

macOS or Linux:

```bash
export EIKON_APP_KEY="YOUR_API_KEY_HERE"
python run_all.py
```

Windows PowerShell:

```powershell
$env:EIKON_APP_KEY="YOUR_API_KEY_HERE"
python run_all.py
```

---

### Full run with key passed directly

```bash
python run_all.py --app-key YOUR_API_KEY_HERE
```

---

### Run without an API key

```bash
python run_all.py
```

When asked for the key, press Enter.

The scripts that require Refinitiv/Eikon will be skipped. This is useful if you only want to run the public-data or local-file steps, but it will not produce a complete fresh dataset unless the missing Refinitiv/Eikon outputs already exist from a previous run.

---

### Continue even if one step fails

By default, `run_all.py` stops if a script fails.

To continue after failures:

```bash
python run_all.py --keep-going
```

This is useful when debugging, but inspect the output carefully. Later scripts may produce incomplete results if earlier steps failed.

---

### Run only Python scripts

```bash
python run_all.py --python-only
```

---

### Run only R scripts

```bash
python run_all.py --r-only
```

This requires the data files created by the Python pipeline to already exist.

---

## How to run individual scripts

You normally do not need to run scripts one by one. Use `run_all.py` unless you are debugging.

### Run one Python script

From the project root:

```bash
python python/00_fetch_closed_markets.py
```

For scripts that require Refinitiv/Eikon, either set `EIKON_APP_KEY` first or pass the key if the script supports it.

Example:

```bash
python python/02_fetch_corp_info.py --app-key
```

Some scripts read the key from the environment even when `--app-key` is passed as a flag. The safest approach is:

```bash
export EIKON_APP_KEY="YOUR_API_KEY_HERE"
python python/02_fetch_corp_info.py --app-key
```

On Windows PowerShell:

```powershell
$env:EIKON_APP_KEY="YOUR_API_KEY_HERE"
python python/02_fetch_corp_info.py --app-key
```

---

### Run one R script

From the project root:

```bash
Rscript --vanilla R/00_descriptive_statistics.R
```

Some R scripts expect the Python pipeline outputs to exist before they can run.

---

## Script run order

`run_all.py` runs the following scripts in this order.

### Python pipeline

| Order | Script | Requires Refinitiv/Eikon API key? | Main purpose |
|---:|---|---|---|
| 00 | `python/00_fetch_closed_markets.py` | No | Fetch closed Polymarket earnings markets from public Polymarket/Gamma endpoints. |
| 01 | `python/01_check_consistency.py` | Yes | Validate Polymarket market outcomes against Refinitiv/Eikon earnings data. |
| 01b | `python/01b_retry_unmatched_earnings.py` | Yes | Retry markets that were not matched in the consistency check. |
| 02 | `python/02_fetch_corp_info.py` | Yes | Fetch company-level and event-time information from Refinitiv/Eikon. |
| 03 | `python/03_fetch_stock_prices.py` | Yes | Fetch daily stock prices and S&P 500 prices from Refinitiv/Eikon. |
| 04 | `python/04_fetch_poly_prices.py` | No | Fetch historical Polymarket YES/NO prices. |
| 05 | `python/05_brier_scores.py` | No | Compute Brier scores by market and horizon. |
| 06 | `python/06_create_dataset.py` | No | Create complete long and wide analysis datasets. |
| 07 | `python/07_heckman_selection_fetch_universe.py` | Yes | Fetch broader earnings-event universe for Heckman selection analysis. |

### R statistics pipeline

| Order | Script | Main purpose |
|---:|---|---|
| 00 | `R/00_descriptive_statistics.R` | Create descriptive statistics tables, plots, logs, and README for outputs. |
| 01 | `R/01_BSS.R` | Compute Brier Skill Scores and paired tests. |
| 02 | `R/02_factor_analysis.R` | Run factor-analysis regressions and create tables/plots. |
| 03 | `R/03_heckman_selection_robustness.R` | Run Heckman selection robustness analysis. |
| 04 | `R/04_event_study.R` | Run the Polymarket-price event study. |

---

## Folder structure and outputs

### `data/`

The `data/` folder contains datasets created by the Python scripts.

#### `data/markets/`

Created by `00_fetch_closed_markets.py` and updated by `01_check_consistency.py`.

Expected files include:

```text
data/markets/markets.jsonl
data/markets/markets.csv
data/markets/discarded_markets.jsonl
data/markets/summary.txt
```

Use this folder to inspect the raw and filtered Polymarket market universe.

---

#### `data/validation/`

Created by `01_check_consistency.py` and `01b_retry_unmatched_earnings.py`.

Expected files include:

```text
data/validation/unmatched.jsonl
data/validation/unmatched.csv
data/validation/consistency_summary.txt
data/validation/unmatched_retry_matched.jsonl
data/validation/unmatched_retry_matched.csv
data/validation/unmatched_retry_still_unmatched.jsonl
data/validation/unmatched_retry_still_unmatched.csv
data/validation/unmatched_retry_summary.txt
```

Use this folder to inspect markets that were difficult to match to Refinitiv/Eikon earnings data.

---

#### `data/corporate_info/`

Created by `02_fetch_corp_info.py`.

Expected files include:

```text
data/corporate_info/corporate_info_by_market.jsonl
data/corporate_info/corporate_info_by_market.csv
data/corporate_info/corporate_info_by_market_summary.txt
```

Use this folder for company characteristics and earnings-release timing data.

---

#### `data/stock_prices/`

Created by `03_fetch_stock_prices.py`.

Expected files include:

```text
data/stock_prices/stock_prices_daily.csv
data/stock_prices/stock_prices_daily.jsonl
data/stock_prices/stock_prices_daily.json
data/stock_prices/stock_prices_summary.txt
```

Use this folder for daily stock-price data and benchmark data.

---

#### `data/poly_prices/`

Created by `04_fetch_poly_prices.py`.

Expected files include:

```text
data/poly_prices/poly_prices.jsonl
data/poly_prices/failed_poly_markets.jsonl
data/poly_prices/poly_prices_wide.csv
data/poly_prices/poly_prices_long.csv
data/poly_prices/failed_poly_markets.csv
data/poly_prices/summary.txt
```

Use this folder for Polymarket price snapshots before earnings announcements.

---

#### `data/brier_scores/`

Created by `05_brier_scores.py`.

Expected files include:

```text
data/brier_scores/brier_scores_market_horizon.csv
data/brier_scores/brier_scores_market_horizon.json
data/brier_scores/brier_scores_market_horizon.jsonl
data/brier_scores/brier_scores_by_horizon.csv
data/brier_scores/brier_scores_by_horizon.json
data/brier_scores/brier_scores_by_horizon.jsonl
```

Use this folder to inspect Brier scores at the market-horizon level and aggregated by horizon.

---

#### Complete analysis datasets

Created by `06_create_dataset.py`.

Expected files include:

```text
data/complete_dataset_long.csv
data/complete_dataset_long.jsonl
data/complete_dataset_wide.csv
data/complete_dataset_wide.jsonl
```

These are the most important datasets for analysis.

Use:

- `data/complete_dataset_long.csv` when you want one row per market and horizon.
- `data/complete_dataset_wide.csv` when you want one row per market with horizon variables spread across columns.

---

#### `data/heckman_selection_model/`

Created by `07_heckman_selection_fetch_universe.py`.

Expected files include:

```text
data/heckman_selection_model/screener_universe_rics.csv
data/heckman_selection_model/screener_universe_rics.jsonl
data/heckman_selection_model/heckman_universe_companies.csv
data/heckman_selection_model/heckman_universe_events.csv
data/heckman_selection_model/heckman_universe_events.jsonl
data/heckman_selection_model/heckman_missing_summary.json
data/heckman_selection_model/heckman_report.txt
```

Use this folder for the broader earnings-event universe used by the Heckman selection analysis.

---

### `statistics/`

The `statistics/` folder contains the final analysis outputs created by the R scripts.

#### `statistics/descriptive_statistics/`

Created by `R/00_descriptive_statistics.R`.

Contains descriptive tables, plots, logs, and an output README.

---

#### `statistics/brier_analysis/`

Created by `R/01_BSS.R`.

Contains Brier score summaries, Brier Skill Score tables, paired test outputs, HTML tables, and figures.

---

#### `statistics/factor_analysis/`

Created by `R/02_factor_analysis.R`.

Expected files include:

```text
statistics/factor_analysis/factor_analysis_regression_table.html
statistics/factor_analysis/factor_analysis_regression_coefficients.csv
statistics/factor_analysis/factor_analysis_regression_coefficients.jsonl
statistics/factor_analysis/factor_analysis_model_fit.csv
statistics/factor_analysis/factor_analysis_model_fit.jsonl
statistics/factor_analysis/factor_analysis_market_level.csv
statistics/factor_analysis/factor_analysis_market_level.jsonl
statistics/factor_analysis/factor_analysis_plot_data.csv
statistics/factor_analysis/factor_analysis_plot_data.jsonl
statistics/factor_analysis/factor_analysis_coefficients_plot.png
```

---

#### `statistics/heckman_selection/`

Created by `R/03_heckman_selection_robustness.R`.

Contains matching diagnostics, Heckman model panels, coefficient outputs, model-fit files, and HTML tables.

---

#### `statistics/event_study/`

Created by `R/04_event_study.R`.

Contains event-study datasets, plots, CAAR outputs, and HTML tables.

---

## Expected outputs after a successful full run

After a successful full run, the most important outputs are:

```text
data/complete_dataset_long.csv
data/complete_dataset_wide.csv
statistics/descriptive_statistics/
statistics/brier_analysis/
statistics/factor_analysis/
statistics/heckman_selection/
statistics/event_study/
```

For most users, the best starting points are:

1. `data/complete_dataset_long.csv`
2. `data/complete_dataset_wide.csv`
3. HTML tables inside `statistics/`
4. PNG/PDF figures inside `statistics/`
5. Summary text files inside each output folder

---

## Troubleshooting

### `python` is not recognized

Python is either not installed or not added to your PATH.

Fix:

1. Install Python from [https://www.python.org/downloads/](https://www.python.org/downloads/).
2. On Windows, tick **Add Python to PATH** during installation.
3. Restart your terminal.
4. Try again:

```bash
python --version
```

On macOS/Linux, try:

```bash
python3 --version
```

---

### `Rscript` is not recognized or not found

R is either not installed or not added to your PATH.

Fix:

1. Install R from [https://cran.r-project.org/](https://cran.r-project.org/).
2. Restart your terminal.
3. Try:

```bash
Rscript --version
```

If that still fails, pass the Rscript path directly:

```bash
python run_all.py --rscript "C:\Program Files\R\R-4.4.0\bin\Rscript.exe"
```

---

### Python package installation fails

Try upgrading `pip`, `setuptools`, and `wheel`:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If the `eikon` package fails, check that your Python version is supported. Python 3.10, 3.11, or 3.12 is recommended.

---

### R package installation fails

Try running:

```bash
Rscript install_r_packages.R
```

If packages fail to compile on Windows, install Rtools:

- [https://cran.r-project.org/bin/windows/Rtools/](https://cran.r-project.org/bin/windows/Rtools/)

Then rerun:

```bash
Rscript install_r_packages.R
```

---

### Refinitiv/Eikon scripts fail with network or proxy errors

Common causes:

- Workspace/Eikon is not open.
- You are not logged in.
- You are logged in on a different computer.
- The API key is missing or invalid.
- Your account does not have access to the requested data.
- The local desktop API proxy is not responding.

Fix:

1. Open Workspace/Eikon on the same computer.
2. Log in.
3. Confirm that the App Key Generator works.
4. Confirm that the key is set as `EIKON_APP_KEY` or passed to `run_all.py`.
5. Restart Workspace/Eikon if needed.
6. Run again.

Example:

```bash
python run_all.py --app-key YOUR_API_KEY_HERE
```

---

### `Input file not found`

This usually means an earlier script failed or was skipped.

Example: if `data/corporate_info/corporate_info_by_market.jsonl` is missing, then scripts that need corporate information cannot run.

Fix:

1. Look at the first script that failed.
2. Fix that error first.
3. Rerun the pipeline.

Use:

```bash
python run_all.py --dry-run
```

Then run:

```bash
python run_all.py
```

---

### The R scripts cannot find the project root

Some R scripts look for `renv.lock` or an `.Rproj` file to identify the project root.

Fix:

- Run scripts from the project root.
- Keep `renv.lock` or the `.Rproj` file in the project root if your project includes one.
- Prefer running through `run_all.py`, which sets the working directory to the project root for R scripts.

---

### The pipeline stops after one failed script

This is the default behavior.

To continue after failures, use:

```bash
python run_all.py --keep-going
```

Use this mainly for debugging. For final results, rerun after fixing the original failure.

---

### The project folder is in a synced or protected folder

OneDrive, iCloud Drive, Dropbox, or protected system folders can sometimes cause file-locking or permission problems.

Fix:

Move the project to a simple local folder, for example:

Windows:

```text
C:\Users\YOUR_NAME\Documents\Polymarket-Earnings-Study
```

macOS/Linux:

```text
~/Documents/Polymarket-Earnings-Study
```

---

## Reproducibility notes

### Relative paths

The project is designed to use paths relative to the project root. This makes it easier for a new user to download the project folder and run it on another computer.

Do not move individual scripts out of the project folder unless you also update paths accordingly.

---

### API data may change over time

This project retrieves data from public Polymarket endpoints and Refinitiv/Eikon. Re-running the data-collection scripts later may produce differences if:

- Polymarket API responses change.
- Refinitiv/Eikon revises or updates data.
- Your data entitlements differ from another user’s entitlements.
- Markets have been added, corrected, or changed upstream.

For reproducibility, keep copies of generated raw files, especially:

```text
data/markets/
data/corporate_info/
data/stock_prices/
data/poly_prices/
data/heckman_selection_model/
```

---

### Package versions

The file `requirements.txt` installs Python packages using version ranges rather than exact pinned versions.

For stricter reproducibility, create a lock file after a successful run:

```bash
python -m pip freeze > requirements-lock.txt
```

A future user can then install the exact Python package versions with:

```bash
python -m pip install -r requirements-lock.txt
```

For R, the strongest reproducibility option is to use `renv` and keep `renv.lock` in the project root.

---

### API keys and credentials

Never commit API keys to GitHub or include them in shared scripts.

Use one of these instead:

- The interactive prompt in `run_all.py`
- The `EIKON_APP_KEY` environment variable
- A private local `.env` file if you add your own environment-loading workflow

---

### Recommended reproducibility workflow

After a successful full run:

1. Save the generated `data/` folder.
2. Save the generated `statistics/` folder.
3. Save `requirements-lock.txt` if you created one.
4. Save `renv.lock` if using `renv`.
5. Record the run date, operating system, Python version, R version, and whether Refinitiv/Eikon was used.

Useful commands:

```bash
python --version
Rscript --version
python -m pip freeze > requirements-lock.txt
```

---

## Quick command summary

From the project root:

```bash
# 1. Create and activate Python environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

# 2. Install Python packages
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 3. Install R packages
Rscript install_r_packages.R

# 4. Optional: set Refinitiv/Eikon API key
export EIKON_APP_KEY="YOUR_API_KEY_HERE"

# 5. Check the planned run order
python run_all.py --dry-run

# 6. Run everything
python run_all.py
```

On Windows PowerShell, replace the environment-variable command with:

```powershell
$env:EIKON_APP_KEY="YOUR_API_KEY_HERE"
```
