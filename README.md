# Backtesting Project — BMGT 442

A backtesting project for BMGT 442 that analyzes stock return predictors using Fama-French factors, momentum, CRSP data, and open-source asset pricing signals.

---

## How to Set Up (Step by Step)

### 1. Install Python

If you don't have Python installed yet:

1. Go to https://www.python.org/downloads/
2. Click the big yellow **"Download Python"** button
3. Open the installer and **check the box that says "Add Python to PATH"** (this is important!)
4. Click **Install Now** and wait for it to finish

### 2. Download This Project

**Option A — If you have Git:**

```
git clone <your-repo-url-here>
cd backtestingProject
```

**Option B — No Git:**

1. On the GitHub page, click the green **"Code"** button
2. Click **"Download ZIP"**
3. Unzip the folder somewhere you can find it (like your Desktop)

### 3. Open a Terminal

- **Mac:** Open the **Terminal** app (search for "Terminal" in Spotlight)
- **Windows:** Open **Command Prompt** (search for "cmd" in the Start menu)

Then navigate to the project folder. For example, if it's on your Desktop:

```
cd Desktop/backtestingProject
```

### 4. Install Required Packages

Copy and paste this into your terminal and press Enter:

```
pip install -r requirements.txt
```

Wait for everything to finish installing. You'll see a bunch of text scrolling — that's normal.

### 5. Open the Notebook

Copy and paste this into your terminal and press Enter:

```
jupyter notebook
```

This will open a page in your web browser. Click on **main.ipynb** to open the project notebook.

### 6. Download the Data

The data files are too large for GitHub, so they're hosted on Google Drive.

1. In the notebook, the **very first cell** will download all the data for you automatically. Just click on that cell and press **Shift + Enter** to run it.
2. Wait for the download to finish (the large file is ~8 GB so it may take a while).
3. Once you see `Done! All data files downloaded to the 'data/' folder.`, you're good to go.
4. You only need to do this once — after that you can skip the first cell.

### 7. Run the Analysis

You have **two equivalent ways** to run the analysis — pick whichever you prefer.

#### Option A — Notebook

Click on each cell in order and press **Shift + Enter** to run them one by one, or go to **Cell > Run All** in the menu bar.

#### Option B — Python pipeline + Streamlit dashboard

The notebook has been extracted into plain Python modules in `project/` so the pipeline can be run from the command line and results can be viewed in an interactive dashboard.

1. **(Once) Download the data** — same ~8 GB Google Drive pull as the notebook:

   ```
   python project/setup_data.py
   ```

   Files already in `data/` are skipped, so re-running is safe.

2. **Run the pipeline** — load → clean+merge → z-score, writing CSVs to `cleaned_data/`:

   ```
   python project/main.py
   ```

   You'll see progress for each step and the final row counts. It produces:
   - `cleaned_data/cleaned_merged_data.csv`
   - `cleaned_data/ff_factors_clean.csv`
   - `cleaned_data/cleaned_merged_zscored.csv`
   - `cleaned_data/pipeline_stats.json` (used by the dashboard)

3. **Open the dashboard** — interactive Streamlit app:
   ```
   streamlit run project/dashboard.py
   ```
   It opens in your browser and shows:
   1. Pipeline summary (row counts through each stage)
   2. Panel composition (stocks per year, FF49 industry breakdown)
   3. Top 30 factors by missingness
   4. Z-score sanity checks (pick a factor, see before/after histogram, mean, std)
   5. Annual return distribution by year

---

## Project Files

| File                             | What It Is                                                               |
| -------------------------------- | ------------------------------------------------------------------------ |
| `main.ipynb`                     | The original notebook with all the code (still works as-is)              |
| `project/setup_data.py`          | One-time Google Drive downloader                                         |
| `project/step1_data_loading.py`  | Reads the 5 raw source files into DataFrames                             |
| `project/step2_data_cleaning.py` | CRSP filter, FF49 mapping, annual returns, OAP merge, FF-factor cleaning |
| `project/step3_zscoring.py`      | Industry-year z-scoring of the 209 factor columns                        |
| `project/main.py`                | Runs steps 1→2→3 end-to-end and writes outputs                           |
| `project/dashboard.py`           | Streamlit dashboard reading the outputs                                  |
| `data/`                          | Raw datasets (downloaded via step 6 / `setup_data.py`)                   |
| `cleaned_data/`                  | Pipeline outputs written by `project/main.py`                            |
| `requirements.txt`               | List of Python packages needed                                           |
