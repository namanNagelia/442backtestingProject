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

Click on each cell in order and press **Shift + Enter** to run them one by one, or go to **Cell > Run All** in the menu bar.

---

## Project Files

| File | What It Is |
|------|-----------|
| `main.ipynb` | The main notebook with all the code |
| `data/` | Folder with all the datasets (downloaded via Step 6) |
| `requirements.txt` | List of Python packages needed |
