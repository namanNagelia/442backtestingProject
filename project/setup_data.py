"""One-time data download from Google Drive.

Run this once before the pipeline:
    python project/setup_data.py

Downloads the shared folder into ./data/. Already-present files are skipped
by gdown, so re-running is safe and cheap.
"""
from __future__ import annotations

import os
import sys

GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1ei8-m-N2vJHj2y7LHA7QYEC7FT2G-qDN?usp=sharing"
)

EXPECTED_FILES = [
    "crsp.csv",
    "F-F_Momentum_Factor.csv",
    "F-F_Research_Data_Factors.csv",
    "ret.csv",
    "SIC_49_Industry.xlsx",
    "signed_predictors_dl_wide.csv",
]


def main(output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)

    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(output_dir, f))]
    if not missing:
        print(f"All expected data files already present in '{output_dir}/'. Nothing to do.")
        return

    print(f"Missing {len(missing)} file(s): {missing}")
    print("Downloading from Google Drive (the CSV of OAP predictors is ~8 GB)...")

    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        raise

    gdown.download_folder(GOOGLE_DRIVE_FOLDER_URL, output=output_dir, quiet=False)
    print(f"Done. Files are in '{output_dir}/'.")


if __name__ == "__main__":
    main()
