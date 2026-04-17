"""Step 1: Load the five raw source files into DataFrames."""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd

REQUIRED_FILES = {
    "ff_factors": "F-F_Research_Data_Factors.csv",
    "ff_mom": "F-F_Momentum_Factor.csv",
    "sic_mapping": "SIC_49_Industry.xlsx",
    "oap_factors": "signed_predictors_dl_wide.csv",
    "crsp": "crsp.csv",
}


def _check_files_exist(data_dir: str) -> None:
    missing = [name for name in REQUIRED_FILES.values() if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} data file(s) in '{data_dir}/': {missing}. "
            f"Run: python project/setup_data.py"
        )


def load_raw_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Read all five source files and return them in a dict."""
    _check_files_exist(data_dir)

    print("Loading raw data...")
    ff_factors = pd.read_csv(os.path.join(data_dir, REQUIRED_FILES["ff_factors"]), skiprows=3)
    ff_mom = pd.read_csv(os.path.join(data_dir, REQUIRED_FILES["ff_mom"]), skiprows=13)
    sic_mapping = pd.read_excel(os.path.join(data_dir, REQUIRED_FILES["sic_mapping"]))
    oap_factors = pd.read_csv(os.path.join(data_dir, REQUIRED_FILES["oap_factors"]))
    crsp = pd.read_csv(os.path.join(data_dir, REQUIRED_FILES["crsp"]), low_memory=False)

    for name, df in [
        ("FF 3 Factors", ff_factors),
        ("FF Momentum", ff_mom),
        ("SIC 49 Industry Mapping", sic_mapping),
        ("OAP Factors", oap_factors),
        ("CRSP", crsp),
    ]:
        print(f"  {name:28s} shape={df.shape}")

    return {
        "ff_factors": ff_factors,
        "ff_mom": ff_mom,
        "sic_mapping": sic_mapping,
        "oap_factors": oap_factors,
        "crsp": crsp,
    }


if __name__ == "__main__":
    load_raw_data()
