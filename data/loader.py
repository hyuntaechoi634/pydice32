"""
Low-level CSV loading utilities.
Corresponds to GAMS $gdxin / $load phases.
"""

import os
import pandas as pd


def load_csv(data_dir, subdir, name):
    filepath = os.path.join(data_dir, subdir, name)
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required data file not found: {filepath}\n"
            f"Ensure RICE50xmodel/data_maxiso3_csv/ directory is present."
        )


def load_1d(data_dir, subdir, name):
    """Load parameter indexed by region n -> {region: value}."""
    df = load_csv(data_dir, subdir, name)
    return dict(zip(df.iloc[:, 0].str.lower(), df["Val"]))


def load_validation_param(data_dir, subdir, name, dim1_key):
    """Load from multi-dim CSV (Dim1, t, n, Val). Filter Dim1=key, t=1."""
    df = load_csv(data_dir, subdir, name)
    mask = ((df.iloc[:, 0].str.lower() == dim1_key.lower())
            & (df["t"].astype(str) == "1"))
    filtered = df[mask]
    return dict(zip(filtered["n"].str.lower(), filtered["Val"]))
