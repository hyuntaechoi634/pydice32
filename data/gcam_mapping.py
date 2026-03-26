"""
ISO3 → GCAM 32-region mapping and aggregation utilities.
"""

import os
import pandas as pd


def load_rice_regions(project_root):
    """Load 155 ISO3 region codes from n.inc."""
    ninc = os.path.join(project_root, "RICE50xmodel", "data_maxiso3", "n.inc")
    try:
        regions = []
        for line in open(ninc):
            s = line.strip()
            if s:
                regions.append(s)
        return regions
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required data file not found: {ninc}\n"
            f"Ensure RICE50xmodel/data_maxiso3/ directory is present."
        )


def load_gcam_mapping(gcam_csv, rice_regions):
    """Map ISO3 -> GCAM region ID (1-32)."""
    try:
        df = pd.read_csv(gcam_csv, comment="#")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required data file not found: {gcam_csv}\n"
            f"Ensure RICE50xmodel/data_maxiso3_csv/ directory is present."
        )
    df.columns = ["iso", "country_name", "region_GCAM3", "GCAM_region_ID"]
    df["iso"] = df["iso"].str.lower().str.strip()
    mapping = {}
    for _, row in df.iterrows():
        iso = row["iso"]
        try:
            gid = int(row["GCAM_region_ID"])
        except (ValueError, TypeError):
            continue
        if iso in rice_regions and 1 <= gid <= 32:
            mapping[iso] = gid
    return mapping


def load_gcam_region_names(gcam_names_csv):
    try:
        df = pd.read_csv(gcam_names_csv, comment="#")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required data file not found: {gcam_names_csv}\n"
            f"Ensure RICE50xmodel/data_maxiso3_csv/ directory is present."
        )
    df.columns = ["GCAM_region_ID", "region"]
    return {int(row["GCAM_region_ID"]): row["region"].strip()
            for _, row in df.iterrows()}


def aggregate_param_1d(raw_dict, mapping, gcam_names, weight_dict=None):
    """Aggregate 1D param to 32 GCAM regions. Sum or weighted avg."""
    result = {gcam_names[r]: 0.0 for r in range(1, 33)}
    weights = {gcam_names[r]: 0.0 for r in range(1, 33)}
    for iso, val in raw_dict.items():
        if iso in mapping:
            rname = gcam_names[mapping[iso]]
            if weight_dict is not None:
                w = weight_dict.get(iso, 0.0)
                result[rname] += val * w
                weights[rname] += w
            else:
                result[rname] += val
    if weight_dict is not None:
        for rname in result:
            if weights[rname] > 0:
                result[rname] /= weights[rname]
    return result


def aggregate_param_tn(df_raw, ssp, mapping, gcam_names, T,
                       weight_fn=None, ghg_col=None, ghg_filter=None):
    """Aggregate T x N parameter to GCAM 32 regions."""
    result = {}
    weights = {}
    for _, row in df_raw.iterrows():
        row_ssp = str(row.iloc[0]).upper()
        if row_ssp != ssp.upper():
            continue
        t_val = int(row["t"])
        n_val = str(row["n"]).lower()
        if n_val not in mapping or t_val < 1 or t_val > T:
            continue
        rname = gcam_names[mapping[n_val]]
        if ghg_col is not None:
            ghg = str(row.iloc[ghg_col]).lower()
            if ghg_filter and ghg != ghg_filter:
                continue
            key = (t_val, rname, ghg)
        else:
            key = (t_val, rname)
        if key not in result:
            result[key] = 0.0
            weights[key] = 0.0
        if weight_fn is not None:
            w = weight_fn(t_val, n_val)
            result[key] += row["Val"] * w
            weights[key] += w
        else:
            result[key] += row["Val"]
    if weight_fn is not None:
        for key in result:
            if weights[key] > 0:
                result[key] /= weights[key]
    return result
