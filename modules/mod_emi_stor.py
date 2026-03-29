"""
CCS storage module: per-storage-type allocation, cumulative tracking, and leakage.

Based on GAMS mod_emi_stor.gms from RICE50x.

Variables:
    E_STOR(ccs_stor, t, n)     -- CO2 stored per type per period [GtC/yr]
    CUM_E_STOR(ccs_stor, t, n) -- cumulative stored CO2 per type [GtC]
    E_LEAK(t, n)               -- leakage emissions from stored CO2 [GtC/yr]

Equations:
    eq_stor_cum  -- cumulative storage dynamics with leakage
    eq_emi_leak  -- leakage from cumulative storage
    eq_emi_stor_dac -- E_NEG = sum(ccs_stor, E_STOR) * CtoCO2

Integration:
    - eq_emi_stor_dac links E_NEG to E_STOR (replaces the average-cost simplification)
    - eq_cost_cdr uses sum(ccs_stor, E_STOR * ccs_stor_cost) * CtoCO2
    - E_LEAK enters emissions accounting (adds leaked CO2 back)

Data files (from data_mod_emi_stor/):
    - ccs_stor_cap_aqui.csv, ccs_stor_cap_og.csv, ccs_stor_cap_ecbm.csv, ccs_stor_cap_eor.csv
    - ccs_stor_share_onoff.csv
    - ccs_stor_cost_estim.csv
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Set, Parameter, Ord, Card, Sum


# GAMS storage types
CCS_STOR_TYPES = [
    "aqui_on", "aqui_off",
    "oil_gas_no_eor_on", "oil_gas_no_eor_off",
    "eor_on", "eor_off",
    "ecbm",
]

# C-to-CO2 conversion factor
CtoCO2 = 44.0 / 12.0


def _load_storage_data(data_dir, region_names, ssp="SSP2", gcam_map=None, gcam_nm=None,
                       cap_override=None):
    """Load CCS storage capacity and cost data from CSV files.

    Data is at ISO3 country level; aggregated to GCAM regions via gcam_map.

    Parameters
    ----------
    cap_override : str, optional
        Explicit capacity scenario ("low" | "best" | "high").
        When provided, overrides the SSP-derived default.
    """
    stor_dir = os.path.join(data_dir, "data_mod_emi_stor")
    if gcam_map is None:
        gcam_map = {}
    if gcam_nm is None:
        gcam_nm = {}
    result = {}

    # SSP-based cost/capacity scenario selection (GAMS mod_emi_stor.gms lines 20-30)
    ssp_cost_map = {"SSP1": "low", "SSP2": "best", "SSP3": "high", "SSP4": "best", "SSP5": "low"}
    ssp_cap_map = {"SSP1": "low", "SSP2": "best", "SSP3": "high", "SSP4": "high", "SSP5": "high"}
    cost_scenario = ssp_cost_map.get(ssp.upper(), "best")
    cap_scenario = cap_override if cap_override else ssp_cap_map.get(ssp.upper(), "best")

    # Storage cost per type [T$/GtCO2] (same for all regions)
    stor_cost = {}
    cost_file = os.path.join(stor_dir, "ccs_stor_cost_estim.csv")
    if os.path.exists(cost_file):
        df = pd.read_csv(cost_file)
        for _, row in df.iterrows():
            stype = str(row.iloc[0]).lower()
            scenario = str(row.iloc[1]).lower()
            if scenario == cost_scenario:
                stor_cost[stype] = float(row["Val"])
    result["stor_cost"] = stor_cost

    # On/offshore share per region (aggregate ISO3 -> GCAM via pop-weight approx)
    # Simplification: use unweighted average of on/off shares within region
    share_onoff_iso = {}  # (iso, cat) -> fraction
    share_file = os.path.join(stor_dir, "ccs_stor_share_onoff.csv")
    if os.path.exists(share_file):
        df = pd.read_csv(share_file)
        for _, row in df.iterrows():
            n = str(row.iloc[0]).lower()
            cat = str(row.iloc[1]).lower()
            share_onoff_iso[(n, cat)] = float(row["Val"])

    # Load raw capacity data at ISO3 level and aggregate to GCAM regions (sum)
    def _load_cap(fname, has_scenario=True):
        d = {}  # region -> value (summed)
        fpath = os.path.join(stor_dir, fname)
        if not os.path.exists(fpath):
            return d
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            n = str(row.iloc[0]).lower()
            if has_scenario:
                scenario = str(row.iloc[1]).lower()
                if scenario != cap_scenario:
                    continue
            val = float(row["Val"])
            if n in gcam_map:
                r = gcam_nm.get(gcam_map[n])
                if r in region_names:
                    d[r] = d.get(r, 0.0) + val
        return d

    cap_aqui = _load_cap("ccs_stor_cap_aqui.csv")
    cap_og = _load_cap("ccs_stor_cap_og.csv")
    cap_ecbm = _load_cap("ccs_stor_cap_ecbm.csv")
    cap_eor = _load_cap("ccs_stor_cap_eor.csv", has_scenario=False)

    # Aggregate on/off shares to regions (simple average across countries)
    share_onoff = {}
    share_count = {}
    for (iso, cat), val in share_onoff_iso.items():
        if iso in gcam_map:
            r = gcam_nm.get(gcam_map[iso])
            if r in region_names:
                key = (r, cat)
                share_onoff[key] = share_onoff.get(key, 0.0) + val
                share_count[key] = share_count.get(key, 0) + 1
    for key in share_onoff:
        if share_count.get(key, 0) > 0:
            share_onoff[key] /= share_count[key]

    # Compute per-type capacity (GAMS compute_data lines 154-170)
    cap_max = {}  # (region, stype) -> GtCO2
    for r in region_names:
        # Aquifer: split on/off
        aq = cap_aqui.get(r, 0.0)
        cap_max[(r, "aqui_on")] = aq * share_onoff.get((r, "aqui_on"), 0.5)
        cap_max[(r, "aqui_off")] = aq * share_onoff.get((r, "aqui_off"), 0.5)

        # ECBM: on only
        cap_max[(r, "ecbm")] = cap_ecbm.get(r, 0.0)

        # O&G total, split on/off
        og = cap_og.get(r, 0.0)
        og_on = og * share_onoff.get((r, "oil_gas_on"), 0.5)
        og_off = og * share_onoff.get((r, "oil_gas_off"), 0.5)

        # EOR: split on/off
        eor_total = cap_eor.get(r, 0.0)
        eor_on = eor_total * share_onoff.get((r, "oil_gas_on"), 0.5)
        eor_off = eor_total * share_onoff.get((r, "oil_gas_off"), 0.5)
        cap_max[(r, "eor_on")] = eor_on
        cap_max[(r, "eor_off")] = eor_off

        # O&G excluding EOR
        cap_max[(r, "oil_gas_no_eor_on")] = max(og_on - eor_on, 1e-7)
        cap_max[(r, "oil_gas_no_eor_off")] = max(og_off - eor_off, 1e-7)

    result["cap_max"] = cap_max
    return result


def declare_vars(m, sets, params, cfg, v):
    """Create CCS storage variables and storage cost parameter.

    The storage cost parameter is created here (not in define_eqs) so that
    mod_dac.define_eqs() can find it when building eq_cost_cdr.
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # Storage type set
    ccs_stor = Set(m, name="ccs_stor", records=CCS_STOR_TYPES)
    sets["ccs_stor"] = ccs_stor

    # Load storage cost data and create parameter early (before mod_dac.define_eqs)
    from pydice32.data.gcam_mapping import load_rice_regions, load_gcam_mapping, load_gcam_region_names
    region_names = [str(r) for r in n_set.records.iloc[:, 0]]
    try:
        rice_regions = load_rice_regions(cfg.project_root)
        gcam_map = load_gcam_mapping(cfg.gcam_csv, set(rice_regions))
        gcam_nm = load_gcam_region_names(cfg.gcam_names_csv)
    except Exception:
        gcam_map, gcam_nm = {}, {}
    stor_data = _load_storage_data(cfg.data_dir, region_names, cfg.SSP, gcam_map, gcam_nm,
                                       cap_override=getattr(cfg, "ccs_stor_cap_max", None) or None)
    stor_cost = stor_data.get("stor_cost", {})
    cost_recs = [(stype, stor_cost.get(stype, 0.01)) for stype in CCS_STOR_TYPES]
    par_stor_cost = Parameter(m, name="ccs_stor_cost",
                              domain=[ccs_stor], records=cost_recs)
    params["par_ccs_stor_cost"] = par_stor_cost
    v["_ccs_stor_set"] = ccs_stor
    # Cache storage data for define_eqs
    v["_stor_data_cache"] = stor_data

    E_STOR = Variable(m, name="E_STOR",
                      domain=[ccs_stor, t_set, n_set], type="positive")
    CUM_E_STOR = Variable(m, name="CUM_E_STOR",
                          domain=[ccs_stor, t_set, n_set], type="positive")
    E_LEAK = Variable(m, name="E_LEAK",
                      domain=[t_set, n_set], type="positive")

    E_STOR.l[ccs_stor, t_set, n_set] = 0
    CUM_E_STOR.l[ccs_stor, t_set, n_set] = 1e-8
    E_LEAK.l[t_set, n_set] = 0

    # Fix first period
    CUM_E_STOR.fx[ccs_stor, "1", n_set] = 1e-8

    v["E_STOR"] = E_STOR
    v["CUM_E_STOR"] = CUM_E_STOR
    v["E_LEAK"] = E_LEAK


def define_eqs(m, sets, params, cfg, v):
    """Create CCS storage equations."""
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ccs_stor = sets["ccs_stor"]
    TSTEP = cfg.TSTEP
    T = cfg.T

    E_STOR = v["E_STOR"]
    CUM_E_STOR = v["CUM_E_STOR"]
    E_LEAK = v["E_LEAK"]
    E_NEG = v["E_NEG"]

    region_names = [r for r in params["par_pop"].records["n"].unique()]

    # Use cached storage data from declare_vars
    stor_data = v.get("_stor_data_cache")
    if stor_data is None:
        from pydice32.data.gcam_mapping import load_rice_regions, load_gcam_mapping, load_gcam_region_names
        try:
            rice_regions = load_rice_regions(cfg.project_root)
            gcam_map = load_gcam_mapping(cfg.gcam_csv, set(rice_regions))
            gcam_nm = load_gcam_region_names(cfg.gcam_names_csv)
        except Exception:
            gcam_map, gcam_nm = {}, {}
        stor_data = _load_storage_data(cfg.data_dir, region_names, cfg.SSP, gcam_map, gcam_nm,
                                       cap_override=getattr(cfg, "ccs_stor_cap_max", None) or None)
    cap_max = stor_data["cap_max"]
    stor_cost = stor_data["stor_cost"]

    # Leakage rate (GAMS: leak_input = 0 by default)
    leak_rate = getattr(cfg, "leak_input", 0.0)

    # Set CUM_E_STOR upper bounds from capacity
    cap_recs = []
    for r in region_names:
        for stype in CCS_STOR_TYPES:
            cap = cap_max.get((r, stype), 1e-5) / CtoCO2  # GtCO2 -> GtC
            cap = max(cap, 1e-5)
            cap_recs.append((stype, r, cap))
    for stype, r, cap in cap_recs:
        for t in range(2, T + 1):
            CUM_E_STOR.up[stype, str(t), r] = cap

    equations = []

    # eq_emi_stor_dac: E_NEG(t,n) = sum(ccs_stor, E_STOR(ccs_stor,t,n)) * CtoCO2
    eq_emi_stor_dac = Equation(m, name="eq_emi_stor_dac", domain=[t_set, n_set])
    eq_emi_stor_dac[t_set, n_set] = (
        E_NEG[t_set, n_set] == Sum(ccs_stor, E_STOR[ccs_stor, t_set, n_set]) * CtoCO2
    )
    equations.append(eq_emi_stor_dac)

    # eq_stor_cum: CUM_E_STOR(ccs_stor,t+1,n) =
    #   CUM_E_STOR(ccs_stor,t,n) * (1-leak_rate)^tstep + tstep * E_STOR(ccs_stor,t,n)
    eq_stor_cum = Equation(m, name="eq_stor_cum",
                           domain=[ccs_stor, t_set, n_set])
    eq_stor_cum[ccs_stor, t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        CUM_E_STOR[ccs_stor, t_set.lead(1), n_set]
        == CUM_E_STOR[ccs_stor, t_set, n_set] * (1 - leak_rate) ** TSTEP
        + TSTEP * E_STOR[ccs_stor, t_set, n_set]
    )
    equations.append(eq_stor_cum)

    # eq_emi_leak: E_LEAK(t,n) = sum(ccs_stor, (1-(1-leak)^tstep) * CUM_E_STOR) / tstep
    eq_emi_leak = Equation(m, name="eq_emi_leak", domain=[t_set, n_set])
    leak_frac = 1 - (1 - leak_rate) ** TSTEP
    eq_emi_leak[t_set, n_set] = (
        E_LEAK[t_set, n_set] == Sum(
            ccs_stor,
            leak_frac * CUM_E_STOR[ccs_stor, t_set, n_set]
        ) / TSTEP
    )
    equations.append(eq_emi_leak)

    # par_ccs_stor_cost and _ccs_stor_set already set in declare_vars

    return equations
