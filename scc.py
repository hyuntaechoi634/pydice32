"""
Social Cost of Carbon (SCC) via emission pulse method.

Based on GAMS mod_emission_pulse.gms (Emmerling, 2019).

Method:
  1. Run a baseline scenario (e.g., CBA) and extract results
  2. Re-run the same scenario with a small CO2 pulse at t=2 (year 2020)
  3. Compute SCC as discounted monetized damages from the pulse

SCC is reported in $/tCO2 (2005 USD; multiply by 1.2988 for 2020 USD).

Usage:
    from pydice32.scc import compute_scc
    scc = compute_scc(cfg)
    # scc["ramsey"][year] -> SCC at that year using Ramsey discount
    # scc["2.0"][year]    -> SCC at that year using 2% constant discount

Alternatively:
    python -m pydice32.scc --policy=cba --impact=kalkuhl
"""

import time
import math
from pydice32.config import Config
from pydice32.solver import build_model, solve_model


# CO2 pulse size in MtCO2 (GAMS default: 1 MtCO2)
PULSE_SIZE_MT = 1.0
PULSE_SIZE_GT = PULSE_SIZE_MT * 1e-3  # convert to GtCO2

# Discount rates for IAWG-style SCC
DISCOUNT_RATES = {"1.5": 0.015, "2.0": 0.02, "2.5": 0.025, "3.0": 0.03, "5.0": 0.05}

# USD deflator 2005 -> 2020
DEFLATOR_2005_TO_2020 = 1.298763


def _extract_levels(v, var_name, domain="tn"):
    """Extract variable levels into a dict."""
    var = v.get(var_name)
    if var is None or var.records is None or len(var.records) == 0:
        return {}
    result = {}
    for _, row in var.records.iterrows():
        if domain == "t":
            result[str(row.iloc[0])] = row["level"]
        elif domain == "tn":
            result[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]
        elif domain == "tng":
            result[(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = row["level"]
    return result


def compute_scc(cfg=None, pulse_mt=PULSE_SIZE_MT, verbose=True, **cfg_kwargs):
    """Compute SCC via emission pulse method.

    Parameters
    ----------
    cfg : Config, optional
        Base configuration. If None, created from cfg_kwargs.
    pulse_mt : float
        Pulse size in MtCO2 (default: 1).
    verbose : bool
        Print progress.
    **cfg_kwargs
        Passed to Config() if cfg is None.

    Returns
    -------
    dict with keys:
        "scc_2005usd" : {dr_label: {year: scc_value}}
        "scc_2020usd" : {dr_label: {year: scc_value}}
        "tatm_diff"   : {year: temperature_difference}
        "timing"      : {"baseline_s", "pulse_s", "total_s"}
    """
    if cfg is None:
        cfg = Config(**cfg_kwargs)

    pulse_gt = pulse_mt * 1e-3
    T = cfg.T
    TSTEP = cfg.TSTEP
    ELASMU = cfg.ELASMU

    # ── Step 1: Baseline run ──────────────────────────────────────────
    if verbose:
        print("SCC Step 1: Baseline run...")
    t0 = time.perf_counter()
    m_base, rice_base, v_base, data_base = build_model(cfg)
    solve_model(rice_base, cfg)
    t_baseline = time.perf_counter() - t0
    if verbose:
        print(f"  Baseline: {t_baseline:.1f}s, status={rice_base.solve_status}")

    # Extract baseline results
    C_base = _extract_levels(v_base, "C", "tn")
    UTARG_base = _extract_levels(v_base, "UTARG", "tn")
    TATM_base = _extract_levels(v_base, "TATM", "t")
    MIU_base = _extract_levels(v_base, "MIU", "tng")
    S_base = _extract_levels(v_base, "S", "tn")
    ELAND_base = _extract_levels(v_base, "ELAND", "tn")

    pop_dict = data_base.get("pop_dict", {})
    rr_dict = data_base.get("rr", {})
    region_names = data_base.get("region_names", [])

    # ── Step 2: Pulse run ─────────────────────────────────────────────
    # Reuse the same model: fix MIU and S to baseline levels,
    # add CO2 pulse via par_eland_bau at t=2, then re-solve.
    if verbose:
        print("SCC Step 2: Pulse run (reuse model)...")
    t0 = time.perf_counter()

    # Fix MIU, S, and MIULAND to baseline solved levels
    MIU_v = v_base["MIU"]
    S_v = v_base["S"]
    for key, val in MIU_base.items():
        MIU_v.fx[key[0], key[1], key[2]] = val
    for key, val in S_base.items():
        S_v.fx[key[0], key[1]] = val
    MIULAND_base = _extract_levels(v_base, "MIULAND", "tn")
    MIULAND_v = v_base.get("MIULAND")
    if MIULAND_v is not None:
        for key, val in MIULAND_base.items():
            MIULAND_v.fx[key[0], key[1]] = val

    # Add CO2 pulse at t=2 via par_eland_bau
    par_eland = data_base["_params"].get("par_eland_bau")
    if par_eland is not None and par_eland.records is not None:
        pulse_per_region = pulse_gt / len(region_names)
        new_recs = []
        for _, row in par_eland.records.iterrows():
            t_str = str(row.iloc[0])
            r_str = str(row.iloc[1])
            val = float(row.iloc[2])
            if t_str == "2":
                val += pulse_per_region
            new_recs.append((t_str, r_str, val))
        par_eland.setRecords(new_recs)

    solve_model(rice_base, cfg)
    t_pulse = time.perf_counter() - t0
    if verbose:
        print(f"  Pulse: {t_pulse:.1f}s, status={rice_base.solve_status}")

    # Extract pulse results (from the same v_base, now with pulse solution)
    C_pulse = _extract_levels(v_base, "C", "tn")
    UTARG_pulse = _extract_levels(v_base, "UTARG", "tn")
    TATM_pulse = _extract_levels(v_base, "TATM", "t")

    # ── Step 3: Compute SCC ───────────────────────────────────────────
    if verbose:
        print("SCC Step 3: Computing SCC...")

    # Monetized damages: damrt(t,n) using UTARG-based Rennert et al. (2022) formula
    # damrt(t,n) = pop(t,n) * ((UTARG_base^(1-elasmu) - UTARG_pulse^(1-elasmu))
    #              / (1-elasmu)) / marg_util_cons_ref(t,n)
    # where marg_util_cons_ref = UTARG_base^(-elasmu)
    # Simplifies to: pop * (UTARG_base^(1-elasmu) - UTARG_pulse^(1-elasmu)) / (1-elasmu)
    #                / UTARG_base^(-elasmu)
    #              = pop * UTARG_base * (1 - (UTARG_pulse/UTARG_base)^(1-elasmu)) / (elasmu-1)
    damrt = {}  # (t, n) -> monetized damage in T$
    for t in range(1, T + 1):
        ts = str(t)
        for r in region_names:
            pop = pop_dict.get((t, r), 0.0)
            u_base = UTARG_base.get((ts, r), 1.0)
            u_pulse = UTARG_pulse.get((ts, r), 1.0)
            if u_base > 0 and u_pulse > 0 and abs(ELASMU - 1.0) > 1e-10:
                marg_util = u_base ** (-ELASMU)
                welfare_diff = (u_base ** (1 - ELASMU) - u_pulse ** (1 - ELASMU)) / (1 - ELASMU)
                # pop is in millions, result is M$; convert to T$ (* 1e-6)
                damrt[(t, r)] = pop * welfare_diff / marg_util * 1e-6 if marg_util > 0 else 0.0
            else:
                damrt[(t, r)] = 0.0

    # Temperature difference
    tatm_diff = {}
    for t in range(1, T + 1):
        ts = str(t)
        yr = 2015 + (t - 1) * TSTEP
        tb = TATM_base.get(ts, 0)
        tp = TATM_pulse.get(ts, 0)
        tatm_diff[yr] = tp - tb

    # SCC with constant discount rates (IAWG style)
    scc_2005 = {}
    scc_2020 = {}
    for dr_label, dr in DISCOUNT_RATES.items():
        scc_by_year = {}
        for t_ref in range(1, T + 1):
            yr_ref = 2015 + (t_ref - 1) * TSTEP
            total = 0.0
            for t in range(t_ref, T + 1):
                yr = 2015 + (t - 1) * TSTEP
                for r in region_names:
                    d = damrt.get((t, r), 0.0)
                    total += d * (1 + dr) ** (-(yr - yr_ref))
            # SCC in $/tCO2: total (T$) / pulse (MtCO2) * 1e6 (T$/Mt -> $/t)
            # total is in T$, pulse is in MtCO2
            scc_val = total / pulse_mt * 1e6  # $/tCO2
            scc_by_year[yr_ref] = scc_val
        scc_2005[dr_label] = scc_by_year
        scc_2020[dr_label] = {yr: v * DEFLATOR_2005_TO_2020 for yr, v in scc_by_year.items()}

    # Ramsey SCC (endogenous discount rate)
    ramsey_by_year = {}
    for t_ref in range(1, T + 1):
        yr_ref = 2015 + (t_ref - 1) * TSTEP
        ts_ref = str(t_ref)
        # Global consumption per capita at reference
        c_ref_total = sum(C_base.get((ts_ref, r), 0) for r in region_names)
        pop_ref_total = sum(pop_dict.get((t_ref, r), 0) for r in region_names)
        cpc_ref = c_ref_total / max(pop_ref_total, 1e-10) * 1e6  # $/person

        total = 0.0
        for t in range(t_ref, T + 1):
            yr = 2015 + (t - 1) * TSTEP
            ts = str(t)
            # Global CPC at t
            c_t_total = sum(C_base.get((ts, r), 0) for r in region_names)
            pop_t_total = sum(pop_dict.get((t, r), 0) for r in region_names)
            cpc_t = c_t_total / max(pop_t_total, 1e-10) * 1e6

            # Ramsey SDF
            rr_ref = rr_dict.get(t_ref, 1.0)
            rr_t = rr_dict.get(t, 1.0)
            sdf = (rr_t * cpc_t ** (-ELASMU)) / (rr_ref * cpc_ref ** (-ELASMU)) if rr_ref > 0 else 0

            for r in region_names:
                d = damrt.get((t, r), 0.0)
                total += sdf * d

        ramsey_by_year[yr_ref] = total / pulse_mt * 1e6
    scc_2005["ramsey"] = ramsey_by_year
    scc_2020["ramsey"] = {yr: v * DEFLATOR_2005_TO_2020 for yr, v in ramsey_by_year.items()}

    if verbose:
        print(f"\nSCC at 2020 (2020 USD/tCO2):")
        for dr_label in ["2.0", "3.0", "5.0", "ramsey"]:
            val = scc_2020.get(dr_label, {}).get(2020, 0)
            print(f"  DR={dr_label:>6s}: ${val:,.0f}")
        print(f"\nTATM difference at 2100: {tatm_diff.get(2100, 0)*1000:.4f} mK")

    return {
        "scc_2005usd": scc_2005,
        "scc_2020usd": scc_2020,
        "tatm_diff": tatm_diff,
        "damrt": damrt,
        "timing": {
            "baseline_s": t_baseline,
            "pulse_s": t_pulse,
            "total_s": t_baseline + t_pulse,
        },
    }


if __name__ == "__main__":
    import sys
    kwargs = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            key, _, val = arg[2:].partition("=")
            key = key.replace("-", "_")
            # Try numeric conversion
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except ValueError:
                pass
            kwargs[key] = val

    if "policy" not in kwargs:
        kwargs["policy"] = "cba"
    if "impact" not in kwargs:
        kwargs["impact"] = "kalkuhl"

    result = compute_scc(**kwargs)
