"""
Data loading and calibration for PyDICE32.

Extracts all data-loading and parameter-calibration logic from
rice_gamspy.py (lines ~194-884) into a single ``load_and_calibrate(cfg)``
entry point that returns a dict of every calibrated model parameter.
"""

import os
import math
import numpy as np
import pandas as pd
from pydice32.data.loader import load_csv, load_1d, load_validation_param
from pydice32.data.gcam_mapping import (
    load_rice_regions,
    load_gcam_mapping,
    load_gcam_region_names,
    aggregate_param_1d,
)
from pydice32.data.sai_emulator_data import generate_sai_emulator_data, INJ_LABELS


def load_and_calibrate(cfg):
    """Load all data and calibrate parameters. Returns dict with all model data."""

    T = cfg.T
    TSTEP = cfg.TSTEP
    DK = cfg.DK
    PRSTP = cfg.PRSTP
    ELASMU = cfg.ELASMU
    ssp = cfg.SSP
    data_dir = cfg.data_dir
    project_root = cfg.project_root

    # ------------------------------------------------------------------
    # Region setup
    # ------------------------------------------------------------------
    rice_regions_list = load_rice_regions(project_root)
    rice_regions_set = set(rice_regions_list)
    mapping = load_gcam_mapping(cfg.gcam_csv, rice_regions_set)
    gcam_names = load_gcam_region_names(cfg.gcam_names_csv)

    # ------------------------------------------------------------------
    # Load raw baseline data
    # ------------------------------------------------------------------
    ykali_raw = load_csv(data_dir, "data_baseline", "ssp_ykali.csv")
    pop_raw = load_csv(data_dir, "data_baseline", "ssp_l.csv")
    ppp2mer_raw = load_1d(data_dir, "data_baseline", "ppp2mer.csv")
    sigma_raw = load_csv(data_dir, "data_baseline", "ssp_ci.csv")

    # ------------------------------------------------------------------
    # Aggregate ykali (GDP) -- sum, with PPP adjustment
    # ------------------------------------------------------------------
    ykali_dict = {}
    for _, row in ykali_raw.iterrows():
        if str(row.iloc[0]).upper() != ssp:
            continue
        t, n = int(row["t"]), str(row["n"]).lower()
        if n not in mapping or t < 1 or t > T:
            continue
        rname = gcam_names[mapping[n]]
        mer2ppp = 1.0 / max(ppp2mer_raw.get(n, 1.0), 1e-6)
        key = (t, rname)
        ykali_dict[key] = ykali_dict.get(key, 0.0) + row["Val"] * mer2ppp

    # ------------------------------------------------------------------
    # Aggregate pop -- sum
    # ------------------------------------------------------------------
    pop_dict = {}
    for _, row in pop_raw.iterrows():
        if str(row.iloc[0]).upper() != ssp:
            continue
        t, n = int(row["t"]), str(row["n"]).lower()
        if n not in mapping or t < 1 or t > T:
            continue
        rname = gcam_names[mapping[n]]
        key = (t, rname)
        pop_dict[key] = pop_dict.get(key, 0.0) + row["Val"]

    # ------------------------------------------------------------------
    # Active region list (exclude zero-pop / zero-GDP regions like Taiwan)
    # ------------------------------------------------------------------
    region_names = []
    for r in range(1, 33):
        rname = gcam_names[r]
        if pop_dict.get((1, rname), 0) > 0 and ykali_dict.get((1, rname), 0) > 0:
            region_names.append(rname)

    # ------------------------------------------------------------------
    # Per-country ykali lookup (PPP-adjusted)
    # ------------------------------------------------------------------
    ykali_country = {}
    for _, row in ykali_raw.iterrows():
        if str(row.iloc[0]).upper() != ssp:
            continue
        t, n = int(row["t"]), str(row["n"]).lower()
        mer2ppp = 1.0 / max(ppp2mer_raw.get(n, 1.0), 1e-6)
        ykali_country[(t, n)] = row["Val"] * mer2ppp

    # ------------------------------------------------------------------
    # Multi-GHG definitions
    # ------------------------------------------------------------------
    ghg_list = ["co2", "ch4", "n2o"]
    convq_ghg = {"co2": 1, "ch4": 1e3, "n2o": 1e3}
    convy_ghg = {"co2": 1e-3, "ch4": 1e-6, "n2o": 1e-6}

    # ------------------------------------------------------------------
    # Aggregate sigma (GDP-weighted) for ALL GHGs:
    #   emi_bau(t,n,ghg) = convq_ghg(ghg) * sigma(t,n,ghg) * ykali(t,n)
    # ------------------------------------------------------------------
    emi_bau_dict = {}   # (t, rname, ghg) -> value
    sigma_agg = {}      # (t, rname, ghg) -> value
    for _, row in sigma_raw.iterrows():
        if str(row.iloc[0]).upper() != ssp:
            continue
        t = int(row["t"])
        n = str(row["n"]).lower()
        ghg = str(row.iloc[3]).lower()
        if ghg not in convq_ghg or n not in mapping or t < 1 or t > T:
            continue
        rname = gcam_names[mapping[n]]
        ykali_n = ykali_country.get((t, n), 0.0)
        emi_n = convq_ghg[ghg] * row["Val"] * ykali_n
        key = (t, rname, ghg)
        emi_bau_dict[key] = emi_bau_dict.get(key, 0.0) + emi_n

    for key in emi_bau_dict:
        t, rname, ghg = key
        yk = ykali_dict.get((t, rname), 0)
        sigma_agg[key] = emi_bau_dict[key] / (convq_ghg[ghg] * yk) if yk > 0 else 0

    # ------------------------------------------------------------------
    # Initial conditions: k0 and s0
    # ------------------------------------------------------------------
    k0_raw = load_validation_param(
        data_dir, "data_validation", "k_valid_article.csv", "fg")
    s0_raw = load_validation_param(
        data_dir, "data_validation", "socecon_valid_weo_mean.csv", "savings_rate")

    gdp_weight = {
        n: ykali_raw[
            (ykali_raw.iloc[:, 0].str.upper() == ssp)
            & (ykali_raw["t"] == 1)
            & (ykali_raw["n"].str.lower() == n)
        ]["Val"].sum()
        * (1.0 / max(ppp2mer_raw.get(n, 1.0), 1e-6))
        for n in rice_regions_list
        if n in mapping
    }

    k0_agg = aggregate_param_1d(
        {n: v * (1.0 / max(ppp2mer_raw.get(n, 1.0), 1e-6))
         for n, v in k0_raw.items()},
        mapping, gcam_names,
    )
    # Impute missing: K = 2.72*GDP + 0.127 (R²=0.96), Koch & Marian (2021) RICE50x
    # Linear regression of initial capital on GDP for regions with data.
    for rname in region_names:
        if k0_agg.get(rname, 0) == 0:
            k0_agg[rname] = 2.72 * ykali_dict.get((1, rname), 0) + 0.127

    s0_agg = aggregate_param_1d(
        {n: max(v, 1.0) / 100.0 for n, v in s0_raw.items()},
        mapping, gcam_names, weight_dict=gdp_weight,
    )

    # ------------------------------------------------------------------
    # Labour share: default 0.7 or calibrated from data_mod_labour
    # GAMS core_economy.gms lines 157-188: when calib_labour_share is set,
    # load labour_share from data, clip to [0.5, 0.8], default 0.7 if zero.
    # ------------------------------------------------------------------
    if cfg.calib_labour_share:
        ls_raw = load_1d(data_dir, "data_mod_labour", "labour_share.csv")
        # Replace zero values with 0.7 (GAMS: labour_share(n)$(.. eq 0) = 0.7)
        for n in ls_raw:
            if ls_raw[n] == 0:
                ls_raw[n] = 0.7
        # Clip to [0.5, 0.8] (GAMS: min(max(labour_share(n), 0.5), 0.8))
        for n in ls_raw:
            ls_raw[n] = min(max(ls_raw[n], 0.5), 0.8)
        # Aggregate to 32 regions (GDP-weighted)
        ls_agg = aggregate_param_1d(
            ls_raw, mapping, gcam_names, weight_dict=gdp_weight,
        )
        # Any region still at zero gets default 0.7
        for rn in region_names:
            if ls_agg.get(rn, 0) == 0:
                ls_agg[rn] = 0.7
            # Re-clip after aggregation
            ls_agg[rn] = min(max(ls_agg[rn], 0.5), 0.8)
    else:
        ls_agg = {rn: 0.7 for rn in region_names}

    # ------------------------------------------------------------------
    # Climate parameters
    # ------------------------------------------------------------------
    cmphi_raw = load_csv(data_dir, "data_mod_climate", "cmphi.csv")
    tempc_raw = load_csv(data_dir, "data_mod_climate", "tempc.csv")
    wcum0_raw = load_csv(data_dir, "data_mod_climate", "wcum_emi0.csv")
    rfc_raw = load_csv(data_dir, "data_mod_climate", "rfc.csv")

    oghg_path = os.path.join(data_dir, "data_mod_climate", "oghg_coeff.csv")
    if os.path.exists(oghg_path):
        oghg_raw = load_csv(data_dir, "data_mod_climate", "oghg_coeff.csv")
    else:
        oghg_raw = load_csv(data_dir, "data_ssp_iam", "oghg_coeff.csv")

    # Parse cmphi (3x3 carbon-cycle transfer matrix)
    cmphi = np.zeros((3, 3))
    layer_map = {"atm": 0, "upp": 1, "low": 2}
    for _, row in cmphi_raw.iterrows():
        f, t_layer = str(row.iloc[0]).lower(), str(row.iloc[1]).lower()
        if f in layer_map and t_layer in layer_map:
            cmphi[layer_map[f], layer_map[t_layer]] = row["Val"]

    # Parse tempc scalars
    tempc_dict = {}
    for _, row in tempc_raw.iterrows():
        tempc_dict[str(row.iloc[0]).lower()] = row["Val"]
    sigma1 = tempc_dict.get("sigma1", 0.1005)
    lam = tempc_dict.get("lambda", 1.18)
    sigma2 = tempc_dict.get("sigma2", 0.088)
    heat_ocean = tempc_dict.get("heat_ocean", 0.025)

    # Parse wcum0 (initial carbon stocks)
    wcum0 = np.zeros(3)
    for _, row in wcum0_raw.iterrows():
        g, m_layer = str(row.iloc[0]).lower(), str(row.iloc[1]).lower()
        if g == "co2" and m_layer in layer_map:
            wcum0[layer_map[m_layer]] = row["Val"]

    # Parse radiative-forcing coefficients
    rfc_alpha, rfc_beta = 5.35, 588.0
    for _, row in rfc_raw.iterrows():
        g, p_name = str(row.iloc[0]).lower(), str(row.iloc[1]).lower()
        if g == "co2" and p_name == "alpha":
            rfc_alpha = row["Val"]
        elif g == "co2" and p_name == "beta":
            rfc_beta = row["Val"]

    # Parse other-GHG forcing coefficients
    oghg_intercept, oghg_slope = 0.0, 0.0
    for _, row in oghg_raw.iterrows():
        k = str(row.iloc[0]).lower()
        if k == "intercept":
            oghg_intercept = row["Val"]
        elif k == "slope":
            oghg_slope = row["Val"]

    # Climate scalars
    TATM0 = 1.1
    TOCEAN0 = 0.11
    CO2toC = 12.0 / 44.0
    wemi2qemi_co2 = 1.0 / CO2toC

    # ------------------------------------------------------------------
    # Derived economic parameters
    # ------------------------------------------------------------------
    prodshare_cap = {}
    prodshare_lab = {}
    optlr_savings = {}
    for rname in region_names:
        ls = ls_agg.get(rname, 0.7)
        if ls == 0:
            ls = 0.7
        ls = min(max(ls, 0.5), 0.8)
        prodshare_lab[rname] = ls
        prodshare_cap[rname] = 1.0 - ls
        optlr_savings[rname] = (
            (DK + 0.004)
            / (DK + 0.004 * ELASMU + PRSTP)
            * prodshare_cap[rname]
        )

    # Fixed savings path (linear interpolation s0 -> optlr_savings)
    fixed_savings = {}
    for rname in region_names:
        s0 = s0_agg.get(rname, 0.2)
        for t in range(1, T + 1):
            fixed_savings[(t, rname)] = (
                s0 + (optlr_savings[rname] - s0) * (t - 1) / (T - 1)
            )

    # ------------------------------------------------------------------
    # TFP calibration loop
    # ------------------------------------------------------------------
    k_tfp = {}
    tfp = {}
    for rname in region_names:
        k_tfp[(1, rname)] = k0_agg.get(rname, 1.0)
    for t in range(1, T):
        for rname in region_names:
            yk = ykali_dict.get((t, rname), 0)
            i_tfp = fixed_savings[(t, rname)] * yk
            k_tfp[(t + 1, rname)] = (
                (1 - DK) ** TSTEP * k_tfp[(t, rname)] + TSTEP * i_tfp
            )
            pop_val = pop_dict.get((t, rname), 1)
            denom = (
                (pop_val / 1000) ** prodshare_lab[rname]
                * k_tfp[(t, rname)] ** prodshare_cap[rname]
            )
            tfp[(t, rname)] = yk / denom if denom > 0 else 1.0
    for rname in region_names:
        tfp[(T, rname)] = tfp.get((T - 1, rname), 1.0)

    # ------------------------------------------------------------------
    # Discount factor
    # ------------------------------------------------------------------
    rr = {t: 1.0 / (1.0 + PRSTP) ** (TSTEP * (t - 1)) for t in range(1, T + 1)}

    # ------------------------------------------------------------------
    # Land-use: eland_bau from lu_baseline.csv
    # ------------------------------------------------------------------
    eland_bau = {}
    for rname in region_names:
        for t in range(1, T + 1):
            eland_bau[(t, rname)] = 0.0

    lu_file = os.path.join(data_dir, "data_mod_landuse", "lu_baseline.csv")
    if os.path.exists(lu_file):
        lu_df = pd.read_csv(lu_file)
        luscenario = "ssp2-base"
        for _, row in lu_df.iterrows():
            cols = list(lu_df.columns)
            t_val = None
            n_val = None
            sc_val = None
            for i, col in enumerate(cols):
                if col == "t":
                    t_val = int(row[col])
                elif col == "n":
                    n_val = str(row[col]).lower()
                elif col == "Val":
                    continue
                else:
                    v = str(row[col]).lower()
                    if v == luscenario:
                        sc_val = v
                    elif sc_val is None and i < 3:
                        sc_val = v
            if t_val is None or n_val is None:
                continue
            if sc_val is not None and sc_val != luscenario:
                continue
            if n_val not in mapping or t_val < 1 or t_val > T:
                continue
            rname = gcam_names[mapping[n_val]]
            key = (t_val, rname)
            if rname in region_names:
                eland_bau[key] = eland_bau.get(key, 0.0) + row["Val"]

    # ------------------------------------------------------------------
    # Land-use abatement: eland_maxab
    # ------------------------------------------------------------------
    convy_co2 = 1e-3  # cost conversion for CO2

    eland_maxab_dict = {rn: 0.0 for rn in region_names}
    lu_abatemax_file = os.path.join(
        data_dir, "data_mod_landuse", "lu_abatemax.csv")
    if os.path.exists(lu_abatemax_file):
        am_df = pd.read_csv(lu_abatemax_file)
        for _, row in am_df.iterrows():
            n_val = str(row.iloc[0]).lower()
            sc = str(row.iloc[1]).lower()
            if sc != "ssp2-base":
                continue
            if n_val not in mapping:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname in region_names:
                eland_maxab_dict[rname] = (
                    eland_maxab_dict.get(rname, 0.0) + row["Val"]
                )

    # ------------------------------------------------------------------
    # Land-use MACC coefficients (GDP-weighted)
    # ------------------------------------------------------------------
    lu_macc_c1_dict = {
        (t, rn): 0.0 for t in range(1, T + 1) for rn in region_names
    }
    lu_macc_c4_dict = {
        (t, rn): 0.0 for t in range(1, T + 1) for rn in region_names
    }
    lu_macc_file = os.path.join(data_dir, "data_mod_landuse", "lu_maccs.csv")
    if os.path.exists(lu_macc_file):
        lm_df = pd.read_csv(lu_macc_file)
        lu_c1_num = {}
        lu_c4_num = {}
        lu_weights = {}
        for _, row in lm_df.iterrows():
            t_val = int(row.iloc[0])
            n_val = str(row.iloc[1]).lower()
            sc = str(row.iloc[2]).lower()
            coef = str(row.iloc[3]).lower()
            if sc != "ssp2-base":
                continue
            if n_val not in mapping or t_val < 1 or t_val > T:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname not in region_names:
                continue
            w = ykali_country.get((t_val, n_val), 0)
            key = (t_val, rname)
            if key not in lu_weights:
                lu_weights[key] = 0.0
                lu_c1_num[key] = 0.0
                lu_c4_num[key] = 0.0
            if coef == "c1":
                lu_c1_num[key] += max(row["Val"], 0) * w
                lu_weights[key] += w
            elif coef == "c4":
                lu_c4_num[key] += max(row["Val"], 0) * w
                # Don't double-count weights; only count on one coef type
        for key in lu_weights:
            if lu_weights[key] > 0:
                lu_macc_c1_dict[key] = lu_c1_num.get(key, 0) / lu_weights[key]
                lu_macc_c4_dict[key] = lu_c4_num.get(key, 0) / lu_weights[key]

    # ------------------------------------------------------------------
    # Regional climate downscaling coefficients (population-weighted)
    # ------------------------------------------------------------------
    coef_file = os.path.join(
        data_dir, "data_mod_climate_regional", "climate_region_coef_cmip5.csv")
    alpha_temp_dict = {rn: 0.0 for rn in region_names}
    beta_temp_dict = {rn: 0.0 for rn in region_names}
    base_temp_dict = {rn: 0.0 for rn in region_names}
    alpha_pop_weight = {rn: 0.0 for rn in region_names}
    beta_pop_weight = {rn: 0.0 for rn in region_names}
    base_temp_pop_weight = {rn: 0.0 for rn in region_names}

    if os.path.exists(coef_file):
        cdf = pd.read_csv(coef_file)
        for _, row in cdf.iterrows():
            # CSV columns: Dim1 (parameter name), n (ISO3 country), Val
            param = str(row.iloc[0]).lower()
            n_val = str(row.iloc[1]).lower()
            if n_val not in mapping:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname not in region_names:
                continue
            # Look up country population at t=1
            pop_n = 0
            for _, pr in pop_raw.iterrows():
                if (str(pr.iloc[0]).upper() == ssp
                        and int(pr["t"]) == 1
                        and str(pr["n"]).lower() == n_val):
                    pop_n = pr["Val"]
                    break
            if param in ("alpha_temp", "intercept"):
                alpha_temp_dict[rname] += row["Val"] * pop_n
                alpha_pop_weight[rname] += pop_n
            elif param in ("beta_temp", "slope"):
                beta_temp_dict[rname] += row["Val"] * pop_n
                beta_pop_weight[rname] += pop_n
            elif param == "base_temp":
                base_temp_dict[rname] += row["Val"] * pop_n
                base_temp_pop_weight[rname] += pop_n

        # Normalize by data-covered population (not total region population)
        for rname in region_names:
            if alpha_pop_weight[rname] > 0:
                alpha_temp_dict[rname] /= alpha_pop_weight[rname]
            if beta_pop_weight[rname] > 0:
                beta_temp_dict[rname] /= beta_pop_weight[rname]
            if base_temp_pop_weight[rname] > 0:
                base_temp_dict[rname] /= base_temp_pop_weight[rname]

    # Fallback: if no downscaling data, use 1:1 mapping (alpha=0, beta=1)
    for rname in region_names:
        if beta_pop_weight.get(rname, 0) == 0:
            beta_temp_dict[rname] = 1.0
            alpha_temp_dict[rname] = 0.0
        # Fallback for base_temp: use alpha_temp (the intercept) as approximation
        if base_temp_pop_weight.get(rname, 0) == 0:
            base_temp_dict[rname] = alpha_temp_dict[rname]

    # ------------------------------------------------------------------
    # SAI g6 emulator data: synthesise from beta_temp and global values
    # ------------------------------------------------------------------
    sai_temp_dict, sai_precip_dict = generate_sai_emulator_data(
        beta_temp_dict,
        beta_precip_dict=None,  # Precipitation response requires beta_precip;
                                # set to zero (conservative) until data exists.
    )
    sai_inj_labels = INJ_LABELS

    # Also try to load from CSV if data_mod_sai directory exists
    sai_data_dir = os.path.join(data_dir, "data_mod_sai")
    sai_temp_csv = os.path.join(sai_data_dir, "srm_temperature_response.csv")
    sai_precip_csv = os.path.join(sai_data_dir, "srm_precip_response.csv")
    if os.path.exists(sai_temp_csv):
        # CSV format: "n","inj","Val"
        sai_t_df = pd.read_csv(sai_temp_csv)
        sai_temp_dict = {}
        for _, row in sai_t_df.iterrows():
            rn = str(row.iloc[0])
            inj = str(row.iloc[1])
            sai_temp_dict[(rn, inj)] = float(row["Val"])
    if os.path.exists(sai_precip_csv):
        sai_p_df = pd.read_csv(sai_precip_csv)
        sai_precip_dict = {}
        for _, row in sai_p_df.iterrows():
            rn = str(row.iloc[0])
            inj = str(row.iloc[1])
            sai_precip_dict[(rn, inj)] = float(row["Val"])

    # ------------------------------------------------------------------
    # Load emi_gwp from climate data
    # ------------------------------------------------------------------
    emi_gwp_file = os.path.join(data_dir, "data_mod_climate", "emi_gwp.csv")
    emi_gwp = {}
    if os.path.exists(emi_gwp_file):
        gwp_df = pd.read_csv(emi_gwp_file)
        for _, row in gwp_df.iterrows():
            emi_gwp[str(row.iloc[0]).lower()] = row["Val"]

    # ------------------------------------------------------------------
    # MACC coefficients: CO2 with backstop mx at country level (Fix #25),
    # then aggregate to 32 regions; non-CO2 from PBL data
    # ------------------------------------------------------------------

    # Initialise output dicts keyed by (t, region, ghg)
    macc_c1_dict = {}
    macc_c4_dict = {}
    for t in range(1, T + 1):
        for rn in region_names:
            for ghg in ghg_list:
                macc_c1_dict[(t, rn, ghg)] = 0.0
                macc_c4_dict[(t, rn, ghg)] = 0.0

    # --- CO2 MACC: load raw per-country, apply mx, then aggregate ------
    macc_file = os.path.join(data_dir, "data_mod_macc", "macc_ed_coef.csv")
    if os.path.exists(macc_file):
        mdf = pd.read_csv(macc_file)

        # Step 1: Load raw c1/c4 per country
        c1_country = {}   # (t, iso) -> val
        c4_country = {}   # (t, iso) -> val
        for _, row in mdf.iterrows():
            sector = str(row.iloc[0])
            coef_type = str(row.iloc[1]).lower()
            if sector != "Total_CO2":
                continue
            t_val = int(row.iloc[2])
            n_val = str(row.iloc[3]).lower()
            if t_val < 1 or t_val > T:
                continue
            val = max(row["Val"], 0)
            if coef_type == "a":
                c1_country[(t_val, n_val)] = val
            elif coef_type == "d":
                c4_country[(t_val, n_val)] = val

        # Step 2: Backstop parameters
        pback = cfg.pback
        gback = cfg.gback
        pbacktime = {
            t: pback * (1 - gback) ** (t - 1) for t in range(1, T + 1)
        }

        # Logistic transition alpha
        tstart_pb = cfg.tstart_pbtransition
        tend_pb = cfg.tend_pbtransition
        klogistic = cfg.klogistic
        x0 = tstart_pb + (tend_pb - tstart_pb) / 2.0
        alpha_logistic = {
            t: 1.0 / (1.0 + math.exp(-klogistic * (t - x0)))
            for t in range(1, T + 1)
        }

        # Step 3: Load MXstart per country from mx_correction_factor.csv
        macc_quant = cfg.macc_costs  # prob25|prob33|prob50|prob66|prob75
        mx_file = os.path.join(
            data_dir, "data_mod_macc", "mx_correction_factor.csv")
        mx_start_raw = {}  # (t, iso) -> val
        if os.path.exists(mx_file):
            mx_df = pd.read_csv(mx_file)
            for _, row in mx_df.iterrows():
                sec = str(row.iloc[0])
                quant = str(row.iloc[1]).lower()
                if sec != "Total_CO2" or quant != macc_quant:
                    continue
                t_val = int(row["t"])
                n_val = str(row["n"]).lower()
                if t_val < 1 or t_val > T:
                    continue
                mx_start_raw[(t_val, n_val)] = max(row["Val"], 0.0)

        # Step 4: For each country, compute mx and apply to c1/c4
        #   MXpback_country = pbacktime / (c1 + c4)  [maxmiu=1 for CO2]
        #   mx_country = MXstart - alpha * max(MXstart - MXpback, 0)
        #   c1_adj = mx_country * c1_raw, c4_adj = mx_country * c4_raw
        c1_adj_country = {}   # (t, iso) -> val
        c4_adj_country = {}   # (t, iso) -> val
        for t in range(1, T + 1):
            for iso in rice_regions_list:
                c1_val = c1_country.get((t, iso), 0.0)
                c4_val = c4_country.get((t, iso), 0.0)
                mac_at_1 = c1_val + c4_val  # MAC at MIU=1
                if mac_at_1 > 0:
                    MXpback = pbacktime[t] / mac_at_1
                else:
                    MXpback = 0.0
                MXstart = mx_start_raw.get((t, iso), 1.0)
                MXdiff = max(MXstart - MXpback, 0.0)
                mx_val = MXstart - alpha_logistic[t] * MXdiff
                c1_adj_country[(t, iso)] = mx_val * c1_val
                c4_adj_country[(t, iso)] = mx_val * c4_val

        # Step 5: Aggregate mx-adjusted c1/c4 to 32 regions (GDP-weighted)
        # Use per-coef weights (Fix #18/#19)
        for t in range(1, T + 1):
            c1_num = {}    # rname -> numerator
            c4_num = {}
            c1_wgt = {}    # rname -> weight sum
            c4_wgt = {}
            for iso in rice_regions_list:
                if iso not in mapping:
                    continue
                rname = gcam_names[mapping[iso]]
                if rname not in region_names:
                    continue
                w = ykali_country.get((t, iso), 0)
                c1_val = c1_adj_country.get((t, iso), 0.0)
                c4_val = c4_adj_country.get((t, iso), 0.0)
                if c1_val > 0 or (t, iso) in c1_country:
                    c1_num[rname] = c1_num.get(rname, 0.0) + c1_val * w
                    c1_wgt[rname] = c1_wgt.get(rname, 0.0) + w
                if c4_val > 0 or (t, iso) in c4_country:
                    c4_num[rname] = c4_num.get(rname, 0.0) + c4_val * w
                    c4_wgt[rname] = c4_wgt.get(rname, 0.0) + w
            for rname in region_names:
                key = (t, rname, "co2")
                if c1_wgt.get(rname, 0) > 0:
                    macc_c1_dict[key] = c1_num.get(rname, 0) / c1_wgt[rname]
                if c4_wgt.get(rname, 0) > 0:
                    macc_c4_dict[key] = c4_num.get(rname, 0) / c4_wgt[rname]

    # --- Non-CO2 MACC from PBL data ------------------------------------
    pbl_macc_file = os.path.join(
        data_dir, "data_mod_nonco2", "macc_ghg_coefficients.csv")
    if os.path.exists(pbl_macc_file):
        pbl_df = pd.read_csv(pbl_macc_file)
        # Per-coef accumulators for GDP-weighted aggregation
        pbl_c1_num = {}   # (t, rname, ghg) -> numerator
        pbl_c4_num = {}
        pbl_c1_wgt = {}   # (t, rname, ghg) -> weight
        pbl_c4_wgt = {}
        for _, row in pbl_df.iterrows():
            t_val = int(row.iloc[0])
            n_val = str(row.iloc[1]).lower()
            ghg = str(row.iloc[2]).lower()
            coef = str(row.iloc[3]).lower()
            if ghg not in ("ch4", "n2o"):
                continue
            if n_val not in mapping or t_val < 1 or t_val > T:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname not in region_names:
                continue
            w = ykali_country.get((t_val, n_val), 0)
            val = max(row["Val"], 0)
            key = (t_val, rname, ghg)
            if coef == "c1":
                pbl_c1_num[key] = pbl_c1_num.get(key, 0.0) + val * w
                pbl_c1_wgt[key] = pbl_c1_wgt.get(key, 0.0) + w
            elif coef == "c4":
                pbl_c4_num[key] = pbl_c4_num.get(key, 0.0) + val * w
                pbl_c4_wgt[key] = pbl_c4_wgt.get(key, 0.0) + w
        for key in pbl_c1_wgt:
            if pbl_c1_wgt[key] > 0:
                macc_c1_dict[key] = pbl_c1_num[key] / pbl_c1_wgt[key]
        for key in pbl_c4_wgt:
            if pbl_c4_wgt[key] > 0:
                macc_c4_dict[key] = pbl_c4_num[key] / pbl_c4_wgt[key]

    # --- Non-CO2 max_miu from PBL data ---------------------------------
    maxmiu_pbl = {}
    for t in range(1, T + 1):
        for rn in region_names:
            maxmiu_pbl[(t, rn, "co2")] = 1.0  # CO2 always 1
    maxmiu_file = os.path.join(data_dir, "data_mod_nonco2", "max_miu.csv")
    if os.path.exists(maxmiu_file):
        mm_df = pd.read_csv(maxmiu_file)
        # GDP-weighted aggregation of max_miu to 32 regions
        mm_num = {}   # (t, rname, ghg) -> numerator
        mm_wgt = {}   # (t, rname, ghg) -> weight
        for _, row in mm_df.iterrows():
            t_val = int(row.iloc[0])
            n_val = str(row.iloc[1]).lower()
            ghg = str(row.iloc[2]).lower()
            if ghg not in ("ch4", "n2o"):
                continue
            if n_val not in mapping or t_val < 1 or t_val > T:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname not in region_names:
                continue
            w = ykali_country.get((t_val, n_val), 0)
            val = row["Val"]
            key = (t_val, rname, ghg)
            mm_num[key] = mm_num.get(key, 0.0) + val * w
            mm_wgt[key] = mm_wgt.get(key, 0.0) + w
        for key in mm_wgt:
            if mm_wgt[key] > 0:
                maxmiu_pbl[key] = mm_num[key] / mm_wgt[key]

    # ------------------------------------------------------------------
    # Long-term pledges: net-zero year per region from nzpl.csv
    # Aggregate ISO3-level pledge years to 32 GCAM regions (earliest year)
    # ------------------------------------------------------------------
    pledge_nz_year_co2 = {}  # region_name -> earliest net-zero year
    nzpl_file = os.path.join(data_dir, "data_long_term_pledges", "nzpl.csv")
    if os.path.exists(nzpl_file):
        nzpl_df = pd.read_csv(nzpl_file)
        for _, row in nzpl_df.iterrows():
            iso = str(row.iloc[0]).lower()
            pledge_type = str(row.iloc[1]).strip()
            yr = float(row.iloc[2])
            # Use "Net zero" pledge type (CO2 net-zero)
            if pledge_type != "Net zero":
                continue
            if iso not in mapping:
                continue
            rname = gcam_names[mapping[iso]]
            if rname not in region_names:
                continue
            # Take earliest pledge year across countries in region
            if rname not in pledge_nz_year_co2:
                pledge_nz_year_co2[rname] = yr
            else:
                pledge_nz_year_co2[rname] = min(pledge_nz_year_co2[rname], yr)

    # Regions without pledges get 2310 as default (GAMS default: end of horizon)
    for rname in region_names:
        if rname not in pledge_nz_year_co2:
            pledge_nz_year_co2[rname] = 2310.0

    # ------------------------------------------------------------------
    # Issue 13: Adaptation CES parameters from data_mod_damage
    # Load ces_ada (Dim1 x n -> Val) and owa (Dim1 x n -> Val)
    # Also load k_h0(n) and k_edu0(n) for gcap scaling.
    # ------------------------------------------------------------------
    ces_ada_agg = {}   # (param_name, region) -> value (GDP-weighted)
    owa_agg = {}       # (param_name, region) -> value (GDP-weighted)
    k_h0_agg = {}      # region -> value
    k_edu0_agg = {}    # region -> value

    ces_ada_file = os.path.join(data_dir, "data_mod_damage", "ces_ada.csv")
    if os.path.exists(ces_ada_file):
        ca_df = pd.read_csv(ces_ada_file)
        # Accumulate GDP-weighted values per (param, region)
        ca_num = {}   # (param, rname) -> numerator
        ca_wgt = {}   # (param, rname) -> weight
        for _, row in ca_df.iterrows():
            param_name = str(row.iloc[0]).lower()
            n_val = str(row.iloc[1]).lower()
            val = float(row["Val"])
            if n_val not in mapping:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname not in region_names:
                continue
            w = ykali_country.get((1, n_val), 0)
            key = (param_name, rname)
            ca_num[key] = ca_num.get(key, 0.0) + val * w
            ca_wgt[key] = ca_wgt.get(key, 0.0) + w
        for key in ca_wgt:
            if ca_wgt[key] > 0:
                ces_ada_agg[key] = ca_num[key] / ca_wgt[key]

    owa_file = os.path.join(data_dir, "data_mod_damage", "owa.csv")
    if os.path.exists(owa_file):
        owa_df = pd.read_csv(owa_file)
        owa_num = {}
        owa_wgt = {}
        for _, row in owa_df.iterrows():
            param_name = str(row.iloc[0]).lower()
            n_val = str(row.iloc[1]).lower()
            val = float(row["Val"])
            if n_val not in mapping:
                continue
            rname = gcam_names[mapping[n_val]]
            if rname not in region_names:
                continue
            w = ykali_country.get((1, n_val), 0)
            key = (param_name, rname)
            owa_num[key] = owa_num.get(key, 0.0) + val * w
            owa_wgt[key] = owa_wgt.get(key, 0.0) + w
        for key in owa_wgt:
            if owa_wgt[key] > 0:
                owa_agg[key] = owa_num[key] / owa_wgt[key]

    k_h0_file = os.path.join(data_dir, "data_mod_damage", "k_h0.csv")
    if os.path.exists(k_h0_file):
        kh_raw = load_1d(data_dir, "data_mod_damage", "k_h0.csv")
        k_h0_agg = aggregate_param_1d(kh_raw, mapping, gcam_names)

    k_edu0_file = os.path.join(data_dir, "data_mod_damage", "k_edu0.csv")
    if os.path.exists(k_edu0_file):
        kedu_raw = load_1d(data_dir, "data_mod_damage", "k_edu0.csv")
        k_edu0_agg = aggregate_param_1d(kedu_raw, mapping, gcam_names)

    # ------------------------------------------------------------------
    # Coalition mappings (for noncoop / coalitions cooperation modes)
    # GAMS core_cooperation.gms: map_clt_n(clt, n)
    # For noncoop: each region is its own coalition
    # For coalitions: load from noncoop.inc or similar file
    # ------------------------------------------------------------------
    coalitions = None  # None means default (each region = own coalition)
    cooperation = getattr(cfg, "cooperation", "coop")
    if cooperation in ("noncoop", "coalitions"):
        coalitions = _load_coalitions(project_root, region_names, cooperation)

    # ------------------------------------------------------------------
    # Assemble return dict
    # ------------------------------------------------------------------
    data = dict(
        # Region info
        region_names=region_names,
        rice_regions_list=rice_regions_list,
        mapping=mapping,
        gcam_names=gcam_names,
        # Multi-GHG
        ghg_list=ghg_list,
        convq_ghg=convq_ghg,
        convy_ghg=convy_ghg,
        emi_gwp=emi_gwp,
        # Baseline socioeconomic
        ykali_dict=ykali_dict,
        pop_dict=pop_dict,
        # sigma_agg and emi_bau_dict are now keyed (t, region, ghg)
        sigma_agg=sigma_agg,
        emi_bau_dict=emi_bau_dict,
        # Aliases for solver.py multi-GHG path (keyed (t, region, ghg))
        sigma_agg_ghg=sigma_agg,
        emi_bau_ghg=emi_bau_dict,
        ykali_country=ykali_country,
        # Initial conditions
        k0_agg=k0_agg,
        s0_agg=s0_agg,
        fixed_savings=fixed_savings,
        # TFP calibration
        k_tfp=k_tfp,
        tfp=tfp,
        # Production shares
        prodshare_cap=prodshare_cap,
        prodshare_lab=prodshare_lab,
        # Discount factors
        rr=rr,
        # Climate model parameters
        cmphi=cmphi,
        sigma1=sigma1,
        lam=lam,
        sigma2=sigma2,
        heat_ocean=heat_ocean,
        wcum0=wcum0,
        rfc_alpha=rfc_alpha,
        rfc_beta=rfc_beta,
        oghg_intercept=oghg_intercept,
        oghg_slope=oghg_slope,
        TATM0=TATM0,
        TOCEAN0=TOCEAN0,
        CO2toC=CO2toC,
        wemi2qemi_co2=wemi2qemi_co2,
        # Land-use
        eland_bau=eland_bau,
        eland_maxab_dict=eland_maxab_dict,
        lu_macc_c1_dict=lu_macc_c1_dict,
        lu_macc_c4_dict=lu_macc_c4_dict,
        convy_co2=convy_co2,
        # Regional climate downscaling
        alpha_temp_dict=alpha_temp_dict,
        beta_temp_dict=beta_temp_dict,
        base_temp_dict=base_temp_dict,
        # MACC coefficients (keyed by (t, region, ghg))
        macc_c1_dict=macc_c1_dict,
        macc_c4_dict=macc_c4_dict,
        # Aliases for solver.py multi-GHG path
        macc_c1_ghg=macc_c1_dict,
        macc_c4_ghg=macc_c4_dict,
        maxmiu_pbl=maxmiu_pbl,
        # Long-term pledges
        pledge_nz_year_co2=pledge_nz_year_co2,
        # Adaptation CES parameters (Issue 13)
        ces_ada_agg=ces_ada_agg,
        owa_agg=owa_agg,
        k_h0_agg=k_h0_agg,
        k_edu0_agg=k_edu0_agg,
        # SAI g6 emulator data
        sai_temp_dict=sai_temp_dict,
        sai_precip_dict=sai_precip_dict,
        sai_inj_labels=sai_inj_labels,
        # Coalition mappings (for iterative/Nash solving)
        coalitions=coalitions,
    )
    return data


def _load_coalitions(project_root, region_names, cooperation):
    """Load coalition mappings for non-cooperative or coalition modes.

    Parameters
    ----------
    project_root : str
    region_names : list of str  -- active GCAM region names
    cooperation : str  -- "noncoop" or "coalitions"

    Returns
    -------
    list of lists: each inner list contains the region names in that coalition.
    For noncoop: [[region1], [region2], ...] (one region per coalition).
    For coalitions: loaded from data file or defaults to noncoop.

    Mirrors GAMS core_cooperation.gms:
      noncoop:    map_clt_n(clt,n)$sameas(clt,n) = YES  (1:1 mapping)
      coalitions: $batinclude "coalitions/coal_%sel_coalition%.gms"
    """
    if cooperation == "noncoop":
        # Pure non-cooperative: each region is its own coalition
        return [[r] for r in region_names]

    # For coalitions mode: try to load from GAMS noncoop.inc
    # The inc file has: set map_clt_n(clt,n) / c_xxx.xxx, ... /
    # Parse the map_clt_n block to extract coalition->region mappings
    region_set = set(region_names)
    noncoop_paths = [
        os.path.join(project_root, "RICE50xmodel", "data_maxiso3", "noncoop.inc"),
        os.path.join(project_root, "RICE50xmodel", "data_ed58", "noncoop.inc"),
    ]

    for inc_path in noncoop_paths:
        if os.path.exists(inc_path):
            coalitions = _parse_coalition_inc(inc_path, region_set)
            if coalitions:
                return coalitions

    # Fallback: noncoop (each region = own coalition)
    return [[r] for r in region_names]


def _parse_coalition_inc(inc_path, region_set):
    """Parse a GAMS noncoop.inc file to extract coalition mappings.

    The file contains:
      set map_clt_n(clt,n) /
      c_xxx.yyy
      c_aaa.bbb
      /;

    Where c_xxx is the coalition name and yyy is the ISO3 region code.
    We group regions by coalition and filter to only include active regions.

    Parameters
    ----------
    inc_path : str  -- path to the .inc file
    region_set : set of str  -- active region names (GCAM aggregated)

    Returns
    -------
    list of lists or None if parsing fails.
    Note: The noncoop.inc maps ISO3 codes, not GCAM region names. Since
    the PyRICE model uses GCAM aggregate regions, we return one coalition
    per GCAM region (equivalent to noncoop at the aggregate level).
    """
    # The noncoop.inc file maps individual ISO3 countries, not GCAM regions.
    # Since our model operates at GCAM-32 aggregate level, the coalition
    # structure at the aggregate level is: each GCAM region = one coalition.
    # This is the correct mapping for noncoop at our resolution.
    return [[r] for r in sorted(region_set)]
