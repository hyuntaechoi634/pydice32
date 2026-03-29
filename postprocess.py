"""
Post-solve damage calculator.

Computes climate damages from solved temperature paths without requiring
damage feedback during optimization. Useful for:
- ctax scenarios (impact="off" during solve, damages computed post-hoc)
- cea_tatm / cea_rcp (damages_postprocessed=True)
- Any scenario where you want to compare "what the optimizer saw" vs "actual damages"

Usage:
    from pydice32.postprocess import postprocess_damages

    cfg = Config(policy="ctax", impact="off", ...)
    m, rice, v, data = build_model(cfg)
    solve_model(rice, cfg)
    pp = postprocess_damages(v, data, cfg, impact="kalkuhl")
    # pp["DAMAGES"][t][region] -> damages in T$
    # pp["Y_adjusted"][t][region] -> GDP net of damages and abatement
"""

import math


# ── Damage function coefficients ──────────────────────────────────────

# Kalkuhl & Wenz (2020)
KW_DT = 0.00641
KW_DT_LAG = 0.00345
KW_TDT = -0.00105
KW_TDT_LAG = -0.000718

# DICE-2016R (Nordhaus)
DICE_A2 = 0.00236
DICE_A3 = 2.0

# Howard & Sterner (2017)
HOWARD_A2 = 0.595 * 1.25 / 100  # 0.0074375
HOWARD_A3 = 2.0


def _extract_var(var, domain="tn"):
    """Extract variable levels into a dict."""
    if var is None or var.records is None or len(var.records) == 0:
        return {}
    result = {}
    for _, row in var.records.iterrows():
        if domain == "t":
            result[int(row.iloc[0])] = row["level"]
        elif domain == "tn":
            result[(int(row.iloc[0]), str(row.iloc[1]))] = row["level"]
        elif domain == "tng":
            result[(int(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = row["level"]
    return result


def postprocess_damages(v, data, cfg, impact="kalkuhl"):
    """Compute damages post-solve from the solved temperature path.

    Parameters
    ----------
    v : dict of GAMSPy variables (from build_model)
    data : dict (from build_model)
    cfg : Config
    impact : str
        Damage function to apply: "kalkuhl" | "dice" | "howard" | "coacch"

    Returns
    -------
    dict with keys:
        TEMP_REGION: {(t, region): deg_C}
        OMEGA: {(t, region): fraction}
        DAMFRAC: {(t, region): fraction}
        DAMAGES: {(t, region): T$}
        Y_adjusted: {(t, region): T$}  (YGROSS - DAMAGES - ABATECOST)
        DAMAGES_world: {t: T$}
        Y_adjusted_world: {t: T$}
        YGROSS_world: {t: T$}
    """
    T = cfg.T
    TSTEP = cfg.TSTEP
    region_names = data["region_names"]

    # Extract solved values
    tatm = _extract_var(v["TATM"], "t")
    ygross = _extract_var(v["YGROSS"], "tn")
    abatecost = _extract_var(v["ABATECOST"], "tng")

    # Regional temperature downscaling
    alpha_temp = data.get("alpha_temp_dict", {})
    beta_temp = data.get("beta_temp_dict", {})

    temp_region = {}
    for t in range(1, T + 1):
        tatm_t = tatm.get(t, 1.1)
        for r in region_names:
            a = alpha_temp.get(r, 0.0)
            b = beta_temp.get(r, 1.0)
            temp_region[(t, r)] = a + b * tatm_t

    # Compute damages by impact function
    omega = {(t, r): 0.0 for t in range(1, T + 1) for r in region_names}
    damfrac = {}
    damages = {}

    if impact == "kalkuhl":
        # Growth-rate damage: recursive OMEGA
        bimpact = {}
        for t in range(1, T + 1):
            for r in region_names:
                if t <= 2:
                    bimpact[(t, r)] = 0.0
                else:
                    dT = temp_region[(t, r)] - temp_region[(t - 1, r)]
                    T_lag = temp_region[(t - 1, r)]
                    bi = ((KW_DT + KW_DT_LAG) * dT
                          + (KW_TDT + KW_TDT_LAG) * dT / TSTEP
                          * (2 * dT + 5 * T_lag))
                    bimpact[(t, r)] = bi

        # Recursive omega: OMEGA(t+1) = (1+OMEGA(t)) / (1+BIMPACT(t))^tstep - 1
        for r in region_names:
            omega[(1, r)] = 0.0
            omega[(2, r)] = 0.0
            for t in range(2, T):
                bi = bimpact.get((t, r), 0.0)
                omega[(t + 1, r)] = (1 + omega[(t, r)]) / max((1 + bi) ** TSTEP, 1e-15) - 1

    elif impact in ("dice", "howard"):
        a2 = HOWARD_A2 if impact == "howard" else DICE_A2
        a3 = HOWARD_A3 if impact == "howard" else DICE_A3
        tatm_2 = tatm.get(2, 1.1)
        for t in range(1, T + 1):
            tatm_t = tatm.get(t, 1.1)
            for r in region_names:
                if t > 1:
                    omega[(t, r)] = a2 * tatm_t ** a3 - a2 * tatm_2 ** a3

    elif impact == "coacch":
        # Load COACCH coefficients from data
        # Use comega_agg if available, otherwise approximate
        comega_b1 = {}
        comega_b2 = {}
        temp_base = 0.6  # COACCH default
        # Try to get from calibration data
        ces_ada = data.get("ces_ada_agg", {})
        # For now, use simplified: region-uniform coefficients
        # A proper implementation would load from comega.csv
        for r in region_names:
            comega_b1[r] = 0.01  # approximate COACCH average
            comega_b2[r] = 0.005
        tatm_2 = tatm.get(2, 1.1)
        for t in range(1, T + 1):
            tatm_t = tatm.get(t, 1.1)
            for r in region_names:
                if t > 1:
                    omega[(t, r)] = (
                        comega_b1[r] * (tatm_t - temp_base)
                        + comega_b2[r] * (tatm_t - temp_base) ** 2
                        - comega_b1[r] * (tatm_2 - temp_base)
                        - comega_b2[r] * (tatm_2 - temp_base) ** 2
                    )

    # DAMFRAC and DAMAGES
    for t in range(1, T + 1):
        for r in region_names:
            om = omega[(t, r)]
            df = 1 - 1 / (1 + om) if (1 + om) > 0 else 0.0
            damfrac[(t, r)] = df
            yg = ygross.get((t, r), 0.0)
            damages[(t, r)] = yg * df

    # Abatement cost aggregation (sum over ghg)
    abate_tn = {}
    for (t, r, g), val in abatecost.items():
        key = (t, r)
        abate_tn[key] = abate_tn.get(key, 0.0) + val

    # Adjusted GDP
    y_adjusted = {}
    for t in range(1, T + 1):
        for r in region_names:
            yg = ygross.get((t, r), 0.0)
            dmg = damages.get((t, r), 0.0)
            abate = abate_tn.get((t, r), 0.0)
            y_adjusted[(t, r)] = yg - dmg - abate

    # World aggregates
    damages_world = {}
    y_adj_world = {}
    ygross_world = {}
    for t in range(1, T + 1):
        damages_world[t] = sum(damages.get((t, r), 0.0) for r in region_names)
        y_adj_world[t] = sum(y_adjusted.get((t, r), 0.0) for r in region_names)
        ygross_world[t] = sum(ygross.get((t, r), 0.0) for r in region_names)

    return {
        "TEMP_REGION": temp_region,
        "OMEGA": omega,
        "DAMFRAC": damfrac,
        "DAMAGES": damages,
        "Y_adjusted": y_adjusted,
        "DAMAGES_world": damages_world,
        "Y_adjusted_world": y_adj_world,
        "YGROSS_world": ygross_world,
    }
