"""
Direct Air Capture (DAC) module.

Implements negative emissions via DAC with learning-curve cost dynamics.
Based on GAMS mod_dac.gms from RICE50x.

Variables:
    E_NEG(t,n)    -- negative emissions (installed DAC capacity) [GtCO2/yr]
    I_CDR(t,n)    -- yearly DAC investment [T$/yr]
    COST_CDR(t,n) -- yearly total DAC cost [T$/yr]

Equations:
    eq_depr_e_neg    -- capacity depreciation and investment accumulation
    eq_cost_cdr      -- total cost of CDR (investment + O&M + CCS storage)
    eq_mkt_growth_dac -- market growth constraint on DAC deployment

Integration points:
    - E_NEG enters core_emissions eq_e: E(t,n,'co2') = EIND - E_NEG + ELAND
    - COST_CDR enters core_economy eq_yy: Y = YNET - ABATECOST - ABCOSTLAND - COST_CDR

Parameters (from GAMS):
    dac_tot0      = 0.453  T$/GtCO2  (initial LCOD, from RFF expert elicitation)
    dac_totfloor  = 0.100  T$/GtCO2  (long-term floor cost)
    capex         = 0.4    (fraction of LCOD from investment)
    lifetime      = 20     years
    dac_learn     = 0.136  (learning rate for SSP2 'best' scenario)
    max_cdr       = 40     GtCO2 global max
    mkt_growth_rate = 0.06 (annual capacity growth rate)
    avg_ccs_stor_cost = 0.01414 T$/GtC (average CCS storage cost from
                        data_mod_emi_stor/ccs_stor_cost_estim.csv 'best' values)
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Ord, Card


# --------------- Default DAC parameters (GAMS mod_dac.gms) ---------------

# Initial levelized cost of DAC [T$/GtCO2]
DAC_TOT0 = 453e-3

# Floor levelized cost [T$/GtCO2]
DAC_TOTFLOOR = 100e-3

# Fraction of LCOD attributable to capital investment
CAPEX = 0.4

# DAC plant lifetime [years]
LIFETIME = 20

# Learning rates by SSP cost assumption
LEARN_RATES = {"low": 0.22, "best": 0.136, "high": 0.06}

# Market growth rate by scenario
GROWTH_RATES = {"low": 0.03, "medium": 0.06, "high": 0.10}

# Free market growth per period
MKT_GROWTH_FREE = 0.001 / 5

# Global max CDR [GtCO2/yr]
MAX_CDR = 40

# CCS storage cost [T$/GtC] -- simplified: average of "best" estimates
# from data_mod_emi_stor/ccs_stor_cost_estim.csv.
# GAMS mod_dac.gms line 213: + sum(ccs_stor, E_STOR * ccs_stor_cost) * CtoCO2
# Simplified: all captured CO2 stored at avg cost.
# The conversion works out so the term is E_NEG * avg_ccs_stor_cost:
#   E_STOR (GtC) = E_NEG (GtCO2) / CtoCO2
#   cost = sum(E_STOR * stor_cost) * CtoCO2 = (E_NEG/CtoCO2) * avg_cost * CtoCO2 = E_NEG * avg_cost
AVG_CCS_STOR_COST_DEFAULT = 0.01414  # T$/GtC, fallback if data not found

# C-to-CO2 conversion factor (GAMS: c2co2 = 44/12)
CtoCO2 = 44.0 / 12.0


def _load_avg_ccs_stor_cost(data_dir):
    """Load average CCS storage cost from data_mod_emi_stor/ccs_stor_cost_estim.csv.

    Returns the mean of all 'best' scenario storage costs [T$/GtC].
    Falls back to AVG_CCS_STOR_COST_DEFAULT if file not found.
    """
    fpath = os.path.join(data_dir, "data_mod_emi_stor", "ccs_stor_cost_estim.csv")
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
            best_rows = df[df["Dim2"] == "best"]
            if not best_rows.empty:
                return float(best_rows["Val"].mean())
        except Exception:
            pass
    return AVG_CCS_STOR_COST_DEFAULT


def _dac_delta():
    """Depreciation factor for DAC capacity (constant, as in GAMS)."""
    return 1 - __import__("math").exp(1 / (-LIFETIME + 0.005 * LIFETIME**2))


def declare_vars(m, sets, params, cfg, v):
    """Create DAC variables, set bounds/starting values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config  -- expects cfg.dac (bool) and cfg.dac_cost ('low'|'best'|'high')
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # Variables
    E_NEG = Variable(m, name="E_NEG", domain=[t_set, n_set], type="positive")
    I_CDR = Variable(m, name="I_CDR", domain=[t_set, n_set], type="positive")
    COST_CDR = Variable(m, name="COST_CDR", domain=[t_set, n_set], type="positive")

    # Starting levels
    E_NEG.l[t_set, n_set] = 1e-8
    I_CDR.l[t_set, n_set] = 0
    COST_CDR.l[t_set, n_set] = 0

    # Bounds -- GAMS: E_NEG.lo = 1e-15, stability
    E_NEG.lo[t_set, n_set] = 1e-15

    # GAMS: COST_CDR.up = 0.25 * ykali for stability
    par_ykali = params["par_ykali"]
    COST_CDR.up[t_set, n_set] = 0.25 * par_ykali[t_set, n_set]

    # Fix first period
    I_CDR.fx["1", n_set] = 0
    E_NEG.fx["1", n_set] = 1e-8

    # Register
    v["E_NEG"] = E_NEG
    v["I_CDR"] = I_CDR
    v["COST_CDR"] = COST_CDR


def define_eqs(m, sets, params, cfg, v):
    """Create DAC equations.

    Parameters
    ----------
    m : Container
    sets : dict
    params : dict
    cfg : Config
    v : dict of all variables

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    TSTEP = cfg.TSTEP

    # Retrieve DAC config
    dac_cost_scenario = getattr(cfg, "dac_cost", "best")
    dac_learn = LEARN_RATES.get(dac_cost_scenario, LEARN_RATES["best"])
    dac_growth_scenario = getattr(cfg, "dac_growth", "medium")
    mkt_growth_rate = GROWTH_RATES.get(dac_growth_scenario, GROWTH_RATES["medium"])

    # Use initial cost (learning is iterated externally in before_solve in GAMS;
    # here we use the initial cost as a constant approximation for the NLP).
    dac_totcost = DAC_TOT0

    # CCS storage cost: load from data or use default
    # GAMS mod_dac.gms line 213: + sum(ccs_stor, E_STOR * ccs_stor_cost) * CtoCO2
    avg_ccs_stor_cost = _load_avg_ccs_stor_cost(cfg.data_dir)

    delta_en = _dac_delta()

    # Own variables
    E_NEG = v["E_NEG"]
    I_CDR = v["I_CDR"]
    COST_CDR = v["COST_CDR"]

    equations = []

    # eq_depr_e_neg: capacity accumulation / depreciation
    # E_NEG(t+1) = E_NEG(t) * (1 - delta)^tstep + tstep * I_CDR(t) / (capex * lifetime * dac_totcost)
    eq_depr_e_neg = Equation(m, name="eq_depr_e_neg", domain=[t_set, n_set])
    inv_factor = 1.0 / (CAPEX * LIFETIME * dac_totcost)
    eq_depr_e_neg[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        E_NEG[t_set.lead(1), n_set]
        == E_NEG[t_set, n_set] * (1 - delta_en) ** TSTEP
        + TSTEP * I_CDR[t_set, n_set] * inv_factor
    )
    equations.append(eq_depr_e_neg)

    # eq_cost_cdr: total cost = investment + O&M + CCS storage
    # GAMS mod_dac.gms line 211-213:
    #   COST_CDR(t,n) = I_CDR(t,n) + E_NEG(t,n) * dac_totcost * (1-capex)
    #                 + sum(ccs_stor, E_STOR(ccs_stor,t,n) * ccs_stor_cost(ccs_stor,n)) * CtoCO2
    # Simplified: all captured CO2 stored at average cost.
    # E_STOR total = E_NEG / CtoCO2 (GtC), so storage cost = E_NEG * avg_ccs_stor_cost.
    eq_cost_cdr = Equation(m, name="eq_cost_cdr", domain=[t_set, n_set])
    eq_cost_cdr[t_set, n_set] = (
        COST_CDR[t_set, n_set]
        == I_CDR[t_set, n_set]
        + E_NEG[t_set, n_set] * dac_totcost * (1 - CAPEX)
        + E_NEG[t_set, n_set] * avg_ccs_stor_cost
    )
    equations.append(eq_cost_cdr)

    # eq_mkt_growth_dac: growth constraint
    # I_CDR(t+1) / factor <= I_CDR(t) / factor * (1+growth)^tstep + tstep * free_growth
    eq_mkt_growth_dac = Equation(m, name="eq_mkt_growth_dac", domain=[t_set, n_set])
    eq_mkt_growth_dac[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        I_CDR[t_set.lead(1), n_set] * inv_factor
        <= I_CDR[t_set, n_set] * inv_factor * (1 + mkt_growth_rate) ** TSTEP
        + TSTEP * MKT_GROWTH_FREE
    )
    equations.append(eq_mkt_growth_dac)

    return equations
