"""
Exogenous temperature climate module.

Based on GAMS mod_climate_tatm_exogen.gms.
Replaces the endogenous climate system entirely: TATM is fixed to an
external trajectory, and all other climate variables (FORC, TOCEAN,
WCUM_EMI, W_EMI) are created as free variables with no equations.

Usage:
    cfg = Config(climate="tatm_exogen", ...)
    cfg.tatm_exogen_path = {1: 1.1, 2: 1.2, ...}  # t_index -> deg C
    # or {2020: 1.1, 2025: 1.2, ...}  # year -> deg C
"""

from gamspy import Variable, Equation, Sum


def declare_vars(m, sets, params, cfg, v):
    """Create climate variables and fix TATM to exogenous path."""
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]
    T = cfg.T
    TSTEP = cfg.TSTEP

    TATM0_param = params.get("TATM0")
    if TATM0_param is not None and hasattr(TATM0_param, "records") and TATM0_param.records is not None:
        TATM0 = float(TATM0_param.records.iloc[0, -1])
    else:
        TATM0 = 1.1

    # Create climate variables that other modules expect
    TATM = Variable(m, name="TATM", domain=[t_set])
    FORC = Variable(m, name="FORC", domain=[t_set])
    TOCEAN = Variable(m, name="TOCEAN", domain=[t_set])
    WCUM_EMI = Variable(m, name="WCUM_EMI", domain=[t_set])
    W_EMI = Variable(m, name="W_EMI", domain=[ghg_set, t_set])

    # Starting values
    FORC.l[t_set] = 0
    TOCEAN.l[t_set] = 0.007
    WCUM_EMI.l[t_set] = 0
    W_EMI.l[ghg_set, t_set] = 0

    # Fix TATM to exogenous path
    tatm_path = getattr(cfg, "tatm_exogen_path", None)
    for t in range(1, T + 1):
        yr = 2015 + (t - 1) * TSTEP
        if tatm_path is not None and isinstance(tatm_path, dict):
            temp = tatm_path.get(t, tatm_path.get(yr, TATM0))
        else:
            temp = TATM0
        TATM.fx[str(t)] = temp
        TATM.l[str(t)] = temp

    # Register
    v["TATM"] = TATM
    v["FORC"] = FORC
    v["TOCEAN"] = TOCEAN
    v["WCUM_EMI"] = WCUM_EMI
    v["W_EMI"] = W_EMI


def define_eqs(m, sets, params, cfg, v):
    """Define trivial equations for FORC and W_EMI so the model has them.

    TATM is fixed, so no climate dynamics equation is needed.
    FORC and W_EMI get trivial definitions so they appear in the model
    without creating conflicts.
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]

    FORC = v["FORC"]
    W_EMI = v["W_EMI"]
    E = v.get("E")

    equations = []

    # W_EMI = sum(n, E(t,n,ghg)) -- world emissions per GHG
    if E is not None:
        eq_wemi = Equation(m, name="eq_wemi", domain=[ghg_set, t_set])
        eq_wemi[ghg_set, t_set] = (
            W_EMI[ghg_set, t_set] == Sum(n_set, E[t_set, n_set, ghg_set])
        )
        equations.append(eq_wemi)

    # FORC = 0 (placeholder -- no endogenous forcing calculation)
    # Since TATM is externally fixed, FORC is not needed for temperature.
    # Leave it as a free variable without equation (determined by solver slack).

    return equations
