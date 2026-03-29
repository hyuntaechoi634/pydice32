"""
Regional climate downscaling module.

Variables: TEMP_REGION, TEMP_REGION_DAM
Equations: eq_temp_region, eq_temp_region_dam

Variables: TEMP_REGION, TEMP_REGION_DAM, PRECIP_REGION
Equations: eq_temp_region, eq_temp_region_dam, eq_precip_region

When ``cfg.temp_region_cap`` is True, TEMP_REGION_DAM is capped at
``cfg.max_temp_region_dam`` (default 30 deg C) using a smooth NLP
min() approximation, following Burke et al. (2015) conservative approach.

PRECIP_REGION uses the GAMS multiplicative form:
    PRECIP_REGION = (alpha_precip + beta_precip * TATM) * 12 * (1 + DPRECIP_REGION_SAI)
where DPRECIP_REGION_SAI is added when SAI g6 is active.
"""

from gamspy import Variable, Equation, Number
from gamspy.math import sqrt, sqr


def declare_vars(m, sets, params, cfg, v):
    """Create regional temperature variables, set starting values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    TATM0 = params["TATM0"]

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    TEMP_REGION = Variable(m, name="TEMP_REGION", domain=[t_set, n_set])
    TEMP_REGION_DAM = Variable(m, name="TEMP_REGION_DAM", domain=[t_set, n_set])
    # GAMS mod_climate_regional.gms line 88: PRECIP_REGION(t,n)
    PRECIP_REGION = Variable(m, name="PRECIP_REGION", domain=[t_set, n_set])

    # ------------------------------------------------------------------
    # Starting values
    # Ideally these would be alpha_temp + beta_temp * TATM0 (absolute
    # regional temperatures) as GAMS sets in before_solve.  But GAMSPy
    # .l assignments don't support Parameter arithmetic, so we use the
    # alpha_temp parameter alone (dominant term) as the starting hint.
    # The equations will enforce the correct values regardless.
    # ------------------------------------------------------------------
    par_alpha_temp = params["par_alpha_temp"]
    par_alpha_precip = params.get("par_alpha_precip")
    TEMP_REGION.l[t_set, n_set] = par_alpha_temp[n_set]
    TEMP_REGION_DAM.l[t_set, n_set] = par_alpha_temp[n_set]
    if par_alpha_precip is not None:
        PRECIP_REGION.l[t_set, n_set] = par_alpha_precip[n_set]
    else:
        PRECIP_REGION.l[t_set, n_set] = 0

    # Register in shared variable dict
    v["TEMP_REGION"] = TEMP_REGION
    v["TEMP_REGION_DAM"] = TEMP_REGION_DAM
    v["PRECIP_REGION"] = PRECIP_REGION


def define_eqs(m, sets, params, cfg, v):
    """Create regional temperature equations.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    par_alpha_temp = params["par_alpha_temp"]
    par_beta_temp = params["par_beta_temp"]

    # Own variables
    TEMP_REGION = v["TEMP_REGION"]
    TEMP_REGION_DAM = v["TEMP_REGION_DAM"]

    # Cross-module variables
    TATM = v["TATM"]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_temp_region: TEMP_REGION = alpha_temp + beta_temp * TATM
    #   - DTEMP_REGION_SAI  (when SAI g6 is active)
    # GAMS mod_climate_regional.gms line 136-137:
    #   TEMP_REGION(t,n) = alpha + beta * TATM - DTEMP_REGION_SAI(t,n)
    temp_region_rhs = par_alpha_temp[n_set] + par_beta_temp[n_set] * TATM[t_set]
    if "DTEMP_REGION_SAI" in v:
        temp_region_rhs = temp_region_rhs - v["DTEMP_REGION_SAI"][t_set, n_set]

    eq_temp_region = Equation(m, name="eq_temp_region", domain=[t_set, n_set])
    eq_temp_region[t_set, n_set] = (
        TEMP_REGION[t_set, n_set] == temp_region_rhs
    )

    # eq_temp_region_dam: with or without temperature cap
    eq_temp_region_dam = Equation(m, name="eq_temp_region_dam", domain=[t_set, n_set])

    if cfg.temp_region_cap:
        # Smooth NLP approximation for min(TEMP_REGION, max_temp_region_dam):
        #   (f(x) + g(y) - sqrt(sqr(f(x)-g(y)) + sqr(delta))) / 2
        # GAMS mod_climate_regional.gms lines 152-155
        delta_tempcap = 1e-4
        max_t = cfg.max_temp_region_dam
        eq_temp_region_dam[t_set, n_set] = (
            TEMP_REGION_DAM[t_set, n_set] == (
                TEMP_REGION[t_set, n_set] + max_t
                - sqrt(sqr(TEMP_REGION[t_set, n_set] - max_t)
                       + sqr(Number(delta_tempcap)))
            ) / 2
        )
    else:
        # No cap: damages evaluated at effective local temperatures
        eq_temp_region_dam[t_set, n_set] = (
            TEMP_REGION_DAM[t_set, n_set] == TEMP_REGION[t_set, n_set]
        )

    # ------------------------------------------------------------------
    # eq_precip_region: PRECIP_REGION = alpha_precip + beta_precip * TATM
    #   + DPRECIP_REGION_SAI (when SAI g6 is active)
    # GAMS mod_climate_regional.gms lines 145-149
    # ------------------------------------------------------------------
    PRECIP_REGION = v["PRECIP_REGION"]
    par_alpha_precip = params.get("par_alpha_precip")
    par_beta_precip = params.get("par_beta_precip")

    # GAMS mod_climate_regional.gms lines 145-149:
    #   PRECIP_REGION = (alpha_precip + beta_precip * TATM) * 12 * (1 + DPRECIP_REGION_SAI)
    # Note: * 12 converts monthly to annual, DPRECIP_REGION_SAI is fractional
    eq_precip_region = Equation(m, name="eq_precip_region", domain=[t_set, n_set])
    if par_alpha_precip is not None and par_beta_precip is not None:
        precip_base = (par_alpha_precip[n_set] + par_beta_precip[n_set] * TATM[t_set]) * 12
        if "DPRECIP_REGION_SAI" in v:
            precip_rhs = precip_base * (1 + v["DPRECIP_REGION_SAI"][t_set, n_set])
        else:
            precip_rhs = precip_base
        eq_precip_region[t_set, n_set] = (
            PRECIP_REGION[t_set, n_set] == precip_rhs
        )
    else:
        eq_precip_region[t_set, n_set] = PRECIP_REGION[t_set, n_set] == 0

    return [eq_temp_region, eq_temp_region_dam, eq_precip_region]
