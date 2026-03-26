"""
Hub climate module (WITCH-CO2): global carbon cycle, radiative forcing,
and temperature dynamics.

Variables: W_EMI, WCUM_EMI, RF_CO2, RFoth, FORC, TATM, TOCEAN
Equations: eq_w_emi, eq_wcum, eq_rf_co2, eq_rf_oghg, eq_forc, eq_tatm, eq_tocean
"""

from gamspy import Variable, Equation, Ord, Card, Sum, Number
from gamspy.math import log


def declare_vars(m, sets, params, cfg, v):
    """Create climate variables, set bounds/starting values/fixed values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, layers, layers_alias, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    layers = sets["layers"]

    # Initial conditions (scalars passed via params)
    TATM0 = params["TATM0"]
    TOCEAN0 = params["TOCEAN0"]
    wcum0 = params["wcum0"]          # list [atm, upp, low]

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    # W_EMI stays (t) only for witchco2: only CO2 goes through the carbon cycle
    W_EMI = Variable(m, name="W_EMI", domain=[t_set])
    WCUM_EMI = Variable(m, name="WCUM_EMI", domain=[layers, t_set], type="positive")
    RF_CO2 = Variable(m, name="RF_CO2", domain=[t_set])
    RFoth = Variable(m, name="RFoth", domain=[t_set])
    FORC = Variable(m, name="FORC", domain=[t_set])
    TATM = Variable(m, name="TATM", domain=[t_set])
    TOCEAN = Variable(m, name="TOCEAN", domain=[t_set])

    # ------------------------------------------------------------------
    # Bounds (GAMS mod_climate_witchco2.gms compute_vars)
    # ------------------------------------------------------------------
    TATM.lo[t_set] = -10
    TATM.up[t_set] = 10
    TOCEAN.lo[t_set] = -1
    TOCEAN.up[t_set] = 20
    WCUM_EMI.lo[layers, t_set] = 0.0001
    WCUM_EMI.up[layers, t_set] = 8000

    # GAMS: W_EMI.lo('co2',t) = -200; W_EMI.up(ghg,t) = 200
    W_EMI.lo[t_set] = -200
    W_EMI.up[t_set] = 200

    # GAMS: RF.lo(ghg,t) = -10; RF.up(ghg,t) = 40
    RF_CO2.lo[t_set] = -10
    RF_CO2.up[t_set] = 40

    # ------------------------------------------------------------------
    # Starting values
    # ------------------------------------------------------------------
    TATM.l[t_set] = TATM0
    TOCEAN.l[t_set] = TOCEAN0

    # ------------------------------------------------------------------
    # Fixed initial conditions
    # ------------------------------------------------------------------
    TATM.fx["1"] = TATM0
    TOCEAN.fx["1"] = TOCEAN0
    WCUM_EMI.fx["atm", "1"] = wcum0[0]
    WCUM_EMI.fx["upp", "1"] = wcum0[1]
    WCUM_EMI.fx["low", "1"] = wcum0[2]

    # Register in shared variable dict
    v["W_EMI"] = W_EMI
    v["WCUM_EMI"] = WCUM_EMI
    v["RF_CO2"] = RF_CO2
    v["RFoth"] = RFoth
    v["FORC"] = FORC
    v["TATM"] = TATM
    v["TOCEAN"] = TOCEAN


def define_eqs(m, sets, params, cfg, v):
    """Create climate equations.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, layers, layers_alias, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]
    layers = sets["layers"]
    layers_alias = sets["layers_alias"]

    par_cmphi = params["par_cmphi"]
    par_sigma1 = params["par_sigma1"]
    par_lambda = params["par_lambda"]
    par_sigma2 = params["par_sigma2"]
    par_heat_ocean = params["par_heat_ocean"]
    par_rfc_alpha = params["par_rfc_alpha"]
    par_rfc_beta = params["par_rfc_beta"]
    par_oghg_intercept = params["par_oghg_intercept"]
    par_oghg_slope = params["par_oghg_slope"]

    TSTEP = cfg.TSTEP
    CO2toC = 12.0 / 44.0
    wemi2qemi_co2 = 1.0 / CO2toC

    rfc_beta = params["rfc_beta_scalar"]  # raw float for Number()

    # Own variables
    W_EMI = v["W_EMI"]
    WCUM_EMI = v["WCUM_EMI"]
    RF_CO2 = v["RF_CO2"]
    RFoth = v["RFoth"]
    FORC = v["FORC"]
    TATM = v["TATM"]
    TOCEAN = v["TOCEAN"]

    # Cross-module variables
    E = v["E"]  # now E(t,n,ghg)

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_w_emi: W_EMI = Sum(n, E[t,n,'co2']) / wemi2qemi_co2
    # GAMS: eq_w_emi_co2(t).. W_EMI('co2',t) =E= sum(n$reg(n), E(t,n,'co2')) / wemi2qemi('co2')
    # W_EMI stays (t) only since only CO2 goes through witchco2 carbon cycle.
    # We sum E over n for ghg='co2' only (Ord(ghg_set)==1).
    eq_w_emi = Equation(m, name="eq_w_emi", domain=[t_set])
    eq_w_emi[t_set] = W_EMI[t_set] == (
        Sum(n_set, E[t_set, n_set, "co2"]) / wemi2qemi_co2
    )

    # eq_wcum: WCUM_EMI[layer, t+1] = Sum(mm, cmphi[mm,layer]*WCUM[mm,t]) + TSTEP*W_EMI[t] (atm only)
    eq_wcum = Equation(m, name="eq_wcum", domain=[layers, t_set])
    eq_wcum[layers, t_set].where[Ord(t_set) < Card(t_set)] = (
        WCUM_EMI[layers, t_set.lead(1)] ==
        Sum(layers_alias, par_cmphi[layers_alias, layers] * WCUM_EMI[layers_alias, t_set])
        + (TSTEP * W_EMI[t_set]).where[Ord(layers) == 1]
    )

    # eq_rf_co2: RF_CO2 = rfc_alpha * (log(WCUM[atm,t]) - log(rfc_beta))
    eq_rf_co2 = Equation(m, name="eq_rf_co2", domain=[t_set])
    eq_rf_co2[t_set] = RF_CO2[t_set] == par_rfc_alpha * (
        log(WCUM_EMI["atm", t_set]) - log(Number(rfc_beta))
    )

    # eq_rf_oghg: RFoth = oghg_intercept + oghg_slope * RF_CO2
    eq_rf_oghg = Equation(m, name="eq_rf_oghg", domain=[t_set])
    eq_rf_oghg[t_set] = RFoth[t_set] == par_oghg_intercept + par_oghg_slope * RF_CO2[t_set]

    # eq_forc: FORC = RF_CO2 + RFoth [+ geoeng_forcing * W_SAI when SAI active]
    eq_forc = Equation(m, name="eq_forc", domain=[t_set])
    forc_rhs = RF_CO2[t_set] + RFoth[t_set]
    # Issue 9: SAI forcing offset
    # For g0: _GEOENG_FORCING = -0.2 (global forcing reduction)
    # For g6: _GEOENG_FORCING = 0.0 (temperature effect via regional emulator)
    # The value is stored in v["_GEOENG_FORCING"] by mod_sai.declare_vars,
    # avoiding reliance on module-level mutable global state.
    if "W_SAI" in v:
        geoeng_f = v.get("_GEOENG_FORCING", -0.2)
        if geoeng_f != 0.0:
            forc_rhs = forc_rhs + geoeng_f * v["W_SAI"][t_set]
    eq_forc[t_set] = FORC[t_set] == forc_rhs

    # eq_tatm: TATM[t+1] = TATM[t] + sigma1*(FORC - lambda*TATM - sigma2*(TATM-TOCEAN))
    eq_tatm = Equation(m, name="eq_tatm", domain=[t_set])
    eq_tatm[t_set].where[Ord(t_set) < Card(t_set)] = (
        TATM[t_set.lead(1)] == TATM[t_set] + par_sigma1 * (
            FORC[t_set] - par_lambda * TATM[t_set]
            - par_sigma2 * (TATM[t_set] - TOCEAN[t_set])
        )
    )

    # eq_tocean: TOCEAN[t+1] = TOCEAN[t] + heat_ocean*(TATM-TOCEAN)
    eq_tocean = Equation(m, name="eq_tocean", domain=[t_set])
    eq_tocean[t_set].where[Ord(t_set) < Card(t_set)] = (
        TOCEAN[t_set.lead(1)] == TOCEAN[t_set]
        + par_heat_ocean * (TATM[t_set] - TOCEAN[t_set])
    )

    return [eq_w_emi, eq_wcum, eq_rf_co2, eq_rf_oghg, eq_forc, eq_tatm, eq_tocean]
