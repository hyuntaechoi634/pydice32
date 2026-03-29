"""
Adaptation module: proactive, reactive, and specific adaptive capacity.

Based on GAMS mod_adaptation.gms from RICE50x (Bosello and De Cian, 2014).

Adaptation is modelled through a CES function combining:
  - Proactive adaptation (prada): stock that accumulates via investment
  - Reactive adaptation (rada):   flow investment
  - Specific adaptive capacity (scap): stock that accumulates
  - Generic adaptive capacity (gcap): exogenous, grows with TFP

Variables:
    K_ADA(g,t,n)  -- adaptation capital by sector [T$ 2005]
    I_ADA(g,t,n)  -- adaptation investment by sector [T$ 2005/yr]
    Q_ADA(iq,t,n) -- adaptation output (ada, cap, act, gcap) [T$ 2005/yr]

Equations:
    eqq_ada   -- CES: total adaptation = f(actions, capacity)
    eqq_act   -- CES: actions = f(rada, prada stock)
    eqq_cap   -- CES: capacity = f(gcap, scap stock)
    eqq_gcap  -- generic capacity = exogenous (TFP-scaled)
    eqk_prada -- proactive stock accumulation
    eqk_scap  -- specific stock accumulation

Integration point:
    Q_ADA modifies OMEGA in hub_impact (adaptation reduces damages).

GAMS adaptation sectors:
    g = {prada, rada, scap}     (investment sectors)
    iq = {ada, cap, act, gcap}  (output nodes in the CES tree)
"""

from gamspy import Variable, Equation, Set, Ord, Card


# --------------- Default adaptation parameters ---------------

# Depreciation rates by sector
DK_ADA = {"prada": 0.1, "rada": 1.0, "scap": 0.03}

# Default CES parameters -- used as fallback when CSV data from
# data_mod_damage.gdx (ces_ada.csv, owa.csv) is not loaded.
# Issue 13: when adaptation data IS loaded, these are overridden
# by per-region values from params["par_ces_ada_*"] / params["par_owa_*"].
DEFAULT_CES_ADA = {"ada": -0.11111, "act": 0.17, "cap": -4.0, "tfpada": 1.0, "eff": 1.0, "exp": 0.6}
DEFAULT_OWA = {
    "act": 0.5, "cap": 0.5,
    "rada": 0.5, "prada": 0.5,
    "gcap": 0.5, "scap": 0.5,
    "actc": 1.0,
}


def declare_vars(m, sets, params, cfg, v):
    """Create adaptation variables.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config  -- expects cfg.adaptation (bool)
    v : dict of all variables (mutated)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # Create adaptation-specific sets
    g_set = Set(m, name="g_ada", records=["prada", "rada", "scap"])
    iq_set = Set(m, name="iq_ada", records=["ada", "cap", "act", "gcap"])
    sets["g_ada"] = g_set
    sets["iq_ada"] = iq_set

    # Variables
    K_ADA = Variable(m, name="K_ADA", domain=[g_set, t_set, n_set])
    I_ADA = Variable(m, name="I_ADA", domain=[g_set, t_set, n_set])
    Q_ADA = Variable(m, name="Q_ADA", domain=[iq_set, t_set, n_set])

    # Bounds (GAMS)
    Q_ADA.lo[iq_set, t_set, n_set] = 1e-8
    Q_ADA.up[iq_set, t_set, n_set] = 1e3
    Q_ADA.l[iq_set, t_set, n_set] = 1e-5
    K_ADA.lo[g_set, t_set, n_set] = 1e-8
    K_ADA.up[g_set, t_set, n_set] = 1e3
    K_ADA.l[g_set, t_set, n_set] = 1e-8
    I_ADA.lo[g_set, t_set, n_set] = 1e-8
    I_ADA.up[g_set, t_set, n_set] = 1e3
    I_ADA.l[g_set, t_set, n_set] = 1e-8

    # Fix first period capital
    K_ADA.fx["prada", "1", n_set] = 1e-5
    K_ADA.fx["rada", "1", n_set] = 1e-5
    K_ADA.fx["scap", "1", n_set] = 1e-5

    # Register
    v["K_ADA"] = K_ADA
    v["I_ADA"] = I_ADA
    v["Q_ADA"] = Q_ADA


def define_eqs(m, sets, params, cfg, v):
    """Create adaptation equations.

    The CES tree structure is:
        ada = tfpada * CES(act, cap; rho_ada)
        act = eff * actc * CES(rada, prada_stock; rho_act)
        cap = CES(gcap, scap_stock; rho_cap)
        gcap = ((k_h0 + k_edu0)/2) * tfp

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

    K_ADA = v["K_ADA"]
    I_ADA = v["I_ADA"]
    Q_ADA = v["Q_ADA"]

    # Retrieve TFP parameter for gcap equation
    par_tfp = params["par_tfp"]

    equations = []

    # ------------------------------------------------------------------
    # Issue 13: use data-loaded CES/OWA parameters when available,
    # otherwise fall back to module defaults.
    # Data-loaded params are registered in solver.py as par_ces_ada_*
    # and par_owa_* (per-region Parameters).  For scalar CES usage we
    # average across regions at load time -- the GAMS model uses
    # per-region values but these are constant across regions in the
    # current dataset.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # eqq_ada: Q_ADA('ada',t,n) = tfpada * (owa_act * Q_ADA('act')^rho_ada
    #                              + owa_cap * Q_ADA('cap')^rho_ada)^(1/rho_ada)
    # ------------------------------------------------------------------
    # Use data-loaded parameters if available, else defaults
    par_ces_tfpada = params.get("par_ces_ada_tfpada")
    par_ces_ada_rho = params.get("par_ces_ada_ada")
    par_ces_ada_exp = params.get("par_ces_ada_exp")
    par_owa_act = params.get("par_owa_act")
    par_owa_cap = params.get("par_owa_cap")

    tfpada = DEFAULT_CES_ADA["tfpada"]
    rho_ada = DEFAULT_CES_ADA["ada"]
    owa_act = DEFAULT_OWA["act"]
    owa_cap = DEFAULT_OWA["cap"]

    # When data-loaded per-region parameters are available, use them;
    # otherwise use scalar defaults.  Per-region params are [n_set] domain.
    if par_ces_tfpada is not None:
        tfpada_expr = par_ces_tfpada[n_set]
    else:
        tfpada_expr = tfpada
    if par_ces_ada_rho is not None:
        rho_ada_expr = par_ces_ada_rho[n_set]
    else:
        rho_ada_expr = rho_ada
    if par_owa_act is not None:
        owa_act_expr = par_owa_act[n_set]
    else:
        owa_act_expr = owa_act
    if par_owa_cap is not None:
        owa_cap_expr = par_owa_cap[n_set]
    else:
        owa_cap_expr = owa_cap

    eqq_ada = Equation(m, name="eqq_ada", domain=[t_set, n_set])
    eqq_ada[t_set, n_set] = (
        Q_ADA["ada", t_set, n_set]
        == tfpada_expr
        * (
            owa_act_expr * Q_ADA["act", t_set, n_set] ** rho_ada_expr
            + owa_cap_expr * Q_ADA["cap", t_set, n_set] ** rho_ada_expr
        )
        ** (1.0 / rho_ada_expr)
    )
    equations.append(eqq_ada)

    # ------------------------------------------------------------------
    # eqq_act: Q_ADA('act',t,n) = eff * actc * (owa_rada * I_ADA('rada')^rho_act
    #                              + owa_prada * K_ADA('prada')^rho_act)^(1/rho_act)
    # ------------------------------------------------------------------
    # GAMS: ces_ada('eff',n) varies by SSP: ssp2=1.0, ssp1/ssp5=1.25, ssp3=0.75
    # Use SSP-dependent efficiency when available via config, otherwise default.
    _ssp_eff = {"ssp1": 1.25, "ssp2": 1.0, "ssp3": 0.75, "ssp4": 1.0, "ssp5": 1.25}
    eff = _ssp_eff.get(getattr(cfg, "SSP", "ssp2"), DEFAULT_CES_ADA["eff"])
    actc = DEFAULT_OWA["actc"]
    rho_act = DEFAULT_CES_ADA["act"]
    owa_rada = DEFAULT_OWA["rada"]
    owa_prada = DEFAULT_OWA["prada"]

    par_ces_ada_act = params.get("par_ces_ada_act")
    par_owa_rada = params.get("par_owa_rada")
    par_owa_prada = params.get("par_owa_prada")
    par_owa_actc = params.get("par_owa_actc")

    if par_ces_ada_act is not None:
        rho_act_expr = par_ces_ada_act[n_set]
    else:
        rho_act_expr = rho_act
    if par_owa_rada is not None:
        owa_rada_expr = par_owa_rada[n_set]
    else:
        owa_rada_expr = owa_rada
    if par_owa_prada is not None:
        owa_prada_expr = par_owa_prada[n_set]
    else:
        owa_prada_expr = owa_prada
    if par_owa_actc is not None:
        actc_expr = par_owa_actc[n_set]
    else:
        actc_expr = actc

    eqq_act = Equation(m, name="eqq_act", domain=[t_set, n_set])
    eqq_act[t_set, n_set] = (
        Q_ADA["act", t_set, n_set]
        == eff
        * actc_expr
        * (
            owa_rada_expr * I_ADA["rada", t_set, n_set] ** rho_act_expr
            + owa_prada_expr * K_ADA["prada", t_set, n_set] ** rho_act_expr
        )
        ** (1.0 / rho_act_expr)
    )
    equations.append(eqq_act)

    # ------------------------------------------------------------------
    # eqq_cap: Q_ADA('cap',t,n) = (owa_gcap * Q_ADA('gcap')^rho_cap
    #                              + owa_scap * K_ADA('scap')^rho_cap)^(1/rho_cap)
    # ------------------------------------------------------------------
    rho_cap = DEFAULT_CES_ADA["cap"]
    owa_gcap = DEFAULT_OWA["gcap"]
    owa_scap = DEFAULT_OWA["scap"]

    par_ces_ada_cap = params.get("par_ces_ada_cap")
    par_owa_gcap = params.get("par_owa_gcap")
    par_owa_scap = params.get("par_owa_scap")

    if par_ces_ada_cap is not None:
        rho_cap_expr = par_ces_ada_cap[n_set]
    else:
        rho_cap_expr = rho_cap
    if par_owa_gcap is not None:
        owa_gcap_expr = par_owa_gcap[n_set]
    else:
        owa_gcap_expr = owa_gcap
    if par_owa_scap is not None:
        owa_scap_expr = par_owa_scap[n_set]
    else:
        owa_scap_expr = owa_scap

    eqq_cap = Equation(m, name="eqq_cap", domain=[t_set, n_set])
    eqq_cap[t_set, n_set] = (
        Q_ADA["cap", t_set, n_set]
        == (
            owa_gcap_expr * Q_ADA["gcap", t_set, n_set] ** rho_cap_expr
            + owa_scap_expr * K_ADA["scap", t_set, n_set] ** rho_cap_expr
        )
        ** (1.0 / rho_cap_expr)
    )
    equations.append(eqq_cap)

    # ------------------------------------------------------------------
    # eqq_gcap: Q_ADA('gcap',t,n) = ((k_h0+k_edu0)/2) * tfp(t,n)
    # Issue 13: use data-loaded k_h0/k_edu0 when available
    # ------------------------------------------------------------------
    par_gcap_scale = params.get("par_gcap_scale")
    if par_gcap_scale is not None:
        gcap_scale_expr = par_gcap_scale[n_set]
    else:
        gcap_scale_expr = 1e-3  # placeholder for (k_h0+k_edu0)/2

    eqq_gcap = Equation(m, name="eqq_gcap", domain=[t_set, n_set])
    eqq_gcap[t_set, n_set] = (
        Q_ADA["gcap", t_set, n_set] == gcap_scale_expr * par_tfp[t_set, n_set]
    )
    equations.append(eqq_gcap)

    # ------------------------------------------------------------------
    # eqk_prada: K_ADA('prada',t+1) = (1-dk_prada)^tstep * K_ADA('prada',t) + tstep * I_ADA('prada',t)
    # ------------------------------------------------------------------
    dk_prada = DK_ADA["prada"]
    eqk_prada = Equation(m, name="eqk_prada", domain=[t_set, n_set])
    eqk_prada[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        K_ADA["prada", t_set.lead(1), n_set]
        == (1 - dk_prada) ** TSTEP * K_ADA["prada", t_set, n_set]
        + TSTEP * I_ADA["prada", t_set, n_set]
    )
    equations.append(eqk_prada)

    # ------------------------------------------------------------------
    # eqk_scap: K_ADA('scap',t+1) = (1-dk_scap)^tstep * K_ADA('scap',t) + tstep * I_ADA('scap',t)
    # ------------------------------------------------------------------
    dk_scap = DK_ADA["scap"]
    eqk_scap = Equation(m, name="eqk_scap", domain=[t_set, n_set])
    eqk_scap[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        K_ADA["scap", t_set.lead(1), n_set]
        == (1 - dk_scap) ** TSTEP * K_ADA["scap", t_set, n_set]
        + TSTEP * I_ADA["scap", t_set, n_set]
    )
    equations.append(eqk_scap)

    return equations
