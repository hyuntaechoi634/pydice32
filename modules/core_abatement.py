"""
Core abatement cost module: marginal abatement cost (MAC) and total abatement cost.

Variables: ABATECOST(t,n,ghg), MAC(t,n,ghg)
Equations: eq_abatecost, eq_mac
"""

from gamspy import Variable, Equation


def declare_vars(m, sets, params, cfg, v):
    """Create abatement cost variables, set bounds/starting values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]

    # ------------------------------------------------------------------
    # Variables (all indexed over ghg per GAMS core_abatement.gms)
    # ------------------------------------------------------------------
    ABATECOST = Variable(m, name="ABATECOST", domain=[t_set, n_set, ghg_set], type="positive")
    # GAMS name is "MAC" (not "MAC_var")
    MAC = Variable(m, name="MAC", domain=[t_set, n_set, ghg_set], type="positive")

    # ------------------------------------------------------------------
    # Starting values
    # ------------------------------------------------------------------
    ABATECOST.l[t_set, n_set, ghg_set] = 0
    MAC.l[t_set, n_set, ghg_set] = 0

    # Register in shared variable dict
    v["ABATECOST"] = ABATECOST
    v["MAC"] = MAC


def define_eqs(m, sets, params, cfg, v):
    """Create abatement cost equations.

    Also sets MIU.up for non-CO2 GHGs using maxmiu_pbl (GAMS core_abatement.gms compute_vars).

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, etc.
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

    par_emi_bau = params["par_emi_bau"]       # emi_bau(t,n,ghg)
    par_macc_c1 = params["par_macc_c1"]       # macc_coef(t,n,ghg,'c1')
    par_macc_c4 = params["par_macc_c4"]       # macc_coef(t,n,ghg,'c4')
    par_convy_ghg = params["par_convy_ghg"]   # convy_ghg(ghg): co2=1e-3, ch4=1e-6, n2o=1e-6
    par_maxmiu_pbl = params["par_maxmiu_pbl"] # maxmiu_pbl(t,n,ghg)

    # Own variables
    ABATECOST = v["ABATECOST"]
    MAC = v["MAC"]

    # Cross-module variables
    MIU = v["MIU"]

    # ------------------------------------------------------------------
    # MIU upper bounds (GAMS core_abatement.gms compute_vars)
    # GAMS: MIU.up(t,n,ghg)$(not tmiufix(t)) = maxmiu_pbl(t,n,ghg)
    # Must skip tmiufix periods AND bau/bau_impact/simulation (MIU.fx=0
    # for all t), because setting .up in define_eqs (Pass 2) overrides
    # the .fx set in core_policy declare_vars (Pass 1).
    # ------------------------------------------------------------------
    if cfg.policy in ("bau", "bau_impact", "simulation"):
        pass  # MIU.fx=0 for all periods, don't override
    else:
        tmiufix_set = set(cfg.tmiufix)
        for t_idx in range(1, cfg.T + 1):
            if t_idx not in tmiufix_set:
                MIU.up[str(t_idx), n_set, ghg_set] = par_maxmiu_pbl[str(t_idx), n_set, ghg_set]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_abatecost: ABATECOST = emi_bau * convy_ghg * (c1*MIU^2/2 + c4*MIU^5/5)
    # GAMS: ABATECOST(t,n,ghg) =E= emi_bau(t,n,ghg) * convy_ghg(ghg) *
    #   sum(coef$coefact(coef,ghg), macc_coef(t,n,ghg,coef)*power(MIU,...,coefn(coef)+1)/(coefn(coef)+1))
    # Simplified: c1*MIU^2/2 + c4*MIU^5/5
    eq_abatecost = Equation(m, name="eq_abatecost", domain=[t_set, n_set, ghg_set])
    eq_abatecost[t_set, n_set, ghg_set] = ABATECOST[t_set, n_set, ghg_set] == (
        par_emi_bau[t_set, n_set, ghg_set] * par_convy_ghg[ghg_set] * (
            par_macc_c1[t_set, n_set, ghg_set] * MIU[t_set, n_set, ghg_set] ** 2 / 2
            + par_macc_c4[t_set, n_set, ghg_set] * MIU[t_set, n_set, ghg_set] ** 5 / 5
        )
    )

    # eq_mac: MAC = c1*MIU + c4*MIU^4
    # GAMS: MAC(t,n,ghg) =E= sum(coef$coefact(coef,ghg), macc_coef(t,n,ghg,coef)*power(MIU(t,n,ghg),coefn(coef)))
    eq_mac = Equation(m, name="eq_mac", domain=[t_set, n_set, ghg_set])
    eq_mac[t_set, n_set, ghg_set] = MAC[t_set, n_set, ghg_set] == (
        par_macc_c1[t_set, n_set, ghg_set] * MIU[t_set, n_set, ghg_set]
        + par_macc_c4[t_set, n_set, ghg_set] * MIU[t_set, n_set, ghg_set] ** 4
    )

    return [eq_abatecost, eq_mac]
