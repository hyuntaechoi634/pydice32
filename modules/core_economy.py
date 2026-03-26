"""
Core economy module: production, consumption, capital accumulation.

Variables: YGROSS, YNET, Y, C, CPC, K, I, S, RI
Equations: eq_ygross, eq_ynet, eq_yy, eq_s, eq_cc, eq_cpc, eq_kk, eq_ri
"""

from gamspy import Variable, Equation, Ord, Card, Sum


def declare_vars(m, sets, params, cfg, v):
    """Create economy variables, set bounds/starting values/fixed values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, t_alias, layers, layers_alias, n_alias
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # Unpack parameters
    par_ykali = params["par_ykali"]
    par_fixed_savings = params["par_fixed_savings"]
    par_k0 = params["par_k0"]
    par_k_start = params["par_k_start"]
    par_cpc_start = params["par_cpc_start"]

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    YGROSS = Variable(m, name="YGROSS", domain=[t_set, n_set], type="positive")
    YNET = Variable(m, name="YNET", domain=[t_set, n_set], type="positive")
    Y = Variable(m, name="Y", domain=[t_set, n_set], type="positive")
    C = Variable(m, name="C", domain=[t_set, n_set], type="positive")
    CPC = Variable(m, name="CPC", domain=[t_set, n_set], type="positive")
    K = Variable(m, name="K", domain=[t_set, n_set], type="positive")
    # GAMSPy name "I_inv" to avoid Python complex-number clash; GAMS name is "I"
    I = Variable(m, name="I_inv", domain=[t_set, n_set], type="positive")
    S = Variable(m, name="S", domain=[t_set, n_set], type="positive")
    RI = Variable(m, name="RI", domain=[t_set, n_set])

    # ------------------------------------------------------------------
    # Bounds (GAMS compute_vars stability constraints)
    # ------------------------------------------------------------------
    YGROSS.lo[t_set, n_set] = 1e-8
    YNET.lo[t_set, n_set] = 1e-8
    Y.lo[t_set, n_set] = 1e-8
    C.lo[t_set, n_set] = 1e-8
    CPC.lo[t_set, n_set] = 1e-8
    K.lo[t_set, n_set] = 1e-8
    S.up[t_set, n_set] = 1

    # GAMS: UTARG.lo(t,n) = 1e-8 (set in core_economy compute_vars)
    # Applied here because UTARG is declared in core_welfare but bound in core_economy per GAMS
    if "UTARG" in v:
        v["UTARG"].lo[t_set, n_set] = 1e-8

    # ------------------------------------------------------------------
    # Starting values
    # ------------------------------------------------------------------
    YGROSS.l[t_set, n_set] = par_ykali[t_set, n_set]
    YNET.l[t_set, n_set] = par_ykali[t_set, n_set]
    Y.l[t_set, n_set] = par_ykali[t_set, n_set]
    S.l[t_set, n_set] = par_fixed_savings[t_set, n_set]
    CPC.l[t_set, n_set] = par_cpc_start[t_set, n_set]
    C.l[t_set, n_set] = par_ykali[t_set, n_set] * (1 - par_fixed_savings[t_set, n_set])
    K.l[t_set, n_set] = par_k_start[t_set, n_set]
    I.l[t_set, n_set] = par_fixed_savings[t_set, n_set] * par_ykali[t_set, n_set]
    RI.l[t_set, n_set] = 0.05

    # ------------------------------------------------------------------
    # Fixed initial conditions
    # ------------------------------------------------------------------
    K.fx["1", n_set] = par_k0[n_set]
    # GAMS doesn't explicitly fix YGROSS(tfirst), but eq_ygross skips t=1
    # and GAMS keeps .l values for variables not in any active equation.
    # In GAMSPy, YGROSS(1) appears in eq_ynet/eq_eind which apply at all t,
    # so we must fix it to prevent the optimizer from inflating it.
    YGROSS.fx["1", n_set] = par_ykali["1", n_set]

    # Register in shared variable dict
    v["YGROSS"] = YGROSS
    v["YNET"] = YNET
    v["Y"] = Y
    v["C"] = C
    v["CPC"] = CPC
    v["K"] = K
    v["I"] = I
    v["S"] = S
    v["RI"] = RI


def define_eqs(m, sets, params, cfg, v):
    """Create economy equations.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, t_alias, layers, layers_alias, n_alias
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

    # Unpack parameters
    par_tfp = params["par_tfp"]
    par_pop = params["par_pop"]
    par_prodshare_cap = params["par_prodshare_cap"]
    par_prodshare_lab = params["par_prodshare_lab"]

    DK = cfg.DK
    TSTEP = cfg.TSTEP
    PRSTP = cfg.PRSTP
    ELASMU = cfg.ELASMU

    # Own variables
    YGROSS = v["YGROSS"]
    YNET = v["YNET"]
    Y = v["Y"]
    C = v["C"]
    CPC = v["CPC"]
    K = v["K"]
    I = v["I"]
    S = v["S"]
    RI = v["RI"]

    # Cross-module variables
    DAMAGES = v["DAMAGES"]
    ABATECOST = v["ABATECOST"]  # now (t,n,ghg)
    ABCOSTLAND = v["ABCOSTLAND"]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_ygross: YGROSS = tfp * K^cap * (pop/1000)^lab, only t>1
    eq_ygross = Equation(m, name="eq_ygross", domain=[t_set, n_set])
    eq_ygross[t_set, n_set].where[Ord(t_set) > 1] = (
        YGROSS[t_set, n_set] == par_tfp[t_set, n_set]
        * (K[t_set, n_set] ** par_prodshare_cap[n_set])
        * ((par_pop[t_set, n_set] / 1000) ** par_prodshare_lab[n_set])
    )

    # eq_ynet: YNET = YGROSS - DAMAGES
    # When damages_postprocessed is True, DAMAGES are decoupled from the
    # optimization: YNET = YGROSS (damages are computed post-solve).
    # This corresponds to GAMS: eq_ynet.. YNET = YGROSS - DAMAGES.l
    # where .l means the fixed level, effectively removing the feedback.
    eq_ynet = Equation(m, name="eq_ynet", domain=[t_set, n_set])
    if cfg.damages_postprocessed:
        # Damages handled post-solve; optimization sees no damage feedback
        eq_ynet[t_set, n_set] = (
            YNET[t_set, n_set] == YGROSS[t_set, n_set]
        )
    else:
        eq_ynet[t_set, n_set] = (
            YNET[t_set, n_set] == YGROSS[t_set, n_set] - DAMAGES[t_set, n_set]
        )

    # eq_yy: Y = YNET - Sum(ghg, ABATECOST) - ABCOSTLAND
    #         - COST_CDR (when DAC active, Issue 8)
    #         - Sum(ghg, ctax_corrected * convy_ghg * (E - E_bau))  [ctax fiscal revenue]
    # GAMS always includes abatement costs (eq_yy is unconditional on policy).
    # For BAU, MIU=0 makes ABATECOST=0 naturally.
    eq_yy = Equation(m, name="eq_yy", domain=[t_set, n_set])
    yy_rhs = (
        YNET[t_set, n_set]
        - Sum(ghg_set, ABATECOST[t_set, n_set, ghg_set])
        - ABCOSTLAND[t_set, n_set]
    )
    # GAMS eq_yy conditionals: COST_CDR (mod_dac), COST_SAI (mod_sai)
    if "COST_CDR" in v:
        yy_rhs = yy_rhs - v["COST_CDR"][t_set, n_set]
    if "COST_SAI" in v:
        yy_rhs = yy_rhs - v["COST_SAI"][t_set, n_set]
    # Item 6: Carbon tax fiscal revenue term
    # GAMS: - sum(ghg, ctax_corrected(t,n,ghg) * convy_ghg(ghg) * (E(t,n,ghg) - E.l(t,n,ghg)))
    # When ctax policy is active (and not ctax_marginal), add fiscal revenue.
    # Uses par_emi_bau as proxy for E.l (BAU emission level).
    if (cfg.policy in ("ctax", "cbudget_regional")
            and "par_ctax_corrected" in v
            and not cfg.ctax_marginal):
        par_ctax_corr = v["par_ctax_corrected"]
        par_convy_ghg = params["par_convy_ghg"]
        par_emi_bau = params["par_emi_bau"]
        E = v["E"]
        yy_rhs = yy_rhs - Sum(
            ghg_set,
            par_ctax_corr[t_set, ghg_set]
            * par_convy_ghg[ghg_set]
            * (E[t_set, n_set, ghg_set] - par_emi_bau[t_set, n_set, ghg_set])
        )
    eq_yy[t_set, n_set] = Y[t_set, n_set] == yy_rhs

    # eq_s: I = S * Y
    eq_s = Equation(m, name="eq_s", domain=[t_set, n_set])
    eq_s[t_set, n_set] = I[t_set, n_set] == S[t_set, n_set] * Y[t_set, n_set]

    # eq_cc: C = Y - I [- Sum(g, I_ADA) when adaptation] [- Sum(type, NAT_INV) when nat_cap]
    eq_cc = Equation(m, name="eq_cc", domain=[t_set, n_set])
    cc_rhs = Y[t_set, n_set] - I[t_set, n_set]
    # Issue 11: adaptation investment deduction
    if "I_ADA" in v:
        g_ada = sets.get("g_ada")
        if g_ada is not None:
            cc_rhs = cc_rhs - Sum(g_ada, v["I_ADA"][g_ada, t_set, n_set])
    # Issue 11: natural capital investment deduction
    if "NAT_INV" in v:
        nc_type = sets.get("nc_type")
        if nc_type is not None:
            cc_rhs = cc_rhs - Sum(nc_type, v["NAT_INV"][nc_type, t_set, n_set])
    eq_cc[t_set, n_set] = C[t_set, n_set] == cc_rhs

    # eq_cpc: CPC = C / pop * 1e6
    eq_cpc = Equation(m, name="eq_cpc", domain=[t_set, n_set])
    eq_cpc[t_set, n_set] = CPC[t_set, n_set] == C[t_set, n_set] / par_pop[t_set, n_set] * 1e6

    # eq_kk: K[t+1] = (1-DK)^TSTEP * K[t] + TSTEP * I[t], t<T
    eq_kk = Equation(m, name="eq_kk", domain=[t_set, n_set])
    eq_kk[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        K[t_set.lead(1), n_set] == (1 - DK) ** TSTEP * K[t_set, n_set]
        + TSTEP * I[t_set, n_set]
    )

    # eq_ri: RI = (1+PRSTP) * (CPC[t+1]/CPC[t])^(ELASMU/TSTEP) - 1, t<T
    eq_ri = Equation(m, name="eq_ri", domain=[t_set, n_set])
    eq_ri[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        RI[t_set, n_set] == (1 + PRSTP)
        * (CPC[t_set.lead(1), n_set] / CPC[t_set, n_set]) ** (ELASMU / TSTEP) - 1
    )

    return [eq_ygross, eq_ynet, eq_yy, eq_s, eq_cc, eq_cpc, eq_kk, eq_ri]
