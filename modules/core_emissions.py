"""
Core emissions module: industrial emissions, total emissions, abated emissions,
and MIU inertia constraints.

Variables: E(t,n,ghg), EIND(t,n,ghg), MIU(t,n,ghg), ABATEDEMI(t,n,ghg)
Equations: eq_eind, eq_e, eq_abatedemi, eq_miu_up, eq_miu_dn
"""

from gamspy import Variable, Equation, Ord, Card


def declare_vars(m, sets, params, cfg, v):
    """Create emissions variables, set bounds/starting values/fixed values.

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
    # Variables (all indexed over ghg per GAMS core_emissions.gms)
    # ------------------------------------------------------------------
    # E is free (can be negative with CDR/abatement)
    E = Variable(m, name="E", domain=[t_set, n_set, ghg_set])
    # GAMS line 122: only ABATEDEMI and MIU are POSITIVE; EIND is FREE
    EIND = Variable(m, name="EIND", domain=[t_set, n_set, ghg_set])
    MIU = Variable(m, name="MIU", domain=[t_set, n_set, ghg_set], type="positive")
    ABATEDEMI = Variable(m, name="ABATEDEMI", domain=[t_set, n_set, ghg_set], type="positive")

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    # CO2 MIU upper bound (non-CO2 bounds set in core_abatement via maxmiu_pbl)
    MIU.up[t_set, n_set, ghg_set] = cfg.max_miuup
    MIU.fx["1", n_set, ghg_set] = 0  # No abatement in first period

    # ------------------------------------------------------------------
    # Starting values
    # ------------------------------------------------------------------
    par_emi_start = params["par_emi_start"]
    par_e_start = params["par_e_start"]
    # GAMS: EIND.l = convq_ghg * sigma * ykali  (no max(,0) floor)
    EIND.l[t_set, n_set, ghg_set] = par_emi_start[t_set, n_set, ghg_set]
    E.l[t_set, n_set, ghg_set] = par_e_start[t_set, n_set, ghg_set]
    MIU.l[t_set, n_set, ghg_set] = 0
    ABATEDEMI.l[t_set, n_set, ghg_set] = 0

    # Register in shared variable dict
    v["E"] = E
    v["EIND"] = EIND
    v["MIU"] = MIU
    v["ABATEDEMI"] = ABATEDEMI


def define_eqs(m, sets, params, cfg, v):
    """Create emissions equations.

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

    par_sigma = params["par_sigma"]         # sigma(t,n,ghg)
    par_convq_ghg = params["par_convq_ghg"] # convq_ghg(ghg): co2=1, ch4=1e3, n2o=1e3

    # Own variables
    E = v["E"]
    EIND = v["EIND"]
    MIU = v["MIU"]
    ABATEDEMI = v["ABATEDEMI"]

    # Cross-module variables
    YGROSS = v["YGROSS"]
    ELAND = v["ELAND"]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_eind: EIND = sigma * convq_ghg * YGROSS * (1 - MIU)
    # GAMS: eq_eind(t,n,ghg).. EIND(t,n,ghg) =E= sigma(t,n,ghg) * convq_ghg(ghg) * YGROSS(t,n) * (1-MIU(t,n,ghg))
    eq_eind = Equation(m, name="eq_eind", domain=[t_set, n_set, ghg_set])
    eq_eind[t_set, n_set, ghg_set] = (
        EIND[t_set, n_set, ghg_set] == par_sigma[t_set, n_set, ghg_set]
        * par_convq_ghg[ghg_set] * YGROSS[t_set, n_set]
        * (1 - MIU[t_set, n_set, ghg_set])
    )

    # EIND.fx for t=1: GAMS fixes EIND(tfirst,n,ghg) = sigma(tfirst,n,ghg) * convq_ghg(ghg) * ykali(tfirst,n)
    par_ykali = params["par_ykali"]
    EIND.fx["1", n_set, ghg_set] = (
        par_sigma["1", n_set, ghg_set] * par_convq_ghg[ghg_set] * par_ykali["1", n_set]
    )

    # eq_e: E = EIND + ELAND (only for CO2; ELAND is zero for non-CO2)
    #        - E_NEG (when DAC is active, CO2 only)
    # GAMS: E(t,n,ghg) =E= EIND(t,n,ghg) + ELAND(t,n)$(sameas(ghg,'co2'))
    #                      - E_NEG(t,n)$(sameas(ghg,'co2'))   [mod_dac]
    eq_e = Equation(m, name="eq_e", domain=[t_set, n_set, ghg_set])
    # Issue 8: when DAC is active, subtract E_NEG for CO2
    if "E_NEG" in v:
        E_NEG = v["E_NEG"]
        eq_e[t_set, n_set, ghg_set] = (
            E[t_set, n_set, ghg_set] == EIND[t_set, n_set, ghg_set]
            + ELAND[t_set, n_set].where[Ord(ghg_set) == 1]
            - E_NEG[t_set, n_set].where[Ord(ghg_set) == 1]
        )
    else:
        eq_e[t_set, n_set, ghg_set] = (
            E[t_set, n_set, ghg_set] == EIND[t_set, n_set, ghg_set]
            + ELAND[t_set, n_set].where[Ord(ghg_set) == 1]
        )
    # Note: ghg_set ordering is (co2, ch4, n2o), so Ord==1 selects co2.
    # The caller must ensure 'co2' is the first element of ghg_set.

    # eq_abatedemi: ABATEDEMI = MIU * sigma * convq_ghg * YGROSS
    # GAMS: eq_abatedemi(t,n,ghg).. ABATEDEMI(t,n,ghg) =E= MIU(t,n,ghg) * sigma(t,n,ghg) * convq_ghg(ghg) * YGROSS(t,n)
    eq_abatedemi = Equation(m, name="eq_abatedemi", domain=[t_set, n_set, ghg_set])
    eq_abatedemi[t_set, n_set, ghg_set] = (
        ABATEDEMI[t_set, n_set, ghg_set] == MIU[t_set, n_set, ghg_set]
        * par_sigma[t_set, n_set, ghg_set] * par_convq_ghg[ghg_set]
        * YGROSS[t_set, n_set]
    )

    # MIU inertia constraints
    # GAMS: eq_miuinertiaplus(t,tp1,n,ghg)$(tperiod(t) gt 1 and pre(t,tp1) and not tmiufix(tp1))..
    #        MIU(tp1,n,ghg) =L= MIU(t,n,ghg) + miu_inertia(ghg)*tstep
    # Note on tmiufix: The GAMS condition `not tmiufix(tp1)` with tmiufix={1,2}
    # excludes constraints where tp1 is period 1 or 2. Since tp1=t.lead(1),
    # tp1=1 corresponds to t=0 (unused) and tp1=2 corresponds to t=1.
    # Our condition `Ord(t) > 1` already ensures t >= 2, which means tp1 >= 3,
    # so the tmiufix exclusion for {1,2} is already implicitly satisfied.
    # If tmiufix were extended, this condition would need updating.
    miu_inertia_yr = cfg.miu_inertia
    TSTEP = cfg.TSTEP

    eq_miu_up = Equation(m, name="eq_miu_up", domain=[t_set, n_set, ghg_set])
    eq_miu_up[t_set, n_set, ghg_set].where[
        (Ord(t_set) > 1) & (Ord(t_set) < Card(t_set))
    ] = (
        MIU[t_set.lead(1), n_set, ghg_set]
        <= MIU[t_set, n_set, ghg_set] + miu_inertia_yr * TSTEP
    )

    eq_miu_dn = Equation(m, name="eq_miu_dn", domain=[t_set, n_set, ghg_set])
    eq_miu_dn[t_set, n_set, ghg_set].where[
        (Ord(t_set) > 1) & (Ord(t_set) < Card(t_set))
    ] = (
        MIU[t_set.lead(1), n_set, ghg_set]
        >= MIU[t_set, n_set, ghg_set] - miu_inertia_yr * TSTEP
    )

    return [eq_eind, eq_e, eq_abatedemi, eq_miu_up, eq_miu_dn]
