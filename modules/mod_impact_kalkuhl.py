"""
Kalkuhl impact module: growth-rate damages (Kalkuhl & Wenz style).

Variables: BIMPACT
Equations: eq_bimpact (Kalkuhl growth damage), eq_omega (simple recursive)
"""

from gamspy import Variable, Equation, Ord, Card


def declare_vars(m, sets, params, cfg, v):
    """Create Kalkuhl impact variables, set bounds/starting values/fixed values.

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

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    BIMPACT = Variable(m, name="BIMPACT", domain=[t_set, n_set])
    BIMPACT.l[t_set, n_set] = 0
    BIMPACT.lo[t_set, n_set] = -1 + 1e-6
    # BIMPACT fixed to 0 for t=1,2 (year(t) <= 2020)
    BIMPACT.fx["1", n_set] = 0
    BIMPACT.fx["2", n_set] = 0

    v["BIMPACT"] = BIMPACT

    # Item 12: KOMEGA variable for full omega formulation
    omega_eq = getattr(cfg, "omega_eq", "simple")
    if omega_eq == "full":
        KOMEGA = Variable(m, name="KOMEGA", domain=[t_set, n_set])
        KOMEGA.lo[t_set, n_set] = 0
        KOMEGA.l[t_set, n_set] = 1
        v["KOMEGA"] = KOMEGA


def define_eqs(m, sets, params, cfg, v):
    """Create Kalkuhl impact equations.

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

    TSTEP = cfg.TSTEP

    # Kalkuhl & Wenz (2020) damage coefficients.
    # NOTE (audit false-positive): The GAMS code (mod_impact_kalkuhl.gms
    # lines 83-84) sums kw_DT+kw_DT_lag and kw_TDT+kw_TDT_lag and applies
    # them to a SINGLE 5-year temperature difference (TEMP(t)-TEMP(tm1)).
    # This is the original Kalkuhl & Wenz 2020 specification where current
    # and one-period-lagged effects are combined into aggregate coefficients
    # on a single temperature change.  The Python correctly replicates this.
    kw_DT = 0.00641
    kw_DT_lag = 0.00345
    kw_TDT = -0.00105
    kw_TDT_lag = -0.000718

    # Own variables
    BIMPACT = v["BIMPACT"]

    # Cross-module variables
    OMEGA = v["OMEGA"]
    TEMP_REGION_DAM = v["TEMP_REGION_DAM"]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_bimpact: Kalkuhl growth damage (t > 2)
    eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
    eq_bimpact[t_set, n_set].where[Ord(t_set) > 2] = (
        BIMPACT[t_set, n_set] ==
        (kw_DT + kw_DT_lag) * (
            TEMP_REGION_DAM[t_set, n_set] - TEMP_REGION_DAM[t_set.lag(1), n_set]
        )
        + (kw_TDT + kw_TDT_lag) * (
            TEMP_REGION_DAM[t_set, n_set] - TEMP_REGION_DAM[t_set.lag(1), n_set]
        ) / TSTEP
        * (2 * (TEMP_REGION_DAM[t_set, n_set] - TEMP_REGION_DAM[t_set.lag(1), n_set])
           + 5 * TEMP_REGION_DAM[t_set.lag(1), n_set])
    )

    # Item 12: Full omega formulation (when omega_eq='full')
    omega_eq = getattr(cfg, "omega_eq", "simple")
    equations = [eq_bimpact]

    if omega_eq == "full":
        # GAMS full omega:
        # OMEGA(tp1,n) = ((1+OMEGA(t,n)) * (tfp(tp1)/tfp(t))
        #   * ((pop(tp1)/pop(t))^lab_share * pop(t)/pop(tp1))
        #   * KOMEGA(t,n) / (1+basegrowthcap(t,n)+BIMPACT(t,n))^tstep) - 1
        # KOMEGA(t,n) = (((1-dk)^tstep*K + tstep*S*tfp*K^cap*(pop/1000)^lab
        #   * 1/(1+OMEGA)) / K)^cap
        KOMEGA = v["KOMEGA"]
        K = v["K"]
        S = v["S"]
        par_tfp = params["par_tfp"]
        par_pop = params["par_pop"]
        par_prodshare_cap = params["par_prodshare_cap"]
        par_prodshare_lab = params["par_prodshare_lab"]
        DK = cfg.DK

        # Compute basegrowthcap parameter: (ykali(tp1)/pop(tp1)) / (ykali(t)/pop(t)) - 1
        # This is the baseline per-capita GDP growth rate (annualized)
        # Approximate from ykali and pop data
        par_ykali = params["par_ykali"]
        T = cfg.T
        region_names = [r for r in par_pop.records["n"].unique()]

        bgc_recs = []
        yk_d = {}
        pp_d = {}
        if hasattr(par_ykali, 'records') and par_ykali.records is not None:
            for _, row in par_ykali.records.iterrows():
                yk_d[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])
        if hasattr(par_pop, 'records') and par_pop.records is not None:
            for _, row in par_pop.records.iterrows():
                pp_d[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])

        for t in range(1, T):
            for r in region_names:
                yk_t = yk_d.get((str(t), r), 1)
                yk_t1 = yk_d.get((str(t + 1), r), 1)
                pp_t = pp_d.get((str(t), r), 1)
                pp_t1 = pp_d.get((str(t + 1), r), 1)
                gdppc_t = yk_t / max(pp_t, 1e-6)
                gdppc_t1 = yk_t1 / max(pp_t1, 1e-6)
                bgc = (gdppc_t1 / max(gdppc_t, 1e-12)) ** (1.0 / TSTEP) - 1
                bgc_recs.append((str(t), r, bgc))
        from gamspy import Parameter as GParam
        par_bgc = GParam(m, name="basegrowthcap",
                         domain=[t_set, n_set], records=bgc_recs)

        # eq_komega: KOMEGA = (((1-dk)^tstep * K + tstep * S * tfp * K^cap * (pop/1000)^lab
        #   * 1/(1+OMEGA)) / K)^cap
        eq_komega = Equation(m, name="eq_komega", domain=[t_set, n_set])
        eq_komega[t_set, n_set] = (
            KOMEGA[t_set, n_set] ==
            (((1 - DK) ** TSTEP * K[t_set, n_set]
              + TSTEP * S[t_set, n_set] * par_tfp[t_set, n_set]
              * K[t_set, n_set] ** par_prodshare_cap[n_set]
              * (par_pop[t_set, n_set] / 1000) ** par_prodshare_lab[n_set]
              * (1 / (1 + OMEGA[t_set, n_set])))
             / K[t_set, n_set]) ** par_prodshare_cap[n_set]
        )
        equations.append(eq_komega)

        # eq_omega (full): recursive with TFP, pop, and KOMEGA factors
        eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])
        eq_omega[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
            OMEGA[t_set.lead(1), n_set] ==
            ((1 + OMEGA[t_set, n_set])
             * (par_tfp[t_set.lead(1), n_set] / par_tfp[t_set, n_set])
             * ((par_pop[t_set.lead(1), n_set] / par_pop[t_set, n_set])
                ** par_prodshare_lab[n_set])
             * (par_pop[t_set, n_set] / par_pop[t_set.lead(1), n_set])
             * KOMEGA[t_set, n_set]
             / (1 + par_bgc[t_set, n_set] + BIMPACT[t_set, n_set]) ** TSTEP
             ) - 1
        )
        equations.append(eq_omega)

    else:
        # Simple omega: OMEGA(t+1) = 1/(BIMPACT(t+1) + 1/(1+OMEGA(t))) - 1
        #
        # NOTE: This formula is INTENTIONALLY different from the Burke module's
        # simple omega: OMEGA(tp1) = (1+OMEGA(t)) / (1+BIMPACT(t))^tstep - 1.
        # The difference reflects distinct economic models:
        #   - Kalkuhl models growth-rate impacts where BIMPACT enters as a
        #     reciprocal term (harmonic accumulation of damage fractions),
        #     matching GAMS mod_impact_kalkuhl.gms line 107.
        #   - Burke models level impacts with compound growth over tstep years,
        #     matching GAMS mod_impact_burke.gms line 218.
        # Both correctly replicate their respective GAMS originals.
        eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])
        eq_omega[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
            OMEGA[t_set.lead(1), n_set] ==
            1 / (BIMPACT[t_set.lead(1), n_set] + 1 / (1 + OMEGA[t_set, n_set])) - 1
        )
        equations.append(eq_omega)

    return equations
