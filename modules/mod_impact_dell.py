"""
Dell, Jones & Olken (2012) impact module: growth-rate damages.

Based on GAMS mod_impact_dell.gms.
Temperature affects GDP growth rate with rich/poor differentiation.
Rich: beta_T = 0.00261 (not statistically significant)
Poor: beta_T = 0.00261 - 0.01655 = -0.01394 (significant negative)

DJOIMPACT = beta_T * (TEMP_REGION_DAM - base_temp)
OMEGA(t+1) = (1+OMEGA(t)) / (1+DJOIMPACT)^tstep - 1  (simple)
"""

from gamspy import Variable, Equation, Ord, Card, Parameter


# DJO coefficients
DJO_RICH_T = 0.00261
DJO_POOR_T = 0.00261 - 0.01655  # -0.01394


def declare_vars(m, sets, params, cfg, v):
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    BIMPACT = Variable(m, name="BIMPACT", domain=[t_set, n_set])
    BIMPACT.l[t_set, n_set] = 0
    BIMPACT.lo[t_set, n_set] = -1 + 1e-6
    BIMPACT.fx["1", n_set] = 0
    BIMPACT.fx["2", n_set] = 0
    v["BIMPACT"] = BIMPACT

    # DJOIMPACT: yearly local impact
    DJOIMPACT = Variable(m, name="DJOIMPACT", domain=[t_set, n_set])
    DJOIMPACT.l[t_set, n_set] = 0
    DJOIMPACT.lo[t_set, n_set] = -1 + 1e-5
    DJOIMPACT.fx["1", n_set] = 0
    v["DJOIMPACT"] = DJOIMPACT

    # Full omega: KOMEGA
    omega_eq = getattr(cfg, "omega_eq", "simple")
    if omega_eq == "full":
        KOMEGA = Variable(m, name="KOMEGA", domain=[t_set, n_set])
        KOMEGA.lo[t_set, n_set] = 0
        KOMEGA.l[t_set, n_set] = 1
        v["KOMEGA"] = KOMEGA


def define_eqs(m, sets, params, cfg, v):
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    TSTEP = cfg.TSTEP
    T = cfg.T

    DJOIMPACT = v["DJOIMPACT"]
    BIMPACT = v["BIMPACT"]
    OMEGA = v["OMEGA"]
    TEMP_REGION_DAM = v["TEMP_REGION_DAM"]

    par_base_temp = params["par_base_temp"]

    # Rich/poor classification (same approach as Burke)
    # Use region-level GDP per capita at t=1 with median cutoff
    par_ykali = params["par_ykali"]
    par_pop = params["par_pop"]
    region_names = [r for r in par_pop.records["n"].unique()]

    ykali_recs = {}
    pop_recs = {}
    if hasattr(par_ykali, 'records') and par_ykali.records is not None:
        for _, row in par_ykali.records.iterrows():
            ykali_recs[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])
    if hasattr(par_pop, 'records') and par_pop.records is not None:
        for _, row in par_pop.records.iterrows():
            pop_recs[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])

    # Compute GDPpc at t=1 and classify
    gdppc_t1 = {}
    for r in region_names:
        yk = ykali_recs.get(("1", r), 0)
        pp = pop_recs.get(("1", r), 1)
        gdppc_t1[r] = yk * 1e6 / max(pp, 1e-6)

    vals = sorted(gdppc_t1.values())
    cutoff = vals[len(vals) // 2] if vals else 13205.0

    # Create beta_djo_T parameter (time-invariant for simplicity, matches GAMS line 68-70)
    beta_recs = []
    for r in region_names:
        if gdppc_t1.get(r, 0) > cutoff:
            beta_recs.append((r, DJO_RICH_T))
        else:
            beta_recs.append((r, DJO_POOR_T))

    par_beta_djo = Parameter(m, name="beta_djo_T", domain=[n_set], records=beta_recs)

    equations = []

    # eq_djoimpact: DJOIMPACT = beta_T * (TEMP_REGION_DAM - base_temp)
    eq_djoimpact = Equation(m, name="eq_djoimpact", domain=[t_set, n_set])
    eq_djoimpact[t_set, n_set].where[Ord(t_set) > 1] = (
        DJOIMPACT[t_set, n_set] == par_beta_djo[n_set]
        * (TEMP_REGION_DAM[t_set, n_set] - par_base_temp[n_set])
    )
    equations.append(eq_djoimpact)

    # eq_bimpact: trivially 0 (impact channeled through DJOIMPACT → OMEGA)
    eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
    eq_bimpact[t_set, n_set] = BIMPACT[t_set, n_set] == 0
    equations.append(eq_bimpact)

    # eq_omega: growth-rate recursive
    omega_eq = getattr(cfg, "omega_eq", "simple")
    eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])

    if omega_eq == "full":
        # Full omega with capital-omega factor
        KOMEGA = v["KOMEGA"]
        K = v["K"]
        S = v["S"]
        par_tfp = params["par_tfp"]
        par_prodshare_cap = params["par_prodshare_cap"]
        par_prodshare_lab = params["par_prodshare_lab"]
        DK = cfg.DK

        # eq_komega
        eq_komega = Equation(m, name="eq_komega", domain=[t_set, n_set])
        eq_komega[t_set, n_set] = (
            KOMEGA[t_set, n_set] == (
                ((1 - DK) ** TSTEP * K[t_set, n_set]
                 + TSTEP * S[t_set, n_set] * par_tfp[t_set, n_set]
                 * (K[t_set, n_set] ** par_prodshare_cap[n_set])
                 * ((par_pop[t_set, n_set] / 1000) ** par_prodshare_lab[n_set])
                 * (1 / (1 + OMEGA[t_set, n_set])))
                / K[t_set, n_set]
            ) ** par_prodshare_cap[n_set]
        )
        equations.append(eq_komega)

        # Full omega
        basegrowthcap = params.get("par_basegrowthcap")
        eq_omega[t_set, n_set].where[
            (Ord(t_set) > 1) & (Ord(t_set) < Card(t_set))
        ] = (
            OMEGA[t_set.lead(1), n_set] == (
                (1 + OMEGA[t_set, n_set])
                * (par_tfp[t_set.lead(1), n_set] / par_tfp[t_set, n_set])
                * ((par_pop[t_set.lead(1), n_set] / par_pop[t_set, n_set])
                   ** par_prodshare_lab[n_set])
                * (par_pop[t_set, n_set] / par_pop[t_set.lead(1), n_set])
                * KOMEGA[t_set, n_set]
                / ((1 + basegrowthcap[t_set, n_set] + DJOIMPACT[t_set, n_set]) ** TSTEP)
            ) - 1
        )
    else:
        # Simple omega: OMEGA(t+1) = (1+OMEGA(t)) / (1+DJOIMPACT)^tstep - 1
        eq_omega[t_set, n_set].where[
            (Ord(t_set) > 1) & (Ord(t_set) < Card(t_set))
        ] = (
            OMEGA[t_set.lead(1), n_set] == (
                (1 + OMEGA[t_set, n_set])
                / (1 + DJOIMPACT[t_set, n_set]) ** TSTEP
            ) - 1
        )
    equations.append(eq_omega)

    return equations
