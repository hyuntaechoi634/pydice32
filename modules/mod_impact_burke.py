"""
Burke impact module: growth-rate damages (Burke et al. 2015 style).

This is distinct from the Kalkuhl module (mod_impact_kalkuhl.py).
Burke uses level-temperature coefficients on regional temperature
(T and T^2 relative to base temperature), while Kalkuhl uses
temperature-change coefficients (delta-T style).

Variables: BIMPACT
Equations: eq_bimpact (Burke growth damage), eq_omega (simple recursive)

References
----------
- Burke, Hsiang & Miguel (2015), "Global non-linear effect of temperature
  on economic production", Nature 527, 235-239.
"""

from gamspy import Variable, Equation, Ord, Card


# Burke et al. 2015 damage coefficients
# ── Short run (baseline)
BHM_SR_T = 0.0127184
BHM_SR_T2 = -0.0004871
# ── Long run
BHM_LR_T = -0.0037497
BHM_LR_T2 = -0.0000955
# ── Short run differentiated
BHM_SRDIFF_RICH_T = 0.0088951
BHM_SRDIFF_RICH_T2 = -0.0003155
BHM_SRDIFF_POOR_T = 0.0254342
BHM_SRDIFF_POOR_T2 = -0.000772
# ── Long run differentiated
BHM_LRDIFF_RICH_T = -0.0026918
BHM_LRDIFF_RICH_T2 = -0.000022
BHM_LRDIFF_POOR_T = -0.0186
BHM_LRDIFF_POOR_T2 = 0.0001513


def declare_vars(m, sets, params, cfg, v):
    """Create Burke impact variables, set bounds/starting values/fixed values.

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

    # Fix first period to zero (GAMS: tfirst only)
    BIMPACT.fx["1", n_set] = 0
    # Note: GAMS Burke only fixes t=1 (tfirst). t=2 is left free since
    # BIMPACT depends on TEMP_REGION_DAM via eq_bimpact where Ord(t)>1.

    v["BIMPACT"] = BIMPACT

    # Item 12: KOMEGA for full omega formulation
    omega_eq = getattr(cfg, "omega_eq", "simple")
    if omega_eq == "full":
        KOMEGA = Variable(m, name="KOMEGA", domain=[t_set, n_set])
        KOMEGA.lo[t_set, n_set] = 0
        KOMEGA.l[t_set, n_set] = 1
        v["KOMEGA"] = KOMEGA


def define_eqs(m, sets, params, cfg, v):
    """Create Burke impact equations.

    Uses the 'sr' (short run, undifferentiated) specification by default.
    The bhm_spec config option selects among: sr, lr, srdiff, lrdiff.

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

    # Select BHM specification coefficients
    bhm_spec = getattr(cfg, "bhm_spec", "sr")

    # Own variables
    BIMPACT = v["BIMPACT"]

    # Cross-module variables
    OMEGA = v["OMEGA"]
    TEMP_REGION_DAM = v["TEMP_REGION_DAM"]

    # Base temperature parameter
    par_base_temp = params["par_base_temp"]

    # ------------------------------------------------------------------
    # Item 8: Region-varying coefficients for srdiff/lrdiff
    # Use GDP per capita to classify rich/poor per region per period.
    # Rich/poor cutoff: median GDP per capita (~$13,205 in 2020 PPP
    # from World Bank upper-middle/high income threshold).
    # ------------------------------------------------------------------
    if bhm_spec in ("srdiff", "lrdiff"):
        from gamspy import Parameter as GParam
        par_ykali = params["par_ykali"]
        par_pop = params["par_pop"]
        region_names = [r for r in par_pop.records["n"].unique()]
        T = cfg.T

        if bhm_spec == "srdiff":
            rich_T, rich_T2 = BHM_SRDIFF_RICH_T, BHM_SRDIFF_RICH_T2
            poor_T, poor_T2 = BHM_SRDIFF_POOR_T, BHM_SRDIFF_POOR_T2
        else:  # lrdiff
            rich_T, rich_T2 = BHM_LRDIFF_RICH_T, BHM_LRDIFF_RICH_T2
            poor_T, poor_T2 = BHM_LRDIFF_POOR_T, BHM_LRDIFF_POOR_T2

        # Compute per-region, per-period GDP per capita
        ykali_recs = {}
        pop_recs = {}
        if hasattr(par_ykali, 'records') and par_ykali.records is not None:
            for _, row in par_ykali.records.iterrows():
                ykali_recs[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])
        if hasattr(par_pop, 'records') and par_pop.records is not None:
            for _, row in par_pop.records.iterrows():
                pop_recs[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])

        # Compute median GDP per capita per period
        beta_T_recs = []
        beta_T2_recs = []
        for t in range(1, T + 1):
            # Compute GDP per capita for all regions
            gdppc = {}
            for r in region_names:
                yk = ykali_recs.get((str(t), r), 0)
                pp = pop_recs.get((str(t), r), 1)
                gdppc[r] = yk * 1e6 / max(pp, 1e-6)

            # Median cutoff
            vals = sorted(gdppc.values())
            if len(vals) > 0:
                cutoff = vals[len(vals) // 2]
            else:
                cutoff = 13205.0

            for r in region_names:
                if gdppc.get(r, 0) > cutoff:
                    beta_T_recs.append((str(t), r, rich_T))
                    beta_T2_recs.append((str(t), r, rich_T2))
                else:
                    beta_T_recs.append((str(t), r, poor_T))
                    beta_T2_recs.append((str(t), r, poor_T2))

        par_beta_T = GParam(m, name="beta_bhm_T",
                            domain=[t_set, n_set], records=beta_T_recs)
        par_beta_T2 = GParam(m, name="beta_bhm_T2",
                             domain=[t_set, n_set], records=beta_T2_recs)

        # eq_bimpact with region-varying coefficients
        eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
        eq_bimpact[t_set, n_set].where[Ord(t_set) > 1] = (
            BIMPACT[t_set, n_set] ==
            par_beta_T[t_set, n_set]
            * (TEMP_REGION_DAM[t_set, n_set] - par_base_temp[n_set])
            + par_beta_T2[t_set, n_set]
            * (TEMP_REGION_DAM[t_set, n_set] ** 2 - par_base_temp[n_set] ** 2)
        )

    else:
        # Undifferentiated (sr or lr)
        if bhm_spec == "sr":
            bhm_T = BHM_SR_T
            bhm_T2 = BHM_SR_T2
        elif bhm_spec == "lr":
            bhm_T = BHM_LR_T
            bhm_T2 = BHM_LR_T2
        else:
            bhm_T = BHM_SR_T
            bhm_T2 = BHM_SR_T2

        # ------------------------------------------------------------------
        # eq_bimpact: Burke level-temperature damage (t > 1)
        # GAMS:  BIMPACT(t,n) = bhm_T * (TEMP_REGION_DAM - base_temp)
        #                      + bhm_T2 * (TEMP_REGION_DAM^2 - base_temp^2)
        eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
        eq_bimpact[t_set, n_set].where[Ord(t_set) > 1] = (
            BIMPACT[t_set, n_set] ==
            bhm_T * (TEMP_REGION_DAM[t_set, n_set] - par_base_temp[n_set])
            + bhm_T2 * (TEMP_REGION_DAM[t_set, n_set] ** 2
                         - par_base_temp[n_set] ** 2)
        )

    # Item 12: omega formulation (simple or full)
    omega_eq = getattr(cfg, "omega_eq", "simple")
    equations = [eq_bimpact]

    if omega_eq == "full" and "KOMEGA" in v:
        # Full omega with KOMEGA (same as Kalkuhl full omega)
        KOMEGA = v["KOMEGA"]
        K = v["K"]
        S = v["S"]
        par_tfp = params["par_tfp"]
        par_pop = params["par_pop"]
        par_prodshare_cap = params["par_prodshare_cap"]
        par_prodshare_lab = params["par_prodshare_lab"]
        DK = cfg.DK

        # basegrowthcap parameter
        par_ykali = params["par_ykali"]
        T = cfg.T
        rn = [r for r in par_pop.records["n"].unique()]
        yk_d = {}
        pp_d = {}
        if hasattr(par_ykali, 'records') and par_ykali.records is not None:
            for _, row in par_ykali.records.iterrows():
                yk_d[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])
        if hasattr(par_pop, 'records') and par_pop.records is not None:
            for _, row in par_pop.records.iterrows():
                pp_d[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])

        bgc_recs = []
        for t_idx in range(1, T):
            for r in rn:
                yk_t = yk_d.get((str(t_idx), r), 1)
                yk_t1 = yk_d.get((str(t_idx + 1), r), 1)
                pp_t = pp_d.get((str(t_idx), r), 1)
                pp_t1 = pp_d.get((str(t_idx + 1), r), 1)
                gdppc_t = yk_t / max(pp_t, 1e-6)
                gdppc_t1 = yk_t1 / max(pp_t1, 1e-6)
                bgc = (gdppc_t1 / max(gdppc_t, 1e-12)) ** (1.0 / TSTEP) - 1
                bgc_recs.append((str(t_idx), r, bgc))
        from gamspy import Parameter as GParam2
        par_bgc = GParam2(m, name="basegrowthcap_burke",
                          domain=[t_set, n_set], records=bgc_recs)

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
        # Simple recursive omega
        # GAMS: OMEGA(tp1,n) = (1 + OMEGA(t,n)) / (1 + BIMPACT(t,n))^tstep - 1
        eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])
        eq_omega[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
            OMEGA[t_set.lead(1), n_set] ==
            ((1 + OMEGA[t_set, n_set])
             / ((1 + BIMPACT[t_set, n_set]) ** TSTEP)) - 1
        )
        equations.append(eq_omega)

    return equations
