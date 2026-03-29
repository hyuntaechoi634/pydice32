"""
Policy constraint module: policy-specific variable fixings and constraint equations.
Corresponds to GAMS core_policy.gms.
Policies: bau, bau_impact, cba, cbudget, cea_tatm, ctax, global_netzero,
          long_term_pledges, simulation
"""

from gamspy import Equation, Variable, Parameter, Ord, Card, Sum


def declare_vars(m, sets, params, cfg, v):
    """Apply policy-specific variable fixings."""
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]
    policy = cfg.policy

    MIU = v["MIU"]
    MIULAND = v["MIULAND"]
    S = v["S"]

    # Savings mode
    par_fixed_savings = params["par_fixed_savings"]
    if cfg.savings_mode == "fixed":
        S.fx[t_set, n_set] = par_fixed_savings[t_set, n_set]

    elif cfg.savings_mode == "flexible":
        # GAMS core_economy.gms lines 290-302 (compute_vars):
        # Savings are left free to be optimized with gradual bounds.
        #
        # S.lo(t,n) = 0.1;
        # S.up(t,n) = 0.45;
        # S.lo(t,n) = s0 + (S.lo('58',n) - s0) * (tperiod(t)-1)/(smax(tt,tperiod(tt))-1);
        # S.up(t,n) = s0 + (S.up('58',n) - s0) * (tperiod(t)-1)/(smax(tt,tperiod(tt))-1);
        # S.fx(tfirst,n) = s0;
        #
        # GAMS before_solve (lines 384-387):
        # S.fx(t,n)$(tperiod(t) gt (smax(tt,tperiod(tt)) - 10)) = S.l('48',n);
        #
        # In our notation: tperiod(t) = t (1-based), smax(tt,tperiod(tt)) = T = 58.
        # S.lo('58',n) = 0.1, S.up('58',n) = 0.45 (the terminal bounds).
        # The linear interpolation ramps S bounds from s0 at t=1 to [0.1, 0.45] at t=58.
        # Terminal fix: for t > T-10 (i.e., t >= 49), S is fixed to S.l('48',n).
        # In single-pass mode, we approximate S.l('48',n) with the fixed_savings value
        # at t=48 (since we don't have prior solution levels).

        par_s0 = params["par_s0"]
        T = cfg.T  # 58

        # Set base bounds: S.lo = 0.1, S.up = 0.45
        S.lo[t_set, n_set] = 0.1
        S.up[t_set, n_set] = 0.45

        # Linear interpolation of bounds from s0 at t=1 to terminal bounds at t=T
        # GAMS: S.lo(t,n) = s0 + (S.lo('58',n) - s0) * (t-1)/(T-1)
        # GAMS: S.up(t,n) = s0 + (S.up('58',n) - s0) * (t-1)/(T-1)
        # We implement via GAMSPy domain operations:
        #   S.lo(t,n) = s0(n) + (0.1 - s0(n)) * (Ord(t)-1)/(T-1)
        #   S.up(t,n) = s0(n) + (0.45 - s0(n)) * (Ord(t)-1)/(T-1)
        # Since GAMSPy .lo/.up assignment with expressions is not directly
        # supported in the same way as GAMS, we set bounds per-period.
        # GAMSPy column naming: the domain column is typically named after
        # the set (e.g. "n"), and the value column is "value" for Parameters.
        # Use .iloc positional access as a defensive fallback in case GAMSPy
        # renames columns across versions.
        s0_recs = params["par_s0"].records
        region_names = s0_recs.iloc[:, 0].tolist()
        s0_vals = dict(zip(
            s0_recs.iloc[:, 0].tolist(),
            s0_recs.iloc[:, 1].tolist(),
        ))
        for t in range(1, T + 1):
            frac = (t - 1) / (T - 1)
            for rname in region_names:
                s0_val = s0_vals.get(rname, 0.2)
                lo_val = s0_val + (0.1 - s0_val) * frac
                up_val = s0_val + (0.45 - s0_val) * frac
                S.lo[str(t), rname] = lo_val
                S.up[str(t), rname] = up_val

        # Fix initial period: S.fx(tfirst,n) = s0
        S.fx["1", n_set] = par_s0[n_set]

        # Terminal fix: for t > T-10 (t >= 49), fix S to the fixed_savings
        # at t=48 as a proxy for S.l('48',n) (GAMS before_solve).
        for t in range(T - 10 + 1, T + 1):  # t = 49..58
            for rname in region_names:
                # Use .iloc for positional access (col 0=t, col 1=n, col 2=value)
                # to guard against GAMSPy column name changes across versions.
                s48 = params["par_fixed_savings"].records
                s48_val = s48[(s48.iloc[:, 0] == "48") & (s48.iloc[:, 1] == rname)]
                if len(s48_val) > 0:
                    S.fx[str(t), rname] = s48_val.iloc[0, 2]
                else:
                    S.fx[str(t), rname] = s0_vals.get(rname, 0.2)

    # Policy-specific fixings
    if policy == "bau":
        # GAMS: MIU.fx(t,n,ghg) = 0 (which makes ABATECOST=0 via eq_abatecost)
        MIU.fx[t_set, n_set, ghg_set] = 0
        # GAMS BAU sets impact="off", which zeros DAMAGES via hub_impact eq_damages.
        # Do NOT fix DAMAGES here; it is controlled by the impact module.
        MIULAND.fx[t_set, n_set] = 0

    elif policy == "bau_impact":
        # BAU with impacts: MIU=0, MIULAND=0, but damages ARE computed
        # (impact is not "off" -- policy_with_damages returns True)
        MIU.fx[t_set, n_set, ghg_set] = 0
        MIULAND.fx[t_set, n_set] = 0

    elif policy == "simulation":
        # Simulation: no abatement, fixed savings enforced
        # GAMS only fixes MIU=0, NOT MIULAND (land-use abatement remains free)
        MIU.fx[t_set, n_set, ghg_set] = 0

    elif policy in ("cba", "cbudget", "cbudget_regional", "cea_tatm",
                     "cea_rcp", "ctax", "global_netzero",
                     "long_term_pledges"):
        # GAMS: MIU.fx(t,n,ghg)$tmiufix(t) = miu_fixed(t,n,ghg)
        # miu_fixed defaults to 0 for periods in tmiufix
        for t_idx in cfg.tmiufix:
            MIU.fx[str(t_idx), n_set, ghg_set] = 0

    # ------------------------------------------------------------------
    # NDC MIU floors (pol_ndc): load NDC MIU values for t=3 (2025), t=4 (2030)
    # GAMS: miu_fixed('3',n,'co2') = (1/2)*miu_ndcs_2030(n)
    #        miu_fixed('4',n,'co2') = miu_ndcs_2030(n)
    # ------------------------------------------------------------------
    if getattr(cfg, "pol_ndc", False):
        import os
        import pandas as pd
        ndc_file = os.path.join(cfg.data_dir, "data_pol_ndc", "pbl_cond_2030.csv")
        if os.path.exists(ndc_file):
            ndc_df = pd.read_csv(ndc_file)
            from pydice32.data.gcam_mapping import (
                load_rice_regions, load_gcam_mapping, load_gcam_region_names,
            )
            rice_regions = load_rice_regions(cfg.project_root)
            rice_set = set(rice_regions)
            gcam_map = load_gcam_mapping(cfg.gcam_csv, rice_set)
            gcam_nm = load_gcam_region_names(cfg.gcam_names_csv)

            ndc_by_region = {}
            for _, row in ndc_df.iterrows():
                n_val = str(row.iloc[0]).lower()
                val = float(row["Val"])
                if n_val in gcam_map:
                    rname = gcam_nm[gcam_map[n_val]]
                    ndc_by_region[rname] = max(ndc_by_region.get(rname, 0), val)

            for rname, ndc_val in ndc_by_region.items():
                if ndc_val > 0:
                    # GAMS: ndcs_bound defaults to ".lo" (lower bound), allowing
                    # the optimizer to exceed NDC targets if beneficial.
                    MIU.lo["3", rname, "co2"] = 0.5 * ndc_val
                    MIU.lo["4", rname, "co2"] = ndc_val


def define_eqs(m, sets, params, cfg, v):
    """Define policy constraint equations."""
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]
    policy = cfg.policy
    equations = []

    # ------------------------------------------------------------------
    # Carbon budget: per-period cumulative CO2 constraint for tt >= 18
    # GAMS: eq_carbon_budget(tt)$years_budget(tt)..
    #   sum((t,n)$(year(t)>2020 and year(t)<year(tt)), E(t,n,'co2'))*tstep
    #   + 3.5*sum(n,E('2',n,'co2')) + 2.5*sum(n,E(tt,n,'co2')) =L= cbudget
    # where years_budget(t) = year(t) >= 2100, i.e. tt >= 18.
    #
    # Implementation: use a cumulative emissions tracking variable CUMEMI(t)
    # to avoid complex nested sums with dynamic ranges.
    # ------------------------------------------------------------------
    if policy == "cbudget":
        E = v["E"]  # E(t,n,ghg)

        # Cumulative CO2 emissions variable (GtCO2, positive)
        # CUMEMI(t) tracks: tstep * sum(s=3..t-1, sum(n, E(s,n,'co2')))
        # i.e. the full-weight interior sum from the GAMS equation.
        # The partial-period terms (3.5*E(2) and 2.5*E(tt)) are added
        # directly in the budget constraint.
        #
        # GAMS reference:
        #   eq_carbon_budget(tt)$years_budget(tt)..
        #     sum((t,n)$(year(t)>2020 and year(t)<year(tt)), E(t,n,'co2')) * tstep
        #     + 3.5 * sum(n, E('2',n,'co2'))
        #     + 2.5 * sum(n, E(tt,n,'co2'))
        #     =L= cbudget
        #   where years_budget(t) = year(t) >= 2100, i.e. tt >= 18.
        #
        # For tt=18 (year 2100): inner sum covers t=3..17 at weight tstep.
        # For tt=19 (year 2105): inner sum covers t=3..18 at weight tstep.
        CUMEMI = Variable(m, name="CUMEMI", domain=[t_set], type="positive")
        CUMEMI.l[t_set] = 0

        # Fix CUMEMI(1) = CUMEMI(2) = CUMEMI(3) = 0
        # (no full-period emissions accumulated before t=4)
        CUMEMI.fx["1"] = 0
        CUMEMI.fx["2"] = 0
        CUMEMI.fx["3"] = 0

        # Recursion for t >= 4:
        #   CUMEMI(t) = CUMEMI(t-1) + tstep * sum(n, E(t-1, n, 'co2'))
        eq_cumemi = Equation(m, name="eq_cumemi", domain=[t_set])
        eq_cumemi[t_set].where[Ord(t_set) >= 4] = (
            CUMEMI[t_set] == CUMEMI[t_set.lag(1)]
            + cfg.TSTEP * Sum(n_set, E[t_set.lag(1), n_set, "co2"])
        )
        equations.append(eq_cumemi)

        # Per-period budget constraint for tt >= 18 (year >= 2100):
        #   3.5 * sum(n, E('2',n,'co2'))   [partial 2020 period]
        #   + CUMEMI(tt)                     [full periods t=3..tt-1]
        #   + 2.5 * sum(n, E(tt,n,'co2'))   [partial final period]
        #   <= cbudget
        eq_cbudget = Equation(m, name="eq_cbudget", domain=[t_set], type="leq")
        eq_cbudget[t_set].where[Ord(t_set) >= 18] = (
            3.5 * Sum(n_set, E["2", n_set, "co2"])
            + CUMEMI[t_set]
            + 2.5 * Sum(n_set, E[t_set, n_set, "co2"])
            <= cfg.cbudget
        )
        equations.append(eq_cbudget)

        v["CUMEMI"] = CUMEMI

    # ------------------------------------------------------------------
    # Temperature ceiling: TATM <= limit from 2100 (overshoot allowed before)
    # GAMS: eq_tatm_limit(t)$(year(t) ge 2100).. TATM(t) =L= tatm_limit
    # ------------------------------------------------------------------
    if policy == "cea_tatm":
        TATM = v["TATM"]
        eq_tatm_limit = Equation(m, name="eq_tatm_limit", domain=[t_set], type="leq")
        eq_tatm_limit[t_set].where[Ord(t_set) >= 18] = (
            TATM[t_set] <= cfg.tatm_limit
        )
        equations.append(eq_tatm_limit)

    # ------------------------------------------------------------------
    # Carbon tax: fix MAC to exogenous tax schedule (ctax_marginal variant)
    #
    # GAMS core_policy.gms lines 273-297 / 477 implement the full fiscal-
    # revenue approach where ctax_corrected = min(ctax*1e3*emi_gwp, cprice_max)
    # and the revenue term enters eq_yy.  That approach requires iteratively
    # updated E.l (BAU emission levels) and is not feasible in single-pass
    # optimization.  Instead we use the "ctax_marginal" simplification:
    #   MAC(t,n,ghg) == ctax_sched(t,ghg)
    # which forces the marginal abatement cost to equal the tax schedule,
    # achieving the same abatement outcome without tracking fiscal revenue.
    #
    # Issue 6: tax is capped at its year-2100 value for all years > 2100,
    # matching GAMS core_policy.gms post-2100 cap behaviour.
    # ------------------------------------------------------------------
    if policy in ("ctax", "cbudget_regional"):
        MAC = v["MAC"]

        # Compute base tax schedule (T$/tCO2)
        # Issue 6: cap at year-2100 level
        base_tax_records = {}  # t -> base_tax (without GWP)
        tax_at_2100 = None
        for t in range(1, cfg.T + 1):
            yr = cfg.year(t)
            if yr >= cfg.ctax_start:
                effective_yr = min(yr, 2100)
                tax = (cfg.ctax_initial / 1000.0
                       * (1 + cfg.ctax_slope) ** (effective_yr - cfg.ctax_start))
                if yr == 2100 or (yr > 2100 and tax_at_2100 is None):
                    tax_at_2100 = tax
                if yr > 2100 and tax_at_2100 is not None:
                    tax = tax_at_2100
            else:
                tax = 0.0
            base_tax_records[t] = tax

        # Item 7: Apply GWP conversion for non-CO2 + cprice_max cap
        # GAMS: ctax_corrected(t,n,ghg) = min(ctax*1e3*emi_gwp(ghg), cprice_max(t,n,ghg))
        # emi_gwp: CO2=1, CH4=28, N2O=265 (AR4/AR5)
        emi_gwp = {"co2": 1.0, "ch4": 28.0, "n2o": 265.0}
        par_emi_gwp = params.get("par_emi_gwp")
        if par_emi_gwp is not None and hasattr(par_emi_gwp, 'records') and par_emi_gwp.records is not None:
            for _, row in par_emi_gwp.records.iterrows():
                g = str(row.iloc[0]).lower()
                emi_gwp[g] = float(row.iloc[1])

        ctax_corrected_records = []
        for t in range(1, cfg.T + 1):
            base_tax = base_tax_records[t]
            for ghg in cfg.ghg_list:
                # ctax * 1e3 * emi_gwp(ghg): convert T$/GtCO2 to T$/Gt-species
                corrected = base_tax * 1e3 * emi_gwp.get(ghg, 1.0)
                # Cap at cprice_max (approximated as a large number since
                # cprice_max depends on MIU.up and MACC coefficients)
                # In practice, the MAC equation naturally caps at the MACC
                # curve maximum, so we use a conservative large cap.
                corrected = min(corrected, 1e6)
                ctax_corrected_records.append((str(t), ghg, corrected))

        par_ctax_corrected = Parameter(
            m, name="ctax_corrected",
            domain=[t_set, ghg_set], records=ctax_corrected_records)
        v["par_ctax_corrected"] = par_ctax_corrected

        if cfg.ctax_marginal:
            # ctax_marginal: fix CO2 MAC to ctax schedule.
            # GAMS disables this by default (infeasibility risk), but for
            # single-pass cooperative solve, fiscal revenue doesn't work
            # (needs iterative E.l update).  Apply CO2-only to avoid
            # non-CO2 MACC infeasibility.  Non-CO2 abatement is implicit
            # via GWP-scaled carbon tax in the fiscal revenue term.
            tmiufix_set = set(cfg.tmiufix)
            # Compute cprice_max per (t, region) = MAC at MIU=maxmiu (CO2)
            # GAMS: cprice_max(t,n,ghg) = c1*maxmiu^1 + c4*maxmiu^4
            cprice_max_min = {}  # t -> min across regions of cprice_max
            par_macc_c1 = params.get("par_macc_c1")
            par_macc_c4 = params.get("par_macc_c4")
            par_maxmiu = params.get("par_maxmiu_pbl")
            if par_macc_c1 is not None and par_macc_c1.records is not None:
                c1_df = par_macc_c1.records
                c4_df = par_macc_c4.records if par_macc_c4 is not None else None
                mu_df = par_maxmiu.records if par_maxmiu is not None else None
                for _, row in c1_df[c1_df.iloc[:, 2] == "co2"].iterrows():
                    t_s = str(row.iloc[0])
                    n_s = str(row.iloc[1])
                    c1_v = float(row["value"])
                    c4_v = 0.0
                    if c4_df is not None:
                        c4r = c4_df[(c4_df.iloc[:,0]==t_s) & (c4_df.iloc[:,1]==n_s) & (c4_df.iloc[:,2]=="co2")]
                        if len(c4r) > 0:
                            c4_v = float(c4r["value"].iloc[0])
                    mu = 1.0
                    if mu_df is not None:
                        mur = mu_df[(mu_df.iloc[:,0]==t_s) & (mu_df.iloc[:,1]==n_s) & (mu_df.iloc[:,2]=="co2")]
                        if len(mur) > 0:
                            mu = float(mur["value"].iloc[0])
                    cp = c1_v * mu + c4_v * mu**4
                    if t_s not in cprice_max_min or cp < cprice_max_min[t_s]:
                        cprice_max_min[t_s] = cp

            ctax_sched_records = []
            for t in range(1, cfg.T + 1):
                for ghg in cfg.ghg_list:
                    # Only CO2, skip tmiufix periods
                    if ghg != "co2" or t in tmiufix_set:
                        ctax_sched_records.append((str(t), ghg, 0.0))
                    else:
                        # base_tax is in T$/GtCO2; GAMS: ctax_corrected = min(ctax*1e3, cprice_max)
                        tax = base_tax_records[t] * 1e3
                        cap = cprice_max_min.get(str(t), 1e6)
                        tax = min(tax, cap * 0.95)  # 5% margin for solver
                        ctax_sched_records.append((str(t), ghg, tax))
            par_ctax = Parameter(m, name="ctax_sched",
                                 domain=[t_set, ghg_set],
                                 records=ctax_sched_records)

            eq_ctax = Equation(m, name="eq_ctax",
                               domain=[t_set, n_set, ghg_set])
            eq_ctax[t_set, n_set, ghg_set].where[par_ctax[t_set, ghg_set] > 0] = (
                MAC[t_set, n_set, ghg_set] == par_ctax[t_set, ghg_set]
            )
            equations.append(eq_ctax)

    # ------------------------------------------------------------------
    # Global net-zero: sum(n, E(t,n,'co2')) == 0 for year(t) >= nz_year
    # GAMS: eq_global_netzero(t)$(year(t) ge nzyear).. sum(n, E(t,n,'co2')) =E= EPS
    # ------------------------------------------------------------------
    if policy == "global_netzero":
        E = v["E"]
        nz_t = (cfg.nz_year - 2015) // cfg.TSTEP + 1  # convert year to t index
        eq_netzero = Equation(m, name="eq_global_netzero",
                              domain=[t_set], type="eq")
        eq_netzero[t_set].where[Ord(t_set) >= nz_t] = (
            Sum(n_set, E[t_set, n_set, "co2"]) == 0
        )
        equations.append(eq_netzero)

    # ------------------------------------------------------------------
    # Long-term pledges: per-region net-zero year for CO2
    # GAMS: eq_long_term_pledges_co2(t,n)$(reg(n) and year(t) ge nz_year_co2(n))..
    #   E(t,n,'co2') - ELAND(t,n) =L= EPS
    # Meaning: industrial emissions <= land absorption (net-zero fossil)
    # ------------------------------------------------------------------
    if policy == "long_term_pledges":
        E = v["E"]
        ELAND = v["ELAND"]

        # Load pledge years from calibrated data (set during data loading)
        # par_nz_year_co2 is a parameter [n] with net-zero year for each region
        # We need to create a parameter [t, n] that is 1 when year(t) >= nz_year(n)
        pledge_years = params.get("par_pledge_nz_year_co2")
        if pledge_years is not None:
            # Build an indicator parameter: 1 if year(t) >= nz_year(n), else 0
            region_names = [r for r in v["E"].domain[1].records["uni"].tolist()]
            pledge_active_records = []
            # Read the pledge year parameter records back
            nz_year_dict = {}
            if hasattr(pledge_years, 'records') and pledge_years.records is not None:
                for _, row in pledge_years.records.iterrows():
                    nz_year_dict[row.iloc[0]] = row.iloc[1]

            for t in range(1, cfg.T + 1):
                yr = cfg.year(t)
                for rname in nz_year_dict:
                    nz_yr = nz_year_dict[rname]
                    if yr >= nz_yr:
                        pledge_active_records.append((str(t), rname, 1.0))

            par_pledge_active = Parameter(
                m, name="pledge_active", domain=[t_set, n_set],
                records=pledge_active_records if pledge_active_records else None)

            eq_pledges_co2 = Equation(
                m, name="eq_pledges_co2", domain=[t_set, n_set], type="leq")
            eq_pledges_co2[t_set, n_set].where[par_pledge_active[t_set, n_set] > 0] = (
                E[t_set, n_set, "co2"] - ELAND[t_set, n_set] <= 0
            )
            equations.append(eq_pledges_co2)

            # ----------------------------------------------------------
            # Issue 5: GHG pledge constraint (all greenhouse gases)
            # GAMS: eq_long_term_pledges_ghg(t,n)$(...) ..
            #   sum(ghg, EIND(t,n,ghg) * emi_gwp(ghg)) - ELAND(t,n) =L= 0
            # ----------------------------------------------------------
            EIND = v["EIND"]
            par_emi_gwp = params.get("par_emi_gwp")
            if par_emi_gwp is not None:
                eq_pledges_ghg = Equation(
                    m, name="eq_pledges_ghg", domain=[t_set, n_set], type="leq")
                eq_pledges_ghg[t_set, n_set].where[par_pledge_active[t_set, n_set] > 0] = (
                    Sum(ghg_set, EIND[t_set, n_set, ghg_set] * par_emi_gwp[ghg_set])
                    - ELAND[t_set, n_set] <= 0
                )
                equations.append(eq_pledges_ghg)

    # ------------------------------------------------------------------
    # Item 9: Radiative forcing ceiling (cea_rcp policy)
    # GAMS: eq_forc_limit(t)$(year(t) ge 2100).. FORC(t) =L= forc_limit
    # ------------------------------------------------------------------
    if policy == "cea_rcp":
        FORC = v["FORC"]
        eq_forc_limit = Equation(m, name="eq_forc_limit",
                                 domain=[t_set], type="leq")
        eq_forc_limit[t_set].where[Ord(t_set) >= 18] = (
            FORC[t_set] <= cfg.forc_limit
        )
        equations.append(eq_forc_limit)

    # ------------------------------------------------------------------
    # Item 11: Regional carbon budget (cbudget_regional policy)
    # GAMS: eq_carbon_budget_reg(n,tt)$(reg(n) and years_budget(tt))..
    #   sum(t$(year(t)>2020 and year(t)<year(tt)), E(t,n,'co2'))*tstep
    #   + 3.5*E('2',n,'co2') + 2.5*E(tt,n,'co2')
    #   =L= cbudget * burden_share(n)
    # Reformulated as recursive per-region cumulative emissions.
    # ------------------------------------------------------------------
    if policy == "cbudget_regional":
        E = v["E"]

        # Compute burden_share based on cfg.burden setting
        region_names = [r for r in params["par_pop"].records["n"].unique()]
        par_pop_p = params["par_pop"]

        burden_shares = {}
        if cfg.burden == "equal_per_capita":
            # burden_share(n) = sum(t<=2100, pop(t,n)) / sum(t<=2100, sum(nn, pop(t,nn)))
            total_pop = 0
            region_pop = {r: 0.0 for r in region_names}
            for t in range(1, min(cfg.T + 1, 19)):  # t=1..18 -> year <= 2100
                for r in region_names:
                    p_val = params["par_pop"].records
                    p_row = p_val[(p_val["t"] == str(t)) & (p_val["n"] == r)]
                    if len(p_row) > 0:
                        # Use positional access for GAMSPy version robustness
                        pv = p_row.iloc[0, -1]  # last column = value
                        region_pop[r] += pv
                        total_pop += pv
            for r in region_names:
                burden_shares[r] = region_pop[r] / total_pop if total_pop > 0 else 1.0 / len(region_names)

        elif cfg.burden == "grandfathering":
            # burden_share(n) = emi_bau('1',n,'co2') / sum(nn, emi_bau('1',nn,'co2'))
            par_emi_bau = params["par_emi_bau"]
            total_emi = 0
            region_emi = {r: 0.0 for r in region_names}
            if hasattr(par_emi_bau, 'records') and par_emi_bau.records is not None:
                for _, row in par_emi_bau.records.iterrows():
                    if str(row.iloc[0]) == "1" and str(row.iloc[2]).lower() == "co2":
                        r = str(row.iloc[1])
                        v_val = float(row.iloc[3])
                        region_emi[r] = v_val
                        total_emi += v_val
            for r in region_names:
                burden_shares[r] = region_emi.get(r, 0) / total_emi if total_emi > 0 else 1.0 / len(region_names)

        elif cfg.burden == "historical_responsibility":
            # Proxy for historical responsibility: use cumulative BAU emissions
            # at t=1..2 as a stand-in for GAMS PRIMAP data (1960-2010).
            # GAMS: carbon_debt weighted average with population.
            # Here we combine each region's share of early-period BAU CO2
            # emissions as a proxy for historical cumulative emissions.
            par_emi_bau = params["par_emi_bau"]
            total_emi = 0
            region_emi = {r: 0.0 for r in region_names}
            if hasattr(par_emi_bau, 'records') and par_emi_bau.records is not None:
                for _, row in par_emi_bau.records.iterrows():
                    t_str = str(row.iloc[0])
                    ghg_str = str(row.iloc[2]).lower()
                    if t_str in ("1", "2") and ghg_str == "co2":
                        r = str(row.iloc[1])
                        v_val = float(row.iloc[3])
                        region_emi[r] = region_emi.get(r, 0) + v_val
                        total_emi += v_val
            for r in region_names:
                burden_shares[r] = region_emi.get(r, 0) / total_emi if total_emi > 0 else 1.0 / len(region_names)

        elif cfg.burden == "cost_efficiency":
            # cost_efficiency: very large share (makes regional constraint non-binding,
            # only the global ctax constraint binds)
            for r in region_names:
                burden_shares[r] = 1e4

        else:
            # Unknown burden type: default to equal shares
            for r in region_names:
                burden_shares[r] = 1.0 / len(region_names)

        par_burden = Parameter(m, name="burden_share", domain=[n_set],
                               records=[(r, burden_shares.get(r, 1e4))
                                        for r in region_names])

        # Per-region cumulative emissions variable
        CUMEMI_REG = Variable(m, name="CUMEMI_REG",
                              domain=[t_set, n_set], type="positive")
        CUMEMI_REG.l[t_set, n_set] = 0
        CUMEMI_REG.fx["1", n_set] = 0
        CUMEMI_REG.fx["2", n_set] = 0
        CUMEMI_REG.fx["3", n_set] = 0

        eq_cumemi_reg = Equation(m, name="eq_cumemi_reg",
                                 domain=[t_set, n_set])
        eq_cumemi_reg[t_set, n_set].where[Ord(t_set) >= 4] = (
            CUMEMI_REG[t_set, n_set] == CUMEMI_REG[t_set.lag(1), n_set]
            + cfg.TSTEP * E[t_set.lag(1), n_set, "co2"]
        )
        equations.append(eq_cumemi_reg)

        eq_cbudget_reg = Equation(m, name="eq_cbudget_reg",
                                  domain=[t_set, n_set], type="leq")
        eq_cbudget_reg[t_set, n_set].where[Ord(t_set) >= 18] = (
            3.5 * E["2", n_set, "co2"]
            + CUMEMI_REG[t_set, n_set]
            + 2.5 * E[t_set, n_set, "co2"]
            <= cfg.cbudget * par_burden[n_set]
        )
        equations.append(eq_cbudget_reg)

        v["CUMEMI_REG"] = CUMEMI_REG

    return equations
