"""
Decile-disaggregated damage module.

Based on GAMS mod_impact_deciles.gms.
Requires cfg.inequality=True (mod_inequality must be loaded first).

Computes per-decile damages using BHM-style growth-rate impact with
decile-specific coefficients. DAMAGES_DIST feeds into the inequality
module's eq_ynetdist instead of the standard DAMAGES → DAMFRAC path.

Variables:
    BIMPACT(t,n,dist)          -- per-decile yearly impact
    YNET_UNBOUNDED(t,n,dist)   -- unbounded per-decile GDP net of damages
    YNET_UPBOUND(t,n,dist)     -- upper-bounded per-decile GDP
    YNET_ESTIMATED(t,n,dist)   -- final bounded per-decile GDP
    DAMAGES_DIST(t,n,dist)     -- per-decile damages
    DAMFRAC_DIST(t,n,dist)     -- per-decile damage fraction

Equations:
    eq_bimpact     -- BHM decile-varying yearly impact
    eq_ynet_nobnd  -- recursive growth with impact
    eq_ynet_upbnd  -- upper bound (damage cap)
    eq_ynet_estim  -- lower bound (damage cap)
    eq_damages     -- DAMAGES_DIST = YGROSS_DIST - YNET_ESTIMATED
    eq_damfrac     -- DAMFRAC_DIST = DAMAGES_DIST / YGROSS_DIST
    eq_damagestot  -- DAMAGES = sum(dist, DAMAGES_DIST)
"""

from gamspy import Variable, Equation, Parameter, Ord, Card, Sum, Number
from gamspy.math import sqrt, sqr, log


# Beta coefficients by decile (GAMS mod_impact_deciles.gms lines 45-49)
BETA_DECILES = {
    "T":     [0.1657, 0.1390, 0.1375, 0.1356, 0.1304, 0.1281, 0.1275, 0.1288, 0.1278, 0.1180],
    "T2":    [-0.0045, -0.0032, -0.0033, -0.0033, -0.0032, -0.0031, -0.0031, -0.0031, -0.0031, -0.0028],
    "TxGDP": [-0.0152, -0.0124, -0.0126, -0.0124, -0.0122, -0.0120, -0.0120, -0.0122, -0.0121, -0.0107],
    "T2xGDP":[0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003],
}

DIST_NAMES = [f"D{i}" for i in range(1, 11)]


def declare_vars(m, sets, params, cfg, v):
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    dist_set = sets["dist_set"]

    BIMPACT = Variable(m, name="BIMPACT", domain=[t_set, n_set, dist_set])
    YNET_UNBOUNDED_D = Variable(m, name="YNET_UNBOUNDED_D",
                                 domain=[t_set, n_set, dist_set])
    YNET_UPBOUND_D = Variable(m, name="YNET_UPBOUND_D",
                               domain=[t_set, n_set, dist_set])
    YNET_ESTIMATED_D = Variable(m, name="YNET_ESTIMATED_D",
                                 domain=[t_set, n_set, dist_set], type="positive")
    DAMAGES_DIST = Variable(m, name="DAMAGES_DIST",
                            domain=[t_set, n_set, dist_set])
    DAMFRAC_DIST = Variable(m, name="DAMFRAC_DIST",
                            domain=[t_set, n_set, dist_set])

    BIMPACT.lo[t_set, n_set, dist_set] = -1 + 1e-6
    BIMPACT.l[t_set, n_set, dist_set] = 0
    # GAMS: BIMPACT.fx(t,n,dist)$(not t_damages(t)) = 0
    # t_damages = {t : tperiod(t) > 2 and tperiod(t) < T-20}
    # i.e. active for t=3..T-21. Fix periods 1, 2, and t > T-20 to 0.
    T = cfg.T
    max_t_dam = T - 20
    for t in range(1, T + 1):
        if t <= 2 or t > max_t_dam:
            BIMPACT.fx[str(t), n_set, dist_set] = 0

    # Starting values from ykali * quantiles_ref
    par_ykali = params["par_ykali"]
    par_qref = params.get("par_quantiles_ref")
    if par_qref is not None:
        YNET_UNBOUNDED_D.l[t_set, n_set, dist_set] = par_ykali[t_set, n_set] * par_qref[t_set, n_set, dist_set]
        YNET_UNBOUNDED_D.fx["1", n_set, dist_set] = par_ykali["1", n_set] * par_qref["1", n_set, dist_set]
    YNET_ESTIMATED_D.l[t_set, n_set, dist_set] = YNET_UNBOUNDED_D.l[t_set, n_set, dist_set]
    DAMAGES_DIST.l[t_set, n_set, dist_set] = 0
    DAMFRAC_DIST.l[t_set, n_set, dist_set] = 0

    v["BIMPACT"] = BIMPACT
    v["YNET_UNBOUNDED_D"] = YNET_UNBOUNDED_D
    v["YNET_UPBOUND_D"] = YNET_UPBOUND_D
    v["YNET_ESTIMATED_D"] = YNET_ESTIMATED_D
    v["DAMAGES_DIST"] = DAMAGES_DIST
    v["DAMFRAC_DIST"] = DAMFRAC_DIST

    # Also create the standard OMEGA variable (set to 0, not used)
    # needed by hub_impact
    if "OMEGA" not in v:
        OMEGA = Variable(m, name="OMEGA", domain=[t_set, n_set])
        OMEGA.fx[t_set, n_set] = 0
        v["OMEGA"] = OMEGA


def define_eqs(m, sets, params, cfg, v):
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    dist_set = sets["dist_set"]
    T = cfg.T
    TSTEP = cfg.TSTEP

    BIMPACT = v["BIMPACT"]
    YNET_UNBOUNDED_D = v["YNET_UNBOUNDED_D"]
    YNET_UPBOUND_D = v["YNET_UPBOUND_D"]
    YNET_ESTIMATED_D = v["YNET_ESTIMATED_D"]
    DAMAGES_DIST = v["DAMAGES_DIST"]
    DAMFRAC_DIST = v["DAMFRAC_DIST"]
    DAMAGES = v["DAMAGES"]
    YGROSS_DIST = v["YGROSS_DIST"]
    TEMP_REGION_DAM = v["TEMP_REGION_DAM"]

    par_pop = params["par_pop"]
    par_ykali = params["par_ykali"]
    par_qref = params.get("par_quantiles_ref")

    region_names = [r for r in par_pop.records["n"].unique()]

    # Temperature reference: GAMS uses TEMP_REGION.l('1',n) (first-period level)
    TEMP_REGION = v.get("TEMP_REGION")
    temp_ref_recs = []
    if TEMP_REGION is not None and TEMP_REGION.records is not None:
        for _, row in TEMP_REGION.records.iterrows():
            if str(row.iloc[0]) == "1":
                temp_ref_recs.append((str(row.iloc[1]), float(row["level"])))
    if not temp_ref_recs:
        # Fallback to par_base_temp
        par_base_temp = params["par_base_temp"]
        for r in region_names:
            bt = par_base_temp.records[par_base_temp.records.iloc[:, 0] == r]
            if len(bt) > 0:
                temp_ref_recs.append((r, float(bt.iloc[0, -1])))
    par_temp_ref = Parameter(m, name="temp_region_reference",
                             domain=[n_set], records=temp_ref_recs)
    # Quadratic term uses base_temp (regression intercept), not TEMP_REGION.l
    # GAMS line 130: power(climate_region_coef('base_temp',n), 2)
    par_base_temp = params["par_base_temp"]

    # Create beta_deciles parameters
    beta_recs_T = [(d, BETA_DECILES["T"][i]) for i, d in enumerate(DIST_NAMES)]
    beta_recs_T2 = [(d, BETA_DECILES["T2"][i]) for i, d in enumerate(DIST_NAMES)]
    beta_recs_TxGDP = [(d, BETA_DECILES["TxGDP"][i]) for i, d in enumerate(DIST_NAMES)]
    beta_recs_T2xGDP = [(d, BETA_DECILES["T2xGDP"][i]) for i, d in enumerate(DIST_NAMES)]

    par_beta_T = Parameter(m, name="beta_dec_T", domain=[dist_set], records=beta_recs_T)
    par_beta_T2 = Parameter(m, name="beta_dec_T2", domain=[dist_set], records=beta_recs_T2)
    par_beta_TxGDP = Parameter(m, name="beta_dec_TxGDP", domain=[dist_set], records=beta_recs_TxGDP)
    par_beta_T2xGDP = Parameter(m, name="beta_dec_T2xGDP", domain=[dist_set], records=beta_recs_T2xGDP)

    # GDP per capita (baseline) parameter for the log(gdppc) interaction
    gdppc_recs = []
    for t in range(1, T + 1):
        for r in region_names:
            yk = par_ykali.records
            pp = par_pop.records
            yk_val = yk[(yk.iloc[:, 0] == str(t)) & (yk.iloc[:, 1] == r)]
            pp_val = pp[(pp.iloc[:, 0] == str(t)) & (pp.iloc[:, 1] == r)]
            if len(yk_val) > 0 and len(pp_val) > 0:
                gdppc = yk_val.iloc[0, -1] * 1e6 / max(pp_val.iloc[0, -1], 1e-6)
                gdppc_recs.append((str(t), r, max(gdppc, 1.0)))
    par_gdppc = Parameter(m, name="gdppc_kali_dec", domain=[t_set, n_set],
                          records=gdppc_recs)

    # basegrowthcap_dist: per-decile baseline growth
    bgc_recs = []
    if par_qref is not None and par_qref.records is not None:
        qref_dict = {}
        for _, row in par_qref.records.iterrows():
            qref_dict[(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = float(row.iloc[3])
        ykali_dict = {}
        for _, row in par_ykali.records.iterrows():
            ykali_dict[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])
        pop_dict_local = {}
        for _, row in par_pop.records.iterrows():
            pop_dict_local[(str(row.iloc[0]), str(row.iloc[1]))] = float(row.iloc[2])

        for t in range(1, T):
            tp1 = t + 1
            for r in region_names:
                for d in DIST_NAMES:
                    yk_t = ykali_dict.get((str(t), r), 0)
                    yk_tp1 = ykali_dict.get((str(tp1), r), 0)
                    qr_t = qref_dict.get((str(t), r, d), 0.1)
                    qr_tp1 = qref_dict.get((str(tp1), r, d), 0.1)
                    p_t = pop_dict_local.get((str(t), r), 1)
                    p_tp1 = pop_dict_local.get((str(tp1), r), 1)
                    num = yk_tp1 * qr_tp1 / max(p_tp1, 1e-6)
                    den = yk_t * qr_t / max(p_t, 1e-6)
                    if den > 0 and num > 0:
                        bgc = (num / den) ** (1 / TSTEP) - 1
                    else:
                        bgc = 0.0
                    bgc_recs.append((str(t), r, d, bgc))
    par_bgc = Parameter(m, name="basegrowthcap_dist",
                        domain=[t_set, n_set, dist_set], records=bgc_recs)

    # t_damages: active periods (t > 2 and t < T-20)
    max_t_dam = T - 20
    delta = 1e-8

    equations = []

    # eq_bimpact: per-decile BHM impact
    eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set, dist_set])
    eq_bimpact[t_set, n_set, dist_set].where[
        (Ord(t_set) > 2) & (Ord(t_set) <= max_t_dam)
    ] = (
        BIMPACT[t_set, n_set, dist_set] ==
        (par_beta_T[dist_set] + par_beta_TxGDP[dist_set] * log(1.3 * par_gdppc[t_set, n_set]))
        * (TEMP_REGION_DAM[t_set, n_set] - par_temp_ref[n_set])
        + (par_beta_T2[dist_set] + par_beta_T2xGDP[dist_set] * log(1.3 * par_gdppc[t_set, n_set]))
        * (TEMP_REGION_DAM[t_set, n_set] ** 2 - par_base_temp[n_set] ** 2)
    )
    equations.append(eq_bimpact)

    # eq_ynet_nobnd: recursive growth
    eq_ynet_nobnd = Equation(m, name="eq_ynet_nobnd",
                             domain=[t_set, n_set, dist_set])
    eq_ynet_nobnd[t_set, n_set, dist_set].where[
        (Ord(t_set) > 1) & (Ord(t_set) < Card(t_set))
    ] = (
        YNET_UNBOUNDED_D[t_set.lead(1), n_set, dist_set]
        * par_pop[t_set, n_set] / par_pop[t_set.lead(1), n_set]
        == YNET_UNBOUNDED_D[t_set, n_set, dist_set]
        * (1 + par_bgc[t_set, n_set, dist_set] + BIMPACT[t_set, n_set, dist_set]) ** TSTEP
    )
    equations.append(eq_ynet_nobnd)

    # Damage cap
    if cfg.damage_cap:
        max_gain = getattr(cfg, "max_gain", 1.1)
        max_damage = getattr(cfg, "max_damage", 1e-5)

        # ynet_maximum/minimum as parameters
        ymax_recs = []
        ymin_recs = []
        if par_qref is not None:
            for t in range(1, T + 1):
                for r in region_names:
                    yk = ykali_dict.get((str(t), r), 0)
                    for d in DIST_NAMES:
                        qr = qref_dict.get((str(t), r, d), 0.1)
                        ymax_recs.append((str(t), r, d, max_gain * yk * qr))
                        ymin_recs.append((str(t), r, d, max_damage * yk * qr))

        par_ymax = Parameter(m, name="ynet_maximum",
                             domain=[t_set, n_set, dist_set], records=ymax_recs)
        par_ymin = Parameter(m, name="ynet_minimum",
                             domain=[t_set, n_set, dist_set], records=ymin_recs)

        eq_ynet_upbnd = Equation(m, name="eq_ynet_upbnd",
                                 domain=[t_set, n_set, dist_set])
        eq_ynet_upbnd[t_set, n_set, dist_set] = (
            YNET_UPBOUND_D[t_set, n_set, dist_set] == (
                YNET_UNBOUNDED_D[t_set, n_set, dist_set] + par_ymax[t_set, n_set, dist_set]
                - sqrt(sqr(YNET_UNBOUNDED_D[t_set, n_set, dist_set] - par_ymax[t_set, n_set, dist_set])
                       + sqr(Number(delta)))
            ) / 2
        )
        equations.append(eq_ynet_upbnd)

        eq_ynet_estim = Equation(m, name="eq_ynet_estim",
                                 domain=[t_set, n_set, dist_set])
        eq_ynet_estim[t_set, n_set, dist_set] = (
            YNET_ESTIMATED_D[t_set, n_set, dist_set] == (
                YNET_UPBOUND_D[t_set, n_set, dist_set] + par_ymin[t_set, n_set, dist_set]
                + sqrt(sqr(YNET_UPBOUND_D[t_set, n_set, dist_set] - par_ymin[t_set, n_set, dist_set])
                       + sqr(Number(delta)))
            ) / 2
        )
        equations.append(eq_ynet_estim)

        eq_damages = Equation(m, name="eq_damages_dist",
                              domain=[t_set, n_set, dist_set])
        eq_damages[t_set, n_set, dist_set] = (
            DAMAGES_DIST[t_set, n_set, dist_set] ==
            YGROSS_DIST[t_set, n_set, dist_set] - YNET_ESTIMATED_D[t_set, n_set, dist_set]
        )
        equations.append(eq_damages)
    else:
        eq_damages = Equation(m, name="eq_damages_dist",
                              domain=[t_set, n_set, dist_set])
        eq_damages[t_set, n_set, dist_set] = (
            DAMAGES_DIST[t_set, n_set, dist_set] ==
            YGROSS_DIST[t_set, n_set, dist_set] - YNET_UNBOUNDED_D[t_set, n_set, dist_set]
        )
        equations.append(eq_damages)

    # eq_damfrac
    eq_damfrac = Equation(m, name="eq_damfrac_dist",
                          domain=[t_set, n_set, dist_set])
    eq_damfrac[t_set, n_set, dist_set] = (
        DAMFRAC_DIST[t_set, n_set, dist_set] ==
        DAMAGES_DIST[t_set, n_set, dist_set] / YGROSS_DIST[t_set, n_set, dist_set]
    )
    equations.append(eq_damfrac)

    # eq_damagestot: DAMAGES = sum(dist, DAMAGES_DIST)
    eq_damagestot = Equation(m, name="eq_damagestot", domain=[t_set, n_set])
    eq_damagestot[t_set, n_set] = (
        DAMAGES[t_set, n_set] == Sum(dist_set, DAMAGES_DIST[t_set, n_set, dist_set])
    )
    equations.append(eq_damagestot)

    return equations
