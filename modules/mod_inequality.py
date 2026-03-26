"""
Inequality module.

Based on GAMS mod_inequality.gms from RICE50x.

Tracks within-country income distribution across deciles, allocating climate
damages and abatement costs according to income-dependent weights. Optionally
modifies the welfare function to account for within-country inequality aversion.

Data loaded from data_mod_inequality_converted/ CSVs.
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Set, Parameter, Ord, Card, Sum


# GAMS scalar defaults
GAMMAINT = 0.50       # within-country inequality aversion (gammaint/100)
OMEGA_EL = 0.5        # elasticity of abatement cost distribution (omega/10)
XI_EL = 0.85          # elasticity of damage distribution (xi/100)
EL_REDIST = 0.0       # elasticity of redistribution (el_redist/10)
SUBSISTANCE_LEVEL = 273.3  # half of 1.9 USD/day in 2005 USD/yr


def _load_quantiles(data_dir, ssp, region_names, T):
    """Load quantile shares from data_mod_inequality_converted/quantiles.csv.

    Returns dict: (ssp, t, region, dist) -> share value.
    """
    ineq_dir = os.path.join(data_dir, "data_mod_inequality_converted")
    q_file = os.path.join(ineq_dir, "quantiles.csv")

    quantiles = {}
    rn_lower = {r.lower(): r for r in region_names}

    if os.path.exists(q_file):
        df = pd.read_csv(q_file)
        # CSV format varies; try to parse adaptively
        cols = list(df.columns)
        for _, row in df.iterrows():
            try:
                ssp_val = str(row.iloc[0]).lower()
                t_val = int(row.iloc[1])
                n_val = str(row.iloc[2]).lower()
                dist_val = str(row.iloc[3]).upper()
                val = float(row.iloc[4]) if len(cols) > 4 else float(row["Val"])

                if ssp_val != ssp.lower():
                    continue
                if n_val not in rn_lower or t_val < 1 or t_val > T:
                    continue
                quantiles[(t_val, rn_lower[n_val], dist_val)] = val
            except (ValueError, IndexError):
                continue

    return quantiles


def declare_vars(m, sets, params, cfg, v):
    """Create inequality variables.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # Decile set
    dist_set = Set(m, name="dist", records=[f"D{i}" for i in range(1, 11)])
    sets["dist_set"] = dist_set

    # Variables
    YGROSS_DIST = Variable(m, name="YGROSS_DIST",
                           domain=[t_set, n_set, dist_set])
    YNET_DIST = Variable(m, name="YNET_DIST",
                         domain=[t_set, n_set, dist_set])
    Y_DIST_PRE = Variable(m, name="Y_DIST_PRE",
                          domain=[t_set, n_set, dist_set])
    Y_DIST = Variable(m, name="Y_DIST",
                      domain=[t_set, n_set, dist_set])
    CPC_DIST = Variable(m, name="CPC_DIST",
                        domain=[t_set, n_set, dist_set])
    TRANSFER = Variable(m, name="TRANSFER",
                        domain=[t_set, n_set, dist_set])

    # CTX: total carbon tax cost (from GAMS mod_inequality eq_ctx)
    CTX = Variable(m, name="CTX", domain=[t_set, n_set])

    # Bounds
    TRANSFER.lo[t_set, n_set, dist_set] = 0
    YGROSS_DIST.lo[t_set, n_set, dist_set] = 0
    Y_DIST.lo[t_set, n_set, dist_set] = 0
    CPC_DIST.lo[t_set, n_set, dist_set] = 1e-3

    # Starting values (uniform deciles)
    par_ykali = params["par_ykali"]
    YGROSS_DIST.l[t_set, n_set, dist_set] = 0.1 * par_ykali[t_set, n_set]
    YNET_DIST.l[t_set, n_set, dist_set] = 0.1 * par_ykali[t_set, n_set]
    Y_DIST_PRE.l[t_set, n_set, dist_set] = 0.1 * par_ykali[t_set, n_set]
    Y_DIST.l[t_set, n_set, dist_set] = 0.1 * par_ykali[t_set, n_set]
    CPC_DIST.l[t_set, n_set, dist_set] = 1e3
    TRANSFER.l[t_set, n_set, dist_set] = 0
    CTX.l[t_set, n_set] = 0

    # Register
    v["YGROSS_DIST"] = YGROSS_DIST
    v["YNET_DIST"] = YNET_DIST
    v["Y_DIST_PRE"] = Y_DIST_PRE
    v["Y_DIST"] = Y_DIST
    v["CPC_DIST"] = CPC_DIST
    v["TRANSFER"] = TRANSFER
    v["CTX"] = CTX


def define_eqs(m, sets, params, cfg, v):
    """Create inequality equations.

    Implements:
    - eq_ygrossdist: YGROSS_DIST = quantiles_ref * YGROSS
    - eq_ynetdist_unbnd: YNET_DIST = YGROSS_DIST - DAMAGES * ineq_weights('damages')
    - eq_ydist_unbnd: Y_DIST_PRE = YNET_DIST - (ABATECOST + CTX) * ineq_weights('abatement')
    - eq_ydist: Y_DIST = Y_DIST_PRE + TRANSFER
    - eq_cpcdist: CPC_DIST = 1e6 * Y_DIST * (1-S) / (pop * quant_share)
    - eq_transfer: TRANSFER = CTX * ineq_weights('redist')
    - eq_ctx: CTX = sum(ghg, MAC * EIND * convy_ghg)

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]
    dist_set = sets["dist_set"]

    T = cfg.T
    QUANT_SHARE = 1.0 / 10.0  # 10 deciles

    YGROSS = v["YGROSS"]
    DAMAGES = v["DAMAGES"]
    ABATECOST = v["ABATECOST"]
    S = v["S"]
    MAC = v["MAC"]
    EIND = v["EIND"]
    par_pop = params["par_pop"]
    par_convy_ghg = params["par_convy_ghg"]

    YGROSS_DIST = v["YGROSS_DIST"]
    YNET_DIST = v["YNET_DIST"]
    Y_DIST_PRE = v["Y_DIST_PRE"]
    Y_DIST = v["Y_DIST"]
    CPC_DIST = v["CPC_DIST"]
    TRANSFER = v["TRANSFER"]
    CTX = v["CTX"]

    region_names = [r for r in params["par_pop"].records["n"].unique()]

    # Load quantile data
    quantiles = _load_quantiles(cfg.data_dir, cfg.SSP, region_names, T)

    # Create quantiles_ref parameter
    dist_names = [f"D{i}" for i in range(1, 11)]
    qref_recs = []
    for t in range(1, T + 1):
        for r in region_names:
            for d in dist_names:
                val = quantiles.get((t, r, d), 0.1)  # default: equal shares
                qref_recs.append((str(t), r, d, val))
    par_qref = Parameter(m, name="quantiles_ref",
                         domain=[t_set, n_set, dist_set], records=qref_recs)

    # Compute inequality weights
    # ineq_weights(t,n,dist,elast) = qref^el / sum(dd, qref^el)
    # We compute for damages, abatement, and redist elasticities
    def _compute_weights(el):
        w_recs = []
        for t in range(1, T + 1):
            for r in region_names:
                shares = []
                for d in dist_names:
                    q = quantiles.get((t, r, d), 0.1)
                    shares.append(max(q, 1e-12) ** el)
                total = sum(shares)
                if total <= 0:
                    total = 1.0
                for i, d in enumerate(dist_names):
                    w_recs.append((str(t), r, d, shares[i] / total))
        return w_recs

    par_ineq_w_dam = Parameter(m, name="ineq_w_dam",
                               domain=[t_set, n_set, dist_set],
                               records=_compute_weights(XI_EL))
    par_ineq_w_abate = Parameter(m, name="ineq_w_abate",
                                 domain=[t_set, n_set, dist_set],
                                 records=_compute_weights(OMEGA_EL))
    par_ineq_w_redist = Parameter(m, name="ineq_w_redist",
                                  domain=[t_set, n_set, dist_set],
                                  records=_compute_weights(OMEGA_EL))  # neutral: same as abatement

    equations = []

    # ------------------------------------------------------------------
    # eq_ygrossdist: YGROSS_DIST = quantiles_ref * YGROSS
    # ------------------------------------------------------------------
    eq_ygrossdist = Equation(m, name="eq_ygrossdist",
                             domain=[t_set, n_set, dist_set])
    eq_ygrossdist[t_set, n_set, dist_set] = (
        YGROSS_DIST[t_set, n_set, dist_set] ==
        par_qref[t_set, n_set, dist_set] * YGROSS[t_set, n_set]
    )
    equations.append(eq_ygrossdist)

    # ------------------------------------------------------------------
    # eq_ynetdist_unbnd: YNET_DIST = YGROSS_DIST - DAMAGES * ineq_weights('damages')
    # ------------------------------------------------------------------
    eq_ynetdist = Equation(m, name="eq_ynetdist_unbnd",
                           domain=[t_set, n_set, dist_set])
    eq_ynetdist[t_set, n_set, dist_set] = (
        YNET_DIST[t_set, n_set, dist_set] ==
        YGROSS_DIST[t_set, n_set, dist_set]
        - DAMAGES[t_set, n_set] * par_ineq_w_dam[t_set, n_set, dist_set]
    )
    equations.append(eq_ynetdist)

    # ------------------------------------------------------------------
    # eq_ctx: CTX = sum(ghg, MAC * EIND * convy_ghg)
    # ------------------------------------------------------------------
    eq_ctx = Equation(m, name="eq_ctx", domain=[t_set, n_set])
    eq_ctx[t_set, n_set] = (
        CTX[t_set, n_set] == Sum(ghg_set,
            MAC[t_set, n_set, ghg_set]
            * EIND[t_set, n_set, ghg_set]
            * par_convy_ghg[ghg_set]
        )
    )
    equations.append(eq_ctx)

    # ------------------------------------------------------------------
    # eq_ydist_unbnd: Y_DIST_PRE = YNET_DIST - (ABATECOST + CTX) * ineq_weights('abatement')
    # ------------------------------------------------------------------
    eq_ydist_unbnd = Equation(m, name="eq_ydist_unbnd",
                              domain=[t_set, n_set, dist_set])
    eq_ydist_unbnd[t_set, n_set, dist_set] = (
        Y_DIST_PRE[t_set, n_set, dist_set] ==
        YNET_DIST[t_set, n_set, dist_set]
        - (Sum(ghg_set, ABATECOST[t_set, n_set, ghg_set]) + CTX[t_set, n_set])
        * par_ineq_w_abate[t_set, n_set, dist_set]
    )
    equations.append(eq_ydist_unbnd)

    # ------------------------------------------------------------------
    # eq_ydist: Y_DIST = Y_DIST_PRE + TRANSFER
    # ------------------------------------------------------------------
    eq_ydist = Equation(m, name="eq_ydist",
                        domain=[t_set, n_set, dist_set])
    eq_ydist[t_set, n_set, dist_set] = (
        Y_DIST[t_set, n_set, dist_set] ==
        Y_DIST_PRE[t_set, n_set, dist_set] + TRANSFER[t_set, n_set, dist_set]
    )
    equations.append(eq_ydist)

    # ------------------------------------------------------------------
    # eq_cpcdist: CPC_DIST = 1e6 * Y_DIST * (1-S) / (pop * quant_share)
    # ------------------------------------------------------------------
    eq_cpcdist = Equation(m, name="eq_cpcdist",
                          domain=[t_set, n_set, dist_set])
    eq_cpcdist[t_set, n_set, dist_set] = (
        CPC_DIST[t_set, n_set, dist_set] ==
        1e6 * Y_DIST[t_set, n_set, dist_set]
        * (1 - S[t_set, n_set])
        / (par_pop[t_set, n_set] * QUANT_SHARE)
    )
    equations.append(eq_cpcdist)

    # ------------------------------------------------------------------
    # eq_transfer: TRANSFER = CTX * ineq_weights('redist')
    # ------------------------------------------------------------------
    eq_transfer = Equation(m, name="eq_transfer",
                           domain=[t_set, n_set, dist_set])
    eq_transfer[t_set, n_set, dist_set] = (
        TRANSFER[t_set, n_set, dist_set] ==
        CTX[t_set, n_set] * par_ineq_w_redist[t_set, n_set, dist_set]
    )
    equations.append(eq_transfer)

    return equations
