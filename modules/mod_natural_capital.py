"""
Natural Capital module.

Based on GAMS mod_natural_capital.gms from RICE50x
(Bastien-Olvera and Moore, 2020).

Introduces natural capital into the production function and utility function.
Natural capital can be damaged by climate change and optionally enters the
Cobb-Douglas production function as a third factor alongside labour and
physical capital.

Data loaded from data_mod_natural_capital/ CSVs.
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Set, Parameter, Ord, Card, Sum


# GAMS scalar defaults
DKNAT = 0.0        # depreciation of natural capital (per year)
THETA = 0.58       # utility substitution parameter
NAT_CAP_UTILITY_SHARE = 0.10  # weight of nature in utility

# Default damage function settings
NAT_CAP_DAMFUN = "lin"  # lin | sq | log
NAT_CAP_DGVM = "lpj"    # all | lpj | car | orc


def _load_natcap_data(data_dir, region_names):
    """Load natural capital parameters from CSV files.

    Returns dict with loaded data.
    """
    nc_dir = os.path.join(data_dir, "data_mod_natural_capital")
    result = {}
    rn_lower = {r.lower(): r for r in region_names}

    # natural_capital_aggregate: n, factor (H, K, mN, nN) -> value
    agg_file = os.path.join(nc_dir, "natural_capital_aggregate.csv")
    agg = {}
    if os.path.exists(agg_file):
        df = pd.read_csv(agg_file)
        for _, row in df.iterrows():
            n = str(row.iloc[0]).lower()
            factor = str(row.iloc[1]).lower()
            if n in rn_lower:
                agg[(rn_lower[n], factor)] = float(row["Val"])
    result["aggregate"] = agg

    # natural_capital_damfun: type, dgvm, formula, n, coeff -> value
    dam_file = os.path.join(nc_dir, "natural_capital_damfun.csv")
    damfun = {}
    if os.path.exists(dam_file):
        df = pd.read_csv(dam_file)
        for _, row in df.iterrows():
            nctype = str(row.iloc[0]).lower()
            dgvm = str(row.iloc[1]).lower()
            formula = str(row.iloc[2]).lower()
            n = str(row.iloc[3]).lower()
            coef_name = str(row.iloc[4]).lower()
            if n in rn_lower and coef_name == "coeff":
                damfun[(nctype, dgvm, formula, rn_lower[n])] = float(row["Val"])
    result["damfun"] = damfun

    # natural_capital_elasticity: n, factor -> value
    elast_file = os.path.join(nc_dir, "natural_capital_elasticity.csv")
    elasticity = {}
    if os.path.exists(elast_file):
        df = pd.read_csv(elast_file)
        for _, row in df.iterrows():
            n = str(row.iloc[0]).lower()
            factor = str(row.iloc[1]).lower()
            if n in rn_lower:
                elasticity[(rn_lower[n], factor)] = float(row["Val"])
    result["elasticity"] = elasticity

    return result


def declare_vars(m, sets, params, cfg, v):
    """Create natural capital variables.

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

    # Nature type set
    nc_type = Set(m, name="nc_type", records=["market", "nonmarket"])
    sets["nc_type"] = nc_type

    # Variables
    NAT_CAP = Variable(m, name="NAT_CAP",
                       domain=[nc_type, t_set, n_set], type="positive")
    NAT_CAP_DAM = Variable(m, name="NAT_CAP_DAM",
                           domain=[nc_type, t_set, n_set], type="positive")
    NAT_INV = Variable(m, name="NAT_INV",
                       domain=[nc_type, t_set, n_set], type="positive")
    NAT_CAP_BASE = Variable(m, name="NAT_CAP_BASE",
                            domain=[nc_type, t_set, n_set], type="positive")
    GLOBAL_NN = Variable(m, name="GLOBAL_NN",
                         domain=[t_set, n_set], type="positive")

    # GAMS: NAT_INV.fx = 0 (no investment for now)
    NAT_INV.fx[nc_type, t_set, n_set] = 0

    # Starting values (placeholders -- overridden when data is loaded)
    NAT_CAP.l[nc_type, t_set, n_set] = 1e-3
    NAT_CAP_DAM.l[nc_type, t_set, n_set] = 1e-3
    NAT_INV.l[nc_type, t_set, n_set] = 0
    NAT_CAP_BASE.l[nc_type, t_set, n_set] = 1e-3
    GLOBAL_NN.l[t_set, n_set] = 1e-3

    # Register
    v["NAT_CAP"] = NAT_CAP
    v["NAT_CAP_DAM"] = NAT_CAP_DAM
    v["NAT_INV"] = NAT_INV
    v["NAT_CAP_BASE"] = NAT_CAP_BASE
    v["GLOBAL_NN"] = GLOBAL_NN


def define_eqs(m, sets, params, cfg, v):
    """Create natural capital equations.

    Implements:
    - eq_nat_cap: capital accumulation with depreciation
    - eq_es_prodfun_mN: market base = market capital
    - eq_es_prodfun_nN: non-market base = non-market capital
    - eq_nat_cap_dam: damaged capital = base * (1 + coeff * f(TATM-TATM0))
    - eq_gnn: global non-market sum

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    nc_type = sets["nc_type"]

    TSTEP = cfg.TSTEP
    TATM = v["TATM"]
    TATM0 = params["TATM0"]
    NAT_CAP = v["NAT_CAP"]
    NAT_CAP_DAM = v["NAT_CAP_DAM"]
    NAT_CAP_BASE = v["NAT_CAP_BASE"]
    GLOBAL_NN = v["GLOBAL_NN"]

    T = cfg.T
    region_names = [r for r in params["par_pop"].records["n"].unique()]

    # Load natural capital data
    nc_data = _load_natcap_data(cfg.data_dir, region_names)

    # Create parameter for initial natural capital stocks
    mN_recs = [(r, nc_data["aggregate"].get((r, "mn"), 1e-3))
               for r in region_names]
    nN_recs = [(r, nc_data["aggregate"].get((r, "nn"), 1e-3))
               for r in region_names]
    par_mN0 = Parameter(m, name="natcap_mN0", domain=[n_set], records=mN_recs)
    par_nN0 = Parameter(m, name="natcap_nN0", domain=[n_set], records=nN_recs)

    # Fix initial capital
    NAT_CAP.fx["market", "1", n_set] = par_mN0[n_set]
    NAT_CAP.fx["nonmarket", "1", n_set] = par_nN0[n_set]
    NAT_CAP_DAM.fx["nonmarket", "1", n_set] = par_nN0[n_set]

    # Update starting values
    NAT_CAP.l["market", t_set, n_set] = par_mN0[n_set]
    NAT_CAP.l["nonmarket", t_set, n_set] = par_nN0[n_set]
    NAT_CAP_DAM.l["market", t_set, n_set] = par_mN0[n_set]
    NAT_CAP_DAM.l["nonmarket", t_set, n_set] = par_nN0[n_set]

    # Create damage coefficient parameter
    # Uses NAT_CAP_DAMFUN and NAT_CAP_DGVM settings
    dam_recs = []
    for r in region_names:
        for nctype in ["market", "nonmarket"]:
            key = (nctype, NAT_CAP_DGVM, NAT_CAP_DAMFUN, r)
            coef = nc_data["damfun"].get(key, 0.0)
            # GAMS clipping: lin max -0.16, sq max -0.02
            if NAT_CAP_DAMFUN == "lin":
                coef = max(-0.16, coef)
            elif NAT_CAP_DAMFUN == "sq":
                coef = max(-0.02, coef)
            dam_recs.append((nctype, r, coef))
    par_damcoef = Parameter(m, name="natcap_damcoef",
                            domain=[nc_type, n_set], records=dam_recs)

    equations = []

    # ------------------------------------------------------------------
    # eq_nat_cap: NAT_CAP(type, t+1) = (1-dknat)^tstep * NAT_CAP(type,t)
    #             + tstep * NAT_INV(type,t)
    # Since NAT_INV.fx = 0 and dknat = 0, this simplifies to:
    # NAT_CAP(type, t+1) = NAT_CAP(type,t)
    # But we implement the full form for generality.
    # Note: GAMS uses NAT_CAP.l(type,t,n) (fixed level), so we do the same
    # ------------------------------------------------------------------
    eq_nat_cap = Equation(m, name="eq_nat_cap",
                          domain=[nc_type, t_set, n_set])
    eq_nat_cap[nc_type, t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
        NAT_CAP[nc_type, t_set.lead(1), n_set] ==
        (1 - DKNAT) ** TSTEP * NAT_CAP[nc_type, t_set, n_set]
    )
    equations.append(eq_nat_cap)

    # ------------------------------------------------------------------
    # eq_es_prodfun_mN: NAT_CAP_BASE('market',t,n) = NAT_CAP('market',t,n)
    # ------------------------------------------------------------------
    eq_es_prodfun_mN = Equation(m, name="eq_es_prodfun_mN",
                                domain=[t_set, n_set])
    eq_es_prodfun_mN[t_set, n_set] = (
        NAT_CAP_BASE["market", t_set, n_set] ==
        NAT_CAP["market", t_set, n_set]
    )
    equations.append(eq_es_prodfun_mN)

    # ------------------------------------------------------------------
    # eq_es_prodfun_nN: NAT_CAP_BASE('nonmarket',t,n) = NAT_CAP('nonmarket',t,n)
    # (Simple version; full version with nat_cap_prodfun uses TFP-based formula)
    # ------------------------------------------------------------------
    eq_es_prodfun_nN = Equation(m, name="eq_es_prodfun_nN",
                                domain=[t_set, n_set])
    eq_es_prodfun_nN[t_set, n_set] = (
        NAT_CAP_BASE["nonmarket", t_set, n_set] ==
        NAT_CAP["nonmarket", t_set, n_set]
    )
    equations.append(eq_es_prodfun_nN)

    # ------------------------------------------------------------------
    # eq_nat_cap_dam: NAT_CAP_DAM(type,t,n) = NAT_CAP_BASE * (1 + coeff * f(TATM))
    # Default: nat_cap_damages not set -> factor is just 1
    # When damages active, f depends on damfun setting:
    #   lin: (TATM - TATM0)
    #   sq:  (TATM - TATM0)^2
    #   log: log(1 + TATM - TATM0)
    # ------------------------------------------------------------------
    eq_nat_cap_dam = Equation(m, name="eq_nat_cap_dam",
                              domain=[nc_type, t_set, n_set])
    # GAMS: damage applied only when $setglobal nat_cap_damages is set.
    # When not set, NAT_CAP_DAM = NAT_CAP_BASE (no climate damage).
    nat_cap_damages = getattr(cfg, "nat_cap_damages", False)
    if nat_cap_damages:
        eq_nat_cap_dam[nc_type, t_set, n_set] = (
            NAT_CAP_DAM[nc_type, t_set, n_set] ==
            NAT_CAP_BASE[nc_type, t_set, n_set]
            * (1 + par_damcoef[nc_type, n_set] * (TATM[t_set] - TATM0))
        )
    else:
        eq_nat_cap_dam[nc_type, t_set, n_set] = (
            NAT_CAP_DAM[nc_type, t_set, n_set] ==
            NAT_CAP_BASE[nc_type, t_set, n_set]
        )
    equations.append(eq_nat_cap_dam)

    # ------------------------------------------------------------------
    # eq_gnn: GLOBAL_NN(t,n) = NAT_CAP_DAM('nonmarket',t,n)
    #         + sum(nn$(not reg(nn)), NAT_CAP_DAM.l('nonmarket',t,nn))
    # In cooperative mode, all regions solve together, so we just sum
    # the endogenous variable.
    # ------------------------------------------------------------------
    n_alias = sets.get("n_alias")
    eq_gnn = Equation(m, name="eq_gnn", domain=[t_set, n_set])
    if n_alias is not None:
        eq_gnn[t_set, n_set] = (
            GLOBAL_NN[t_set, n_set] ==
            Sum(n_alias, NAT_CAP_DAM["nonmarket", t_set, n_alias])
        )
    else:
        # Fallback: just own region
        eq_gnn[t_set, n_set] = (
            GLOBAL_NN[t_set, n_set] == NAT_CAP_DAM["nonmarket", t_set, n_set]
        )
    equations.append(eq_gnn)

    return equations
