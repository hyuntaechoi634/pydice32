"""
Ocean capital and ecosystem services module.

Based on GAMS mod_ocean.gms from RICE50x (Bastien-Olvera et al., 2025).

Represents ocean natural capital (coral reefs, mangroves, ports, fisheries)
and their valuation under climate change. Ocean area responds to warming,
and non-market/non-use values scale with income growth.

Data is loaded from data_mod_ocean/ CSVs where available; parameters default
to zero for regions without ocean capital.
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Set, Parameter, Ord, Card, Sum
from gamspy.math import exp as gams_exp


# Scalar parameters from GAMS mod_ocean.gms
OCEAN_THETA_1 = 0.21
OCEAN_THETA_2 = 0.21
OCEAN_INCOME_ELASTICITY_USENM = 0.222
OCEAN_INCOME_ELASTICITY_NONUSE = 0.243
OCEAN_S1_1 = 1.0
OCEAN_S1_2 = 1.0
OCEAN_S2_1 = 1.0
OCEAN_S2_2 = 1.0
VSL_START = 7.4  # Million USD per capita (US)
OCEAN_HEALTH_ETA = 0.05


def _load_ocean_data(data_dir, oc_names, region_names):
    """Load ocean parameters from CSV files.

    Returns dict of parameter name -> {(oc_capital, region): value}.
    """
    ocean_dir = os.path.join(data_dir, "data_mod_ocean")
    result = {}

    param_files = {
        "area_damage_coef": "ocean_area_damage_coef.csv",
        "area_damage_coef_sq": "ocean_area_damage_coef_sq.csv",
        "area_start": "ocean_area_start.csv",
        "consump_damage_coef": "ocean_consump_damage_coef.csv",
        "consump_damage_coef_sq": "ocean_consump_damage_coef_sq.csv",
        "health_tame": "ocean_health_tame.csv",
        "health_beta": "ocean_health_beta.csv",
        "health_mu": "ocean_health_mu.csv",
        "unm_start": "ocean_unm_start.csv",
        "nu_start": "ocean_nu_start.csv",
        "value_intercept_unm": "ocean_value_intercept_unm.csv",
        "value_intercept_nu": "ocean_value_intercept_nu.csv",
        "value_exp_unm": "ocean_value_exp_unm.csv",
        "value_exp_nu": "ocean_value_exp_nu.csv",
        "value_exp_um": "ocean_value_exp_um.csv",
    }

    rn_lower = {r.lower(): r for r in region_names}

    for pname, fname in param_files.items():
        fpath = os.path.join(ocean_dir, fname)
        d = {}
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # CSV format: Dim1 (oc_capital), n (region), Val
            for _, row in df.iterrows():
                oc = str(row.iloc[0]).lower()
                n = str(row.iloc[1]).lower()
                if oc in oc_names and n in rn_lower:
                    d[(oc, rn_lower[n])] = float(row["Val"])
        result[pname] = d

    return result


def declare_vars(m, sets, params, cfg, v):
    """Create ocean module variables.

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

    # Create ocean capital set
    oc_capital = Set(m, name="oc_capital",
                     records=["coral", "mangrove", "ports", "fisheries"])
    sets["oc_capital"] = oc_capital

    # Subsets
    oc_nonmkt = Set(m, name="oc_nonmkt",
                    records=["coral", "mangrove", "fisheries"])
    oc_nonuse = Set(m, name="oc_nonuse",
                    records=["coral", "mangrove"])
    sets["oc_nonmkt"] = oc_nonmkt
    sets["oc_nonuse"] = oc_nonuse

    # Variables
    CPC_OCEAN_DAM = Variable(m, name="CPC_OCEAN_DAM",
                             domain=[t_set, n_set], type="positive")
    OCEAN_AREA = Variable(m, name="OCEAN_AREA",
                          domain=[oc_capital, t_set, n_set], type="positive")
    OCEAN_USENM_VALUE = Variable(m, name="OCEAN_USENM_VALUE",
                                 domain=[oc_capital, t_set, n_set], type="positive")
    OCEAN_USENM_VALUE_PERKM2 = Variable(m, name="OCEAN_USENM_VALUE_PERKM2",
                                        domain=[oc_capital, t_set, n_set], type="positive")
    OCEAN_NONUSE_VALUE = Variable(m, name="OCEAN_NONUSE_VALUE",
                                 domain=[oc_capital, t_set, n_set], type="positive")
    OCEAN_NONUSE_VALUE_PERKM2 = Variable(m, name="OCEAN_NONUSE_VALUE_PERKM2",
                                         domain=[oc_capital, t_set, n_set], type="positive")
    VSL = Variable(m, name="VSL", domain=[t_set, n_set], type="positive")

    # Starting values
    CPC_OCEAN_DAM.l[t_set, n_set] = 1
    OCEAN_AREA.l[oc_capital, t_set, n_set] = 1
    OCEAN_USENM_VALUE.l[oc_capital, t_set, n_set] = 1
    OCEAN_NONUSE_VALUE.l[oc_capital, t_set, n_set] = 1
    VSL.l[t_set, n_set] = VSL_START

    # Register
    v["CPC_OCEAN_DAM"] = CPC_OCEAN_DAM
    v["OCEAN_AREA"] = OCEAN_AREA
    v["OCEAN_USENM_VALUE"] = OCEAN_USENM_VALUE
    v["OCEAN_USENM_VALUE_PERKM2"] = OCEAN_USENM_VALUE_PERKM2
    v["OCEAN_NONUSE_VALUE"] = OCEAN_NONUSE_VALUE
    v["OCEAN_NONUSE_VALUE_PERKM2"] = OCEAN_NONUSE_VALUE_PERKM2
    v["VSL"] = VSL


def define_eqs(m, sets, params, cfg, v):
    """Create ocean equations.

    Implements all GAMS mod_ocean.gms equations:
      1. eq_ocean_area -- area response to warming
      2. eq_ocean_cpc -- consumption per capita with ocean damages
      3. eq_ocean_vsl -- Value of Statistical Life
      4. eq_ocean_nmuse_value_perkm2_coral -- per-km2 non-market use value (coral)
      5. eq_ocean_nmuse_value_coral -- total non-market use value (coral)
      6. eq_ocean_nonuse_value_perkm2_coral -- per-km2 non-use value (coral)
      7. eq_ocean_nonuse_value_coral -- total non-use value (coral)
      8. eq_ocean_nmuse_value_mangrove -- non-market use value (mangrove)
      9. eq_ocean_nonuse_value_mangrove -- non-use value (mangrove)
     10. eq_ocean_nmuse_value_fisheries -- health benefit from fisheries
     11. eq_utarg (welfare_ocean) -- CES nested utility replacing default CPC

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    oc_capital = sets["oc_capital"]
    oc_nonmkt = sets["oc_nonmkt"]
    oc_nonuse = sets["oc_nonuse"]

    TATM = v["TATM"]
    CPC = v["CPC"]
    YNET = v["YNET"]
    CPC_OCEAN_DAM = v["CPC_OCEAN_DAM"]
    OCEAN_AREA = v["OCEAN_AREA"]
    OCEAN_USENM_VALUE = v["OCEAN_USENM_VALUE"]
    OCEAN_USENM_VALUE_PERKM2 = v["OCEAN_USENM_VALUE_PERKM2"]
    OCEAN_NONUSE_VALUE = v["OCEAN_NONUSE_VALUE"]
    OCEAN_NONUSE_VALUE_PERKM2 = v["OCEAN_NONUSE_VALUE_PERKM2"]
    VSL = v["VSL"]
    UTARG = v["UTARG"]

    par_pop = params["par_pop"]
    par_ykali = params["par_ykali"]
    TATM0 = params["TATM0"]

    T = cfg.T
    region_names = [r for r in params["par_pop"].records["n"].unique()]

    # Load ocean data
    oc_names = ["coral", "mangrove", "ports", "fisheries"]
    ocean_data = _load_ocean_data(cfg.data_dir, oc_names, region_names)

    # Create GAMSPy parameters from loaded data
    def _make_oc_param(name, data_key):
        recs = [(oc, r, val) for (oc, r), val in ocean_data[data_key].items()]
        if not recs:
            recs = None
        return Parameter(m, name=name, domain=[oc_capital, n_set], records=recs)

    par_area_dam = _make_oc_param("oc_area_dam", "area_damage_coef")
    par_area_dam_sq = _make_oc_param("oc_area_dam_sq", "area_damage_coef_sq")
    par_area_start = _make_oc_param("oc_area_start", "area_start")
    par_consump_dam = _make_oc_param("oc_consump_dam", "consump_damage_coef")
    par_consump_dam_sq = _make_oc_param("oc_consump_dam_sq", "consump_damage_coef_sq")
    par_health_tame = _make_oc_param("oc_health_tame", "health_tame")
    par_health_beta = _make_oc_param("oc_health_beta", "health_beta")
    par_health_mu = _make_oc_param("oc_health_mu", "health_mu")
    par_unm_start = _make_oc_param("oc_unm_start", "unm_start")
    par_nu_start = _make_oc_param("oc_nu_start", "nu_start")
    par_value_intercept_unm = _make_oc_param("oc_val_int_unm", "value_intercept_unm")
    par_value_intercept_nu = _make_oc_param("oc_val_int_nu", "value_intercept_nu")
    par_value_exp_unm = _make_oc_param("oc_val_exp_unm", "value_exp_unm")
    par_value_exp_nu = _make_oc_param("oc_val_exp_nu", "value_exp_nu")
    par_value_exp_um = _make_oc_param("oc_val_exp_um", "value_exp_um")

    # Map of which regions have ocean capital (any non-zero parameter)
    # Create an indicator parameter
    map_recs = []
    for oc in oc_names:
        for r in region_names:
            has_data = any(
                ocean_data[k].get((oc, r), 0) != 0
                for k in ["consump_damage_coef", "area_start", "health_beta"]
            )
            if has_data:
                map_recs.append((oc, r, 1.0))
    par_map_n_oc = Parameter(m, name="map_n_oc", domain=[oc_capital, n_set],
                             records=map_recs if map_recs else None)

    # Per-capita GDP in baseline (ykali/pop, used for income-elastic values)
    # GAMS: gdppc_kali(t,n) = ykali(t,n) / pop(t,n) * 1e6
    # For ratios, the 1e6 cancels out, so we use ykali/pop directly.

    # Fix initial per-km2 values for coral at t=1
    # GAMS: OCEAN_USENM_VALUE_PERKM2.fx('coral','1',n)$map_n_oc(n,'coral') = ocean_unm_start('coral',n) * 1e6
    # GAMS: OCEAN_NONUSE_VALUE_PERKM2.fx('coral','1',n)$map_n_oc(n,'coral') = ocean_nu_start('coral',n) * 1e6
    for r in region_names:
        if ocean_data["unm_start"].get(("coral", r), 0) != 0:
            OCEAN_USENM_VALUE_PERKM2.fx["coral", "1", r] = (
                ocean_data["unm_start"][("coral", r)] * 1e6
            )
        if ocean_data["nu_start"].get(("coral", r), 0) != 0:
            OCEAN_NONUSE_VALUE_PERKM2.fx["coral", "1", r] = (
                ocean_data["nu_start"][("coral", r)] * 1e6
            )

    n_alias = sets["n_alias"]
    equations = []

    # ------------------------------------------------------------------
    # 1. eq_ocean_area: OCEAN_AREA = area_start * (1 + coef*dT + coef_sq*dT^2)
    # GAMS: uses TATM.l(t) (fixed level, NOT endogenous) to decouple ocean
    # area from the optimizer. We use par_tatm_level (a Parameter updated
    # between iterations in _before_solve) instead of the TATM Variable.
    # ------------------------------------------------------------------
    # Create a parameter to hold TATM level values (updated iteratively)
    par_tatm_level = params.get("par_tatm_level")
    if par_tatm_level is None:
        # Initialize from TATM starting levels
        tatm_recs = [(str(t), TATM0) for t in range(1, cfg.T + 1)]
        par_tatm_level = Parameter(m, name="tatm_level", domain=[t_set],
                                   records=tatm_recs)
        params["par_tatm_level"] = par_tatm_level

    eq_ocean_area = Equation(m, name="eq_ocean_area",
                             domain=[oc_capital, t_set, n_set])
    eq_ocean_area[oc_capital, t_set, n_set].where[
        par_area_start[oc_capital, n_set] > 0
    ] = (
        OCEAN_AREA[oc_capital, t_set, n_set] ==
        par_area_start[oc_capital, n_set]
        * (1 + par_area_dam[oc_capital, n_set] * (par_tatm_level[t_set] - TATM0)
           + par_area_dam_sq[oc_capital, n_set] * (par_tatm_level[t_set] - TATM0) ** 2)
    )
    equations.append(eq_ocean_area)

    # GAMS: TATM.lo(t) = TATM.l('1') -- prevent temperature below initial
    TATM.lo[t_set] = TATM0

    # ------------------------------------------------------------------
    # 2. eq_ocean_cpc: CPC with ocean consumption damages
    # GAMS: eq_ocean_cpc(t,n)$reg(n)
    # ------------------------------------------------------------------
    eq_ocean_cpc = Equation(m, name="eq_ocean_cpc", domain=[t_set, n_set])
    eq_ocean_cpc[t_set, n_set] = (
        CPC_OCEAN_DAM[t_set, n_set] == CPC[t_set, n_set]
        * (1 + Sum(oc_capital, par_consump_dam[oc_capital, n_set])
           * (TATM[t_set] - TATM0)
           + Sum(oc_capital, par_consump_dam_sq[oc_capital, n_set])
           * (TATM[t_set] - TATM0) ** 2)
    )
    equations.append(eq_ocean_cpc)

    # ------------------------------------------------------------------
    # 3. eq_ocean_vsl: Value of Statistical Life
    # GAMS: VSL(t,n) = vsl_start * (global_gdppc(1)/US_gdppc(1))
    #                            * (global_gdppc(t)/global_gdppc(1))
    # ------------------------------------------------------------------
    # GAMS: VSL = vsl_start * (global_gdppc_1 / US_gdppc_1) * (global_gdppc_t / global_gdppc_1)
    #      = vsl_start * global_gdppc_t / US_gdppc_1
    # The US GDP per capita normalization is needed because VSL_START is
    # calibrated to the US. Load US region GDP per capita at t=1 from data.
    us_gdppc_1 = params.get("par_us_gdppc_1")
    if us_gdppc_1 is not None:
        us_denom = us_gdppc_1
    else:
        # Fallback: use global average (original Python behavior)
        us_denom = (Sum(n_alias, par_ykali["1", n_alias])
                    / Sum(n_alias, par_pop["1", n_alias]))

    eq_ocean_vsl = Equation(m, name="eq_ocean_vsl", domain=[t_set, n_set])
    eq_ocean_vsl[t_set, n_set] = (
        VSL[t_set, n_set] == VSL_START
        * (Sum(n_alias, par_ykali[t_set, n_alias])
           / Sum(n_alias, par_pop[t_set, n_alias]))
        / us_denom
    )
    equations.append(eq_ocean_vsl)

    # ------------------------------------------------------------------
    # 4. eq_ocean_nmuse_value_perkm2_coral: per-km2 non-market use value (coral)
    # GAMS: OCEAN_USENM_VALUE_PERKM2('coral',t,n) =
    #   OCEAN_USENM_VALUE_PERKM2('coral',tm1,n) *
    #   [1 + (gdppc_kali(t,n)/gdppc_kali(tm1,n) - 1) * income_elasticity_usenm]
    # Condition: pre(tm1,t) and map_n_oc(n,'coral')
    # ------------------------------------------------------------------
    eq_ocean_nmuse_perkm2_coral = Equation(
        m, name="eq_ocean_nmuse_perkm2_coral", domain=[t_set, n_set])
    # Income growth ratio: (ykali(t)/pop(t)) / (ykali(t-1)/pop(t-1))
    # 1e6 factors cancel in the ratio.
    gdppc_ratio = (
        (par_ykali[t_set, n_set] / par_pop[t_set, n_set])
        / (par_ykali[t_set.lag(1), n_set] / par_pop[t_set.lag(1), n_set])
    )
    eq_ocean_nmuse_perkm2_coral[t_set, n_set].where[
        (Ord(t_set) > 1) & (par_map_n_oc["coral", n_set] > 0)
    ] = (
        OCEAN_USENM_VALUE_PERKM2["coral", t_set, n_set] ==
        OCEAN_USENM_VALUE_PERKM2["coral", t_set.lag(1), n_set]
        * (1 + (gdppc_ratio - 1) * OCEAN_INCOME_ELASTICITY_USENM)
    )
    equations.append(eq_ocean_nmuse_perkm2_coral)

    # ------------------------------------------------------------------
    # 5. eq_ocean_nmuse_value_coral: total = perkm2 * area
    # GAMS: OCEAN_USENM_VALUE('coral',t,n) =
    #   OCEAN_USENM_VALUE_PERKM2('coral',t,n) * OCEAN_AREA('coral',t,n)
    # ------------------------------------------------------------------
    eq_ocean_nmuse_coral = Equation(
        m, name="eq_ocean_nmuse_coral", domain=[t_set, n_set])
    eq_ocean_nmuse_coral[t_set, n_set].where[
        par_map_n_oc["coral", n_set] > 0
    ] = (
        OCEAN_USENM_VALUE["coral", t_set, n_set] ==
        OCEAN_USENM_VALUE_PERKM2["coral", t_set, n_set]
        * OCEAN_AREA["coral", t_set, n_set]
    )
    equations.append(eq_ocean_nmuse_coral)

    # ------------------------------------------------------------------
    # 6. eq_ocean_nonuse_value_perkm2_coral: per-km2 non-use value (coral)
    # GAMS: OCEAN_NONUSE_VALUE_PERKM2('coral',t,n) =
    #   OCEAN_NONUSE_VALUE_PERKM2('coral',tm1,n) *
    #   [1 + (gdppc_kali(t,n)/gdppc_kali(tm1,n) - 1) * income_elasticity_nonuse]
    # ------------------------------------------------------------------
    eq_ocean_nonuse_perkm2_coral = Equation(
        m, name="eq_ocean_nonuse_perkm2_coral", domain=[t_set, n_set])
    eq_ocean_nonuse_perkm2_coral[t_set, n_set].where[
        (Ord(t_set) > 1) & (par_map_n_oc["coral", n_set] > 0)
    ] = (
        OCEAN_NONUSE_VALUE_PERKM2["coral", t_set, n_set] ==
        OCEAN_NONUSE_VALUE_PERKM2["coral", t_set.lag(1), n_set]
        * (1 + (gdppc_ratio - 1) * OCEAN_INCOME_ELASTICITY_NONUSE)
    )
    equations.append(eq_ocean_nonuse_perkm2_coral)

    # ------------------------------------------------------------------
    # 7. eq_ocean_nonuse_value_coral: total = perkm2 * area
    # GAMS: OCEAN_NONUSE_VALUE('coral',t,n) =
    #   OCEAN_NONUSE_VALUE_PERKM2('coral',t,n) * OCEAN_AREA('coral',t,n)
    # ------------------------------------------------------------------
    eq_ocean_nonuse_coral = Equation(
        m, name="eq_ocean_nonuse_coral", domain=[t_set, n_set])
    eq_ocean_nonuse_coral[t_set, n_set].where[
        par_map_n_oc["coral", n_set] > 0
    ] = (
        OCEAN_NONUSE_VALUE["coral", t_set, n_set] ==
        OCEAN_NONUSE_VALUE_PERKM2["coral", t_set, n_set]
        * OCEAN_AREA["coral", t_set, n_set]
    )
    equations.append(eq_ocean_nonuse_coral)

    # ------------------------------------------------------------------
    # 8. eq_ocean_nmuse_value_mangrove: non-market use value (power function)
    # GAMS: OCEAN_USENM_VALUE('mangrove',t,n) =
    #   exp(value_intercept_unm('mangrove',n)) *
    #   (YNET.l(t,n)/pop(t,n)*1e6)^value_exp_unm('mangrove',n) *
    #   OCEAN_AREA('mangrove',t,n) * 1e6
    # ------------------------------------------------------------------
    eq_ocean_nmuse_mangrove = Equation(
        m, name="eq_ocean_nmuse_mangrove", domain=[t_set, n_set])
    eq_ocean_nmuse_mangrove[t_set, n_set].where[
        par_map_n_oc["mangrove", n_set] > 0
    ] = (
        OCEAN_USENM_VALUE["mangrove", t_set, n_set] ==
        gams_exp(par_value_intercept_unm["mangrove", n_set])
        * (YNET[t_set, n_set] / par_pop[t_set, n_set] * 1e6)
          ** par_value_exp_unm["mangrove", n_set]
        * OCEAN_AREA["mangrove", t_set, n_set] * 1e6
    )
    equations.append(eq_ocean_nmuse_mangrove)

    # ------------------------------------------------------------------
    # 9. eq_ocean_nonuse_value_mangrove: non-use value (power function)
    # GAMS: OCEAN_NONUSE_VALUE('mangrove',t,n) =
    #   exp(value_intercept_nu('mangrove',n)) *
    #   (YNET.l(t,n)/pop(t,n)*1e6)^value_exp_nu('mangrove',n) *
    #   OCEAN_AREA('mangrove',t,n) * 1e6
    # ------------------------------------------------------------------
    eq_ocean_nonuse_mangrove = Equation(
        m, name="eq_ocean_nonuse_mangrove", domain=[t_set, n_set])
    eq_ocean_nonuse_mangrove[t_set, n_set].where[
        par_map_n_oc["mangrove", n_set] > 0
    ] = (
        OCEAN_NONUSE_VALUE["mangrove", t_set, n_set] ==
        gams_exp(par_value_intercept_nu["mangrove", n_set])
        * (YNET[t_set, n_set] / par_pop[t_set, n_set] * 1e6)
          ** par_value_exp_nu["mangrove", n_set]
        * OCEAN_AREA["mangrove", t_set, n_set] * 1e6
    )
    equations.append(eq_ocean_nonuse_mangrove)

    # ------------------------------------------------------------------
    # 10. eq_ocean_nmuse_value_fisheries: health benefit from fisheries
    # GAMS: OCEAN_USENM_VALUE('fisheries',t,n) =
    #   (1 + health_beta('fisheries',n) * (TATM(t) - TATM.l('1')))
    #   * health_tame('fisheries',n) * pop(t,n) * 1e6
    #   * health_mu('fisheries',n) * health_eta * VSL(t,n)
    # ------------------------------------------------------------------
    eq_ocean_nmuse_fisheries = Equation(
        m, name="eq_ocean_nmuse_fisheries", domain=[t_set, n_set])
    eq_ocean_nmuse_fisheries[t_set, n_set].where[
        (par_map_n_oc["fisheries", n_set] > 0)
        & (par_health_beta["fisheries", n_set] != 0)
    ] = (
        OCEAN_USENM_VALUE["fisheries", t_set, n_set] ==
        (1 + par_health_beta["fisheries", n_set] * (TATM[t_set] - TATM0))
        * par_health_tame["fisheries", n_set]
        * par_pop[t_set, n_set] * 1e6
        * par_health_mu["fisheries", n_set]
        * OCEAN_HEALTH_ETA
        * VSL[t_set, n_set]
    )
    equations.append(eq_ocean_nmuse_fisheries)

    # ------------------------------------------------------------------
    # 11. eq_utarg (welfare_ocean): CES nested utility
    # GAMS (when $set welfare_ocean):
    #   eq_utility_arg(t,n)$reg(n)..
    #     UTARG(t,n) =E= (
    #       ( s2_1 * (
    #           s1_1 * CPC_OCEAN_DAM(t,n)^theta_1 +
    #           s1_2 * (sum(oc_nonmkt$map, USENM_VALUE) / pop)^theta_1
    #         )^(theta_2/theta_1)
    #         + s2_2 * (sum(oc_nonuse$map, NONUSE_VALUE) / pop)^theta_2
    #       )^(1/theta_2)
    #     )
    #
    # This replaces the default UTARG = CPC in core_welfare.
    # When ocean is active, core_welfare skips its eq_utarg definition.
    # ------------------------------------------------------------------
    eq_utarg = Equation(m, name="eq_utarg", domain=[t_set, n_set])
    eq_utarg[t_set, n_set] = (
        UTARG[t_set, n_set] == (
            (
                OCEAN_S2_1 * (
                    OCEAN_S1_1
                    * CPC_OCEAN_DAM[t_set, n_set] ** OCEAN_THETA_1
                    + OCEAN_S1_2 * (
                        Sum(oc_nonmkt.where[par_map_n_oc[oc_nonmkt, n_set] > 0],
                            OCEAN_USENM_VALUE[oc_nonmkt, t_set, n_set])
                        / par_pop[t_set, n_set]
                    ) ** OCEAN_THETA_1
                ) ** (OCEAN_THETA_2 / OCEAN_THETA_1)
                + OCEAN_S2_2 * (
                    (Sum(oc_nonuse.where[par_map_n_oc[oc_nonuse, n_set] > 0],
                         OCEAN_NONUSE_VALUE[oc_nonuse, t_set, n_set])
                     / par_pop[t_set, n_set]) ** OCEAN_THETA_2
                )
            ) ** (1.0 / OCEAN_THETA_2)
        )
    )
    equations.append(eq_utarg)

    return equations
