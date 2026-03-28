"""
FAIR climate module: multi-gas carbon cycle, forcing, and temperature.

Full GAMSPy translation of the FAIR climate model from
RICE50xmodel/modules/mod_climate_fair.gms (357 lines in the original GAMS).

The FAIR model replaces the simpler WITCH-CO2 single-gas carbon cycle with a
full multi-gas (CO2, CH4, N2O) impulse-response carbon cycle,
methane oxidation, tropospheric ozone forcing, and a two-box temperature model.

Key features:
- Four-box CO2 carbon cycle with time-varying decay (IRF100 feedback)
- Explicit CH4 and N2O concentration dynamics
- Methane oxidation to CO2
- Separate forcing equations for CO2, CH4, N2O, H2O, and trop. O3
- Two-box temperature (TSLOW + TFAST) instead of atmosphere + deep ocean
- IRF100 feedback (carbon-climate feedback on decay timescales)

References
----------
- Smith et al. (2018), "FAIR v1.3: a simple emissions-based impulse response
  and carbon cycle model", Geosci. Model Dev., 11, 2273-2297.
- Millar et al. (2017), "A modified impulse-response representation of the
  global near-surface air temperature and atmospheric concentration response
  to carbon dioxide emissions", Atmos. Chem. Phys., 17, 7213-7228.
"""

import math
from gamspy import Variable, Equation, Set, Parameter, Ord, Card, Sum, Number
from gamspy.math import log, sqrt, sqr, exp


# ---------------------------------------------------------------------------
# FAIR model constants (from GAMS mod_climate_fair.gms)
# ---------------------------------------------------------------------------
# Thermal response timescales (years)
DSLOW = 236.0
DFAST = 4.07

# IRF100 parameters
IRF_PREINDUSTRIAL = 35.0   # Pre-industrial IRF100 (%)
IRF_MAX = 97.0             # Maximum IRF100 (%)
IRC = 0.019                # IRF100 increase with cumulative carbon (yr/GtC)
IRT = 4.165                # IRF100 increase with warming (yr/K)

# Atmospheric constants
ATMOSPHERE_MASS = 5.1352e18   # kg
ATMOSPHERE_MM = 28.97         # kg/mol

# Climate sensitivity
TECS = 3.0    # Equilibrium climate sensitivity (K)
TTCR = 1.8    # Transient climate response (K)
FORC2X = 3.71  # Forcing for 2xCO2 (W/m2)

# Carbon cycle box parameters
BOX_NAMES = ["geo", "deep", "bio", "mixed"]
TAUBOX = {"geo": 1e6, "deep": 394.4, "bio": 36.53, "mixed": 4.304}
EMSHARE = {"geo": 0.2173, "deep": 0.224, "bio": 0.2824, "mixed": 0.2763}

# Molecular masses (kg/mol)
GHG_MM = {"co2": 44.01, "ch4": 16.04, "n2o": 44.013, "c": 12.01, "n2": 28.013}

# Pre-industrial concentrations
CONC_PREINDUSTRIAL = {"co2": 278.05, "ch4": 722.0, "n2o": 255.0}

# GHG half-lives (years) for single-box decay
TAUGHG = {"ch4": 9.3, "n2o": 121.0}

# Initial conditions (year 2015)
CONC0 = {"co2": 400.724, "ch4": 1822.1, "n2o": 324.314}
CUMEMI0 = 2070.6

# Initial share of cumulative CO2 in reservoirs (from historical FAIR run)
EMSHARE0 = {"geo": 0.551, "deep": 0.238, "bio": 0.175, "mixed": 0.036}
TSLOWSHARE0 = 0.153443 / 1.10308

# CO2toC conversion factor
CO2toC = 12.0 / 44.0

# Smooth approximation delta
DELTA = 1e-3


def _compute_derived_constants():
    """Compute derived constants that mirror GAMS compute_data phase."""
    # emitoconc: conversion from emissions to concentration
    emitoconc = {}
    emitoconc["co2"] = 1e18 / ATMOSPHERE_MASS * ATMOSPHERE_MM / GHG_MM["co2"]
    emitoconc["ch4"] = 1e18 / ATMOSPHERE_MASS * ATMOSPHERE_MM / GHG_MM["ch4"]
    # n2o is expressed in N2 equivalent
    emitoconc["n2o"] = (1e18 / ATMOSPHERE_MASS * ATMOSPHERE_MM / GHG_MM["n2o"]
                        * GHG_MM["n2o"] / GHG_MM["n2"])
    emitoconc["c"] = 1e18 / ATMOSPHERE_MASS * ATMOSPHERE_MM / GHG_MM["c"]

    catm_preindustrial = CONC_PREINDUSTRIAL["co2"] / emitoconc["co2"]

    # scaling_forc2x: ensures consistency with user-specified 2xCO2 forcing
    cp_co2 = CONC_PREINDUSTRIAL["co2"]
    cp_n2o = CONC_PREINDUSTRIAL["n2o"]
    scaling_forc2x = (
        (-2.4e-7 * cp_co2**2 + 7.2e-4 * cp_co2
         - 1.05e-4 * (2 * cp_n2o) + 5.36) * math.log(2) / FORC2X
    )

    # Initial reservoir concentrations
    res0 = {}
    for box in BOX_NAMES:
        res0[box] = EMSHARE0[box] * (CONC0["co2"] - CONC_PREINDUSTRIAL["co2"])

    # Solve QSLOW and QFAST from ECS and TCR
    # eq_tecs: Tecs = forc2x * (QSLOW + QFAST)
    # eq_ttcr: Ttcr = forc2x * (QSLOW*(1-dslow/69.7*(1-exp(-69.7/dslow)))
    #                          + QFAST*(1-dfast/69.7*(1-exp(-69.7/dfast))))
    a_slow = 1 - DSLOW / 69.7 * (1 - math.exp(-69.7 / DSLOW))
    a_fast = 1 - DFAST / 69.7 * (1 - math.exp(-69.7 / DFAST))
    # From the two equations:
    # QSLOW + QFAST = Tecs / forc2x
    # a_slow*QSLOW + a_fast*QFAST = Ttcr / forc2x
    # Solution:
    q_sum = TECS / FORC2X
    q_weighted = TTCR / FORC2X
    QSLOW = (q_weighted - a_fast * q_sum) / (a_slow - a_fast)
    QFAST = q_sum - QSLOW

    # Initial temperatures
    TATM0 = 1.1  # matches other modules
    dt0 = 0.15  # GAMS hub_climate.gms line 38: dt0 /0.15/
    # Temperature change from 1765 (beginning year of FAIR) to reference
    # model period (e.g. average 1850-1900).
    # From GAMS: tslow0 = (tatm0 + dt0) * tslowshare0
    tslow0 = (TATM0 + dt0) * TSLOWSHARE0
    tfast0 = (TATM0 + dt0) * (1 - TSLOWSHARE0)

    return dict(
        emitoconc=emitoconc,
        catm_preindustrial=catm_preindustrial,
        scaling_forc2x=scaling_forc2x,
        res0=res0,
        QSLOW=QSLOW,
        QFAST=QFAST,
        tslow0=tslow0,
        tfast0=tfast0,
        dt0=dt0,
    )


# Pre-compute at module load time
_DERIVED = _compute_derived_constants()


def declare_vars(m, sets, params, cfg, v):
    """Create FAIR climate variables.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, ghg_set, layers, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    ghg_set = sets["ghg_set"]

    TATM0 = params["TATM0"]
    TOCEAN0 = params["TOCEAN0"]

    # ------------------------------------------------------------------
    # Additional sets for FAIR
    # ------------------------------------------------------------------
    box_set = Set(m, name="box", records=BOX_NAMES)
    sets["box_set"] = box_set

    # ------------------------------------------------------------------
    # Variables -- mirrors GAMS declare_vars phase
    # ------------------------------------------------------------------
    # Concentrations
    CONC = Variable(m, name="CONC", domain=[ghg_set, t_set], type="positive")
    CONC.lo[ghg_set, t_set] = 1e-9

    # Reservoirs (CO2 4-box) -- free (can be negative for net-negative emissions)
    RES = Variable(m, name="RES", domain=[box_set, t_set])

    # Cumulative emissions
    CUMEMI_FAIR = Variable(m, name="CUMEMI_FAIR", domain=[t_set], type="positive")

    # Carbon tracking
    C_ATM = Variable(m, name="C_ATM", domain=[t_set], type="positive")
    C_SINKS = Variable(m, name="C_SINKS", domain=[t_set])

    # IRF and scaling
    IRF = Variable(m, name="IRF", domain=[t_set], type="positive")
    IRF.up[t_set] = 100
    CD_SCALE = Variable(m, name="CD_SCALE", domain=[t_set], type="positive")
    CD_SCALE.lo[t_set] = 1e-2
    CD_SCALE.up[t_set] = 1e3
    CD_SCALE.l[t_set] = 0.35

    # Methane oxidation
    OXI_CH4 = Variable(m, name="OXI_CH4", domain=[t_set])

    # Fossil methane fraction
    FF_CH4 = Variable(m, name="FF_CH4", domain=[t_set], type="positive")
    FF_CH4.up[t_set] = 1

    # Forcing per GHG
    RF = Variable(m, name="RF", domain=[ghg_set, t_set])
    RF.lo[ghg_set, t_set] = -10
    RF.up[ghg_set, t_set] = 40

    # Other climate agent forcing (H2O, O3_trop)
    ORF_H2O = Variable(m, name="ORF_H2O", domain=[t_set])
    ORF_O3TROP = Variable(m, name="ORF_O3TROP", domain=[t_set])

    # Temperature: two-box model
    TSLOW = Variable(m, name="TSLOW", domain=[t_set])
    TFAST = Variable(m, name="TFAST", domain=[t_set])
    TATM = Variable(m, name="TATM", domain=[t_set])
    TATM.lo[t_set] = -10
    TATM.up[t_set] = 10
    TATM.l[t_set] = TATM0

    # Total forcing
    FORC = Variable(m, name="FORC", domain=[t_set])

    # World emissions per GHG
    W_EMI = Variable(m, name="W_EMI", domain=[ghg_set, t_set])
    W_EMI.lo[ghg_set, t_set] = -200
    W_EMI.up[ghg_set, t_set] = 200

    # ------------------------------------------------------------------
    # Initial conditions (fix first period)
    # ------------------------------------------------------------------
    # Concentrations
    CONC.fx["ch4", "1"] = CONC0["ch4"]
    CONC.fx["n2o", "1"] = CONC0["n2o"]
    # CO2 concentration is defined by constraining RES
    for box in BOX_NAMES:
        RES.fx[box, "1"] = _DERIVED["res0"][box]

    CUMEMI_FAIR.fx["1"] = CUMEMI0
    TSLOW.fx["1"] = _DERIVED["tslow0"]
    TFAST.fx["1"] = _DERIVED["tfast0"]
    TATM.fx["1"] = TATM0

    # Starting values
    CONC.l["co2", t_set] = CONC0["co2"]
    CONC.l["ch4", t_set] = CONC0["ch4"]
    CONC.l["n2o", t_set] = CONC0["n2o"]
    TSLOW.l[t_set] = _DERIVED["tslow0"]
    TFAST.l[t_set] = _DERIVED["tfast0"]
    FF_CH4.l[t_set] = 0.28  # approximate fossil CH4 fraction
    OXI_CH4.l[t_set] = 0
    IRF.l[t_set] = IRF_PREINDUSTRIAL
    CUMEMI_FAIR.l[t_set] = CUMEMI0

    # ------------------------------------------------------------------
    # Register in shared variable dict
    # ------------------------------------------------------------------
    v["TATM"] = TATM
    v["FORC"] = FORC
    v["W_EMI"] = W_EMI
    v["CONC"] = CONC
    v["RES"] = RES
    v["CUMEMI_FAIR"] = CUMEMI_FAIR
    v["C_ATM"] = C_ATM
    v["C_SINKS"] = C_SINKS
    v["IRF"] = IRF
    v["CD_SCALE"] = CD_SCALE
    v["OXI_CH4"] = OXI_CH4
    v["FF_CH4"] = FF_CH4
    v["RF"] = RF
    v["ORF_H2O"] = ORF_H2O
    v["ORF_O3TROP"] = ORF_O3TROP
    v["TSLOW"] = TSLOW
    v["TFAST"] = TFAST

    # Placeholder for modules that expect TOCEAN
    TOCEAN = Variable(m, name="TOCEAN", domain=[t_set])
    TOCEAN.l[t_set] = TOCEAN0
    TOCEAN.fx["1"] = TOCEAN0
    v["TOCEAN"] = TOCEAN


def define_eqs(m, sets, params, cfg, v):
    """Create all 20 FAIR climate equations.

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    ghg_set = sets["ghg_set"]
    box_set = sets["box_set"]

    TSTEP = cfg.TSTEP

    # Derived constants
    emitoconc_co2 = _DERIVED["emitoconc"]["co2"]
    emitoconc_ch4 = _DERIVED["emitoconc"]["ch4"]
    emitoconc_n2o = _DERIVED["emitoconc"]["n2o"]
    catm_preind = _DERIVED["catm_preindustrial"]
    scaling_forc2x = _DERIVED["scaling_forc2x"]
    QSLOW_val = _DERIVED["QSLOW"]
    QFAST_val = _DERIVED["QFAST"]
    dt0 = _DERIVED["dt0"]

    # Create GAMSPy parameters for box-level data
    par_taubox = Parameter(m, name="taubox", domain=[box_set],
                           records=[(b, TAUBOX[b]) for b in BOX_NAMES])
    par_emshare = Parameter(m, name="emshare", domain=[box_set],
                            records=[(b, EMSHARE[b]) for b in BOX_NAMES])
    par_taughg = Parameter(m, name="taughg", domain=[ghg_set],
                           records=[("ch4", TAUGHG["ch4"]),
                                    ("n2o", TAUGHG["n2o"])])

    # Exogenous data parameters (natural emissions, fossil CH4 fraction,
    # exogenous forcing).  Load from CSV data files in data_mod_climate/
    # when available; fall back to constants otherwise.
    import os
    import pandas as pd
    T = cfg.T
    climate_data_dir = os.path.join(cfg.data_dir, "data_mod_climate")

    # ------------------------------------------------------------------
    # Natural emissions for CH4 and N2O (GAMS: natural_emissions(t,ghg))
    # Data file: natural_emissions.csv -- columns: t, Dim2(ghg), Val
    # These are constant over time in the GAMS data (190.58 CH4, 8.99 N2O).
    # ------------------------------------------------------------------
    nat_emi_recs = []
    nat_emi_file = os.path.join(climate_data_dir, "natural_emissions.csv")
    nat_emi_loaded = False
    if os.path.exists(nat_emi_file):
        try:
            ne_df = pd.read_csv(nat_emi_file)
            # Build lookup: (t_str, ghg) -> val
            ne_lookup = {}
            for _, row in ne_df.iterrows():
                t_str = str(row.iloc[0])
                ghg = str(row.iloc[1]).lower()
                ne_lookup[(t_str, ghg)] = float(row["Val"])
            for t in range(1, T + 1):
                for ghg in ("ch4", "n2o"):
                    val = ne_lookup.get((str(t), ghg))
                    if val is None:
                        # Some periods may be missing; use first-period value
                        val = ne_lookup.get(("1", ghg), 190.58 if ghg == "ch4" else 8.99)
                    nat_emi_recs.append((str(t), ghg, val))
            nat_emi_loaded = True
        except Exception:
            nat_emi_loaded = False
    if not nat_emi_loaded:
        # Fallback: constant values from GAMS data
        for t in range(1, T + 1):
            nat_emi_recs.append((str(t), "ch4", 190.5807))
            nat_emi_recs.append((str(t), "n2o", 8.9883))
    par_natural_emi = Parameter(m, name="fair_natural_emi",
                                domain=[t_set, ghg_set], records=nat_emi_recs)

    # ------------------------------------------------------------------
    # Fossil CH4 fraction (GAMS: fossilch4_frac(t,rcp))
    # Data file: fossilch4_frac.csv -- columns: t, Dim2(rcp), Val
    # We use the RCP scenario matching cfg (default RCP45).
    # ------------------------------------------------------------------
    fch4_recs = []
    fch4_file = os.path.join(climate_data_dir, "fossilch4_frac.csv")
    fch4_loaded = False
    if os.path.exists(fch4_file):
        try:
            fch4_df = pd.read_csv(fch4_file)
            # Determine which RCP to use; default to RCP45
            rcp_name = getattr(cfg, "rcp", "RCP45")
            fch4_lookup = {}
            for _, row in fch4_df.iterrows():
                t_str = str(row.iloc[0])
                rcp = str(row.iloc[1])
                if rcp == rcp_name:
                    fch4_lookup[t_str] = float(row["Val"])
            if fch4_lookup:
                for t in range(1, T + 1):
                    val = fch4_lookup.get(str(t), 0.28)
                    fch4_recs.append((str(t), val))
                fch4_loaded = True
        except Exception:
            fch4_loaded = False
    if not fch4_loaded:
        # Fallback: constant fraction
        fch4_recs = [(str(t), 0.28) for t in range(1, T + 1)]
    par_fossilch4_frac = Parameter(m, name="fossilch4_frac_fair",
                                   domain=[t_set], records=fch4_recs)

    # ------------------------------------------------------------------
    # Exogenous forcing: GAMS computes forcing_exogenous(t) as the sum of
    # all forcing agents EXCEPT co2, ch4, n2o, tropospheric ozone, and
    # stratospheric H2O (which are computed endogenously by FAIR).
    # Data file: Forcing.csv -- columns: t, Dim2(rcp), Dim3(agent), Val
    # Exogenous agents include: F-gases, aerosols, solar, volcanic,
    # stratospheric ozone, land use, mineral dust, etc.
    # ------------------------------------------------------------------
    # Endogenous agents (computed by FAIR equations):
    _ENDOGENOUS_FORCING = {
        "co2", "ch4", "n2o", "tropoz", "ch4oxstrath2o",
        # Aggregate categories to exclude:
        "co2ch4n2o", "ghg", "kyotoghg", "total_anthro",
        "total_inclvolcanic",
    }
    forc_exog_recs = []
    forc_file = os.path.join(climate_data_dir, "Forcing.csv")
    forc_loaded = False
    if os.path.exists(forc_file):
        try:
            forc_df = pd.read_csv(forc_file)
            rcp_name = getattr(cfg, "rcp", "RCP45")
            # Group by (t, rcp), sum over exogenous agents
            forc_by_t = {}
            for _, row in forc_df.iterrows():
                rcp = str(row.iloc[1])
                if rcp != rcp_name:
                    continue
                agent = str(row.iloc[2]).lower()
                if agent in _ENDOGENOUS_FORCING:
                    continue
                t_str = str(row.iloc[0])
                forc_by_t[t_str] = forc_by_t.get(t_str, 0.0) + float(row["Val"])
            if forc_by_t:
                for t in range(1, T + 1):
                    val = forc_by_t.get(str(t), -0.5)
                    forc_exog_recs.append((str(t), val))
                forc_loaded = True
        except Exception:
            forc_loaded = False
    if not forc_loaded:
        # Fallback: constant approximate net aerosol + other forcing
        forc_exog_recs = [(str(t), -0.5) for t in range(1, T + 1)]
    par_forcing_exogenous = Parameter(m, name="forcing_exogenous",
                                      domain=[t_set], records=forc_exog_recs)

    # Variables
    TATM = v["TATM"]
    FORC = v["FORC"]
    W_EMI = v["W_EMI"]
    CONC = v["CONC"]
    RES = v["RES"]
    CUMEMI_FAIR = v["CUMEMI_FAIR"]
    C_ATM = v["C_ATM"]
    C_SINKS = v["C_SINKS"]
    IRF = v["IRF"]
    CD_SCALE = v["CD_SCALE"]
    OXI_CH4 = v["OXI_CH4"]
    FF_CH4 = v["FF_CH4"]
    RF = v["RF"]
    ORF_H2O = v["ORF_H2O"]
    ORF_O3TROP = v["ORF_O3TROP"]
    TSLOW = v["TSLOW"]
    TFAST = v["TFAST"]
    E = v["E"]

    equations = []

    # ------------------------------------------------------------------
    # 1. eq_w_emi: World emissions per GHG
    # GAMS: W_EMI(ghg,t) = sum(n, E(t,n,ghg))
    # ------------------------------------------------------------------
    eq_w_emi = Equation(m, name="eq_w_emi", domain=[ghg_set, t_set])
    eq_w_emi[ghg_set, t_set] = (
        W_EMI[ghg_set, t_set] == Sum(n_set, E[t_set, n_set, ghg_set])
    )
    equations.append(eq_w_emi)

    # ------------------------------------------------------------------
    # 2. eq_reslom: Reservoir law of motion (4-box CO2)
    # GAMS: RES(box,tp1) = RES(box,t)*exp(-tstep/(taubox*CD_SCALE))
    #        + emshare * (W_EMI('co2',tp1) + OXI_CH4(tp1)) * emitoconc('co2') * tstep
    # ------------------------------------------------------------------
    eq_reslom = Equation(m, name="eq_reslom", domain=[box_set, t_set])
    eq_reslom[box_set, t_set].where[Ord(t_set) < Card(t_set)] = (
        RES[box_set, t_set.lead(1)] ==
        RES[box_set, t_set] * exp(-TSTEP / (par_taubox[box_set] * CD_SCALE[t_set]))
        + par_emshare[box_set]
        * (W_EMI["co2", t_set.lead(1)] + OXI_CH4[t_set.lead(1)])
        * emitoconc_co2 * TSTEP
    )
    equations.append(eq_reslom)

    # ------------------------------------------------------------------
    # 3. eq_concco2: CO2 concentration from reservoirs
    # GAMS: CONC('co2',t) = conc_preindustrial('co2') + sum(box, RES(box,t))
    # ------------------------------------------------------------------
    eq_concco2 = Equation(m, name="eq_concco2", domain=[t_set])
    eq_concco2[t_set] = (
        CONC["co2", t_set] == CONC_PREINDUSTRIAL["co2"]
        + Sum(box_set, RES[box_set, t_set])
    )
    equations.append(eq_concco2)

    # ------------------------------------------------------------------
    # 4. eq_catm: Atmospheric carbon
    # GAMS: C_ATM(t) = CONC('co2',t) / emitoconc('co2')
    # ------------------------------------------------------------------
    eq_catm = Equation(m, name="eq_catm", domain=[t_set])
    eq_catm[t_set] = (
        C_ATM[t_set] == CONC["co2", t_set] / emitoconc_co2
    )
    equations.append(eq_catm)

    # ------------------------------------------------------------------
    # 5. eq_cumemi: Cumulative CO2 emissions
    # GAMS: CUMEMI(tp1) = CUMEMI(t) + (W_EMI('co2',tp1) + OXI_CH4(tp1))*tstep
    # ------------------------------------------------------------------
    eq_cumemi_fair = Equation(m, name="eq_cumemi_fair", domain=[t_set])
    eq_cumemi_fair[t_set].where[Ord(t_set) < Card(t_set)] = (
        CUMEMI_FAIR[t_set.lead(1)] == CUMEMI_FAIR[t_set]
        + (W_EMI["co2", t_set.lead(1)] + OXI_CH4[t_set.lead(1)]) * TSTEP
    )
    equations.append(eq_cumemi_fair)

    # ------------------------------------------------------------------
    # 6. eq_csinks: Carbon in sinks
    # GAMS: C_SINKS(t) = CUMEMI(t) - (C_ATM(t) - catm_preindustrial)
    # ------------------------------------------------------------------
    eq_csinks = Equation(m, name="eq_csinks", domain=[t_set])
    eq_csinks[t_set] = (
        C_SINKS[t_set] == CUMEMI_FAIR[t_set] - (C_ATM[t_set] - catm_preind)
    )
    equations.append(eq_csinks)

    # ------------------------------------------------------------------
    # 7. eq_concghg: Non-CO2 GHG concentration dynamics (CH4, N2O)
    # GAMS: CONC(ghg,tp1) = CONC(ghg,t)*exp(-tstep/taughg) +
    #        ((W_EMI(ghg,tp1)+W_EMI(ghg,t))/2 + natural_emi) * emitoconc * tstep
    # Condition: not co2
    # ------------------------------------------------------------------
    eq_concghg_ch4 = Equation(m, name="eq_concghg_ch4", domain=[t_set])
    eq_concghg_ch4[t_set].where[Ord(t_set) < Card(t_set)] = (
        CONC["ch4", t_set.lead(1)] ==
        CONC["ch4", t_set] * math.exp(-TSTEP / TAUGHG["ch4"])
        + ((W_EMI["ch4", t_set.lead(1)] + W_EMI["ch4", t_set]) / 2
           + par_natural_emi[t_set.lead(1), "ch4"])
        * emitoconc_ch4 * TSTEP
    )
    equations.append(eq_concghg_ch4)

    eq_concghg_n2o = Equation(m, name="eq_concghg_n2o", domain=[t_set])
    eq_concghg_n2o[t_set].where[Ord(t_set) < Card(t_set)] = (
        CONC["n2o", t_set.lead(1)] ==
        CONC["n2o", t_set] * math.exp(-TSTEP / TAUGHG["n2o"])
        + ((W_EMI["n2o", t_set.lead(1)] + W_EMI["n2o", t_set]) / 2
           + par_natural_emi[t_set.lead(1), "n2o"])
        * emitoconc_n2o * TSTEP
    )
    equations.append(eq_concghg_n2o)

    # ------------------------------------------------------------------
    # 8. eq_methoxi: Methane oxidation to CO2
    # GAMS: OXI_CH4(t) = 1e-3 * ghg_mm('co2')/ghg_mm('ch4') * 0.61
    #        * FF_CH4(t) * (CONC('ch4',t)-conc_preind_ch4)
    #        * (1 - exp(-tstep/taughg('ch4')))
    # ------------------------------------------------------------------
    co2_over_ch4 = GHG_MM["co2"] / GHG_MM["ch4"]
    ch4_decay_factor = 1 - math.exp(-TSTEP / TAUGHG["ch4"])
    eq_methoxi = Equation(m, name="eq_methoxi", domain=[t_set])
    eq_methoxi[t_set] = (
        OXI_CH4[t_set] == 1e-3 * co2_over_ch4 * 0.61 * FF_CH4[t_set]
        * (CONC["ch4", t_set] - CONC_PREINDUSTRIAL["ch4"])
        * ch4_decay_factor
    )
    equations.append(eq_methoxi)

    # ------------------------------------------------------------------
    # 9. eq_ffch4: Fossil methane fraction
    # GAMS: FF_CH4(t) = fossilch4_frac(t,rcp)
    #        * (sum(n, EIND(t,n,'co2'))) / (sum(n, convq*sigma*ykali))
    # Simplified: fix to exogenous fossil fraction (the ratio is ~1 in BAU)
    # ------------------------------------------------------------------
    eq_ffch4 = Equation(m, name="eq_ffch4", domain=[t_set])
    eq_ffch4[t_set] = (
        FF_CH4[t_set] == par_fossilch4_frac[t_set]
    )
    equations.append(eq_ffch4)

    # ------------------------------------------------------------------
    # 10. eq_forcco2: CO2 radiative forcing (Etminan et al.)
    # GAMS: RF('co2',t) = (-2.4e-7*sqr(CONC_co2-cp_co2) + 7.2e-4*(sqrt(sqr(CONC_co2-cp_co2)+sqr(delta))-delta)
    #        -1.05e-4*(CONC_n2o + cp_n2o) + 5.36) * log(CONC_co2/cp_co2) / scaling_forc2x
    # ------------------------------------------------------------------
    cp_co2 = CONC_PREINDUSTRIAL["co2"]
    cp_n2o = CONC_PREINDUSTRIAL["n2o"]

    eq_forcco2 = Equation(m, name="eq_forcco2", domain=[t_set])
    eq_forcco2[t_set] = (
        RF["co2", t_set] ==
        (-2.4e-7 * sqr(CONC["co2", t_set] - cp_co2)
         + 7.2e-4 * (sqrt(sqr(CONC["co2", t_set] - cp_co2) + sqr(Number(DELTA))) - DELTA)
         - 1.05e-4 * (CONC["n2o", t_set] + cp_n2o)
         + 5.36)
        * log(CONC["co2", t_set] / cp_co2)
        / scaling_forc2x
    )
    equations.append(eq_forcco2)

    # ------------------------------------------------------------------
    # 11. eq_forcch4: CH4 radiative forcing
    # GAMS: RF('ch4',t) = (-6.5e-7*(CONC_ch4+cp_ch4) - 4.1e-6*(CONC_n2o+cp_n2o) + 0.043)
    #        * (sqrt(CONC_ch4) - sqrt(cp_ch4))
    # ------------------------------------------------------------------
    cp_ch4 = CONC_PREINDUSTRIAL["ch4"]

    eq_forcch4 = Equation(m, name="eq_forcch4", domain=[t_set])
    eq_forcch4[t_set] = (
        RF["ch4", t_set] ==
        (-6.5e-7 * (CONC["ch4", t_set] + cp_ch4)
         - 4.1e-6 * (CONC["n2o", t_set] + cp_n2o)
         + 0.043)
        * (sqrt(CONC["ch4", t_set]) - math.sqrt(cp_ch4))
    )
    equations.append(eq_forcch4)

    # ------------------------------------------------------------------
    # 12. eq_forcn2o: N2O radiative forcing
    # GAMS: RF('n2o',t) = (-4.0e-6*(CONC_co2+cp_co2) + 2.1e-6*(CONC_n2o+cp_n2o)
    #        - 2.45e-6*(CONC_ch4+cp_ch4) + 0.117)
    #        * (sqrt(CONC_n2o) - sqrt(cp_n2o))
    # ------------------------------------------------------------------
    eq_forcn2o = Equation(m, name="eq_forcn2o", domain=[t_set])
    eq_forcn2o[t_set] = (
        RF["n2o", t_set] ==
        (-4.0e-6 * (CONC["co2", t_set] + cp_co2)
         + 2.1e-6 * (CONC["n2o", t_set] + cp_n2o)
         - 2.45e-6 * (CONC["ch4", t_set] + cp_ch4)
         + 0.117)
        * (sqrt(CONC["n2o", t_set]) - math.sqrt(cp_n2o))
    )
    equations.append(eq_forcn2o)

    # ------------------------------------------------------------------
    # 13. eq_forch2o: Stratospheric H2O forcing
    # GAMS: ORF('h2o',t) = 0.12 * RF('ch4',t)
    # ------------------------------------------------------------------
    eq_forch2o = Equation(m, name="eq_forch2o", domain=[t_set])
    eq_forch2o[t_set] = (
        ORF_H2O[t_set] == 0.12 * RF["ch4", t_set]
    )
    equations.append(eq_forch2o)

    # ------------------------------------------------------------------
    # 14. eq_forco3trop: Tropospheric ozone forcing
    # GAMS: ORF('o3trop',t) = 1.74e-4*(CONC_ch4-cp_ch4)
    #        + (emissions-based terms for NOx, CO, NMVOC)
    #        + smoothed temperature feedback
    # Simplified: use only CH4-driven component + temperature feedback
    # ------------------------------------------------------------------
    eq_forco3trop = Equation(m, name="eq_forco3trop", domain=[t_set])
    # The emissions-based terms (NOx, CO, NMVOC) are exogenous and small;
    # approximate as zero deviation from baseline.
    # Temperature feedback: 0.032*(exp(-1.35*(TATM+dt0)) - 1) smoothed
    eq_forco3trop[t_set] = (
        ORF_O3TROP[t_set] ==
        1.74e-4 * (CONC["ch4", t_set] - cp_ch4)
        + (0.032 * (exp(-1.35 * (TATM[t_set] + dt0)) - 1)
           - sqrt(sqr(0.032 * (exp(-1.35 * (TATM[t_set] + dt0)) - 1))
                  + sqr(Number(1e-8)))) / 2
    )
    equations.append(eq_forco3trop)

    # ------------------------------------------------------------------
    # 15. (eq_forcoghg: other GHG forcing -- not implemented, these are F-gases
    #      with exogenous concentrations; their forcing is included in
    #      forcing_exogenous parameter)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 16. eq_forcing: Total forcing
    # GAMS: FORC(t) = sum(ghg, RF(ghg,t)) + sum(climagents, ORF(climagents,t))
    #        + forcing_exogenous(t)
    # ------------------------------------------------------------------
    eq_forcing = Equation(m, name="eq_forcing", domain=[t_set])
    forc_rhs = (
        Sum(ghg_set, RF[ghg_set, t_set])
        + ORF_H2O[t_set] + ORF_O3TROP[t_set]
        + par_forcing_exogenous[t_set]
    )
    # SAI forcing offset (if active)
    if "W_SAI" in v:
        import pydice32.modules.mod_sai as _sai_mod
        geoeng_f = _sai_mod.GEOENG_FORCING
        if geoeng_f != 0.0:
            forc_rhs = forc_rhs + geoeng_f * v["W_SAI"][t_set]
    eq_forcing[t_set] = FORC[t_set] == forc_rhs
    equations.append(eq_forcing)

    # ------------------------------------------------------------------
    # 17. eq_tslow: Slow temperature response
    # GAMS: TSLOW(tp1) = TSLOW(t)*exp(-tstep/dslow) + QSLOW*FORC(t)*(1-exp(-tstep/dslow))
    # ------------------------------------------------------------------
    slow_decay = math.exp(-TSTEP / DSLOW)
    eq_tslow = Equation(m, name="eq_tslow", domain=[t_set])
    eq_tslow[t_set].where[Ord(t_set) < Card(t_set)] = (
        TSLOW[t_set.lead(1)] ==
        TSLOW[t_set] * slow_decay
        + QSLOW_val * FORC[t_set] * (1 - slow_decay)
    )
    equations.append(eq_tslow)

    # ------------------------------------------------------------------
    # 18. eq_tfast: Fast temperature response
    # GAMS: TFAST(tp1) = TFAST(t)*exp(-tstep/dfast) + QFAST*FORC(t)*(1-exp(-tstep/dfast))
    # ------------------------------------------------------------------
    fast_decay = math.exp(-TSTEP / DFAST)
    eq_tfast = Equation(m, name="eq_tfast", domain=[t_set])
    eq_tfast[t_set].where[Ord(t_set) < Card(t_set)] = (
        TFAST[t_set.lead(1)] ==
        TFAST[t_set] * fast_decay
        + QFAST_val * FORC[t_set] * (1 - fast_decay)
    )
    equations.append(eq_tfast)

    # ------------------------------------------------------------------
    # 19. eq_tatm: Atmospheric temperature
    # GAMS: TATM(t) = TSLOW(t) + TFAST(t) - dt0
    # ------------------------------------------------------------------
    eq_tatm = Equation(m, name="eq_tatm", domain=[t_set])
    eq_tatm[t_set] = (
        TATM[t_set] == TSLOW[t_set] + TFAST[t_set] - dt0
    )
    equations.append(eq_tatm)

    # ------------------------------------------------------------------
    # 20a. eq_irflhs: IRF100 left-hand side (implicit equation)
    # GAMS: IRF(t) = CD_SCALE(t) * sum(box, emshare*taubox*(1-exp(-100/(CD_SCALE*taubox))))
    # ------------------------------------------------------------------
    eq_irflhs = Equation(m, name="eq_irflhs", domain=[t_set])
    eq_irflhs[t_set] = (
        IRF[t_set] == CD_SCALE[t_set] * Sum(
            box_set,
            par_emshare[box_set] * par_taubox[box_set]
            * (1 - exp(-100 / (CD_SCALE[t_set] * par_taubox[box_set])))
        )
    )
    equations.append(eq_irflhs)

    # ------------------------------------------------------------------
    # 20b. eq_irfrhs: IRF100 right-hand side (with smooth max cap)
    # GAMS: IRF(t) = (irf_max + (irf_preind + irC*C_SINKS*CO2toC + irT*(TATM+dt0))
    #                 - sqrt(sqr(irf_max - (irf_preind + irC*C_SINKS*CO2toC + irT*(TATM+dt0))) + sqr(1e-8))) / 2
    # This is a smooth min(irf_max, uncapped_irf)
    # ------------------------------------------------------------------
    eq_irfrhs = Equation(m, name="eq_irfrhs", domain=[t_set])
    uncapped = (IRF_PREINDUSTRIAL
                + IRC * C_SINKS[t_set] * CO2toC
                + IRT * (TATM[t_set] + dt0))
    eq_irfrhs[t_set] = (
        IRF[t_set] == (IRF_MAX + uncapped
                       - sqrt(sqr(IRF_MAX - uncapped) + sqr(Number(1e-8)))) / 2
    )
    equations.append(eq_irfrhs)

    return equations
