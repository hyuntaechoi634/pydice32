"""
Stratospheric Aerosol Injection (SAI) geoengineering module.

Based on GAMS mod_sai.gms from RICE50x.

This module implements both the 'g0' experiment (uniform solar constant
reduction) and the full 'g6' multi-latitude injection emulator.  Select
via ``cfg.sai_experiment``:

  - "g0": Simplified mode.  SAI reduces global radiative forcing uniformly.
          No regional injection latitudes or emulator.  FORC is reduced by
          geoeng_forcing * W_SAI.

  - "g6": Full emulator.  SAI at 9 injection latitudes (60S..60N) produces
          region-specific temperature and precipitation responses via a
          linear emulator.  Regional downscaling is offset by
          DTEMP_REGION_SAI and DPRECIP_REGION_SAI.  The global forcing
          channel is DISABLED (geoeng_forcing=0); the temperature effect
          goes through the regional emulator instead.

g6 Variables (additional to g0):
    SAI(t,n,inj)           -- TgS injected per region per latitude
    Z_SAI(t,inj)           -- zonal (total across regions) SAI per latitude
    DTEMP_REGION_SAI(t,n)  -- regional temperature reduction from SAI [deg C]
    DPRECIP_REGION_SAI(t,n)-- regional precipitation change from SAI [mm/yr]

g6 Equations:
    eqz_sai               -- Z_SAI = sum(n, SAI(t,n,inj))
    eqn_sai               -- N_SAI = sum(inj, SAI(t,n,inj))
    eq_temp_region_sai     -- DTEMP = sum(inj, Z_SAI * sai_temp / 12)
    eq_precip_region_sai   -- DPRECIP = sum(inj, Z_SAI * sai_precip / 12)
    eq_tempconstup (g6)    -- regional safe ramp-down limit
    eq_tempconstdo (g6)    -- regional safe ramp-up limit

g0 Variables:
    W_SAI(t)       -- global SAI deployment [TgS]
    N_SAI(t,n)     -- regional SAI contribution [TgS]
    COST_SAI(t,n)  -- SAI costs [T$]

g0 Equations:
    eqw_sai         -- W_SAI = sum(n, N_SAI)
    eq_sai_cost     -- cost = (sai_cost_tgs/residence/1000) * N_SAI
    eq_tempconstup  -- TATM(t+1) >= TATM(t) - safe_temp*tstep/10
    eq_tempconstdo  -- TATM(t+1) <= TATM(t) + max_warming_projected

Integration points:
    - SAI affects global forcing (g0):  FORC += geoeng_forcing * W_SAI
      (patched in hub_climate.py when "W_SAI" in v)
    - SAI affects regional temperature (g6): eq_temp_region subtracts
      DTEMP_REGION_SAI, eq_precip_region adds DPRECIP_REGION_SAI
      (patched in mod_climate_regional.py when "DTEMP_REGION_SAI" in v)
    - SAI damage in hub_impact.py eq_damfrac_nobnd:
        + damage_geoeng_amount * (W_SAI/12)^2
    - COST_SAI enters eq_yy: Y = YNET - ABATECOST - ABCOSTLAND - COST_SAI

GAMS scalar parameters:
    sai_cost_tgs           = 10     [billion USD/TgS]
    geoeng_forcing         = -0.2   [W/m^2 per TgS]  (0 for g6)
    geoeng_residence_in_atm = 2     [years]
    safe_temp              = 0.3    [deg C per decade max temp change]
    max_warming_projected  = 0.2    [deg C per period BAU reference]
    geoeng_start           = 2050
    geoeng_end             = 2200
    sai_damage_coef        = 0.03   [g0 only: fraction GDP loss for 12 TgS]
"""

from gamspy import Variable, Equation, Set, Parameter, Ord, Card, Sum

# --------------- Default SAI parameters (GAMS mod_sai.gms) ---------------

SAI_COST_TGS = 10          # billion USD / TgS
GEOENG_FORCING_G0 = -0.2   # W/m^2 per TgS  (used by g0; g6 sets this to 0)
GEOENG_FORCING_G6 = 0.0    # g6 disables global forcing channel
GEOENG_RESIDENCE = 2       # atmospheric residence time [years]
SAFE_TEMP = 0.3            # max deg C change per decade
MAX_WARMING_PROJECTED = 0.2  # max warming per period (BAU reference)
SAI_DAMAGE_COEF_G0 = 0.03  # damage coefficient for g0: 3% GDP loss for 12 TgS/yr

# Injection latitudes
INJ_LABELS = ["60S", "45S", "30S", "15S", "0", "15N", "30N", "45N", "60N"]
NUM_TO_INJ = {
    "60S": -60, "45S": -45, "30S": -30, "15S": -15, "0": 0,
    "15N": 15, "30N": 30, "45N": 45, "60N": 60,
}

# Global temperature change from 12 TgS/yr SAI single-point injection
# (GAMS mod_sai.gms lines 92-100)
SAI_TEMP_GLOBAL = {
    "60S": 0.95, "45S": 1.2, "30S": 1.3, "15S": 1.12,
    "0": 0.93,
    "15N": 1.09, "30N": 1.28, "45N": 1.22, "60N": 1.06,
}

def get_geoeng_forcing(cfg):
    """Return the geoengineering forcing value based on SAI experiment mode.

    g0: uniform forcing reduction (-0.2 W/m^2 per TgS).
    g6: forcing handled by regional emulator, so global channel is 0.
    """
    experiment = getattr(cfg, "sai_experiment", "g0")
    if experiment == "g6":
        return GEOENG_FORCING_G6
    return GEOENG_FORCING_G0


def _get_experiment(cfg):
    """Return the SAI experiment mode from config."""
    return getattr(cfg, "sai_experiment", "g0")


def declare_vars(m, sets, params, cfg, v):
    """Create SAI variables, set bounds/starting values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict
    cfg : Config  -- expects cfg.sai, cfg.sai_start, cfg.sai_end, cfg.sai_experiment
    v : dict of all variables (mutated)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    experiment = _get_experiment(cfg)

    # Store the forcing value in v so hub_climate can read it without
    # relying on module-level mutable global state.
    v["_GEOENG_FORCING"] = get_geoeng_forcing(cfg)

    sai_start = getattr(cfg, "sai_start", 2050)
    sai_end = getattr(cfg, "sai_end", 2200)
    sai_mode = getattr(cfg, "sai_mode", "free")

    # ── Common variables (both g0 and g6) ─────────────────────
    W_SAI = Variable(m, name="W_SAI", domain=[t_set])
    N_SAI = Variable(m, name="N_SAI", domain=[t_set, n_set])
    COST_SAI = Variable(m, name="COST_SAI", domain=[t_set, n_set])

    W_SAI.l[t_set] = 0
    N_SAI.l[t_set, n_set] = 0
    COST_SAI.l[t_set, n_set] = 0

    W_SAI.lo[t_set] = 0
    N_SAI.lo[t_set, n_set] = 0
    N_SAI.up[t_set, n_set] = 100
    COST_SAI.lo[t_set, n_set] = 0

    # Fix N_SAI to zero outside deployment window (for both modes)
    for tp in range(1, cfg.T + 1):
        yr = cfg.year(tp)
        if yr < sai_start or yr > sai_end:
            N_SAI.fx[str(tp), n_set] = 0

    # GAMS can_deploy gate: restrict which regions can inject
    # GAMS mod_sai.gms lines 9, 31-46: can_inject(n) gates N_SAI.fx
    can_deploy = getattr(cfg, "can_deploy", "no")
    region_names = [str(r) for r in n_set.records.iloc[:, 0]]
    if can_deploy == "no":
        # No region can deploy SAI
        N_SAI.fx[t_set, n_set] = 0
    elif can_deploy == "all":
        pass  # all regions can deploy (no restriction)
    elif can_deploy == "coal":
        # Coalition-based deployment: only regions in the SAI coalition can inject
        # GAMS: can_inject(n) = yes$(coalitions("%sai_coalition%",n))
        coalition_def = getattr(cfg, "coalition_def", None)
        sai_coalition = getattr(cfg, "sai_coalition", "sai")
        coal_members = set()
        if coalition_def is not None and isinstance(coalition_def, dict):
            coal_members = set(coalition_def.get(sai_coalition, []))
        for rn in region_names:
            if rn not in coal_members:
                N_SAI.fx[t_set, rn] = 0
    else:
        # Specific region name: only that region can deploy
        for rn in region_names:
            if rn.lower() != can_deploy.lower():
                N_SAI.fx[t_set, rn] = 0

    v["W_SAI"] = W_SAI
    v["N_SAI"] = N_SAI
    v["COST_SAI"] = COST_SAI

    # ── g6-specific variables ─────────────────────────────────
    if experiment == "g6":
        # Injection latitude set (reuse if already created by _create_parameters)
        if "inj_set" in sets:
            inj_set = sets["inj_set"]
        else:
            inj_set = Set(m, name="inj", records=INJ_LABELS,
                          description="possible injection points for SAI")
            sets["inj_set"] = inj_set
        v["inj_set"] = inj_set

        # SAI(t,n,inj) -- per-region per-latitude injection
        SAI = Variable(m, name="SAI", domain=[t_set, n_set, inj_set])
        SAI.l[t_set, n_set, inj_set] = 0
        SAI.lo[t_set, n_set, inj_set] = 0

        # Z_SAI(t,inj) -- zonal total
        Z_SAI = Variable(m, name="Z_SAI", domain=[t_set, inj_set])
        Z_SAI.l[t_set, inj_set] = 0
        Z_SAI.lo[t_set, inj_set] = 0

        # DTEMP_REGION_SAI(t,n) -- regional temperature reduction
        DTEMP_REGION_SAI = Variable(m, name="DTEMP_REGION_SAI",
                                    domain=[t_set, n_set])
        DTEMP_REGION_SAI.l[t_set, n_set] = 0

        # DPRECIP_REGION_SAI(t,n) -- regional precipitation change
        DPRECIP_REGION_SAI = Variable(m, name="DPRECIP_REGION_SAI",
                                      domain=[t_set, n_set])
        DPRECIP_REGION_SAI.l[t_set, n_set] = 0

        # Determine belong_inj: which (n, inj) pairs are allowed
        # Build a parameter to encode belong_inj as 0/1 for equation conditions
        sai_data = params.get("sai_data", {})
        sai_temp_data = sai_data.get("sai_temp", {})

        belong_inj_records = []
        region_names = [str(r) for r in n_set.records.iloc[:, 0]]

        if sai_mode == "free":
            for rn in region_names:
                for inj in INJ_LABELS:
                    belong_inj_records.append((rn, inj, 1.0))
        elif sai_mode == "equator":
            for rn in region_names:
                belong_inj_records.append((rn, "0", 1.0))
        elif sai_mode == "tropics":
            for rn in region_names:
                belong_inj_records.append((rn, "15N", 1.0))
                belong_inj_records.append((rn, "15S", 1.0))
        elif sai_mode == "symmetric":
            for rn in region_names:
                for inj in ["15N", "15S", "30N", "30S"]:
                    belong_inj_records.append((rn, inj, 1.0))
        elif sai_mode == "max_efficiency":
            # For each region, only allow the injection latitude with
            # maximum temperature effect
            for rn in region_names:
                best_inj = None
                best_val = -1
                for inj in INJ_LABELS:
                    val = sai_temp_data.get((rn, inj), 0.0)
                    if val > best_val:
                        best_val = val
                        best_inj = inj
                if best_inj is not None:
                    belong_inj_records.append((rn, best_inj, 1.0))
        else:
            # Default: all allowed
            for rn in region_names:
                for inj in INJ_LABELS:
                    belong_inj_records.append((rn, inj, 1.0))

        par_belong_inj = Parameter(m, name="belong_inj",
                                   domain=[n_set, inj_set],
                                   records=belong_inj_records)
        v["par_belong_inj"] = par_belong_inj

        # Fix SAI to zero outside deployment window and for
        # disallowed (n, inj) combinations
        for tp in range(1, cfg.T + 1):
            yr = cfg.year(tp)
            if yr < sai_start or yr > sai_end:
                SAI.fx[str(tp), n_set, inj_set] = 0

        # Note: GAMSPy does not support .fx conditional on a parameter
        # easily at the Python level for individual records.
        # The constraint is enforced via the belong_inj parameter in
        # the equation definitions (eqz_sai, eqn_sai).
        # We fix SAI=0 for disallowed combinations by iterating.
        _belong_set = set()
        for rn, inj, val in belong_inj_records:
            if val > 0:
                _belong_set.add((rn, inj))

        for tp in range(1, cfg.T + 1):
            yr = cfg.year(tp)
            if yr < sai_start or yr > sai_end:
                continue
            for rn in region_names:
                for inj in INJ_LABELS:
                    if (rn, inj) not in _belong_set:
                        SAI.fx[str(tp), rn, inj] = 0

        v["SAI"] = SAI
        v["Z_SAI"] = Z_SAI
        v["DTEMP_REGION_SAI"] = DTEMP_REGION_SAI
        v["DPRECIP_REGION_SAI"] = DPRECIP_REGION_SAI


def define_eqs(m, sets, params, cfg, v):
    """Create SAI equations.

    For g0: simplified global equations.
    For g6: full regional emulator equations.

    Parameters
    ----------
    m : Container
    sets : dict
    params : dict
    cfg : Config
    v : dict of all variables

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    experiment = _get_experiment(cfg)
    TSTEP = cfg.TSTEP

    W_SAI = v["W_SAI"]
    N_SAI = v["N_SAI"]
    COST_SAI = v["COST_SAI"]
    TATM = v["TATM"]

    # Cost per TgS per year in T$ (accounting for atmospheric residence)
    # GAMS: (sai_cost_tgs / geoeng_residence_in_atm) / 1000
    cost_per_tgs = (SAI_COST_TGS / GEOENG_RESIDENCE) / 1000

    equations = []

    # ── eq_sai_cost (both modes) ─────────────────────────────
    eq_sai_cost = Equation(m, name="eq_sai_cost", domain=[t_set, n_set])
    eq_sai_cost[t_set, n_set] = (
        COST_SAI[t_set, n_set] == cost_per_tgs * N_SAI[t_set, n_set]
    )
    equations.append(eq_sai_cost)

    if experiment == "g6":
        # ── g6 experiment: full regional emulator ─────────────
        inj_set = sets["inj_set"]
        SAI = v["SAI"]
        Z_SAI = v["Z_SAI"]
        DTEMP_REGION_SAI = v["DTEMP_REGION_SAI"]
        DPRECIP_REGION_SAI = v["DPRECIP_REGION_SAI"]
        TEMP_REGION = v["TEMP_REGION"]

        # Load emulator parameters from params
        sai_data = params.get("sai_data", {})
        par_sai_temp = params.get("par_sai_temp")
        par_sai_precip = params.get("par_sai_precip")
        par_beta_temp = params.get("par_beta_temp")

        # eqz_sai: Z_SAI(t,inj) = sum(n, SAI(t,n,inj))
        eqz_sai = Equation(m, name="eqz_sai", domain=[t_set, inj_set])
        eqz_sai[t_set, inj_set] = (
            Z_SAI[t_set, inj_set] == Sum(n_set, SAI[t_set, n_set, inj_set])
        )
        equations.append(eqz_sai)

        # eqn_sai: N_SAI(t,n) = sum(inj, SAI(t,n,inj))
        eqn_sai = Equation(m, name="eqn_sai", domain=[t_set, n_set])
        eqn_sai[t_set, n_set] = (
            N_SAI[t_set, n_set] == Sum(inj_set, SAI[t_set, n_set, inj_set])
        )
        equations.append(eqn_sai)

        # eqw_sai: W_SAI(t) = sum(inj, Z_SAI(t,inj))
        eqw_sai = Equation(m, name="eqw_sai", domain=[t_set])
        eqw_sai[t_set] = W_SAI[t_set] == Sum(inj_set, Z_SAI[t_set, inj_set])
        equations.append(eqw_sai)

        # eq_temp_region_sai: DTEMP_REGION_SAI(t,n) =
        #   sum(inj, Z_SAI(t,inj) * sai_temp(n,inj) / 12)
        eq_temp_region_sai = Equation(m, name="eq_temp_region_sai",
                                      domain=[t_set, n_set])
        eq_temp_region_sai[t_set, n_set] = (
            DTEMP_REGION_SAI[t_set, n_set] == Sum(
                inj_set,
                Z_SAI[t_set, inj_set] * par_sai_temp[n_set, inj_set] / 12
            )
        )
        equations.append(eq_temp_region_sai)

        # eq_precip_region_sai: DPRECIP_REGION_SAI(t,n) =
        #   sum(inj, Z_SAI(t,inj) * sai_precip(n,inj) / 12)
        eq_precip_region_sai = Equation(m, name="eq_precip_region_sai",
                                        domain=[t_set, n_set])
        eq_precip_region_sai[t_set, n_set] = (
            DPRECIP_REGION_SAI[t_set, n_set] == Sum(
                inj_set,
                Z_SAI[t_set, inj_set] * par_sai_precip[n_set, inj_set] / 12
            )
        )
        equations.append(eq_precip_region_sai)

        # Temperature safety constraints (g6 version: regional TEMP_REGION)
        # eq_tempconstup: TEMP_REGION(t+1,n) >= TEMP_REGION(t,n) - safe_temp*tstep/10
        safe_drop = SAFE_TEMP * TSTEP / 10
        eq_tempconstup = Equation(m, name="eq_tempconstup",
                                  domain=[t_set, n_set])
        eq_tempconstup[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
            TEMP_REGION[t_set.lead(1), n_set]
            >= TEMP_REGION[t_set, n_set] - safe_drop
        )
        equations.append(eq_tempconstup)

        # eq_tempconstdo: TEMP_REGION(t+1,n) <= TEMP_REGION(t,n) +
        #   max_warming_projected * beta_temp(n)
        # GAMS: max_warming_projected * climate_region_coef('beta_temp', n)
        eq_tempconstdo = Equation(m, name="eq_tempconstdo",
                                  domain=[t_set, n_set])
        eq_tempconstdo[t_set, n_set].where[Ord(t_set) < Card(t_set)] = (
            TEMP_REGION[t_set.lead(1), n_set]
            <= TEMP_REGION[t_set, n_set]
            + MAX_WARMING_PROJECTED * par_beta_temp[n_set]
        )
        equations.append(eq_tempconstdo)

    else:
        # ── g0 experiment: simplified global mode ─────────────

        # eqw_sai: W_SAI(t) = sum(n, N_SAI(t,n))
        eqw_sai = Equation(m, name="eqw_sai", domain=[t_set])
        eqw_sai[t_set] = W_SAI[t_set] == Sum(n_set, N_SAI[t_set, n_set])
        equations.append(eqw_sai)

        # Temperature safety constraints (g0 version: global TATM)
        # eq_tempconstup: TATM(t+1) >= TATM(t) - safe_temp * tstep/10
        safe_drop = SAFE_TEMP * TSTEP / 10
        eq_tempconstup = Equation(m, name="eq_tempconstup", domain=[t_set])
        eq_tempconstup[t_set].where[Ord(t_set) < Card(t_set)] = (
            TATM[t_set.lead(1)] >= TATM[t_set] - safe_drop
        )
        equations.append(eq_tempconstup)

        # eq_tempconstdo: TATM(t+1) <= TATM(t) + max_warming_projected
        eq_tempconstdo = Equation(m, name="eq_tempconstdo", domain=[t_set])
        eq_tempconstdo[t_set].where[Ord(t_set) < Card(t_set)] = (
            TATM[t_set.lead(1)] <= TATM[t_set] + MAX_WARMING_PROJECTED
        )
        equations.append(eq_tempconstdo)

    return equations
