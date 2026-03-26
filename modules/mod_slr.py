"""
Sea-Level Rise (SLR) module.

Based on GAMS mod_slr.gms from RICE50x (Li et al., 2020).

Computes global mean sea level rise from four components:
  1. Thermal expansion (thermo) -- driven by ocean temperature layers
  2. Greenland ice sheet melt (gris)
  3. Antarctic ice sheet melt (antis)
  4. Mountain glaciers and ice cap melting (mg)

The ocean has three temperature layers (ml, tc, dp) that respond to
atmospheric temperature with different timescales.

The cumulative-sum equations (gris, antis, mg) from GAMS use preds(t,tt)
which sums over all preceding periods. In GAMSPy these are reformulated
as recursive difference equations using lag/lead.
"""

from gamspy import Variable, Equation, Set, Ord, Card


# GAMS scalar parameters
SLR_BETA = 1.33
SLR_D_ML = 50       # m
SLR_D_TC = 500      # m
SLR_D_DP = 3150     # m
SLR_W_E = 0.5e-6    # m/s
SLR_W_D = 0.2e-6    # m/s
SLR_GAMMA_ML = 2.4e-4   # 1/K
SLR_GAMMA_TC = 2.0e-4   # 1/K
SLR_GAMMA_DP = 2.1e-4   # 1/K
NBSECYEAR = 31556952    # seconds per year
SLR_GRIS0 = 0.0055      # cm (initial Greenland contribution)
SLR_ANTIS0 = 0.009      # cm (initial Antarctic contribution)
SLR_MG0 = 0.026         # cm (initial mountain glaciers contribution)


def declare_vars(m, sets, params, cfg, v):
    """Create SLR variables.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated)
    """
    t_set = sets["t_set"]

    # SLR component set
    slr_w = Set(m, name="slr_w", records=["thermo", "gris", "antis", "mg"])
    sets["slr_w"] = slr_w

    # Ocean temperature layer set
    slr_m = Set(m, name="slr_m", records=["ml", "tc", "dp"])
    sets["slr_m"] = slr_m

    # Variables
    TEMP_SLR = Variable(m, name="TEMP_SLR", domain=[slr_m, t_set])
    GMSLR = Variable(m, name="GMSLR", domain=[t_set])
    SLR = Variable(m, name="SLR", domain=[slr_w, t_set])

    # Bounds
    GMSLR.lo[t_set] = 0
    GMSLR.up[t_set] = 3
    SLR.lo[slr_w, t_set] = 0
    SLR.up[slr_w, t_set] = 3
    SLR.up["mg", t_set] = 0.4  # max from mountain glaciers

    # Starting values
    TEMP_SLR.l[slr_m, t_set] = 0
    TEMP_SLR.l["tc", "1"] = 0
    TEMP_SLR.l["dp", "1"] = 0
    GMSLR.l[t_set] = 0
    SLR.l[slr_w, t_set] = 0

    # Register
    v["TEMP_SLR"] = TEMP_SLR
    v["GMSLR"] = GMSLR
    v["SLR"] = SLR


def define_eqs(m, sets, params, cfg, v):
    """Create SLR equations.

    The GAMS preds(t,tt)-based cumulative sums are reformulated as recursive
    equations: SLR(component, t+1) = SLR(component, t) + 5-year increment.

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    slr_w = sets["slr_w"]
    slr_m = sets["slr_m"]

    TSTEP = cfg.TSTEP  # = 5

    TEMP_SLR = v["TEMP_SLR"]
    GMSLR = v["GMSLR"]
    SLR = v["SLR"]
    TATM = v["TATM"]

    equations = []

    # ------------------------------------------------------------------
    # eqslr_tot: GMSLR(t) = sum(w, SLR(w,t))
    # ------------------------------------------------------------------
    eqslr_tot = Equation(m, name="eqslr_tot", domain=[t_set])
    eqslr_tot[t_set] = GMSLR[t_set] == (
        SLR["thermo", t_set] + SLR["gris", t_set]
        + SLR["antis", t_set] + SLR["mg", t_set]
    )
    equations.append(eqslr_tot)

    # ------------------------------------------------------------------
    # eqslr_gris: Greenland ice sheet melt (recursive)
    # GAMS cumulative: SLR('gris',t) = sum(tt$preds, 5*(71.5*TATM+20.4*TATM^2+2.8*TATM^3))/3.61e5 + gris0
    # Recursive: SLR('gris',t+1) = SLR('gris',t) + 5*(71.5*TATM(t)+20.4*TATM(t)^2+2.8*TATM(t)^3)/3.61e5
    # Initial: SLR('gris','1') = gris0
    # ------------------------------------------------------------------
    SLR.fx["gris", "1"] = SLR_GRIS0
    eqslr_gris = Equation(m, name="eqslr_gris", domain=[t_set])
    eqslr_gris[t_set].where[Ord(t_set) < Card(t_set)] = (
        SLR["gris", t_set.lead(1)] == SLR["gris", t_set]
        + TSTEP * (71.5 * TATM[t_set]
                   + 20.4 * TATM[t_set] ** 2
                   + 2.8 * TATM[t_set] ** 3)
        / 3.61e5
    )
    equations.append(eqslr_gris)

    # ------------------------------------------------------------------
    # eqslr_antis: Antarctic ice sheet melt (recursive)
    # Recursive: SLR('antis',t+1) = SLR('antis',t) + 5*(0.00074 + 0.00022*TATM(t))
    # ------------------------------------------------------------------
    SLR.fx["antis", "1"] = SLR_ANTIS0
    eqslr_antis = Equation(m, name="eqslr_antis", domain=[t_set])
    eqslr_antis[t_set].where[Ord(t_set) < Card(t_set)] = (
        SLR["antis", t_set.lead(1)] == SLR["antis", t_set]
        + TSTEP * (0.00074 + 0.00022 * TATM[t_set])
    )
    equations.append(eqslr_antis)

    # ------------------------------------------------------------------
    # eqslr_mg: Mountain glaciers (self-limiting, recursive)
    # Recursive: SLR('mg',t+1) = SLR('mg',t) + 5*0.0008*TATM(t)*(1-SLR('mg',t)/0.41)^1.646
    # ------------------------------------------------------------------
    SLR.fx["mg", "1"] = SLR_MG0
    eqslr_mg = Equation(m, name="eqslr_mg", domain=[t_set])
    eqslr_mg[t_set].where[Ord(t_set) < Card(t_set)] = (
        SLR["mg", t_set.lead(1)] == SLR["mg", t_set]
        + TSTEP * 0.0008 * TATM[t_set]
        * (1 - SLR["mg", t_set] / 0.41) ** 1.646
    )
    equations.append(eqslr_mg)

    # ------------------------------------------------------------------
    # eqtemp_ml: Mixed layer temperature
    # GAMS: TEMP('ml',t) = TATM(t) / slr_beta
    # ------------------------------------------------------------------
    eqtemp_ml = Equation(m, name="eqtemp_ml", domain=[t_set])
    eqtemp_ml[t_set] = (
        TEMP_SLR["ml", t_set] == TATM[t_set] / SLR_BETA
    )
    equations.append(eqtemp_ml)

    # ------------------------------------------------------------------
    # eqtemp_tc: Thermocline diffusion (recursive)
    # GAMS: TEMP('tc',tp1) = TEMP('tc',t)
    #        + (w_e/d_tc*(TEMP_ml - TEMP_tc) - w_d/d_tc*(TEMP_tc - TEMP_dp)) * nbsecyear * tlen
    # ------------------------------------------------------------------
    TEMP_SLR.fx["tc", "1"] = 0
    eqtemp_tc = Equation(m, name="eqtemp_tc", domain=[t_set])
    eqtemp_tc[t_set].where[Ord(t_set) < Card(t_set)] = (
        TEMP_SLR["tc", t_set.lead(1)] == TEMP_SLR["tc", t_set]
        + (SLR_W_E / SLR_D_TC * (TEMP_SLR["ml", t_set] - TEMP_SLR["tc", t_set])
           - SLR_W_D / SLR_D_TC * (TEMP_SLR["tc", t_set] - TEMP_SLR["dp", t_set]))
        * NBSECYEAR * TSTEP
    )
    equations.append(eqtemp_tc)

    # ------------------------------------------------------------------
    # eqtemp_dp: Deep ocean diffusion (recursive)
    # GAMS: TEMP('dp',tp1) = TEMP('dp',t)
    #        + (w_d/d_dp*(TEMP_tc - TEMP_dp)) * nbsecyear * tlen
    # ------------------------------------------------------------------
    TEMP_SLR.fx["dp", "1"] = 0
    eqtemp_dp = Equation(m, name="eqtemp_dp", domain=[t_set])
    eqtemp_dp[t_set].where[Ord(t_set) < Card(t_set)] = (
        TEMP_SLR["dp", t_set.lead(1)] == TEMP_SLR["dp", t_set]
        + SLR_W_D / SLR_D_DP
        * (TEMP_SLR["tc", t_set] - TEMP_SLR["dp", t_set])
        * NBSECYEAR * TSTEP
    )
    equations.append(eqtemp_dp)

    # ------------------------------------------------------------------
    # eqslr_thermo: Thermal expansion
    # GAMS: SLR('thermo',t) = gamma_ml*TEMP_ml*d_ml + gamma_tc*TEMP_tc*d_tc + gamma_dp*TEMP_dp*d_dp
    # ------------------------------------------------------------------
    eqslr_thermo = Equation(m, name="eqslr_thermo", domain=[t_set])
    eqslr_thermo[t_set] = (
        SLR["thermo", t_set] ==
        SLR_GAMMA_ML * TEMP_SLR["ml", t_set] * SLR_D_ML
        + SLR_GAMMA_TC * TEMP_SLR["tc", t_set] * SLR_D_TC
        + SLR_GAMMA_DP * TEMP_SLR["dp", t_set] * SLR_D_DP
    )
    equations.append(eqslr_thermo)

    return equations
