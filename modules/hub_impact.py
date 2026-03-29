"""
Hub impact module: damage fraction and damages.

Variables: OMEGA, DAMFRAC, DAMFRAC_UNBOUNDED, DAMFRAC_UPBOUND, DAMAGES, OMEGA_SLR
Equations: eq_damfrac_nobnd, eq_damfrac, eq_damfrac_upbnd (conditional), eq_damages,
           eq_omega_slr (when cfg.slr=True)
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Ord, Number, Parameter
from gamspy.math import sqrt, sqr

try:
    from gamspy.math import errorf
except ImportError:
    # Fallback: errorf(x) = normal CDF = 0.5*(1+erf(x/sqrt(2)))
    # GAMSPy may expose it as 'normal_cdf' or similar in some versions.
    # If errorf is unavailable, threshold_damage will raise ImportError
    # at equation-build time (see define_eqs).
    errorf = None


def declare_vars(m, sets, params, cfg, v):
    """Create impact hub variables, set bounds/starting values/fixed values.

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
    OMEGA = Variable(m, name="OMEGA", domain=[t_set, n_set])
    DAMFRAC = Variable(m, name="DAMFRAC", domain=[t_set, n_set])
    DAMFRAC_UNBOUNDED = Variable(m, name="DAMFRAC_UNBOUNDED", domain=[t_set, n_set])
    DAMFRAC_UPBOUND = Variable(m, name="DAMFRAC_UPBOUND", domain=[t_set, n_set])
    # DAMAGES should be free (can be negative for warming benefits)
    DAMAGES = Variable(m, name="DAMAGES", domain=[t_set, n_set])

    # ------------------------------------------------------------------
    # Bounds and starting values
    # ------------------------------------------------------------------
    OMEGA.l[t_set, n_set] = 0
    # GAMS hub_impact.gms line 66: OMEGA.lo = (-1 + 1e-5) only for full omega
    if getattr(cfg, "omega_eq", "simple") == "full":
        OMEGA.lo[t_set, n_set] = -1 + 1e-5
    OMEGA.fx["1", n_set] = 0
    DAMFRAC_UNBOUNDED.l[t_set, n_set] = 0
    DAMFRAC_UNBOUNDED.fx["1", n_set] = 0
    DAMFRAC.l[t_set, n_set] = 0
    DAMFRAC.lo[t_set, n_set] = -1.01
    DAMFRAC.up[t_set, n_set] = 0.91
    DAMFRAC_UPBOUND.l[t_set, n_set] = 0
    DAMAGES.l[t_set, n_set] = 0
    # GAMS hub_impact does NOT fix DAMAGES for t=1.
    # DAMAGES("1") is determined by eq_damages = YGROSS * DAMFRAC, with
    # DAMFRAC_UNBOUNDED.fx("1")=0 ensuring zero first-period damages.

    # SLR damage contribution to OMEGA (active when cfg.slr=True)
    # GAMS mod_impact_coacch.gms line 83:
    #   + comega_qmul * (comega_slr('b1') * GMSLR + comega_slr('b2') * GMSLR^2)
    OMEGA_SLR = Variable(m, name="OMEGA_SLR", domain=[t_set, n_set])
    OMEGA_SLR.l[t_set, n_set] = 0
    OMEGA_SLR.fx["1", n_set] = 0

    # Register in shared variable dict
    v["OMEGA"] = OMEGA
    v["OMEGA_SLR"] = OMEGA_SLR
    v["DAMFRAC"] = DAMFRAC
    v["DAMFRAC_UNBOUNDED"] = DAMFRAC_UNBOUNDED
    v["DAMFRAC_UPBOUND"] = DAMFRAC_UPBOUND
    v["DAMAGES"] = DAMAGES


def define_eqs(m, sets, params, cfg, v):
    """Create impact hub equations.

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

    # Own variables
    OMEGA = v["OMEGA"]
    OMEGA_SLR = v["OMEGA_SLR"]
    DAMFRAC = v["DAMFRAC"]
    DAMFRAC_UNBOUNDED = v["DAMFRAC_UNBOUNDED"]
    DAMFRAC_UPBOUND = v["DAMFRAC_UPBOUND"]
    DAMAGES = v["DAMAGES"]

    # Cross-module variables
    YGROSS = v["YGROSS"]
    TATM = v["TATM"]

    policy_with_damages = cfg.policy_with_damages
    damage_cap = cfg.damage_cap

    delta_dam = 1e-2  # smooth NLP approximation tolerance
    max_gain = 1.0
    max_damage = 0.9

    # ------------------------------------------------------------------
    # SLR damage equation (when cfg.slr=True)
    # GAMS mod_impact_coacch.gms line 83:
    #   + comega_qmul('%damcostslr%',n,'%damcostpb%')
    #     * (comega_slr('%damcostslr%',n,'b1') * GMSLR(t)
    #        + comega_slr('%damcostslr%',n,'b2') * GMSLR(t)**2)
    # ------------------------------------------------------------------
    slr_equations = []
    if getattr(cfg, "slr", False) and "GMSLR" in v:
        GMSLR = v["GMSLR"]
        # Load comega_slr data
        comega_slr_file = os.path.join(
            cfg.data_dir, "data_mod_damage", "comega_slr.csv")
        damcostslr = getattr(cfg, "damcostslr", "COACCH_SLR_Ad")
        slr_b1_recs = []
        slr_b2_recs = []
        if os.path.exists(comega_slr_file):
            try:
                df_slr = pd.read_csv(comega_slr_file)
                region_names = [r for r in
                    params["par_pop"].records["n"].unique()]
                rn_lower = {r.lower(): r for r in region_names}
                for _, row in df_slr.iterrows():
                    scenario = str(row["Dim1"])
                    if scenario != damcostslr:
                        continue
                    n_val = str(row["n"]).lower()
                    coef_type = str(row["Dim3"])
                    val = float(row["Val"])
                    if n_val not in rn_lower:
                        continue
                    r = rn_lower[n_val]
                    if coef_type == "b1":
                        # GAMS: no positive impacts from SLR
                        # comega_slr('COACCH_SLR_Ad',n,'b1')$(... le 0) = 0
                        slr_b1_recs.append((r, max(val, 0.0)))
                    elif coef_type == "b2":
                        slr_b2_recs.append((r, val))
            except Exception:
                pass

        par_slr_b1 = Parameter(m, name="comega_slr_b1", domain=[n_set],
                               records=slr_b1_recs if slr_b1_recs else None)
        par_slr_b2 = Parameter(m, name="comega_slr_b2", domain=[n_set],
                               records=slr_b2_recs if slr_b2_recs else None)

        # eq_omega_slr: OMEGA_SLR = b1 * GMSLR + b2 * GMSLR^2
        # Subtract the base-period value (GMSLR('2')) as in the GAMS
        eq_omega_slr = Equation(m, name="eq_omega_slr",
                                domain=[t_set, n_set])
        eq_omega_slr[t_set, n_set].where[Ord(t_set) > 1] = (
            OMEGA_SLR[t_set, n_set] ==
            (par_slr_b1[n_set] * GMSLR[t_set]
             + par_slr_b2[n_set] * GMSLR[t_set] ** 2)
            - (par_slr_b1[n_set] * GMSLR["2"]
               + par_slr_b2[n_set] * GMSLR["2"] ** 2)
        )
        slr_equations.append(eq_omega_slr)
    else:
        # No SLR: fix OMEGA_SLR to zero
        eq_omega_slr = Equation(m, name="eq_omega_slr",
                                domain=[t_set, n_set])
        eq_omega_slr[t_set, n_set] = OMEGA_SLR[t_set, n_set] == 0
        slr_equations.append(eq_omega_slr)

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_damfrac_nobnd: DAMFRAC_UNBOUNDED = 1 - 1/(1+OMEGA+OMEGA_SLR)
    #   + optional threshold_damage term
    #   + optional gradient_damage term
    #
    # GAMS hub_impact.gms lines 96-101:
    #   DAMFRAC_UNBOUNDED = 1 - 1/(1+OMEGA/adaptation)
    #     + threshold_d * errorf((TATM - threshold_temp)/threshold_sigma)
    #     + gradient_d * power(sqrt(sqr(TATM-TATM(tm1)) + sqr(delta))/0.35, 4)
    eq_damfrac_nobnd = Equation(m, name="eq_damfrac_nobnd", domain=[t_set, n_set])
    if policy_with_damages:
        # Build the base OMEGA expression.
        # Issue 10: when adaptation is active, OMEGA is divided by
        #   (1 + Q_ADA('ada',t,n)^exp) per GAMS hub_impact.gms:
        #   DAMFRAC_UNBOUNDED = 1 - 1/(1 + OMEGA/(1 + Q_ADA^exp))
        # Without adaptation: 1 - 1/(1 + OMEGA + OMEGA_SLR)
        # GAMS: OMEGA includes both temperature and SLR damage.
        # OMEGA_SLR is the SLR contribution (zero when slr=False).
        omega_expr = OMEGA[t_set, n_set] + OMEGA_SLR[t_set, n_set]
        if "Q_ADA" in v:
            Q_ADA = v["Q_ADA"]
            # GAMS hub_impact.gms line 97:
            #   OMEGA / (1 + Q_ADA('ada',t,n)**ces_ada('exp',n))$(OMEGA.l(t,n) gt 0)
            # ces_ada('exp',n) is per-region; use par_ces_ada_exp Parameter.
            par_ces_ada_exp = params.get("par_ces_ada_exp")
            if par_ces_ada_exp is not None:
                ada_exp_expr = par_ces_ada_exp[n_set]
            else:
                ada_exp_expr = 2  # fallback scalar

            # GAMS $(OMEGA.l(t,n) gt 0) gate: adaptation divisor only applies
            # when OMEGA is positive (net damages). When OMEGA <= 0 (warming
            # benefits), no adaptation is needed.
            #
            # par_omega_positive is an indicator parameter [t, n] that is 1
            # when OMEGA.l > 0, else 0. Initialized to 1 (conservative: assume
            # damages), updated in _before_solve for iterative solver.
            #
            # Formula: divisor = 1 + Q_ADA^exp * omega_gate
            # When omega_gate=1: full adaptation.  When omega_gate=0: divisor=1.
            par_omega_gate = params.get("par_omega_positive")
            if par_omega_gate is not None:
                ada_divisor = 1 + Q_ADA["ada", t_set, n_set] ** ada_exp_expr * par_omega_gate[t_set, n_set]
            else:
                ada_divisor = 1 + Q_ADA["ada", t_set, n_set] ** ada_exp_expr
            base_expr = 1 - 1 / (1 + omega_expr / ada_divisor)
        else:
            base_expr = 1 - 1 / (1 + omega_expr)

        # Issue 9: SAI damage term.  GAMS hub_impact.gms line 101:
        #   + damage_geoeng_amount(n) * power(W_SAI(t) / 12, 2)
        # For g0: damage_geoeng_amount = sai_damage_coef (default 0.03)
        # For g6: damage_geoeng_amount = 0 (side effects handled by emulator)
        if "W_SAI" in v:
            W_SAI = v["W_SAI"]
            sai_dam_coef = getattr(cfg, "sai_damage_coef", 0.0)
            experiment = getattr(cfg, "sai_experiment", "g0")
            if experiment == "g6":
                # g6: side effects captured by regional emulator, no extra term
                sai_dam_coef = 0.0
            if sai_dam_coef > 0:
                base_expr = base_expr + sai_dam_coef * (W_SAI[t_set] / 12) ** 2

        # Threshold damage: errorf is the standard GAMS error function
        # (the normal CDF, equivalent to 0.5*(1+erf(x/sqrt(2))) ).
        # In GAMSPy, gamspy.math.errorf maps to the same GAMS function.
        if cfg.threshold_damage:
            if errorf is None:
                raise ImportError(
                    "threshold_damage requires gamspy.math.errorf "
                    "(standard normal CDF). Upgrade GAMSPy or disable "
                    "threshold_damage."
                )
            base_expr = base_expr + cfg.threshold_d * errorf(
                (TATM[t_set] - cfg.threshold_temp) / cfg.threshold_sigma
            )

        # Gradient damage: penalises rapid temperature changes
        # Uses smooth abs via sqrt(sqr(dT) + sqr(delta)) instead of |dT|
        if cfg.gradient_damage:
            base_expr = base_expr + cfg.gradient_d * (
                sqrt(sqr(TATM[t_set] - TATM[t_set.lag(1)])
                     + sqr(Number(delta_dam)))
                / 0.35
            ) ** 4

        eq_damfrac_nobnd[t_set, n_set].where[Ord(t_set) > 1] = (
            DAMFRAC_UNBOUNDED[t_set, n_set] == base_expr
        )
    else:
        eq_damfrac_nobnd[t_set, n_set] = DAMFRAC_UNBOUNDED[t_set, n_set] == 0

    # eq_damfrac: with or without damage cap
    eq_damfrac = Equation(m, name="eq_damfrac", domain=[t_set, n_set])
    equations = [eq_damfrac_nobnd, eq_damfrac]

    if damage_cap and policy_with_damages:
        # Smooth min/max bounds
        eq_damfrac_upbnd = Equation(m, name="eq_damfrac_upbnd", domain=[t_set, n_set])
        eq_damfrac_upbnd[t_set, n_set] = DAMFRAC_UPBOUND[t_set, n_set] == (
            DAMFRAC_UNBOUNDED[t_set, n_set] + max_damage
            - sqrt(sqr(DAMFRAC_UNBOUNDED[t_set, n_set] - max_damage) + sqr(Number(delta_dam)))
        ) / 2

        eq_damfrac[t_set, n_set] = DAMFRAC[t_set, n_set] == (
            DAMFRAC_UPBOUND[t_set, n_set] - max_gain
            + sqrt(sqr(DAMFRAC_UPBOUND[t_set, n_set] + max_gain) + sqr(Number(delta_dam)))
        ) / 2

        equations.append(eq_damfrac_upbnd)
    else:
        # No damage cap: DAMFRAC = DAMFRAC_UNBOUNDED
        eq_damfrac[t_set, n_set] = (
            DAMFRAC[t_set, n_set] == DAMFRAC_UNBOUNDED[t_set, n_set]
        )

    # eq_damages: DAMAGES = YGROSS * DAMFRAC
    # When impact_deciles is active, DAMAGES = sum(dist, DAMAGES_DIST) is defined
    # in mod_impact_deciles instead. Skip the aggregate equation here.
    if not getattr(cfg, "_decile_damages", False):
        eq_damages = Equation(m, name="eq_damages", domain=[t_set, n_set])
        if policy_with_damages:
            eq_damages[t_set, n_set] = (
                DAMAGES[t_set, n_set] == YGROSS[t_set, n_set] * DAMFRAC[t_set, n_set]
            )
        else:
            eq_damages[t_set, n_set] = DAMAGES[t_set, n_set] == 0
        equations.append(eq_damages)

    # Add SLR damage equations
    equations.extend(slr_equations)

    return equations
