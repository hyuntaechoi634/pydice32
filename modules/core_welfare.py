"""
Core welfare module: utility target and social welfare function.

Supports three SWF modes (cfg.swf):
  - "disentangled" (default): Berger-Emmerling equity-equivalents SWF
  - "dice": Original DICE welfare function adapted to multi-region
  - "stochastic": Stochastic SWF with probability-weighted branches and
    disentangled risk aversion (GAMS core_welfare.gms lines 137-143,
    mod_stochastic.gms)

Variables: UTARG, UTILITY, PERIODU (dice), CEMUTOTPER (dice), TUTILITY (stochastic)
Equations: eq_utarg, eq_util, eq_periodu (dice), eq_cemutotper (dice),
           eq_welfare (stochastic)

GAMS reference: core_welfare.gms lines 121-152, mod_stochastic.gms
"""

from gamspy import Variable, Equation, Sum


def declare_vars(m, sets, params, cfg, v):
    """Create welfare variables, set bounds/starting values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, n_alias, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    par_cpc_start = params["par_cpc_start"]

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    UTARG = Variable(m, name="UTARG", domain=[t_set, n_set], type="positive")
    UTILITY = Variable(m, name="UTILITY")

    # ------------------------------------------------------------------
    # Bounds and starting values
    # ------------------------------------------------------------------
    UTARG.lo[t_set, n_set] = 1e-8
    UTARG.l[t_set, n_set] = par_cpc_start[t_set, n_set]

    # Register in shared variable dict
    v["UTARG"] = UTARG
    v["UTILITY"] = UTILITY

    # ------------------------------------------------------------------
    # DICE SWF variables: PERIODU and CEMUTOTPER
    # GAMS core_welfare.gms lines 96-97:
    #   PERIODU(t,n)      'One period utility function'
    #   CEMUTOTPER(t,n)   'Period utility'
    # ------------------------------------------------------------------
    if cfg.swf == "dice":
        PERIODU = Variable(m, name="PERIODU", domain=[t_set, n_set])
        CEMUTOTPER = Variable(m, name="CEMUTOTPER", domain=[t_set, n_set])

        # Starting values based on CPC
        ELASMU = cfg.ELASMU
        # PERIODU.l = (CPC^(1-elasmu) - 1) / (1-elasmu) - 1
        # We just set a reasonable starting value
        PERIODU.l[t_set, n_set] = 0.0
        CEMUTOTPER.l[t_set, n_set] = 0.0

        v["PERIODU"] = PERIODU
        v["CEMUTOTPER"] = CEMUTOTPER

    # ------------------------------------------------------------------
    # Stochastic SWF: TUTILITY (intra-region utility per period)
    # GAMS core_welfare.gms line 98:
    #   TUTILITY(t) 'Intra-region utility'
    # ------------------------------------------------------------------
    if cfg.swf == "stochastic":
        TUTILITY = Variable(m, name="TUTILITY", domain=[t_set], type="positive")
        TUTILITY.lo[t_set] = 1e-3
        TUTILITY.l[t_set] = 1.0
        v["TUTILITY"] = TUTILITY


def define_eqs(m, sets, params, cfg, v):
    """Create welfare equations.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, n_alias, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    n_alias = sets["n_alias"]

    par_rr = params["par_rr"]
    par_pop = params["par_pop"]
    par_nweights = params["par_nweights"]

    ELASMU = cfg.ELASMU
    GAMMA = cfg.GAMMA

    # Own variables
    UTARG = v["UTARG"]
    UTILITY = v["UTILITY"]

    # Cross-module variables
    CPC = v["CPC"]

    # ------------------------------------------------------------------
    # eq_utarg: UTARG = CPC (common to all SWF modes)
    # GAMS core_welfare.gms line 131:
    #   eq_utility_arg(t,n).. UTARG(t,n) =E= CPC(t,n)
    #
    # When ocean module is active (cfg.ocean=True), the ocean module
    # provides a CES-nested eq_utarg that replaces this default.
    # GAMS: $ifthen.ut set welfare_ocean ... eq_utility_arg ... $endif.ut
    # ------------------------------------------------------------------
    equations = []
    if not getattr(cfg, "ocean", False):
        eq_utarg = Equation(m, name="eq_utarg", domain=[t_set, n_set])
        eq_utarg[t_set, n_set] = UTARG[t_set, n_set] == CPC[t_set, n_set]
        equations.append(eq_utarg)

    # ==================================================================
    # SWF-specific objective equation
    # ==================================================================

    if cfg.swf == "disentangled":
        # ------------------------------------------------------------------
        # Disentangled SWF (default)
        # GAMS core_welfare.gms line 135:
        #   eq_util.. UTILITY =E= sum(t, PROB.l(t) *
        #     ( ( ( sum(n, pop(t,n)/sum(nn,pop(t,nn)) * UTARG(t,n)^(1-gamma) )
        #       )^((1-elasmu)/(1-gamma)) ) / (1-elasmu) - 1 ) * rr(t) )
        #
        # In cooperative single-pass mode, PROB.l(t) = 1 for all t.
        # Population weights are built into the equation.
        # ------------------------------------------------------------------
        exp_inner = 1.0 - GAMMA           # 0.5
        exp_outer = (1.0 - ELASMU) / (1.0 - GAMMA)  # -0.9
        denom_util = 1.0 - ELASMU         # -0.45

        eq_util = Equation(m, name="eq_util")
        eq_util[...] = UTILITY == Sum(t_set,
            par_rr[t_set] * (
                (Sum(n_set, par_pop[t_set, n_set] / Sum(n_alias, par_pop[t_set, n_alias])
                        * (UTARG[t_set, n_set] ** exp_inner))
                ) ** exp_outer
                / denom_util - 1
            )
        )
        equations.append(eq_util)

    elif cfg.swf == "dice":
        # ------------------------------------------------------------------
        # DICE SWF (multi-region adaptation)
        # GAMS core_welfare.gms lines 147-151:
        #
        # eq_periodu(t,n)..
        #   PERIODU(t,n) =E= (UTARG(t,n)^(1-elasmu) - 1) / (1-elasmu) - 1
        #
        # eq_cemutotper(t,n)..
        #   CEMUTOTPER(t,n) =E= PERIODU(t,n) * rr(t) * pop(t,n)
        #
        # eq_util..
        #   UTILITY =E= dice_scale1 * tstep
        #     * sum((t,n), nweights(t,n) * PROB.l(t) * CEMUTOTPER(t,n))
        #     + dice_scale2
        #
        # In cooperative single-pass mode: PROB.l(t) = 1.
        # nweights = 1 for pop weights, or Negishi weights if configured.
        # ------------------------------------------------------------------
        PERIODU = v["PERIODU"]
        CEMUTOTPER = v["CEMUTOTPER"]

        dice_scale1 = cfg.dice_scale1
        dice_scale2 = cfg.dice_scale2
        TSTEP = cfg.TSTEP

        # eq_periodu: instantaneous utility
        eq_periodu = Equation(m, name="eq_periodu", domain=[t_set, n_set])
        eq_periodu[t_set, n_set] = (
            PERIODU[t_set, n_set] == (
                (UTARG[t_set, n_set] ** (1.0 - ELASMU) - 1.0)
                / (1.0 - ELASMU) - 1.0
            )
        )
        equations.append(eq_periodu)

        # eq_cemutotper: period utility (discounted, population-weighted)
        eq_cemutotper = Equation(m, name="eq_cemutotper", domain=[t_set, n_set])
        eq_cemutotper[t_set, n_set] = (
            CEMUTOTPER[t_set, n_set] == PERIODU[t_set, n_set]
            * par_rr[t_set] * par_pop[t_set, n_set]
        )
        equations.append(eq_cemutotper)

        # eq_util: objective function
        eq_util = Equation(m, name="eq_util")
        eq_util[...] = (
            UTILITY == dice_scale1 * TSTEP
            * Sum([t_set, n_set],
                  par_nweights[t_set, n_set] * CEMUTOTPER[t_set, n_set])
            + dice_scale2
        )
        equations.append(eq_util)

    elif cfg.swf == "stochastic":
        # ------------------------------------------------------------------
        # Stochastic SWF
        #
        # GAMS core_welfare.gms lines 137-143 + mod_stochastic.gms:
        #
        # eq_welfare(t)..
        #   TUTILITY(t) =E= sum(n, pop(t,n)/sum(nn,pop(t,nn))
        #                    * UTARG(t,n)^(1-gamma))
        #
        # eq_util.. UTILITY =E= (
        #   sum(t$(tperiod(t) lt t_resolution_one),
        #     rr(t)/sum(ttt,rr(ttt)*PROB(ttt))
        #     * TUTILITY(t)^((1-elasmu)/(1-gamma)))
        #   + sum(t$branch_node(t, 'branch_1'),
        #     rr(t)/sum(ttt,rr(ttt)*PROB(ttt))
        #     * (sum(tt$(year(tt) eq year(t)),
        #            PROB(tt) * TUTILITY(tt)^((1-rra)/(1-gamma)))
        #       )^((1-elasmu)/(1-rra)))
        # )^(1/(1-elasmu)) * 1e6
        #
        # Parameters from solver.py:
        #   par_prob[t]          : probability weight per time-state node
        #   par_is_pre_res[t]    : 1 if Ord(t) < t_resolution, else 0
        #   par_is_branch1[t]    : 1 if t in branch_1, else 0
        #   par_year_map[t, tt]  : 1 if year(t)==year(tt) (cross-branch)
        #
        # Deterministic behaviour: when par_is_branch1 is all-zero and
        # par_is_pre_res is all-one, the branch sum vanishes and the
        # pre-resolution sum covers all periods, yielding a CES temporal
        # aggregation that is a monotonic transform of the disentangled
        # formula (same optimal allocation).
        # ------------------------------------------------------------------
        t_alias = sets["t_alias"]
        t_alias2 = sets["t_alias2"]

        TUTILITY = v["TUTILITY"]

        par_prob = params["par_prob"]
        par_is_pre_res = params["par_is_pre_res"]
        par_is_branch1 = params["par_is_branch1"]
        par_year_map = params["par_year_map"]

        RRA = cfg.rra

        # Exponents
        exp_inner = 1.0 - GAMMA                          # for UTARG^(1-gamma)
        exp_elasmu_gamma = (1.0 - ELASMU) / (1.0 - GAMMA)  # pre-res exponent
        exp_rra_gamma = (1.0 - RRA) / (1.0 - GAMMA)     # branch inner exponent
        exp_elasmu_rra = (1.0 - ELASMU) / (1.0 - RRA)   # branch outer exponent
        exp_outer = 1.0 / (1.0 - ELASMU)                 # final CES exponent

        # ── eq_welfare: intra-region population-weighted CES aggregation ──
        # GAMS: TUTILITY(t) = sum(n, pop(t,n)/sum(nn,pop(t,nn))
        #                      * UTARG(t,n)^(1-gamma))
        eq_welfare = Equation(m, name="eq_welfare", domain=[t_set])
        eq_welfare[t_set] = (
            TUTILITY[t_set] == Sum(n_set,
                par_pop[t_set, n_set] / Sum(n_alias, par_pop[t_set, n_alias])
                * (UTARG[t_set, n_set] ** exp_inner)
            )
        )
        equations.append(eq_welfare)

        # ── eq_util: stochastic objective function ────────────────────────
        # The normalising denominator: sum(ttt, rr(ttt) * PROB(ttt))
        rr_prob_denom = Sum(t_alias2,
                            par_rr[t_alias2] * par_prob[t_alias2])

        # Term 1: Pre-resolution periods (common trunk, no uncertainty)
        # sum(t$(t < t_resolution), rr(t)/denom * TUTILITY(t)^((1-elasmu)/(1-gamma)))
        pre_res_term = Sum(t_set,
            par_is_pre_res[t_set]
            * par_rr[t_set] / rr_prob_denom
            * TUTILITY[t_set] ** exp_elasmu_gamma
        )

        # Term 2: Post-resolution branch periods (stochastic branches)
        # sum(t$branch_node(t,'branch_1'),
        #   rr(t)/denom
        #   * (sum(tt$(year(tt)==year(t)), PROB(tt) * TUTILITY(tt)^((1-rra)/(1-gamma)))
        #     )^((1-elasmu)/(1-rra))
        # )
        #
        # par_is_branch1[t] selects branch_1 periods (one per calendar year).
        # par_year_map[t, tt] selects all tt sharing the same calendar year as t
        # (i.e., all branches at that year).
        branch_term = Sum(t_set,
            par_is_branch1[t_set]
            * par_rr[t_set] / rr_prob_denom
            * (Sum(t_alias,
                   par_year_map[t_set, t_alias]
                   * par_prob[t_alias]
                   * TUTILITY[t_alias] ** exp_rra_gamma
              )) ** exp_elasmu_rra
        )

        # Combined: (pre_res_term + branch_term)^(1/(1-elasmu)) * 1e6
        eq_util = Equation(m, name="eq_util")
        eq_util[...] = UTILITY == (
            (pre_res_term + branch_term) ** exp_outer * 1e6
        )
        equations.append(eq_util)

    else:
        raise ValueError(
            f"Unknown SWF mode: {cfg.swf!r}. Must be 'disentangled', 'dice', or 'stochastic'."
        )

    return equations
