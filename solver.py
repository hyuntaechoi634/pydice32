"""
Model assembly and solving.
Collects all modules, builds the GAMSPy Model, and solves.

Provides three solving modes:
  - solve_model()            -- single-pass (fast, no iteration)
  - solve_model_iterative()  -- cooperative iterative solve mirroring
                                GAMS algorithm/solve_regions.gms
  - solve_model_nash()       -- non-cooperative Nash equilibrium via
                                iterative best-response

Cooperation routing (cfg.cooperation):
  "coop"            -> solve_model() for single-pass, or
                       solve_model_iterative() for iterative cooperative
                       (grand-coalition Negishi-weighted social planner).
  "coop_iterative"  -> solve_model_iterative() always; same grand-coalition
                       but forces the iterative before_solve / after_solve
                       loop (DAC learning, Negishi weights, savings fix).
  "noncoop"         -> solve_model_iterative() dispatches to
                       solve_model_nash(); each region is its own coalition,
                       sequential best-response (Gauss-Seidel) per iteration.
  "coalitions"      -> solve_model_iterative() dispatches to
                       solve_model_nash(); coalitions defined in
                       data["coalitions"], otherwise one region per coalition.
"""

import math
from gamspy import (
    Container, Set, Alias, Parameter, Model, Sense, Options, Number,
    ModelStatus,
)
from pydice32.config import Config
from pydice32.data import load_and_calibrate
from pydice32.modules import (
    core_economy, core_emissions, core_abatement, core_welfare,
    hub_climate, hub_impact, mod_impact_dice, mod_impact_kalkuhl,
    mod_impact_burke, mod_impact_howard, mod_impact_dell, mod_impact_coacch,
    mod_impact_deciles,
    mod_climate_regional, mod_climate_fair, mod_climate_tatm_exogen,
    mod_landuse, core_policy,
)
from pydice32.modules import (
    mod_dac, mod_emi_stor, mod_sai, mod_adaptation,
    mod_ocean, mod_natural_capital, mod_inequality, mod_labour, mod_slr,
)


def build_model(cfg: Config):
    """Build the full RICE model. Returns (container, model, variables, data)."""

    print("Loading data...")
    data = load_and_calibrate(cfg)
    region_names = data["region_names"]

    print(f"  Active regions: {len(region_names)}")
    print("Building GAMSPy model...")

    m = Container()

    # ── Sets ──────────────────────────────────────────────────
    t_set = Set(m, name="t", records=[str(i) for i in range(1, cfg.T + 1)])
    n_set = Set(m, name="n", records=region_names)
    t_alias = Alias(m, name="tp1", alias_with=t_set)
    t_alias2 = Alias(m, name="ttt", alias_with=t_set)
    n_alias = Alias(m, name="nn", alias_with=n_set)
    layers = Set(m, name="layers", records=["atm", "upp", "low"])
    layers_alias = Alias(m, name="mm", alias_with=layers)
    ghg_set = Set(m, name="ghg", records=list(cfg.ghg_list))

    sets = dict(t_set=t_set, n_set=n_set, t_alias=t_alias, t_alias2=t_alias2,
                n_alias=n_alias,
                layers=layers, layers_alias=layers_alias, ghg_set=ghg_set)

    # ── Parameters ────────────────────────────────────────────
    params = _create_parameters(m, sets, data, cfg)

    # ── Modules: two-pass (mirrors GAMS declare_vars → eqs phases) ──
    v = {}  # shared variable registry
    equations = []

    modules = _module_order(cfg)

    # Pass 1: declare all variables (no cross-module lookups needed)
    for mod in modules:
        if hasattr(mod, "declare_vars"):
            mod.declare_vars(m, sets, params, cfg, v)

    # Pass 2: define equations (can reference any variable via v)
    for mod in modules:
        eqs = mod.define_eqs(m, sets, params, cfg, v)
        equations.extend(eqs)

    # ── Model ─────────────────────────────────────────────────
    rice = Model(m, name="RICE", equations=equations,
                 problem="NLP", sense=Sense.MAX, objective=v["UTILITY"])

    # Store params and sets on data dict for iterative solver access
    data["_params"] = params
    data["_sets"] = sets

    return m, rice, v, data


# ---------------------------------------------------------------------------
#  Single-pass solve (original, fast)
# ---------------------------------------------------------------------------

def solve_model(rice, cfg: Config):
    """Solve the model in a single pass (no iteration)."""
    print("Solving with CONOPT...")
    rice.solve(solver="conopt", options=Options(iteration_limit=99900))
    print(f"  Status: {rice.status} / {rice.solve_status}")
    return rice


def update_macc_params(data, macc_bundle):
    """Swap MACC parameters in an existing model for batch re-solve.

    Parameters
    ----------
    data : dict
        The data dict from build_model(), containing '_params' with
        par_macc_c1 and par_macc_c4 GAMSPy Parameters.
    macc_bundle : dict
        Output of compute_macc_bundle(): {'macc_c1': {...}, 'macc_c4': {...}}
    """
    p = data["_params"]
    par_c1 = p["par_macc_c1"]
    par_c4 = p["par_macc_c4"]
    c1 = macc_bundle["macc_c1"]
    c4 = macc_bundle["macc_c4"]

    c1_records = [(str(t), r, g, v) for (t, r, g), v in c1.items()]
    c4_records = [(str(t), r, g, v) for (t, r, g), v in c4.items()]

    par_c1.setRecords(c1_records)
    par_c4.setRecords(c4_records)


# ---------------------------------------------------------------------------
#  Iterative cooperative solve  (mirrors GAMS algorithm/solve_regions.gms)
# ---------------------------------------------------------------------------

def solve_model_iterative(m, rice, v, data, cfg):
    """Iterative solve with before_solve/after_solve callbacks.

    Mirrors GAMS algorithm/solve_regions.gms:
      1. Run before_solve callbacks (update parameters: DAC cost, Negishi
         weights, ctax, flexible-savings terminal fix)
      2. Solve the NLP
      3. Run after_solve callbacks (propagate climate, track convergence vars)
      4. Check convergence
      5. Repeat

    Parameters
    ----------
    m : Container
    rice : Model
    v : dict  -- variable registry
    data : dict  -- calibration data (includes _params, _sets)
    cfg : Config

    Returns
    -------
    dict with keys: converged (bool), iterations (int), allerr (dict),
                    viter (dict of iteration snapshots)
    """
    # D1: Non-cooperative and coalition modes should use Nash solver.
    # "coop" and cooperative-iterative modes (e.g. "coop_iterative") stay here.
    if cfg.cooperation == "coalitions" and getattr(cfg, "coalition_def", None) is None:
        import warnings
        warnings.warn(
            "Coalitions mode without coalition_def falls back to noncoop "
            "(one region per coalition). Set cfg.coalition_def to a dict "
            "or JSON path to define multi-region coalitions.",
            UserWarning,
        )
    if cfg.cooperation in ("noncoop", "coalitions"):
        return solve_model_nash(m, rice, v, data, cfg)

    max_iter = cfg.max_iter
    min_iter = cfg.min_iter
    tol = cfg.convergence_tol

    region_names = data["region_names"]
    T = cfg.T

    # GAMS vcheck: base {MIU, S, Y, TATM} + module additions (E_NEG, MIULAND, SAI).
    # All variables in vcheck gate convergence (not just monitoring).
    tracked_vars = ["MIU", "S", "Y", "TATM"]
    if cfg.dac and "E_NEG" in v:
        tracked_vars.append("E_NEG")
    if cfg.sai and "W_SAI" in v:
        tracked_vars.append("W_SAI")
    if "MIULAND" in v:
        tracked_vars.append("MIULAND")

    # D14: Per-variable convergence tolerances (Y is tighter)
    tolerances = {vn: tol for vn in tracked_vars}
    tolerances["Y"] = tol * 0.1

    # Iteration history: viter[iteration][varname] -> {(t, n): value}
    viter = {}
    prev_values = None
    converged = False

    print(f"Iterative solve: max_iter={max_iter}, min_iter={min_iter}, "
          f"tol={tol}")

    for iteration in range(1, max_iter + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Phase 1: before_solve
        _before_solve(m, v, data, cfg, iteration)

        # Phase 2: solve
        print("  Solving...")
        rice.solve(solver="conopt", options=Options(iteration_limit=99900))
        print(f"  Status: {rice.status} / {rice.solve_status}")

        # D2: Retry once with fresh solver if infeasible
        if rice.status in (ModelStatus.InfeasibleGlobal, ModelStatus.InfeasibleLocal):
            print("  Infeasible -- retrying with relaxed iteration limit...")
            rice.solve(solver="conopt", options=Options(iteration_limit=99900))
            print(f"  Retry status: {rice.status} / {rice.solve_status}")

        # D3: Track optimality for convergence gating
        is_optimal = rice.status in (
            ModelStatus.OptimalGlobal, ModelStatus.OptimalLocal,
            ModelStatus.Feasible,
        )

        # Phase 3: after_solve
        _after_solve(m, v, data, cfg, iteration)

        # Phase 4: snapshot and convergence check
        current = _snapshot(v, data, cfg, tracked_vars)
        viter[iteration] = current

        allerr = {}
        if prev_values is not None:
            allerr = _compute_errors(current, prev_values, tracked_vars, cfg)
            _print_errors(allerr, iteration)

            # Gate on ALL tracked_vars (GAMS vcheck includes module additions)
            if iteration >= min_iter and is_optimal:
                if all(allerr.get(vn, 1.0) < tolerances.get(vn, tol)
                       for vn in tracked_vars):
                    converged = True
                    print(f"\n  Converged at iteration {iteration}.")
                    break
        else:
            print("  (first iteration -- no convergence check)")

        # Update Negishi weights after first iteration
        # GAMS solve_regions.gms line 185:
        #   nweights(t,n)$((not converged)) = calc_nweights
        if cfg.region_weights == "negishi" and not converged:
            _update_negishi_weights(v, data, cfg)

        prev_values = current

    if not converged:
        print(f"\n  WARNING: not converged after {max_iter} iterations")

    return dict(converged=converged, iterations=iteration,
                allerr=allerr, viter=viter)


# ---------------------------------------------------------------------------
#  Nash non-cooperative solve (iterative best-response)
# ---------------------------------------------------------------------------

def solve_model_nash(m, rice, v, data, cfg):
    """Non-cooperative Nash equilibrium via iterative best-response.

    For each iteration, each coalition (or region in pure noncoop) solves
    independently while all other coalitions' decision variables are fixed
    at their current levels. After all coalitions solve, the climate is
    propagated and convergence is checked.

    Parameters
    ----------
    m : Container
    rice : Model
    v : dict  -- variable registry
    data : dict  -- calibration data (includes _params, _sets, coalitions)
    cfg : Config

    Returns
    -------
    dict with keys: converged (bool), iterations (int), allerr (dict),
                    viter (dict of iteration snapshots)
    """
    max_iter = cfg.max_iter
    min_iter = cfg.min_iter
    tol = cfg.convergence_tol

    region_names = data["region_names"]
    # D22/D23: coalitions mode is not fully supported for GCAM-32.
    # For noncoop, each region is its own coalition.  The "coalitions"
    # key in data is used only for the (unsupported) coalitions cooperation
    # mode; for noncoop the default one-region-per-coalition is correct.
    coalitions = data.get("coalitions",
                          [[r] for r in region_names])

    # GAMS vcheck: base {MIU, S, Y, TATM} + module additions (E_NEG, MIULAND, SAI).
    tracked_vars = ["MIU", "S", "Y", "TATM"]
    if cfg.dac and "E_NEG" in v:
        tracked_vars.append("E_NEG")
    if cfg.sai and "W_SAI" in v:
        tracked_vars.append("W_SAI")
    if "MIULAND" in v:
        tracked_vars.append("MIULAND")

    # D14: Per-variable convergence tolerances (Y is tighter)
    tolerances = {vn: tol for vn in tracked_vars}
    tolerances["Y"] = tol * 0.1

    viter = {}
    prev_values = None
    converged = False

    print(f"Nash solve: {len(coalitions)} coalitions, "
          f"max_iter={max_iter}, min_iter={min_iter}, tol={tol}")

    for iteration in range(1, max_iter + 1):
        print(f"\n--- Nash iteration {iteration} ---")

        # before_solve phase (common updates)
        _before_solve(m, v, data, cfg, iteration)

        # D17: Sequential best-response (Gauss-Seidel) is an intentional
        # design choice.  It is a standard approach in IAMs where each
        # coalition updates its decisions using the latest available
        # information from previously-solved coalitions within the same
        # iteration.  This typically converges faster than the simultaneous
        # (Jacobi) alternative and is used in the reference GAMS RICE model.
        all_optimal = True
        for clt_idx, clt in enumerate(coalitions):
            clt_regions = set(clt)
            other_regions = [r for r in region_names if r not in clt_regions]

            if other_regions:
                _fix_other_regions(v, other_regions, cfg)

            print(f"  Solving coalition {clt_idx + 1}/{len(coalitions)}: "
                  f"{clt}")
            rice.solve(solver="conopt",
                       options=Options(iteration_limit=99900))
            clt_status = rice.status

            # D2: Retry once if infeasible
            if clt_status in (ModelStatus.InfeasibleGlobal,
                              ModelStatus.InfeasibleLocal):
                print("    Infeasible -- retrying...")
                rice.solve(solver="conopt",
                           options=Options(iteration_limit=99900))
                clt_status = rice.status

            # Track optimality across ALL coalitions (not just the last one)
            if clt_status not in (ModelStatus.OptimalGlobal,
                                  ModelStatus.OptimalLocal,
                                  ModelStatus.Feasible):
                all_optimal = False
                print(f"    WARNING: coalition {clt} status = {clt_status}")

            if other_regions:
                _unfix_other_regions(v, other_regions, cfg)

        # after_solve phase (climate propagation, tracking)
        _after_solve(m, v, data, cfg, iteration)

        # D3: Track optimality for convergence gating (all coalitions must be optimal)
        is_optimal = all_optimal

        # Snapshot and convergence
        current = _snapshot(v, data, cfg, tracked_vars)
        viter[iteration] = current

        allerr = {}
        if prev_values is not None:
            allerr = _compute_errors(current, prev_values, tracked_vars, cfg)
            _print_errors(allerr, iteration)

            # Gate on ALL tracked_vars (GAMS vcheck includes module additions)
            if iteration >= min_iter and is_optimal:
                if all(allerr.get(vn, 1.0) < tolerances.get(vn, tol)
                       for vn in tracked_vars):
                    converged = True
                    print(f"\n  Nash converged at iteration {iteration}.")
                    break

        if cfg.region_weights == "negishi" and not converged:
            _update_negishi_weights(v, data, cfg)

        prev_values = current

    if not converged:
        print(f"\n  WARNING: Nash not converged after {max_iter} iterations")

    return dict(converged=converged, iterations=iteration,
                allerr=allerr, viter=viter)


# ---------------------------------------------------------------------------
#  Internal: before_solve callbacks
# ---------------------------------------------------------------------------

def _before_solve(m, v, data, cfg, iteration):
    """Update parameters between iterations (mirrors GAMS before_solve phases).

    Updates applied:
      - DAC learning curve (mod_dac before_solve)
      - Flexible savings terminal fix (core_economy before_solve)
      - ctax_corrected from current E.l and MAC.l (core_policy before_solve)
    """
    # --- DAC learning curve update ---
    # GAMS mod_dac.gms before_solve:
    #   wcum_dac(tp1) = tlen(t) * sum(n, E_NEG.l(t,n)) + wcum_dac(t)
    #   dac_totcost(t,n) = max(dac_tot0 * (wcum_dac(t)/wcum_dac('1'))^(-learn), floor)
    if cfg.dac and "E_NEG" in v:
        _update_dac_learning(v, data, cfg)

    # --- Flexible savings terminal fix ---
    # GAMS core_economy.gms before_solve:
    #   S.fx(t,n)$(tperiod(t) gt (smax(tt,tperiod(tt)) - 10)) = S.l('48',n)
    if cfg.savings_mode == "flexible" and iteration > 1:
        _update_savings_terminal(v, data, cfg)

    # --- Fiscal revenue updates (ctax_corrected + E.l) ---
    # GAMS core_policy.gms before_solve:
    #   ctax_corrected(t,n,ghg) = min(ctax*1e3*emi_gwp(ghg), cprice_max(t,n,ghg))
    #   if tax_oghg_as_co2: ctax_corrected(non-co2) = min(MAC.l(co2)*emi_gwp, cprice_max)
    # Also update par_emi_level to E.l for fiscal revenue term in eq_yy.
    if cfg.policy in ("ctax", "cbudget_regional") and not cfg.ctax_marginal and iteration > 1:
        _update_fiscal_revenue(v, data, cfg, iteration)

    # D8: Update cprice_max from current MAC levels (for ctax policy).
    if cfg.policy in ("ctax", "cbudget_regional") and "MAC" in v and iteration > 1:
        _update_cprice_max(v, data, cfg)

    # --- NDC MAC.lo extrapolation + ctax_corrected recomputation ---
    # GAMS pol_ndc.gms before_solve: re-extrapolate MAC.lo post-2030 using
    # current MAC.l('4') values, then recompute ctax_corrected.
    # This is critical for long_term_pledges policy with iterative solver.
    if cfg.policy == "long_term_pledges" and getattr(cfg, "pol_ndc", False) and iteration > 1:
        _update_ndc_mac_extrap(v, data, cfg)

    # --- Ocean YNET.l update ---
    # GAMS mod_ocean.gms: VSL and mangrove valuation use YNET.l
    if cfg.ocean and "YNET" in v and iteration > 1:
        YNET = v["YNET"]
        par_ynet_lev = data.get("_params", {}).get("par_ynet_level")
        if par_ynet_lev is not None and YNET.records is not None and len(YNET.records) > 0:
            new_recs = [(str(row.iloc[0]), str(row.iloc[1]), row["level"])
                        for _, row in YNET.records.iterrows()]
            par_ynet_lev.setRecords(new_recs)

    # --- Adaptation OMEGA gate update ---
    # GAMS hub_impact.gms: $(OMEGA.l(t,n) gt 0) controls whether adaptation
    # divisor is applied. Update par_omega_positive from solved OMEGA.l.
    if cfg.adaptation and "OMEGA" in v and iteration > 1:
        _update_omega_gate(v, data, cfg)

    # D9: Update TEMP_REGION starting values from current TATM levels
    # to provide better initial guesses for the NLP solver.
    if "TEMP_REGION" in v and "TATM" in v and iteration > 1:
        _update_temp_region_levels(v, data)


def _update_dac_learning(v, data, cfg):
    """Update DAC total cost parameter based on solved cumulative E_NEG.

    Mirrors GAMS mod_dac.gms before_solve:
      loop((t,tp1)$(pre(t,tp1) and year(t) ge 2015),
           wcum_dac(tp1) = tlen(t) * sum(n, E_NEG.l(t,n)) + wcum_dac(t));
      dac_totcost(t,n) = max(dac_tot0*(wcum_dac(t)/wcum_dac('1'))^(-learn), floor);
    """
    from pydice32.modules.mod_dac import (
        DAC_TOT0, DAC_TOTFLOOR, LEARN_RATES,
    )

    region_names = data["region_names"]
    T = cfg.T
    TSTEP = cfg.TSTEP

    dac_learn = LEARN_RATES.get(
        getattr(cfg, "dac_cost", "best"), LEARN_RATES["best"])

    E_NEG = v["E_NEG"]
    e_neg_recs = E_NEG.records
    if e_neg_recs is None or len(e_neg_recs) == 0:
        return

    # Build E_NEG level lookup: (t_str, n) -> level
    e_neg_levels = {}
    for _, row in e_neg_recs.iterrows():
        e_neg_levels[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

    # D6: wcum_dac initial value = 0.005 matching GAMS mod_dac.gms
    # GAMS: wcum_dac('1') = 0.005 (5 kt cumulative initial)
    wcum_dac = {1: 0.005}  # initial cumulative DAC (GAMS default)
    for t in range(1, T):
        total_eneg = sum(
            e_neg_levels.get((str(t), r), 0.0) for r in region_names)
        wcum_dac[t + 1] = TSTEP * total_eneg + wcum_dac.get(t, 0.005)

    # D5: The learning rate (dac_learn) is applied as a scalar across all
    # regions.  This is a known simplification -- the GAMS model also uses
    # a single global learning rate rather than per-region rates.  A
    # per-region learning curve would require region-specific cumulative
    # deployment tracking, which is left for future work.

    # Compute updated dac_totcost and store for reporting.
    dac_totcost_vals = {}
    wcum_base = max(wcum_dac.get(1, 0.005), 1e-12)
    for t in range(1, T + 1):
        ratio = max(wcum_dac.get(t, 0.005), 1e-12) / wcum_base
        cost = max(DAC_TOT0 * ratio ** (-dac_learn), DAC_TOTFLOOR)
        for r in region_names:
            dac_totcost_vals[(t, r)] = cost

    data["_dac_totcost"] = dac_totcost_vals
    data["_wcum_dac"] = wcum_dac

    # D7: Write updated dac_totcost back to GAMSPy Parameter so the
    # NLP solver uses the updated learning-curve costs in the next solve.
    par_dac_cost = data["_params"].get("par_dac_totcost")
    if par_dac_cost is not None:
        new_records = [(str(t), rn, cost)
                       for (t, rn), cost in dac_totcost_vals.items()]
        par_dac_cost.setRecords(new_records)


def _update_savings_terminal(v, data, cfg):
    """Fix savings rate for last 10 periods to S.l(T-10, n).

    GAMS core_economy.gms before_solve:
      S.fx(t,n)$(tperiod(t) gt (smax(tt,tperiod(tt)) - 10)) = S.l('48',n)
    H1: Use T-10 dynamically instead of hardcoded '48' so this works for any T.
    """
    T = cfg.T
    S = v["S"]
    region_names = data["region_names"]

    s_recs = S.records
    if s_recs is None or len(s_recs) == 0:
        return

    # Get S.l(T-10, n) for each region (period 48 when T=58)
    ref_period = str(T - 10)
    s_ref = {}
    for _, row in s_recs.iterrows():
        if str(row.iloc[0]) == ref_period:
            s_ref[str(row.iloc[1])] = row["level"]

    if not s_ref:
        return

    # Fix S for t > T - 10 (i.e., t >= T-9)
    for t in range(T - 10 + 1, T + 1):
        for r in region_names:
            if r in s_ref:
                S.fx[str(t), r] = s_ref[r]


def _update_omega_gate(v, data, cfg):
    """Update par_omega_positive from solved OMEGA.l values.

    GAMS hub_impact.gms line 97: $(OMEGA.l(t,n) gt 0) gates the adaptation
    divisor. When OMEGA.l <= 0 (warming benefit), adaptation is not applied.
    """
    OMEGA = v.get("OMEGA")
    if OMEGA is None or OMEGA.records is None or len(OMEGA.records) == 0:
        return

    par_omega_pos = data.get("_params", {}).get("par_omega_positive")
    if par_omega_pos is None:
        return

    new_records = []
    for _, row in OMEGA.records.iterrows():
        t_str = str(row.iloc[0])
        n_str = str(row.iloc[1])
        omega_val = row["level"]
        new_records.append((t_str, n_str, 1.0 if omega_val > 0 else 0.0))

    par_omega_pos.setRecords(new_records)


def _update_fiscal_revenue(v, data, cfg, iteration):
    """Update par_emi_level to E.l and ctax_corrected for fiscal revenue.

    GAMS core_policy.gms before_solve:
      ctax_corrected(t,n,ghg) = min(ctax*1e3*emi_gwp(ghg), cprice_max(t,n,ghg))
      if tax_oghg_as_co2: ctax_corrected(non-co2) = min(MAC.l(co2)*emi_gwp, cprice_max)
    Also for cbudget_regional + cost_efficiency: iterative ctax_var adjustment.
    """
    p = data.get("_params", {})
    E = v.get("E")
    MAC = v.get("MAC")

    # --- Update par_emi_level to current E.l ---
    par_emi_level = p.get("par_emi_level")
    if par_emi_level is not None and E is not None and E.records is not None and len(E.records) > 0:
        new_records = []
        for _, row in E.records.iterrows():
            new_records.append((str(row.iloc[0]), str(row.iloc[1]),
                                str(row.iloc[2]), row["level"]))
        par_emi_level.setRecords(new_records)

    # --- cbudget_regional + cost_efficiency: iterative ctax_var adjustment ---
    # GAMS core_policy.gms lines 423-457: adjust ctax_var to hit carbon budget
    if cfg.policy == "cbudget_regional" and cfg.burden == "cost_efficiency" and iteration > 1:
        # Compute current cbudget_2020_2100 from E.l
        if E is not None and E.records is not None and len(E.records) > 0:
            region_names = data.get("region_names", [])
            e_levels = {}
            for _, row in E.records.iterrows():
                key = (str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))
                e_levels[key] = row["level"]

            cbudget_actual = 0.0
            for t in range(3, cfg.T + 1):  # year(t) > 2020 and year(t) <= 2095
                yr = cfg.year(t)
                if yr > 2020 and yr <= 2095:
                    for r in region_names:
                        cbudget_actual += e_levels.get((str(t), r, "co2"), 0.0) * cfg.TSTEP
            # + 3.5 * E('2',n,'co2')
            for r in region_names:
                cbudget_actual += 3.5 * e_levels.get(("2", r, "co2"), 0.0)
            # + 2.5 * E at year 2100 (t=18)
            for r in region_names:
                cbudget_actual += 2.5 * e_levels.get(("18", r, "co2"), 0.0)

            ctax_target = cfg.cbudget
            conv_budget = getattr(cfg, "conv_budget", 1.0)

            ctax_var = data.get("_ctax_var")
            if ctax_var is None:
                # Initialize: GAMS formula
                import math
                ctax_var = max(581.12 - 74.3 * math.log(ctax_target), 1.0)

            if abs(cbudget_actual - ctax_target) > conv_budget:
                ratio = min((6000 - ctax_target) / max(6000 - cbudget_actual, 1), 3)
                ctax_var = ctax_var * ratio ** 2.2

            if ctax_var > 10000:
                print(f"  WARNING: ctax_var={ctax_var:.1f} exceeds 10000, clamping")
                ctax_var = 10000
            data["_ctax_var"] = ctax_var

            # Recompute tax schedule and ctax_corrected
            _recompute_ctax_schedule(v, p, data, cfg, ctax_var)

    # Initialize ctax_schedule if not yet set
    if "_ctax_schedule" not in data:
        tax_schedule = {}
        ctax_init = data.get("_ctax_var", cfg.ctax_initial)
        for t in range(1, cfg.T + 1):
            yr = cfg.year(t)
            if yr >= cfg.ctax_start:
                effective_yr = min(yr, 2100)
                tax = (ctax_init / 1000.0) * (1 + cfg.ctax_slope) ** (effective_yr - cfg.ctax_start)
            else:
                tax = 1e-8
            tax_schedule[t] = tax
        # Cap post-2100 at 2100 value
        t2100 = (2100 - 2015) // cfg.TSTEP + 1
        for t in range(t2100 + 1, cfg.T + 1):
            tax_schedule[t] = tax_schedule.get(t2100, tax_schedule[t])
        data["_ctax_schedule"] = tax_schedule

    # --- Update ctax_corrected from current MAC.l ---
    # GAMS: ctax_corrected = min(ctax*1e3*emi_gwp, cprice_max)
    # With tax_oghg_as_co2: non-CO2 uses MAC.l(co2)*emi_gwp
    par_ctax_corr = v.get("par_ctax_corrected")
    if par_ctax_corr is not None and MAC is not None and MAC.records is not None:
        mac_co2_levels = {}  # (t, n) -> MAC.l for CO2
        for _, row in MAC.records.iterrows():
            if str(row.iloc[2]) == "co2":
                mac_co2_levels[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

        emi_gwp = {"co2": 1.0, "ch4": 28.0, "n2o": 265.0}
        par_emi_gwp = p.get("par_emi_gwp")
        if par_emi_gwp is not None and hasattr(par_emi_gwp, 'records') and par_emi_gwp.records is not None:
            for _, row in par_emi_gwp.records.iterrows():
                emi_gwp[str(row.iloc[0]).lower()] = float(row.iloc[1])

        # Read cprice_max (MAC at MIU.up)
        macc_c1_p = p.get("par_macc_c1")
        macc_c4_p = p.get("par_macc_c4")
        maxmiu_p = p.get("par_maxmiu_pbl")
        cp_max_dict = {}
        if macc_c1_p is not None and macc_c4_p is not None and maxmiu_p is not None:
            c1d, c4d, mud = {}, {}, {}
            for _, r in macc_c1_p.records.iterrows():
                c1d[(str(r.iloc[0]), str(r.iloc[1]), str(r.iloc[2]))] = float(r.iloc[3])
            for _, r in macc_c4_p.records.iterrows():
                c4d[(str(r.iloc[0]), str(r.iloc[1]), str(r.iloc[2]))] = float(r.iloc[3])
            for _, r in maxmiu_p.records.iterrows():
                mud[(str(r.iloc[0]), str(r.iloc[1]), str(r.iloc[2]))] = float(r.iloc[3])
            for k in c1d:
                mu = mud.get(k, 1.0)
                cp_max_dict[k] = c1d[k] * mu + c4d.get(k, 0) * mu ** 4

        region_names = data.get("region_names", [])
        new_ctax_records = []
        for t in range(1, cfg.T + 1):
            for ghg in cfg.ghg_list:
                # Base ctax_corrected from tax schedule
                base = data.get("_ctax_schedule", {}).get(t, 0.0) * 1e3 * emi_gwp.get(ghg, 1.0)
                # For non-CO2 with tax_oghg_as_co2: use MAC.l(co2) * emi_gwp
                # averaged across regions as approximation
                if ghg != "co2":
                    avg_mac_co2 = 0.0
                    cnt = 0
                    for r in region_names:
                        m_val = mac_co2_levels.get((str(t), r), 0.0)
                        if m_val > 0:
                            avg_mac_co2 += m_val
                            cnt += 1
                    if cnt > 0:
                        avg_mac_co2 /= cnt
                    base = avg_mac_co2 * emi_gwp.get(ghg, 1.0)
                # Cap at cprice_max (use region-average as approximation for [t, ghg] parameter)
                cp_vals = [cp_max_dict.get((str(t), r, ghg), 1e6) for r in region_names]
                cp_avg = sum(cp_vals) / max(len(cp_vals), 1)
                corrected = min(base, cp_avg)
                new_ctax_records.append((str(t), ghg, corrected))

        par_ctax_corr.setRecords(new_ctax_records)


def _recompute_ctax_schedule(v, p, data, cfg, ctax_var):
    """Recompute ctax schedule from updated ctax_var (cbudget_regional convergence).

    GAMS core_policy.gms lines 434-455: recompute ctax(t,n) from ctax_var.
    """
    tax_schedule = {}
    for t in range(1, cfg.T + 1):
        yr = cfg.year(t)
        if yr >= cfg.ctax_start:
            effective_yr = min(yr, 2100)
            tax = (ctax_var / 1000.0) * (1 + cfg.ctax_slope) ** (effective_yr - cfg.ctax_start)
        else:
            tax = 1e-8
        if yr > 2100:
            tax = tax_schedule.get(18, tax)  # cap at 2100 value (t=18)
        tax_schedule[t] = tax
    data["_ctax_schedule"] = tax_schedule


def _update_cprice_max(v, data, cfg):
    """D8: Update cprice_max upper bound from current MAC levels.

    GAMS before_solve: cprice_max(t,n) = max(cprice_max(t,n), MAC.l(t,n,'co2'))
    This ensures the carbon price upper bound tracks the solved MAC for the
    ctax policy.
    """
    MAC = v.get("MAC")
    if MAC is None or MAC.records is None or len(MAC.records) == 0:
        return

    region_names = data["region_names"]
    T = cfg.T

    # Read current MAC levels for CO2
    cprice_max = data.get("_cprice_max", {})
    for _, row in MAC.records.iterrows():
        t_str = str(row.iloc[0])
        n_str = str(row.iloc[1])
        ghg_str = str(row.iloc[2])
        if ghg_str == "co2":
            key = (t_str, n_str)
            mac_val = row["level"]
            old_val = cprice_max.get(key, 0.0)
            cprice_max[key] = max(old_val, mac_val)

    data["_cprice_max"] = cprice_max


def _update_ndc_mac_extrap(v, data, cfg):
    """Re-extrapolate MAC.lo post-2030 from current solved MAC.l levels.

    GAMS pol_ndc.gms before_solve (ndcs_extr="linear"):
      MAC.lo(t,n,ghg)$(year(t) gt 2030 and MAC.l('4') ne 0) =
        min(MAC.lo('4') * (1 + slope*(year(t)-2030)), cprice_max(t,n,ghg))
      ctax_corrected(t,n,ghg) = min(MAC.lo(t,n,ghg), cprice_max(t,n,ghg))
    """
    MAC = v.get("MAC")
    if MAC is None or MAC.records is None or len(MAC.records) == 0:
        return

    region_names = data.get("region_names", [])
    p = data.get("_params", {})
    macc_c1_p = p.get("par_macc_c1")
    macc_c4_p = p.get("par_macc_c4")
    maxmiu_p = p.get("par_maxmiu_pbl")
    if macc_c1_p is None or macc_c4_p is None:
        return

    # Build lookup dicts
    c1_dict, c4_dict, miu_up_dict = {}, {}, {}
    for _, row in macc_c1_p.records.iterrows():
        c1_dict[(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = float(row.iloc[3])
    for _, row in macc_c4_p.records.iterrows():
        c4_dict[(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = float(row.iloc[3])
    if maxmiu_p is not None and maxmiu_p.records is not None:
        for _, row in maxmiu_p.records.iterrows():
            miu_up_dict[(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = float(row.iloc[3])

    # Read current MAC.l at t=4 and t=2
    mac_l = {}
    for _, row in MAC.records.iterrows():
        mac_l[(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]))] = row["level"]

    for rname in region_names:
        for ghg in cfg.ghg_list:
            m4 = mac_l.get(("4", rname, ghg), 0.0)
            m2 = mac_l.get(("2", rname, ghg), 0.0)
            if m4 == 0:
                continue
            slope = (m4 - m2) / (m4 * cfg.TSTEP * 2) if m4 != 0 else 0
            for t in range(5, cfg.T + 1):
                yr = cfg.year(t)
                if yr <= 2030:
                    continue
                mac_extrap = m4 * (1 + slope * (yr - 2030))
                miu_up = miu_up_dict.get((str(t), rname, ghg), 1.0)
                c1 = c1_dict.get((str(t), rname, ghg), 0.0)
                c4 = c4_dict.get((str(t), rname, ghg), 0.0)
                cp_max = c1 * miu_up + c4 * miu_up ** 4
                mac_floor = min(mac_extrap, cp_max)
                if mac_floor > 0:
                    MAC.lo[str(t), rname, ghg] = mac_floor


def _update_temp_region_levels(v, data):
    """D9: Update TEMP_REGION.l starting values from current TATM.l.

    Provides better initial guesses for the NLP solver by setting regional
    temperatures consistent with the global temperature from the last solve.
    """
    TEMP_REGION = v.get("TEMP_REGION")
    TATM = v.get("TATM")
    if TEMP_REGION is None or TATM is None:
        return

    tatm_recs = TATM.records
    if tatm_recs is None or len(tatm_recs) == 0:
        return

    alpha_temp = data.get("alpha_temp_dict", {})
    beta_temp = data.get("beta_temp_dict", {})
    region_names = data.get("region_names", [])

    tatm_levels = {}
    for _, row in tatm_recs.iterrows():
        tatm_levels[str(row.iloc[0])] = row["level"]

    # Set TEMP_REGION.l = alpha + beta * TATM.l for each (t, n)
    for t_str, tatm_val in tatm_levels.items():
        for r in region_names:
            alpha = alpha_temp.get(r, 0.0)
            beta = beta_temp.get(r, 1.0)
            TEMP_REGION.l[t_str, r] = alpha + beta * tatm_val

    # Update par_tatm_level parameter for ocean module (GAMS: TATM.l pattern)
    # This parameter is used by eq_ocean_area to decouple area from optimizer.
    par_tatm_level = data.get("_params", {}).get("par_tatm_level")
    if par_tatm_level is not None:
        T = data.get("T", len(tatm_levels))
        par_tatm_level.setRecords(
            [(t_str, tatm_levels.get(t_str, 1.1)) for t_str in tatm_levels]
        )


# ---------------------------------------------------------------------------
#  Internal: after_solve callbacks
# ---------------------------------------------------------------------------

def _after_solve(m, v, data, cfg, iteration):
    """Post-solve updates (mirrors GAMS after_solve phases).

    D10: In the cooperative case (single grand coalition), the NLP already
    ensures climate consistency -- no re-solve needed.

    For Nash (noncoop/coalitions), the GAMS witchco2 after_solve pattern
    fixes W_EMI to current solved values, re-evaluates climate forward,
    and unfixes W_EMI.  Here we propagate climate state from the solved
    emission levels to ensure global temperature is consistent after the
    sequential coalition solves.
    """
    # H3: Only propagate climate in Nash modes where coalitions solve
    # separately.  In coop/coop_iterative the NLP already ensures consistency.
    if cfg.cooperation not in ("coop", "coop_iterative"):
        _propagate_climate(m, v, data, cfg)

    # Update TEMP_REGION.l for solver hints in the next iteration
    _update_temp_region_levels(v, data)


def _propagate_climate(m, v, data, cfg):
    """Re-evaluate climate forward from current W_EMI levels (Nash after_solve).

    Mirrors the GAMS witchco2 after_solve pattern:
      1. Fix W_EMI to current solved values
      2. Forward-propagate carbon cycle and temperature
      3. Unfix W_EMI

    Rather than re-solving a CNS system, we do a simple forward calculation
    of the carbon cycle and temperature equations using the solved emission
    levels.  This updates the .l (level) values of WCUM_EMI, TATM, TOCEAN
    so the next iteration's NLP starts from a consistent climate state.
    """
    # Dispatch to appropriate climate propagation
    if cfg.climate == "fair":
        _propagate_climate_fair(m, v, data, cfg)
        return

    T = cfg.T
    TSTEP = cfg.TSTEP

    W_EMI = v.get("W_EMI")
    WCUM_EMI = v.get("WCUM_EMI")
    TATM = v.get("TATM")
    TOCEAN = v.get("TOCEAN")
    if W_EMI is None or WCUM_EMI is None or TATM is None or TOCEAN is None:
        return

    # Read current climate parameters
    params = data["_params"]
    sigma1 = params.get("par_sigma1")
    lam = params.get("par_lambda")
    sigma2 = params.get("par_sigma2")
    heat_ocean = params.get("par_heat_ocean")

    # Read raw scalars for forward calculation
    # Fallback values match RICE50x WITCH calibration (data_maxiso3/tempc.csv)
    sigma1_val = data.get("sigma1", 0.37848906)
    lam_val = data.get("lam", 1.36666667)
    sigma2_val = data.get("sigma2", 0.36275881)
    heat_ocean_val = data.get("heat_ocean", 0.05748796)

    # Read carbon cycle transfer matrix
    cmphi = data.get("cmphi")
    if cmphi is None:
        return

    # Read current W_EMI levels
    w_emi_recs = W_EMI.records
    if w_emi_recs is None or len(w_emi_recs) == 0:
        return
    w_emi_levels = {}
    for _, row in w_emi_recs.iterrows():
        w_emi_levels[str(row.iloc[0])] = row["level"]

    # Read current WCUM levels
    wcum_recs = WCUM_EMI.records
    if wcum_recs is None or len(wcum_recs) == 0:
        return
    wcum_levels = {}  # (layer, t_str) -> level
    for _, row in wcum_recs.iterrows():
        wcum_levels[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

    # Read current TATM/TOCEAN levels
    tatm_recs = TATM.records
    tocean_recs = TOCEAN.records
    if tatm_recs is None or tocean_recs is None:
        return
    tatm_levels = {}
    for _, row in tatm_recs.iterrows():
        tatm_levels[str(row.iloc[0])] = row["level"]
    tocean_levels = {}
    for _, row in tocean_recs.iterrows():
        tocean_levels[str(row.iloc[0])] = row["level"]

    # Forward propagate: update .l values for periods 2..T
    layer_names = ["atm", "upp", "low"]
    for t in range(1, T):
        t_str = str(t)
        tp1_str = str(t + 1)

        # Carbon cycle: WCUM(layer, t+1) = sum_mm cmphi(mm,layer)*WCUM(mm,t) + TSTEP*W_EMI(t) [atm only]
        for li, layer in enumerate(layer_names):
            val = 0.0
            for mi, mm in enumerate(layer_names):
                val += cmphi[mi, li] * wcum_levels.get((mm, t_str), 0.001)
            if li == 0:  # atm
                val += TSTEP * w_emi_levels.get(t_str, 0.0)
            wcum_levels[(layer, tp1_str)] = max(val, 0.0001)

    # Temperature forward propagation: compute FORC from WCUM then update
    # TATM and TOCEAN using the DICE two-box energy balance model.
    rfc_alpha_val = data.get("rfc_alpha", 3.68)
    rfc_beta_val = data.get("rfc_beta", 588.0)
    oghg_intercept = data.get("oghg_intercept", 0.5)
    oghg_slope = data.get("oghg_slope", 0.0)

    for t in range(1, T):
        t_str = str(t)
        tp1_str = str(t + 1)

        # RF_CO2 = rfc_alpha * (log(WCUM_atm) - log(rfc_beta))
        wcum_atm = max(wcum_levels.get(("atm", t_str), 851.0), 1e-6)
        rf_co2 = rfc_alpha_val * (math.log(wcum_atm) - math.log(rfc_beta_val))

        # RFoth = oghg_intercept + oghg_slope * RF_CO2
        rfoth = oghg_intercept + oghg_slope * rf_co2

        # FORC = RF_CO2 + RFoth
        forc = rf_co2 + rfoth

        tatm_t = tatm_levels.get(t_str, 1.1)
        tocean_t = tocean_levels.get(t_str, 0.11)

        # TATM(t+1) = TATM(t) + sigma1*(FORC - lambda*TATM - sigma2*(TATM-TOCEAN))
        tatm_tp1 = tatm_t + sigma1_val * (
            forc - lam_val * tatm_t - sigma2_val * (tatm_t - tocean_t))
        # TOCEAN(t+1) = TOCEAN(t) + heat_ocean*(TATM-TOCEAN)
        tocean_tp1 = tocean_t + heat_ocean_val * (tatm_t - tocean_t)

        tatm_levels[tp1_str] = tatm_tp1
        tocean_levels[tp1_str] = tocean_tp1

    # Write updated levels back to GAMSPy variables
    wcum_records = [(layer, str(t), max(wcum_levels.get((layer, str(t)), 0.0001), 0.0001))
                    for layer in layer_names for t in range(1, T + 1)]
    WCUM_EMI.setRecords(wcum_records)

    tatm_records = [(str(t), tatm_levels.get(str(t), 1.1))
                    for t in range(1, T + 1)]
    TATM.setRecords(tatm_records)

    tocean_records = [(str(t), tocean_levels.get(str(t), 0.11))
                      for t in range(1, T + 1)]
    TOCEAN.setRecords(tocean_records)


def _propagate_climate_fair(m, v, data, cfg):
    """Forward-propagate FAIR climate model from solved W_EMI levels.

    Mirrors the GAMS 'solve fair using cns' after_solve pattern by
    forward-evaluating the FAIR 4-box CO2 cycle, non-CO2 decay, forcing,
    and two-box temperature from current emission levels.
    """
    from pydice32.modules.mod_climate_fair import (
        BOX_NAMES, TAUBOX, EMSHARE, TAUGHG,
        CONC_PREINDUSTRIAL, CO2toC, DELTA, FORC2X,
        IRF_PREINDUSTRIAL, IRC, IRT,
        _DERIVED,
    )

    T = cfg.T
    TSTEP = cfg.TSTEP

    emitoconc = _DERIVED["emitoconc"]
    catm_pre = _DERIVED["catm_preindustrial"]
    scaling_forc2x = _DERIVED["scaling_forc2x"]
    QSLOW = _DERIVED["QSLOW"]
    QFAST = _DERIVED["QFAST"]
    dt0 = _DERIVED["dt0"]

    slow_decay = math.exp(-TSTEP / 236.0)
    fast_decay = math.exp(-TSTEP / 4.07)

    # Read W_EMI levels: (ghg, t_str) -> float
    W_EMI_var = v.get("W_EMI")
    if W_EMI_var is None or W_EMI_var.records is None:
        return
    w_emi = {}
    for _, row in W_EMI_var.records.iterrows():
        w_emi[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

    # Read initial state from variable records
    def _read_var_1d(varname):
        var = v.get(varname)
        if var is None or var.records is None:
            return {}
        d = {}
        for _, row in var.records.iterrows():
            d[str(row.iloc[0])] = row["level"]
        return d

    def _read_var_2d(varname):
        var = v.get(varname)
        if var is None or var.records is None:
            return {}
        d = {}
        for _, row in var.records.iterrows():
            d[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]
        return d

    res_levels = _read_var_2d("RES")        # (box, t) -> level
    conc_levels = _read_var_2d("CONC")       # (ghg, t) -> level
    cumemi_levels = _read_var_1d("CUMEMI_FAIR")
    cd_scale = _read_var_1d("CD_SCALE")
    ff_ch4 = _read_var_1d("FF_CH4")
    tslow_levels = _read_var_1d("TSLOW")
    tfast_levels = _read_var_1d("TFAST")
    tatm_levels = _read_var_1d("TATM")
    forc_levels = _read_var_1d("FORC")

    cp_co2 = CONC_PREINDUSTRIAL["co2"]
    cp_ch4 = CONC_PREINDUSTRIAL["ch4"]
    cp_n2o = CONC_PREINDUSTRIAL["n2o"]

    ch4_decay = math.exp(-TSTEP / TAUGHG["ch4"])
    n2o_decay = math.exp(-TSTEP / TAUGHG["n2o"])
    co2_over_ch4 = 44.01 / 16.04

    for t in range(1, T):
        ts = str(t)
        tp1 = str(t + 1)

        # Current CD_SCALE
        cds = cd_scale.get(ts, 1.0)

        # 1. CO2 reservoirs: RES(box, t+1)
        w_co2 = w_emi.get(("co2", tp1), w_emi.get(("co2", ts), 0.0))

        # H2: Compute non-CO2 concentrations FIRST, then use CONC(ch4,tp1) for
        # methane oxidation (matching GAMS: OXI_CH4(tp1) uses CONC('ch4',tp1)).
        # 2. Non-CO2 concentrations (moved before CO2 reservoirs)
        for ghg in ["ch4", "n2o"]:
            conc_old = conc_levels.get((ghg, ts), CONC_PREINDUSTRIAL[ghg])
            w1 = w_emi.get((ghg, tp1), w_emi.get((ghg, ts), 0.0))
            w0 = w_emi.get((ghg, ts), 0.0)
            tau_ghg = TAUGHG[ghg]
            decay = math.exp(-TSTEP / tau_ghg)
            nat_emi = (CONC_PREINDUSTRIAL[ghg] / emitoconc[ghg]) * (1 - decay) / TSTEP
            conc_new = (conc_old * decay
                        + ((w1 + w0) / 2 + nat_emi) * emitoconc[ghg] * TSTEP)
            conc_levels[(ghg, tp1)] = max(conc_new, 1e-9)

        # Methane oxidation using CONC(ch4, tp1) -- GAMS eq_methoxi(tp1)
        conc_ch4_tp1 = conc_levels.get(("ch4", tp1), CONC_PREINDUSTRIAL["ch4"])
        ff_ch4_t = ff_ch4.get(ts, 0.3)
        ch4_oxi = (1e-3 * co2_over_ch4 * 0.61 * ff_ch4_t
                    * (conc_ch4_tp1 - cp_ch4) * (1 - ch4_decay))

        for box in BOX_NAMES:
            tau = TAUBOX[box]
            old = res_levels.get((box, ts), 0.0)
            new_val = (old * math.exp(-TSTEP / (tau * cds))
                       + EMSHARE[box] * (w_co2 + ch4_oxi) * emitoconc["co2"] * TSTEP)
            res_levels[(box, tp1)] = new_val

        # CO2 concentration
        conc_co2_tp1 = cp_co2 + sum(res_levels.get((b, tp1), 0) for b in BOX_NAMES)
        conc_levels[("co2", tp1)] = conc_co2_tp1

        # 3. Forcing
        c_co2 = conc_levels.get(("co2", tp1), cp_co2)
        c_ch4 = conc_levels.get(("ch4", tp1), cp_ch4)
        c_n2o = conc_levels.get(("n2o", tp1), cp_n2o)

        dc_co2 = c_co2 - cp_co2
        dc_ch4 = c_ch4 - cp_ch4
        dc_n2o = c_n2o - cp_n2o

        rf_co2 = ((-2.4e-7 * dc_co2**2
                   + 7.2e-4 * (math.sqrt(dc_co2**2 + DELTA**2) - DELTA)
                   - 1.05e-4 * (c_n2o + cp_n2o)
                   + 5.36) * math.log(max(c_co2, 1e-6) / cp_co2) / scaling_forc2x)

        rf_ch4 = ((-6.5e-7 * (c_ch4 + cp_ch4)
                   - 4.1e-6 * (c_n2o + cp_n2o) + 0.043)
                  * (math.sqrt(max(c_ch4, 0)) - math.sqrt(cp_ch4)))

        rf_n2o = ((-4.0e-6 * (c_co2 + cp_co2)
                   + 2.1e-6 * (c_n2o + cp_n2o)
                   - 2.45e-6 * (c_ch4 + cp_ch4) + 0.117)
                  * (math.sqrt(max(c_n2o, 0)) - math.sqrt(cp_n2o)))

        rf_h2o = 0.12 * rf_ch4

        # O3trop (simplified: no exogenous NOx/CO/NMVOC terms, same as Python module)
        tatm_t = tatm_levels.get(ts, 1.1)
        o3_temp_term = 0.032 * (math.exp(-1.35 * (tatm_t + dt0)) - 1)
        rf_o3 = (1.74e-4 * dc_ch4
                 + (o3_temp_term - math.sqrt(o3_temp_term**2 + 1e-16)) / 2)

        # Exogenous forcing (read from parameter if available)
        par_fex = data.get("_params", {}).get("par_forcing_exogenous")
        fex = 0.0
        if par_fex is not None and par_fex.records is not None:
            fex_row = par_fex.records[par_fex.records.iloc[:, 0] == tp1]
            if not fex_row.empty:
                fex = float(fex_row["value"].iloc[0])

        forc_tp1 = rf_co2 + rf_ch4 + rf_n2o + rf_h2o + rf_o3 + fex
        forc_levels[tp1] = forc_tp1

        # 4. Temperature
        tslow_t = tslow_levels.get(ts, _DERIVED["tslow0"])
        tfast_t = tfast_levels.get(ts, _DERIVED["tfast0"])
        forc_t = forc_levels.get(ts, forc_tp1)

        tslow_tp1 = tslow_t * slow_decay + QSLOW * forc_t * (1 - slow_decay)
        tfast_tp1 = tfast_t * fast_decay + QFAST * forc_t * (1 - fast_decay)
        tatm_tp1 = tslow_tp1 + tfast_tp1 - dt0

        tslow_levels[tp1] = tslow_tp1
        tfast_levels[tp1] = tfast_tp1
        tatm_levels[tp1] = tatm_tp1

        # 5. Update cumulative emissions and CD_SCALE for next step
        cum_old = cumemi_levels.get(ts, CONC_PREINDUSTRIAL["co2"] / emitoconc["co2"])
        cumemi_levels[tp1] = cum_old + (w_co2 + ch4_oxi) * TSTEP

        c_atm_tp1 = conc_co2_tp1 / emitoconc["co2"]
        c_sinks_tp1 = cumemi_levels[tp1] - (c_atm_tp1 - catm_pre)
        irf_rhs = min(IRF_PREINDUSTRIAL + IRC * c_sinks_tp1 * CO2toC
                      + IRT * (tatm_tp1 + dt0), 97.0)
        # Solve CD_SCALE from IRF (approximate: use Newton-style iteration)
        cds_new = cds  # start from current
        for _ in range(5):  # few Newton iterations
            irf_val = cds_new * sum(
                EMSHARE[b] * TAUBOX[b] * (1 - math.exp(-100 / (cds_new * TAUBOX[b])))
                for b in BOX_NAMES)
            irf_deriv = sum(
                EMSHARE[b] * TAUBOX[b] * (1 - math.exp(-100 / (cds_new * TAUBOX[b]))
                                           - 100 / (cds_new * TAUBOX[b])
                                           * math.exp(-100 / (cds_new * TAUBOX[b])))
                for b in BOX_NAMES)
            if abs(irf_deriv) < 1e-15:
                break
            cds_new = max(0.01, cds_new - (irf_val - irf_rhs) / irf_deriv)
        cd_scale[tp1] = min(max(cds_new, 0.01), 1000.0)

    # Write back all updated levels
    def _write_1d(varname, levels_dict):
        var = v.get(varname)
        if var is not None:
            recs = [(str(t), levels_dict.get(str(t), 0.0))
                    for t in range(1, T + 1)]
            var.setRecords(recs)

    def _write_2d(varname, levels_dict, dim1_keys):
        var = v.get(varname)
        if var is not None:
            recs = [(k1, str(t), levels_dict.get((k1, str(t)), 0.0))
                    for k1 in dim1_keys for t in range(1, T + 1)]
            var.setRecords(recs)

    _write_2d("RES", res_levels, BOX_NAMES)
    _write_2d("CONC", conc_levels, ["co2", "ch4", "n2o"])
    _write_1d("CUMEMI_FAIR", cumemi_levels)
    _write_1d("CD_SCALE", cd_scale)
    _write_1d("TSLOW", tslow_levels)
    _write_1d("TFAST", tfast_levels)
    _write_1d("TATM", tatm_levels)
    _write_1d("FORC", forc_levels)

    # Also update TOCEAN-equivalent (FAIR uses TSLOW+TFAST, no TOCEAN)
    # but TATM is the key variable for convergence checking.


# ---------------------------------------------------------------------------
#  Internal: convergence machinery
# ---------------------------------------------------------------------------

def _snapshot(v, data, cfg, tracked_vars):
    """Take a snapshot of tracked variable levels for convergence checking.

    Returns dict: varname -> {(t_str, n_str): float}

    GAMS tracked variables and their normalizations:
      viter(iter,'MIU',t,n) = MIU.l(t,n,'co2')     -- already 0..1
      viter(iter,'S',t,n)   = S.l(t,n)              -- already 0..1
      viter(iter,'Y',t,n)   = Y.l(t,n)/ykali(t,n)   -- ratio to baseline
      viter(iter,'TATM',t,n) = TATM.l(t)            -- absolute (all n)
    """
    region_names = data["region_names"]
    snap = {}

    for vname in tracked_vars:
        vals = {}
        if vname == "MIU":
            MIU = v.get("MIU")
            if MIU is not None and MIU.records is not None:
                for _, row in MIU.records.iterrows():
                    t_str = str(row.iloc[0])
                    n_str = str(row.iloc[1])
                    ghg_str = str(row.iloc[2])
                    if ghg_str == "co2":
                        vals[(t_str, n_str)] = row["level"]

        elif vname == "S":
            S = v.get("S")
            if S is not None and S.records is not None:
                for _, row in S.records.iterrows():
                    vals[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

        elif vname == "Y":
            Y = v.get("Y")
            ykali = data.get("ykali_dict", {})
            if Y is not None and Y.records is not None:
                for _, row in Y.records.iterrows():
                    t_str = str(row.iloc[0])
                    n_str = str(row.iloc[1])
                    t_int = int(t_str)
                    yk = ykali.get((t_int, n_str), 1.0)
                    vals[(t_str, n_str)] = (
                        row["level"] / yk if yk > 0 else row["level"])

        elif vname == "TATM":
            TATM = v.get("TATM")
            if TATM is not None and TATM.records is not None:
                for _, row in TATM.records.iterrows():
                    t_str = str(row.iloc[0])
                    tatm_val = row["level"]
                    for r in region_names:
                        vals[(t_str, r)] = tatm_val

        # D12: Additional tracked variables for optional modules
        elif vname == "E_NEG":
            E_NEG = v.get("E_NEG")
            if E_NEG is not None and E_NEG.records is not None:
                for _, row in E_NEG.records.iterrows():
                    vals[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

        elif vname == "W_SAI":
            W_SAI = v.get("W_SAI")
            if W_SAI is not None and W_SAI.records is not None:
                for _, row in W_SAI.records.iterrows():
                    t_str = str(row.iloc[0])
                    sai_val = row["level"]
                    for r in region_names:
                        vals[(t_str, r)] = sai_val

        elif vname == "MIULAND":
            MIULAND = v.get("MIULAND")
            if MIULAND is not None and MIULAND.records is not None:
                for _, row in MIULAND.records.iterrows():
                    vals[(str(row.iloc[0]), str(row.iloc[1]))] = row["level"]

        snap[vname] = vals

    return snap


def _compute_errors(current, prev, tracked_vars, cfg=None):
    """Compute max absolute error for each tracked variable.

    Mirrors GAMS:
      savediff(t,n,iter,v) = abs(viter(iter,v,t,n) - viter(iter-1,v,t,n))
      allerr(iter,v) = smax((t,n)$(not t5last(t)), savediff(t,n,iter,v))

    Excludes last 5 periods (GAMS: t5last) to avoid terminal noise.
    """
    # D13: Derive t5last cutoff from cfg.T rather than hardcoding 53
    T = cfg.T if cfg is not None else 58
    t5last_cutoff = T - 5  # periods > this are excluded

    errors = {}
    for vname in tracked_vars:
        cur_vals = current.get(vname, {})
        prv_vals = prev.get(vname, {})
        max_diff = 0.0

        for key, cur_v in cur_vals.items():
            t_int = int(key[0])
            if t_int > t5last_cutoff:
                continue
            prv_v = prv_vals.get(key, 0.0)
            diff = abs(cur_v - prv_v)
            if diff > max_diff:
                max_diff = diff

        errors[vname] = max_diff

    return errors


def _check_convergence(allerr, tol):
    """Check if all tracked variables have converged below tolerance."""
    return all(err < tol for err in allerr.values())


def _print_errors(allerr, iteration):
    """Print convergence error summary for an iteration."""
    parts = [f"{vn}={err:.6f}" for vn, err in allerr.items()]
    print(f"  Convergence errors: {', '.join(parts)}")


# ---------------------------------------------------------------------------
#  Internal: Negishi weight update
# ---------------------------------------------------------------------------

def _update_negishi_weights(v, data, cfg):
    """Update Negishi weights from solved CPC levels.

    GAMS core_cooperation.gms line 18: Negishi weights use the POSITIVE
    exponent so that richer regions (higher CPC) receive higher weight,
    equalizing weighted marginal utilities at the solution:
      nweights(t,n) = CPC.l(t,n)^(elasmu) / sum(nn, CPC.l(t,nn)^(elasmu))
    """
    CPC = v.get("CPC")
    if CPC is None or CPC.records is None or len(CPC.records) == 0:
        return

    region_names = data["region_names"]
    T = cfg.T
    elasmu = cfg.ELASMU

    cpc_levels = {}
    for _, row in CPC.records.iterrows():
        cpc_levels[(str(row.iloc[0]), str(row.iloc[1]))] = max(
            row["level"], 1e-8)

    params = data["_params"]
    par_nweights = params.get("par_nweights")
    if par_nweights is None:
        return

    new_records = []
    for t in range(1, T + 1):
        t_str = str(t)
        numerators = {}
        denom = 0.0
        for r in region_names:
            cpc_val = cpc_levels.get((t_str, r), 1e-8)
            num = cpc_val ** elasmu
            numerators[r] = num
            denom += num

        for r in region_names:
            nw = numerators[r] / denom if denom > 0 else 1.0
            new_records.append((t_str, r, nw))

    par_nweights.setRecords(new_records)


# ---------------------------------------------------------------------------
#  Internal: Nash region fixing/unfixing
# ---------------------------------------------------------------------------

_NASH_FIX_VARS_3D = ["MIU"]           # domain (t, n, ghg)
# All decision variables that need fixing in Nash mode.
# GAMS implicitly excludes other regions' equations via reg(n);
# Python must explicitly fix their decision variables.
_NASH_FIX_VARS_2D = [
    "S", "MIULAND",
    "E_NEG", "I_CDR",        # DAC (mod_dac)
    "N_SAI",                  # SAI (mod_sai)
]
# 3D variables where region is at index 2 (domain: [extra, t, n])
# These need region-based fixing like _NASH_FIX_VARS_3D but with
# the extra dimension at position 0 instead of position 2.
_NASH_FIX_VARS_3D_EXTRA = [
    "I_ADA",                  # Adaptation: (g, t, n)
    "K_ADA",                  # Adaptation: (g, t, n)
    "SAI",                    # SAI g6: (t, n, inj)
    "E_STOR",                 # CCS storage: (ccs_stor, t, n)
]
# Explicit region-column index for each extra 3D variable.
# Avoids fragile heuristic detection that can mis-identify SAI's inj column.
_3D_EXTRA_REGION_COL = {
    "I_ADA": 2,   # (g, t, n)
    "K_ADA": 2,   # (g, t, n)
    "SAI": 1,     # (t, n, inj)
    "E_STOR": 2,  # (ccs_stor, t, n)
}


def _safe_bound(val, default):
    """D19: Return val if finite and not NaN, else default."""
    import math as _math
    if val is None or (_math.isnan(val) if isinstance(val, float) else False):
        return default
    if _math.isinf(val):
        return default
    return val


def _fix_other_regions(v, other_regions, cfg):
    """Fix decision variables for regions NOT in the active coalition."""
    other_set = set(other_regions)

    for vname in _NASH_FIX_VARS_3D:
        var = v.get(vname)
        if var is None or var.records is None:
            continue
        if not hasattr(var, "_saved_bounds"):
            var._saved_bounds = {}
        for _, row in var.records.iterrows():
            n_str = str(row.iloc[1])
            if n_str in other_set:
                t_str = str(row.iloc[0])
                ghg_str = str(row.iloc[2])
                level = row["level"]
                key = (t_str, n_str, ghg_str)
                # D19: Explicit NaN/inf check instead of .get() fallback
                lo_raw = row.get("lower", None)
                up_raw = row.get("upper", None)
                var._saved_bounds[key] = (
                    _safe_bound(lo_raw, -float("inf")),
                    _safe_bound(up_raw, float("inf")))
                var.fx[t_str, n_str, ghg_str] = level

    for vname in _NASH_FIX_VARS_2D:
        var = v.get(vname)
        if var is None or var.records is None:
            continue
        if not hasattr(var, "_saved_bounds"):
            var._saved_bounds = {}
        for _, row in var.records.iterrows():
            n_str = str(row.iloc[1])
            if n_str in other_set:
                t_str = str(row.iloc[0])
                level = row["level"]
                key = (t_str, n_str)
                # D19: Explicit NaN/inf check
                lo_raw = row.get("lower", None)
                up_raw = row.get("upper", None)
                var._saved_bounds[key] = (
                    _safe_bound(lo_raw, -float("inf")),
                    _safe_bound(up_raw, float("inf")))
                var.fx[t_str, n_str] = level

    # Extra 3D vars with explicit region-column mapping
    for vname in _NASH_FIX_VARS_3D_EXTRA:
        var = v.get(vname)
        if var is None or var.records is None:
            continue
        if not hasattr(var, "_saved_bounds"):
            var._saved_bounds = {}
        ncols = var.records.shape[1]
        n_domain_cols = ncols - 4  # subtract level/marginal/lower/upper
        col_idx = _3D_EXTRA_REGION_COL.get(vname, 2)
        for _, row in var.records.iterrows():
            n_str = str(row.iloc[col_idx])
            if n_str in other_set:
                level = row["level"]
                key = tuple(str(row.iloc[i]) for i in range(n_domain_cols))
                if key not in var._saved_bounds:
                    lo_raw = row.get("lower", None)
                    up_raw = row.get("upper", None)
                    var._saved_bounds[key] = (
                        _safe_bound(lo_raw, -float("inf")),
                        _safe_bound(up_raw, float("inf")))
                var.fx[key] = level


def _unfix_other_regions(v, other_regions, cfg):
    """Restore decision variable bounds for non-active regions."""
    for vname in _NASH_FIX_VARS_3D:
        var = v.get(vname)
        if var is None:
            continue
        saved = getattr(var, "_saved_bounds", {})
        for key, (lo, up) in saved.items():
            t_str, n_str, ghg_str = key
            var.lo[t_str, n_str, ghg_str] = lo
            var.up[t_str, n_str, ghg_str] = up
        var._saved_bounds = {}

    for vname in _NASH_FIX_VARS_2D:
        var = v.get(vname)
        if var is None:
            continue
        saved = getattr(var, "_saved_bounds", {})
        for key, (lo, up) in saved.items():
            t_str, n_str = key
            var.lo[t_str, n_str] = lo
            var.up[t_str, n_str] = up
        var._saved_bounds = {}

    # Restore extra 3D variables (adaptation, SAI g6, E_STOR)
    for vname in _NASH_FIX_VARS_3D_EXTRA:
        var = v.get(vname)
        if var is None:
            continue
        saved = getattr(var, "_saved_bounds", {})
        for key, (lo, up) in saved.items():
            var.lo[key] = lo
            var.up[key] = up
        var._saved_bounds = {}


# ---------------------------------------------------------------------------
#  Internal: module ordering and parameter creation
# ---------------------------------------------------------------------------

def _module_order(cfg):
    """Return modules in dependency order."""
    # Impact submodule selection
    impact_map = {
        "dice": mod_impact_dice,
        "kalkuhl": mod_impact_kalkuhl,
        "burke": mod_impact_burke,
        "howard": mod_impact_howard,
        "dell": mod_impact_dell,
        "coacch": mod_impact_coacch,
        "climcost": mod_impact_coacch,  # same module, different damcost key
    }
    impact_mod = impact_map.get(cfg.impact, mod_impact_kalkuhl)

    # Climate submodule selection
    if cfg.climate == "fair":
        climate_mod = mod_climate_fair
    elif cfg.climate == "tatm_exogen":
        # Exogenous temperature: skip endogenous climate entirely.
        # mod_climate_tatm_exogen creates TATM/FORC/TOCEAN stubs and fixes TATM.
        climate_mod = mod_climate_tatm_exogen
    else:
        climate_mod = hub_climate  # witchco2 (default)

    # When impact_deciles is active, DAMAGES comes from mod_impact_deciles
    # instead of hub_impact. We still need hub_impact for OMEGA/DAMFRAC vars
    # but tell it to skip the damages equation.
    use_decile_damages = (getattr(cfg, "impact_deciles", False)
                          and getattr(cfg, "inequality", False))
    if use_decile_damages:
        cfg._decile_damages = True  # signal to hub_impact

    # Order follows modules/__init__.py docstring (respects variable dependencies)
    mods = [
        mod_landuse,           # ELAND, MIULAND, MACLAND, ABCOSTLAND
        hub_impact,            # OMEGA, DAMFRAC*, DAMAGES (needs YGROSS stub)
        climate_mod,           # W_EMI, FORC, TATM, etc.
    ]
    mods.append(mod_climate_regional)  # TEMP_REGION, TEMP_REGION_DAM
    # In decile mode, skip the normal impact module entirely to avoid
    # duplicate BIMPACT/eq_bimpact symbols. mod_impact_deciles replaces it.
    if not use_decile_damages:
        mods.append(impact_mod)       # BIMPACT + eq_omega

    # Optional: DAC (needs to come before core_emissions for E_NEG variable)
    if getattr(cfg, "dac", False):
        mods.append(mod_dac)
        # CCS storage module (per-type storage, cumulative tracking, leakage)
        # Must come after mod_dac (needs E_NEG) and before core_emissions
        mods.append(mod_emi_stor)

    # Optional: SAI (needs TATM from climate module)
    if getattr(cfg, "sai", False):
        mods.append(mod_sai)

    # Optional: Adaptation (needs TATM, provides Q_ADA for OMEGA modification)
    if getattr(cfg, "adaptation", False):
        mods.append(mod_adaptation)

    # Core modules
    mods.extend([
        core_emissions,        # E, EIND, MIU, ABATEDEMI (needs YGROSS, ELAND)
        core_abatement,        # ABATECOST, MAC (needs MIU)
        core_economy,          # YGROSS..RI (needs DAMAGES, ABATECOST, ABCOSTLAND)
    ])

    # Optional: Inequality (needs YGROSS, DAMAGES, ABATECOST, CPC)
    if getattr(cfg, "inequality", False):
        mods.append(mod_inequality)
        # Optional: Per-decile damages (needs YGROSS_DIST from inequality)
        if getattr(cfg, "impact_deciles", False):
            mods.append(mod_impact_deciles)

    mods.append(core_welfare)  # UTARG, UTILITY (needs CPC)

    # Optional: Ocean (needs CPC, TATM, YNET)
    if getattr(cfg, "ocean", False):
        mods.append(mod_ocean)

    # Optional: Natural Capital (needs TATM, YGROSS)
    if getattr(cfg, "natural_capital", False):
        mods.append(mod_natural_capital)

    # Optional: Labour (placeholder)
    if getattr(cfg, "labour", False):
        mods.append(mod_labour)

    # Optional: SLR (needs TATM)
    if getattr(cfg, "slr", False):
        mods.append(mod_slr)

    mods.append(core_policy)   # policy fixings (BAU/CBA) -- always last

    return mods


def _create_parameters(m, sets, data, cfg):
    """Create all GAMSPy Parameter objects from calibrated data."""
    t_set = sets["t_set"]
    n_set = sets["n_set"]
    layers = sets["layers"]
    ghg_set = sets["ghg_set"]
    rn = data["region_names"]
    T = cfg.T
    ghg_list = cfg.ghg_list

    def tn_records(d):
        return [(str(t), r, v) for (t, r), v in d.items() if r in rn]

    def tng_records(d):
        """Records from a dict keyed by (t, region, ghg)."""
        return [(str(t), r, g, v) for (t, r, g), v in d.items() if r in rn]

    def g_records(d):
        """Records from a dict keyed by ghg name."""
        return [(g, v) for g, v in d.items() if g in ghg_list]

    p = {}

    p["par_ykali"] = Parameter(m, name="ykali", domain=[t_set, n_set],
                                records=tn_records(data["ykali_dict"]))
    p["par_pop"] = Parameter(m, name="pop", domain=[t_set, n_set],
                              records=tn_records(data["pop_dict"]))
    p["par_tfp"] = Parameter(m, name="tfp", domain=[t_set, n_set],
                              records=tn_records(data["tfp"]))

    # sigma: now [t, n, ghg] -- fall back to CO2-only 2D dict if 3D not available
    sigma_data = data.get("sigma_agg_ghg")
    if sigma_data is not None:
        p["par_sigma"] = Parameter(m, name="sigma", domain=[t_set, n_set, ghg_set],
                                    records=tng_records(sigma_data))
    else:
        # Legacy CO2-only: broadcast to (t, n, "co2")
        sigma_co2 = data["sigma_agg"]
        legacy_sigma = {(t, r, "co2"): v for (t, r), v in sigma_co2.items()}
        p["par_sigma"] = Parameter(m, name="sigma", domain=[t_set, n_set, ghg_set],
                                    records=tng_records(legacy_sigma))

    p["par_k0"] = Parameter(m, name="k0", domain=[n_set],
                              records=[(r, v) for r, v in data["k0_agg"].items() if r in rn])
    p["par_fixed_savings"] = Parameter(m, name="fixed_savings", domain=[t_set, n_set],
                                        records=tn_records(data["fixed_savings"]))
    p["par_prodshare_cap"] = Parameter(m, name="prodshare_cap", domain=[n_set],
                                        records=[(r, v) for r, v in data["prodshare_cap"].items()])
    p["par_prodshare_lab"] = Parameter(m, name="prodshare_lab", domain=[n_set],
                                        records=[(r, v) for r, v in data["prodshare_lab"].items()])
    p["par_rr"] = Parameter(m, name="rr", domain=[t_set],
                             records=[(str(t), v) for t, v in data["rr"].items()])

    # emi_bau: now [t, n, ghg]
    emi_bau_data = data.get("emi_bau_ghg")
    if emi_bau_data is not None:
        p["par_emi_bau"] = Parameter(m, name="emi_bau", domain=[t_set, n_set, ghg_set],
                                      records=tng_records(emi_bau_data))
    else:
        legacy_emi = {(t, r, "co2"): v for (t, r), v in data["emi_bau_dict"].items()}
        p["par_emi_bau"] = Parameter(m, name="emi_bau", domain=[t_set, n_set, ghg_set],
                                      records=tng_records(legacy_emi))

    # par_emi_level: mirrors GAMS E.l in eq_yy fiscal revenue term.
    # Initialized to BAU emissions; updated to E.l each iteration in _before_solve.
    if emi_bau_data is not None:
        p["par_emi_level"] = Parameter(m, name="emi_level", domain=[t_set, n_set, ghg_set],
                                        records=tng_records(emi_bau_data))
    else:
        p["par_emi_level"] = Parameter(m, name="emi_level", domain=[t_set, n_set, ghg_set],
                                        records=tng_records(legacy_emi))

    p["par_eland_bau"] = Parameter(m, name="eland_bau", domain=[t_set, n_set],
                                    records=tn_records(data["eland_bau"]))
    p["par_eland_maxab"] = Parameter(m, name="eland_maxab", domain=[n_set],
                                      records=[(r, v) for r, v in data["eland_maxab_dict"].items()])
    p["par_lu_macc_c1"] = Parameter(m, name="lu_macc_c1", domain=[t_set, n_set],
                                     records=tn_records(data["lu_macc_c1_dict"]))
    p["par_lu_macc_c4"] = Parameter(m, name="lu_macc_c4", domain=[t_set, n_set],
                                     records=tn_records(data["lu_macc_c4_dict"]))

    # macc_c1, macc_c4: now [t, n, ghg]
    macc_c1_data = data.get("macc_c1_ghg")
    if macc_c1_data is not None:
        p["par_macc_c1"] = Parameter(m, name="macc_c1", domain=[t_set, n_set, ghg_set],
                                      records=tng_records(macc_c1_data))
    else:
        legacy_c1 = {(t, r, "co2"): v for (t, r), v in data["macc_c1_dict"].items()}
        p["par_macc_c1"] = Parameter(m, name="macc_c1", domain=[t_set, n_set, ghg_set],
                                      records=tng_records(legacy_c1))

    macc_c4_data = data.get("macc_c4_ghg")
    if macc_c4_data is not None:
        p["par_macc_c4"] = Parameter(m, name="macc_c4", domain=[t_set, n_set, ghg_set],
                                      records=tng_records(macc_c4_data))
    else:
        legacy_c4 = {(t, r, "co2"): v for (t, r), v in data["macc_c4_dict"].items()}
        p["par_macc_c4"] = Parameter(m, name="macc_c4", domain=[t_set, n_set, ghg_set],
                                      records=tng_records(legacy_c4))

    # GHG conversion factors (quantity and cost units)
    convq_data = data.get("convq_ghg", {"co2": 1.0, "ch4": 1e3, "n2o": 1e3})
    p["par_convq_ghg"] = Parameter(m, name="convq_ghg", domain=[ghg_set],
                                    records=g_records(convq_data))
    convy_data = data.get("convy_ghg", {"co2": 1e-3, "ch4": 1e-6, "n2o": 1e-6})
    p["par_convy_ghg"] = Parameter(m, name="convy_ghg", domain=[ghg_set],
                                    records=g_records(convy_data))

    # maxmiu_pbl: [t, n, ghg]
    maxmiu_data = data.get("maxmiu_pbl")
    if maxmiu_data is not None:
        p["par_maxmiu_pbl"] = Parameter(m, name="maxmiu_pbl", domain=[t_set, n_set, ghg_set],
                                         records=tng_records(maxmiu_data))
    else:
        # Default: MIU upper bound 1.0 for CO2, 1.0 for all GHGs
        default_maxmiu = {(t, r, g): 1.0
                          for t in range(1, T + 1) for r in rn for g in ghg_list}
        p["par_maxmiu_pbl"] = Parameter(m, name="maxmiu_pbl", domain=[t_set, n_set, ghg_set],
                                         records=tng_records(default_maxmiu))

    # GWP factors (optional)
    emi_gwp_data = data.get("emi_gwp")
    if emi_gwp_data is not None:
        p["par_emi_gwp"] = Parameter(m, name="emi_gwp", domain=[ghg_set],
                                      records=g_records(emi_gwp_data))

    p["par_alpha_temp"] = Parameter(m, name="alpha_temp", domain=[n_set],
                                     records=[(r, data["alpha_temp_dict"].get(r, 0.0)) for r in rn])
    p["par_beta_temp"] = Parameter(m, name="beta_temp", domain=[n_set],
                                    records=[(r, data["beta_temp_dict"].get(r, 1.0)) for r in rn])
    p["par_base_temp"] = Parameter(m, name="base_temp", domain=[n_set],
                                    records=[(r, data["base_temp_dict"].get(r, 0.0)) for r in rn])
    p["par_alpha_precip"] = Parameter(m, name="alpha_precip", domain=[n_set],
                                       records=[(r, data.get("alpha_precip_dict", {}).get(r, 0.0)) for r in rn])
    p["par_beta_precip"] = Parameter(m, name="beta_precip", domain=[n_set],
                                      records=[(r, data.get("beta_precip_dict", {}).get(r, 0.0)) for r in rn])

    # Climate scalars
    cmphi = data["cmphi"]
    p["par_cmphi"] = Parameter(m, name="cmphi", domain=[layers, layers],
                                records=[("atm", "atm", cmphi[0, 0]), ("atm", "upp", cmphi[0, 1]),
                                         ("atm", "low", cmphi[0, 2]), ("upp", "atm", cmphi[1, 0]),
                                         ("upp", "upp", cmphi[1, 1]), ("upp", "low", cmphi[1, 2]),
                                         ("low", "atm", cmphi[2, 0]), ("low", "upp", cmphi[2, 1]),
                                         ("low", "low", cmphi[2, 2])])
    p["par_sigma1"] = Parameter(m, name="sigma1", records=data["sigma1"])
    p["par_lambda"] = Parameter(m, name="lambda_clim", records=data["lam"])
    p["par_sigma2"] = Parameter(m, name="sigma2", records=data["sigma2"])
    p["par_heat_ocean"] = Parameter(m, name="heat_ocean", records=data["heat_ocean"])
    p["par_rfc_alpha"] = Parameter(m, name="rfc_alpha", records=data["rfc_alpha"])
    p["par_rfc_beta"] = Parameter(m, name="rfc_beta", records=data["rfc_beta"])
    p["par_oghg_intercept"] = Parameter(m, name="oghg_intercept", records=data["oghg_intercept"])
    p["par_oghg_slope"] = Parameter(m, name="oghg_slope", records=data["oghg_slope"])

    # Emission starting values -- GHG-dimensioned (Fix #11: removed max(,0) floor)
    emi_bau_src = data.get("emi_bau_ghg")
    if emi_bau_src is not None:
        p["par_emi_start"] = Parameter(m, name="emi_start", domain=[t_set, n_set, ghg_set],
                                        records=[(str(t), r, g,
                                                  emi_bau_src.get((t, r, g), 0.0))
                                                 for t in range(1, T + 1) for r in rn
                                                 for g in ghg_list])
        p["par_e_start"] = Parameter(m, name="e_start", domain=[t_set, n_set, ghg_set],
                                      records=[(str(t), r, g,
                                                emi_bau_src.get((t, r, g), 0.0)
                                                + (data["eland_bau"].get((t, r), 0.0)
                                                   if g == "co2" else 0.0))
                                               for t in range(1, T + 1) for r in rn
                                               for g in ghg_list])
    else:
        # Legacy CO2-only path (no max(,0) floor -- Fix #11)
        p["par_emi_start"] = Parameter(m, name="emi_start", domain=[t_set, n_set, ghg_set],
                                        records=[(str(t), r, "co2",
                                                  data["emi_bau_dict"].get((t, r), 0.0))
                                                 for t in range(1, T + 1) for r in rn])
        p["par_e_start"] = Parameter(m, name="e_start", domain=[t_set, n_set, ghg_set],
                                      records=[(str(t), r, "co2",
                                                data["emi_bau_dict"].get((t, r), 0.0)
                                                + data["eland_bau"].get((t, r), 0.0))
                                               for t in range(1, T + 1) for r in rn])

    # Raw scalars (for modules that need plain floats, not GAMSPy Parameters)
    p["TATM0"] = data["TATM0"]
    p["TOCEAN0"] = data["TOCEAN0"]
    p["wcum0"] = data["wcum0"]
    p["rfc_beta_scalar"] = data["rfc_beta"]
    p["wemi2qemi_co2"] = data["wemi2qemi_co2"]
    p["CO2toC"] = data["CO2toC"]
    p["convy_co2"] = data["convy_co2"]

    # K starting values for all periods (from TFP calibration)
    p["par_k_start"] = Parameter(m, name="k_start", domain=[t_set, n_set],
                                  records=[(str(t), r, max(data["k_tfp"].get((t, r), 1.0), 1e-4))
                                           for t in range(1, T + 1) for r in rn])

    # CPC starting values
    p["par_cpc_start"] = Parameter(m, name="cpc_start", domain=[t_set, n_set],
                                    records=[(str(t), r,
                                              max(data["ykali_dict"].get((t, r), 1)
                                                  * (1 - data["fixed_savings"].get((t, r), 0.2))
                                                  / max(data["pop_dict"].get((t, r), 1), 1e-6) * 1e6, 1.0))
                                             for t in range(1, T + 1) for r in rn])

    # Long-term pledges: net-zero year per region
    pledge_data = data.get("pledge_nz_year_co2")
    if pledge_data:
        p["par_pledge_nz_year_co2"] = Parameter(
            m, name="pledge_nz_year_co2", domain=[n_set],
            records=[(r, pledge_data[r]) for r in rn if r in pledge_data])

    # GHG pledge years (GAMS default: 2310 = effectively disabled)
    pledge_ghg_data = data.get("pledge_nz_year_ghg")
    if pledge_ghg_data:
        p["par_pledge_nz_year_ghg"] = Parameter(
            m, name="pledge_nz_year_ghg", domain=[n_set],
            records=[(r, pledge_ghg_data[r]) for r in rn if r in pledge_ghg_data])

    # Country-level BAU emissions for NDC weighting (raw Python dict, not GAMSPy Parameter)
    emi_bau_country = data.get("emi_bau_country")
    if emi_bau_country is not None:
        p["_emi_bau_country"] = emi_bau_country

    # Carbon debt for historical_responsibility burden sharing (raw dict)
    carbon_debt = data.get("carbon_debt_by_region")
    if carbon_debt is not None:
        p["_carbon_debt_by_region"] = carbon_debt

    # Country-level ykali/pop for Burke rich/poor classification (raw dicts)
    ykali_cty = data.get("ykali_country")
    if ykali_cty is not None:
        p["_ykali_country"] = ykali_cty
    pop_cty = data.get("pop_country")
    if pop_cty is not None:
        p["_pop_country"] = pop_cty

    # Initial savings rate s0 (for flexible savings mode)
    # GAMS core_economy.gms: s0('savings_rate', '1', n)
    p["par_s0"] = Parameter(m, name="s0", domain=[n_set],
                             records=[(r, data["s0_agg"].get(r, 0.2)) for r in rn])

    # Ocean VSL: US GDP per capita at t=1 for VSL normalization
    # GAMS: VSL = vsl_start * global_gdppc(t) / US_gdppc(1)
    # Find the US region in GCAM-32 and compute its GDP per capita
    us_region_name = None
    for r in rn:
        if r.lower() in ("usa", "united states"):
            us_region_name = r
            break
    if us_region_name is not None:
        us_yk = data["ykali_dict"].get((1, us_region_name), 0)
        us_pop = data["pop_dict"].get((1, us_region_name), 1)
        if us_pop > 0 and us_yk > 0:
            us_gdppc_1 = us_yk / us_pop * 1e6  # T$/million -> $/person
            p["par_us_gdppc_1"] = Parameter(
                m, name="us_gdppc_1", records=us_gdppc_1)

    # par_ynet_level: mirrors GAMS YNET.l for ocean VSL/mangrove.
    # Initialized to ykali; updated to YNET.l in _before_solve.
    p["par_ynet_level"] = Parameter(
        m, name="ynet_level", domain=[t_set, n_set],
        records=tn_records(data["ykali_dict"]))

    # D7: DAC total cost parameter (for iterative learning-curve updates)
    # Created unconditionally so _update_dac_learning can write back to it.
    if getattr(cfg, "dac", False):
        from pydice32.modules.mod_dac import DAC_TOT0
        dac_cost_records = [(str(t), r, DAC_TOT0)
                            for t in range(1, T + 1) for r in rn]
        p["par_dac_totcost"] = Parameter(
            m, name="dac_totcost", domain=[t_set, n_set],
            records=dac_cost_records)

    # ------------------------------------------------------------------
    # Issue 13: Adaptation CES parameters (per-region, from data_mod_damage)
    # ------------------------------------------------------------------
    ces_ada = data.get("ces_ada_agg", {})
    owa = data.get("owa_agg", {})

    def _ada_param(name, source, param_key):
        """Create per-region adaptation Parameter if data exists."""
        recs = [(r, source.get((param_key, r), 0.0)) for r in rn
                if (param_key, r) in source]
        if recs:
            p[name] = Parameter(m, name=name.replace("par_", ""),
                                domain=[n_set], records=recs)

    _ada_param("par_ces_ada_tfpada", ces_ada, "tfpada")
    _ada_param("par_ces_ada_ada", ces_ada, "ada")
    _ada_param("par_ces_ada_act", ces_ada, "act")
    _ada_param("par_ces_ada_cap", ces_ada, "cap")
    _ada_param("par_ces_ada_exp", ces_ada, "exp")
    _ada_param("par_owa_act", owa, "act")
    _ada_param("par_owa_cap", owa, "cap")
    _ada_param("par_owa_rada", owa, "rada")
    _ada_param("par_owa_prada", owa, "prada")
    _ada_param("par_owa_gcap", owa, "gcap")
    _ada_param("par_owa_scap", owa, "scap")
    _ada_param("par_owa_actc", owa, "actc")

    # gcap scale: (k_h0 + k_edu0) / 2
    k_h0 = data.get("k_h0_agg", {})
    k_edu0 = data.get("k_edu0_agg", {})
    if k_h0 or k_edu0:
        gcap_recs = [(r, (k_h0.get(r, 0.0) + k_edu0.get(r, 0.0)) / 2.0)
                     for r in rn]
        p["par_gcap_scale"] = Parameter(m, name="gcap_scale",
                                         domain=[n_set], records=gcap_recs)

    # ada_exp scalar for hub_impact (legacy fallback)
    if ces_ada:
        exp_vals = [v for (k, r), v in ces_ada.items() if k == "exp"]
        if exp_vals:
            p["ada_exp_scalar"] = sum(exp_vals) / len(exp_vals)

    # OMEGA positive indicator for adaptation gate (GAMS: $(OMEGA.l(t,n) gt 0))
    # Initialized to 1 for all (t, n) = assume damages. Updated in _before_solve.
    if cfg.adaptation:
        omega_pos_records = [(str(t), r, 1.0)
                             for t in range(1, T + 1) for r in rn]
        p["par_omega_positive"] = Parameter(
            m, name="omega_positive", domain=[t_set, n_set],
            records=omega_pos_records)

    # Region weights for welfare function (Negishi or population-based)
    # GAMS core_cooperation.gms line 18: Negishi weights use the POSITIVE
    # exponent, equalizing weighted marginal utilities at the solution:
    #   nweights(t,n) = CPC(t,n)^(elasmu) / sum(nn, CPC(t,nn)^(elasmu))
    # Updated iteratively in solve_model_iterative/_update_negishi_weights.
    # For single-pass optimization: compute initial weights from starting CPC.
    if cfg.region_weights == "negishi":
        elasmu = cfg.ELASMU
        nweights_records = []
        for t in range(1, T + 1):
            cpc_vals = {}
            for r in rn:
                yk = data["ykali_dict"].get((t, r), 1.0)
                s = data["fixed_savings"].get((t, r), 0.2)
                pop_val = max(data["pop_dict"].get((t, r), 1.0), 1e-6)
                cpc_vals[r] = max(yk * (1 - s) / pop_val * 1e6, 1e-8)
            numerators = {r: cpc_vals[r] ** elasmu for r in rn}
            denom = sum(numerators.values())
            for r in rn:
                nw = numerators[r] / denom if denom > 0 else 1.0
                nweights_records.append((str(t), r, nw))
        p["par_nweights"] = Parameter(m, name="nweights", domain=[t_set, n_set],
                                       records=nweights_records)
    else:
        # Population weights: nweights = 1 (population weighting is applied
        # directly in the disentangled SWF equation; for DICE SWF, unit
        # weights mean all regions contribute equally before pop scaling)
        p["par_nweights"] = Parameter(m, name="nweights", domain=[t_set, n_set],
                                       records=[(str(t), r, 1.0)
                                                for t in range(1, T + 1) for r in rn])

    # ------------------------------------------------------------------
    # Stochastic SWF parameters
    # GAMS mod_stochastic.gms: PROB(t), branch_node(t,branch), rra
    #
    # par_prob[t]           : probability weight per time-state node
    # par_is_pre_res[t]     : 1.0 if Ord(t) < t_resolution, else 0.0
    # par_is_branch1[t]     : 1.0 if t belongs to branch_1, else 0.0
    # par_year_map[t, tt]   : 1.0 if year(t) == year(tt), else 0.0
    #                         (for cross-branch aggregation in stochastic SWF)
    # ------------------------------------------------------------------
    t_resolution = cfg.t_resolution
    num_branches = cfg.num_branches
    branch_probs = cfg.branch_probs

    if cfg.swf == "stochastic":
        # Build branch_node mapping (reproduces GAMS mod_stochastic.gms lines 47-51)
        #
        # In the GAMS model, post-resolution periods are split into branches.
        # Each branch has `span` periods, and `t` indices are allocated
        # sequentially: branch_0 gets t_resolution..(t_resolution+span-1),
        # branch_1 gets (t_resolution+span)..(t_resolution+2*span-1), etc.
        #
        # GAMS branch allocation (mod_stochastic.gms lines 47-51):
        # span = t_resolution_two - t_resolution_one (periods per branch)
        # branch_of_t = round((t - (t_resolution-1)) / span + 0.499)
        #
        # Dynamic adjustment: if T < t_resolution_two, reduce t_res2 so that
        # branches fit within T.  With the GAMS default t_resolution_two=59,
        # T must be >= 59 + num_branches*51 for full branching.  When T is
        # smaller (e.g. T=32 or T=58), we set effective_t_res2 = min(configured,
        # T+1) so that span shrinks to fit all branches within the available
        # time horizon.  For example:
        #   T=58: effective_t_res2 = min(59, 59) = 59, span = 51
        #   T=32: effective_t_res2 = min(59, 33) = 33, span = 25
        effective_t_res2 = min(cfg.t_resolution_two, T + 1)
        t_res2 = effective_t_res2
        span = max(t_res2 - t_resolution, 1)

        branch_of_t = {}  # t -> branch_index (0-based)
        for t in range(1, T + 1):
            if t >= t_resolution:
                # GAMS formula: round(((t-(t_resolution-1))/span)+0.499) gives 1-based branch
                gams_branch = round(((t - (t_resolution - 1)) / span) + 0.499)
                b = max(0, min(int(gams_branch) - 1, num_branches - 1))  # 0-based
                branch_of_t[t] = b

        # PROB: pre-resolution = 1.0, post-resolution = branch probability
        prob_records = []
        for t in range(1, T + 1):
            if t < t_resolution:
                prob_records.append((str(t), 1.0))
            else:
                b = branch_of_t.get(t, 0)
                prob_records.append((str(t), branch_probs[b]))
        p["par_prob"] = Parameter(m, name="prob", domain=[t_set],
                                   records=prob_records)

        # is_pre_res: 1 where t < t_resolution
        p["par_is_pre_res"] = Parameter(
            m, name="is_pre_res", domain=[t_set],
            records=[(str(t), 1.0 if t < t_resolution else 0.0)
                     for t in range(1, T + 1)])

        # is_branch1: 1 where t belongs to branch_1 (branch index 0)
        # In the GAMS formulation, the outer sum iterates over branch_1 only
        # (one representative per calendar year), so this picks one branch.
        p["par_is_branch1"] = Parameter(
            m, name="is_branch1", domain=[t_set],
            records=[(str(t), 1.0 if branch_of_t.get(t, -1) == 0 else 0.0)
                     for t in range(1, T + 1)])

        # year_map[t, tt]: 1 where year(t) == year(tt) across branches
        # GAMS: $sameas(year(tt), year(t)) — calendar year matching
        # Two periods share a year if their calendar year is the same.
        # year(t) = 2015 + TSTEP * (t - 1)
        year_of_t = {t: cfg.year(t) for t in range(1, T + 1) if t >= t_resolution}

        year_map_records = []
        for t1, yr1 in year_of_t.items():
            for t2, yr2 in year_of_t.items():
                if yr1 == yr2:
                    year_map_records.append((str(t1), str(t2), 1.0))
        p["par_year_map"] = Parameter(
            m, name="year_map", domain=[t_set, sets["t_alias"]],
            records=year_map_records)

    else:
        # Deterministic default: PROB = 1.0 everywhere, no branches
        p["par_prob"] = Parameter(m, name="prob", domain=[t_set],
                                   records=[(str(t), 1.0)
                                            for t in range(1, T + 1)])
        p["par_is_pre_res"] = Parameter(
            m, name="is_pre_res", domain=[t_set],
            records=[(str(t), 1.0) for t in range(1, T + 1)])
        p["par_is_branch1"] = Parameter(
            m, name="is_branch1", domain=[t_set],
            records=[(str(t), 0.0) for t in range(1, T + 1)])
        p["par_year_map"] = Parameter(
            m, name="year_map", domain=[t_set, sets["t_alias"]],
            records=[("1", "1", 0.0)])  # dummy record for dimensionality

    # ------------------------------------------------------------------
    # SAI g6 emulator parameters (when sai=True and sai_experiment="g6")
    # ------------------------------------------------------------------
    if getattr(cfg, "sai", False) and getattr(cfg, "sai_experiment", "g0") == "g6":
        from pydice32.modules.mod_sai import INJ_LABELS as _SAI_INJ
        if "inj_set" not in sets:
            inj_set = Set(m, name="inj", records=_SAI_INJ,
                           description="possible injection points for SAI")
            sets["inj_set"] = inj_set
        else:
            inj_set = sets["inj_set"]

        sai_temp_data = data.get("sai_temp_dict", {})
        sai_precip_data = data.get("sai_precip_dict", {})

        sai_temp_recs = [(r, inj, sai_temp_data.get((r, inj), 0.0))
                         for r in rn for inj in _SAI_INJ]
        p["par_sai_temp"] = Parameter(
            m, name="sai_temp", domain=[n_set, inj_set],
            records=sai_temp_recs)

        sai_precip_recs = [(r, inj, sai_precip_data.get((r, inj), 0.0))
                           for r in rn for inj in _SAI_INJ]
        p["par_sai_precip"] = Parameter(
            m, name="sai_precip", domain=[n_set, inj_set],
            records=sai_precip_recs)

        # Also store the raw dict for declare_vars to use
        p["sai_data"] = {
            "sai_temp": sai_temp_data,
            "sai_precip": sai_precip_data,
        }

    return p
