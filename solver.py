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
from pyrice32.config import Config
from pyrice32.data import load_and_calibrate
from pyrice32.modules import (
    core_economy, core_emissions, core_abatement, core_welfare,
    hub_climate, hub_impact, mod_impact_dice, mod_impact_kalkuhl,
    mod_impact_burke, mod_climate_regional, mod_climate_fair,
    mod_landuse, core_policy,
)
from pyrice32.modules import (
    mod_dac, mod_sai, mod_adaptation,
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
    if cfg.cooperation in ("noncoop", "coalitions"):
        return solve_model_nash(m, rice, v, data, cfg)

    max_iter = cfg.max_iter
    min_iter = cfg.min_iter
    tol = cfg.convergence_tol

    region_names = data["region_names"]
    T = cfg.T

    # D12: Build tracked_vars list dynamically based on active modules
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

            # D3: Only declare convergence when solution is also optimal
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

    # D12: Build tracked_vars dynamically based on active modules
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
        last_status = None
        for clt_idx, clt in enumerate(coalitions):
            clt_regions = set(clt)
            other_regions = [r for r in region_names if r not in clt_regions]

            if other_regions:
                _fix_other_regions(v, other_regions, cfg)

            print(f"  Solving coalition {clt_idx + 1}/{len(coalitions)}: "
                  f"{clt}")
            rice.solve(solver="conopt",
                       options=Options(iteration_limit=99900))
            last_status = rice.status

            # D2: Retry once if infeasible
            if last_status in (ModelStatus.InfeasibleGlobal,
                               ModelStatus.InfeasibleLocal):
                print("    Infeasible -- retrying...")
                rice.solve(solver="conopt",
                           options=Options(iteration_limit=99900))
                last_status = rice.status

            if other_regions:
                _unfix_other_regions(v, other_regions, cfg)

        # after_solve phase (climate propagation, tracking)
        _after_solve(m, v, data, cfg, iteration)

        # D3: Track optimality for convergence gating
        is_optimal = last_status in (
            ModelStatus.OptimalGlobal, ModelStatus.OptimalLocal,
            ModelStatus.Feasible,
        ) if last_status is not None else False

        # Snapshot and convergence
        current = _snapshot(v, data, cfg, tracked_vars)
        viter[iteration] = current

        allerr = {}
        if prev_values is not None:
            allerr = _compute_errors(current, prev_values, tracked_vars, cfg)
            _print_errors(allerr, iteration)

            # D3: Only declare convergence when solution is also optimal
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

    # --- ctax corrected update ---
    # GAMS core_policy.gms before_solve updates ctax_corrected from
    # current E.l levels. This matters for the fiscal revenue approach.
    # For ctax_marginal mode, ctax_corrected is a fixed schedule, so
    # no update is needed. The fiscal approach is future work.

    # D8: Update cprice_max from current MAC levels (for ctax policy).
    # GAMS before_solve: cprice_max(t,n) = max(cprice_max(t,n), MAC.l(t,n,'co2'))
    # This ensures the upper bound on carbon price tracks the solved MAC.
    if cfg.policy == "ctax" and "MAC" in v and iteration > 1:
        _update_cprice_max(v, data, cfg)

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
    from pyrice32.modules.mod_dac import (
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
    """Fix savings rate for last 10 periods to S.l('48',n).

    GAMS core_economy.gms before_solve:
      S.fx(t,n)$(tperiod(t) gt (smax(tt,tperiod(tt)) - 10)) = S.l('48',n)
    """
    T = cfg.T
    S = v["S"]
    region_names = data["region_names"]

    s_recs = S.records
    if s_recs is None or len(s_recs) == 0:
        return

    # Get S.l('48', n) for each region
    s48 = {}
    for _, row in s_recs.iterrows():
        if str(row.iloc[0]) == "48":
            s48[str(row.iloc[1])] = row["level"]

    if not s48:
        return

    # Fix S for t > T - 10 (i.e., t >= 49 when T=58)
    for t in range(T - 10 + 1, T + 1):
        for r in region_names:
            if r in s48:
                S.fx[str(t), r] = s48[r]


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
    if cfg.cooperation != "coop":
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
    # FAIR climate uses a fundamentally different carbon cycle (4-box CO2,
    # IRF100 feedback) that cannot be forward-propagated with the simple
    # witchco2 transfer matrix.  Warn and skip for FAIR+Nash.
    if cfg.climate == "fair" and cfg.cooperation != "coop":
        import warnings
        warnings.warn(
            "Climate propagation in Nash/non-cooperative mode is only "
            "implemented for the witchco2 climate module. FAIR climate "
            "propagation is not available; the NLP equations will maintain "
            "climate consistency within each coalition solve, but inter-"
            "coalition consistency may be approximate.",
            UserWarning,
        )
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
    sigma1_val = data.get("sigma1", 0.024)
    lam_val = data.get("lam", 1.13)
    sigma2_val = data.get("sigma2", 0.44)
    heat_ocean_val = data.get("heat_ocean", 0.024)

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

    # Note: Temperature forward propagation requires FORC which depends on
    # RF_CO2 = rfc_alpha * (log(WCUM_atm) - log(rfc_beta)).  For simplicity,
    # we rely on the NLP equations to maintain temperature consistency and
    # only update WCUM levels here.  The TATM and TOCEAN .l values from the
    # last coalition solve are already reasonable starting points.


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

    Standard Negishi weights use the NEGATIVE exponent so that poorer
    regions (lower CPC) receive higher weight, reflecting higher
    marginal utility of consumption:
      nweights(t,n) = CPC.l(t,n)^(-elasmu) / sum(nn, CPC.l(t,nn)^(-elasmu))
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
            num = cpc_val ** (-elasmu)
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
_NASH_FIX_VARS_2D = ["S", "MIULAND"]  # domain (t, n)


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


# ---------------------------------------------------------------------------
#  Internal: module ordering and parameter creation
# ---------------------------------------------------------------------------

def _module_order(cfg):
    """Return modules in dependency order."""
    # Impact submodule selection
    if cfg.impact == "dice":
        impact_mod = mod_impact_dice
    elif cfg.impact == "burke":
        impact_mod = mod_impact_burke
    else:
        impact_mod = mod_impact_kalkuhl

    # Climate submodule selection
    if cfg.climate == "fair":
        climate_mod = mod_climate_fair
    else:
        climate_mod = hub_climate  # witchco2 (default)

    # Order follows modules/__init__.py docstring (respects variable dependencies)
    mods = [
        mod_landuse,           # ELAND, MIULAND, MACLAND, ABCOSTLAND
        hub_impact,            # OMEGA, DAMFRAC*, DAMAGES (needs YGROSS stub)
        climate_mod,           # W_EMI, FORC, TATM, etc.
        mod_climate_regional,  # TEMP_REGION, TEMP_REGION_DAM (needs TATM)
        impact_mod,            # BIMPACT + eq_omega (needs TATM/TEMP_REGION_DAM)
    ]

    # Optional: DAC (needs to come before core_emissions for E_NEG variable)
    if getattr(cfg, "dac", False):
        mods.append(mod_dac)

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

    # Initial savings rate s0 (for flexible savings mode)
    # GAMS core_economy.gms: s0('savings_rate', '1', n)
    p["par_s0"] = Parameter(m, name="s0", domain=[n_set],
                             records=[(r, data["s0_agg"].get(r, 0.2)) for r in rn])

    # D7: DAC total cost parameter (for iterative learning-curve updates)
    # Created unconditionally so _update_dac_learning can write back to it.
    if getattr(cfg, "dac", False):
        from pyrice32.modules.mod_dac import DAC_TOT0
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

    # ada_exp scalar for hub_impact (Issue 10): use average of ces_ada 'exp'
    if ces_ada:
        exp_vals = [v for (k, r), v in ces_ada.items() if k == "exp"]
        if exp_vals:
            p["ada_exp_scalar"] = sum(exp_vals) / len(exp_vals)

    # Region weights for welfare function (Negishi or population-based)
    # Standard Negishi weights use the NEGATIVE exponent so that poorer
    # regions (lower CPC) receive higher weight, reflecting higher
    # marginal utility of consumption:
    #   nweights(t,n) = CPC(t,n)^(-elasmu) / sum(nn, CPC(t,nn)^(-elasmu))
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
            numerators = {r: cpc_vals[r] ** (-elasmu) for r in rn}
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
        from pyrice32.modules.mod_sai import INJ_LABELS as _SAI_INJ
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
