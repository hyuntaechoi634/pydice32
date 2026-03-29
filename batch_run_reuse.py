"""
Batch runner with MACC reuse: build once per family, swap MACC & re-solve.

A 'family' is a set of scenarios that share the same policy/impact/config
but differ only in macc_costs (prob25/prob50/prob75). Within a family,
the GAMSPy model is built once and MACC parameters are swapped via
setRecords() before each re-solve, leveraging CONOPT warm starts.

Usage:
    python -m pydice32.batch_run_reuse
"""

import os
import time
import pandas as pd

from pydice32.config import Config
from pydice32.solver import build_model, solve_model, update_macc_params
from pydice32.data.calibration import compute_macc_bundle


# ── Scenario families ────────────────────────────────────────────────
# Each family: (name, config_kwargs)
# MACC variants within each family: prob50 (center) -> prob25 -> prob75
MACC_ORDER = ["prob50", "prob25", "prob75"]

FAMILIES = [
    ("bau_kalkuhl",    dict(policy="bau_impact", impact="kalkuhl")),
    ("cba_kalkuhl",    dict(policy="cba", impact="kalkuhl")),
    ("ctax_kalkuhl",   dict(policy="ctax", impact="kalkuhl", ctax_initial=50, ctax_slope=0.03)),
    ("nz2050_kalkuhl", dict(policy="global_netzero", nz_year=2050, impact="kalkuhl")),
    ("cbudget_1150",   dict(policy="cbudget", cbudget=1150)),
    ("cea_2C",         dict(policy="cea_tatm", tatm_limit=2.0)),
]

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), "results_batch")


def extract_results(v, cfg, scenario_name):
    """Extract key results into a dict of time series."""
    T = cfg.T
    TSTEP = cfg.TSTEP
    t_range = list(range(1, min(T + 1, 59)))
    years = [2015 + (t - 1) * TSTEP for t in t_range]
    rows = []

    tatm_recs = v["TATM"].records if "TATM" in v else None
    e_recs = v["E"].records if "E" in v else None
    miu_recs = v["MIU"].records if "MIU" in v else None
    y_recs = v["Y"].records if "Y" in v else None

    for t, yr in zip(t_range, years):
        ts = str(t)
        row = {"scenario": scenario_name, "t": t, "year": yr}

        # TATM
        if tatm_recs is not None:
            mask = tatm_recs.iloc[:, 0] == ts
            row["TATM"] = float(tatm_recs.loc[mask, "level"].iloc[0]) if mask.any() else 0

        # World E_CO2
        if e_recs is not None:
            mask = (e_recs.iloc[:, 0] == ts) & (e_recs.iloc[:, 2] == "co2")
            row["E_CO2_world"] = e_recs.loc[mask, "level"].sum()

        # World MIU_CO2 average
        if miu_recs is not None:
            mask = (miu_recs.iloc[:, 0] == ts) & (miu_recs.iloc[:, 2] == "co2")
            vals = miu_recs.loc[mask, "level"]
            row["MIU_CO2_avg"] = vals.mean() if len(vals) > 0 else 0

        # World Y
        if y_recs is not None:
            mask = y_recs.iloc[:, 0] == ts
            row["Y_world"] = y_recs.loc[mask, "level"].sum()

        rows.append(row)

    return pd.DataFrame(rows)


def run_family(family_name, base_kwargs):
    """Run a single family: build once, swap MACC & re-solve for each variant."""
    print(f"\n{'='*60}")
    print(f"FAMILY: {family_name}")
    print(f"{'='*60}")

    family_results = []
    timing = []
    m, rice, v, data = None, None, None, None

    for i, macc in enumerate(MACC_ORDER):
        scenario_name = f"{family_name}__{macc}"
        print(f"\n  --- {scenario_name} ---")

        if i == 0:
            # First run: full build + solve
            cfg = Config(**base_kwargs, macc_costs=macc)

            t0 = time.perf_counter()
            m, rice, v, data = build_model(cfg)
            t_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            solve_model(rice, cfg)
            t_solve = time.perf_counter() - t0

            timing.append(dict(
                scenario=scenario_name, build_s=t_build, swap_s=0,
                solve_s=t_solve, total_s=t_build + t_solve,
                reuse=False, status=str(rice.solve_status)))
        else:
            # Subsequent runs: MACC swap + re-solve (warm start)
            base_ctx = data.get("_macc_base_ctx")
            if base_ctx is None:
                print("  WARNING: no MACC base context, falling back to full build")
                cfg = Config(**base_kwargs, macc_costs=macc)
                t0 = time.perf_counter()
                m, rice, v, data = build_model(cfg)
                t_build = time.perf_counter() - t0
                t0 = time.perf_counter()
                solve_model(rice, cfg)
                t_solve = time.perf_counter() - t0
                timing.append(dict(
                    scenario=scenario_name, build_s=t_build, swap_s=0,
                    solve_s=t_solve, total_s=t_build + t_solve,
                    reuse=False, status=str(rice.solve_status)))
            else:
                t0 = time.perf_counter()
                bundle = compute_macc_bundle(base_ctx, macc)
                update_macc_params(data, bundle)
                t_swap = time.perf_counter() - t0

                t0 = time.perf_counter()
                solve_model(rice, cfg)
                t_solve = time.perf_counter() - t0

                timing.append(dict(
                    scenario=scenario_name, build_s=0, swap_s=t_swap,
                    solve_s=t_solve, total_s=t_swap + t_solve,
                    reuse=True, status=str(rice.solve_status)))

        print(f"  Status: {rice.status} / {rice.solve_status}")
        t_info = timing[-1]
        if t_info["reuse"]:
            print(f"  Time: swap={t_info['swap_s']:.1f}s, solve={t_info['solve_s']:.1f}s")
        else:
            print(f"  Time: build={t_info['build_s']:.1f}s, solve={t_info['solve_s']:.1f}s")

        # Extract results
        df = extract_results(v, cfg, scenario_name)
        family_results.append(df)

    return family_results, timing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    all_timing = []

    for family_name, base_kwargs in FAMILIES:
        results, timing = run_family(family_name, base_kwargs)
        all_results.extend(results)
        all_timing.extend(timing)

    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, "all_scenarios.csv"), index=False)
        print(f"\nResults saved to {OUT_DIR}/all_scenarios.csv")

    # Save timing
    if all_timing:
        timing_df = pd.DataFrame(all_timing)
        timing_df.to_csv(os.path.join(OUT_DIR, "timing.csv"), index=False)
        print(f"Timing saved to {OUT_DIR}/timing.csv")

        # Summary
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        for _, row in timing_df.iterrows():
            reuse_str = "REUSE" if row["reuse"] else "FRESH"
            print(f"  {row['scenario']:40s} {row['total_s']:6.1f}s  [{reuse_str}] {row['status']}")
        total_time = timing_df["total_s"].sum()
        reuse_time = timing_df[timing_df["reuse"]]["total_s"].sum()
        fresh_time = timing_df[~timing_df["reuse"]]["total_s"].sum()
        print(f"\n  Total: {total_time:.1f}s = fresh {fresh_time:.1f}s + reuse {reuse_time:.1f}s")


if __name__ == "__main__":
    main()
