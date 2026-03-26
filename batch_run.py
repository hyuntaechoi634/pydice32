"""
Batch scenario runner: 5 policies × 2 damage functions × 3 MACC costs = 30 runs.
Saves results to pyrice32/results/ as CSV files.
Tracks time, memory for each run.
"""

import os
import time
import traceback
import pandas as pd

# Memory tracking
try:
    import resource
    def get_mem_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on Linux
except ImportError:
    def get_mem_mb():
        return 0

from pyrice32.config import Config
from pyrice32.solver import build_model, solve_model


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

POLICIES = [
    ("bau", {}),
    ("cba", {}),
    ("cbudget", {"cbudget": 1150}),  # ~2°C
    ("cbudget", {"cbudget": 500}),   # ~1.5°C
    ("cea_tatm", {"tatm_limit": 2.0}),
]

# Note: these are explicit user choices -- they override policy defaults
# because Config.impact sentinel "" is only the initial default.
IMPACTS = ["dice", "kalkuhl"]

MACC_COSTS = ["prob25", "prob50", "prob75"]


def scenario_name(policy, policy_kw, impact, macc):
    suffix = ""
    if policy == "cbudget":
        suffix = f"_{int(policy_kw.get('cbudget', 0))}"
    elif policy == "cea_tatm":
        suffix = f"_{policy_kw.get('tatm_limit', 2.0)}"
    return f"{policy}{suffix}__{impact}__{macc}"


def extract_results(m, v, cfg):
    """Extract key time series into a dict of DataFrames."""
    results = {}

    T = cfg.T
    years = [2015 + 5 * (t - 1) for t in range(1, T + 1)]
    t_strs = [str(t) for t in range(1, T + 1)]

    # Global aggregates per period
    rows = []
    for t_str, yr in zip(t_strs, years):
        row = {"year": yr, "t": int(t_str)}
        for vname in ["Y", "YGROSS", "YNET", "C"]:
            if vname in v and v[vname].records is not None:
                vdf = v[vname].records
                vals = vdf[vdf["t"] == t_str]["level"]
                row[f"{vname}_world"] = vals.sum()
        # E and EIND: filter CO2 only (ghg dimension present)
        for vname in ["E", "EIND"]:
            if vname in v and v[vname].records is not None:
                vdf = v[vname].records
                if "ghg" in vdf.columns:
                    vals = vdf[(vdf["t"] == t_str) & (vdf["ghg"] == "co2")]["level"]
                else:
                    vals = vdf[vdf["t"] == t_str]["level"]
                row[f"{vname}_world"] = vals.sum()
        # TATM (scalar per t)
        if "TATM" in v and v["TATM"].records is not None:
            tatm_df = v["TATM"].records
            t_row = tatm_df[tatm_df["t"] == t_str]
            row["TATM"] = t_row["level"].iloc[0] if len(t_row) > 0 else None
        # MIU average
        if "MIU" in v and v["MIU"].records is not None:
            miu_df = v["MIU"].records
            row["MIU_avg"] = miu_df[miu_df["t"] == t_str]["level"].mean()
        # ABATECOST total
        if "ABATECOST" in v and v["ABATECOST"].records is not None:
            ac_df = v["ABATECOST"].records
            row["ABATECOST_world"] = ac_df[ac_df["t"] == t_str]["level"].sum()
        # DAMAGES total
        if "DAMAGES" in v and v["DAMAGES"].records is not None:
            dm_df = v["DAMAGES"].records
            row["DAMAGES_world"] = dm_df[dm_df["t"] == t_str]["level"].sum()
        rows.append(row)

    results["global"] = pd.DataFrame(rows)

    # UTILITY
    if "UTILITY" in v and v["UTILITY"].records is not None:
        util = v["UTILITY"].records["level"].iloc[0]
    else:
        util = None
    results["utility"] = util

    return results


def run_one(policy, policy_kw, impact, macc, quiet=True):
    """Run a single scenario. Returns (name, results_dict, elapsed, mem_mb, status)."""
    name = scenario_name(policy, policy_kw, impact, macc)

    cfg = Config(policy=policy, impact=impact, macc_costs=macc, **policy_kw)

    mem_before = get_mem_mb()
    t0 = time.time()

    try:
        # Suppress stdout during build/solve
        if quiet:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                m_cont, rice, v, data = build_model(cfg)
                solve_model(rice, cfg)
        else:
            m_cont, rice, v, data = build_model(cfg)
            solve_model(rice, cfg)

        elapsed = time.time() - t0
        mem_after = get_mem_mb()
        status = str(rice.status)
        results = extract_results(m_cont, v, cfg)

    except Exception as e:
        elapsed = time.time() - t0
        mem_after = get_mem_mb()
        status = f"ERROR: {e}"
        results = None
        traceback.print_exc()

    return name, results, elapsed, mem_after - mem_before, mem_after, status


def main():
    summary_rows = []
    all_global = {}

    total = len(POLICIES) * len(IMPACTS) * len(MACC_COSTS)
    i = 0

    for policy, policy_kw in POLICIES:
        for impact in IMPACTS:
            for macc in MACC_COSTS:
                i += 1
                name = scenario_name(policy, policy_kw, impact, macc)
                print(f"[{i:2d}/{total}] {name} ...", end=" ", flush=True)

                name, results, elapsed, mem_delta, mem_total, status = run_one(
                    policy, policy_kw, impact, macc)

                print(f"{elapsed:.1f}s  mem={mem_total:.0f}MB  {status}")

                summary_rows.append({
                    "scenario": name,
                    "policy": policy,
                    "impact": impact,
                    "macc_costs": macc,
                    "elapsed_s": round(elapsed, 1),
                    "mem_total_mb": round(mem_total, 0),
                    "status": status,
                    "utility": results["utility"] if results else None,
                })

                if results and results["global"] is not None:
                    gdf = results["global"].copy()
                    gdf["scenario"] = name
                    all_global[name] = gdf

                    # Save individual CSV
                    gdf.to_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), index=False)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(f"\nSummary saved: {os.path.join(RESULTS_DIR, 'summary.csv')}")

    # Save combined global results
    if all_global:
        combined = pd.concat(all_global.values(), ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "all_scenarios.csv"), index=False)
        print(f"Combined saved: {os.path.join(RESULTS_DIR, 'all_scenarios.csv')}")

    # Print summary table
    print(f"\n{'Scenario':<45} {'Time':>6} {'Mem':>7} {'Status':<30} {'Utility':>12}")
    print("-" * 105)
    for r in summary_rows:
        print(f"{r['scenario']:<45} {r['elapsed_s']:>5.1f}s {r['mem_total_mb']:>6.0f}MB "
              f"{r['status']:<30} {r['utility'] if r['utility'] else 'N/A':>12}")


if __name__ == "__main__":
    main()
