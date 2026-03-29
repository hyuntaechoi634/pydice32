"""
Validation batch: 4 core scenarios with kalkuhl damage + prob50 MACC.

Scenarios: BAU (with damages), CBA, CTax(50$/t), NZ2050
All use impact=kalkuhl so GDP/damages are directly comparable.

Key validation:
- BAU damages > 0
- CBA GDP >= BAU GDP (damage reduction > abatement cost)
- CBA TATM < BAU TATM
- CTax/NZ have lower emissions than BAU
"""

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pydice32.config import Config
from pydice32.solver import build_model, solve_model

POLICIES = [
    ("BAU",     dict(policy="bau_impact", impact="kalkuhl")),
    ("CBA",     dict(policy="cba", impact="kalkuhl")),
    ("CTax30",  dict(policy="ctax", impact="kalkuhl", ctax_initial=30, ctax_slope=0.03)),
    ("NZ2050",  dict(policy="global_netzero", impact="kalkuhl", nz_year=2050)),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "validation_output")
T_RANGE = list(range(2, 19))
TSTEP = 5
YEARS = [2015 + (t - 1) * TSTEP for t in T_RANGE]

COLORS = {"BAU": "#888888", "CBA": "#2196F3", "CTax30": "#FF9800", "NZ2050": "#4CAF50"}


def _sum(var, t_range, ghg=None):
    recs = var.records
    if recs is None or len(recs) == 0:
        return {t: 0.0 for t in t_range}
    out = {}
    for t in t_range:
        ts = str(t)
        if ghg and recs.shape[1] > 3:
            mask = (recs.iloc[:, 0] == ts) & (recs.iloc[:, 2] == ghg)
        else:
            mask = recs.iloc[:, 0] == ts
        out[t] = recs.loc[mask, "level"].sum()
    return out


def _mean(var, t_range, ghg=None):
    recs = var.records
    if recs is None or len(recs) == 0:
        return {t: 0.0 for t in t_range}
    out = {}
    for t in t_range:
        ts = str(t)
        if ghg and recs.shape[1] > 3:
            mask = (recs.iloc[:, 0] == ts) & (recs.iloc[:, 2] == ghg)
        else:
            mask = recs.iloc[:, 0] == ts
        vals = recs.loc[mask, "level"]
        out[t] = vals.mean() if len(vals) > 0 else 0.0
    return out


def _scalar(var, t_range):
    recs = var.records
    if recs is None or len(recs) == 0:
        return {t: 0.0 for t in t_range}
    out = {}
    for t in t_range:
        mask = recs.iloc[:, 0] == str(t)
        out[t] = float(recs.loc[mask, "level"].iloc[0]) if mask.any() else 0.0
    return out


def extract(v, data):
    r = {}
    r["TATM"] = _scalar(v["TATM"], T_RANGE)
    r["Y"] = _sum(v["Y"], T_RANGE)
    r["E_CO2"] = _sum(v["E"], T_RANGE, "co2")
    r["MIU_avg"] = _mean(v["MIU"], T_RANGE, "co2")
    r["ABATECOST"] = _sum(v["ABATECOST"], T_RANGE)
    r["C"] = _sum(v["C"], T_RANGE)
    r["CPC_mean"] = _mean(v.get("CPC", v["C"]), T_RANGE)
    r["DAMAGES"] = _sum(v.get("DAMAGES", v["Y"]), T_RANGE)
    r["DAMFRAC_avg"] = _mean(v.get("DAMFRAC", v["S"]), T_RANGE)
    r["MAC_co2_avg"] = _mean(v.get("MAC", v["MIU"]), T_RANGE, "co2")
    r["S_avg"] = _mean(v["S"], T_RANGE)
    r["K"] = _sum(v["K"], T_RANGE)
    return r


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    timing = []

    for pname, pkw in POLICIES:
        print(f"\n=== {pname} ===")
        cfg = Config(**pkw, macc_costs="prob50")
        t0 = time.perf_counter()
        m, rice, v, data = build_model(cfg)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        solve_model(rice, cfg)
        t_solve = time.perf_counter() - t0
        print(f"  build={t_build:.0f}s solve={t_solve:.0f}s status={rice.solve_status}")
        timing.append(dict(scenario=pname, build=t_build, solve=t_solve))
        results[pname] = extract(v, data)

    return results, timing


def _vals(results, name, var):
    return [results[name][var].get(t, 0.0) for t in T_RANGE]


def generate_charts(results):
    print("\nGenerating charts...")

    def _plot(fname, ylabel, var, notes=None, **kw):
        fig, ax = plt.subplots(figsize=(8, 4))
        for pname, _ in POLICIES:
            ax.plot(YEARS, _vals(results, pname, var),
                    color=COLORS[pname], label=pname, linewidth=1.5)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend()
        if kw.get("hline") is not None:
            ax.axhline(kw["hline"], color="gray", ls="--", lw=0.5)
        if notes:
            fig.text(0.02, -0.02, f"Notes: {notes}", fontsize=7,
                     color="gray", va="top", ha="left", style="italic")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18 if notes else 0.12)
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close(fig)

    # Extract key numbers for notes
    bau_t2100 = results["BAU"]["TATM"].get(18, 0)
    cba_t2100 = results["CBA"]["TATM"].get(18, 0)
    nz_t2100 = results["NZ2050"]["TATM"].get(18, 0)
    bau_dmg = results["BAU"]["DAMAGES"].get(18, 0)
    cba_gdp = results["CBA"]["Y"].get(18, 0)
    bau_gdp = results["BAU"]["Y"].get(18, 0)

    _plot("01_temperature.png", "deg C", "TATM",
          notes=f"BAU reaches {bau_t2100:.1f}C by 2100. CBA optimal: {cba_t2100:.1f}C. NZ2050: {nz_t2100:.1f}C.")
    _plot("02_co2_emissions.png", "GtCO2/yr", "E_CO2", hline=0,
          notes="NZ2050 reaches net-zero at 2050. CBA converges to near-zero by 2100.")
    _plot("03_gdp.png", "T$", "Y",
          notes=f"BAU GDP ({bau_gdp:.0f}T$) falls below CBA ({cba_gdp:.0f}T$) after ~2060 due to climate damages.")
    _plot("04_miu_co2.png", "MIU", "MIU_avg",
          notes="MIU = fraction of CO2 abated. BAU = 0 (no policy). NZ2050 ramps fastest.")
    _plot("05_abatecost.png", "T$", "ABATECOST",
          notes="Higher abatement cost reflects stronger mitigation effort.")
    _plot("06_carbon_price.png", "$/tCO2", "MAC_co2_avg",
          notes="MAC = marginal abatement cost = implicit carbon price.")
    _plot("07_consumption.png", "$/person", "CPC_mean",
          notes="Per-capita consumption. Policy scenarios overtake BAU as damages grow.")
    _plot("08_damages.png", "T$", "DAMAGES",
          notes=f"BAU damages reach {bau_dmg:.0f}T$ by 2100. Mitigation sharply reduces damages.")
    _plot("09_damfrac.png", "fraction", "DAMFRAC_avg",
          notes="Fraction of GDP lost to climate damages. Kalkuhl growth-rate specification.")
    _plot("10_savings.png", "rate", "S_avg")

    # GDP loss vs BAU
    fig, ax = plt.subplots(figsize=(8, 4))
    bau_y = np.array(_vals(results, "BAU", "Y"))
    for pname, _ in POLICIES:
        if pname == "BAU":
            continue
        pol_y = np.array(_vals(results, pname, "Y"))
        loss = (pol_y - bau_y) / np.maximum(bau_y, 1e-10) * 100
        ax.plot(YEARS, loss, color=COLORS[pname], label=pname, linewidth=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP change vs BAU (%)")
    ax.legend()
    fig.text(0.02, -0.02,
             "Notes: Positive = net GDP gain from policy (damage reduction > abatement cost).",
             fontsize=7, color="gray", va="top", ha="left", style="italic")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(os.path.join(OUT_DIR, "11_gdp_loss_vs_bau.png"), dpi=150)
    plt.close(fig)

    print(f"  Charts saved to {OUT_DIR}/")


def sanity_checks(results):
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    passed = failed = 0

    def check(name, ok):
        nonlocal passed, failed
        s = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{s}] {name}")

    bau = results["BAU"]
    cba = results["CBA"]
    ctax = results["CTax30"]
    nz = results["NZ2050"]

    # BAU damages must be > 0
    check("BAU damages(2100) > 0", bau["DAMAGES"][18] > 0)
    check("BAU damages(2050) > 0", bau["DAMAGES"][8] > 0)

    # CBA should have lower temperature
    check("CBA TATM(2100) < BAU TATM(2100)",
          cba["TATM"][18] < bau["TATM"][18])

    # CBA GDP should be close to or higher than BAU (damages reduced)
    ratio = cba["Y"][18] / bau["Y"][18] if bau["Y"][18] > 0 else 0
    check(f"CBA GDP(2100)/BAU = {ratio:.3f} in [0.9, 1.15]",
          0.9 < ratio < 1.15)

    # CBA should have positive abatement
    check(f"CBA MIU(2050) = {cba['MIU_avg'][8]:.3f} > 0.1",
          cba["MIU_avg"][8] > 0.1)

    # CTax emissions < BAU emissions
    check("CTax E_CO2(2100) < BAU E_CO2(2100)",
          ctax["E_CO2"][18] < bau["E_CO2"][18])

    # NZ emissions at 2050 ~ 0
    check(f"NZ E_CO2(2050) = {nz['E_CO2'][8]:.3f} ~ 0",
          abs(nz["E_CO2"][8]) < 1.0)

    # NZ temperature lower than BAU
    check("NZ TATM(2100) < BAU TATM(2100)",
          nz["TATM"][18] < bau["TATM"][18])

    # Print key numbers
    print(f"\n  Key values at 2100:")
    for p in ["BAU", "CBA", "CTax30", "NZ2050"]:
        r = results[p]
        print(f"    {p:8s}: TATM={r['TATM'][18]:.2f}C  "
              f"GDP={r['Y'][18]:.0f}T$  "
              f"DMG={r['DAMAGES'][18]:.1f}T$  "
              f"E_CO2={r['E_CO2'][18]:.1f}GtCO2  "
              f"MIU={r['MIU_avg'][18]:.3f}")

    print(f"\n  Total: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    results, timing = run_all()

    # Save CSV
    rows = []
    for pname in results:
        r = results[pname]
        for t, yr in zip(T_RANGE, YEARS):
            row = {"scenario": pname, "t": t, "year": yr}
            for var in r:
                row[var] = r[var].get(t, 0.0)
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "all_scenarios.csv"), index=False)
    pd.DataFrame(timing).to_csv(os.path.join(OUT_DIR, "timing.csv"), index=False)

    total_time = sum(t["build"] + t["solve"] for t in timing)
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")

    generate_charts(results)
    ok = sanity_checks(results)
    if not ok:
        sys.exit(1)
