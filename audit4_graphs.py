"""
Audit 4: Comprehensive validation graphs across scenarios.
Runs 6 scenarios, extracts key variables, and generates detailed plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pydice32.config import Config
from pydice32.solver import build_model, solve_model
from pydice32.report import print_results


# ── Scenario definitions ──────────────────────────────────────────────
SCENARIOS = [
    ("BAU",                dict(policy="bau")),
    ("BAU+DICE dmg",       dict(policy="bau_impact", impact="dice")),
    ("BAU+Kalkuhl dmg",    dict(policy="bau_impact", impact="kalkuhl")),
    ("CBA-DICE",           dict(policy="cba", impact="dice")),
    ("CBA-Kalkuhl",        dict(policy="cba", impact="kalkuhl")),
    ("CEA 2.0°C",          dict(policy="cea_tatm", tatm_limit=2.0)),
    ("CTax $50/3%",        dict(policy="ctax", ctax_initial=50, ctax_slope=0.03)),
    ("NetZero 2050",       dict(policy="global_netzero", nz_year=2050)),
]


def extract_var_trajectory(var, t_range, region_names, aggregate="sum"):
    """Extract world-aggregate trajectory of a variable."""
    recs = var.records
    if recs is None or len(recs) == 0:
        return {t: 0.0 for t in t_range}
    result = {}
    for t in t_range:
        ts = str(t)
        if recs.shape[1] > 3:  # (t, n, ..., level)
            mask = recs.iloc[:, 0] == ts
            vals = recs.loc[mask, "level"]
        else:
            mask = recs.iloc[:, 0] == ts
            vals = recs.loc[mask, "level"]
        if aggregate == "sum":
            result[t] = vals.sum()
        elif aggregate == "mean":
            result[t] = vals.mean() if len(vals) > 0 else 0.0
        elif aggregate == "first":
            result[t] = float(vals.iloc[0]) if len(vals) > 0 else 0.0
    return result


def extract_scalar_trajectory(var, t_range):
    """Extract trajectory of a scalar-indexed-by-t variable (like TATM)."""
    recs = var.records
    if recs is None or len(recs) == 0:
        return {t: 0.0 for t in t_range}
    result = {}
    for t in t_range:
        ts = str(t)
        mask = recs.iloc[:, 0] == ts
        if mask.any():
            result[t] = float(recs.loc[mask, "level"].iloc[0])
        else:
            result[t] = 0.0
    return result


def extract_regional_var(var, t_idx, region_names, ghg_filter=None):
    """Extract per-region values at a specific time period."""
    recs = var.records
    if recs is None or len(recs) == 0:
        return {r: 0.0 for r in region_names}
    ts = str(t_idx)
    result = {}
    for _, row in recs.iterrows():
        if str(row.iloc[0]) != ts:
            continue
        if ghg_filter is not None and recs.shape[1] > 3:
            if str(row.iloc[2]) != ghg_filter:
                continue
        n = str(row.iloc[1])
        result[n] = float(row["level"])
    return result


def run_all_scenarios():
    """Run all scenarios and collect results."""
    all_results = {}

    for name, kwargs in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        cfg = Config(**kwargs)
        cfg._apply_policy_defaults()
        cfg._validate()

        m, rice, v, data = build_model(cfg)
        solve_model(rice, cfg)

        T = cfg.T
        TSTEP = cfg.TSTEP
        # t range covering 2020-2100 (periods 2-18)
        t_range = list(range(2, 19))  # 2020 to 2100
        years = [2015 + (t - 1) * TSTEP for t in t_range]

        results = {
            "years": years,
            "t_range": t_range,
            "cfg": cfg,
        }

        # Key variables
        region_names = data["region_names"]

        # TATM (scalar over t)
        results["TATM"] = extract_scalar_trajectory(v["TATM"], t_range)

        # GDP (Y), sum over regions
        results["Y"] = extract_var_trajectory(v["Y"], t_range, region_names)

        # CO2 emissions - need to filter ghg=co2
        E = v["E"]
        e_dict = {}
        if E.records is not None:
            for t in t_range:
                ts = str(t)
                mask = (E.records.iloc[:, 0] == ts) & (E.records.iloc[:, 2] == "co2")
                e_dict[t] = E.records.loc[mask, "level"].sum()
        results["E_CO2"] = e_dict

        # MIU (CO2, average across regions)
        MIU = v["MIU"]
        miu_dict = {}
        if MIU.records is not None:
            for t in t_range:
                ts = str(t)
                mask = (MIU.records.iloc[:, 0] == ts) & (MIU.records.iloc[:, 2] == "co2")
                vals = MIU.records.loc[mask, "level"]
                miu_dict[t] = vals.mean() if len(vals) > 0 else 0.0
        results["MIU_avg"] = miu_dict

        # ABATECOST (sum across regions and ghg)
        results["ABATECOST"] = extract_var_trajectory(v["ABATECOST"], t_range, region_names)

        # Consumption
        results["C"] = extract_var_trajectory(v["C"], t_range, region_names)

        # CPC (mean across regions)
        CPC = v.get("CPC")
        if CPC is not None:
            cpc_dict = {}
            if CPC.records is not None:
                for t in t_range:
                    ts = str(t)
                    mask = CPC.records.iloc[:, 0] == ts
                    vals = CPC.records.loc[mask, "level"]
                    cpc_dict[t] = vals.mean() if len(vals) > 0 else 0.0
            results["CPC_mean"] = cpc_dict
        else:
            results["CPC_mean"] = {t: 0.0 for t in t_range}

        # Capital
        results["K"] = extract_var_trajectory(v["K"], t_range, region_names)

        # Savings rate (mean)
        S = v["S"]
        s_dict = {}
        if S.records is not None:
            for t in t_range:
                ts = str(t)
                mask = S.records.iloc[:, 0] == ts
                vals = S.records.loc[mask, "level"]
                s_dict[t] = vals.mean() if len(vals) > 0 else 0.0
        results["S_avg"] = s_dict

        # DAMAGES
        DMG = v.get("DAMAGES")
        if DMG is not None and DMG.records is not None and len(DMG.records) > 0:
            results["DAMAGES"] = extract_var_trajectory(DMG, t_range, region_names)
        else:
            results["DAMAGES"] = {t: 0.0 for t in t_range}

        # DAMFRAC (mean)
        DF = v.get("DAMFRAC")
        if DF is not None and DF.records is not None and len(DF.records) > 0:
            df_dict = {}
            for t in t_range:
                ts = str(t)
                mask = DF.records.iloc[:, 0] == ts
                vals = DF.records.loc[mask, "level"]
                df_dict[t] = vals.mean() if len(vals) > 0 else 0.0
            results["DAMFRAC_avg"] = df_dict
        else:
            results["DAMFRAC_avg"] = {t: 0.0 for t in t_range}

        # MAC (CO2, mean across regions)
        MAC = v.get("MAC")
        if MAC is not None and MAC.records is not None:
            mac_dict = {}
            for t in t_range:
                ts = str(t)
                mask = (MAC.records.iloc[:, 0] == ts) & (MAC.records.iloc[:, 2] == "co2")
                vals = MAC.records.loc[mask, "level"]
                mac_dict[t] = vals.mean() if len(vals) > 0 else 0.0
            results["MAC_co2_avg"] = mac_dict
        else:
            results["MAC_co2_avg"] = {t: 0.0 for t in t_range}

        # TOCEAN
        TOC = v.get("TOCEAN")
        if TOC is not None:
            results["TOCEAN"] = extract_scalar_trajectory(TOC, t_range)
        else:
            results["TOCEAN"] = {t: 0.0 for t in t_range}

        # FORC
        FORC = v.get("FORC")
        if FORC is not None:
            results["FORC"] = extract_scalar_trajectory(FORC, t_range)
        else:
            results["FORC"] = {t: 0.0 for t in t_range}

        # Regional breakdown at 2050 (period 8) for CO2 emissions
        results["E_CO2_regional_2050"] = extract_regional_var(
            E, 8, region_names, ghg_filter="co2")

        # Regional GDP at 2050
        results["Y_regional_2050"] = extract_regional_var(
            v["Y"], 8, region_names)

        all_results[name] = results
        print(f"  Done: TATM(2100)={results['TATM'].get(18, 'N/A'):.2f}°C")

    return all_results


def plot_all(all_results, output_dir):
    """Generate all validation graphs."""
    os.makedirs(output_dir, exist_ok=True)

    scenario_names = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_names)))
    color_map = dict(zip(scenario_names, colors))

    # Helper: get years and values
    def get_xy(results, var_key):
        years = results["years"]
        vals = results[var_key]
        t_range = results["t_range"]
        return years, [vals.get(t, 0.0) for t in t_range]

    # ── 1. Temperature pathway ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "TATM")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("°C above pre-industrial")
    ax.set_title("Global Mean Temperature (TATM)")
    ax.set_xlim(2020, 2100)
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="2°C target")
    ax.axhline(y=1.5, color="orange", linestyle="--", alpha=0.5, label="1.5°C target")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_temperature.png"), dpi=150)
    plt.close(fig)

    # ── 2. CO2 emissions ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "E_CO2")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("GtCO2/yr")
    ax.set_title("Global CO2 Emissions")
    ax.set_xlim(2020, 2100)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_co2_emissions.png"), dpi=150)
    plt.close(fig)

    # ── 3. GDP (Y) ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "Y")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion $ (PPP)")
    ax.set_title("Global GDP (Y)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_gdp.png"), dpi=150)
    plt.close(fig)

    # ── 4. Abatement rate (MIU CO2 average) ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "MIU_avg")
        ax.plot(x, [v * 100 for v in y], label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 abatement rate (%)")
    ax.set_title("Average CO2 Abatement Rate (MIU)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_miu_co2.png"), dpi=150)
    plt.close(fig)

    # ── 5. Abatement cost ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "ABATECOST")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion $")
    ax.set_title("Global Abatement Cost")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_abatecost.png"), dpi=150)
    plt.close(fig)

    # ── 6. Carbon price (MAC CO2) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        if name == "BAU":
            continue
        x, y = get_xy(all_results[name], "MAC_co2_avg")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("MAC (model units)")
    ax.set_title("Average Carbon Price (MAC CO2)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_carbon_price.png"), dpi=150)
    plt.close(fig)

    # ── 7. Consumption ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "C")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion $")
    ax.set_title("Global Consumption")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_consumption.png"), dpi=150)
    plt.close(fig)

    # ── 8. Damages ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "DAMAGES")
        if max(y) > 0.001:
            ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion $")
    ax.set_title("Global Climate Damages")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_damages.png"), dpi=150)
    plt.close(fig)

    # ── 9. Damage fraction ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "DAMFRAC_avg")
        if max(abs(v) for v in y) > 1e-6:
            ax.plot(x, [v * 100 for v in y], label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Damage fraction (%)")
    ax.set_title("Average Damage Fraction (DAMFRAC)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_damfrac.png"), dpi=150)
    plt.close(fig)

    # ── 10. Savings rate ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "S_avg")
        ax.plot(x, [v * 100 for v in y], label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Savings rate (%)")
    ax.set_title("Average Savings Rate")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_savings_rate.png"), dpi=150)
    plt.close(fig)

    # ── 11. Radiative Forcing ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "FORC")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("W/m²")
    ax.set_title("Radiative Forcing (FORC)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "11_forcing.png"), dpi=150)
    plt.close(fig)

    # ── 12. Capital stock ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "K")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion $")
    ax.set_title("Global Capital Stock (K)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "12_capital.png"), dpi=150)
    plt.close(fig)

    # ── 13. Regional CO2 emissions at 2050 (bar chart) ────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    # Select top 10 emitting regions in BAU
    bau_e = all_results["BAU"]["E_CO2_regional_2050"]
    top_regions = sorted(bau_e.keys(), key=lambda r: bau_e.get(r, 0), reverse=True)[:10]

    x_pos = np.arange(len(top_regions))
    width = 0.12
    for i, name in enumerate(scenario_names):
        e_reg = all_results[name]["E_CO2_regional_2050"]
        vals = [e_reg.get(r, 0) for r in top_regions]
        ax.bar(x_pos + i * width, vals, width, label=name, color=color_map[name])
    ax.set_xlabel("Region")
    ax.set_ylabel("GtCO2/yr")
    ax.set_title("Regional CO2 Emissions at 2050 (Top 10 BAU emitters)")
    ax.set_xticks(x_pos + width * len(scenario_names) / 2)
    ax.set_xticklabels(top_regions, rotation=45, ha="right", fontsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "13_regional_co2_2050.png"), dpi=150)
    plt.close(fig)

    # ── 14. Regional GDP at 2050 (bar chart) ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    bau_y = all_results["BAU"]["Y_regional_2050"]
    top_gdp = sorted(bau_y.keys(), key=lambda r: bau_y.get(r, 0), reverse=True)[:10]

    x_pos = np.arange(len(top_gdp))
    for i, name in enumerate(scenario_names):
        y_reg = all_results[name]["Y_regional_2050"]
        vals = [y_reg.get(r, 0) for r in top_gdp]
        ax.bar(x_pos + i * width, vals, width, label=name, color=color_map[name])
    ax.set_xlabel("Region")
    ax.set_ylabel("Trillion $")
    ax.set_title("Regional GDP at 2050 (Top 10)")
    ax.set_xticks(x_pos + width * len(scenario_names) / 2)
    ax.set_xticklabels(top_gdp, rotation=45, ha="right", fontsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "14_regional_gdp_2050.png"), dpi=150)
    plt.close(fig)

    # ── 15. GDP loss vs BAU ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    bau_y_traj = all_results["BAU"]["Y"]
    bau_t_range = all_results["BAU"]["t_range"]
    for name in scenario_names:
        if name == "BAU":
            continue
        x = all_results[name]["years"]
        y_traj = all_results[name]["Y"]
        t_range = all_results[name]["t_range"]
        loss = [(bau_y_traj.get(t, 1) - y_traj.get(t, 0)) / max(bau_y_traj.get(t, 1), 0.001) * 100
                for t in t_range]
        ax.plot(x, loss, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP loss vs BAU (%)")
    ax.set_title("GDP Loss Relative to BAU")
    ax.set_xlim(2020, 2100)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "15_gdp_loss_vs_bau.png"), dpi=150)
    plt.close(fig)

    # ── 16. Ocean temperature ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "TOCEAN")
        ax.plot(x, y, label=name, color=color_map[name], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("°C above pre-industrial")
    ax.set_title("Ocean Temperature (TOCEAN)")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "16_ocean_temp.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll 16 graphs saved to {output_dir}/")

    # ── README spaghetti emission plot ────────────────────────────────
    docs_dir = os.path.join(os.path.dirname(output_dir), "docs")
    os.makedirs(docs_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    for name in scenario_names:
        x, y = get_xy(all_results[name], "E_CO2")
        lw = 2.5 if name in ("BAU", "CBA-Kalkuhl", "NetZero 2050") else 1.8
        ls = "--" if "BAU+" in name else "-"
        ax.plot(x, y, label=name, color=color_map[name], linewidth=lw, linestyle=ls)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("GtCO2/yr", fontsize=13)
    ax.set_title("Global CO2 Emission Pathways", fontsize=15, fontweight="bold")
    ax.set_xlim(2020, 2100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10,
              frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(docs_dir, "spaghetti_emissions.png"), dpi=150)
    plt.close(fig)
    print(f"README plot saved to {docs_dir}/spaghetti_emissions.png")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "audit4_output")
    all_results = run_all_scenarios()
    plot_all(all_results, output_dir)
    print("\nDone!")
