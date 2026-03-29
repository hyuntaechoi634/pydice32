"""
COACCH impact module: regional level damages with quantile uncertainty.

Based on GAMS mod_impact_coacch.gms (van der Wijst et al., 2023).

OMEGA = qmul * (b1*(TATM-Tbase) + b2*(TATM-Tbase)^2 + c)
        - qmul * (b1*(TATM('2')-Tbase) + b2*(TATM('2')-Tbase)^2 + c)

Optional SLR addon (when cfg.slr=True):
        + qmul_slr * (slr_b1*GMSLR + slr_b2*GMSLR^2)
        - qmul_slr * (slr_b1*GMSLR('2') + slr_b2*GMSLR('2')^2)

Data from data_mod_damage/:
    comega.csv       -- (damcost, n, {b1,b2,c}) -> val
    comega_qmul.csv  -- (damcost, n, percentile) -> multiplier
    temp_base.csv    -- (damcost) -> base temperature offset

Config options:
    cfg.damcost    -- damage cost set: 'COACCH_NoSLR' (default)
    cfg.damcostpb  -- percentile: 'p50' (default)
    cfg.damcostslr -- SLR damage set: 'none' / 'COACCH_SLR_Ad' / 'COACCH_SLR_NoAd'
"""

import os
import pandas as pd
from gamspy import Variable, Equation, Ord, Parameter
from pydice32.data.gcam_mapping import load_rice_regions, load_gcam_mapping, load_gcam_region_names


def declare_vars(m, sets, params, cfg, v):
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    BIMPACT = Variable(m, name="BIMPACT", domain=[t_set, n_set])
    BIMPACT.l[t_set, n_set] = 0
    BIMPACT.lo[t_set, n_set] = -1 + 1e-6
    BIMPACT.fx["1", n_set] = 0
    BIMPACT.fx["2", n_set] = 0
    v["BIMPACT"] = BIMPACT


def define_eqs(m, sets, params, cfg, v):
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    BIMPACT = v["BIMPACT"]
    TATM = v["TATM"]
    OMEGA = v["OMEGA"]

    region_names = [r for r in params["par_pop"].records["n"].unique()]

    # Config
    damcost = getattr(cfg, "damcost", "COACCH_NoSLR")
    damcostpb = getattr(cfg, "damcostpb", "p50")
    damcostslr = getattr(cfg, "damcostslr", "none")

    # Load ISO -> GCAM mapping for aggregation
    try:
        rice_regions = load_rice_regions(cfg.project_root)
        gcam_map = load_gcam_mapping(cfg.gcam_csv, set(rice_regions))
        gcam_nm = load_gcam_region_names(cfg.gcam_names_csv)
    except Exception:
        gcam_map, gcam_nm = {}, {}

    dam_dir = os.path.join(cfg.data_dir, "data_mod_damage")

    # --- Load comega (b1, b2, c) per region ---
    # Aggregate from ISO3 to GCAM regions (GDP-weighted)
    comega_iso = {}  # (iso, coef) -> val
    comega_file = os.path.join(dam_dir, "comega.csv")
    if os.path.exists(comega_file):
        df = pd.read_csv(comega_file)
        for _, row in df.iterrows():
            dc = str(row.iloc[0])
            if dc != damcost:
                continue
            iso = str(row["n"]).lower()
            coef = str(row.iloc[2]).lower()
            comega_iso[(iso, coef)] = float(row["Val"])

    # GDP-weighted aggregation
    ykali_country = params.get("_ykali_country", {})
    comega_agg = {}  # (region, coef) -> val
    for coef in ("b1", "b2", "c"):
        num = {}
        wgt = {}
        for iso in gcam_map:
            r = gcam_nm.get(gcam_map[iso])
            if r not in region_names:
                continue
            w = ykali_country.get((1, iso), 0.0)
            val = comega_iso.get((iso, coef), 0.0)
            num[r] = num.get(r, 0.0) + val * w
            wgt[r] = wgt.get(r, 0.0) + w
        for r in region_names:
            comega_agg[(r, coef)] = num.get(r, 0.0) / wgt[r] if wgt.get(r, 0) > 0 else 0.0

    # --- Load comega_qmul ---
    qmul_iso = {}
    qmul_file = os.path.join(dam_dir, "comega_qmul.csv")
    if os.path.exists(qmul_file):
        df = pd.read_csv(qmul_file)
        for _, row in df.iterrows():
            dc = str(row.iloc[0])
            iso = str(row["n"]).lower()
            pb = str(row.iloc[2]).lower()
            if dc == damcost and pb == damcostpb:
                qmul_iso[iso] = max(float(row["Val"]), 0.0)  # GAMS clips <=0 to 0

    # Aggregate qmul (GDP-weighted)
    qmul_agg = {}
    qnum, qwgt = {}, {}
    for iso in gcam_map:
        r = gcam_nm.get(gcam_map[iso])
        if r not in region_names:
            continue
        w = ykali_country.get((1, iso), 0.0)
        val = qmul_iso.get(iso, 1.0)
        qnum[r] = qnum.get(r, 0.0) + val * w
        qwgt[r] = qwgt.get(r, 0.0) + w
    for r in region_names:
        qmul_agg[r] = qnum.get(r, 1.0) / qwgt[r] if qwgt.get(r, 0) > 0 else 1.0

    # --- Load temp_base ---
    temp_base_val = 0.6  # default
    tb_file = os.path.join(dam_dir, "temp_base.csv")
    if os.path.exists(tb_file):
        df = pd.read_csv(tb_file)
        for _, row in df.iterrows():
            if str(row.iloc[0]) == damcost:
                temp_base_val = float(row["Val"])
                break

    # Create GAMSPy parameters
    b1_recs = [(r, comega_agg.get((r, "b1"), 0.0)) for r in region_names]
    b2_recs = [(r, comega_agg.get((r, "b2"), 0.0)) for r in region_names]
    c_recs = [(r, comega_agg.get((r, "c"), 0.0)) for r in region_names]
    qmul_recs = [(r, qmul_agg.get(r, 1.0)) for r in region_names]

    par_b1 = Parameter(m, name="comega_b1", domain=[n_set], records=b1_recs)
    par_b2 = Parameter(m, name="comega_b2", domain=[n_set], records=b2_recs)
    par_c = Parameter(m, name="comega_c", domain=[n_set], records=c_recs)
    par_qmul = Parameter(m, name="comega_qmul", domain=[n_set], records=qmul_recs)

    equations = []

    # eq_bimpact: trivially 0 (level damage, not growth-rate)
    eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
    eq_bimpact[t_set, n_set] = BIMPACT[t_set, n_set] == 0
    equations.append(eq_bimpact)

    # eqomega: COACCH level damage
    # OMEGA = qmul * (b1*(TATM-Tb) + b2*(TATM-Tb)^2 + c)
    #       - qmul * (b1*(TATM('2')-Tb) + b2*(TATM('2')-Tb)^2 + c)
    eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])

    omega_rhs = (
        par_qmul[n_set] * (
            par_b1[n_set] * (TATM[t_set] - temp_base_val)
            + par_b2[n_set] * (TATM[t_set] - temp_base_val) ** 2
            + par_c[n_set]
        )
        - par_qmul[n_set] * (
            par_b1[n_set] * (TATM["2"] - temp_base_val)
            + par_b2[n_set] * (TATM["2"] - temp_base_val) ** 2
            + par_c[n_set]
        )
    )

    # SLR addon (handled by hub_impact OMEGA_SLR, not here)
    # COACCH SLR is already in hub_impact via comega_slr data

    eq_omega[t_set, n_set].where[Ord(t_set) > 1] = (
        OMEGA[t_set, n_set] == omega_rhs
    )
    equations.append(eq_omega)

    return equations
