"""
Result extraction and reporting.
Corresponds to GAMS 'report' and 'gdx_items' phases.
"""


def print_results(m, rice, cfg, v, data):
    """Print solve results."""
    region_names = data["region_names"]
    policy = cfg.policy

    UTILITY = v["UTILITY"]
    TATM = v["TATM"]
    Y = v["Y"]
    E = v["E"]
    C = v["C"]
    YGROSS = v["YGROSS"]
    YNET = v["YNET"]
    K = v["K"]
    S = v["S"]

    print(f"\n=== PyRICE32 {policy.upper()} Results ({len(region_names)} regions, CONOPT) ===")
    util_rec = UTILITY.records
    if util_rec is not None and len(util_rec) > 0:
        print(f"UTILITY: {util_rec['level'].iloc[0]:.6f}")
    else:
        print("UTILITY: N/A")

    tatm_df = TATM.records
    y_df = Y.records
    e_df = E.records
    c_df = C.records
    ygross_df = YGROSS.records
    ynet_df = YNET.records
    k_df = K.records
    s_df = S.records

    for yr_idx, yr in [(1, 2015), (8, 2050), (18, 2100)]:
        t_str = str(yr_idx)
        gdp = y_df[y_df["t"] == t_str]["level"].sum()
        # Filter to CO2 only for headline emissions (E now has ghg dimension)
        if "ghg" in e_df.columns:
            e_co2 = e_df[(e_df["t"] == t_str) & (e_df["ghg"] == "co2")]
        else:
            e_co2 = e_df[e_df["t"] == t_str]
        co2 = e_co2["level"].sum()
        tatm = tatm_df[tatm_df["t"] == t_str]["level"].iloc[0]
        print(f"  {yr}: GDP={gdp:.1f}T$  CO2={co2:.1f}Gt  TATM={tatm:.2f}°C")

    # World aggregates
    print("\nWorld aggregates:")
    for yr_idx, yr in [(1, 2015), (8, 2050), (18, 2100), (38, 2200), (58, 2300)]:
        t_str = str(yr_idx)
        world_c = c_df[c_df["t"] == t_str]["level"].sum()
        world_y = y_df[y_df["t"] == t_str]["level"].sum()
        world_ygross = ygross_df[ygross_df["t"] == t_str]["level"].sum()
        world_ynet = ynet_df[ynet_df["t"] == t_str]["level"].sum()
        world_k = k_df[k_df["t"] == t_str]["level"].sum()
        world_s = s_df[s_df["t"] == t_str]["level"].mean()
        print(f"  {yr}: C={world_c:.1f} Y={world_y:.1f} YGROSS={world_ygross:.1f} "
              f"YNET={world_ynet:.1f} K={world_k:.1f} S_avg={world_s:.4f}")

    policy_with_damages = cfg.policy_with_damages

    # SCC
    if policy_with_damages:
        _print_scc(m, v)

    # CBA diagnostics
    if policy_with_damages:
        _print_cba_diagnostics(v)

    # TATM trajectory
    print("\nFull TATM trajectory:")
    for _, row in tatm_df.iterrows():
        t_idx = int(row["t"])
        yr = 2015 + 5 * (t_idx - 1)
        print(f"  {yr}: {row['level']:.2f}°C")


def _print_scc(m, v):
    try:
        eq_e_rec = m.data["eq_e"].records
        eq_cc_rec = m.data["eq_cc"].records
        if eq_e_rec is not None and eq_cc_rec is not None:
            print("\nSocial Cost of Carbon (SCC) [$/tCO2]:")
            for yr_idx, yr in [(1, 2015), (4, 2030), (8, 2050), (18, 2100)]:
                t_str = str(yr_idx)
                e_marg = eq_e_rec[eq_e_rec["t"] == t_str]["marginal"].values
                cc_marg = eq_cc_rec[eq_cc_rec["t"] == t_str]["marginal"].values
                if len(e_marg) > 0 and len(cc_marg) > 0:
                    sum_e_m = e_marg.sum()
                    sum_cc_m = cc_marg.sum()
                    scc = -1e3 * sum_e_m / sum_cc_m if abs(sum_cc_m) > 1e-20 else 0
                    print(f"  {yr}: SCC={scc:.2f} $/tCO2")
    except Exception:
        pass


def _print_cba_diagnostics(v):
    MIU = v["MIU"]
    ABATECOST = v["ABATECOST"]
    DAMAGES = v["DAMAGES"]
    DAMFRAC = v["DAMFRAC"]
    OMEGA = v["OMEGA"]
    MAC = v["MAC"]

    miu_df = MIU.records
    abate_df = ABATECOST.records
    dam_df = DAMAGES.records
    damfrac_df = DAMFRAC.records
    omega_df = OMEGA.records
    mac_df = MAC.records

    print("\nCBA diagnostics:")
    for yr_idx, yr in [(1, 2015), (4, 2030), (8, 2050), (18, 2100)]:
        t_str = str(yr_idx)
        miu_avg = miu_df[miu_df["t"] == t_str]["level"].mean()
        miu_max = miu_df[miu_df["t"] == t_str]["level"].max()
        abate_tot = abate_df[abate_df["t"] == t_str]["level"].sum()
        dam_tot = dam_df[dam_df["t"] == t_str]["level"].sum()
        df_val = damfrac_df[damfrac_df["t"] == t_str]["level"].mean() if len(damfrac_df) > 0 else 0
        om_val = omega_df[omega_df["t"] == t_str]["level"].mean() if len(omega_df) > 0 else 0
        mac_avg = mac_df[mac_df["t"] == t_str]["level"].mean() if mac_df is not None and len(mac_df) > 0 else 0
        print(f"  {yr}: MIU_avg={miu_avg:.4f} MIU_max={miu_max:.4f} "
              f"MAC_avg={mac_avg:.2f}$/tCO2 ABATECOST={abate_tot:.3f}T$ "
              f"DAMAGES={dam_tot:.3f}T$ DAMFRAC_avg={df_val:.5f} OMEGA_avg={om_val:.5f}")


