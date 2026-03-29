"""
Howard & Sterner (2017) impact module: meta-analysis quadratic level damages.

Based on GAMS mod_impact_howard.gms.
Uses preferred specification (4) from Table 2: a2=0.595% with 25% catastrophic
uplift, yielding a2=0.7438% (0.007438 in decimal).

OMEGA = a2 * TATM^2 - a2 * TATM('2')^2

Compared to DICE (a2=0.00236), Howard damages are ~3.2x higher.
"""

from gamspy import Variable, Equation, Ord


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

    # GAMS mod_impact_howard.gms lines 23-31:
    #   a2 = 0.595 * 1.25 / 100 = 0.0074375
    #   a3 = 2.0
    a1 = 0.0       # intercept (zero in preferred spec)
    a2 = 0.595 * 1.25 / 100  # 0.0074375
    a3 = 2.0

    BIMPACT = v["BIMPACT"]
    TATM = v["TATM"]
    OMEGA = v["OMEGA"]

    eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
    eq_bimpact[t_set, n_set] = BIMPACT[t_set, n_set] == 0

    # GAMS: OMEGA = (a1*TATM + a2*TATM^a3) - (a1*TATM('2') + a2*TATM('2')^a3)
    eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])
    eq_omega[t_set, n_set].where[Ord(t_set) > 1] = (
        OMEGA[t_set, n_set]
        == (a1 * TATM[t_set] + a2 * TATM[t_set] ** a3)
        - (a1 * TATM["2"] + a2 * TATM["2"] ** a3)
    )

    return [eq_bimpact, eq_omega]
