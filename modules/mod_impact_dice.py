"""
DICE impact module: quadratic level damages (DICE-2016 style).

Equations: eq_bimpact (trivial, =0), eq_omega (DICE: a2*TATM^2 - a2*TATM(2)^2)
"""

from gamspy import Variable, Equation, Ord


def declare_vars(m, sets, params, cfg, v):
    """Create DICE impact variables, set bounds/starting values/fixed values.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    # BIMPACT not used in DICE mode but must exist for hub_impact
    BIMPACT = Variable(m, name="BIMPACT", domain=[t_set, n_set])
    BIMPACT.l[t_set, n_set] = 0
    BIMPACT.lo[t_set, n_set] = -1 + 1e-6
    BIMPACT.fx["1", n_set] = 0
    BIMPACT.fx["2", n_set] = 0

    v["BIMPACT"] = BIMPACT


def define_eqs(m, sets, params, cfg, v):
    """Create DICE impact equations.

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config
    v : dict of all variables (mutated: this module adds its own)

    Returns
    -------
    list of Equation objects
    """
    t_set = sets["t_set"]
    n_set = sets["n_set"]

    # DICE-2016R damage parameters (Nordhaus, 2018, AEJ: Economic Policy)
    # GAMS mod_impact_dice.gms lines 22-24
    a2_dice = 0.00236   # damage quadratic coefficient
    a3_dice = 2.0       # damage exponent

    # Own variables
    BIMPACT = v["BIMPACT"]

    # Cross-module variables
    TATM = v["TATM"]
    OMEGA = v["OMEGA"]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_bimpact: trivially 0 in DICE mode
    eq_bimpact = Equation(m, name="eq_bimpact", domain=[t_set, n_set])
    eq_bimpact[t_set, n_set] = BIMPACT[t_set, n_set] == 0

    # eq_omega: OMEGA = a2*TATM^a3 - a2*TATM('2')^a3
    eq_omega = Equation(m, name="eq_omega", domain=[t_set, n_set])
    eq_omega[t_set, n_set].where[Ord(t_set) > 1] = (
        OMEGA[t_set, n_set] == a2_dice * TATM[t_set] ** a3_dice
                              - a2_dice * TATM["2"] ** a3_dice
    )

    return [eq_bimpact, eq_omega]
