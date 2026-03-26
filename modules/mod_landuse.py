"""
Land-use emissions and abatement module.

Variables: ELAND, MIULAND, MACLAND, ABCOSTLAND
Equations: eq_eland, eq_mcland, eq_abcostland
"""

from gamspy import Variable, Equation


def declare_vars(m, sets, params, cfg, v):
    """Create land-use variables, set bounds/starting values/fixed values.

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

    par_eland_bau = params["par_eland_bau"]

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    ELAND = Variable(m, name="ELAND", domain=[t_set, n_set])
    MIULAND = Variable(m, name="MIULAND", domain=[t_set, n_set], type="positive")
    MACLAND = Variable(m, name="MACLAND", domain=[t_set, n_set], type="positive")
    ABCOSTLAND = Variable(m, name="ABCOSTLAND", domain=[t_set, n_set], type="positive")

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    MIULAND.up[t_set, n_set] = 1.0

    # ------------------------------------------------------------------
    # Starting values
    # ------------------------------------------------------------------
    ELAND.l[t_set, n_set] = par_eland_bau[t_set, n_set]
    MIULAND.l[t_set, n_set] = 0
    MACLAND.l[t_set, n_set] = 0
    ABCOSTLAND.l[t_set, n_set] = 0

    # ------------------------------------------------------------------
    # Fixed initial conditions
    # ------------------------------------------------------------------
    MIULAND.fx["1", n_set] = 0
    MIULAND.fx["2", n_set] = 0

    # Register in shared variable dict
    v["ELAND"] = ELAND
    v["MIULAND"] = MIULAND
    v["MACLAND"] = MACLAND
    v["ABCOSTLAND"] = ABCOSTLAND


def define_eqs(m, sets, params, cfg, v):
    """Create land-use equations.

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

    par_eland_bau = params["par_eland_bau"]
    par_eland_maxab = params["par_eland_maxab"]
    par_lu_macc_c1 = params["par_lu_macc_c1"]
    par_lu_macc_c4 = params["par_lu_macc_c4"]

    convy_co2 = 1e-3  # cost conversion for CO2

    # Own variables
    ELAND = v["ELAND"]
    MIULAND = v["MIULAND"]
    MACLAND = v["MACLAND"]
    ABCOSTLAND = v["ABCOSTLAND"]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    # eq_eland: ELAND = eland_bau - eland_maxab * MIULAND
    eq_eland = Equation(m, name="eq_eland", domain=[t_set, n_set])
    eq_eland[t_set, n_set] = (
        ELAND[t_set, n_set] == par_eland_bau[t_set, n_set]
        - par_eland_maxab[n_set] * MIULAND[t_set, n_set]
    )

    # eq_mcland: MACLAND = lu_c1*MIULAND + lu_c4*MIULAND^4
    eq_mcland = Equation(m, name="eq_mcland", domain=[t_set, n_set])
    eq_mcland[t_set, n_set] = (
        MACLAND[t_set, n_set] == par_lu_macc_c1[t_set, n_set] * MIULAND[t_set, n_set]
        + par_lu_macc_c4[t_set, n_set] * MIULAND[t_set, n_set] ** 4
    )

    # eq_abcostland: ABCOSTLAND = convy * eland_maxab * (lu_c1*MIULAND^2/2 + lu_c4*MIULAND^5/5)
    eq_abcostland = Equation(m, name="eq_abcostland", domain=[t_set, n_set])
    eq_abcostland[t_set, n_set] = ABCOSTLAND[t_set, n_set] == (
        convy_co2 * par_eland_maxab[n_set] * (
            par_lu_macc_c1[t_set, n_set] * MIULAND[t_set, n_set] ** 2 / 2
            + par_lu_macc_c4[t_set, n_set] * MIULAND[t_set, n_set] ** 5 / 5
        )
    )

    return [eq_eland, eq_mcland, eq_abcostland]
