"""
Labour module (STUB).

There is no standalone mod_labour.gms in the RICE50x modules directory.
Labour share calibration is handled within core_economy.gms using data from
data_mod_labour.gdx (labour_share parameter from Guerriero, 2019).

This module exists as a placeholder for future labour-market extensions
(e.g., labour productivity impacts from climate change, labour reallocation,
or endogenous labour supply).

Data available in data_mod_labour/:
    labour_share(n) -- labour share in GDP by region (from Guerriero 2019)
                       Loaded in core_economy for Cobb-Douglas calibration.
                       Values clamped to [0.5, 0.8]; default 0.7 when missing.

Current integration in core_economy:
    prodshare('labour', n) = labour_share(n)
    prodshare('capital', n) = 1 - labour_share(n)
    YGROSS = tfp * K^prodshare_cap * (pop/1000)^prodshare_lab

Future extensions could add:
    - Climate impacts on labour productivity (heat stress, etc.)
    - Endogenous labour supply / participation rates
    - Human capital accumulation
    - Sectoral labour reallocation

Variables: (none yet)
Equations: (none yet)

TODO: Implement labour productivity damage channel if needed.
"""

def declare_vars(m, sets, params, cfg, v):
    """Create labour module variables (STUB -- currently no variables).

    Parameters
    ----------
    m : Container
    sets : dict with t_set, n_set, etc.
    params : dict of GAMSPy Parameter objects
    cfg : Config  -- expects cfg.labour (bool)
    v : dict of all variables (mutated)
    """
    # No additional variables for now.
    # Labour share is used in core_economy via par_prodshare_lab/par_prodshare_cap.
    pass


def define_eqs(m, sets, params, cfg, v):
    """Create labour equations (STUB -- returns empty list).

    TODO: Implement labour productivity impacts:
        - Heat stress damage to effective labour
        - Sectoral labour reallocation costs
        - Human capital accumulation

    Returns
    -------
    list of Equation objects (empty for stub)
    """
    return []
