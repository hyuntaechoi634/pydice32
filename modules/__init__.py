"""
PyDICE32 modules -- translated from GAMS .gms module files.

NOTE: This is not a 1:1 file-for-file port. Some GAMS modules are merged
(core_time, core_regions, pol_ndc), some are partial translations, and
several source modules are not yet implemented (see README.md Coverage).

Each module exposes a ``define(m, sets, params, cfg, v)`` function:
  m      : GAMSPy Container
  sets   : dict  {t_set, n_set, t_alias, layers, layers_alias, n_alias}
  params : dict  {par_ykali, par_pop, par_tfp, ...}
  cfg    : Config object
  v      : dict  of all variables (mutated -- each module ADDs its own)

Returns a list of Equation objects to include in the Model.

Loading order (respects cross-module variable dependencies):
  1. mod_landuse        -- ELAND, MIULAND, MACLAND, ABCOSTLAND
  2. hub_impact         -- OMEGA, DAMFRAC*, DAMAGES  (needs YGROSS stub from v)
  3. hub_climate | mod_climate_fair -- W_EMI, FORC, TATM, etc.
  4. mod_climate_regional -- TEMP_REGION, TEMP_REGION_DAM
  5. mod_impact_dice | mod_impact_kalkuhl | mod_impact_burke -- BIMPACT (+ eq_omega)
  6. core_emissions     -- E, EIND, MIU, ABATEDEMI (needs YGROSS, ELAND)
  7. core_abatement     -- ABATECOST, MAC (needs MIU)
  8. core_economy       -- YGROSS..RI (needs DAMAGES, ABATECOST, ABCOSTLAND)
  9. core_welfare       -- UTARG, UTILITY (needs CPC)
  10. core_policy       -- policy fixings (BAU/CBA)

Optional extension modules (enabled via config flags):
  mod_dac              -- Direct Air Capture (cfg.dac=True)
  mod_emi_stor         -- CCS Storage per-type (loaded with mod_dac)
  mod_sai              -- Stratospheric Aerosol Injection (cfg.sai=True)
  mod_adaptation       -- Adaptation investment (cfg.adaptation=True)
  mod_ocean            -- Ocean capital & ecosystem services (cfg.ocean=True)
  mod_natural_capital  -- Natural capital in production/utility (cfg.natural_capital=True)
  mod_inequality       -- Within-country inequality (cfg.inequality=True)
  mod_labour           -- Labour market extensions (cfg.labour=True) [STUB]
  mod_slr              -- Sea-level rise (cfg.slr=True)
"""

from pydice32.modules import (
    core_economy,
    core_emissions,
    core_abatement,
    core_welfare,
    hub_climate,
    hub_impact,
    mod_impact_dice,
    mod_impact_kalkuhl,
    mod_impact_burke,
    mod_impact_howard,
    mod_impact_dell,
    mod_impact_coacch,
    mod_impact_deciles,
    mod_climate_regional,
    mod_climate_fair,
    mod_climate_tatm_exogen,
    mod_landuse,
    core_policy,
    mod_dac,
    mod_emi_stor,
    mod_sai,
    mod_adaptation,
    mod_ocean,
    mod_natural_capital,
    mod_inequality,
    mod_labour,
    mod_slr,
)

__all__ = [
    "core_economy",
    "core_emissions",
    "core_abatement",
    "core_welfare",
    "hub_climate",
    "hub_impact",
    "mod_impact_dice",
    "mod_impact_kalkuhl",
    "mod_impact_burke",
    "mod_impact_howard",
    "mod_impact_dell",
    "mod_impact_coacch",
    "mod_impact_deciles",
    "mod_climate_regional",
    "mod_climate_fair",
    "mod_landuse",
    "core_policy",
    "mod_dac",
    "mod_emi_stor",
    "mod_sai",
    "mod_adaptation",
    "mod_ocean",
    "mod_natural_capital",
    "mod_inequality",
    "mod_labour",
    "mod_slr",
]
