# Module Coverage Matrix

Legend: `translated`, `merged`, `partial`, `missing`, `excluded`, `n/a`

| Source module | Current status | Target location | Notes |
| --- | --- | --- | --- |
| `core_time.gms` | `merged` | `pydice32/config.py`, `pydice32/solver.py` | Time horizon and set construction handled in Python setup. |
| `mod_stochastic.gms` | `merged` | `pydice32/config.py`, `pydice32/modules/core_welfare.py`, `pydice32/solver.py` | Stochastic SWF branch logic is integrated. |
| `core_regions.gms` | `merged` | `pydice32/data/calibration.py`, `pydice32/solver.py` | Region sets and calibrated inputs are handled outside a standalone module. |
| `core_economy.gms` | `translated` | `pydice32/modules/core_economy.py` | Main production/income/capital path looked close on static pass. |
| `core_emissions.gms` | `translated` | `pydice32/modules/core_emissions.py` | Core multi-GHG emissions path present. |
| `core_welfare.gms` | `translated` | `pydice32/modules/core_welfare.py` | Disentangled, DICE, and stochastic welfare modes present. |
| `core_abatement.gms` | `translated` | `pydice32/modules/core_abatement.py` | MACC path present. |
| `core_cooperation.gms` | `partial` | `pydice32/config.py`, `pydice32/data/calibration.py`, `pydice32/solver.py` | User-defined `coalition_def` exists, but source preset coalition library / full Nash gating are still incomplete. |
| `core_algorithm.gms` | `merged` | `pydice32/solver.py` | Solve loops and convergence logic handled in solver code. |
| `mod_landuse.gms` | `translated` | `pydice32/modules/mod_landuse.py` | Present. |
| `hub_climate.gms` | `translated` | `pydice32/modules/hub_climate.py` | WITCH-CO2 hub present. |
| `mod_climate_fair.gms` | `translated` | `pydice32/modules/mod_climate_fair.py` | Present. |
| `mod_climate_tatm_exogen.gms` | `translated` | `pydice32/modules/mod_climate_tatm_exogen.py` | Exogenous TATM path now exists. |
| `mod_climate_regional.gms` | `translated` | `pydice32/modules/mod_climate_regional.py` | Temperature and precipitation equations now match source algebra more closely; docstring is stale. |
| `mod_climate_witchco2.gms` | `merged` | `pydice32/modules/hub_climate.py` | Folded into climate hub. |
| `mod_climate_witchghg.gms` | `excluded` | not audited in this pass | Explicitly excluded by user request. |
| `hub_impact.gms` | `translated / partial` | `pydice32/modules/hub_impact.py` | Normal aggregate-damage path looks translated; decile mode wiring remains wrong because hub should be bypassed there. |
| `mod_impact_dice.gms` | `translated` | `pydice32/modules/mod_impact_dice.py` | Present. |
| `mod_impact_kalkuhl.gms` | `translated` | `pydice32/modules/mod_impact_kalkuhl.py` | Present. |
| `mod_impact_burke.gms` | `translated` | `pydice32/modules/mod_impact_burke.py` | Present. |
| `mod_impact_howard.gms` | `translated` | `pydice32/modules/mod_impact_howard.py` | Present. |
| `mod_impact_dell.gms` | `translated` | `pydice32/modules/mod_impact_dell.py` | Present. |
| `mod_impact_coacch.gms` | `translated` | `pydice32/modules/mod_impact_coacch.py` | Present. `climcost` is routed through this module via `cfg.damcost`. |
| `mod_impact_deciles.gms` | `partial` | `pydice32/modules/mod_impact_deciles.py`, `pydice32/modules/mod_inequality.py`, `pydice32/solver.py` | Module exists, but source replacement/wiring semantics are not yet correct. |
| `mod_impact_sai.gms` | `excluded` | not audited in this pass | Explicitly excluded by user request. |
| `core_policy.gms` | `partial` | `pydice32/modules/core_policy.py` | Broad policy surface exists; `ctax_marginal` branch still differs from source. |
| `pol_ndc.gms` | `merged` | `pydice32/modules/core_policy.py`, `pydice32/solver.py` | NDC logic integrated into policy/solver code. |
| `mod_adaptation.gms` | `translated / partial` | `pydice32/modules/mod_adaptation.py` | Main equations present; Nash/coalition fixing does not currently cover adaptation controls. |
| `mod_labour.gms` | `n/a / stub` | `pydice32/modules/mod_labour.py` | Python file is an explicit stub; no standalone source file exists in this checkout. |
| `mod_inequality.gms` | `partial` | `pydice32/modules/mod_inequality.py` | Fiscal term fix is in place, but `transfer="opt"` / `omegacalib` / direct `DAMAGES_DIST` branch remain incomplete. |
| `mod_sai.gms` | `partial` | `pydice32/modules/mod_sai.py` | Core path exists; Nash/coalition fixing does not currently cover all g6 control variables. |
| `mod_slr.gms` | `translated` | `pydice32/modules/mod_slr.py` | Present. |
| `mod_natural_capital.gms` | `partial` | `pydice32/modules/mod_natural_capital.py` | Core path exists; `nat_cap_prodfun` branch remains missing. |
| `mod_emission_pulse.gms` | `missing` | none found | Still not implemented. |
| `mod_emi_stor.gms` | `partial` | `pydice32/modules/mod_emi_stor.py`, `pydice32/modules/mod_dac.py` | Storage module now exists, but DAC cost/bound wiring is still not fully source-faithful. |
| `mod_dac.gms` | `partial` | `pydice32/modules/mod_dac.py` | Residual build-order and upper-bound gaps remain. |
| `mod_ocean.gms` | `translated` | `pydice32/modules/mod_ocean.py` | Looked structurally close on static pass. |
| `mod_template.gms` | `n/a` | none needed | Template only. |

Coverage takeaway
- Relative to the first audit, the codebase now covers substantially more of the source surface.
- The biggest remaining risks are no longer “module completely missing”, but “module present with residual wiring/order semantics that still diverge from GAMS”.
