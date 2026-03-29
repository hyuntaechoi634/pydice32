# Equation and Wiring Gaps

This file focuses on the structural gaps that remain after the latest fixes.

## DAC and Storage

| Source item | GAMS source behavior | Current PyDICE32 behavior | Status |
| --- | --- | --- | --- |
| DAC capacity share | Reuses derived `ccs_stor_cap_max`, so OG/EOR are not double-counted | Now reuses the translated per-type capacity derivation from `mod_emi_stor` | `translated` |
| `eq_cost_cdr` | Uses per-type storage cost `sum(E_STOR * ccs_stor_cost) * CtoCO2` | Now wired through `par_ccs_stor_cost` created in `declare_vars()` | `translated` |
| 2020 `E_NEG.up` cap | Early cap is share-scaled by `capstorreg(n)/totcapstor` | Share-scaled early cap is now present | `translated` |
| Nash fixing of DAC storage controls | Non-active regions are implicitly excluded via `reg(n)` gating | `E_STOR` is now in the fix/unfix path | `translated / approximate` |
| Explicit `ccs_stor_cap_max` override | User can override the capacity scenario independent of baseline default | `Config` has the field, but `_load_storage_data()` still only uses SSP-derived selection | `partial` |

## Decile Damage Mode

| Source item | GAMS source behavior | Current PyDICE32 behavior | Status |
| --- | --- | --- | --- |
| Module replacement | `mod_impact_deciles` replaces the normal impact submodule | Normal impact module is now skipped in decile mode | `translated` |
| `eq_ynetdist_unbnd` in inequality | Uses `DAMAGES_DIST` directly when decile damages are active | Now uses `DAMAGES_DIST` directly | `translated` |
| Linear temperature reference | Uses `TEMP_REGION.l('1',n)` in the linear term | Now mirrors that when available | `translated` |
| Quadratic temperature reference | Uses `base_temp(n)^2` in the quadratic term | Uses `temp_region_reference(n)^2` instead | `altered` |
| `BIMPACT` outside `t_damages` | Fixed to zero outside active periods | Only `t=1` is fixed; period 2 and tail periods remain free | `incorrect` |

## Coalition / Nash Fixing

| Source item | GAMS source behavior | Current PyDICE32 behavior | Status |
| --- | --- | --- | --- |
| Coalition presets | Uses `sel_coalition` + `coalitions/coal_*.gms` | User-defined `coalition_def` only; difference is now documented | `partial but explicit` |
| Extra 3D best-response controls | Inactive for non-active regions via `reg(n)` gating | Restore path exists, but region-column inference for `SAI(t,n,inj)` is still heuristic and fragile | `partial` |

## Post-Process Damage Calculator

| Item | Behavior | Known Limitation | Status |
| --- | --- | --- | --- |
| `postprocess_damages()` | Computes damages from solved TATM path using pure Python (no GAMSPy) | Damages are computed on YGROSS (no-feedback GDP), not on feedback-reduced GDP. BAU postprocess gives ~4x higher damages than endogenous BAU because YGROSS >> YNET when damages are large. This is structural, not a bug — same as GAMS `damages_postprocessed`. | `known limitation` |
| `ctax` + `ctax_marginal=False` + `impact=kalkuhl` | Fiscal revenue approach does not constrain MAC; optimizer finds CBA optimal regardless of tax level | Use `ctax_marginal=True` for tax-driven scenarios, or `impact="off"` with postprocess | `by design` |
| `ctax` + `impact="off"` + `ctax_marginal=False` | Tax enters eq_yy but MAC is unconstrained; near-zero abatement results | Same issue: fiscal term alone insufficient without MAC constraint | `by design` |

## Residual Legacy Gap

| Source item | GAMS source behavior | Current PyDICE32 behavior | Status |
| --- | --- | --- | --- |
| `eq_ctax` under `ctax_marginal` | `MAC = ctax_corrected` | `MAC <= ctax_sched` | `altered` |
