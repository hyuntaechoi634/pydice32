# Final Findings

This file reflects a fresh static re-audit of the current codebase after the reported follow-up fixes.

## Resolved Since The Previous Audit

- DAC now reuses the translated storage-capacity derivation instead of summing raw OG/EOR files directly: `pydice32/modules/mod_dac.py:130-160`, `pydice32/modules/mod_emi_stor.py:124-152`.
- The old DAC build-order issue is resolved because `par_ccs_stor_cost` is now created in `mod_emi_stor.declare_vars()`: `pydice32/modules/mod_emi_stor.py:155-185`, `pydice32/modules/mod_dac.py:276-300`, `pydice32/solver.py:77-90`.
- `E_STOR` is now included in the Nash fix/unfix lists: `pydice32/solver.py:1455-1460`, `pydice32/solver.py:1574-1583`.
- `impact_deciles` now skips the normal impact submodule, so the earlier `BIMPACT` / `eq_bimpact` symbol conflict is resolved: `pydice32/solver.py:1629-1633`.
- `mod_inequality.py` now uses `DAMAGES_DIST` directly when decile damages are active: `pydice32/modules/mod_inequality.py:224-241`.
- `mod_impact_deciles.py` now uses first-period `TEMP_REGION.l('1',n)` as the linear-term reference when available: `pydice32/modules/mod_impact_deciles.py:113-129`.
- `ccs_stor_cap_max` now exists in `Config` and gets SSP-based defaults: `pydice32/config.py:118`, `pydice32/config.py:208-212`.
- CLI support for `--coalition-def=` now exists, and README now documents coalition mode / translated modules more accurately: `pydice32/__main__.py:92-99`, `pydice32/README.md:44-46`, `pydice32/README.md:65`, `pydice32/README.md:227-230`.

## High

### A01. `impact_deciles` still has a material equation-fidelity bug

Impact
- The previous structural wiring issue is largely fixed.
- But the current decile-damage equation path still differs from the source in ways that can materially change decile damage dynamics.

Evidence
- Source fixes `BIMPACT(t,n,dist)` to zero for every period outside `t_damages`, i.e. not just the first period but also period 2 and the final tail: `RICE50xmodel/modules/mod_impact_deciles.gms:102-104`.
- Target only fixes `BIMPACT` at period 1: `pydice32/modules/mod_impact_deciles.py:61-63`.
- Target defines `eq_bimpact` only for `(Ord(t) > 2) & (Ord(t) <= T-20)`: `pydice32/modules/mod_impact_deciles.py:194-205`.
- But `eq_ynet_nobnd` still uses `BIMPACT(t,n,dist)` for every `t > 1` through the penultimate period: `pydice32/modules/mod_impact_deciles.py:207-218`.
- That leaves `BIMPACT` free at least for period 2 and for all tail periods after `T-20`, whereas source pins those periods to zero.
- Source linear term uses `TEMP_REGION_DAM - temp_region_reference(n)` but the quadratic term uses `TEMP_REGION_DAM^2 - base_temp(n)^2`: `RICE50xmodel/modules/mod_impact_deciles.gms:128-130`.
- Target currently uses `par_temp_ref[n]^2` in the quadratic term as well: `pydice32/modules/mod_impact_deciles.py:199-204`.

Why this matters
- The earlier “decile path collides with the normal impact path” issue is mostly gone.
- But the remaining decile equation differences are still large enough to change damages in exactly the mode where source fidelity matters most.

Recommendation
- Fix `BIMPACT` to zero outside the active `t_damages` window, not just at `t=1`.
- Use `base_temp(n)^2` in the quadratic term while keeping `TEMP_REGION.l('1',n)` only for the linear-term reference.

## Medium

### A02. Nash fixing for extra 3D controls is still fragile for `SAI(t,n,inj)`

Impact
- The extra fix/unfix path is much better than before.
- But the current implementation can still mis-detect which column is the region dimension for `SAI(t,n,inj)`.

Evidence
- `_fix_other_regions()` tries to infer the region column for `_NASH_FIX_VARS_3D_EXTRA` by inspecting only the first record and checking whether one of its first three columns belongs to `other_regions`: `pydice32/solver.py:1518-1533`.
- If no match is found in that first row, it falls back to column index `2` for all extra-3D variables: `pydice32/solver.py:1531-1533`.
- That fallback is correct for `(extra, t, n)` variables like `I_ADA`, `K_ADA`, and `E_STOR`, but not for `SAI(t,n,inj)`, where the region is at index `1`: `pydice32/modules/mod_sai.py:195-201`, `pydice32/solver.py:1455-1459`.

Why this matters
- If the first `SAI` record happens to belong to the active coalition rather than `other_regions`, the fallback path can mistake `inj` for the region column and fail to fix the intended non-active regions correctly.

Recommendation
- Replace the heuristic with explicit per-variable region-column metadata.
- At minimum, special-case `SAI` as `(t, n, inj)` rather than treating all extra-3D variables as `(extra, t, n)`.

### A03. Explicit `ccs_stor_cap_max` override is still not wired through storage loading

Impact
- Default SSP-based capacity selection now matches the source more closely.
- But an explicit user override is still not propagated through the storage-data loader.

Evidence
- `Config` now exposes `ccs_stor_cap_max` and fills it from SSP when empty: `pydice32/config.py:118`, `pydice32/config.py:208-212`.
- `mod_dac.py` now calls `_load_storage_data(cfg.data_dir, region_names, cfg.SSP, ...)`: `pydice32/modules/mod_dac.py:134-147`.
- `mod_emi_stor.py` also calls `_load_storage_data(..., cfg.SSP, ...)`: `pydice32/modules/mod_emi_stor.py:168-178`.
- `_load_storage_data()` still derives `cap_scenario` entirely from the `ssp` argument and has no parameter for an explicit `ccs_stor_cap_max` override: `pydice32/modules/mod_emi_stor.py:44-60`.

Why this matters
- For default runs this is much better than before.
- For users who intentionally set `cfg.ccs_stor_cap_max`, the model still behaves as if SSP alone chose the capacity scenario.

Recommendation
- Thread `cfg.ccs_stor_cap_max` through `_load_storage_data()` and honor it when non-empty.

## Acknowledged Partial Translation

- Coalition presets are still not source-ported. The difference is now explicit rather than hidden: `pydice32/config.py:155-163`, `pydice32/__main__.py:92-99`, `pydice32/README.md:65`, `pydice32/README.md:227`.

## Self-Identified Limitation: Post-Process Damage Calculator

`postprocess_damages()` computes damages on YGROSS (the no-feedback GDP path). When damages are large (e.g., BAU at 3.5C), this overstates damages by ~4x compared to the endogenous path where damages reduce GDP which in turn reduces damages. This is the same behavior as GAMS `damages_postprocessed` mode — it is structural, not a bug.

Practical impact: postprocess damages are useful for **relative comparison** between policies (e.g., CTax30 vs CTax50 post-hoc damage difference) but should not be compared directly against endogenous damage scenarios.

Additionally, `ctax` with `ctax_marginal=False` (fiscal revenue) does not constrain MAC, so the tax level has no effect on abatement when `impact="off"`. Use `ctax_marginal=True` for meaningful carbon tax scenarios.

## Legacy Residual Still Present

- `ctax_marginal` still uses `MAC <= tax` instead of the source equality branch `MAC = tax`: `pydice32/modules/core_policy.py:428-437`, `RICE50xmodel/modules/core_policy.gms:405-406`.

## Areas That Looked Mostly Sound On This Pass

- `pydice32/modules/mod_dac.py`
- `pydice32/modules/mod_emi_stor.py`
- `pydice32/modules/mod_inequality.py`
- `pydice32/solver.py` module-order changes for decile mode
- `pydice32/config.py` / `pydice32/__main__.py` coalition and storage-scenario surface
- `pydice32/README.md` coverage/status updates

That list means “no additional material static red flag found on this pass”, not “numerically validated”.
