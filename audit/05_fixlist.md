# Recommended Fix Order

## Priority 1: decile-damage equation fidelity

1. Fix `BIMPACT` outside the active `t_damages` window.
   Files: `pydice32/modules/mod_impact_deciles.py`
   Goal: mirror `BIMPACT.fx(t,n,dist)$(not t_damages(t)) = 0` from the source so period 2 and the tail are not free variables.

2. Fix the quadratic temperature-reference term in `eq_bimpact`.
   Files: `pydice32/modules/mod_impact_deciles.py`
   Goal: use `base_temp(n)^2` in the quadratic term while keeping `TEMP_REGION.l('1',n)` only for the linear-term reference.

## Priority 2: Nash / coalition robustness

3. Replace heuristic region-column inference for `_NASH_FIX_VARS_3D_EXTRA`.
   Files: `pydice32/solver.py`
   Goal: use explicit per-variable region-column positions so `SAI(t,n,inj)` is fixed correctly regardless of record ordering.

## Priority 3: storage-capacity override completeness

4. Thread `cfg.ccs_stor_cap_max` through `_load_storage_data()`.
   Files: `pydice32/modules/mod_emi_stor.py`, `pydice32/modules/mod_dac.py`
   Goal: honor explicit user overrides instead of always deriving the capacity scenario from SSP.

## Optional residual

5. Revisit `ctax_marginal` only if that branch matters for your workflow.
   Files: `pydice32/modules/core_policy.py`
   Goal: match the GAMS equality branch or explicitly label the current inequality as an intentional relaxed variant.
