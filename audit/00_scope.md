# pydice32 Static Re-Audit Scope

Generated: 2026-03-29
Audit mode: deep, independent, static-only
Source model: `RICE50xmodel` (GAMS)
Target model: `pydice32` (Python/GAMSPy, GCAM v8+ 32-region aggregation)

This re-audit supersedes the earlier static audit in this folder.

Explicit exclusions for this pass
- `mod_climate_witchghg`
- `mod_impact_sai`

Checked
- Current module coverage against `RICE50xmodel/modules.gms`
- Equation semantics for the areas that were reported fixed after the first audit
- Solver/build ordering, especially module ordering and two-pass side effects
- Nash / coalition best-response fixing for optional decision variables
- Documentation and coverage notes that changed since the first audit

Out of scope
- No model execution
- No numerical regression or scenario comparison
- No convergence / solver diagnostics
- No quantitative validation of GCAM-32 aggregation choices

Assessment labels
- `translated`: close source-to-target port with no material static gap found on this pass
- `merged`: source functionality exists, but is folded into different Python files
- `partial`: functionality exists, but with altered algebra, ordering issues, missing branches, or unsupported options
- `missing`: no static implementation found
- `excluded`: intentionally not audited in this pass

Reading guide
- `01_coverage_matrix.md` gives module-level status for the current codebase
- `02_findings.md` gives the current prioritized findings and resolved items
- `03_equation_gaps.md` gives subsystem-level structural gaps
- `04_status.json` is the machine-readable summary
- `05_fixlist.md` gives the new remediation order
