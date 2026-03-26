"""
CLI entry point: python -m pyrice32 <policy> [options]

Usage:
    python -m pyrice32 bau --dice
    python -m pyrice32 bau_impact --kalkuhl
    python -m pyrice32 cba --dice
    python -m pyrice32 cbudget --dice --cbudget=1150
    python -m pyrice32 cbudget_regional --kalkuhl --burden=equal_per_capita
    python -m pyrice32 cea_tatm --dice --tatm-limit=2.0
    python -m pyrice32 cea_rcp --kalkuhl --forc-limit=4.5
    python -m pyrice32 ctax --dice --ctax-initial=5 --ctax-slope=0.05
    python -m pyrice32 global_netzero --dice --nz-year=2050
    python -m pyrice32 long_term_pledges --kalkuhl
    python -m pyrice32 simulation --dice
    python -m pyrice32 cba --kalkuhl --macc=prob75
"""

import sys
from pyrice32.config import Config
from pyrice32.solver import build_model, solve_model, solve_model_iterative, solve_model_nash
from pyrice32.report import print_results


VALID_POLICIES = (
    "bau", "bau_impact", "cba", "cbudget", "cbudget_regional",
    "cea_tatm", "cea_rcp",
    "ctax", "global_netzero", "long_term_pledges", "simulation",
)


def main():
    args = sys.argv[1:]

    policy = args[0] if args else "bau"
    if policy.startswith("--"):
        # No policy given, default to bau
        policy = "bau"
        flag_args = args
    else:
        flag_args = args[1:]

    if policy not in VALID_POLICIES:
        print(f"Unknown policy '{policy}'. Valid: {', '.join(VALID_POLICIES)}")
        sys.exit(1)

    cfg = Config(policy=policy)

    for arg in flag_args:
        if arg == "--dice":
            cfg.impact = "dice"
        elif arg == "--kalkuhl":
            cfg.impact = "kalkuhl"
        elif arg == "--damage-cap":
            cfg.damage_cap = True
        elif arg == "--flexible":
            cfg.savings_mode = "flexible"
        elif arg.startswith("--prstp="):
            cfg.PRSTP = float(arg.split("=")[1])
        elif arg.startswith("--cbudget="):
            cfg.cbudget = float(arg.split("=")[1])
        elif arg.startswith("--tatm-limit="):
            cfg.tatm_limit = float(arg.split("=")[1])
        elif arg.startswith("--macc="):
            cfg.macc_costs = arg.split("=")[1]
        elif arg.startswith("--ctax-initial="):
            cfg.ctax_initial = float(arg.split("=")[1])
        elif arg.startswith("--ctax-start="):
            cfg.ctax_start = int(arg.split("=")[1])
        elif arg.startswith("--ctax-slope="):
            cfg.ctax_slope = float(arg.split("=")[1])
        elif arg.startswith("--nz-year="):
            cfg.nz_year = int(arg.split("=")[1])
        elif arg.startswith("--burden="):
            cfg.burden = arg.split("=")[1]
        elif arg.startswith("--forc-limit="):
            cfg.forc_limit = float(arg.split("=")[1])
        elif arg == "--iterative":
            cfg.cooperation = "coop_iterative"
        elif arg == "--noncoop":
            cfg.cooperation = "noncoop"
        elif arg == "--coalitions":
            cfg.cooperation = "coalitions"
        elif arg.startswith("--max-iter="):
            cfg.max_iter = int(arg.split("=")[1])
        elif arg.startswith("--min-iter="):
            cfg.min_iter = int(arg.split("=")[1])
        elif arg.startswith("--convergence-tol="):
            cfg.convergence_tol = float(arg.split("=")[1])
        else:
            print(f"Warning: unknown argument '{arg}'")

    m, rice, v, data = build_model(cfg)

    # Route to appropriate solver based on cooperation mode
    if cfg.cooperation in ("noncoop", "coalitions"):
        solve_model_nash(m, rice, v, data, cfg)
    elif cfg.cooperation == "coop_iterative":
        solve_model_iterative(m, rice, v, data, cfg)
    else:
        # Default: single-pass cooperative solve
        solve_model(rice, cfg)

    print_results(m, rice, cfg, v, data)


if __name__ == "__main__":
    main()
