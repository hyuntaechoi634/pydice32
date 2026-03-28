"""
Configuration and global parameters for PyDICE32.
Corresponds to GAMS 'conf' and 'include_data' phases across all modules.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # Time
    T: int = 58                     # number of periods (2015–2300)
    TSTEP: int = 5                  # years per period

    # Economy
    DK: float = 0.1                 # capital depreciation rate (per year)
    PRSTP: float = 0.015            # pure rate of social time preference
    ELASMU: float = 1.45            # elasticity of marginal utility of consumption
    GAMMA: float = 0.5              # inequality aversion (disentangled SWF)
    calib_labour_share: bool = False # use calibrated labour shares from data

    # Welfare
    swf: str = "disentangled"       # disentangled | dice | stochastic
    region_weights: str = "pop"     # pop | negishi
    dice_scale1: float = 1e-4       # DICE SWF scaling coefficient 1
    dice_scale2: float = 0.0        # DICE SWF scaling coefficient 2

    # Stochastic SWF parameters (only used when swf='stochastic')
    # GAMS mod_stochastic.gms defaults
    rra: float = 10.0               # relative risk aversion (GAMS default=10)
    t_resolution: int = 8           # t_resolution_one: period where uncertainty resolves
    t_resolution_two: int = 59      # t_resolution_two: end of branch window
    num_branches: int = 3           # number of stochastic branches
    branch_probs: tuple = (0.333, 0.334, 0.333)  # probability per branch

    # Baseline
    SSP: str = "SSP2"
    EXCHANGE_RATE: str = "PPP"

    # Policy
    policy: str = "bau"             # bau | bau_impact | cba | cbudget | cea_tatm
                                    # | cea_rcp | ctax | cbudget_regional
                                    # | global_netzero | long_term_pledges
                                    # | simulation
    savings_mode: str = "fixed"     # fixed | flexible
    damage_cap: bool = False
    cbudget: float = 1150.0         # carbon budget GtCO2 (2020-2100), for policy=cbudget
    tatm_limit: float = 2.0         # temperature limit °C, for policy=cea_tatm
    macc_costs: str = "prob50"      # prob25 | prob33 | prob50 | prob66 | prob75

    # Carbon tax (ctax policy)
    ctax_initial: float = 5.0       # $/tCO2 in 2015
    ctax_start: int = 2025          # year tax begins
    ctax_slope: float = 0.05        # annual growth rate
    ctax_shape: str = "exponential" # exponential (only option for now)
    ctax_marginal: bool = True      # True = MAC-fixing approach (ctax_marginal variant);
                                    # False = fiscal revenue approach (ctax_corrected in eq_yy)

    # Net-zero (global_netzero policy)
    nz_year: int = 2050             # year global net-zero CO2 is enforced

    # Radiative forcing ceiling (cea_rcp policy)
    forc_limit: float = 4.5         # W/m2 forcing limit, for policy=cea_rcp

    # NDC (Nationally Determined Contributions)
    pol_ndc: bool = False           # activate NDC MIU floors for t=3,4

    # Regional carbon budget (cbudget_regional policy)
    burden: str = "cost_efficiency" # burden sharing: equal_per_capita | historical_responsibility
                                    # | grandfathering | cost_efficiency

    # Impact
    impact: str = ""                # "" (sentinel: use policy default) | dice | kalkuhl | burke
    bhm_spec: str = "sr"            # Burke spec: sr | lr | srdiff | lrdiff
    omega_eq: str = "simple"        # simple | full (omega formulation for Kalkuhl/Burke)

    # Impact: threshold and gradient damage (hub_impact extras)
    threshold_damage: bool = False
    threshold_d: float = 0.20       # threshold damage magnitude (fraction of GDP)
    threshold_temp: float = 3.0     # threshold temperature [deg C above pre-industrial]
    threshold_sigma: float = 0.05   # smoothing width for threshold [deg C]
    gradient_damage: bool = False
    gradient_d: float = 0.01        # gradient damage magnitude (fraction of GDP)

    # Impact: damages post-processing mode
    damages_postprocessed: bool = False  # True = decouple DAMAGES from optimization

    # Climate
    climate: str = "witchco2"       # witchco2 | fair

    # Climate: regional temperature cap (Burke conservative approach)
    temp_region_cap: bool = False
    max_temp_region_dam: float = 30.0  # maximum regional temperature for damages [deg C]

    # Abatement / MIU
    miu_inertia: float = 0.034      # per year (default RICE)
    max_miuup: float = 1.0          # MIU upper bound for CO2
    tmiufix: tuple = (1, 2)         # periods where MIU is fixed

    # GHG species
    ghg_list: tuple = ("co2", "ch4", "n2o")

    # Backstop
    pback: float = 550.0            # backstop price 2015 $/tCO2
    gback: float = 0.025            # annual decline rate
    tstart_pbtransition: int = 7
    tend_pbtransition: int = 28
    klogistic: float = 0.25

    # ── Optional extension modules ──────────────────────────
    # DAC (Direct Air Capture)
    dac: bool = False               # activate DAC module
    dac_cost: str = "best"          # DAC cost scenario: low | best | high
    dac_growth: str = "medium"      # DAC market growth: low | medium | high

    # SAI (Stratospheric Aerosol Injection)
    sai: bool = False               # activate SAI module
    sai_start: int = 2050           # year SAI deployment begins
    sai_end: int = 2200             # year SAI deployment ends
    sai_experiment: str = "g0"      # g0 (uniform solar constant) | g6 (emulator)
    sai_mode: str = "free"          # free | max_efficiency | equator | tropics | symmetric
    sai_damage_coef: float = 0.03   # damage coef for g0: GDP loss fraction for 12 TgS/yr

    # Adaptation
    adaptation: bool = False        # activate adaptation module

    # Ocean (ocean capital & ecosystem services)
    ocean: bool = False             # activate ocean module

    # Natural Capital
    natural_capital: bool = False   # activate natural capital module

    # Inequality (within-country income distribution)
    inequality: bool = False        # activate inequality module

    # Labour (labour market extensions)
    labour: bool = False            # activate labour module

    # Sea-Level Rise
    slr: bool = False               # activate SLR module
    damcostslr: str = "COACCH_SLR_Ad"  # SLR damage scenario (GAMS: %damcostslr%)

    # Cooperation mode (mirrors GAMS core_cooperation.gms)
    cooperation: str = "coop"       # coop | noncoop | coalitions
    max_iter: int = 100             # max iterations for iterative solve
    min_iter: int = 4               # min iterations before convergence check
    convergence_tol: float = 1e-2   # max relative variation for convergence

    # Paths (set by resolve_paths)
    project_root: str = ""
    data_dir: str = ""
    gcam_csv: str = ""
    gcam_names_csv: str = ""

    def __post_init__(self):
        self.resolve_paths()
        self._apply_policy_defaults()
        self._validate()

    def _apply_policy_defaults(self):
        """Apply GAMS-style per-policy defaults for impact mode and other settings.

        Only overrides impact if the user has not explicitly set it (impact is
        still the sentinel empty string "").  Per-policy defaults mirror GAMS
        core_policy.gms flag resolution.
        """
        # Issue 3: policy_with_damages alignment -- map each policy to its
        # GAMS-default impact setting so that bau/simulation get "off" etc.
        policy_impact_defaults = {
            "bau": "off",
            "simulation": "off",
            "bau_impact": "kalkuhl",
            "cba": "kalkuhl",
            "cbudget": "off",
            "cbudget_regional": "off",
            "ctax": "off",
            "cea_tatm": "kalkuhl",
            "cea_rcp": "kalkuhl",
            "global_netzero": "off",
            "long_term_pledges": "off",
        }
        # Only override impact if user didn't set it (still sentinel "")
        if self.impact == "":
            self.impact = policy_impact_defaults.get(self.policy, "kalkuhl")
        # If user explicitly set it (non-empty), respect their choice

        # Issue 1: cea_tatm and cea_rcp auto-set damages_postprocessed
        if self.policy in ("cea_tatm", "cea_rcp"):
            self.damages_postprocessed = True

        # Issue 4: long_term_pledges doubles miu_inertia (GAMS line 128)
        if self.policy == "long_term_pledges":
            self.miu_inertia = 0.068
            self.pol_ndc = True  # GAMS: long_term_pledges activates NDCs

        # When pol_ndc is active, GAMS extends tmiufix to {1,2,3,4} so that
        # inertia constraints won't apply to the NDC-fixed periods.
        if self.pol_ndc:
            self.tmiufix = (1, 2, 3, 4)

        # cbudget_regional doubles miu_inertia (GAMS line 91)
        if self.policy == "cbudget_regional":
            self.miu_inertia = 0.068

        # SAI-3: Auto-set sai_damage_coef for g0 experiment.
        # When SAI is enabled with the g0 experiment and sai_damage_coef
        # is still at its dataclass default, ensure it is set to 0.03
        # (3% GDP loss for 12 TgS/yr, matching GAMS mod_sai.gms).
        if self.sai and self.sai_experiment == "g0":
            # Only set if still at default; a user-specified value should
            # not be overridden.
            if self.sai_damage_coef == 0.03:
                pass  # already correct default
            # If user explicitly set it to something else, respect that.

    def _validate(self):
        # GAMS core_welfare.gms line 46:
        # Negishi weights require DICE welfare function
        if self.swf == "disentangled" and self.region_weights == "negishi":
            raise ValueError(
                "Negishi weights require DICE welfare function "
                "(set swf='dice' when region_weights='negishi')"
            )
        # FAIR climate module is now implemented
        # (no longer raises NotImplementedError)
        # Stochastic SWF: validate branch_probs length matches num_branches
        if self.swf == "stochastic":
            if len(self.branch_probs) != self.num_branches:
                raise ValueError(
                    f"branch_probs length ({len(self.branch_probs)}) must equal "
                    f"num_branches ({self.num_branches})"
                )
            if abs(sum(self.branch_probs) - 1.0) > 1e-6:
                import warnings
                warnings.warn(
                    f"branch_probs sum to {sum(self.branch_probs):.6f}, not 1.0.",
                    UserWarning,
                )
            # Validate T vs. branch requirements.
            #
            # The GAMS model uses T>160 with t_resolution_two=59 for 3 full
            # branches of 51 periods each.  When T is smaller, solver.py
            # dynamically adjusts t_resolution_two to min(configured, T+1)
            # so that branches are shorter but still functional.
            #
            # With T=58: effective span = min(59,59)-8 = 51, branches fit
            #   within T (8+51*3=161 > 58, so branches share periods via
            #   the GAMS round() formula -- branch 0 covers all post-res).
            # With T=32: effective span = min(59,33)-8 = 25, each branch
            #   gets 25 periods, so 3 branches need 8+75=83 > 32 periods.
            #   Branch allocation still works: the GAMS round() formula
            #   assigns early post-resolution periods to branch 0, so
            #   shorter T gives fewer effective branch periods.
            #
            # Full branching (all branches get equal periods) requires:
            effective_t_res2 = min(self.t_resolution_two, self.T + 1)
            effective_span = max(effective_t_res2 - self.t_resolution, 1)
            max_t_for_full = (self.t_resolution
                              + self.num_branches * effective_span)
            if self.T < max_t_for_full:
                import warnings
                warnings.warn(
                    f"Stochastic SWF: T={self.T} < {max_t_for_full} for "
                    f"{self.num_branches} full branches (effective span="
                    f"{effective_span}). Later branches will have fewer "
                    f"or zero periods. Consider increasing T or reducing "
                    f"num_branches for full stochastic functionality.",
                    UserWarning,
                )
        # Burke srdiff/lrdiff now use region-varying coefficients
        # computed from GDP per capita cutoff in mod_impact_burke.py

    def resolve_paths(self):
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(
            self.project_root, "RICE50xmodel", "data_maxiso3_csv")
        self.gcam_csv = os.path.join(
            self.project_root, "gcam-core", "input", "gcamdata", "inst",
            "extdata", "common", "iso_GCAM_regID.csv")
        self.gcam_names_csv = os.path.join(
            self.project_root, "gcam-core", "input", "gcamdata", "inst",
            "extdata", "common", "GCAM_region_names.csv")

    @property
    def policy_with_damages(self):
        """Whether damage feedback is active in the optimization.

        After _apply_policy_defaults, impact is set per GAMS conventions:
        - "off" for bau, simulation, cbudget, ctax, global_netzero, long_term_pledges
        - "kalkuhl" (or user-chosen) for bau_impact, cba, cea_tatm, cea_rcp
        The user can override via CLI (--kalkuhl, --dice) which changes impact
        away from the default, enabling damages for any policy.
        """
        if self.policy in ("bau", "simulation"):
            return False
        if self.policy in ("bau_impact", "cba"):
            return True
        # For other policies: damages are active unless impact is "off"
        return self.impact != "off"

    @property
    def miu_inertia_per_period(self):
        return self.miu_inertia * self.TSTEP

    def year(self, t):
        return 2015 + self.TSTEP * (t - 1)
