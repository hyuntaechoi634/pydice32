# PyDICE32

Python/GAMSPy implementation of the [RICE50x](https://github.com/witch-team/RICE50xmodel) integrated assessment model with 32 GCAM v8+ regions.

## Overview

PyDICE32 translates the GAMS-based RICE50x model into a modular Python package using [GAMSPy](https://gamspy.readthedocs.io/). The model aggregates 155 ISO3 countries into 32 [GCAM v8+](https://jgcri.github.io/gcam-doc/) regions and solves a Ramsey-type optimal growth model coupled with a climate system, damage functions, and abatement cost curves.

Modular architecture with ~110 equations verified against the native GAMS source.

![Global CO2 Emission Pathways](docs/spaghetti_emissions.png)

*Global CO2 emission pathways across 30 scenarios (5 policies x 2 damage functions x 3 MACC cost assumptions). Solid lines show the median MACC (prob50); shaded bands span prob25-prob75 uncertainty.*

## Features

### Policies (11)
| Policy | Description |
|--------|-------------|
| `bau` | Business-as-usual, no climate policy |
| `bau_impact` | BAU with climate damages computed |
| `cba` | Cost-benefit analysis (optimal mitigation) |
| `cbudget` | Cumulative CO2 budget (e.g., 1150 GtCO2 for ~2°C) |
| `cbudget_regional` | Regional carbon budgets with burden sharing |
| `cea_tatm` | Temperature ceiling (e.g., 2°C from 2100) |
| `cea_rcp` | Radiative forcing ceiling |
| `ctax` | Carbon tax with exponential growth schedule |
| `global_netzero` | Global net-zero CO2 by target year |
| `long_term_pledges` | Per-country net-zero pledges (Paris Agreement) |
| `simulation` | Fixed MIU=0, no mitigation |

### Damage Functions (3)
- **DICE** — Nordhaus (2018) quadratic temperature-level damages
- **Kalkuhl** — Kalkuhl & Wenz (2020) growth-rate damages (simple + full omega)
- **Burke** — Burke, Hsiang & Miguel (2015) with sr/lr/srdiff/lrdiff specifications

### Climate Modules (2)
- **WITCH-CO2** — 3-box carbon cycle with 2-layer temperature (default)
- **FAIR** — Impulse response model with 20 equations (Smith et al., 2018)

### Social Welfare Functions (3)
- **Disentangled** — Berger & Emmerling (2020) equity-equivalents (default)
- **DICE** — Original DICE SWF with Negishi weights
- **Stochastic** — Branch-node probability tree with ambiguity aversion

### Multi-GHG
CO2, CH4, N2O with species-specific:
- Carbon intensity (sigma) from SSP scenarios
- Marginal abatement cost curves (MACC) with backstop calibration
- Global warming potential (GWP) conversion
- Emission quantity/cost unit conversion (convq_ghg, convy_ghg)

### Solve Modes
- **Single-pass** — Fast cooperative optimization (default)
- **Cooperative iterative** — Multi-iteration with Negishi weight updates, DAC learning
- **Nash non-cooperative** — Per-coalition best-response with convergence tracking

### Extension Modules
| Module | Description |
|--------|-------------|
| DAC | Direct air capture with learning curve and CCS storage costs |
| SAI | Stratospheric aerosol injection (g0 uniform + g6 multi-latitude emulator) |
| Adaptation | CES-nested adaptive capacity (proactive, reactive, specific, generic) |
| Ocean | Ocean capital ecosystem services (coral, mangrove, fisheries) |
| Natural capital | Green/blue capital in production and utility |
| Inequality | Within-country income distribution by deciles |
| Sea-level rise | Thermal expansion + ice sheet dynamics with economic feedback |

## Installation

### Requirements
- Python 3.10+
- [GAMSPy](https://gamspy.readthedocs.io/) with GAMS license
- NumPy, Pandas

### Setup
```bash
conda create -n pydice32 python=3.11
conda activate pydice32
pip install gamspy numpy pandas
```

### Data
PyDICE32 reads CSV data from the RICE50x model directory. Ensure this structure exists:
```
rice-fund-gcam/
├── pydice32/              # this package
├── RICE50xmodel/
│   └── data_maxiso3_csv/  # CSV exports of GAMS GDX data
└── gcam-core/
    └── input/gcamdata/inst/extdata/common/
        ├── iso_GCAM_regID.csv
        └── GCAM_region_names.csv
```

## Usage

### Command Line
```bash
# Basic scenarios
python -m pydice32 bau --dice
python -m pydice32 cba --dice
python -m pydice32 cba --kalkuhl

# Carbon budget (2°C and 1.5°C)
python -m pydice32 cbudget --dice --cbudget=1150
python -m pydice32 cbudget --dice --cbudget=500

# Temperature ceiling
python -m pydice32 cea_tatm --dice --tatm-limit=2.0

# Carbon tax
python -m pydice32 ctax --dice --ctax-initial=50 --ctax-slope=0.05

# Net-zero
python -m pydice32 global_netzero --dice --nz-year=2050

# With options
python -m pydice32 cba --dice --prstp=0.03 --macc=prob75

# Iterative cooperative solve
python -m pydice32 cba --dice --iterative

# Nash non-cooperative
python -m pydice32 cba --dice --noncoop
```

### Batch Scenarios
```bash
# Run 30 scenarios (5 policies × 2 damage functions × 3 MACC costs)
python -m pydice32.batch_run
```
Results saved to `pydice32/results/`.

### Python API
```python
from pydice32.config import Config
from pydice32.solver import build_model, solve_model
from pydice32.report import print_results

cfg = Config(policy="cba", impact="dice", PRSTP=0.015)
m, rice, v, data = build_model(cfg)
solve_model(rice, cfg)
print_results(m, rice, cfg, v, data)
```

## Architecture

```
pydice32/
├── config.py                  # All parameters in one dataclass
├── solver.py                  # Model assembly + single-pass/iterative/Nash solve
├── report.py                  # Result extraction and printing
├── batch_run.py               # Multi-scenario batch runner
├── data/
│   ├── loader.py              # CSV I/O utilities
│   ├── gcam_mapping.py        # 155 ISO3 → 32 GCAM region aggregation
│   ├── calibration.py         # TFP, savings, sigma, MACC, climate parameters
│   └── sai_emulator_data.py   # SAI g6 regional response synthesis
└── modules/                   # 1:1 mapping with GAMS .gms files
    ├── core_economy.py        # Cobb-Douglas production, capital, consumption
    ├── core_emissions.py      # Multi-GHG emissions with MIU inertia
    ├── core_abatement.py      # MACC polynomial cost curves
    ├── core_welfare.py        # Disentangled / DICE / Stochastic SWF
    ├── core_policy.py         # 11 policy constraint equations
    ├── hub_climate.py         # WITCH-CO2 carbon cycle + temperature
    ├── hub_impact.py          # Damage fraction with cap, adaptation, SLR
    ├── mod_climate_fair.py    # FAIR impulse response (20 equations)
    ├── mod_climate_regional.py # Regional temperature downscaling
    ├── mod_impact_dice.py     # DICE quadratic damage
    ├── mod_impact_kalkuhl.py  # Kalkuhl growth-rate damage + full omega
    ├── mod_impact_burke.py    # Burke level damage + differentiated specs
    ├── mod_landuse.py         # Land-use emissions and abatement
    ├── mod_dac.py             # Direct air capture with learning
    ├── mod_sai.py             # SAI geoengineering (g0 + g6)
    ├── mod_adaptation.py      # CES adaptive capacity
    ├── mod_ocean.py           # Ocean ecosystem services
    ├── mod_natural_capital.py # Natural capital in production
    ├── mod_inequality.py      # Decile income distribution
    ├── mod_slr.py             # Sea-level rise components
    └── mod_labour.py          # Labour market (stub)
```

Each module follows a two-pass pattern mirroring GAMS phases:
1. `declare_vars()` — Create variables, set bounds and starting values
2. `define_eqs()` — Create equations referencing cross-module variables

## References

- Emmerling, J., et al. (2024). RICE50x: The RICE50+ Integrated Assessment Model. [GitHub](https://github.com/witch-team/RICE50xmodel)
- Nordhaus, W. (2018). Projections and Uncertainties about Climate Change in an Era of Minimal Climate Policies. *AEJ: Economic Policy*, 10(3), 333-360.
- Kalkuhl, M. & Wenz, L. (2020). The impact of climate conditions on economic production. *Journal of Environmental Economics and Management*, 103, 102360.
- Burke, M., Hsiang, S. & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527, 235-239.
- Smith, C.J., et al. (2018). FAIR v1.3: A simple emissions-based impulse response and carbon cycle model. *Geoscientific Model Development*, 11, 2273-2297.
- Berger, L. & Emmerling, J. (2020). Welfare as Equity Equivalents. *Journal of Economic Surveys*, 34(4), 727-752.

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
