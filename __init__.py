"""
PyRICE 32 — Python GAMSPy implementation of RICE50x with 32 GCAM regions.

Modular structure mirroring the native GAMS RICE50xmodel.
"""

from pyrice32.config import Config
from pyrice32.solver import build_model, solve_model
