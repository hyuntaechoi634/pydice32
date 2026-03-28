"""
Generate SAI g6 emulator data for PyDICE32.

The GAMS RICE50x model loads ``srm_temperature_response`` and
``srm_precip_response`` from a GDX file (data_mod_sai.gdx) that is not
available in CSV form.  This module synthesises reasonable emulator
coefficients from the data that IS available:

  1. ``sai_temp_global(inj)`` -- global mean temperature change from
     12 TgS/yr single-point injection (hard-coded in GAMS mod_sai.gms)
  2. ``beta_temp(n)`` -- regional climate sensitivity amplification
     factor from ``climate_region_coef_cmip5.csv``

Approach
--------
For temperature:
    sai_temp(n, inj) = sai_temp_global(inj) * beta_temp(n)

    This uses the well-established pattern polar amplification
    (regions with beta > 1 warm faster AND cool faster under SAI).

For precipitation:
    sai_precip(n, inj) = -PRECIP_SCALE * sai_temp_global(inj) * beta_precip(n)

    SAI generally reduces precipitation (PDRMIP studies show ~2% per W/m^2
    global mean precipitation reduction).  We scale by ``beta_precip``
    to get regional variation.  The negative sign means positive
    sai_precip values correspond to REDUCED precipitation.

The data is produced at the ISO3 country level and then aggregated to
GCAM-32 regions using population weights (same as beta_temp aggregation).
"""

# Global temperature change from 12 TgS/yr single-point injection
# GAMS mod_sai.gms lines 92-100
SAI_TEMP_GLOBAL = {
    "60S": 0.95, "45S": 1.2, "30S": 1.3, "15S": 1.12,
    "0": 0.93,
    "15N": 1.09, "30N": 1.28, "45N": 1.22, "60N": 1.06,
}

INJ_LABELS = ["60S", "45S", "30S", "15S", "0", "15N", "30N", "45N", "60N"]

# Precipitation scaling factor:
# PDRMIP: ~2-3% global precipitation change per degree of SAI cooling
# Normalised to 12 TgS reference (matching sai_temp_global units).
# A value of 0.015 means ~1.5% precipitation change per unit sai_temp.
PRECIP_SCALE = 0.015


def generate_sai_emulator_data(beta_temp_dict, beta_precip_dict=None):
    """Generate SAI temperature and precipitation response dicts.

    Parameters
    ----------
    beta_temp_dict : dict
        {region_name: beta_temp_value} at GCAM-32 region level.
    beta_precip_dict : dict or None
        {region_name: beta_precip_value}.  If None, precipitation
        response is set to zero (conservative default).

    Returns
    -------
    sai_temp : dict
        {(region, inj_label): value}  Temperature reduction [deg C]
        from 12 TgS/yr at injection latitude inj.
    sai_precip : dict
        {(region, inj_label): value}  Precipitation change [fraction]
        from 12 TgS/yr at injection latitude inj.
    """
    sai_temp = {}
    sai_precip = {}

    for rn, bt in beta_temp_dict.items():
        for inj in INJ_LABELS:
            # Temperature response: scale global by regional sensitivity
            sai_temp[(rn, inj)] = SAI_TEMP_GLOBAL[inj] * bt

            # Precipitation response: scale by beta_precip if available
            if beta_precip_dict and rn in beta_precip_dict:
                bp = beta_precip_dict[rn]
                sai_precip[(rn, inj)] = -PRECIP_SCALE * SAI_TEMP_GLOBAL[inj] * bp
            else:
                sai_precip[(rn, inj)] = 0.0

    return sai_temp, sai_precip
