"""
ksmesopy
=========
Python API for the Kansas Mesonet.
https://mesonet.k-state.edu
"""

from ksmesopy.core import (
    get_stations,
    get_stations_active,
    list_variables,
    VARIABLES,
    RENAME_PRESET,
    _VWC_DEPS,
    _ALL_VWC,
    _VALID_FOR,
    request_data,
    request_data_multi,
    rename_columns,
    calibrate_vwc,
    compute_soil_water_storage,
    srad_to_mj,
    atmospheric_pressure,
    saturation_vapor_pressure,
    actual_vapor_pressure,
    vapor_pressure_deficit,
    slope_saturation_vapor_pressure,
    psychrometric_constant,
    extraterrestrial_radiation,
    net_radiation,
    reference_et_penman_monteith,
    reference_et_hargreaves,
)

from ksmesopy.utils import (
    growing_degree_days,
    heat_index,
    wind_chill,
    temperature_humidity_index,
)

__all__ = [
    # Station metadata
    "get_stations",
    "get_stations_active",
    # Variable catalogue
    "list_variables",
    "VARIABLES",
    "RENAME_PRESET",
    "_VWC_DEPS",
    "_ALL_VWC",
    "_VALID_FOR",
    # Data retrieval
    "request_data",
    "request_data_multi",
    "rename_columns",
    # Soil processing
    "calibrate_vwc",
    "compute_soil_water_storage",
    # Atmospheric helpers
    "srad_to_mj",
    "atmospheric_pressure",
    "saturation_vapor_pressure",
    "actual_vapor_pressure",
    "vapor_pressure_deficit",
    "slope_saturation_vapor_pressure",
    "psychrometric_constant",
    "extraterrestrial_radiation",
    "net_radiation",
    # Reference ET
    "reference_et_penman_monteith",
    "reference_et_hargreaves",
    # Derived variables
    "growing_degree_days",
    "heat_index",
    "wind_chill",
    "temperature_humidity_index",
]
