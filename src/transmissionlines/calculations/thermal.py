# The heat balance below is ported from DynamicLineRatings (dlr/physics.py,
# https://github.com/NatLabRockies/DynamicLineRatings), which carries this notice:
#
# BSD 3-Clause License
#
# Copyright (c) 2024, Alliance for Sustainable Energy, LLC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
r"""Steady-state thermal rating of a bare overhead conductor (IEEE 738-2012).

The rating is the current whose Joule heating balances convective and radiative
cooling minus solar heating at the allowed conductor temperature:

.. math:: I = \sqrt{\frac{q_c + q_r - q_s}{R(T_c)}}

Convective cooling is the largest of the natural, low-wind and high-wind forms.
Solar heating is ``GHI · D · α``, without the standard's sun-angle geometry.
Air density comes from the ideal gas law for moist air at the film temperature.
The water vapor partial pressure is relative humidity times the Tetens saturation
pressure at the *air* temperature; heating air to the film temperature at constant
total pressure leaves that partial pressure unchanged.
"""

from bisect import bisect
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from transmissionlines.models.cables import BundleSpec, ConductorSpec, ResistanceAtTemperature

C_TO_K = 273.15
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.058
WATER_VAPOR_GAS_CONSTANT_J_KG_K = 461.495
METERS_PER_INCH = 0.0254
METERS_PER_KILOFOOT = 304.8

WEATHER_COLUMNS = ("air_temperature_k", "wind_speed_m_s", "pressure_pa", "ghi_w_m2")
# Optional weather column, a fraction; DynamicLineRatings' default is used when absent.
HUMIDITY_COLUMN = "relative_humidity"
DEFAULT_RELATIVE_HUMIDITY = 0.5
# Bounds that catch values given in the wrong unit (°C for K, kPa for Pa).
_AIR_TEMPERATURE_RANGE_K = (180.0, 340.0)
_PRESSURE_RANGE_PA = (40e3, 110e3)


def ampacity(
    *,
    diameter_m: ArrayLike,
    resistance_ohm_m: ArrayLike,
    conductor_temperature_k: ArrayLike,
    air_temperature_k: ArrayLike,
    wind_speed_m_s: ArrayLike,
    wind_angle_deg: ArrayLike,
    pressure_pa: ArrayLike,
    relative_humidity: ArrayLike,
    ghi_w_m2: ArrayLike,
    emissivity: ArrayLike,
    absorptivity: ArrayLike,
) -> np.ndarray:
    """Return the steady-state rating of one conductor, in amperes.

    All inputs are SI and broadcast together; ``relative_humidity`` is a fraction
    (0–1) of saturation at the air temperature. ``wind_angle_deg`` is the angle
    between the wind and the conductor axis; any angle is accepted and folded into
    0–90°. Conditions where the conductor cannot be held at its allowed
    temperature (air hotter than the conductor, or sun alone heating it past
    that) rate as zero.
    """
    d, r = np.asarray(diameter_m, float), np.asarray(resistance_ohm_m, float)
    t_c, t_a = np.asarray(conductor_temperature_k, float), np.asarray(air_temperature_k, float)
    angle = np.asarray(wind_angle_deg, float)
    film_k = (t_c + t_a) / 2
    film_c = film_k - C_TO_K
    air_c = t_a - C_TO_K
    vapor_pa = np.asarray(relative_humidity, float) * 610.78 * np.exp(17.27 * air_c / (air_c + 237.3))
    density = (
        (np.asarray(pressure_pa, float) - vapor_pa) / DRY_AIR_GAS_CONSTANT_J_KG_K
        + vapor_pa / WATER_VAPOR_GAS_CONSTANT_J_KG_K
    ) / film_k
    viscosity = 1.458e-6 * film_k**1.5 / (film_c + 383.4)
    conductivity = 2.424e-2 + 7.477e-5 * film_c - 4.407e-9 * film_c**2
    folded = np.deg2rad(90 - np.abs(np.remainder(np.abs(angle), 180) - 90))
    k_angle = 1.194 - np.cos(folded) + 0.194 * np.cos(2 * folded) + 0.368 * np.sin(2 * folded)
    reynolds = d * density * np.asarray(wind_speed_m_s, float) / viscosity
    rise = np.maximum(t_c - t_a, 0)
    convection = np.maximum(
        3.645 * density**0.5 * d**0.75 * rise**1.25,
        k_angle * conductivity * rise * np.maximum(1.01 + 1.35 * reynolds**0.52, 0.754 * reynolds**0.6),
    )
    radiation = np.pi * d * np.asarray(emissivity, float) * STEFAN_BOLTZMANN_W_M2_K4 * (t_c**4 - t_a**4)
    solar = np.asarray(ghi_w_m2, float) * d * np.asarray(absorptivity, float)
    return np.sqrt(np.maximum(convection + radiation - solar, 0) / r)


def ac_resistance_at(points: Sequence[ResistanceAtTemperature], temperature_c: float) -> float:
    """Interpolate published AC resistance (ohm/kft) linearly in temperature.

    Uses the two published temperatures either side of the requested one, or the
    two nearest it beyond the published range. A conductor with one published
    temperature rates only at that temperature.
    """
    for point in points:
        if point.temperature_c == temperature_c:
            return point.ac_resistance_ohm_kft
    if len(points) < 2:
        published = [point.temperature_c for point in points]
        raise ValueError(
            f"AC resistance at {temperature_c} °C needs two published temperatures; have {published}"
        )
    ordered = sorted(points, key=lambda p: p.temperature_c)
    above = bisect([p.temperature_c for p in ordered], temperature_c)
    low, high = ordered[min(max(above, 1), len(ordered) - 1) - 1:][:2]
    slope = (high.ac_resistance_ohm_kft - low.ac_resistance_ohm_kft) / (high.temperature_c - low.temperature_c)
    return low.ac_resistance_ohm_kft + slope * (temperature_c - low.temperature_c)


def validate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Check a weather table's columns, units and UTC time index."""
    missing = [column for column in WEATHER_COLUMNS if column not in weather]
    if missing:
        raise ValueError(f"weather is missing columns {missing}")
    if not isinstance(weather.index, pd.DatetimeIndex) or str(weather.index.tz) != "UTC":
        raise ValueError("weather needs a UTC DatetimeIndex")
    values = weather[list(WEATHER_COLUMNS)]
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("weather values must be finite")
    for column, (low, high) in (("air_temperature_k", _AIR_TEMPERATURE_RANGE_K), ("pressure_pa", _PRESSURE_RANGE_PA)):
        if not weather[column].between(low, high).all():
            raise ValueError(f"{column} outside {low}–{high}; check units")
    if (weather["wind_speed_m_s"] < 0).any() or (weather["ghi_w_m2"] < 0).any():
        raise ValueError("wind_speed_m_s and ghi_w_m2 must be nonnegative")
    if HUMIDITY_COLUMN in weather and not weather[HUMIDITY_COLUMN].astype(float).between(0, 1).all():
        raise ValueError(f"{HUMIDITY_COLUMN} must be a fraction between 0 and 1, not a percentage")
    return weather


def conductor_ampacity(
    conductor: ConductorSpec,
    bundle: BundleSpec,
    weather: pd.DataFrame,
    *,
    wind_angle_deg: float | np.ndarray,
    conductor_temperature_c: float = 75.0,
    emissivity: float = 0.8,
    absorptivity: float = 0.8,
) -> pd.Series:
    """Rate one bundled phase for each weather row, in amperes.

    The bundle carries ``subconductor_count`` times the single-conductor rating.
    Emissivity and absorptivity default to DynamicLineRatings' 0.8. Without a
    ``relative_humidity`` column, humidity is 0.5.
    """
    if not (0 <= emissivity <= 1 and 0 <= absorptivity <= 1):
        raise ValueError("emissivity and absorptivity must be fractions between 0 and 1")
    weather = validate_weather(weather)
    equipment = conductor.equipment
    if equipment.diameter_in is None:
        raise ValueError(f"{conductor.name}: missing diameter_in")
    resistance = ac_resistance_at(equipment.ac_resistances, conductor_temperature_c)
    amps = ampacity(
        diameter_m=equipment.diameter_in * METERS_PER_INCH,
        resistance_ohm_m=resistance / METERS_PER_KILOFOOT,
        conductor_temperature_k=conductor_temperature_c + C_TO_K,
        air_temperature_k=weather["air_temperature_k"].to_numpy(),
        wind_speed_m_s=weather["wind_speed_m_s"].to_numpy(),
        wind_angle_deg=np.asarray(wind_angle_deg, float),
        pressure_pa=weather["pressure_pa"].to_numpy(),
        relative_humidity=(
            weather[HUMIDITY_COLUMN].to_numpy() if HUMIDITY_COLUMN in weather else DEFAULT_RELATIVE_HUMIDITY
        ),
        ghi_w_m2=weather["ghi_w_m2"].to_numpy(),
        emissivity=emissivity,
        absorptivity=absorptivity,
    )
    return pd.Series(bundle.subconductor_count * amps, index=weather.index, name="ampacity_a")


__all__ = ["ac_resistance_at", "ampacity", "conductor_ampacity", "validate_weather"]
