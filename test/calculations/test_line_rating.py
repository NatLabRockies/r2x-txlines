import math

import numpy as np
import pandas as pd
import pytest

from transmissionlines.calculations.line_rating import thermal_rating
from transmissionlines.calculations.pieces import pieces
from transmissionlines.calculations.thermal import conductor_ampacity
from transmissionlines.models.assets import CrossSectionTransmissionLine
from transmissionlines.models.routing import RouteGeometry

from test.calculations.test_thermal import _conductor, _weather
from test.integration.test_calculation_input_graph import _routed_line
from test.models.test_input_graph import _configuration

# East along the equator, then north.
_ELBOW = RouteGeometry(coordinates=[(0, 0), (0.1, 0), (0.1, 0.1)])


def _rated_configuration():
    configuration = _configuration()
    circuit = configuration.circuits[0].model_copy(update={"conductor_spec": _conductor()})
    return configuration.model_copy(update={"circuits": [circuit]})


def _cross_section() -> CrossSectionTransmissionLine:
    return CrossSectionTransmissionLine(
        name="bare", nominal_voltage_kv=230, nominal_frequency_hz=60, earth_resistivity_ohm_m=100,
        configuration=_rated_configuration(),
    )


def _windy(rows: int = 3, **overrides) -> pd.DataFrame:
    return _weather(rows, **{"wind_direction_deg": 0.0, **overrides})


def test_one_piece_matches_the_conductor_rating_at_its_wind_angle() -> None:
    route = RouteGeometry(coordinates=[(0, 0), (0.1, 0.1)])
    (piece,) = pieces(route)
    weather = _windy(wind_direction_deg=[0.0, 90.0, 200.0])
    result = thermal_rating(_cross_section(), {(0, 0): weather}, route=route)
    expected = conductor_ampacity(
        _conductor(), _cross_section().configuration.circuits[0].bundle_spec, weather,
        wind_angle_deg=weather["wind_direction_deg"] - piece.bearing_deg,
    )
    np.testing.assert_allclose(result.amps["c1"], expected)


def test_piece_running_along_the_wind_limits_the_line() -> None:
    # A north wind crosses the east-running piece and runs along the north-running one.
    result = thermal_rating(_cross_section(), {(0, 0): _windy()}, route=_ELBOW)
    assert (result.limiting_piece["c1"] == 1).all()
    assert result.weather_location == [(0, 0), (0, 0)]


def test_each_piece_takes_its_nearest_weather() -> None:
    hot, cool = _windy(wind_direction_deg=45.0, air_temperature_k=313.15), _windy(wind_direction_deg=45.0)
    near_start, near_end = (0.05, 0), (0.1, 0.05)
    result = thermal_rating(_cross_section(), {near_start: cool, near_end: hot}, route=_ELBOW)
    assert result.weather_location == [near_start, near_end]
    assert (result.limiting_piece["c1"] == 1).all()
    swapped = thermal_rating(_cross_section(), {near_start: hot, near_end: cool}, route=_ELBOW)
    assert (swapped.limiting_piece["c1"] == 0).all()
    np.testing.assert_allclose(swapped.amps, result.amps)  # 45° wind meets both pieces alike


def test_routed_line_uses_its_spans_and_reports_mva() -> None:
    line, _, _ = _routed_line(configuration=_rated_configuration())
    result = thermal_rating(line, {(-107.5, 45): _windy()})
    assert [piece.span for piece in result.pieces] == line.spans
    np.testing.assert_allclose(result.mva, result.amps * math.sqrt(3) * 230 / 1000)


def test_route_and_weather_are_checked() -> None:
    line, _, _ = _routed_line(configuration=_rated_configuration())
    with pytest.raises(ValueError, match="from its spans"):
        thermal_rating(line, {(0, 0): _windy()}, route=_ELBOW)
    with pytest.raises(ValueError, match="needs a route"):
        thermal_rating(_cross_section(), {(0, 0): _windy()})
    with pytest.raises(ValueError, match="wind_direction_deg"):
        thermal_rating(_cross_section(), {(0, 0): _weather()}, route=_ELBOW)
    with pytest.raises(ValueError, match="different time index"):
        thermal_rating(_cross_section(), {(0, 0): _windy(3), (1, 1): _windy(4)}, route=_ELBOW)
