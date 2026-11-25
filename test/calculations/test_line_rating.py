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


def _cross_section(route: RouteGeometry | None = None) -> CrossSectionTransmissionLine:
    return CrossSectionTransmissionLine(
        name="bare", nominal_voltage_kv=230, nominal_frequency_hz=60, earth_resistivity_ohm_m=100,
        configuration=_rated_configuration(), route=route,
    )


def _windy(rows: int = 3, **overrides) -> pd.DataFrame:
    return _weather(rows, **{"wind_direction_deg": 0.0, **overrides})


def _limiting_bearings(result) -> np.ndarray:
    return np.array([result.pieces[i].bearing_deg for i in result.limiting_piece["c1"]])


def test_one_piece_matches_the_conductor_rating_at_its_wind_angle() -> None:
    route = RouteGeometry(coordinates=[(0, 0), (0.005, 0.005)])
    (piece,) = pieces(route)
    weather = _windy(wind_direction_deg=[0.0, 90.0, 200.0])
    result = thermal_rating(_cross_section(route), {(0, 0): weather})
    expected = conductor_ampacity(
        _conductor(), _cross_section().configuration.circuits[0].bundle_spec, weather,
        wind_angle_deg=weather["wind_direction_deg"] - piece.bearing_deg,
    )
    np.testing.assert_allclose(result.amps["c1"], expected)


def test_piece_running_along_the_wind_limits_the_line() -> None:
    # A north wind crosses the east-running piece and runs along the north-running one.
    result = thermal_rating(_cross_section(_ELBOW), {(0, 0): _windy()})
    np.testing.assert_allclose(_limiting_bearings(result), 0, atol=1e-6)
    assert set(result.weather_locations) == {((0, 0),)}
    assert len(result.pieces) == 2  # one weather cell: no cuts beyond the bend


def test_each_piece_takes_its_nearest_weather() -> None:
    east = RouteGeometry(coordinates=[(0, 0), (0.1, 0)])
    west_cell, east_cell = (0.02, 0.01), (0.08, 0.01)
    hot, cool = _windy(air_temperature_k=313.15), _windy()
    result = thermal_rating(_cross_section(east), {west_cell: cool, east_cell: hot})
    assert (result.limiting_piece["c1"] == 1).all()
    swapped = thermal_rating(_cross_section(east), {west_cell: hot, east_cell: cool})
    assert (swapped.limiting_piece["c1"] == 0).all()
    np.testing.assert_allclose(swapped.amps, result.amps)


def test_route_is_cut_where_the_nearest_weather_changes() -> None:
    east = RouteGeometry(coordinates=[(0, 0), (0.1, 0)])
    west_cell, east_cell = (0.02, 0.01), (0.08, 0.01)
    result = thermal_rating(_cross_section(east), {west_cell: _windy(), east_cell: _windy()})
    assert [loc for (loc,) in result.weather_locations] == [west_cell, east_cell]
    assert abs(result.pieces[0].end[0] - 0.05) * 111_320 < 50


def test_each_source_is_looked_up_separately() -> None:
    # Wind from one cell, sunlight from two: the sunnier half limits the line.
    east = RouteGeometry(coordinates=[(0, 0), (0.1, 0)])
    wind = {(0.05, 0): _windy().drop(columns="ghi_w_m2")}
    shade, sun = _weather()[["ghi_w_m2"]] * 0, _weather()[["ghi_w_m2"]]
    result = thermal_rating(_cross_section(east), [wind, {(0.02, 0): shade, (0.08, 0): sun}])
    assert len(result.pieces) == 2
    assert (result.limiting_piece["c1"] == 1).all()
    assert result.weather_locations[1] == ((0.05, 0), (0.08, 0))
    with pytest.raises(ValueError, match="more than one source"):
        thermal_rating(_cross_section(east), [{(0, 0): _windy()}, {(0, 0): sun}])
    with pytest.raises(ValueError, match="more than one source"):  # shared by two of three
        thermal_rating(_cross_section(east), [wind, {(0, 0): sun}, {(0, 0): sun}])


def test_routed_line_uses_its_spans_and_reports_mva() -> None:
    line, _, _ = _routed_line(configuration=_rated_configuration())
    result = thermal_rating(line, {(-107.5, 45): _windy()})
    assert all(piece.span is line.spans[0] for piece in result.pieces)
    np.testing.assert_allclose(result.mva, result.amps * math.sqrt(3) * 230 / 1000)


def test_route_and_weather_are_checked() -> None:
    with pytest.raises(ValueError, match="needs a route"):
        thermal_rating(_cross_section(), {(0, 0): _windy()})
    with pytest.raises(ValueError, match="wind_direction_deg"):
        thermal_rating(_cross_section(_ELBOW), {(0, 0): _weather()})
    for bad in (80.0, float("nan")):
        with pytest.raises(ValueError, match="fractions between 0 and 1"):
            thermal_rating(_cross_section(_ELBOW), {(0, 0): _windy()}, emissivity=bad)
    with pytest.raises(ValueError, match="different time index"):
        thermal_rating(_cross_section(_ELBOW), {(0, 0): _windy(3), (1, 1): _windy(4)})
