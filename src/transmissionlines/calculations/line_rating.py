"""Hourly thermal rating of a whole line: each piece rated in its own weather.

Every piece of the route takes the weather at the location nearest its midpoint,
and the wind angle between that weather's wind direction and the piece's true
bearing. A circuit's rating each hour is the lowest over its pieces.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from transmissionlines.calculations.pieces import WGS84, pieces
from transmissionlines.calculations.thermal import ampacity, conductor_inputs, validate_weather, weather_inputs
from transmissionlines.models.assets import CrossSectionTransmissionLine, RoutedTransmissionLine
from transmissionlines.models.configurations import CircuitConfiguration
from transmissionlines.models.routing import LineSpan, RouteGeometry
from transmissionlines.models.thermal import ThermalRatingResult

WIND_DIRECTION_COLUMN = "wind_direction_deg"


def _span_circuits(span: LineSpan) -> dict[str, CircuitConfiguration]:
    start, end = (
        {circuit.circuit_id: circuit for circuit in tower.configuration.circuits}
        for tower in (span.start_end.start, span.start_end.end)
    )
    if start.keys() != end.keys() or any(
        start[key].conductor_spec != end[key].conductor_spec or start[key].bundle_spec != end[key].bundle_spec
        for key in start
    ):
        raise ValueError(f"span {span.span_id}: towers at its two ends carry different conductors")
    return start


def thermal_rating(
    line: RoutedTransmissionLine | CrossSectionTransmissionLine,
    weather: Mapping[tuple[float, float], pd.DataFrame],
    *,
    route: RouteGeometry | None = None,
    conductor_temperature_c: float = 75.0,
    emissivity: float = 0.8,
    absorptivity: float = 0.8,
) -> ThermalRatingResult:
    """Rate each circuit of a line, hour by hour, from weather along its route.

    Parameters
    ----------
    line : RoutedTransmissionLine or CrossSectionTransmissionLine
        A line with spans, or a representative cross-section plus ``route``.
    weather : mapping of (longitude, latitude) to DataFrame
        Weather tables sharing one UTC index, with the columns
        :func:`~transmissionlines.calculations.thermal.validate_weather` checks and
        ``wind_direction_deg`` (where the wind blows from, clockwise from true north).
    route : RouteGeometry, optional
        The route of a cross-section line, e.g. from HIFLD. Not allowed with spans.
    conductor_temperature_c, emissivity, absorptivity : float
        As in :func:`~transmissionlines.calculations.thermal.conductor_ampacity`.
    """
    if isinstance(line, RoutedTransmissionLine):
        if route is not None:
            raise ValueError("a routed line takes its route from its spans")
        route_pieces = pieces(line)
        circuits = [_span_circuits(piece.span) for piece in route_pieces if piece.span is not None]
    else:
        if route is None:
            raise ValueError("a cross-section line needs a route")
        route_pieces = pieces(route)
        by_id = {circuit.circuit_id: circuit for circuit in line.configuration.circuits}
        circuits = [by_id] * len(route_pieces)
    if not route_pieces:
        raise ValueError("route has no length")
    circuit_ids = list(circuits[0])
    if any(list(by_piece) != circuit_ids for by_piece in circuits):
        raise ValueError("every span must carry the same circuits")

    if not weather:
        raise ValueError("weather is empty")
    locations = list(weather)
    tables = [validate_weather(weather[location]) for location in locations]
    index = tables[0].index
    for location, table in zip(locations, tables):
        if not table.index.equals(index):
            raise ValueError(f"weather at {location} has a different time index")
        if WIND_DIRECTION_COLUMN not in table or not np.isfinite(table[WIND_DIRECTION_COLUMN].to_numpy(float)).all():
            raise ValueError(f"weather at {location} needs finite {WIND_DIRECTION_COLUMN}")

    # ponytail: dense pieces × locations distance matrix; use a KD-tree if grids get large.
    midpoints = np.array([piece.midpoint for piece in route_pieces])
    grid = np.array(locations, dtype=float)
    _, _, distance = WGS84.inv(
        *np.broadcast_arrays(midpoints[:, None, 0], midpoints[:, None, 1], grid[None, :, 0], grid[None, :, 1])
    )
    nearest = np.asarray(distance).argmin(axis=1)
    bearings = np.array([piece.bearing_deg for piece in route_pieces])

    # Rate pieces in groups sharing weather and hardware, keeping a running minimum
    # so memory stays one group × hours rather than pieces × hours.
    groups: dict[tuple[str, int, object, int], list[int]] = {}
    for position, (location, by_piece) in enumerate(zip(nearest, circuits)):
        for circuit_id, circuit in by_piece.items():
            key = (circuit_id, int(location), circuit.conductor_spec.uuid, circuit.bundle_spec.subconductor_count)
            groups.setdefault(key, []).append(position)

    amps = {circuit_id: np.full(len(index), np.inf) for circuit_id in circuit_ids}
    limiting = {circuit_id: np.zeros(len(index), dtype=int) for circuit_id in circuit_ids}
    for (circuit_id, location, _, _), positions in groups.items():
        circuit = circuits[positions[0]][circuit_id]
        table = tables[location]
        members = np.array(positions)
        rated = circuit.bundle_spec.subconductor_count * ampacity(
            **conductor_inputs(circuit.conductor_spec, conductor_temperature_c),
            **weather_inputs(table),
            wind_angle_deg=table[WIND_DIRECTION_COLUMN].to_numpy(float)[None, :] - bearings[members, None],
            emissivity=emissivity,
            absorptivity=absorptivity,
        )
        rated = np.broadcast_to(rated, (len(members), len(index)))
        lowest = rated.argmin(axis=0)
        candidate = rated[lowest, np.arange(len(index))]
        better = candidate < amps[circuit_id]
        amps[circuit_id][better] = candidate[better]
        limiting[circuit_id][better] = members[lowest[better]]

    return ThermalRatingResult(
        line_name=line.name,
        nominal_voltage_kv=line.nominal_voltage_kv,
        conductor_temperature_c=conductor_temperature_c,
        pieces=route_pieces,
        weather_location=[tuple(locations[i]) for i in nearest],
        amps=pd.DataFrame(amps, index=index),
        limiting_piece=pd.DataFrame(limiting, index=index),
    )


__all__ = ["thermal_rating"]
