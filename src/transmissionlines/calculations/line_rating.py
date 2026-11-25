"""Hourly thermal rating of a whole line: each piece rated in its own weather.

Weather comes from one or more sources, each a mapping of locations to tables
(e.g. WTK wind and temperature on one grid, NSRDB sunlight on another). The route is
cut wherever the nearest location of any source changes, so each piece lies in one
cell of every source. A piece takes the wind angle between its weather's wind
direction and its true bearing, and a circuit's rating each hour is the lowest over
its pieces.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from transmissionlines.calculations.pieces import WGS84, pieces
from transmissionlines.calculations.thermal import ampacity, conductor_inputs, validate_weather, weather_inputs
from transmissionlines.models.assets import CrossSectionTransmissionLine, RoutedTransmissionLine
from transmissionlines.models.configurations import CircuitConfiguration
from transmissionlines.models.routing import LineSpan
from transmissionlines.models.thermal import ThermalRatingResult

WIND_DIRECTION_COLUMN = "wind_direction_deg"
Location = tuple[float, float]
WeatherSource = Mapping[Location, pd.DataFrame]


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


def _nearest(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return the index of the grid location nearest each (longitude, latitude) point."""
    # ponytail: dense points × locations distances; use a KD-tree if sources get large.
    _, _, distance = WGS84.inv(
        *np.broadcast_arrays(points[:, None, 0], points[:, None, 1], grid[None, :, 0], grid[None, :, 1])
    )
    return np.asarray(distance).argmin(axis=1)


def thermal_rating(
    line: RoutedTransmissionLine | CrossSectionTransmissionLine,
    weather: WeatherSource | Sequence[WeatherSource],
    *,
    conductor_temperature_c: float = 75.0,
    emissivity: float = 0.8,
    absorptivity: float = 0.8,
) -> ThermalRatingResult:
    """Rate each circuit of a line, hour by hour, from weather along its route.

    Parameters
    ----------
    line : RoutedTransmissionLine or CrossSectionTransmissionLine
        A line with spans, or a representative cross-section with a ``route``.
    weather : mapping, or sequence of mappings, of (longitude, latitude) to DataFrame
        Weather tables sharing one UTC index. Together, the sources give the columns
        :func:`~transmissionlines.calculations.thermal.validate_weather` checks and
        ``wind_direction_deg`` (where the wind blows from, clockwise from true north),
        each column from exactly one source. Each source should have a location in
        every cell the route crosses, e.g. by fetching at
        :func:`~transmissionlines.calculations.pieces.weather_points`.
    conductor_temperature_c, emissivity, absorptivity : float
        As in :func:`~transmissionlines.calculations.thermal.conductor_ampacity`.
    """
    sources = [weather] if isinstance(weather, Mapping) else list(weather)
    if not sources or not all(sources):
        raise ValueError("weather is empty")
    locations = [list(source) for source in sources]
    grids = [np.array(keys, dtype=float) for keys in locations]
    # Every table must share one hourly index; take it from the first and check the rest.
    first_table = next(iter(sources[0].values()))
    hours = first_table.index
    for source in sources:
        for location, table in source.items():
            if not table.index.equals(hours):
                raise ValueError(f"weather at {location} has a different time index")
    # Each column must come from exactly one source (a source's tables share columns).
    column_counts = Counter(column for source in sources for column in next(iter(source.values())).columns)
    if shared := sorted(column for column, count in column_counts.items() if count > 1):
        raise ValueError(f"weather columns {shared} come from more than one source")

    def cells(points: np.ndarray) -> np.ndarray:
        return np.column_stack([_nearest(points, grid) for grid in grids])

    if isinstance(line, RoutedTransmissionLine):
        route_pieces = pieces(line, cells=cells)
        circuits = [_span_circuits(piece.span) for piece in route_pieces if piece.span is not None]
    else:
        if line.route is None:
            raise ValueError("a cross-section line needs a route")
        route_pieces = pieces(line.route, cells=cells)
        by_id = {circuit.circuit_id: circuit for circuit in line.configuration.circuits}
        circuits = [by_id] * len(route_pieces)
    if not route_pieces:
        raise ValueError("route has no length")
    circuit_ids = list(circuits[0])
    if any(list(by_piece) != circuit_ids for by_piece in circuits):
        raise ValueError("every span must carry the same circuits")

    labels = [tuple(int(i) for i in row) for row in cells(np.array([piece.midpoint for piece in route_pieces]))]
    bearings = np.array([piece.bearing_deg for piece in route_pieces])
    tables: dict[tuple[int, ...], pd.DataFrame] = {}

    def table_for(label: tuple[int, ...]) -> pd.DataFrame:
        if label not in tables:
            table = pd.concat([sources[s][locations[s][i]] for s, i in enumerate(label)], axis=1)
            if WIND_DIRECTION_COLUMN not in table or not np.isfinite(table[WIND_DIRECTION_COLUMN].to_numpy(float)).all():
                raise ValueError(f"weather needs finite {WIND_DIRECTION_COLUMN}")
            tables[label] = validate_weather(table)
        return tables[label]

    # Pieces of one circuit with the same weather cells and the same conductor/bundle
    # differ only in bearing, so one vectorized ampacity call rates them all. Group by
    # (circuit_id, weather cell label, conductor uuid, subconductor count); the last two
    # matter because a circuit's conductor can change from span to span. Keeping a
    # running minimum across groups holds memory to one group × hours, not pieces × hours.
    groups: dict[tuple, list[int]] = {}
    for position, (label, by_piece) in enumerate(zip(labels, circuits)):
        for circuit_id, circuit in by_piece.items():
            key = (circuit_id, label, circuit.conductor_spec.uuid, circuit.bundle_spec.subconductor_count)
            groups.setdefault(key, []).append(position)

    amps = {circuit_id: np.full(len(hours), np.inf) for circuit_id in circuit_ids}
    limiting = {circuit_id: np.zeros(len(hours), dtype=int) for circuit_id in circuit_ids}
    for (circuit_id, label, _, _), positions in groups.items():
        circuit = circuits[positions[0]][circuit_id]
        table = table_for(label)
        members = np.array(positions)
        rated = circuit.bundle_spec.subconductor_count * ampacity(
            **conductor_inputs(circuit.conductor_spec, conductor_temperature_c),
            **weather_inputs(table),
            wind_angle_deg=table[WIND_DIRECTION_COLUMN].to_numpy(float)[None, :] - bearings[members, None],
            emissivity=emissivity,
            absorptivity=absorptivity,
        )
        rated = np.broadcast_to(rated, (len(members), len(hours)))
        # Ties: argmin takes the earliest piece in the group, and strict < keeps the
        # group seen first. Arbitrary but deterministic; tied pieces rate the same anyway.
        lowest = rated.argmin(axis=0)
        candidate = rated[lowest, np.arange(len(hours))]
        better = candidate < amps[circuit_id]
        amps[circuit_id][better] = candidate[better]
        limiting[circuit_id][better] = members[lowest[better]]

    return ThermalRatingResult(
        line_name=line.name,
        nominal_voltage_kv=line.nominal_voltage_kv,
        conductor_temperature_c=conductor_temperature_c,
        pieces=route_pieces,
        weather_locations=[
            tuple(tuple(locations[s][i]) for s, i in enumerate(label)) for label in labels
        ],
        amps=pd.DataFrame(amps, index=hours),
        limiting_piece=pd.DataFrame(limiting, index=hours),
    )


__all__ = ["thermal_rating"]
