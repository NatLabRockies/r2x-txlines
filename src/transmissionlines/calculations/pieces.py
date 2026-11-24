"""Split a line's route into straight pieces with true bearings, for thermal rating.

A piece is a straight stretch of route. Its length and direction come from the WGS84
geodesic; the bearing is the forward azimuth at the piece's midpoint, in degrees
clockwise from true north, so it is the direction the conductor runs where the
piece's weather is sampled.

Without ``cells``, each stretch between route vertices is split evenly into pieces
no longer than ``max_length_m``. With ``cells``, each stretch is sampled every
``max_length_m`` and cut only where the sampled cells change, so every piece lies
in one set of weather cells, to within half a sample.
"""

import math
from collections.abc import Callable

import numpy as np
from pyproj import Geod

from transmissionlines.models.assets import RoutedTransmissionLine
from transmissionlines.models.routing import LineSpan, RouteGeometry
from transmissionlines.models.thermal import Piece

WGS84 = Geod(ellps="WGS84")
MAX_PIECE_LENGTH_M = 1000.0
# Sample step for finding weather cell boundaries: 2.4% of the 2.06 km WTK spacing.
CELL_SAMPLE_M = 50.0

# Maps an (n, 2) array of (longitude, latitude) points to an (n, k) array of cell labels.
CellLabeler = Callable[[np.ndarray], np.ndarray]


def _densify(coordinates: list[tuple[float, float]], max_length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Add geodesic points so no stretch is longer than ``max_length_m``.

    Returns the (n, 2) points and a mask of which were route vertices.
    """
    lon, lat = np.asarray(coordinates, dtype=float).T
    _, _, length = WGS84.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dense, vertex = [coordinates[0]], [True]
    for i, stretch in enumerate(np.atleast_1d(length)):
        if (count := math.ceil(stretch / max_length_m)) > 1:
            dense += WGS84.npts(lon[i], lat[i], lon[i + 1], lat[i + 1], count - 1)
            vertex += [False] * (count - 1)
        dense.append(coordinates[i + 1])
        vertex.append(True)
    return np.asarray(dense, dtype=float), np.asarray(vertex)


def _midpoints(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, ...]:
    azimuth, _, length = WGS84.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    mid_lon, mid_lat, back_azimuth = WGS84.fwd(lon[:-1], lat[:-1], azimuth, length / 2)
    return np.asarray(mid_lon), np.asarray(mid_lat), np.asarray(length), (np.asarray(back_azimuth) + 180) % 360


def _route_pieces(
    coordinates: list[tuple[float, float]], span: LineSpan | None, max_length_m: float, cells: CellLabeler | None,
) -> list[Piece]:
    points, keep = _densify(coordinates, max_length_m)
    if cells is not None and len(points) > 2:
        mid_lon, mid_lat, _, _ = _midpoints(points[:, 0], points[:, 1])
        labels = np.asarray(cells(np.column_stack([mid_lon, mid_lat])))
        keep[1:-1] |= (labels[1:] != labels[:-1]).reshape(len(labels) - 1, -1).any(axis=1)
        points = points[keep]
    lon, lat = points.T
    mid_lon, mid_lat, length, bearing = _midpoints(lon, lat)
    return [
        Piece(
            start=(float(lon[i]), float(lat[i])),
            end=(float(lon[i + 1]), float(lat[i + 1])),
            midpoint=(float(mid_lon[i]), float(mid_lat[i])),
            length_m=float(length[i]),
            bearing_deg=float(bearing[i]),
            span=span,
        )
        for i in range(len(length))
        if length[i] > 0  # repeated vertices have no direction
    ]


def pieces(
    line: RoutedTransmissionLine | RouteGeometry,
    *,
    max_length_m: float | None = None,
    cells: CellLabeler | None = None,
) -> list[Piece]:
    """Return the straight pieces of a routed line's spans, or of a bare route.

    Parameters
    ----------
    line : RoutedTransmissionLine or RouteGeometry
        A line with towers and spans, or a route with no towers (e.g. from HIFLD).
    max_length_m : float, optional
        Without ``cells``, the longest piece (default ``MAX_PIECE_LENGTH_M``). With
        ``cells``, the sample step (default ``CELL_SAMPLE_M``).
    cells : callable, optional
        Labels points with the weather cells they fall in; pieces are cut where the
        labels change.

    Returns
    -------
    list of Piece
        Pieces in route order. Their lengths add up to the route's geodesic length.
    """
    if max_length_m is None:
        max_length_m = MAX_PIECE_LENGTH_M if cells is None else CELL_SAMPLE_M
    if isinstance(line, RouteGeometry):
        return _route_pieces(line.coordinates, None, max_length_m, cells)
    return [
        piece
        for span in line.spans
        for piece in _route_pieces(span.route_geometry.coordinates, span, max_length_m, cells)
    ]


def weather_points(line: RoutedTransmissionLine | RouteGeometry) -> list[tuple[float, float]]:
    """Return points every ``CELL_SAMPLE_M`` along a route, to fetch weather for.

    These are the points :func:`~transmissionlines.calculations.line_rating.thermal_rating`
    samples, so weather fetched at them covers every cell the route passes through.
    """
    return [piece.midpoint for piece in pieces(line, max_length_m=CELL_SAMPLE_M)]


__all__ = ["CELL_SAMPLE_M", "MAX_PIECE_LENGTH_M", "Piece", "pieces", "weather_points"]
