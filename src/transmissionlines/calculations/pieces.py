"""Split a line's route into straight pieces with true bearings, for thermal rating.

A piece is one straight stretch between consecutive route vertices. Its length and
direction come from the WGS84 geodesic. The bearing is the forward azimuth at the
piece's midpoint, in degrees clockwise from true north, so it is the direction the
conductor actually runs where the piece's weather is sampled.
"""

import numpy as np
from pyproj import Geod

from transmissionlines.models.assets import RoutedTransmissionLine
from transmissionlines.models.routing import LineSpan, RouteGeometry
from transmissionlines.models.thermal import Piece

WGS84 = Geod(ellps="WGS84")


def _route_pieces(coordinates: list[tuple[float, float]], span: LineSpan | None) -> list[Piece]:
    lon, lat = np.asarray(coordinates, dtype=float).T
    azimuth, _, length = WGS84.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    mid_lon, mid_lat, back_azimuth = WGS84.fwd(lon[:-1], lat[:-1], azimuth, length / 2)
    bearing = (np.asarray(back_azimuth) + 180) % 360
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


def pieces(line: RoutedTransmissionLine | RouteGeometry) -> list[Piece]:
    """Return the straight pieces of a routed line's spans, or of a bare route.

    Parameters
    ----------
    line : RoutedTransmissionLine or RouteGeometry
        A line with towers and spans, or a route with no towers (e.g. from HIFLD).

    Returns
    -------
    list of Piece
        Pieces in route order. Their lengths add up to the route's geodesic length.
    """
    if isinstance(line, RouteGeometry):
        return _route_pieces(line.coordinates, None)
    return [
        piece
        for span in line.spans
        for piece in _route_pieces(span.route_geometry.coordinates, span)
    ]


__all__ = ["Piece", "pieces"]
