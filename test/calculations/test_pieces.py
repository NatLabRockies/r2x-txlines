import pytest
from pyproj import Geod

from transmissionlines.calculations.pieces import pieces
from transmissionlines.models.routing import RouteGeometry

from test.integration.test_calculation_input_graph import _routed_line


def test_bearings_are_true_azimuths() -> None:
    north, east, south, west = pieces(RouteGeometry(coordinates=[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]))
    assert north.bearing_deg == pytest.approx(0, abs=1e-9)
    assert east.bearing_deg == pytest.approx(90, abs=1e-3)
    assert south.bearing_deg == pytest.approx(180, abs=1e-9)
    assert west.bearing_deg == pytest.approx(270, abs=1e-9)


def test_bearing_is_taken_at_the_midpoint() -> None:
    # Due east along 60°N, the geodesic starts north of east and ends south of it;
    # by symmetry it runs exactly east at its midpoint.
    (piece,) = pieces(RouteGeometry(coordinates=[(0, 60), (10, 60)]))
    assert piece.bearing_deg == pytest.approx(90, abs=1e-9)
    assert piece.midpoint[0] == pytest.approx(5)


def test_length_is_the_wgs84_geodesic() -> None:
    (piece,) = pieces(RouteGeometry(coordinates=[(0, 0), (0, 1)]))
    assert piece.length_m == pytest.approx(110_574.389, abs=0.01)  # one degree of meridian at the equator


def test_repeated_vertices_are_dropped() -> None:
    route = pieces(RouteGeometry(coordinates=[(0, 0), (0, 0), (0, 1)]))
    assert len(route) == 1


def test_span_pieces_cover_the_line_and_keep_their_span() -> None:
    line, _, _ = _routed_line()
    span = line.spans[0]
    bent = span.model_copy(update={"route_geometry": RouteGeometry(coordinates=[(-108, 45), (-107.5, 45.1), (-107, 45)])})
    line = line.model_copy(update={"spans": [bent]})

    route = pieces(line)
    assert [piece.span for piece in route] == [bent, bent]
    assert route[0].end == route[1].start
    lons, lats = zip(*bent.route_geometry.coordinates)
    assert sum(piece.length_m for piece in route) == pytest.approx(Geod(ellps="WGS84").line_length(lons, lats))
    assert all(piece.span is None for piece in pieces(bent.route_geometry))
