"""Route pieces and hourly thermal rating results."""

import math
from dataclasses import dataclass

import pandas as pd
from pydantic import ConfigDict

from transmissionlines.models.result_base import CalculationResultModel
from transmissionlines.models.routing import LineSpan


@dataclass(frozen=True)
class Piece:
    """One straight stretch of a route.

    Coordinates are ``(longitude, latitude)`` in degrees. ``bearing_deg`` is the
    geodesic forward azimuth at the midpoint, clockwise from true north. ``span`` is
    the span the piece belongs to, or ``None`` for a bare route, which has no towers
    and so no sag.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    midpoint: tuple[float, float]
    length_m: float
    bearing_deg: float
    span: LineSpan | None = None


class ThermalRatingResult(CalculationResultModel):
    """Hourly thermal ratings of a line's circuits, the minimum over its pieces.

    ``amps`` and ``limiting_piece`` share a UTC index and have one column per
    circuit. ``amps`` is the whole bundle's rating. ``limiting_piece`` gives the
    position in ``pieces`` of the piece that set each hour's rating, and
    ``weather_locations`` the location each piece used from each weather source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    line_name: str
    nominal_voltage_kv: float
    conductor_temperature_c: float
    pieces: list[Piece]
    weather_locations: list[tuple[tuple[float, float], ...]]
    amps: pd.DataFrame
    limiting_piece: pd.DataFrame

    @property
    def mva(self) -> pd.DataFrame:
        """Three-phase rating, ``sqrt(3) · kV · A`` in MVA."""
        return self.amps * math.sqrt(3) * self.nominal_voltage_kv / 1000


__all__ = ["Piece", "ThermalRatingResult"]
