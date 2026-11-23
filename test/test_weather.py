from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transmissionlines.calculations.line_rating import thermal_rating
from transmissionlines.calculations.pieces import pieces
from transmissionlines.models.routing import RouteGeometry
from transmissionlines.weather import fetch_weather

h5py = pytest.importorskip("h5py")

from test.calculations.test_line_rating import _cross_section  # noqa: E402

HOURS = 48
# WTK cells 0.02° apart, NSRDB cells 0.04° apart, around (-105, 40).
WTK_LAT, WTK_LON = np.meshgrid(np.arange(40, 40.2, 0.02), np.arange(-105, -104.8, 0.02), indexing="ij")
SUN_LAT, SUN_LON = np.meshgrid(np.arange(40, 40.2, 0.04), np.arange(-105, -104.8, 0.04), indexing="ij")


def _write_files(folder: Path) -> tuple[str, str]:
    times = pd.date_range("2012-01-01", periods=2 * HOURS, freq="30min", tz="UTC")
    cells = WTK_LAT.size
    with h5py.File(folder / "wtk_2012.h5", "w") as f:
        f["coordinates"] = np.column_stack([WTK_LAT.ravel(), WTK_LON.ravel()]).astype(np.float32)
        f["time_index"] = times[::2].strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype("S20")
        cell = np.arange(cells)[None, :]
        hour = np.arange(HOURS)[:, None]
        f["temperature_10m"] = (2000 + cell + 0 * hour).astype(np.int16)  # 20 °C + cell/100
        f["windspeed_10m"] = np.full((HOURS, cells), 350, np.uint16)
        f["winddirection_10m"] = (np.zeros((HOURS, cells)) + 100 * (hour % 360)).astype(np.uint16)
        f["pressure_0m"] = np.full((HOURS, cells), 8400, np.uint16)
        f["relativehumidity_2m"] = np.full((HOURS, cells), 10_050, np.uint16)  # 100.5 %
    with h5py.File(folder / "nsrdb_2012.h5", "w") as f:
        meta = np.zeros(SUN_LAT.size, dtype=[("latitude", "f4"), ("longitude", "f4")])
        meta["latitude"], meta["longitude"] = SUN_LAT.ravel(), SUN_LON.ravel()
        f["meta"] = meta
        f["time_index"] = times.strftime("%Y-%m-%d %H:%M:%S+00:00").to_numpy().astype("S25")
        # On the hour: 100 × hour; half past: an obviously wrong 9999.
        half_hours = np.where(np.arange(2 * HOURS) % 2, 9999, 50 * np.arange(2 * HOURS))
        f["ghi"] = np.repeat(half_hours[:, None], SUN_LAT.size, axis=1).astype(np.uint16)
    return str(folder / "wtk_{year}.h5"), str(folder / "nsrdb_{year}.h5")


def _fetch(tmp_path: Path, points, **kwargs) -> dict:
    wtk, nsrdb = _write_files(tmp_path)
    return fetch_weather(points, 2012, cache_dir=tmp_path / "cache", wtk_path=wtk, nsrdb_path=nsrdb, **kwargs)


def test_weather_comes_in_si_units_from_the_nearest_cell(tmp_path: Path) -> None:
    weather = _fetch(tmp_path, [(-104.961, 40.041), (-104.959, 40.039)])
    ((lon, lat), table), = weather.items()  # both points share one cell
    assert (lon, lat) == pytest.approx((-104.96, 40.04))
    cell = int(np.flatnonzero(np.isclose(WTK_LAT.ravel(), 40.04) & np.isclose(WTK_LON.ravel(), -104.96))[0])
    assert table["air_temperature_k"].iloc[0] == pytest.approx(293.15 + cell / 100)
    assert table["pressure_pa"].iloc[0] == pytest.approx(84_000)
    assert table["wind_speed_m_s"].iloc[0] == pytest.approx(3.5)
    assert table["wind_direction_deg"].iloc[3] == pytest.approx(3)
    assert (table["relative_humidity"] == 1).all()
    np.testing.assert_allclose(table["ghi_w_m2"], 100 * np.arange(HOURS))
    assert str(table.index.tz) == "UTC" and len(table) == HOURS


def test_second_fetch_reads_only_the_cache(tmp_path: Path) -> None:
    first = _fetch(tmp_path, [(-104.95, 40.05)])
    for file in tmp_path.glob("*.h5"):
        file.unlink()
    again = fetch_weather(
        [(-104.95, 40.05)], 2012, cache_dir=tmp_path / "cache",
        wtk_path=str(tmp_path / "wtk_{year}.h5"), nsrdb_path=str(tmp_path / "nsrdb_{year}.h5"),
    )
    for key in first:
        pd.testing.assert_frame_equal(again[key], first[key])


def test_points_off_the_grid_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not within"):
        _fetch(tmp_path, [(-110, 40.05)])


def test_route_goes_to_a_rating_with_no_manual_steps(tmp_path: Path) -> None:
    route = RouteGeometry(coordinates=[(-104.99, 40.01), (-104.91, 40.01), (-104.91, 40.15)])
    weather = _fetch(tmp_path, [piece.midpoint for piece in pieces(route)])
    result = thermal_rating(_cross_section(), weather, route=route)
    assert len(weather) == 2
    assert (result.amps["c1"] > 0).all()
