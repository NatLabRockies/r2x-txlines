"""Hourly weather for thermal ratings from NREL's WIND Toolkit (WTK) and NSRDB.

Air temperature and wind at 10 m, surface pressure and relative humidity come from
WTK CONUS; global horizontal irradiance from the NSRDB GOES aggregated v4 grid,
taking the NSRDB cell nearest each WTK cell and its on-the-hour values. Data are
read over NREL's HSDS service, which needs the ``weather`` extra and an
``~/.hscfg`` holding an NREL API key; a path that exists on disk (e.g. on an HPC
system) is read directly instead.

Grids and every cell-year read are cached under ``cache_dir``, so a second request
for the same cells and years makes no network calls.
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transmissionlines.calculations.thermal import C_TO_K

WTK_PATH = "/nrel/wtk/conus/wtk_conus_{year}.h5"
NSRDB_PATH = "/nrel/nsrdb/GOES/aggregated/v4.0.0/nsrdb_{year}.h5"
# Weather column: (WTK dataset, scale, offset). WTK stores integers × 100 and its files
# carry no scale attributes: °C, m/s, degrees from true north, kPa and percent.
_WTK = {
    "air_temperature_k": ("temperature_10m", 0.01, C_TO_K),
    "wind_speed_m_s": ("windspeed_10m", 0.01, 0.0),
    "wind_direction_deg": ("winddirection_10m", 0.01, 0.0),
    "pressure_pa": ("pressure_0m", 10.0, 0.0),
    "relative_humidity": ("relativehumidity_2m", 1e-4, 0.0),
}
_EARTH_RADIUS_KM = 6371.0
# Grid spacing is 2 km (WTK) and 4 km (NSRDB); anything farther is off the grid.
_MAX_CELL_DISTANCE_KM = 20.0


def default_cache_dir() -> Path:
    """Return ``$TRANSMISSIONLINES_CACHE/weather``, or ``~/.cache/transmissionlines/weather``."""
    root = os.environ.get("TRANSMISSIONLINES_CACHE", Path.home() / ".cache" / "transmissionlines")
    return Path(root) / "weather"


def _open(path: str) -> Any:
    if Path(path).exists():
        import h5py

        return h5py.File(path, "r")
    try:
        import h5pyd
    except ImportError as error:
        raise ImportError(
            "reading NREL weather over HSDS needs the 'weather' extra: pip install 'transmissionlines[weather]'"
        ) from error
    return h5pyd.File(path, "r")


def _grid(source: str, path: str, year: int, cache_dir: Path) -> np.ndarray:
    """Return the cell centers of a dataset as (latitude, longitude) rows."""
    file = cache_dir / f"{source}_grid.npy"
    if file.exists():
        return np.load(file)
    with _open(path.format(year=year)) as f:
        if "coordinates" in f:
            grid = f["coordinates"][...]  # WTK; its `meta` times out over HSDS (DLR issue #3)
        else:
            meta = f["meta"][...]
            grid = np.column_stack([meta["latitude"], meta["longitude"]])
    grid = np.asarray(grid, dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(file, grid)
    return grid


def _unit_vectors(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _nearest(grid: np.ndarray, points: np.ndarray, name: str) -> np.ndarray:
    """Return the index of the grid cell nearest each (longitude, latitude) point."""
    lon, lat = points[:, 0], points[:, 1]
    # Wider than _MAX_CELL_DISTANCE_KM at CONUS latitudes (0.5° of longitude ≥ 35 km below 50°N),
    # so every cell close enough to accept is inside the box.
    margin = 0.5
    candidates = np.flatnonzero(
        (grid[:, 0] >= lat.min() - margin) & (grid[:, 0] <= lat.max() + margin)
        & (grid[:, 1] >= lon.min() - margin) & (grid[:, 1] <= lon.max() + margin)
    )
    cells = _unit_vectors(grid[candidates, 0].astype(float), grid[candidates, 1].astype(float))
    wanted = _unit_vectors(lat, lon)
    nearest = np.empty(len(points), dtype=int)
    best = np.full(len(points), -1.0)
    if candidates.size:
        rows = max(1, 10_000_000 // candidates.size)
        for start in range(0, len(points), rows):
            dots = wanted[start:start + rows] @ cells.T
            nearest[start:start + rows] = dots.argmax(axis=1)
            best[start:start + rows] = dots.max(axis=1)
    distance_km = _EARTH_RADIUS_KM * np.arccos(np.clip(best, -1, 1))
    if (far := distance_km > _MAX_CELL_DISTANCE_KM).any():
        lon_far, lat_far = points[far][0]
        raise ValueError(f"({lon_far}, {lat_far}) is not within {_MAX_CELL_DISTANCE_KM} km of a {name} cell")
    return candidates[nearest]


def _cells(
    source: str, path: str, datasets: list[str], cells: Iterable[int], year: int, cache_dir: Path,
) -> dict[int, pd.DataFrame]:
    """Return raw hourly values per cell for one year, reading only uncached cells."""
    folder = cache_dir / source / str(year)
    frames, missing = {}, []
    for cell in sorted(set(cells)):
        file = folder / f"{cell}.parquet"
        if file.exists():
            frames[cell] = pd.read_parquet(file)
        else:
            missing.append(cell)
    if missing:
        with _open(path.format(year=year)) as f:
            times = pd.to_datetime(f["time_index"][...].astype(str), utc=True)
            raw = {name: f[name][:, missing] for name in datasets}
        folder.mkdir(parents=True, exist_ok=True)
        for column, cell in enumerate(missing):
            frames[cell] = pd.DataFrame({name: raw[name][:, column] for name in datasets}, index=times)
            frames[cell].to_parquet(folder / f"{cell}.parquet")
    return frames


def fetch_weather(
    points: Iterable[tuple[float, float]],
    years: int | Iterable[int],
    *,
    cache_dir: str | Path | None = None,
    wtk_path: str = WTK_PATH,
    nsrdb_path: str = NSRDB_PATH,
) -> dict[tuple[float, float], pd.DataFrame]:
    """Return hourly weather at the WTK cells nearest the given points.

    Parameters
    ----------
    points : iterable of (longitude, latitude)
        Places needing weather, e.g. the midpoints of a line's pieces.
    years : int or iterable of int
        Calendar years; WTK CONUS covers 2007–2014.
    cache_dir : path, optional
        Defaults to :func:`default_cache_dir`.
    wtk_path, nsrdb_path : str
        HSDS domains or local files, with ``{year}`` in place of the year.

    Returns
    -------
    dict of (longitude, latitude) to DataFrame
        One table per WTK cell used, keyed by the cell center, ready for
        :func:`~transmissionlines.calculations.line_rating.thermal_rating`.
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    years = [years] if isinstance(years, int) else list(years)
    wanted = np.asarray(list(points), dtype=float).reshape(-1, 2)
    if not years or not len(wanted):
        raise ValueError("need at least one point and one year")

    wtk_grid = _grid("wtk", wtk_path, years[0], cache)
    cells = np.unique(_nearest(wtk_grid, wanted, "WTK"))
    centers = wtk_grid[cells][:, ::-1].astype(float)
    sunlit = _nearest(_grid("nsrdb", nsrdb_path, years[0], cache), centers, "NSRDB")

    tables: dict[int, list[pd.DataFrame]] = {int(cell): [] for cell in cells}
    for year in years:
        wind = _cells("wtk", wtk_path, [dataset for dataset, _, _ in _WTK.values()], cells, year, cache)
        sun = _cells("nsrdb", nsrdb_path, ["ghi"], sunlit, year, cache)
        for cell, sun_cell in zip(cells, sunlit):
            raw = wind[int(cell)]
            ghi = sun[int(sun_cell)]["ghi"].iloc[::2]  # NSRDB is half-hourly; keep the hours
            if not ghi.index.equals(raw.index):
                raise ValueError(f"WTK and NSRDB hours differ in {year}")
            table = pd.DataFrame(
                {column: raw[dataset] * scale + offset for column, (dataset, scale, offset) in _WTK.items()},
                index=raw.index,
            )
            # WRF humidity can overshoot saturation slightly.
            table["relative_humidity"] = table["relative_humidity"].clip(0, 1)
            table["ghi_w_m2"] = ghi.astype(float)
            tables[int(cell)].append(table)
    return {
        (float(lon), float(lat)): pd.concat(tables[int(cell)])
        for cell, (lon, lat) in zip(cells, centers)
    }


__all__ = ["NSRDB_PATH", "WTK_PATH", "default_cache_dir", "fetch_weather"]
