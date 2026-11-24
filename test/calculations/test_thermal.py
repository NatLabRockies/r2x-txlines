import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from transmissionlines.builders.line import conductor_from_record
from transmissionlines.calculations.thermal import ac_resistance_at, ampacity, conductor_ampacity
from transmissionlines.catalog.schemas import ConductorRecord
from transmissionlines.models.cables import (
    BareConductorEquipment, BundleSpec, ConductorSpec, ResistanceAtTemperature,
)

_SUNNY = dict(
    diameter_m=0.028, resistance_ohm_m=8.7e-5, conductor_temperature_k=348.15,
    air_temperature_k=303.15, wind_speed_m_s=2.0, wind_angle_deg=90.0,
    pressure_pa=101325.0, relative_humidity=0.5, ghi_w_m2=800.0, emissivity=0.8, absorptivity=0.8,
)


def _rate(**overrides) -> np.ndarray:
    return ampacity(**{**_SUNNY, **overrides})


def _points(*pairs: tuple[float, float]) -> tuple[ResistanceAtTemperature, ...]:
    return tuple(ResistanceAtTemperature(temperature_c=t, ac_resistance_ohm_kft=r) for t, r in pairs)


def _conductor() -> ConductorSpec:
    return ConductorSpec(name="c", equipment=BareConductorEquipment(
        diameter_in=1.108, ac_resistances=_points((25, 0.0222), (75, 0.0265)),
    ))


def _weather(rows: int = 3, **overrides) -> pd.DataFrame:
    values = dict(air_temperature_k=303.15, wind_speed_m_s=2.0, pressure_pa=101325.0, ghi_w_m2=800.0)
    index = pd.date_range("2012-07-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({**values, **overrides}, index=index)


def test_rating_rises_with_wind_and_falls_with_air_temperature() -> None:
    by_wind = _rate(wind_speed_m_s=np.linspace(0, 15, 50))
    by_air = _rate(air_temperature_k=np.linspace(250, 340, 50))
    assert np.all(np.diff(by_wind) >= 0)
    assert np.all(np.diff(by_air) < 0)


def test_humid_air_cools_less_than_dry_air() -> None:
    # Water vapor is lighter than dry air, so humid air is less dense.
    by_humidity = _rate(relative_humidity=np.linspace(0, 1, 11))
    assert np.all(np.diff(by_humidity) < 0)


def test_weather_humidity_column_overrides_the_default() -> None:
    bundle = BundleSpec(subconductor_count=1)
    default = conductor_ampacity(_conductor(), bundle, _weather(), wind_angle_deg=90)
    dry = conductor_ampacity(_conductor(), bundle, _weather(relative_humidity=0.0), wind_angle_deg=90)
    explicit = conductor_ampacity(_conductor(), bundle, _weather(relative_humidity=0.5), wind_angle_deg=90)
    assert (dry > default).all()
    pd.testing.assert_series_equal(explicit, default)


def test_crosswind_cools_more_than_wind_along_the_line() -> None:
    assert _rate(wind_angle_deg=90) > _rate(wind_angle_deg=45) > _rate(wind_angle_deg=0)


def test_wind_angle_only_matters_relative_to_the_conductor_axis() -> None:
    angle = 30.0
    equivalent = [angle, -angle, 180 - angle, 180 + angle, angle + 360]
    np.testing.assert_allclose(_rate(wind_angle_deg=np.array(equivalent)), _rate(wind_angle_deg=angle))


def test_rating_is_zero_when_the_conductor_cannot_stay_at_its_limit() -> None:
    assert _rate(air_temperature_k=348.15) == 0
    assert _rate(air_temperature_k=360.0) == 0


def test_heat_balance_closes() -> None:
    # Joule heating at the rating equals net cooling, so I²R scales with 1/R.
    assert _rate(resistance_ohm_m=2 * 8.7e-5) ** 2 == pytest.approx(_rate() ** 2 / 2)


def test_bundle_carries_its_subconductor_count_times_one_conductor() -> None:
    weather = _weather()
    single = conductor_ampacity(_conductor(), BundleSpec(subconductor_count=1), weather, wind_angle_deg=90)
    bundled = conductor_ampacity(
        _conductor(), BundleSpec(subconductor_count=3, subconductor_spacing_in=18), weather, wind_angle_deg=90,
    )
    pd.testing.assert_series_equal(bundled, 3 * single)
    assert single.index.equals(weather.index)


_MANUFACTURER_CONDITIONS = dict(  # stated on both sheets below (IEEE 738, sea level, full sun)
    air_temperature_k=[298.15], wind_speed_m_s=[0.6096], pressure_pa=[101325.0], ghi_w_m2=[1033.0],
    relative_humidity=[0.0],  # unstated; IEEE 738's own air density ignores humidity
)


@pytest.mark.parametrize(("diameter_in", "published_resistances", "temperature_c", "published_amps"), [
    # Southwire ACSR Drake 795 kcmil 26/7: 0.0263 ohm/kft AC at 75 °C, 907 A.
    (1.107, [(75, {"value": 0.0263, "unit": "ohm / kilofoot"})], 75, 907),
    # CTC ACCC Drake: AC 0.0892 ohm/mi at 25 °C and 0.1498 at 200 °C; 1036 / 1786 / 1884 A.
    (1.108, [(25, {"value": 0.0892, "unit": "ohm / mile"}), (200, {"value": 0.1498, "unit": "ohm / mile"})], 75, 1036),
    (1.108, [(25, {"value": 0.0892, "unit": "ohm / mile"}), (200, {"value": 0.1498, "unit": "ohm / mile"})], 180, 1786),
    (1.108, [(25, {"value": 0.0892, "unit": "ohm / mile"}), (200, {"value": 0.1498, "unit": "ohm / mile"})], 200, 1884),
])
def test_reproduces_published_manufacturer_ampacity(
    diameter_in: float, published_resistances: list, temperature_c: float, published_amps: float,
) -> None:
    conductor = ConductorSpec(name="published", equipment=BareConductorEquipment(
        diameter_in=diameter_in,
        ac_resistances=tuple(ResistanceAtTemperature(temperature_c=t, ac_resistance_ohm_kft=r)
                             for t, r in published_resistances),
    ))
    weather = pd.DataFrame(_MANUFACTURER_CONDITIONS, index=pd.DatetimeIndex(["2012-06-10 17:00"], tz="UTC"))
    rating = conductor_ampacity(
        conductor, BundleSpec(subconductor_count=1), weather, wind_angle_deg=90,
        conductor_temperature_c=temperature_c, emissivity=0.5, absorptivity=0.5,
    )
    assert rating.iloc[0] == pytest.approx(published_amps, rel=0.01)


def test_resistance_interpolates_and_extrapolates_linearly() -> None:
    points = _points((25, 0.02), (75, 0.03), (200, 0.05))
    assert ac_resistance_at(points, 75) == 0.03
    assert ac_resistance_at(points, 50) == pytest.approx(0.025)
    assert ac_resistance_at(points, 100) == pytest.approx(0.03 + 0.02 * 25 / 125)
    assert ac_resistance_at(points, 250) == pytest.approx(0.05 + 0.02 * 50 / 125)


def test_one_published_temperature_rates_only_at_that_temperature() -> None:
    points = _points((75, 0.0265))
    assert ac_resistance_at(points, 75) == 0.0265
    with pytest.raises(ValueError, match="two published temperatures"):
        ac_resistance_at(points, 100)


def test_resistance_points_need_distinct_temperatures_and_accept_other_units() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        BareConductorEquipment(ac_resistances=_points((75, 0.02), (75, 0.03)))
    point = ResistanceAtTemperature(
        temperature_c={"value": 348.15, "unit": "kelvin"},
        ac_resistance_ohm_kft={"value": 0.1, "unit": "ohm / mile"},
    )
    assert point.temperature_c == pytest.approx(75)
    assert point.ac_resistance_ohm_kft == pytest.approx(0.1 / 5.28)


def test_catalog_conductor_keeps_every_published_resistance() -> None:
    record = ConductorRecord(
        record_id="test", source_id="source", family="ACCC", diameter_inch=1.108, gmr_ft="0.0375",
        ac_resistance_25c_ohm_kft="0.0170", ac_resistance_200c_ohm_kft="0.0282",
    )
    equipment = conductor_from_record(record, catalog_version="v2").equipment
    assert {(p.temperature_c, p.ac_resistance_ohm_kft) for p in equipment.ac_resistances} == {
        (25, 0.0170), (200, 0.0282),
    }


@pytest.mark.parametrize(("overrides", "message"), [
    ({"air_temperature_k": 30.0}, "check units"),
    ({"pressure_pa": 101.3}, "check units"),
    ({"wind_speed_m_s": -1.0}, "nonnegative"),
    ({"relative_humidity": 50.0}, "not a percentage"),
])
def test_weather_in_wrong_units_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        conductor_ampacity(_conductor(), BundleSpec(subconductor_count=1), _weather(**overrides), wind_angle_deg=90)


def test_weather_needs_its_columns_and_a_utc_index() -> None:
    bundle = BundleSpec(subconductor_count=1)
    with pytest.raises(ValueError, match="missing columns"):
        conductor_ampacity(_conductor(), bundle, _weather().drop(columns="ghi_w_m2"), wind_angle_deg=90)
    with pytest.raises(ValueError, match="UTC"):
        conductor_ampacity(_conductor(), bundle, _weather().tz_convert("US/Mountain"), wind_angle_deg=90)


def test_missing_nullable_humidity_is_rejected() -> None:
    humidity = pd.array([0.5, pd.NA, 0.5], dtype="Float64")
    with pytest.raises(ValueError, match="not a percentage"):
        conductor_ampacity(_conductor(), BundleSpec(subconductor_count=1), _weather(relative_humidity=humidity), wind_angle_deg=90)


@pytest.mark.parametrize("surface", [{"emissivity": 80.0}, {"absorptivity": -0.1}, {"emissivity": float("nan")}])
def test_surface_coefficients_must_be_fractions(surface: dict) -> None:
    with pytest.raises(ValueError, match="fractions"):
        conductor_ampacity(_conductor(), BundleSpec(subconductor_count=1), _weather(), wind_angle_deg=90, **surface)
