"""Selected cable components and per-circuit installation inputs."""

from abc import ABC, abstractmethod
import math
from typing import Literal

from infrasys import Component
from pydantic import model_validator

from transmissionlines.models.base import LineDataModel
from transmissionlines.models.common import CatalogReference
from transmissionlines.units import (
    Celsius,
    OhmsPerKilofoot,
    OptionalAmperes,
    OptionalFeet,
    OptionalInches,
    OptionalOhmsPerKilofoot,
    OptionalPounds,
    OptionalPoundsPerKilofoot,
    OptionalSquareInches,
)


class ResistanceAtTemperature(LineDataModel):
    """One published AC resistance and the conductor temperature it applies at."""

    temperature_c: Celsius
    ac_resistance_ohm_kft: OhmsPerKilofoot

    @model_validator(mode="after")
    def validate_resistance(self) -> "ResistanceAtTemperature":
        if not (math.isfinite(self.temperature_c) and math.isfinite(self.ac_resistance_ohm_kft)):
            raise ValueError("temperature and resistance must be finite")
        if self.ac_resistance_ohm_kft <= 0:
            raise ValueError("ac_resistance_ohm_kft must be positive")
        return self


class BareConductorEquipment(LineDataModel):
    """Measured or manufacturer-provided conductor properties.

    ``ac_resistance_ohm_kft`` is the single resistance used for impedances.
    ``ac_resistances`` lists every published (temperature, resistance) pair, which
    thermal ratings interpolate between.
    """

    ac_resistances: tuple[ResistanceAtTemperature, ...] = ()
    diameter_in: OptionalInches = None
    gmr_ft: OptionalFeet = None
    capacitance_radius_ft: OptionalFeet = None
    ampacity_a: OptionalAmperes = None
    ac_resistance_ohm_kft: OptionalOhmsPerKilofoot = None
    emergency_ampacity_a: OptionalAmperes = None
    dc_resistance_ohm_kft: OptionalOhmsPerKilofoot = None
    weight_lb_kft: OptionalPoundsPerKilofoot = None
    rated_strength_lb: OptionalPounds = None
    total_area_in2: OptionalSquareInches = None

    @model_validator(mode="after")
    def validate_measurements(self) -> "BareConductorEquipment":
        """Reject nonpositive or non-finite measurements; absent ones stay None."""
        for field in type(self).model_fields:
            value = getattr(self, field)
            if field != "ac_resistances" and value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{field} must be finite and positive")
        temperatures = [point.temperature_c for point in self.ac_resistances]
        if len(set(temperatures)) != len(temperatures):
            raise ValueError("ac_resistances temperatures must be distinct")
        return self


class BundleSpec(LineDataModel):
    """Describe the number and separation of installed subconductors."""

    subconductor_count: int
    subconductor_spacing_in: OptionalInches = None

    @model_validator(mode="after")
    def validate_spacing(self) -> "BundleSpec":
        if self.subconductor_count < 1:
            raise ValueError("subconductor_count must be positive")
        if self.subconductor_count > 1 and (
            self.subconductor_spacing_in is None or self.subconductor_spacing_in <= 0
        ):
            raise ValueError("subconductor_spacing_in is required and positive for bundles")
        if self.subconductor_count == 1 and self.subconductor_spacing_in is not None:
            raise ValueError("single conductors cannot specify subconductor_spacing_in")
        return self


class InsulatorStringSpec(LineDataModel):
    """Specify one circuit's insulator string."""

    insulator_type: Literal["glass", "porcelain", "polymeric"]
    number_of_insulators: int
    insulator_code: str
    insulator_coupling: Literal["ball_and_socket", "y_clevis"]

    @model_validator(mode="after")
    def validate_count(self) -> "InsulatorStringSpec":
        if self.number_of_insulators < 1:
            raise ValueError("number_of_insulators must be positive")
        return self


class Cable(Component, ABC):
    """Hold a reusable manufacturer's selected static cable measurements."""

    catalog_reference: CatalogReference | None = None
    equipment: BareConductorEquipment

    @abstractmethod
    def _cable_kind(self) -> str:
        """Identify the concrete cable role."""


class ConductorSpec(Cable):
    """Select a reusable phase conductor independently of installation."""

    family: str | None = None
    codeword: str | None = None
    stranding: str | None = None

    def _cable_kind(self) -> str:
        return "phase"


class GroundWireSpec(Cable):
    """Select a reusable unbundled ground wire."""

    def _cable_kind(self) -> str:
        return "ground"


__all__ = [
    "BareConductorEquipment",
    "BundleSpec",
    "Cable",
    "ConductorSpec",
    "GroundWireSpec",
    "InsulatorStringSpec",
    "ResistanceAtTemperature",
]
