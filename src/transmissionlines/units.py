"""Float field types that carry their unit as r2x-core metadata.

Model fields store plain floats in the unit named by the field (``x_ft``,
``nominal_voltage_kv``). A field also accepts a value in another unit as a
mapping of the right dimension, converted on the way in, so
``x_ft={"value": 1, "unit": "meter"}`` stores 3.28. Calculations only ever see
the floats.

The ``Optional`` aliases keep the unit on the outside of the union: ``HasUnits``
does not see a unit nested inside ``Feet | None``.
"""

from typing import Annotated, Any

import pint
from pydantic import BeforeValidator
from r2x_core import Unit


def _reject_quantity(value: Any) -> Any:
    """Stop a Pint quantity reaching float coercion, which would drop its unit."""
    if isinstance(value, pint.Quantity):
        raise ValueError("Pint quantity not accepted; use a mapping with 'value' and 'unit' keys")
    return value


_Float = Annotated[float, BeforeValidator(_reject_quantity)]
_OptionalFloat = Annotated[float | None, BeforeValidator(_reject_quantity)]

Amperes = Annotated[_Float, Unit("ampere")]
Celsius = Annotated[_Float, Unit("degC")]
Degrees = Annotated[_Float, Unit("degree")]
Feet = Annotated[_Float, Unit("foot")]
Hertz = Annotated[_Float, Unit("hertz")]
Inches = Annotated[_Float, Unit("inch")]
Kilovolts = Annotated[_Float, Unit("kilovolt")]
Miles = Annotated[_Float, Unit("mile")]
OhmMeters = Annotated[_Float, Unit("ohm * meter")]
Ohms = Annotated[_Float, Unit("ohm")]
OhmsPerKilofoot = Annotated[_Float, Unit("ohm / kilofoot")]
PerKelvin = Annotated[_Float, Unit("1 / kelvin")]
Pounds = Annotated[_Float, Unit("pound_force")]
PoundsPerKilofoot = Annotated[_Float, Unit("pound_force / kilofoot")]
Psi = Annotated[_Float, Unit("psi")]
SquareInches = Annotated[_Float, Unit("inch ** 2")]

OptionalAmperes = Annotated[_OptionalFloat, Unit("ampere")]
OptionalFeet = Annotated[_OptionalFloat, Unit("foot")]
OptionalInches = Annotated[_OptionalFloat, Unit("inch")]
OptionalOhmsPerKilofoot = Annotated[_OptionalFloat, Unit("ohm / kilofoot")]
OptionalPounds = Annotated[_OptionalFloat, Unit("pound_force")]
OptionalPoundsPerKilofoot = Annotated[_OptionalFloat, Unit("pound_force / kilofoot")]
OptionalSquareInches = Annotated[_OptionalFloat, Unit("inch ** 2")]

__all__ = [
    "Amperes", "Celsius", "Degrees", "Feet", "Hertz", "Inches", "Kilovolts", "Miles",
    "OhmMeters", "Ohms", "OhmsPerKilofoot", "PerKelvin", "Pounds", "PoundsPerKilofoot",
    "OptionalAmperes", "OptionalFeet", "OptionalInches", "OptionalOhmsPerKilofoot",
    "OptionalPounds", "OptionalPoundsPerKilofoot", "OptionalSquareInches",
    "Psi", "SquareInches",
]
