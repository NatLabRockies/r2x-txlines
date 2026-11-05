"""Float field types that carry their unit as r2x-core metadata.

Model fields store plain floats in the unit named by the field (``x_ft``,
``nominal_voltage_kv``). A field also accepts a value in another unit as a
mapping of the right dimension, converted on the way in, so
``x_ft={"value": 1, "unit": "meter"}`` stores 3.28. Calculations only ever see
the floats.
"""

from typing import Annotated

from r2x_core import Unit

Amperes = Annotated[float, Unit("ampere")]
Celsius = Annotated[float, Unit("degC")]
Degrees = Annotated[float, Unit("degree")]
Feet = Annotated[float, Unit("foot")]
Hertz = Annotated[float, Unit("hertz")]
Inches = Annotated[float, Unit("inch")]
Kilovolts = Annotated[float, Unit("kilovolt")]
Miles = Annotated[float, Unit("mile")]
OhmMeters = Annotated[float, Unit("ohm * meter")]
Ohms = Annotated[float, Unit("ohm")]
OhmsPerKilofoot = Annotated[float, Unit("ohm / kilofoot")]
PerKelvin = Annotated[float, Unit("1 / kelvin")]
Pounds = Annotated[float, Unit("pound_force")]
PoundsPerKilofoot = Annotated[float, Unit("pound_force / kilofoot")]
Psi = Annotated[float, Unit("psi")]
SquareInches = Annotated[float, Unit("inch ** 2")]

__all__ = [
    "Amperes", "Celsius", "Degrees", "Feet", "Hertz", "Inches", "Kilovolts", "Miles",
    "OhmMeters", "Ohms", "OhmsPerKilofoot", "PerKelvin", "Pounds", "PoundsPerKilofoot",
    "Psi", "SquareInches",
]
