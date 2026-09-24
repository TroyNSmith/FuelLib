"""Type definitions for FuelLib."""

from __future__ import annotations
from typing import ClassVar, TypeVar, TypeAlias, Any, cast

try:
    import astropy.units as u
    import jax.numpy as resolved_np
    import unxt
    from jaxtyping import Array, Float

    _ = resolved_np.array([1])  # Check that jax initializes properly
    Array1D = Float[Array, "I"]
    Array2D = Float[Array, "I J"]

    _ = unxt.Q(1, "K")  # Check that unxt initializes properly

    # Enable temperature conversions for astropy units
    u.add_enabled_equivalencies(u.temperature())
    # Add additional units to astropy to ensure compatibility with pint
    atm = u.def_unit("atmosphere", 101325 * u.Pa)
    u.add_enabled_units([atm])
    # Enable aliases for astropy units to ensure compatibility with pint
    u.add_enabled_aliases({"celsius": u.deg_C, "atm": atm})

    class UnxtQuantity(unxt.Quantity):
        """Resolved quantity class using unxt."""

        def __init__(
            self,
            value: Array,
            unit: u.UnitBase | u.FunctionUnitBase,
        ) -> None:
            """Initialize a UnxtQuantity with a value and a unit.

            Args:
                value: The numerical value of the quantity.
                unit: The unit associated with the quantity.
            """
            super().__init__(value, unit)

        @property
        def magnitude(self) -> Array | unxt.quantity.StaticValue:
            return self.value

        @property
        def units(self) -> u.UnitBase | u.FunctionUnitBase:
            return self.unit

        def to(self, unit: Any, /) -> "UnxtQuantity":
            return cast("UnxtQuantity", super().to(unit))

    # `AbstractQuantity` (not `Quantity`) since methods like `.to()`/`.uconvert()`
    # are annotated to return the base class, not `Self`.
    UnxtQuantityT = TypeVar("UnxtQuantityT", bound=UnxtQuantity)
    Quantity0D = Float[UnxtQuantity, ""]
    Quantity1D = Float[UnxtQuantity, "I"]
    Quantity2D = Float[UnxtQuantity, "I J"]

    has_jax = True

except (ImportError, Exception):
    import numpy as np
    import pint

    # NumPy
    _ = np.array([1])  # Check that NumPy is available
    Array1D: TypeAlias = np.ndarray[tuple[int,]]
    Array2D: TypeAlias = np.ndarray[tuple[int, int]]

    # Pint
    ureg = pint.UnitRegistry()
    ## NOTE: TypeVar provides a way to define generic types that can be used for type
    ## hinting. This will be important when we begin to implement optional dependencies,
    # as it allows us to define types that can adapt to the available numerical library.
    PintQuantityT = TypeVar("PintQuantityT", bound=pint.Quantity)
    Quantity0D: TypeAlias = pint.Quantity[float]
    Quantity1D: TypeAlias = pint.Quantity[Array1D]
    Quantity2D: TypeAlias = pint.Quantity[Array2D]

    has_jax = False


QuantityT = UnxtQuantityT if has_jax else PintQuantityT


class Units:
    """Wrapper class for the resolved Units registry.

    Currently a placeholder but will be expanded to include the Unxt registry
    """

    Quantity: ClassVar[type[QuantityT]] = ureg.Quantity
    Q: ClassVar[type[QuantityT]] = ureg.Quantity  # Alias for Quantity


__all__ = [
    "Array1D",
    "Array2D",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
]
