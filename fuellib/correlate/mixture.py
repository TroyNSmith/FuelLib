"""Mixture-wise correlation methods."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils.types import FloatVector, PintScalar
from ..utils.units import PintUnits
from . import components, mixing_rules

if TYPE_CHECKING:
    from ..fuel import Fuel


def flash_point_alibakhshi(
    fuel: "Fuel",
    Yi: FloatVector,
    mixing_rule: Literal["linear", "Liaw"] = "Liaw",
    n_iter: int = 8,
) -> PintScalar:
    """
    Calculate the Alibakhshi et al. (2015) mixture flash point for the given fuel.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: FloatVector
    :param mixing_rule: The mixing rule to use for the calculation.
    :type mixing_rule: Literal["linear", "Liaw"]
    :param n_iter: Number of iterations for the Liaw mixing rule calculation. Default is 8.
    :type n_iter: int, optional
    :return: Flash point of the mixture in Kelvin.
    :rtype: PintScalar
    """
    Xi = fuel.Y2X(Yi)
    Tfp = components.flash_point_alibakhshi(fuel)

    if mixing_rule == "linear":
        return mixing_rules.linear(Xi, Tfp)

    elif mixing_rule == "Liaw":
        psat_ref = components.psat_lee_kesler(fuel, Tfp)
        psat_ref = PintUnits.Quantity(np.diag(psat_ref), psat_ref.units)

        def _residual(T):
            psat_T = components.psat_lee_kesler(fuel, T)
            return np.sum(Xi * psat_T / psat_ref) - 1.0

        T = np.sum(Xi * Tfp)
        dT = PintUnits.Quantity(1.0, "K")
        for _ in range(n_iter):
            val = _residual(T)
            dV = (_residual(T + dT) - val) / dT
            T -= np.clip(
                val / (dV + PintUnits.Quantity(1e-30, "K^-1")),
                PintUnits.Quantity(-20.0, "K"),
                PintUnits.Quantity(20.0, "K"),
            )
        return PintUnits.Quantity(T, "K")


def flash_point_alqaheem(
    fuel: "Fuel",
    Yi: FloatVector,
    mixing_rule: Literal["linear", "Liaw"] = "Liaw",
    n_iter: int = 8,
) -> PintScalar:
    """
    Calculate the Alqaheem et al. (2018) mixture flash point for the given fuel.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: FloatVector
    :param mixing_rule: The mixing rule to use for the calculation.
    :type mixing_rule: Literal["linear", "Liaw"]
    :param n_iter: Number of iterations for the Liaw mixing rule calculation.
    :type n_iter: int
    :return: Flash point of the mixture in Kelvin.
    :rtype: PintScalar
    """
    Xi = fuel.Y2X(Yi)
    Tfp = components.flash_point_alqaheem(fuel)

    if mixing_rule == "linear":
        return mixing_rules.linear(Xi, Tfp)

    elif mixing_rule == "Liaw":
        psat_ref = components.psat_lee_kesler(fuel, Tfp)
        psat_ref = PintUnits.Quantity(np.diag(psat_ref), psat_ref.units)

        def _residual(T):
            psat_T = components.psat_lee_kesler(fuel, T)
            return np.sum(Xi * psat_T / psat_ref) - 1.0

        T = np.sum(Xi * Tfp)
        dT = PintUnits.Quantity(1.0, "K")
        for _ in range(n_iter):
            val = _residual(T)
            dV = (_residual(T + dT) - val) / dT
            T -= np.clip(
                val / (dV + PintUnits.Quantity(1e-30, "K^-1")),
                PintUnits.Quantity(-20.0, "K"),
                PintUnits.Quantity(20.0, "K"),
            )
        return PintUnits.Quantity(T, "K")


def yield_sooting_index_mcenally(
    fuel: "Fuel",
    Yi: FloatVector,
) -> PintScalar:
    """
    Calculate the Yield Sooting Index for the given fuel.

    Component values are derived from the McEnally-Pfefferle Yale YSI Database
    Volume 2.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: FloatVector
    :return: Yale Yield Sooting Index of the mixture.
    :rtype: PintScalar
    """
    Xi = fuel.Y2X(Yi)
    return mixing_rules.linear(Xi, fuel.get_gcm_property("empirical", "ysi"))


def derived_cetane_number(
    fuel: "Fuel", Yi: FloatVector, T_ref: PintScalar
) -> PintScalar:
    """
    Calculate the Derived Cetane Number for the given fuel.

    Component values are derived from the empirical DCN data.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: FloatVector
    :param T_ref: Reference temperature for the calculation.
    :type T_ref: PintScalar
    :return: Derived Cetane Number of the mixture.
    :rtype: PintScalar
    """
    Xi = fuel.Y2X(Yi)
    Vi = Yi / fuel.density(T_ref)
    return mixing_rules.linear(Xi, fuel.get_gcm_property("empirical", "dcn"))


__all__ = [
    "flash_point_alibakhshi",
    "flash_point_alqaheem",
    "yield_sooting_index_mcenally",
]
