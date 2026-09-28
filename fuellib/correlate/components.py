"""Component correlation functions for FuelLib."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils import Units, types

if TYPE_CHECKING:
    from ..fuel import Fuel


def molar_liquid_volume(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    r"""Compute molar liquid volume with temperature correction.

    The liquid molar volume is calculated at a specific temperature :math:`T` using
    the generalized Rackett equation\ :footcite:p:`rackett_equation_1970` \
    :footcite:p:`yamada_saturated_1973`
    with an updated :math:`\phi_i` parameter\ :footcite:p:`govindaraju_group_2016`:

    .. math::

       V_{m,i} = V_{m,\textit{stp},i} Z^{\phi_i}_{c,i},

    where

    .. math::

       Z_{c,i} &= 0.29056 - 0.08775 \omega_i,  \\
       \phi_i &=
       \begin{cases}
           (1 - T_{r,i})^{2/7} - (1 - T_{r,\textit{stp},i})^{2/7}, & \text{ if } T \leq
           T_{c,i} \\ - (1 - T_{r,\textit{stp},i})^{2/7}, & \text{ if } T > T_{c,i}
       \end{cases}.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Molar liquid volume in m^3/mol.
    """
    Tstp = Units.Quantity(298, "K")
    T = T.to("K")
    Tc = fuel.Tc.to("K")
    omega = fuel.omega
    Vm_stp = fuel.Vm_stp

    phi = np.zeros_like(Tc.magnitude)
    for i in range(len(Tc)):
        if T > Tc[i]:
            phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
        else:
            phi[i] = (1 - (T / Tc[i])) ** (2.0 / 7.0) - (1 - (Tstp / Tc[i])) ** (
                2.0 / 7.0
            )
    z = 0.29056 - 0.08775 * omega
    return Vm_stp * z**phi


def density(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate the density of each component at temperature T.

    Args:
        fuel: Fuel object.
        T: Temperature of the mixture in Kelvin.

    Returns:
        Density of each compound in kg/m^3.
    """
    return (fuel.MW / molar_liquid_volume(fuel, T)).to("kg/m^3")


def kinematic_viscosity(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    r"""Calculate the viscosity using Dutt's equation.

    The kinematic viscosity of the *i-th* compound of the fuel,

    .. math::

       \nu_i = \frac{\mu_i}{\rho_i},

    is calculated from Dutt's equation (Eq. 4.23 in Viscosity of Liquids\
    :footcite:p:`viswanath_viscosity_2007`) provided :math:`T` in :math:`^{\circ}` C:

    .. math::

       \nu_i = 10^{-6} \times \exp \bigg\{-3.0171 + \frac{442.78 + 1.6452 \,T_{b,i}}{T +
       239 - 0.19 \,T_{b,i}} \bigg\}.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Viscosity of each component in m^2/s.
    """
    # Convert temperature to Celsius
    T: float = T.to("celsius").magnitude
    Tb = fuel.Tb.to("celsius").magnitude
    # RHS of Dutt's equation (4.23) in Viscosity of Liquids
    rhs = -3.0171 + (442.78 + 1.6452 * Tb) / (T + 239 - 0.19 * Tb)
    return Units.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")


def dynamic_viscosity(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate liquid dynamic viscosity based on droplet temperature and density.

    Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Dynamic viscosity in Pa*s.
    """
    nu_i = kinematic_viscosity(fuel, T)
    rho_i = density(fuel, T)
    return (nu_i * rho_i).to("Pa*s")


def molar_specific_heat_capacity(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Compute molar specific heat capacity at a given temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Molar specific heat capacity in J/mol/K.
    """
    T = T.to("K")
    theta = (T - Units.Quantity(298, "K")) / Units.Quantity(700, "K")
    return (fuel.Cp_stp + fuel.Cp_B * theta + fuel.Cp_C * theta**2).to("J/(mol*K)")


def liquid_mass_specific_heat_capacity(
    fuel: "Fuel", T: types.Quantity0D
) -> types.Quantity1D:
    """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Mass specific heat capacity in J/kg/K.
    """
    T = T.to("K")
    return (molar_specific_heat_capacity(fuel, T) / fuel.MW).to("J/(kg*K)")


def saturated_vapor_pressure(
    fuel: "Fuel",
    T: types.Quantity0D,
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> types.Quantity1D:
    """Compute saturated vapor pressure.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
            Default is "Lee-Kesler".

    Returns:
        Saturated vapor pressure in Pa.
    """
    T = T.to("K")

    Tr = T / fuel.Tc
    Pc = fuel.Pc
    omega = fuel.omega

    if correlation.casefold() == "Ambrose-Walton".casefold():
        # May cause trouble at high temperatures
        tau = 1 - Tr
        f0 = (
            -5.97616 * tau
            + 1.29874 * tau**1.5
            - 0.60394 * tau**2.5
            - 1.06841 * tau**5.0
        )
        f0 /= Tr
        f1 = (
            -5.03365 * tau
            + 1.11505 * tau**1.5
            - 5.41217 * tau**2.5
            - 7.46628 * tau**5.0
        )
        f1 /= Tr
        f2 = (
            -0.64771 * tau
            + 2.41539 * tau**1.5
            - 4.26979 * tau**2.5
            - 3.25259 * tau**5.0
        )
        f2 /= Tr
        rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

    else:  # Default correlation is Lee-Kesler
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
        rhs = np.exp(f0 + omega * f1)

    return (Pc * rhs).to("Pa")


def latent_heat_vaporization(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate latent heat of vaporization adjusted for temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Latent heat of vaporization in J/kg.
    """
    T = T.to("K")

    Tc = fuel.Tc.to("K")
    Tb = fuel.Tb.to("K")
    Lv_stp = fuel.Lv_stp.to("J/kg")

    # Reduced temperatures
    Tr = T / Tc
    Trb = Tb / Tc

    Lvi = Units.Quantity(np.zeros_like(Tc.magnitude), "J/kg")
    for i in range(len(Tc)):
        if T > Tc[i]:
            Lvi.magnitude[i] = 0.0
        else:
            Lvi[i] = Lv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

    return Lvi
