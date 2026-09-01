import os

import pandas as pd

import fuellib as fl
from fuellib._data_locator import get_fueldata_props_dir

FUELDATA_PROPS_DIR = get_fueldata_props_dir()


def get_pred_and_data(fuel_name, prop_name):
    # Get the fuel properties based on the GCM
    fuel = fl.fuel(fuel_name)

    data_file = f"{fuel_name}.csv"
    data = pd.read_csv(os.path.join(FUELDATA_PROPS_DIR, data_file), skiprows=[1])

    # Separate properties and associated temperatures from data
    T_data = data.Temperature[data[prop_name].notna()].to_numpy(dtype=float)
    prop_data = data[prop_name].dropna().to_numpy()

    # Vectors for temperature (convert from C to K)
    T_pred = fl.convert.C2K(T_data)

    Y_li = fuel.Y_0

    if prop_name == "Density":
        # Mixture density, converted to CGS (g/cm^3)
        pred = fuel.mixture_density(Y_li, T_pred).ustrip("g/cm^3")

    if prop_name == "VaporPressure":
        # Mixture vapor pressure, converted to kPa
        pred = fuel.mixture_vapor_pressure(Y_li, T_pred).ustrip("kPa")

    if prop_name == "Viscosity":
        # Converted to mm^2/s
        pred = fuel.mixture_kinematic_viscosity(Y_li, T_pred).ustrip("mm^2/s")

    if prop_name == "SurfaceTension":
        pred = fuel.mixture_surface_tension(Y_li, T_pred).ustrip("N/m")

    if prop_name == "ThermalConductivity":
        pred = fuel.mixture_thermal_conductivity(Y_li, T_pred).ustrip("W/(m*K)")

    return T_data, prop_data, pred


# Backward-compatible alias for older call sites.
getPredAndData = get_pred_and_data
