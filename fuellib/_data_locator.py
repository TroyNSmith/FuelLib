"""
Data locator module for FuelLib.

This module provides functions to locate data directories and files embedded
within the fuellib package using importlib.resources.
"""

import os
from importlib.resources import files
from pathlib import Path

__all__ = [
    "get_data_dir",
    "get_fueldata_dir",
    "get_fueldata_props_dir",
    "get_gcmtable_dir",
    "list_fuel_names",
]


def get_data_dir():
    """
    Get the path to FuelLib's data directory.

    :return: Absolute path to the data directory.
    :rtype: str
    """
    data_ref = files("fuellib").joinpath("data")
    # Convert to a concrete path
    return str(data_ref)


def get_gcmtable_dir():
    """
    Get the path to the GCM table data directory.

    :return: Absolute path to gcmTableData directory.
    :rtype: str
    """
    return os.path.join(get_data_dir(), "gcmTableData")


def get_fueldata_dir():
    """
    Get the path to FuelLib's fuel data directory.

    :return: Absolute path to embedded fuelData directory.
    :rtype: str
    """
    return os.path.join(get_data_dir(), "fuelData")


def get_fueldata_props_dir():
    """
    Get the path to FuelLib's properties data subdirectory, or None if not found.

    This directory holds supplementary experimental validation data that isn't tied
    to a single fuel's JSON file (e.g. blend comparisons). It is optional.

    :return: Absolute path to embedded fuelData/propertiesData directory, or None if not found.
    :rtype: str or None
    """
    props_dir = os.path.join(get_fueldata_dir(), "propertiesData")
    return props_dir if os.path.isdir(props_dir) else None


def list_fuel_names(fuel_data_dir=None):
    """
    List available fuel names by discovering "<name>.json" files in a fuel data directory.

    :param fuel_data_dir: Directory containing "<name>.json" fuel files. If None, uses
        embedded data.
    :type fuel_data_dir: str, optional
    :return: Sorted list of fuel names (JSON filenames without the extension).
    :rtype: list[str]
    """
    if fuel_data_dir is None:
        fuel_data_dir = get_fueldata_dir()

    return sorted(p.stem for p in Path(fuel_data_dir).glob("*.json"))
