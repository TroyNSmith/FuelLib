"""
Script that exports critical properties and initial mass fraction data
for use in Pele simulations.

This script is designed to be run from the command line and will create
a file named "sprayPropsGCM_<fuel_name>.inp" or "sprayPropsMP_<fuel_name>.inp"
in the specified directory. The file contains properties for each compound in
the fuel, formatted for Pele.

Usage:
    fl-export-pele -f <fuel_name>

For detailed options, run:
    fl-export-pele -h
"""

import logging
import os
from pathlib import Path
from typing import ClassVar, Literal

import click
import pandas as pd
import quaxed.numpy as qnp
from unxt import Quantity

from fuellib.fuel import Fuel, mixing_rule
from fuellib.fuel.locator import DEFAULT_DATA_DIR
from fuellib.utils.units import convert_temperature

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
