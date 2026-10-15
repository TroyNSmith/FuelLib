"""Core Fuel class for Group Contribution Method calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .._reference_data import (
    DEFAULT_DATA_DIR,
    load_ref_compounds,
    load_ref_gani,
    smiles_series_to_inchi,
)
from ..utils import Units, types
from .component_correlations import ComponentCorrelationsMixin
from .component_properties import ComponentPropertiesMixin
from .mixture_correlations import MixtureCorrelationsMixin


def _match_gc_to_ref(gcData: DataFrame, refData: DataFrame, refName: str) -> DataFrame:
    """Match GCxGC data to a reference data table.

    Rows are matched first by 'Reference Compound' (case-insensitive), then any
    remaining rows are matched by comparing the InChI of their 'SMILES'
    (if the reference table has a 'SMILES' column).

    Args:
        gcData: GCxGC data as a DataFrame.
        refData: Reference data as a DataFrame.
        refName: Name of the reference data (used in error messages).

    Returns:
        Subset of `refData` with one row per GCxGC row, in GCxGC order.

    Raises:
        ValueError: If any GCxGC row cannot be matched to a reference row.
    """
    refIdx = Series(pd.NA, index=gcData.index, dtype="object")

    def _normalize_names(names: Series) -> Series:
        return names.astype("string").str.strip().str.lower()

    if "Reference Compound" in gcData.columns:
        refNames = _normalize_names(refData["Reference Compound"]).dropna()
        nameLookup = dict(zip(refNames, refNames.index))
        refIdx = _normalize_names(gcData["Reference Compound"]).map(nameLookup)

    unmatched = refIdx.isna()
    if unmatched.any() and "SMILES" in gcData.columns and "SMILES" in refData.columns:
        refInchi = smiles_series_to_inchi(refData["SMILES"]).dropna()
        inchiLookup = dict(zip(refInchi, refInchi.index))
        refIdx[unmatched] = smiles_series_to_inchi(gcData.loc[unmatched, "SMILES"]).map(
            inchiLookup
        )

    unmatched = refIdx.isna()
    if unmatched.any():
        idCols = [c for c in ("Reference Compound", "SMILES") if c in gcData.columns]
        msg = (
            f"GCxGC compounds not found in reference data '{refName}':\n"
            f"{gcData.loc[unmatched, idCols].to_string()}"
        )
        raise ValueError(msg)

    return refData.loc[refIdx.astype(int)].reset_index(drop=True)


class Fuel(
    MixtureCorrelationsMixin, ComponentCorrelationsMixin, ComponentPropertiesMixin
):
    """Class for handling calculations of thermodynamic and mixture properties.

    The core class loads the GCxGC and reference data. Functionality is provided by:

    * `ComponentPropertiesMixin`: properties of each compound (e.g. `Tc`, `MW`).
    * `ComponentCorrelationsMixin`: temperature-dependent correlations for each
      compound (e.g. `density`, `psat`).
    * `MixtureCorrelationsMixin`: composition conversions and mixture
      correlations (e.g. `Y2X`, `mixture_density`).
    """

    def __init__(
        self,
        name: str,
        fuelDataDir: str | Path | None = None,
        refCompoundsPath: str | Path | None = None,
        refGaniPath: str | Path | None = None,
        *,
        useRefProperties: bool = True,
    ) -> None:
        """Initialize a Fuel instance.

        Args:
            name: Name of the fuel.
            fuelDataDir: Directory containing fuel data.
                Defaults to `fuelLib/data`.
            refCompoundsPath: Path to the reference compounds CSV file.
                Defaults to `fuellib/data/refCompounds.csv`. A custom table in
                `fuelDataDir`, when present, is appended to the bundled table and
                replaces rows with the same InChI.
            refGaniPath: Path to the reference Gani decomposition CSV file.
                Defaults to `fuellib/data/refGani.csv`. A custom table in
                `fuelDataDir`, when present, is appended to the bundled table and
                replaces rows with the same `Common_Name`.
            useRefProperties: Whether to use reference compound properties for
                calculations. Defaults to `True`.

        Raises:
            FileNotFoundError: If the fuel data directory, reference CSV files,
                or gcxgc data file do not exist.
            ValueError: If the GCxGC data does not contain either 'Reference Compound'
                or 'SMILES' columns or 'Weight %' column is missing.
        """
        self.name: str = name
        """Name of the fuel/mixture."""
        self.useRefProperties: bool = useRefProperties
        """Whether to use reference compound properties for calculations."""
        self.fuelDataDir: Path = (
            Path(fuelDataDir) if fuelDataDir is not None else DEFAULT_DATA_DIR
        )
        """Directory containing fuel data."""
        if not self.fuelDataDir.exists():
            msg = f"Fuel data directory does not exist: {self.fuelDataDir}"
            raise FileNotFoundError(msg)

        gcFile: Path = self.fuelDataDir / "gcData" / f"{name}.csv"
        if not gcFile.exists():
            msg = f"GCxGC data file does not exist: {gcFile}"
            raise FileNotFoundError(msg)

        self.gcData: DataFrame = pd.read_csv(gcFile)
        """GCxGC data as a DataFrame."""
        if (
            "Reference Compound" not in self.gcData.columns
            and "SMILES" not in self.gcData.columns
        ):
            msg = (
                "GCxGC data must contain either 'Reference Compound' or 'SMILES' "
                "columns."
            )
            raise ValueError(msg)
        if "Weight %" not in self.gcData.columns:
            msg = "GCxGC data must contain 'Weight %' column."
            raise ValueError(msg)
        Y_0 = self.gcData["Weight %"].to_numpy(float)
        self.Y_0: types.Quantity1D = (
            Units.Quantity(Y_0 / sum(Y_0), "")
            if sum(Y_0) > 0
            else Units.Quantity(np.zeros_like(Y_0), "")
        )
        """Initial weight fractions from the GCxGC data."""

        self.refCompounds: DataFrame = load_ref_compounds(
            refCompoundsPath, self.fuelDataDir
        )
        """Reference compounds and their properties."""
        self.refGani: DataFrame = load_ref_gani(refGaniPath, self.fuelDataDir).rename(
            columns={"Common_Name": "Reference Compound"}
        )
        """Reference Gani group decompositions."""
        self.compoundsData: DataFrame = _match_gc_to_ref(
            self.gcData, self.refCompounds, "referenceCompounds"
        )
        """Matched GCxGC data to reference compounds."""
        self.compounds: list[str] = self.compoundsData["Reference Compound"].to_list()
        """List of common names for each matched compound."""
        self.smiles: list[str] = self.compoundsData["SMILES"].to_list()
        """List of SMILES strings for each matched compound."""
        self.pelephysics_keys: list[str] | None = (
            self.gcData["PelePhysics Key"].tolist()
            if "PelePhysics Key" in self.gcData.columns
            else None
        )

        self.ganiDecomp: DataFrame = (
            _match_gc_to_ref(self.compoundsData, self.refGani, "referenceGani")
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """Gani group decompositions for each matched compound."""

        propFilePath = self.fuelDataDir / "propertiesData" / f"{name}.csv"
        self.propData: DataFrame | None = (
            pd.read_csv(propFilePath) if propFilePath.exists() else None
        )
        """Property data for the fuel/mixture, if available."""


__all__ = ["Fuel"]
