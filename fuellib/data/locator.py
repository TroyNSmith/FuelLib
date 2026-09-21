"""Fuel data locator utilities."""

from pathlib import Path

from ..utils.logger import logger

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

DEFAULT_FUELDATA_DIR = Path(__file__).parent / "fuelData"


def validate_fuel_data_dir(path: str | Path) -> Path:
    """
    Validate that the given directory is properly structured for FuelLib.

    :param path: Path to validate.
    :type path: str | Path
    :raises FileNotFoundError: If the path is not a directory or required subdirectories are missing.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Invalid directory: {path}")
    if not (path / "gcData").exists():
        raise FileNotFoundError(f"Missing required subdirectory: {path / 'gcData'}")
    if not (path / "groupDecompositionData").exists():
        raise FileNotFoundError(
            f"Missing required subdirectory: {path / 'groupDecompositionData'}"
        )
    return path


def get_metadata_decomp_name(
    fuel_name: str, fuel_data_dir: str | Path = DEFAULT_FUELDATA_DIR
) -> str:
    """
    Load decomposition name mapping from fuel_metadata.yaml.

    :param fuel_name: Name of the fuel to look up.
    :type fuel_name: str
    :param fuel_data_dir: Directory containing fuel data. If None, uses embedded data.
    :type fuel_data_dir: str or Path, optional
    :return: Decomposition name from metadata.
    :rtype: str
    :raises FileNotFoundError: If fuel_metadata.yaml is missing or fuel not found in metadata
    """
    if not HAS_YAML:
        msg = "PyYAML is required to use custom fuels. Install it with: pip install pyyaml"
        raise ImportError(msg)

    fuel_data_dir = Path(fuel_data_dir)
    metadata_file = fuel_data_dir / "fuel_metadata.yaml"
    data_dir_display = (
        "FuelLib embedded data"
        if fuel_data_dir == DEFAULT_FUELDATA_DIR
        else str(fuel_data_dir)
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"fuel_metadata.yaml not found in {data_dir_display}.\n\n"
            f"This file is required for all fuels. Please create:\n"
            f"  {metadata_file}\n\n"
            f"Minimal example:\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}  # or name of your .csv file in groupDecompositionData/\n\n"
            f"See the 'Adding Custom Fuels' documentation for more details."
        )

    try:
        with metadata_file.open("r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(
            f"Error parsing {metadata_file}:\n{e}\n\n"
            f"Make sure the file is valid YAML with proper indentation."
        ) from e

    if not data or "fuels" not in data:
        raise ValueError(
            f"Invalid metadata file {metadata_file}.\n"
            f"File must contain a 'fuels' section.\n\n"
            f"Example:\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}"
        )

    if fuel_name not in data["fuels"]:
        available = list(data["fuels"].keys())
        raise KeyError(
            f"Fuel '{fuel_name}' not found in {metadata_file}.\n\n"
            f"Available fuels: {', '.join(available) if available else 'none'}\n\n"
            f"Add an entry for '{fuel_name}':\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}"
        )

    fuel_meta = data["fuels"][fuel_name]

    if "decomp_name" not in fuel_meta:
        raise ValueError(
            f"Incomplete metadata for fuel '{fuel_name}' in {metadata_file}.\n\n"
            f"Required fields:\n"
            f"  - decomp_name: Name of the .csv file in groupDecompositionData/ (without .csv extension)\n\n"
            f"Current entry:\n"
            f"  {fuel_name}: {fuel_meta}"
        )

    return fuel_meta["decomp_name"]


def file_path(parent_dir: str | Path, old_name: str, new_name: str) -> Path:
    """
    Locate a file using the legacy naming convention.

    :param parent_dir: Directory containing the file to locate.
    :type parent_dir: str or Path
    :param old_name: Old name of the file to look for.
    :type old_name: str
    :param new_name: New name of the file to apply.
    :type new_name: str
    :return: Path to the file following the legacy naming convention.
    :rtype: Path
    """
    parent_dir = Path(parent_dir)
    new_path = parent_dir / new_name
    old_path = parent_dir / old_name
    if new_path.exists():
        return new_path
    if old_path.exists():
        logger.info(
            (
                "GCM data fetched using legacy naming convention %s.\n"
                "To maintain compatibility, consider renaming your file to %s."
            ),
            old_name,
            new_name,
        )
        return old_path
    msg = f"{old_name} (legacy) or {new_name} (current) not found in {parent_dir}."
    raise FileNotFoundError(msg)
