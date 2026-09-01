# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The [`keepachangelog`](https://pypi.org/project/keepachangelog/) package is a dependency
used to parse and validate this file's entries against that format.

## [Unreleased]

### Added
- Pixi task automation (`fmt`, `lint`, `types`, `imports`, `test`, `pre-commit`, `docs-build`, `docs-clean`) so common dev workflows run via `pixi run <task>`.
- New dev dependencies: `ruff`, `ty`, `pytest-cov`, `lefthook`, `import-linter`, and `uv` for a faster local pip/venv workflow.
- `keepachangelog` dependency for maintaining this `CHANGELOG.md` in the Keep a Changelog format.
- Lefthook pre-commit suite (`lefthook.yaml`) running `fmt` → `lint` → `types` → `test` → `check-clean` on commit. The `import-linter` check is not yet wired into pre-commit since the layering contract (`fuellib.fuel` / `fuellib.gcm` / `fuellib.comp`) will fail broadly until the codebase is reorganized to match it; run it manually via `pixi run imports` in the meantime.
- Coverage reporting via `pytest-cov`, with a temporary `fail_under = 20` threshold, to be raised as test coverage improves.
- `unxt` and `jax` dependencies to give all physically-dimensioned values (fuel properties, `constants.k_B`/`N_A`, conversion results, etc.) explicit, unit-checked `unxt.Quantity` types instead of bare floats/arrays.
- `fuellib/units.py` with a shared `load_quantity(arr, native_unit, target_unit=None)` helper for constructing `unxt.Quantity` values.
- `fuel` class gained `num_compounds` and `num_groups` properties.
- `density`, `viscosity_kinematic`, `viscosity_dynamic`, `Cp`, `Cl`, `psat`, `molar_liquid_vol`, `latent_heat_vaporization`, `surface_tension`, `thermal_conductivity`, and `diffusion_coeff` now accept array-valued temperatures (in addition to scalars), returning a batched result equivalent to looping the scalar call over each temperature.

### Changed
- Replaced Black with Ruff + ty: `ruff format`/`ruff check` now handle formatting and linting, and `ty check` handles static type checking; `fl-format` now shells out to `ruff format`.
- Bumped `requires-python` to `>=3.12,<3.14` (from `>=3.8`); CI now runs on Python 3.12.
- CI's `Formatting` job (previously `psf/black`) now runs `ruff format --check`, `ruff check`, and `ty check`.
- Migrated internal numeric arrays from `numpy` to `jax.numpy`, with `jax_enable_x64` enabled at import time to preserve `numpy`'s float64 precision.
- **Breaking:** `fuel.py`, `constants.py`, `convert.py`, `utility.py`, both exporters (`converge.py`/`pele.py`), and `cli/plotting.py` now take/return `unxt.Quantity` for every dimensioned value (fuel attributes, method args/returns, `constants.k_B`/`N_A`, `convert.C2K`/`F2K`). `convert.K2C`/`K2F` still return plain floats since Celsius/Fahrenheit are offset units that can't round-trip as a `Quantity`. Mass/mole fractions (`Y_0`, `Yi`, `Xi`) and categorical/string attributes (`hc_type`, `fam`, `formulas`, `pelephysics_keys`, `compounds`) remain plain arrays.
- CLI temperature/epsilon converters (`fl-C2K`, `fl-K2C`, `fl-F2K`, `fl-K2F`, `fl-eps2K`) updated to construct/strip `unxt.Quantity` around `convert` calls.

### Removed
- Removed Black as a dev dependency.

### Fixed
- Addressed 40+ Ruff linting errors across the codebase:
  - B023: Fixed lambda variable binding in test loops by capturing loop variables with default parameters (15 fixes in `tests/test_api.py`).
  - SIM102: Combined nested `if` statements using `and` operator (7 fixes across `tests/test_hc_identification.py` and `tests/test_source_docstrings.py`).
  - BLE001: Replaced overly broad `except Exception` clauses with specific exception types (4 fixes in `fuellib/__init__.py`, `fuellib/exporters/pele.py`, `tests/test_exporters.py`).
  - PLW1510: Added explicit `check=False` argument to `subprocess.run()` calls (2 fixes in `tests/test_exporters.py`, `tests/test_utilities.py`).
  - RUF059: Prefixed unused unpacked variables with underscore (2 fixes in `tests/test_utilities.py`).
  - DTZ005: Added timezone argument to `datetime.now()` call in `fuellib/exporters/pele.py`.
  - UP036: Updated outdated Python version check in `fuellib/_data_locator.py`.
  - PLC0206: Fixed dictionary iteration to use `.items()` in `tests/baselinePredictions/generate_baseline.py`.

- Addressed 10+ ty typing errors across the codebase.
- Additional ty/Ruff fixes introduced by the `unxt` migration: cast `unxt.Quantity` construction results in `load_quantity` so ty doesn't treat them as an ambiguous/unknown type, and moved `unxt.Quantity` function-argument defaults (e.g. gas-property defaults in `diffusion_coeff`) to plain floats wrapped internally, since Ruff/ty flag function calls in argument defaults.

## [3.0.1] - 2026-06-25

### Added
- New CLI commands: `fl-C2K`, `fl-K2C`, `fl-C2F`, `fl-F2C`, `fl-F2K`, `fl-K2F` (temperature conversions), `fl-eps2K` (Lennard-Jones epsilon to characteristic temperature), `fl-export-converge`, `fl-export-pele` (CFD export), `fl-plt-comp`, `fl-plt-props` (plotting), and `fl-fuels` (list available fuels).
- `fuellib/cli/` subpackage containing all command-line tools.
- `test_exporters.py`: integration tests for export commands.
- `test_utilities.py` and `test_hc_identification.py`: unit tests for utility functions and hydrocarbon classification logic.

### Changed
- Split monolithic `FuelLib.py` into `constants.py`, `convert.py`, `utility.py`, and `fuel.py`.
- Renamed `source` package to `fuellib` and added `pyproject.toml` for distribution via pip and conda, with proper entry point configuration.
- Switched to editable/development installs (`pip install -e .` and `pip install -e '.[dev]'`).
- Simplified CI exporter job from 8 individual steps to a single `test_exporters.py` call.
- Updated `sourcecode.rst` to reflect the new file organization.
- **Breaking:** functions moved from the `fuellib` namespace to submodules: `fl.C2K()` → `fl.convert.C2K()`, `fl.mixing_rule()` → `fl.utility.mixing_rule()`. `fl.k_B` still works, but `fl.constants.k_B` is recommended.

### Fixed
- Fixed CSV file path in `fuelprops.rst`: `../../fuelData/` → `../fuellib/data/fuelData/`.
- Fixed GitHub Actions failures related to decomposition metadata.
- Fixed error handling for Jet A and cycloaromatic compounds.

## [0.0.0] - YYYY-MM-DD

### Added
- Feature 1
- Feature 2...

### Fixed
- Fix 1
- Fix 2...

### Changed
- Change 1
- Change 2...
