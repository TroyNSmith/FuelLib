import inspect
import unittest

import numpy as np
import unxt as u

import fuellib as fl


def _normalize_signature(sig):
    """Normalize path-like defaults so signatures are stable across machines."""

    parts = []
    for name, param in sig.parameters.items():
        text = str(param)
        if (
            name == "path"
            and param.default is not inspect.Parameter.empty
            and isinstance(param.default, str)
            and param.default.endswith("exportData")
        ):
            text = "path='<EXPORTDATA_PATH>'"
        parts.append(text)
    return f"({', '.join(parts)})"


def _public_module_functions(module):
    """Get public functions from module, including re-exported ones from __all__."""
    functions = {}
    # Check all public names in __all__
    if hasattr(module, "__all__"):
        for name in module.__all__:
            if not name.startswith("_"):
                obj = getattr(module, name, None)
                if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    functions[name] = obj
    return functions


def _public_class_methods(cls):
    return {
        name: obj
        for name, obj in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


class ApiContractTestCase(unittest.TestCase):
    def test_fuellib_module_api(self):
        print("\nFuelLib Module API:")
        expected_top_level = {
            "fuel": "class",
            "constants": "module",
            "convert": "module",
            "utility": "module",
        }

        # Check top-level API
        for name, obj_type in expected_top_level.items():
            self.assertTrue(
                hasattr(fl, name),
                msg=f"FuelLib module missing expected attribute: {name}",
            )
            if obj_type == "class":
                self.assertTrue(
                    inspect.isclass(getattr(fl, name)),
                    msg=f"FuelLib.{name} should be a class",
                )
            elif obj_type == "module":
                import types

                self.assertTrue(
                    isinstance(getattr(fl, name), types.ModuleType),
                    msg=f"FuelLib.{name} should be a module",
                )
            else:
                print(f"  ✓ {name} ({obj_type})")

        # Check that constants are available via module
        self.assertTrue(
            hasattr(fl.constants, "k_B"), msg="FuelLib.constants.k_B not found"
        )
        self.assertTrue(
            hasattr(fl.constants, "N_A"), msg="FuelLib.constants.N_A not found"
        )
        print("  ✓ constants.k_B (constant)")
        print("  ✓ constants.N_A (constant)")
        print("\nFuelLib.convert Module API:")
        convert_funcs = {
            "C2K": "(T)",
            "K2C": "(T)",
            "C2F": "(T)",
            "F2C": "(T)",
            "F2K": "(T)",
            "K2F": "(T)",
            "epsilon_to_characteristic_temperature": "(epsilon_j_per_mol)",
        }
        for name, sig_expected in convert_funcs.items():
            self.assertTrue(
                hasattr(fl.convert, name), msg=f"fuellib.convert missing: {name}"
            )
            func = getattr(fl.convert, name)
            actual_sig = _normalize_signature(inspect.signature(func))
            self.assertEqual(
                actual_sig,
                sig_expected,
                msg=f"fuellib.convert.{name} signature changed",
            )
            print(f"  ✓ {name}{actual_sig}")

        # Check utility submodule
        print("\nFuelLib.utility Module API:")
        utility_funcs = {
            "mixing_rule": "(var_n, X, pseudo_prop='arithmetic')",
            "droplet_volume": "(r)",
            "droplet_mass": "(fuel, r, Yi, T)",
        }
        for name, sig_expected in utility_funcs.items():
            self.assertTrue(
                hasattr(fl.utility, name), msg=f"fuellib.utility missing: {name}"
            )
            func = getattr(fl.utility, name)
            actual_sig = _normalize_signature(inspect.signature(func))
            self.assertEqual(
                actual_sig,
                sig_expected,
                msg=f"fuellib.utility.{name} signature changed",
            )
            print(f"  ✓ {name}{actual_sig}")

        # Check constants submodule
        print("\nFuelLib.constants Module API:")
        constants_vals = {
            "k_B": "Boltzmann constant",
            "N_A": "Avogadro number",
        }
        for name in constants_vals:
            self.assertTrue(
                hasattr(fl.constants, name), msg=f"fuellib.constants missing: {name}"
            )
            val = getattr(fl.constants, name)
            self.assertIsInstance(
                val, u.Quantity, msg=f"fuellib.constants.{name} should be a Quantity"
            )
            print(f"  ✓ {name}")

    def test_fuellib_class_api(self):
        print("\nFuelLib.fuel Class API:")
        expected = {
            "Cl": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "Cp": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "X2Y": "(self, Xi: jax.Array)",
            "Y2X": "(self, Yi: jax.Array)",
            "density": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "diffusion_coeff": "(self, p: unxt._src.quantity.quantity.Quantity, T: unxt._src.quantity.quantity.Quantity, sigma_gas: float = 3.62e-10, epsilonByKB_gas: float = 97.0, MW_gas: float = 0.02897, correlation: str = 'Tee')",
            "latent_heat_vaporization": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "mass2X": "(self, mass: unxt._src.quantity.quantity.Quantity)",
            "mass2Y": "(self, mass: unxt._src.quantity.quantity.Quantity)",
            "mean_molecular_weight": "(self, Yi: jax.Array)",
            "mixture_density": "(self, Yi: jax.Array, T: unxt._src.quantity.quantity.Quantity)",
            "mixture_dynamic_viscosity": "(self, Yi: jax.Array, T: unxt._src.quantity.quantity.Quantity, correlation: str = 'Kendall-Monroe')",
            "mixture_kinematic_viscosity": "(self, Yi: jax.Array, T: unxt._src.quantity.quantity.Quantity, correlation: str = 'Kendall-Monroe')",
            "mixture_surface_tension": "(self, Yi: jax.Array, T: unxt._src.quantity.quantity.Quantity, correlation: Literal['Pitzer', 'Brock-Bird'] = 'Brock-Bird')",
            "mixture_thermal_conductivity": "(self, Yi: jax.Array, T: unxt._src.quantity.quantity.Quantity)",
            "mixture_vapor_pressure": "(self, Yi: jax.Array, T: unxt._src.quantity.quantity.Quantity, correlation: str = 'Lee-Kesler')",
            "mixture_vapor_pressure_antoine_coeffs": "(self, Yi: jax.Array, Tvals: numpy.ndarray | None = None, units: str = 'mks', correlation: str = 'Lee-Kesler')",
            "molar_liquid_vol": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "psat": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None, correlation: str = 'Lee-Kesler')",
            "psat_antoine_coeffs": "(self, Tvals: numpy.ndarray | None = None, units: str = 'mks', correlation: str = 'Lee-Kesler')",
            "surface_tension": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None, correlation: str = 'Brock-Bird')",
            "thermal_conductivity": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "viscosity_dynamic": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
            "viscosity_kinematic": "(self, T: unxt._src.quantity.quantity.Quantity, comp_idx: int | None = None)",
        }

        actual = _public_class_methods(fl.fuel)
        self.assertEqual(
            set(actual.keys()),
            set(expected.keys()),
            msg=(
                "FuelLib.fuel public method list changed. "
                f"Expected: {sorted(expected.keys())}; Found: {sorted(actual.keys())}"
            ),
        )

        for name in sorted(expected.keys()):
            actual_sig = _normalize_signature(inspect.signature(actual[name]))
            self.assertEqual(
                actual_sig,
                expected[name],
                msg=f"FuelLib.fuel method signature changed: {name}",
            )
            print(f"  ✓ fuel.{name}{actual_sig}")


class FuelLibFunctionEvalTestCase(unittest.TestCase):
    """Smoke test all current FuelLib public functions for single and multicomponent fuels."""

    @classmethod
    def setUpClass(cls):
        cls.fuels = {
            "decane": fl.fuel("decane"),
            "posf10325": fl.fuel("posf10325"),
        }
        cls.T = u.Q(320.0, "K")
        cls.p = u.Q(101325.0, "Pa")

    def _assert_finite_and_positive(self, value):
        # Quantity can't become a bare array without naming a unit; the raw
        # value is enough for a finite/positive check regardless of unit.
        if isinstance(value, u.Quantity):
            value = value.value
        arr = np.asarray(value)
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertTrue(np.all(arr > 0.0))

    def test_all_fuel_methods_for_single_and_multicomponent(self):
        print("\n")  # Add newline to separate from unittest verbose output

        for fuel_name, fuel in self.fuels.items():
            print(f"\n{fuel_name.upper()}:")

            with self.subTest(fuel=fuel_name):
                Yi = fuel.Y_0.copy()
                self.assertAlmostEqual(np.sum(Yi), 1.0)

                # Utility functions (run per fuel for consistent CI grouping)
                print("  Utility Functions:")
                self.assertAlmostEqual(fl.convert.C2K(25.0).ustrip("K"), 298.15)
                print("    ✓ convert.C2K")
                self.assertAlmostEqual(fl.convert.K2C(u.Q(298.15, "K")), 25.0)
                print("    ✓ convert.K2C")
                self.assertAlmostEqual(
                    fl.utility.droplet_volume(u.Q(1e-4, "m")).ustrip("m^3"),
                    4.0 / 3.0 * np.pi * (1e-4) ** 3,
                )
                print("    ✓ utility.droplet_volume")

                Xi = fuel.Y2X(Yi)
                self._assert_finite_and_positive(
                    fl.utility.mixing_rule(fuel.Tc, Xi, pseudo_prop="arithmetic")
                )
                print("    ✓ utility.mixing_rule (arithmetic)")
                self._assert_finite_and_positive(
                    fl.utility.mixing_rule(fuel.Tc, Xi, pseudo_prop="geometric")
                )
                print("    ✓ utility.mixing_rule (geometric)")

                # Composition conversion methods
                print("  Composition Conversions:")
                methods_to_test = [
                    (
                        "mean_molecular_weight",
                        lambda fuel=fuel, Yi=Yi: fuel.mean_molecular_weight(Yi),
                    ),
                ]
                for method_name, method_call in methods_to_test:
                    try:
                        result = method_call()
                        self._assert_finite_and_positive(result)
                        print(f"    ✓ {method_name}")
                    except AssertionError as e:
                        print(f"    ✗ {method_name}: {e}")
                        raise

                Yi_back = fuel.X2Y(Xi)
                self.assertTrue(np.allclose(Yi, Yi_back, rtol=1e-10, atol=1e-12))
                print("    ✓ Y2X/X2Y roundtrip")

                mass = u.Q(Yi * 1.0e-6, "kg")
                self.assertTrue(
                    np.allclose(fuel.mass2Y(mass), Yi, rtol=1e-10, atol=1e-12)
                )
                print("    ✓ mass2Y")

                Xi_from_mass = fuel.mass2X(mass)
                self.assertTrue(np.allclose(np.sum(Xi_from_mass), 1.0))
                print("    ✓ mass2X")

                # Component properties (all components and explicit component index)
                print("  Component Properties (both all and indexed):")
                component_methods = [
                    "density",
                    "viscosity_kinematic",
                    "viscosity_dynamic",
                    "Cp",
                    "Cl",
                    "psat",
                    "molar_liquid_vol",
                    "latent_heat_vaporization",
                    "surface_tension",
                    "thermal_conductivity",
                ]
                for method_name in component_methods:
                    method = getattr(fuel, method_name)
                    self._assert_finite_and_positive(method(self.T))
                    self._assert_finite_and_positive(method(self.T, comp_idx=0))
                    print(f"    ✓ {method_name}")

                # Alternate correlation branches
                print("  Alternate Correlations:")
                alt_methods = [
                    (
                        "psat (Ambrose-Walton)",
                        lambda fuel=fuel: fuel.psat(
                            self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "surface_tension (Pitzer)",
                        lambda fuel=fuel: fuel.surface_tension(
                            self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "diffusion_coeff (Tee)",
                        lambda fuel=fuel: fuel.diffusion_coeff(
                            self.p, self.T, correlation="Tee"
                        ),
                    ),
                    (
                        "diffusion_coeff (Wilke)",
                        lambda fuel=fuel: fuel.diffusion_coeff(
                            self.p, self.T, correlation="Wilke"
                        ),
                    ),
                ]
                for method_name, method_call in alt_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Antoine coefficient fits (individual compounds)
                A, B, C, D = fuel.psat_antoine_coeffs(
                    Tvals=np.array([300.0, 340.0]),
                    units="atm",
                    correlation="Lee-Kesler",
                )
                self.assertEqual(len(A), fuel.num_compounds)
                self.assertEqual(len(B), fuel.num_compounds)
                self.assertEqual(len(C), fuel.num_compounds)
                self.assertEqual(len(D), fuel.num_compounds)
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive(A)
                self._assert_finite_and_positive(B)
                self._assert_finite_and_positive(D)
                self.assertTrue(np.all(np.isfinite(C)))
                print("    ✓ psat_antoine_coeffs (individual)")

                # Antoine coefficient fits (mixture)
                A_mix, B_mix, C_mix, D_mix = fuel.mixture_vapor_pressure_antoine_coeffs(
                    Yi,
                    Tvals=np.array([300.0, 340.0]),
                    units="bar",
                    correlation="Lee-Kesler",
                )
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive([A_mix, B_mix, D_mix])
                self.assertTrue(np.isfinite(C_mix))
                print("    ✓ mixture_vapor_pressure_antoine_coeffs")

                # Mixture properties
                print("  Mixture Properties:")
                mixture_methods = [
                    (
                        "mixture_density",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_density(Yi, self.T),
                    ),
                    (
                        "mixture_kinematic_viscosity (Kendall-Monroe)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_kinematic_viscosity(
                            Yi, self.T, correlation="Kendall-Monroe"
                        ),
                    ),
                    (
                        "mixture_kinematic_viscosity (Arrhenius)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_kinematic_viscosity(
                            Yi, self.T, correlation="Arrhenius"
                        ),
                    ),
                    (
                        "mixture_dynamic_viscosity",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_dynamic_viscosity(
                            Yi, self.T
                        ),
                    ),
                    (
                        "mixture_vapor_pressure (Lee-Kesler)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_vapor_pressure(
                            Yi, self.T, correlation="Lee-Kesler"
                        ),
                    ),
                    (
                        "mixture_vapor_pressure (Ambrose-Walton)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_vapor_pressure(
                            Yi, self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "mixture_surface_tension (Brock-Bird)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_surface_tension(
                            Yi, self.T, correlation="Brock-Bird"
                        ),
                    ),
                    (
                        "mixture_surface_tension (Pitzer)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_surface_tension(
                            Yi, self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "mixture_thermal_conductivity",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_thermal_conductivity(
                            Yi, self.T
                        ),
                    ),
                ]
                for method_name, method_call in mixture_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Droplet helpers
                print("  Droplet Properties:")
                m = fl.utility.droplet_mass(fuel, u.Q(2.0e-5, "m"), Yi, self.T)
                self.assertEqual(m.shape, fuel.MW.shape)
                self.assertTrue(np.all(m.value >= 0.0))
                zero_mass = fl.utility.droplet_mass(fuel, u.Q(0.0, "m"), Yi, self.T)
                self.assertTrue(np.allclose(zero_mass.ustrip("kg"), 0.0))
                print("    ✓ utility.droplet_mass")

    def test_array_T_matches_scalar_loop(self):
        """Array-valued T must reproduce stacking the scalar-T call for every element."""
        T_vals = [300.0, 320.0, 350.0]
        T_arr = u.Q(np.array(T_vals), "K")

        for fuel_name, fuel in self.fuels.items():
            with self.subTest(fuel=fuel_name):
                Yi = fuel.Y_0.copy()

                # Per-compound methods, comp_idx=None -> (num_times, num_compounds)
                # and comp_idx=0 -> (num_times,)
                component_methods = [
                    "density",
                    "viscosity_kinematic",
                    "viscosity_dynamic",
                    "Cp",
                    "Cl",
                    "psat",
                    "molar_liquid_vol",
                    "latent_heat_vaporization",
                    "surface_tension",
                    "thermal_conductivity",
                ]
                for method_name in component_methods:
                    method = getattr(fuel, method_name)

                    batched = np.asarray(method(T_arr).value)
                    stacked = np.stack(
                        [np.asarray(method(u.Q(t, "K")).value) for t in T_vals]
                    )
                    self.assertEqual(batched.shape, (len(T_vals), fuel.num_compounds))
                    np.testing.assert_allclose(batched, stacked, rtol=1e-10, atol=1e-12)

                    batched_i = np.asarray(method(T_arr, comp_idx=0).value)
                    stacked_i = np.stack(
                        [
                            np.asarray(method(u.Q(t, "K"), comp_idx=0).value)
                            for t in T_vals
                        ]
                    )
                    self.assertEqual(batched_i.shape, (len(T_vals),))
                    np.testing.assert_allclose(
                        batched_i, stacked_i, rtol=1e-10, atol=1e-12
                    )
                print("    ✓ array-T component methods match scalar loop")

                # diffusion_coeff has no comp_idx: always all compounds
                for correlation in ("Tee", "Wilke"):
                    batched = np.asarray(
                        fuel.diffusion_coeff(
                            self.p, T_arr, correlation=correlation
                        ).value
                    )
                    stacked = np.stack(
                        [
                            np.asarray(
                                fuel.diffusion_coeff(
                                    self.p, u.Q(t, "K"), correlation=correlation
                                ).value
                            )
                            for t in T_vals
                        ]
                    )
                    self.assertEqual(batched.shape, (len(T_vals), fuel.num_compounds))
                    np.testing.assert_allclose(batched, stacked, rtol=1e-10, atol=1e-12)
                print("    ✓ array-T diffusion_coeff (Tee/Wilke) match scalar loop")

                # Mixture methods: array T -> (num_times,)
                mixture_calls = [
                    lambda T, fuel=fuel, Yi=Yi: fuel.mixture_density(Yi, T),
                    lambda T, fuel=fuel, Yi=Yi: fuel.mixture_kinematic_viscosity(Yi, T),
                    lambda T, fuel=fuel, Yi=Yi: fuel.mixture_dynamic_viscosity(Yi, T),
                    lambda T, fuel=fuel, Yi=Yi: fuel.mixture_vapor_pressure(Yi, T),
                    lambda T, fuel=fuel, Yi=Yi: fuel.mixture_surface_tension(Yi, T),
                    lambda T, fuel=fuel, Yi=Yi: fuel.mixture_thermal_conductivity(
                        Yi, T
                    ),
                ]
                for call in mixture_calls:
                    batched = np.asarray(call(T_arr).value)
                    stacked = np.array(
                        [float(np.asarray(call(u.Q(t, "K")).value)) for t in T_vals]
                    )
                    self.assertEqual(batched.shape, (len(T_vals),))
                    np.testing.assert_allclose(batched, stacked, rtol=1e-10, atol=1e-12)
                print("    ✓ array-T mixture methods match scalar loop")


if __name__ == "__main__":
    unittest.main()
