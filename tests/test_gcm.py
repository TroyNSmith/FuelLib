"""Tests for the Group Contribution Method (GCM) module in FuelLib."""

from pathlib import Path

import jax.numpy as jnp
import pytest
import quaxed.numpy as qnp

from fuellib import Fuel, gcm
from fuellib.utils.units import strip

from .data.answers import gcm as gcm_answers

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def gani_gcm():
    return gcm.GaniGCM()


@pytest.fixture
def jet_a():
    return Fuel("jet-a", fuelDataDir=DATA_DIR, decompName="refCompounds")


class TestConstantinouGaniMethod:
    """
    Tests for the Constantinou and Gani (1994, 1995) Group Contribution Method.

    Jet-A fuel is used as the test case for the Gani GCM method as it is a
    diverse mixture of hydrocarbons.
    """

    def test_critical_temperature(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the critical temperature prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Tc(jet_a)), gcm_answers.Gani.Tc)

    def test_critical_pressure(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the critical pressure prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Pc(jet_a)), gcm_answers.Gani.Pc)

    def test_critical_volume(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the critical volume prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Vc(jet_a)), gcm_answers.Gani.Vc)

    def test_boiling_point(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the boiling point prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Tb(jet_a)), gcm_answers.Gani.Tb)

    def test_melting_point(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the melting point prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Tm(jet_a)), gcm_answers.Gani.Tm)

    def test_heat_of_formation(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the heat of formation prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Hf(jet_a)), gcm_answers.Gani.Hf)

    def test_gibbs_free_energy(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the Gibbs free energy prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Gf(jet_a)), gcm_answers.Gani.Gf)

    def test_heat_of_vaporization_stp(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the heat of vaporization at standard temperature and pressure prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Hv_stp(jet_a)), gcm_answers.Gani.Hv_stp)

    def test_heat_capacity_stp(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the heat capacity at standard temperature and pressure prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Cp_stp(jet_a)), gcm_answers.Gani.Cp_stp)

    def test_heat_capacity_b_correction(
        self, gani_gcm: gcm.GaniGCM, jet_a: Fuel
    ) -> None:
        """Test the heat capacity B correction prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Cp_B(jet_a)), gcm_answers.Gani.Cp_B)

    def test_heat_capacity_c_correction(
        self, gani_gcm: gcm.GaniGCM, jet_a: Fuel
    ) -> None:
        """Test the heat capacity C correction prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Cp_C(jet_a)), gcm_answers.Gani.Cp_C)

    def test_molar_volume_stp(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the molar volume at standard temperature and pressure prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.Vm_stp(jet_a)), gcm_answers.Gani.Vm_stp)

    def test_acentric_factor(self, gani_gcm: gcm.GaniGCM, jet_a: Fuel) -> None:
        """Test the acentric factor prediction using the Gani GCM method."""
        assert qnp.allclose(strip(gani_gcm.omega(jet_a)), gcm_answers.Gani.omega)
