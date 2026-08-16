"""Unit tests for canister geometry and fleet sizing."""

import math

import numpy as np
import pytest

from aethon.design.canister import (
    canister_count,
    canister_volume,
    effective_density,
    heat_output_per_canister,
    waste_mass_per_canister,
)

_RHO = 2500.0


class TestGeometry:

    def test_volume_hand_calc(self):
        """V = pi R^2 H with H = 6R  =>  V = 6 pi R^3."""
        assert canister_volume(1.0, 6.0) == pytest.approx(6.0 * math.pi)
        assert canister_volume(0.1, 6.0) == pytest.approx(6.0 * math.pi * 1e-3)

    def test_volume_scales_cubically(self):
        """Doubling radius at fixed aspect ratio multiplies volume by 8."""
        small = canister_volume(0.1, 6.0)
        large = canister_volume(0.2, 6.0)
        assert large == pytest.approx(8.0 * small)

    def test_aspect_ratio_is_linear(self):
        """Volume is proportional to the aspect ratio."""
        assert canister_volume(0.15, 12.0) == pytest.approx(
            2.0 * canister_volume(0.15, 6.0)
        )

    def test_effective_density(self):
        """rho_eff = rho_base / (1 - f)."""
        assert effective_density(2500.0, 0.0) == pytest.approx(2500.0)
        assert effective_density(2500.0, 0.1) == pytest.approx(2500.0 / 0.9)


class TestFleetSize:

    def test_waste_mass_hand_calc(self):
        """
        R = 0.1 m, H = 6R, rho_base = 2500, f = 0.1:
            V       = pi * 0.01 * 0.6      = 0.0188496 m^3
            rho_eff = 2500 / 0.9           = 2777.78 kg/m^3
            m_waste = V * rho_eff * 0.1    = 5.2360 kg
        """
        m = waste_mass_per_canister(0.1, 0.1, _RHO, 6.0)
        assert m == pytest.approx(5.2360, rel=1e-4)

    def test_canister_count_rounds_up(self):
        """100 kg / 5.236 kg = 19.098 canisters -> 20; a part canister is a canister."""
        assert canister_count(0.1, 0.1, _RHO, 100.0, 6.0) == 20

    def test_exact_fit_does_not_round_up(self):
        """A campaign that fills canisters exactly needs no extra one."""
        per = waste_mass_per_canister(0.1, 0.1, _RHO, 6.0)
        assert canister_count(0.1, 0.1, _RHO, per * 4.0, 6.0) == 4

    def test_more_loading_needs_fewer_canisters(self):
        """Higher waste loading packs the same campaign into fewer canisters."""
        low = canister_count(0.15, 0.05, _RHO, 500.0, 6.0)
        high = canister_count(0.15, 0.20, _RHO, 500.0, 6.0)
        assert high < low

    def test_bigger_radius_needs_fewer_canisters(self):
        """Fleet size falls as the canister grows."""
        small = canister_count(0.10, 0.10, _RHO, 500.0, 6.0)
        large = canister_count(0.30, 0.10, _RHO, 500.0, 6.0)
        assert large < small

    def test_zero_loading_is_not_a_number(self):
        """A canister holding no waste cannot encapsulate a campaign."""
        assert np.isnan(canister_count(0.1, 0.0, _RHO, 100.0, 6.0))


class TestHeatOutput:

    def test_heat_output_hand_calc(self):
        """Q = V * rho_eff * f * Q_specific(t)."""
        Q = heat_output_per_canister(
            R=0.1,
            loading_fraction=0.1,
            rho_base=_RHO,
            decay_func=lambda t: 10.0,
            t_years=0.0,
            aspect_ratio=6.0,
        )
        # m_waste = 5.2360 kg, at 10 W/kg -> 52.36 W
        assert Q == pytest.approx(52.360, rel=1e-4)

    def test_heat_output_decays(self):
        """Heat output follows the decay curve."""
        decay = lambda t: 100.0 * np.exp(-0.5 * t)  # noqa: E731
        early = heat_output_per_canister(0.1, 0.1, _RHO, decay, 0.0, 6.0)
        late = heat_output_per_canister(0.1, 0.1, _RHO, decay, 5.0, 6.0)
        assert late < early
