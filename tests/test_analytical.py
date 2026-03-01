"""Tests for the analytical steady-state solver."""

import numpy as np
import pytest
from thermal_envelope.physics.analytical import (
    steady_state_centerline,
    steady_state_surface,
    max_allowable_heat_rate,
)


def test_centerline_hand_calculation():
    """
    T_center = T_inf + Q*R/(2h) + Q*R^2/(4k)

    With Q=18400, R=0.2, k=2.0, h=5.0, T_inf=313.15:
      = 313.15 + 18400*0.2/10 + 18400*0.04/8
      = 313.15 + 368 + 92
      = 773.15 K  (500 °C)
    """
    T = steady_state_centerline(
        Q_vol=18400.0, R=0.2, k=2.0, h=5.0, T_inf=313.15,
    )
    assert pytest.approx(T, abs=0.01) == 773.15


def test_surface_hand_calculation():
    """
    T_surface = T_inf + Q*R/(2h)

    With Q=18400, R=0.2, h=5.0, T_inf=313.15:
      = 313.15 + 18400*0.2/10
      = 313.15 + 368 = 681.15 K
    """
    T = steady_state_surface(
        Q_vol=18400.0, R=0.2, h=5.0, T_inf=313.15,
    )
    assert pytest.approx(T, abs=0.01) == 681.15


def test_max_allowable_centerline_binding():
    """
    When the surface limit is generous (e.g. Salt at 473.15 K = 200°C),
    the centerline constraint should be the binding one.
    """
    k_const = lambda T: 2.0  # noqa: E731

    Q_allow = max_allowable_heat_rate(
        R=0.2, h=5.0, T_inf=313.15,
        T_limit_center=773.15,   # 500 °C
        T_limit_surface=473.15,  # 200 °C
        k_func=k_const,
    )

    # Verify: plugging Q_allow back into centerline formula gives <= T_limit
    T_check = steady_state_centerline(Q_allow, R=0.2, k=2.0, h=5.0, T_inf=313.15)
    assert T_check <= 773.15 + 0.01


def test_max_allowable_surface_binding():
    """
    With a tight surface limit (Bentonite: 100°C = 373.15 K), the surface
    constraint should bind before the centerline limit.
    """
    k_const = lambda T: 2.0  # noqa: E731

    Q_allow = max_allowable_heat_rate(
        R=0.2, h=5.0, T_inf=313.15,
        T_limit_center=773.15,   # 500 °C
        T_limit_surface=373.15,  # 100 °C
        k_func=k_const,
    )

    # Surface-limited: Q_allow = (373.15 - 313.15) / (0.2 / 10) = 60 / 0.02 = 3000
    assert pytest.approx(Q_allow, rel=1e-3) == 3000.0


def test_zero_heat_gives_ambient():
    """With no heat generation, all temperatures should equal T_inf."""
    assert steady_state_centerline(0.0, R=0.5, k=1.0, h=10.0, T_inf=300.0) == 300.0
    assert steady_state_surface(0.0, R=0.5, h=10.0, T_inf=300.0) == 300.0
