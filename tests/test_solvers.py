"""Тесты численных методов и модельных задач."""

import numpy as np
import pytest

from ode_methods.problems import (
    Projectile,
    drag_coefficient,
    euler_range,
    fly,
    stiff_exact,
    stiff_rhs,
)
from ode_methods.solvers import explicit_euler, implicit_euler, observed_order


def test_explicit_euler_on_linear_growth():
    # y' = 1, y(0) = 0 метод Эйлера решает точно.
    t, y = explicit_euler(lambda t, y: np.ones_like(y), (0, 2), [0.0], 10)
    np.testing.assert_allclose(y[:, 0], t)


@pytest.mark.parametrize("method", [explicit_euler, implicit_euler])
def test_first_order_convergence(method):
    eps = 0.01
    steps = np.array([1e-4, 5e-5, 2.5e-5])
    errors = []
    for h in steps:
        t, s = method(stiff_rhs(eps), (0, 0.2), [1.0], round(0.2 / h))
        errors.append(np.max(np.abs(s[:, 0] - stiff_exact(t, eps))))
    assert observed_order(steps, np.array(errors)) == pytest.approx(1.0, abs=0.05)


def test_implicit_euler_matches_closed_form():
    # Для ε·s' + s = 0 неявный Эйлер даёт s_n = (1 + h/ε)^(−n).
    eps, h = 0.01, 0.02
    _, s = implicit_euler(stiff_rhs(eps), (0, 0.2), [1.0], 10)
    np.testing.assert_allclose(s[:, 0], (1 + h / eps) ** -np.arange(11))


def test_explicit_euler_is_unstable_beyond_two_eps():
    eps = 0.01
    _, s = explicit_euler(stiff_rhs(eps), (0, 1), [1.0], 40)   # h = 0.025 > 2ε
    assert abs(s[-1, 0]) > 1e3


def test_implicit_euler_is_stable_for_any_step():
    eps = 0.01
    _, s = implicit_euler(stiff_rhs(eps), (0, 1), [1.0], 10)   # h = 0.1 = 10ε
    assert np.all(np.diff(s[:, 0]) <= 0) and s[-1, 0] < 1e-9


def test_vacuum_range_matches_analytic():
    p = Projectile(drag=False, thrust=0.0)
    sol = fly(p)
    expected = p.speed**2 * np.sin(2 * np.radians(p.angle_deg)) / 9.81
    assert sol.y_events[0][0][0] == pytest.approx(expected, rel=1e-6)


def test_drag_shortens_and_thrust_extends_range():
    ranges = [fly(p).y_events[0][0][0] for p in (
        Projectile(drag=False, thrust=0.0),
        Projectile(thrust=0.0),
        Projectile(),
    )]
    assert ranges[1] < ranges[0] < ranges[2]


def test_euler_range_converges_to_rk45():
    p = Projectile()
    reference = fly(p).y_events[0][0][0]
    assert abs(euler_range(p, 20000, 14.0) - reference) < 0.5


@pytest.mark.parametrize(
    "mach, cx", [(0.0, 0.45), (0.5, 0.45), (1.0, 0.95), (1.2, 0.75), (5, 0.25)]
)
def test_drag_coefficient_profile(mach, cx):
    assert drag_coefficient(mach) == pytest.approx(cx)
