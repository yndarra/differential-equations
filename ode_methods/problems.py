"""Модельные задачи лабораторной работы.

1. **Жёсткое уравнение** ``ε·s' + s = 0, s(0) = 1, ε ≪ 1`` — с точным решением
   ``s(t) = exp(−t/ε)``. На нём хорошо видна разница между явным и неявным
   методами Эйлера.
2. **Полёт снаряда** в атмосфере с сопротивлением, зависящим от числа Маха,
   и кратковременной тягой (активно-реактивный снаряд).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .solvers import explicit_euler

# ----------------------------------------------------------------------
# 1. Жёсткое линейное уравнение
# ----------------------------------------------------------------------


def stiff_rhs(eps: float) -> Callable[[float, np.ndarray], np.ndarray]:
    """Правая часть ``s' = −s/ε``."""
    return lambda t, s: -s / eps


def stiff_exact(t: np.ndarray, eps: float) -> np.ndarray:
    """Точное решение ``s(t) = exp(−t/ε)``."""
    return np.exp(-t / eps)


def explicit_euler_multiplier(h: float, eps: float) -> float:
    """Множитель перехода явного Эйлера: ``s_{n+1} = (1 − h/ε)·s_n``.

    Метод устойчив, пока ``|1 − h/ε| ≤ 1``, то есть ``h ≤ 2ε``. При ``h = 2ε``
    решение не затухает, а колеблется ``1, −1, 1, …``; при большем шаге —
    экспоненциально растёт, хотя точное решение стремится к нулю.
    """
    return 1 - h / eps


# ----------------------------------------------------------------------
# 2. Снаряд с сопротивлением воздуха и тягой
# ----------------------------------------------------------------------

G = 9.81             # ускорение свободного падения, м/с²
AIR_DENSITY = 1.24   # плотность воздуха у земли, кг/м³
SOUND_SPEED = 340.0  # скорость звука, м/с


def drag_coefficient(mach: float) -> float:
    """Коэффициент лобового сопротивления сферы ``C_x(M)``.

    Кусочно-линейная аппроксимация: постоянный ``0.45`` на дозвуке, рост до
    пика ``0.95`` в трансзвуковой области (волновой кризис), затем плавный
    спад до ``0.25`` на больших сверхзвуковых скоростях.
    """
    m = abs(mach)
    if m < 0.8:
        return 0.45
    if m < 1.0:
        return 0.45 + (0.95 - 0.45) * (m - 0.8) / 0.2
    if m < 1.2:
        return 0.95 - (0.95 - 0.75) * (m - 1.0) / 0.2
    if m < 3.0:
        return 0.75 - (0.75 - 0.25) * (m - 1.2) / 1.8
    return 0.25


@dataclass
class Projectile:
    """Параметры снаряда и начальные условия.

    Состояние системы — вектор ``Y = (x, y, Vx, Vy)``.
    """

    mass: float = 30.0                 # масса, кг
    diameter: float = 0.1              # калибр, м
    speed: float = 100.0               # начальная скорость, м/с
    angle_deg: float = 30.0            # угол бросания, градусы
    thrust: float = 500.0              # тяга двигателя, Н
    burn_time: float = 0.5             # время работы двигателя, с
    drag: bool = True                  # учитывать сопротивление воздуха

    @property
    def area(self) -> float:
        """Площадь миделя (поперечного сечения), м²."""
        return np.pi * self.diameter**2 / 4

    @property
    def y0(self) -> np.ndarray:
        a = np.radians(self.angle_deg)
        return np.array([0.0, 0.0, self.speed * np.cos(a), self.speed * np.sin(a)])

    def thrust_at(self, t: float) -> float:
        """Профиль тяги ``P(t)``: постоянная тяга в течение ``burn_time``."""
        return self.thrust if 0 <= t <= self.burn_time else 0.0

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Правые части ``dY/dt``.

        Сила сопротивления ``F = ½·ρ·V²·S·C_x(M)`` направлена против скорости,
        тяга — вдоль скорости (снаряд стабилизирован и летит носом вперёд).
        """
        _, _, vx, vy = state
        v = np.hypot(vx, vy)
        if v > 0:
            ex, ey = vx / v, vy / v           # единичный вектор скорости
        else:
            a = np.radians(self.angle_deg)
            ex, ey = np.cos(a), np.sin(a)     # на старте — по направлению пуска

        force = self.thrust_at(t)
        if self.drag:
            force -= 0.5 * AIR_DENSITY * v**2 * self.area * drag_coefficient(v / SOUND_SPEED)

        ax = force * ex / self.mass
        ay = force * ey / self.mass - G
        return np.array([vx, vy, ax, ay])


def _hit_ground(t: float, state: np.ndarray) -> float:
    """Событие для solve_ivp: высота обращается в ноль на нисходящей ветви."""
    return state[1]


_hit_ground.terminal = True
_hit_ground.direction = -1


def fly(projectile: Projectile, max_time: float = 60.0, rtol: float = 1e-8, atol: float = 1e-8):
    """Проинтегрировать полёт до падения на землю методом Дорманда–Принса (RK45).

    Момент падения находится точно, через механизм событий solve_ivp, а не
    по первой точке с отрицательной высотой.
    """
    return solve_ivp(
        projectile.rhs,
        (0.0, max_time),
        projectile.y0,
        method="RK45",
        events=_hit_ground,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        # Ограничение шага — чтобы солвер не «перешагнул» момент выключения
        # двигателя, где правая часть терпит разрыв.
        max_step=projectile.burn_time / 5,
    )


def euler_range(projectile: Projectile, n_steps: int, t_end: float) -> float:
    """Дальность полёта по явному методу Эйлера.

    Интегрируем на ``[0, t_end]`` и ищем первый переход высоты через ноль,
    уточняя точку падения линейной интерполяцией между узлами.
    """
    _, y = explicit_euler(projectile.rhs, (0.0, t_end), projectile.y0, n_steps)
    below = np.flatnonzero((y[1:, 1] < 0) & (y[:-1, 1] >= 0))
    if below.size == 0:
        raise ValueError("Снаряд не упал за отведённое время — увеличьте t_end")
    i = below[0]
    frac = y[i, 1] / (y[i, 1] - y[i + 1, 1])
    return float(y[i, 0] + frac * (y[i + 1, 0] - y[i, 0]))
