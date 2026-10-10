"""Численные методы решения задачи Коши ``y' = f(t, y), y(t0) = y0``.

Реализованы два одношаговых метода первого порядка:

* **явный метод Эйлера** ``y_{n+1} = y_n + h·f(t_n, y_n)`` — прост, но условно
  устойчив: для жёстких задач шаг должен быть очень маленьким;
* **неявный метод Эйлера** ``y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1})`` —
  на каждом шаге решается (в общем случае нелинейное) уравнение, зато метод
  A-устойчив и годится для жёстких задач при любом шаге.

Для сравнения с методами высокого порядка используется
:func:`scipy.integrate.solve_ivp` (Дорманд–Принс 5(4), он же ``dopri5``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def _grid(t0: float, t1: float, n_steps: int) -> tuple[np.ndarray, float]:
    """Равномерная сетка из ``n_steps`` шагов (``n_steps + 1`` узлов)."""
    if n_steps < 1:
        raise ValueError("Число шагов должно быть положительным")
    t = np.linspace(t0, t1, n_steps + 1)
    return t, (t1 - t0) / n_steps


def explicit_euler(f: RHS, t_span: tuple[float, float], y0, n_steps: int):
    """Явный метод Эйлера. Возвращает сетку ``t`` и решение ``y`` (узлы × компоненты)."""
    t, h = _grid(*t_span, n_steps)
    y = np.empty((t.size, np.size(y0)))
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = y[i] + h * np.asarray(f(t[i], y[i]))
    return t, y


def _newton(g: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,
            tol: float = 1e-12, max_iter: int = 50) -> np.ndarray:
    """Решить ``g(x) = 0`` методом Ньютона с конечно-разностным якобианом.

    Критерий остановки — относительный размер поправки: так метод корректно
    работает и на решениях, близких к нулю (там, где абсолютная невязка
    уже на уровне машинной точности).
    """
    x = np.array(x0, dtype=float)
    n = x.size
    for _ in range(max_iter):
        gx = g(x)
        # Якобиан по столбцам: J[:, j] ≈ (g(x + δ·e_j) − g(x)) / δ.
        jac = np.empty((n, n))
        for j in range(n):
            delta = 1e-7 * max(1.0, abs(x[j]))
            dx = np.zeros(n)
            dx[j] = delta
            jac[:, j] = (g(x + dx) - gx) / delta
        step = np.linalg.solve(jac, -gx)
        x += step
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(x)):
            break
    return x


def implicit_euler(f: RHS, t_span: tuple[float, float], y0, n_steps: int):
    """Неявный метод Эйлера.

    На каждом шаге уравнение ``Y − y_n − h·f(t_{n+1}, Y) = 0`` решается
    относительно ``Y`` методом Ньютона; начальное приближение — результат
    явного шага. Для линейной задачи Ньютон сходится за одну итерацию.
    """
    t, h = _grid(*t_span, n_steps)
    y = np.empty((t.size, np.size(y0)))
    y[0] = y0
    for i in range(n_steps):
        guess = y[i] + h * np.asarray(f(t[i], y[i]))
        y[i + 1] = _newton(lambda Y, i=i: Y - y[i] - h * np.asarray(f(t[i + 1], Y)), guess)
    return t, y


def observed_order(steps: np.ndarray, errors: np.ndarray) -> float:
    """Наблюдаемый порядок сходимости — наклон прямой ``log(error)`` от ``log(h)``."""
    slope, _ = np.polyfit(np.log(steps), np.log(errors), 1)
    return float(slope)
