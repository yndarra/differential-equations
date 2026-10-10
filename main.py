"""Точка входа: решает обе задачи и сохраняет графики в папку ``images/``.

Примеры::

    python main.py            # обе задачи
    python main.py stiff      # задача 1: жёсткое уравнение ε·s' + s = 0
    python main.py ballistic  # задача 2: снаряд с сопротивлением и тягой
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ode_methods.problems import (
    SOUND_SPEED,
    Projectile,
    euler_range,
    explicit_euler_multiplier,
    fly,
    stiff_exact,
    stiff_rhs,
)
from ode_methods.solvers import explicit_euler, implicit_euler, observed_order

EXACT_STYLE = {"color": "black", "lw": 1.5, "label": "Exact"}
METHODS = {"Explicit Euler": explicit_euler, "Implicit Euler": implicit_euler}
ROCKET = "Drag + thrust 500 N for 0.5 s"


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Задача 1. Жёсткое уравнение
# ----------------------------------------------------------------------
def task_stiff(out: Path) -> None:
    t = np.linspace(0, 1, 1000)

    # 1. Точные решения при двух ε.
    fig, ax = plt.subplots(figsize=(7, 4))
    for eps in (0.1, 0.01):
        ax.plot(t, stiff_exact(t, eps), label=f"ε = {eps}")
    ax.set(xlabel="t", ylabel="s(t)", title=r"Exact solution $s(t) = e^{-t/\varepsilon}$")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, out / "stiff_exact.png")

    # 2. Явный и неявный Эйлер при ε = 0.01 с шагами 0.02 и 0.002.
    eps = 0.01
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (name, method) in zip(axes, METHODS.items(), strict=True):
        ax.plot(t, stiff_exact(t, eps), **EXACT_STYLE)
        for h, marker in ((0.02, "o-"), (0.002, ".")):
            tn, sn = method(stiff_rhs(eps), (0, 1), [1.0], round(1 / h))
            ax.plot(tn, sn[:, 0], marker, ms=4, lw=1, label=f"Δt = {h}  (Δt/ε = {h / eps:g})")
        ax.set(xlabel="t", title=f"{name}, ε = {eps}", xlim=(0, 0.3))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("s(t)")
    save(fig, out / "stiff_euler.png")
    for h in (0.02, 0.002):
        print(f"  явный Эйлер, Δt = {h}: множитель перехода 1 − Δt/ε = "
              f"{explicit_euler_multiplier(h, eps):+.2f}")

    # 3. Сходимость и устойчивость: ошибка в зависимости от шага.
    steps = np.logspace(-5, np.log10(0.05), 22)
    errors = {"Explicit Euler": [], "Implicit Euler": []}
    for h in steps:
        n = round(1 / h)
        for name, method in METHODS.items():
            tn, sn = method(stiff_rhs(eps), (0, 1), [1.0], n)
            errors[name].append(np.max(np.abs(sn[:, 0] - stiff_exact(tn, eps))))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, err in errors.items():
        ax.loglog(steps, err, "o-", ms=4, label=name)
    ax.loglog(steps, steps / steps[0] * errors["Implicit Euler"][0] * 0.5, "k--", lw=1,
              label="Slope 1 (first order)")
    ax.axvline(2 * eps, color="tab:red", ls=":", label="Stability limit Δt = 2ε")
    ax.set(xlabel="Step Δt", ylabel="Max absolute error", ylim=(1e-4, 1e3),
           title=f"Convergence and stability, ε = {eps}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save(fig, out / "stiff_convergence.png")

    # Порядок оцениваем в асимптотической области — при Δt ≪ ε.
    small = steps <= 0.1 * eps
    for name, err in errors.items():
        order = observed_order(steps[small], np.array(err)[small])
        print(f"  {name:<15} наблюдаемый порядок: {order:.2f}")


# ----------------------------------------------------------------------
# Задача 2. Снаряд с сопротивлением и тягой
# ----------------------------------------------------------------------
def task_ballistic(out: Path) -> None:
    cases = {
        "Vacuum, no thrust": Projectile(drag=False, thrust=0.0),
        "Drag, no thrust": Projectile(thrust=0.0),
        ROCKET: Projectile(),
    }
    solutions = {name: fly(p) for name, p in cases.items()}

    vacuum = cases["Vacuum, no thrust"]
    analytic = vacuum.speed**2 * np.sin(2 * np.radians(vacuum.angle_deg)) / 9.81
    for name, sol in solutions.items():
        x_land, t_land = sol.y_events[0][0][0], sol.t_events[0][0]
        print(f"  {name:<30} дальность {x_land:7.1f} м, время полёта {t_land:5.2f} с, "
              f"высота {sol.y[1].max():6.1f} м")
    print(f"  (аналитическая дальность в вакууме: {analytic:.1f} м)")

    # Траектории.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, sol in solutions.items():
        ax.plot(sol.y[0], sol.y[1], label=name)
    # Явный Эйлер с грубым шагом для сравнения.
    rocket = cases[ROCKET]
    t_end = solutions[ROCKET].t_events[0][0]
    _, yn = explicit_euler(rocket.rhs, (0, t_end), rocket.y0, 20)
    ax.plot(yn[:, 0], yn[:, 1], "o--", ms=3, lw=1, color="gray",
            label="Explicit Euler, 20 steps (drag + thrust)")
    ax.set(xlabel="x, m", ylabel="y, m", title="Trajectories (RK45, Dormand–Prince)",
           ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, out / "trajectories.png")

    # Кинематика и профиль тяги для снаряда с двигателем.
    projectile = cases[ROCKET]
    sol = solutions[ROCKET]
    t = np.linspace(0, sol.t_events[0][0], 600)
    x, y, vx, vy = sol.sol(t)
    speed = np.hypot(vx, vy)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t, [projectile.thrust_at(ti) for ti in t], color="tab:red")
    axes[0].set(xlabel="t, s", ylabel="P, N", title="Thrust profile P(t)")
    axes[1].plot(t, x, label="x, m")
    axes[1].plot(t, y, label="y, m")
    axes[1].set(xlabel="t, s", title="Coordinates")
    axes[2].plot(t, vx, label="Vx")
    axes[2].plot(t, vy, label="Vy")
    axes[2].plot(t, speed, "k", label="|V|")
    max_mach = speed.max() / SOUND_SPEED
    axes[2].set(xlabel="t, s", ylabel="m/s", title=f"Velocity (max Mach {max_mach:.2f})")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend()
    axes[2].legend()
    save(fig, out / "kinematics.png")

    # Сходимость явного Эйлера по дальности.
    reference = sol.y_events[0][0][0]
    n_steps = np.unique(np.logspace(1.5, 4.5, 13).astype(int))
    t_end = 14.0
    errors = np.array([abs(euler_range(projectile, n, t_end) - reference) for n in n_steps])
    h = t_end / n_steps
    print(f"  явный Эйлер по дальности: наблюдаемый порядок {observed_order(h, errors):.2f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(h, errors, "o-", label="Explicit Euler")
    ax.loglog(h, errors[-1] * h / h[-1], "k--", lw=1, label="Slope 1")
    ax.set(xlabel="Step Δt, s", ylabel="Range error vs RK45, m",
           title="Explicit Euler converges with first order")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save(fig, out / "ballistic_convergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerical solution of ODEs")
    parser.add_argument("task", nargs="?", default="all", choices=["all", "stiff", "ballistic"])
    parser.add_argument("--out", type=Path, default=Path("images"), help="папка для картинок")
    args = parser.parse_args()

    if args.task in ("all", "stiff"):
        print("Задача 1: жёсткое уравнение ε·s' + s = 0")
        task_stiff(args.out)
    if args.task in ("all", "ballistic"):
        print("Задача 2: снаряд с сопротивлением воздуха и тягой")
        task_ballistic(args.out)
    print(f"Картинки сохранены в {args.out.resolve()}")


if __name__ == "__main__":
    main()
