#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =========================================================
# 課題19の設定
# f(x) = sum_{i=0}^2 f_i(x)^2
# f_i(x) = y_i - x0 (1 - x1^{i+1})
# y0=1.5, y1=2.25, y2=2.625
# =========================================================


Y_VALUES = np.array([1.5, 2.25, 2.625], dtype=float)


def fi_and_derivatives(x):
    """
    f_i(x), grad f_i(x), Hessian f_i(x) をまとめて返す
    """
    x0, x1 = x[0], x[1]
    fi_list = []
    gi_list = []
    hi_list = []

    for i, yi in enumerate(Y_VALUES):
        a = i + 1
        x1a = x1**a
        fi = yi - x0 + x0 * x1a

        # grad f_i
        dfi_dx0 = -1.0 + x1a
        dfi_dx1 = x0 * a * (x1 ** (a - 1))
        gi = np.array([dfi_dx0, dfi_dx1], dtype=float)

        # Hessian f_i
        d2fi_dx0dx0 = 0.0
        d2fi_dx0dx1 = a * (x1 ** (a - 1))
        if a == 1:
            d2fi_dx1dx1 = 0.0
        else:
            d2fi_dx1dx1 = x0 * a * (a - 1) * (x1 ** (a - 2))
        hi = np.array(
            [[d2fi_dx0dx0, d2fi_dx0dx1], [d2fi_dx0dx1, d2fi_dx1dx1]],
            dtype=float,
        )

        fi_list.append(fi)
        gi_list.append(gi)
        hi_list.append(hi)

    return fi_list, gi_list, hi_list


def evalf(x):
    """
    目的関数 f(x) = sum f_i(x)^2
    """
    fi_list, _, _ = fi_and_derivatives(x)
    return float(np.sum(np.square(fi_list)))


def evalg(x):
    """
    勾配ベクトル ∇f(x) = 2 * sum f_i(x) * ∇f_i(x)
    """
    fi_list, gi_list, _ = fi_and_derivatives(x)
    grad = np.zeros(2, dtype=float)
    for fi, gi in zip(fi_list, gi_list):
        grad += 2.0 * fi * gi
    return grad


def evalh(x):
    """
    ヘッセ行列 ∇^2 f(x)
    = 2 * sum ( f_i(x) * ∇^2 f_i(x) + ∇f_i(x) ∇f_i(x)^T )
    """
    fi_list, gi_list, hi_list = fi_and_derivatives(x)
    hess = np.zeros((2, 2), dtype=float)
    for fi, gi, hi in zip(fi_list, gi_list, hi_list):
        hess += 2.0 * (fi * hi + np.outer(gi, gi))
    return hess


@dataclass(frozen=True)
class IterationHistory2D:
    x: np.ndarray  # shape (n, 2)
    f: np.ndarray  # shape (n,)


# =========================================================
# バックトラック法（共通で使う）
# =========================================================


def backtracking(xk, dk, evalf, evalg, t_init=1.0, rho=0.5, xi=1e-4):
    """
    バックトラック法（アルミホ条件）
    """
    t = t_init
    fk = evalf(xk)
    gk = evalg(xk)
    while True:
        x_new = xk + t * dk
        if evalf(x_new) <= fk + xi * t * np.dot(dk, gk):
            break
        t *= rho
    return t


# =========================================================
# (a) バックトラック法付き最急降下法
# =========================================================


def steepest_descent_bt(
    x0,
    evalf,
    evalg,
    eps=1e-6,
    max_iter=1000,
    xi=1e-4,
    rho=0.5,
    t_init=1.0,
    *,
    return_history=False,
):
    x = x0.copy()
    history_x = [x.copy()]
    history_f = [float(evalf(x))]

    for k in range(max_iter):
        g = evalg(x)
        if np.linalg.norm(g) <= eps:
            if return_history:
                hist = IterationHistory2D(np.stack(history_x, axis=0), np.array(history_f))
                return x, evalf(x), k, hist
            return x, evalf(x), k

        dk = -g
        tk = backtracking(x, dk, evalf, evalg, t_init=t_init, rho=rho, xi=xi)
        x = x + tk * dk

        history_x.append(x.copy())
        history_f.append(float(evalf(x)))

    if return_history:
        hist = IterationHistory2D(np.stack(history_x, axis=0), np.array(history_f))
        return x, evalf(x), max_iter, hist
    return x, evalf(x), max_iter


# =========================================================
# (b) バックトラック法付きニュートン法
# =========================================================


def newton_bt(
    x0,
    evalf,
    evalg,
    evalh,
    eps=1e-6,
    max_iter=1000,
    xi=1e-4,
    rho=0.5,
    t_init=1.0,
    *,
    return_history=False,
):
    x = x0.copy()
    history_x = [x.copy()]
    history_f = [float(evalf(x))]

    for k in range(max_iter):
        g = evalg(x)
        if np.linalg.norm(g) <= eps:
            if return_history:
                hist = IterationHistory2D(np.stack(history_x, axis=0), np.array(history_f))
                return x, evalf(x), k, hist
            return x, evalf(x), k

        H = evalh(x)
        try:
            dk = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break

        tk = backtracking(x, dk, evalf, evalg, t_init=t_init, rho=rho, xi=xi)
        x = x + tk * dk

        history_x.append(x.copy())
        history_f.append(float(evalf(x)))

    if return_history:
        hist = IterationHistory2D(np.stack(history_x, axis=0), np.array(history_f))
        return x, evalf(x), max_iter, hist
    return x, evalf(x), max_iter


# =========================================================
# メイン
# =========================================================


def main():
    xi = 1e-4
    rho = 0.5
    t_init = 1.0
    x0 = np.array([2.0, 0.0])

    print("初期点 x0 =", x0)

    print("\n=== (a) バックトラック法付き最急降下法 ===")
    x_sd, f_sd, k_sd, hist_sd = steepest_descent_bt(
        x0,
        evalf,
        evalg,
        eps=1e-6,
        max_iter=1000,
        xi=xi,
        rho=rho,
        t_init=t_init,
        return_history=True,
    )
    print("最適解近似 x* ?", x_sd)
    print("最適値近似 f(x*) ?", f_sd)
    print("反復回数 =", k_sd)

    print("\n=== (b) バックトラック法付きニュートン法 ===")
    x_nt, f_nt, k_nt, hist_nt = newton_bt(
        x0,
        evalf,
        evalg,
        evalh,
        eps=1e-6,
        max_iter=1000,
        xi=xi,
        rho=rho,
        t_init=t_init,
        return_history=True,
    )
    print("最適解近似 x* ?", x_nt)
    print("最適値近似 f(x*) ?", f_nt)
    print("反復回数 =", k_nt)

    # プロット（任意）
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib が見つからないため、プロットをスキップします。")
        return

    all_points = np.vstack([hist_sd.x, hist_nt.x])
    x0_min, x1_min = np.min(all_points, axis=0)
    x0_max, x1_max = np.max(all_points, axis=0)
    margin0 = 0.2 * (x0_max - x0_min) if x0_max > x0_min else 1.0
    margin1 = 0.2 * (x1_max - x1_min) if x1_max > x1_min else 1.0

    x0_left, x0_right = x0_min - margin0, x0_max + margin0
    x1_bottom, x1_top = x1_min - margin1, x1_max + margin1

    n = 250
    grid_x0 = np.linspace(x0_left, x0_right, n)
    grid_x1 = np.linspace(x1_bottom, x1_top, n)
    X0, X1 = np.meshgrid(grid_x0, grid_x1)

    F = np.zeros_like(X0)
    for i, yi in enumerate(Y_VALUES):
        a = i + 1
        F += (yi - X0 + X0 * (X1**a)) ** 2

    plt.figure(figsize=(7, 6))
    cf = plt.contourf(X0, X1, F, levels=60, cmap="viridis")
    plt.contour(X0, X1, F, levels=20, colors="k", linewidths=0.4, alpha=0.35)
    cbar = plt.colorbar(cf)
    cbar.set_label("f(x0, x1)")

    plt.plot(hist_sd.x[:, 0], hist_sd.x[:, 1], "-o", markersize=3.5, linewidth=1.2, label="Steepest descent")
    plt.plot(hist_nt.x[:, 0], hist_nt.x[:, 1], "-s", markersize=3.5, linewidth=1.2, label="Newton")

    plt.scatter([hist_sd.x[0, 0]], [hist_sd.x[0, 1]], s=80, c="white", edgecolors="black", zorder=4)
    plt.scatter([hist_nt.x[0, 0]], [hist_nt.x[0, 1]], s=80, c="white", edgecolors="black", zorder=4)
    plt.scatter([hist_sd.x[-1, 0]], [hist_sd.x[-1, 1]], s=110, marker="*", c="white", edgecolors="black", zorder=5)
    plt.scatter([hist_nt.x[-1, 0]], [hist_nt.x[-1, 1]], s=110, marker="*", c="white", edgecolors="black", zorder=5)

    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Optimization trajectories on x0-x1 plane")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    out_path = Path(__file__).with_name("task19.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"プロットを保存しました: {out_path}")

    import matplotlib

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
