#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =========================================================
# 課題17の設定
# f(x) = x0^2 + exp(x0) + x1^4 + x1^2 - 2 x0 x1 + 3
# =========================================================


def evalf(x):
    """
    目的関数 f(x) を計算する
    x: np.array shape (2,)
    """
    x0, x1 = x[0], x[1]
    f = x0**2 + np.exp(x0) + x1**4 + x1**2 - 2.0 * x0 * x1 + 3.0
    return f


def evalg(x):
    """
    勾配ベクトル ∇f(x) を計算する
    ∇f(x) = [ 2x0 + exp(x0) - 2x1,
               4x1^3 + 2x1 - 2x0 ]
    """
    x0, x1 = x[0], x[1]
    g0 = 2.0 * x0 + np.exp(x0) - 2.0 * x1
    g1 = 4.0 * x1**3 + 2.0 * x1 - 2.0 * x0
    return np.array([g0, g1])


def evalh(x):
    """
    ヘッセ行列 ∇^2 f(x) を計算する
    ∇^2 f(x) =
      [[ 2 + exp(x0),     -2          ],
       [     -2       , 12 x1^2 + 2   ]]
    """
    x0, x1 = x[0], x[1]
    h00 = 2.0 + np.exp(x0)
    h01 = -2.0
    h10 = -2.0
    h11 = 12.0 * x1**2 + 2.0
    return np.array([[h00, h01], [h10, h11]])


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
    与えられた点 xk と方向 dk に対してステップ幅 t を返す。
    """
    t = t_init
    fk = evalf(xk)
    gk = evalg(xk)
    # アルミホ条件: f(xk + t dk) <= f(xk) + ξ t <dk, gk>
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
    """
    バックトラック法付き最急降下法
    x_{k+1} = x_k + t_k d_k,  d_k = -∇f(x_k)
    """
    x = x0.copy()
    history_x = [x.copy()]
    history_f = [float(evalf(x))]

    for k in range(max_iter):
        g = evalg(x)
        norm_g = np.linalg.norm(g)
        if norm_g <= eps:
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
    """
    バックトラック法付きニュートン法
    d_k はヘッセ行列を用いて ∇^2 f(x_k) d_k = -∇f(x_k) を解く
    """
    x = x0.copy()
    history_x = [x.copy()]
    history_f = [float(evalf(x))]

    for k in range(max_iter):
        g = evalg(x)
        norm_g = np.linalg.norm(g)
        if norm_g <= eps:
            if return_history:
                hist = IterationHistory2D(np.stack(history_x, axis=0), np.array(history_f))
                return x, evalf(x), k, hist
            return x, evalf(x), k

        H = evalh(x)
        # ニュートン方向を解く: H d = -g
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
    # 共通パラメータ（課題文指定）
    xi = 1e-4
    rho = 0.5
    t_init = 1.0
    x0 = np.array([1.0, 1.0])  # 初期点 (1, 1)^T

    print("初期点 x0 =", x0)

    # (a) 最急降下法
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

    # (b) ニュートン法
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

    # x0-x1 平面に f の値を色で表示し、両手法の反復点の遷移を同一グラフに重ねる
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
    F = X0**2 + np.exp(X0) + X1**4 + X1**2 - 2.0 * X0 * X1 + 3.0

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

    out_path = Path(__file__).with_name("task18.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"プロットを保存しました: {out_path}")

    import matplotlib

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()

