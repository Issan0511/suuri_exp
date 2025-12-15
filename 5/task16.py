#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 課題16
# f(x) = (1/3)x^3 - x^2 - 3x + 5/3 の停留点を求める
# (a) 最急降下法（x0 = 1/2, tk = 1/(k+1)）
# (b) ニュートン法（x0 = 5, tk = 1）

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def f(x):
    """目的関数 f(x) = 1/3 x^3 - x^2 - 3x + 5/3"""
    return (1.0 / 3.0) * x**3 - x**2 - 3.0 * x + 5.0 / 3.0


def df(x):
    """1階微分 f'(x) = x^2 - 2x - 3"""
    return x**2 - 2.0 * x - 3.0


def d2f(x):
    """2階微分 f''(x) = 2x - 2"""
    return 2.0 * x - 2.0


@dataclass(frozen=True)
class IterationHistory:
    x: list[float]
    f: list[float]
    grad_abs: list[float]


# -------------------------
# (a) 最急降下法
# -------------------------


def steepest_descent(x0, eps=1e-8, max_iter=1000, *, return_history=False):
    """
    最急降下法（1次元版）
    x_{k+1} = x_k - t_k * f'(x_k)
    t_k = 1 / (k+1)
    """
    x = x0
    history_x = [float(x)]
    history_f = [float(f(x))]
    history_grad_abs = [float(abs(df(x)))]

    for k in range(max_iter):
        g = df(x)  # 勾配（1次元なので単なる導関数）
        if abs(g) <= eps:
            # 停留点に到達したとみなす
            if return_history:
                return x, k, IterationHistory(history_x, history_f, history_grad_abs)
            return x, k

        t_k = 1.0 / (k + 1)  # 指定どおりのステップ幅
        x = x - t_k * g

        history_x.append(float(x))
        history_f.append(float(f(x)))
        history_grad_abs.append(float(abs(df(x))))

    # 最大反復に到達した場合
    if return_history:
        return x, max_iter, IterationHistory(history_x, history_f, history_grad_abs)
    return x, max_iter


# -------------------------
# (b) ニュートン法
# -------------------------


def newton_method(x0, eps=1e-8, max_iter=1000, *, return_history=False):
    """
    ニュートン法
    x_{k+1} = x_k - t_k * f'(x_k) / f''(x_k)
    ここでは t_k = 1
    """
    x = x0
    history_x = [float(x)]
    history_f = [float(f(x))]
    history_grad_abs = [float(abs(df(x)))]

    for k in range(max_iter):
        g = df(x)
        h = d2f(x)

        if abs(g) <= eps:
            # 停留点に到達したとみなす
            if return_history:
                return x, k, IterationHistory(history_x, history_f, history_grad_abs)
            return x, k

        if h == 0.0:
            raise ZeroDivisionError(f"f''(x) = 0 となったため更新できません (x = {x})")

        t_k = 1.0  # 指定どおり常に1
        x = x - t_k * g / h

        history_x.append(float(x))
        history_f.append(float(f(x)))
        history_grad_abs.append(float(abs(df(x))))

    if return_history:
        return x, max_iter, IterationHistory(history_x, history_f, history_grad_abs)
    return x, max_iter


# -------------------------
# メイン
# -------------------------


def main():
    print("=== (a) 最急降下法 ===")
    x0_sd = 0.5  # 初期点 x0 = 1/2
    x_star_sd, it_sd, hist_sd = steepest_descent(x0_sd, return_history=True)
    print("初期値 x0 = {:.6f}".format(x0_sd))
    print("近似停留点 x* ? {:.10f}".format(x_star_sd))
    print("f'(x*) ? {:.3e}".format(df(x_star_sd)))
    print("反復回数 k =", it_sd)
    print()

    print("=== (b) ニュートン法 ===")
    x0_nt = 5.0  # 初期点 x0 = 5
    x_star_nt, it_nt, hist_nt = newton_method(x0_nt, return_history=True)
    print("初期値 x0 = {:.6f}".format(x0_nt))
    print("近似停留点 x* ? {:.10f}".format(x_star_nt))
    print("f'(x*) ? {:.3e}".format(df(x_star_nt)))
    print("反復回数 k =", it_nt)
    print()

    # y=f(x) 上で、反復点が収束していく様子を同一グラフにプロット
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib が見つからないため、プロットをスキップします。")
        return

    all_x = hist_sd.x + hist_nt.x
    x_min = min(all_x)
    x_max = max(all_x)
    margin = 0.15 * (x_max - x_min) if x_max > x_min else 1.0
    x_left = x_min - margin
    x_right = x_max + margin

    n = 600
    xs = [x_left + (x_right - x_left) * i / n for i in range(n + 1)]
    ys = [f(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys, color="black", linewidth=1.5, label="f(x)")

    plt.plot(hist_sd.x, hist_sd.f, marker="o", markersize=4, linewidth=1.2, label="Steepest descent")
    plt.plot(hist_nt.x, hist_nt.f, marker="s", markersize=4, linewidth=1.2, label="Newton")

    plt.scatter([hist_sd.x[0]], [hist_sd.f[0]], s=60, edgecolors="black", zorder=3)
    plt.scatter([hist_nt.x[0]], [hist_nt.f[0]], s=60, edgecolors="black", zorder=3)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Iterations on y = f(x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path = Path(__file__).with_name("task16.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"プロットを保存しました: {out_path}")
    import matplotlib

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
