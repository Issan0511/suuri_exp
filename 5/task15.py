#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 課題15: 関数の定義
# f(x) = x^3 + 2x^2 - 5x - 6
# -------------------------

def f(x):
    return x**3 + 2*x**2 - 5*x - 6

def df(x):
    """f(x) の導関数: f'(x) = 3x^2 + 4x - 5"""
    return 3*x**2 + 4*x - 5


# -------------------------
# (a) グラフ描画
# -------------------------

def plot_function():
    x = np.linspace(-10, 10, 2000)
    y = f(x)

    plt.figure()
    plt.plot(x, y)
    plt.axhline(0, color="black", linewidth=0.8)  # x軸
    plt.axvline(0, color="black", linewidth=0.8)  # y軸
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Graph of f(x) = x^3 + 2x^2 - 5x - 6")
    plt.grid(True)
    plt.show()
    plt.savefig("task15.png")


# -------------------------
# (b) 二分法
# -------------------------

def bisection(f, a, b, eps=1e-10, max_iter=1000):
    """[a, b] で二分法により f(x) = 0 の解を求める.
       f(a) と f(b) の符号は異なることが前提。
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) と f(b) の符号が同じです: a={}, b={}".format(a, b))

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)

        if abs(fc) <= eps or 0.5 * (b - a) < eps:
            return c

        # 符号でどちらの区間を残すか決める
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    # 最大反復に達した場合
    return 0.5 * (a + b)


def solve_with_bisection():
    # グラフから零点が -3, -1, 2 付近にあることが分かるので
    # それを挟む区間を手で指定する
    intervals = [
        (-4.0, -2.0),  # -3 付近
        (-2.0,  0.0),  # -1 付近
        ( 1.0,  3.0),  #  2 付近
    ]

    roots = []
    for (a, b) in intervals:
        r = bisection(f, a, b)
        roots.append(r)
    return roots


# -------------------------
# (c) ニュートン法
# -------------------------

def newton(f, df, x0, eps=1e-10, max_iter=1000):
    """ニュートン法: f(x) = 0 の解を初期値 x0 から探索."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(fx) <= eps:
            return x

        if dfx == 0:
            # 導関数が 0 になると更新できない
            raise ZeroDivisionError("f'(x) = 0 となったため打ち切り (x={})".format(x))

        x = x - fx / dfx

    return x  # 収束しなかった場合は最後の値を返す


def solve_with_newton():
    # グラフから零点が -3, -1, 2 付近にあることを利用
    initial_points = [-2.5, -0.5, 1.5]
    roots = []
    for x0 in initial_points:
        r = newton(f, df, x0)
        roots.append(r)
    return roots


# -------------------------
# メイン
# -------------------------

def main():
    # (a) グラフ描画
    plot_function()

    # (b) 二分法
    bisection_roots = solve_with_bisection()
    print("Bisection method roots:")
    for r in bisection_roots:
        print("  x ≈ {:.10f}, f(x) ≈ {:.3e}".format(r, f(r)))

    # (c) ニュートン法
    newton_roots = solve_with_newton()
    print("\nNewton method roots:")
    for r in newton_roots:
        print("  x0 -> root ≈ {:.10f}, f(x) ≈ {:.3e}".format(r, f(r)))


if __name__ == "__main__":
    main()
