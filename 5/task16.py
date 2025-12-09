#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 課題16
# f(x) = (1/3)x^3 - x^2 - 3x + 5/3 の停留点を求める
# (a) 最急降下法（x0 = 1/2, tk = 1/(k+1)）
# (b) ニュートン法（x0 = 5, tk = 1）

def f(x):
    """目的関数 f(x) = 1/3 x^3 - x^2 - 3x + 5/3"""
    return (1.0/3.0)*x**3 - x**2 - 3.0*x + 5.0/3.0

def df(x):
    """1階微分 f'(x) = x^2 - 2x - 3"""
    return x**2 - 2.0*x - 3.0

def d2f(x):
    """2階微分 f''(x) = 2x - 2"""
    return 2.0*x - 2.0


# -------------------------
# (a) 最急降下法
# -------------------------

def steepest_descent(x0, eps=1e-8, max_iter=1000):
    """
    最急降下法（1次元版）
    x_{k+1} = x_k - t_k * f'(x_k)
    t_k = 1 / (k+1)
    """
    x = x0
    for k in range(max_iter):
        g = df(x)  # 勾配（1次元なので単なる導関数）
        if abs(g) <= eps:
            # 停留点に到達したとみなす
            return x, k

        t_k = 1.0 / (k + 1)   # 指定どおりのステップ幅
        x = x - t_k * g

    # 最大反復に到達した場合
    return x, max_iter


# -------------------------
# (b) ニュートン法
# -------------------------

def newton_method(x0, eps=1e-8, max_iter=1000):
    """
    ニュートン法
    x_{k+1} = x_k - t_k * f'(x_k) / f''(x_k)
    ここでは t_k = 1
    """
    x = x0
    for k in range(max_iter):
        g = df(x)
        h = d2f(x)

        if abs(g) <= eps:
            # 停留点に到達したとみなす
            return x, k

        if h == 0.0:
            raise ZeroDivisionError("f''(x) = 0 となったため更新できません (x = {})".format(x))

        t_k = 1.0  # 指定どおり常に1
        x = x - t_k * g / h

    return x, max_iter


# -------------------------
# メイン
# -------------------------

def main():
    print("=== (a) 最急降下法 ===")
    x0_sd = 0.5  # 初期点 x0 = 1/2
    x_star_sd, it_sd = steepest_descent(x0_sd)
    print("初期値 x0 = {:.6f}".format(x0_sd))
    print("近似停留点 x* ≈ {:.10f}".format(x_star_sd))
    print("f'(x*) ≈ {:.3e}".format(df(x_star_sd)))
    print("反復回数 k =", it_sd)
    print()

    print("=== (b) ニュートン法 ===")
    x0_nt = 5.0  # 初期点 x0 = 5
    x_star_nt, it_nt = newton_method(x0_nt)
    print("初期値 x0 = {:.6f}".format(x0_nt))
    print("近似停留点 x* ≈ {:.10f}".format(x_star_nt))
    print("f'(x*) ≈ {:.3e}".format(df(x_star_nt)))
    print("反復回数 k =", it_nt)


if __name__ == "__main__":
    main()
