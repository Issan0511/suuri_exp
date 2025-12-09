#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    f = x0**2 + np.exp(x0) + x1**4 + x1**2 - 2.0*x0*x1 + 3.0
    return f

def evalg(x):
    """
    勾配ベクトル ∇f(x) を計算する
    ∇f(x) = [ 2x0 + exp(x0) - 2x1,
               4x1^3 + 2x1 - 2x0 ]
    """
    x0, x1 = x[0], x[1]
    g0 = 2.0*x0 + np.exp(x0) - 2.0*x1
    g1 = 4.0*x1**3 + 2.0*x1 - 2.0*x0
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
    h11 = 12.0*x1**2 + 2.0
    return np.array([[h00, h01],
                     [h10, h11]])


# =========================================================
# バックトラック法（共通で使う）
# =========================================================

def backtracking(xk, dk, evalf, evalg,
                 t_init=1.0, rho=0.5, xi=1e-4):
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

def steepest_descent_bt(x0, evalf, evalg,
                        eps=1e-6, max_iter=1000,
                        xi=1e-4, rho=0.5, t_init=1.0):
    """
    バックトラック法付き最急降下法
    x_{k+1} = x_k + t_k d_k,  d_k = -∇f(x_k)
    """
    x = x0.copy()
    for k in range(max_iter):
        g = evalg(x)
        norm_g = np.linalg.norm(g)
        if norm_g <= eps:
            return x, evalf(x), k
        dk = -g
        tk = backtracking(x, dk, evalf, evalg,
                          t_init=t_init, rho=rho, xi=xi)
        x = x + tk * dk
    # 最大反復に達した場合
    return x, evalf(x), max_iter


# =========================================================
# (b) バックトラック法付きニュートン法
# =========================================================

def newton_bt(x0, evalf, evalg, evalh,
              eps=1e-6, max_iter=1000,
              xi=1e-4, rho=0.5, t_init=1.0):
    """
    バックトラック法付きニュートン法
    d_k はヘッセ行列を用いて ∇^2 f(x_k) d_k = -∇f(x_k) を解く
    """
    x = x0.copy()
    for k in range(max_iter):
        g = evalg(x)
        norm_g = np.linalg.norm(g)
        if norm_g <= eps:
            return x, evalf(x), k

        H = evalh(x)
        # ニュートン方向を解く: H d = -g
        try:
            dk = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # 非正則などで解けない場合は勾配方向にフォールバックしてもよいが、
            # ここでは単純に break する
            break

        tk = backtracking(x, dk, evalf, evalg,
                          t_init=t_init, rho=rho, xi=xi)
        x = x + tk * dk

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
    x_sd, f_sd, k_sd = steepest_descent_bt(
        x0, evalf, evalg,
        eps=1e-6, max_iter=1000,
        xi=xi, rho=rho, t_init=t_init
    )
    print("最適解近似 x* ≈", x_sd)
    print("最適値近似 f(x*) ≈", f_sd)
    print("反復回数 =", k_sd)

    # (b) ニュートン法
    print("\n=== (b) バックトラック法付きニュートン法 ===")
    x_nt, f_nt, k_nt = newton_bt(
        x0, evalf, evalg, evalh,
        eps=1e-6, max_iter=1000,
        xi=xi, rho=rho, t_init=t_init
    )
    print("最適解近似 x* ≈", x_nt)
    print("最適値近似 f(x*) ≈", f_nt)
    print("反復回数 =", k_nt)


if __name__ == "__main__":
    main()
