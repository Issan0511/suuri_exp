#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def primal_dual_ipm_qp(Q, c, A, b, *, max_iter=80, tol=1e-12, sigma=0.1, tau=0.99):
    """
    Primal-Dual Interior-Point (Newton) for convex QP:
        minimize   1/2 x^T Q x + c^T x
        subject to A x <= b   (b is scalar or length-m)

    Slack: y >= 0 such that  A x + y - b = 0
    Dual:  λ >= 0
    Perturbed complementarity: y_i λ_i = sigma * mu,
        mu = (y^T λ)/m
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n = Q.shape[0]
    m = A.shape[0]

    # 半正定値チェック（数値誤差許容）
    Qsym = 0.5 * (Q + Q.T)
    if np.linalg.eigvalsh(Qsym).min() < -1e-10:
        raise ValueError("Q must be positive semidefinite.")

    # 初期点（内点）
    x = np.zeros(n)
    y = np.ones(m)
    lam = np.ones(m)
    e = np.ones(m)

    for k in range(max_iter):
        # 残差
        rd = Qsym @ x + c + A.T @ lam          # dual residual
        rp = A @ x + y - b                      # primal residual
        mu = (y @ lam) / m                      # complementarity
        rc = y * lam - sigma * mu * e           # perturbed complementarity

        # 収束判定
        if max(np.linalg.norm(rd), np.linalg.norm(rp), mu) < tol:
            return x, y, lam, {"iter": k, "rd": np.linalg.norm(rd),
                               "rp": np.linalg.norm(rp), "mu": mu}

        # Newton 方程式
        # [ Q  0   A^T ] [dx] = -[rd]
        # [ A  I   0   ] [dy]   -[rp]
        # [ 0  Λ   Y   ] [dλ]   -[rc]
        K = np.block([
            [Qsym,                 np.zeros((n, m)),      A.T],
            [A,                    np.eye(m),             np.zeros((m, m))],
            [np.zeros((m, n)),     np.diag(lam),          np.diag(y)]
        ])
        rhs = -np.concatenate([rd, rp, rc])

        delta = np.linalg.solve(K, rhs)
        dx = delta[:n]
        dy = delta[n:n+m]
        dlam = delta[n+m:]

        # 正性を保つステップ長
        alpha = 1.0
        if np.any(dy < 0):
            alpha = min(alpha, tau * np.min(-y[dy < 0] / dy[dy < 0]))
        if np.any(dlam < 0):
            alpha = min(alpha, tau * np.min(-lam[dlam < 0] / dlam[dlam < 0]))

        # 更新
        x += alpha * dx
        y += alpha * dy
        lam += alpha * dlam

        # 数値安全
        y = np.maximum(y, 1e-18)
        lam = np.maximum(lam, 1e-18)

    return x, y, lam, {"iter": max_iter, "rd": np.linalg.norm(rd),
                       "rp": np.linalg.norm(rp), "mu": mu}


if __name__ == "__main__":
    # 課題4：パラメータ
    Q = np.array([[2.0, 0.0],
                  [0.0, 1.0]])      # 半正定値行列
    c = -np.array([1.0, 1.0])

    # b はスカラー 0 と解釈
    A = np.array([[1.0, 1.0]])      # 1 本の制約
    b = 0.0

    x_star, y_star, lam_star, info = primal_dual_ipm_qp(Q, c, A, b)

    print("x* =", x_star)
    print("slack y =", y_star)
    print("dual λ =", lam_star)
    print("info =", info)
