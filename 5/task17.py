#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# f(x) = x0^2 + exp(x0) + x1^4 + x1^2 - 2 x0 x1 + 3

def evalf(x):
    """
    目的関数の値 f(x) を計算する
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
    g = np.array([g0, g1])
    return g

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
    H = np.array([[h00, h01],
                  [h10, h11]])
    return H

def main():
    # 動作確認用
    x = np.array([0.3, 5.0])
    f = evalf(x)
    g = evalg(x)
    H = evalh(x)

    print("x =", x)
    print("f(x) =", f)
    print("g(x) =", g)
    print("H(x) =\n", H)

if __name__ == "__main__":
    main()
