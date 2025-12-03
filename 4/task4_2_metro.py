import numpy as np
import matplotlib.pyplot as plt
L = 64
# 温度T はz =exp(2/T)−1 = √2の 解

# beta = 1.0 / T
beta = 0.5 * np.log(1.0 + np.sqrt(2.0))
rng = np.random.default_rng()

# 状態配列（唯一の状態配列）
spins = np.ones((L, L), dtype=np.int8)

# チェッカーボードマスク（初回だけ作る）
x = np.arange(L)[:, None]
y = np.arange(L)[None, :]
mask_black = ((x + y) % 2 == 0)
mask_white = ~mask_black


def sweep(spins, beta, rng, itrations=100):
    """
    J=1, H=0 のイジング模型をメトロポリス法 + チェッカーボード更新で itrations ステップ回す。
    spins : (L, L) の int 型配列（要素は ±1）
    beta  : 1 / T
    rng   : np.random.Generator
    """
    ms = []
    for _ in range(itrations):
        for mask in (mask_black, mask_white):

            # ----- 近傍和 n(i,j) = 上下左右の和 -----
            n  = np.roll(spins,  1, axis=0)
            n += np.roll(spins, -1, axis=0)
            n += np.roll(spins,  1, axis=1)
            n += np.roll(spins, -1, axis=1)

            # ----- ΔE = 2 s_i Σ_j s_j (J=1, H=0) -----
            # mask で対象サイトだけ取り出す
            s_sub = spins[mask]      # 現在の s_i
            n_sub = n[mask]          # 近傍和 Σ_j s_j

            deltaE = 2.0 * s_sub * n_sub  # ここは配列（float でも int でも可）

            # ----- メトロポリスの採択確率 -----
            # ΔE <= 0 なら必ず採択、ΔE > 0 なら exp(-beta * ΔE) で採択
            r = rng.random(s_sub.shape)   # [0,1) 一様乱数

            accept = (deltaE <= 0.0) | (r < np.exp(-beta * deltaE))

            # ----- 採択されたサイトだけスピンを反転 -----
            s_sub[accept] *= -1
            spins[mask] = s_sub

        ms.append(spins.mean())

    return np.array(ms)

ms = sweep(spins, beta, rng, itrations=10000)

plt.plot(ms)
plt.title('Magnetization vs Iteration(Metropolis Method)')
plt.xlabel('Iteration')
plt.ylabel('Magnetization')
plt.savefig('task4_2_metro.png')
plt.show()



