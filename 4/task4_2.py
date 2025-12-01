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


def sweep(spins, beta, rng,itrations=100):
    ms = []
    for _ in range(itrations):
        for mask in (mask_black, mask_white):

            # ----- 近傍和（最速） -----
            n  = np.roll(spins,  1, axis=0)
            n += np.roll(spins, -1, axis=0)
            n += np.roll(spins,  1, axis=1)
            n += np.roll(spins, -1, axis=1)

            # ----- 熱浴確率 -----
            p = 1.0 / (1.0 + np.exp(-2 * beta * n))

            # ----- ランダムに更新 -----
            r = rng.random((L, L))
            sub = mask
            spins[sub] = np.where(r[sub] < p[sub], 1, -1)
        ms.append(spins.mean())
    return np.array(ms)
        
ms = sweep(spins, beta, rng, itrations=10000)

plt.plot(ms)
plt.xlabel('Iteration')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Iteration')
plt.show()




