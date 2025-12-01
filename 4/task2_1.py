import numpy as np

def pi_monte_carlo_integration(M):
    s = np.random.rand(10, M)     # (10, M)
    s *= s                        # s = s**2
    s += 1.0                      # 1 + s**2
    np.reciprocal(s, out=s)       # 1 / (1 + s**2)
    s *= 4.0                      # 4 / (1 + s**2)
    mean_fs = s.mean(axis=1)      # 各行の平均
    return mean_fs

M = 100_000_000
mean_fs = pi_monte_carlo_integration(M)
for i in range(10):
    print(f'Sample {i+1}: {mean_fs[i]:.6f}')
print("mean:", mean_fs.mean())
