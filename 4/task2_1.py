import numpy as np

def pi_monte_carlo_integration(M):
    s = np.random.rand(10, M)     # (10, M)
    s *= s                        # s = s**2
    s += 1.0                      # 1 + s**2
    np.reciprocal(s, out=s)       # 1 / (1 + s**2)
    s *= 4.0                      # 4 / (1 + s**2)
    mean_fs = s.mean(axis=1)      # 各行の平均
    return mean_fs

# M = 100_000_000
Ms = [10,1000,100000, 10000000]
for M in Ms:
    mean_fs = pi_monte_carlo_integration(M)

    print(f"M = {M}")
    for f in mean_fs:
        print(f"{f:.6f}")
    print(f"Mean: {mean_fs.mean():.12f}")
    print(f"Variance: {mean_fs.var():.12f}")
    print()
