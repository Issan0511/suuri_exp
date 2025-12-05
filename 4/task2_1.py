import numpy as np

def pi_monte_carlo_integration(M):
    s = np.random.rand(10, M)     # (10, M)
    s *= s                        # s = s**2
    s += 1.0                      # 1 + s**2
    np.reciprocal(s, out=s)       # 1 / (1 + s**2)
    s *= 4.0                      # 4 / (1 + s**2)
    mean_fs = s.mean(axis=1)      # 各行の平均
    # S の不偏推定分散を計算
    s -= mean_fs[:, np.newaxis]  # 各行から平均を引く
    s *= s                       # 平方
    variances = s.sum(axis=1) / (M - 1)  # 不偏分散
    accuracy = np.sqrt(variances / M)  # 標準誤差
    return mean_fs, accuracy

# M = 100_000_000
Ms = [10,1000,100000, 10000000]
for M in Ms:
    mean_fs, accuracy = pi_monte_carlo_integration(M)

    print(f"M = {M}")
    for f in mean_fs:
        print("pi")
        print(f"{f:.12f}")
        print("accuracy")
        print(f"{accuracy[0]:.12f}")
    print(f"Mean: {mean_fs.mean():.12f}")
