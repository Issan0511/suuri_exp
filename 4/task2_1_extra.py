import numpy as np

def pi_pi_monte_carlo_integration(M):
    s = np.random.rand(10, M)     # (10, M)
    s *= s                        # s = s**2
    s += 1.0                      # 1 + s**2
    np.reciprocal(s, out=s)       # 1 / (1 + s**2)
    s *= 4.0                      # 4 / (1 + s**2)
    s *= s
    mean_fs = s.mean(axis=1)      # 各行の平均
    return mean_fs

M = 10_000_000

result = pi_pi_monte_carlo_integration(M)

print("2乗の平均",result.mean())
print("Π^2との誤差",abs(result.mean() - np.pi**2))