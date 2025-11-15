import os
import numpy as np
import matplotlib.pyplot as plt

# パラメータ
alpha = 10.0   # ここは元に戻す
beta  = 0.0
u0    = 1.0
t0    = 0.0
t_end = 10.0

# 右辺と真値
def f(u, t, alpha=alpha, beta=beta):
    return -alpha * u + beta

def u_exact(t, alpha=alpha, beta=beta, u0=u0):
    if alpha == 0.0:
        return u0 + beta * t
    return (u0 - beta/alpha) * np.exp(-alpha * t) + beta/alpha

# --- 1ステップ法 -------------------------------------------------

def step_crank_nicolson(dt, u, t, alpha=alpha, beta=beta):
    num = u + 0.5 * dt * (-alpha * u + beta)
    den = 1.0 + 0.5 * dt * alpha
    return num / den

def step_predictor_corrector(dt, u, t, alpha=alpha, beta=beta):
    # 予測: 前進オイラー
    u_pred = u + dt * f(u, t, alpha, beta)
    # 修正: 後退オイラーを予測値で陽化
    return u + dt * f(u_pred, t + dt, alpha, beta)

def step_heun(dt, u, t, alpha=alpha, beta=beta):
    k1 = f(u, t, alpha, beta)
    u_pred = u + dt * k1
    k2 = f(u_pred, t + dt, alpha, beta)
    return u + 0.5 * dt * (k1 + k2)

# --- 汎用ソルバ ---------------------------------------------------

def solve(step_func, dt_target, alpha=alpha, beta=beta):
    N = int(round((t_end - t0) / dt_target))
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    u = np.empty_like(t)
    u[0] = u0
    for n in range(N):
        u[n+1] = step_func(dt, u[n], t[n], alpha, beta)
    return t, u, dt

# --- グラフ描画 & 保存 ---------------------------------------------

methods = [
    ("crank_nicolson",      "Crank–Nicolson",      step_crank_nicolson),
    ("predictor_corrector", "Predictor–Corrector", step_predictor_corrector),
    ("heun",                "Heun method",         step_heun),
]

# 手法ごとの Δt
dt_lists = {
    "crank_nicolson":      [0.01, 0.19, 0.205],
    "predictor_corrector": [0.01, 0.095, 0.105],
    "heun":                [0.01, 0.19, 0.205],
}

save_dir = "."
os.makedirs(save_dir, exist_ok=True)

for fname_prefix, title, step_func in methods:
    plt.figure(figsize=(6, 4))

    # 真値
    t_exact = np.linspace(t0, t_end, 2000)
    plt.plot(t_exact, u_exact(t_exact), "k--", label="exact")

    # 各 Δt での数値解
    for dt_target in dt_lists[fname_prefix]:
        t, u, dt_actual = solve(step_func, dt_target)
        plt.plot(t, u, label=f"$\\Delta t \\approx {dt_actual:.3f}$")

    plt.xlabel("$t$")
    plt.ylabel("$u$")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 縦軸の範囲
    if fname_prefix == "crank_nicolson":
        plt.ylim(-1.0, 1.0)
    else:
        plt.ylim(-1.0, 10.0)

    plt.tight_layout()
    out_path = f"{save_dir}/task4_{fname_prefix}.png"
    plt.savefig(out_path)
    plt.close()

print("saved:", [f"task4_{m[0]}.png" for m in methods])
