import numpy as np
import matplotlib.pyplot as plt

# ODE: u' = -2u + 1,  u(0) = 1
def f(u, t):
    return -2.0 * u + 1.0

def exact_solution(t, u0=1.0):
    # u(t) = 1/2 + (u0 - 1/2) e^{-2t}
    return 0.5 + (u0 - 0.5) * np.exp(-2.0 * t)

def update_two_step(delta_t, u_nm2, u_nm1):
    """
    2 段法:
        u_n = u_{n-2} + 2 Δt f(t_{n-1}, u_{n-1})
    ここでは f は t に依存しないので t は使わない。
    """
    return u_nm2 + 2.0 * delta_t * f(u_nm1, 0.0)

def solve_two_step(delta_t, t_start=0.0, t_end=10.0, u_start=1.0):
    """
    2 段法で t_start から t_end まで解く。
    u_0 = u_start,
    u_1 は真値 u_exact(Δt) を用いて与える。
    """
    # ステップ数を整数に丸めて dt を取り直す
    n_steps = int(round((t_end - t_start) / delta_t))
    dt = (t_end - t_start) / n_steps

    t = np.linspace(t_start, t_end, n_steps + 1)
    u = np.zeros_like(t)

    u[0] = float(u_start)
    u[1] = float(exact_solution(dt, u_start))

    for n in range(2, n_steps + 1):
        u[n] = update_two_step(dt, u[n-2], u[n-1])

    return t, u, dt

if __name__ == "__main__":
    T = 10.0
    dt_list = [ 0.10 , 0.05,0.01]

    # 図の作成
    plt.figure(figsize=(6, 4))

    # 真値
    t_exact = np.linspace(0.0, T, 2000)
    plt.plot(t_exact, exact_solution(t_exact), "k--", label="exact")

    # 各 Δt での数値解
    for dt_target in dt_list:
        t_num, u_num, dt_actual = solve_two_step(dt_target, 0.0, T, 1.0)
        plt.plot(t_num, u_num, label=f"$\\Delta t \\approx {dt_actual:.3f}$")

    plt.xlabel("$t$")
    plt.ylabel("$u$")
    plt.title("Two-step method in Task 5")
    plt.grid(True)
    plt.legend()

    # 発散の様子が見えるよう縦軸を固定
    plt.ylim(-200.0, 200.0)

    plt.tight_layout()
    plt.savefig("task5_two_step.png")
    plt.show()

    # t = 10 における値と誤差も出力（表用）
    uT_exact = exact_solution(T)
    print("t = 10 の真値:", uT_exact)
    for dt_target in dt_list:
        t_num, u_num, dt_actual = solve_two_step(dt_target, 0.0, T, 1.0)
        err = abs(u_num[-1] - uT_exact)
        print(f"Δt ≈ {dt_actual:.3f}  ->  u_N = {u_num[-1]:.6e},  error = {err:.6e}")
