import numpy as np
import matplotlib.pyplot as plt

sigma = 10.0
beta = 8.0 / 3.0
r = 28.0

def x_dot (x, y, z, t=0):
    return sigma * (y - x)

def y_dot (x, y, z, t=0):
    return x * (r - z) - y

def z_dot (x, y, z, t=0):
    return x * y - beta * z

def update_runge_kutta4_lorenz (delta_t, u1, t1 ,f_x= x_dot, f_y= y_dot, f_z= z_dot):
    x1, y1, z1 = u1

    k1x = f_x(x1, y1, z1, t1)
    k1y = f_y(x1, y1, z1, t1)
    k1z = f_z(x1, y1, z1, t1)

    k2x = f_x(x1 + (delta_t / 2) * k1x, y1 + (delta_t / 2) * k1y, z1 + (delta_t / 2) * k1z, t1 + (delta_t / 2))
    k2y = f_y(x1 + (delta_t / 2) * k1x, y1 + (delta_t / 2) * k1y, z1 + (delta_t / 2) * k1z, t1 + (delta_t / 2))
    k2z = f_z(x1 + (delta_t / 2) * k1x, y1 + (delta_t / 2) * k1y, z1 + (delta_t / 2) * k1z, t1 + (delta_t / 2))

    k3x = f_x(x1 + (delta_t / 2) * k2x, y1 + (delta_t / 2) * k2y, z1 + (delta_t / 2) * k2z, t1 + (delta_t / 2))
    k3y = f_y(x1 + (delta_t / 2) * k2x, y1 + (delta_t / 2) * k2y, z1 + (delta_t / 2) * k2z, t1 + (delta_t / 2))
    k3z = f_z(x1 + (delta_t / 2) * k2x, y1 + (delta_t / 2) * k2y, z1 + (delta_t / 2) * k2z, t1 + (delta_t / 2))

    k4x = f_x(x1 + delta_t * k3x, y1 + delta_t * k3y, z1 + delta_t * k3z, t1 + delta_t)
    k4y = f_y(x1 + delta_t * k3x, y1 + delta_t * k3y, z1 + delta_t * k3z, t1 + delta_t)
    k4z = f_z(x1 + delta_t * k3x, y1 + delta_t * k3y, z1 + delta_t * k3z, t1 + delta_t)

    x_new = x1 + (delta_t / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y1 + (delta_t / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
    z_new = z1 + (delta_t / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
    return np.array([x_new, y_new, z_new])

def update_forward_euler_lorenz(delta_t, u1, t1, f_x=x_dot, f_y=y_dot, f_z=z_dot):
    x1, y1, z1 = u1
    x_new = x1 + delta_t * f_x(x1, y1, z1, t1)
    y_new = y1 + delta_t * f_y(x1, y1, z1, t1)
    z_new = z1 + delta_t * f_z(x1, y1, z1, t1)
    return np.array([x_new, y_new, z_new])

# 共通のサンプリング関数：軌道は保持せず、指定時刻だけ返す
def sample_lorenz(method, n, t_start, t_end, u_start, sample_times):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.array(u_start, dtype=np.float64)
    t = np.float64(t_start)

    sample_times = np.array(sample_times, dtype=np.float64)
    # サンプルしたい時刻に対応するステップ番号
    idx_map = {
        int(round((ts - t_start) / delta_t)): i
        for i, ts in enumerate(sample_times)
    }

    samples = np.empty((len(sample_times), 3), dtype=np.float64)

    for k in range(n + 1):
        # k番目ステップの状態を必要に応じて保存
        if k in idx_map:
            samples[idx_map[k]] = u

        if k == n:
            break

        if method == "rk4":
            u = update_runge_kutta4_lorenz(delta_t, u, t, x_dot, y_dot, z_dot)
        elif method == "fe":
            u = update_forward_euler_lorenz(delta_t, u, t, x_dot, y_dot, z_dot)
        else:
            raise ValueError("method must be 'rk4' or 'fe'")

        t += delta_t

    return samples

# ここからメイン部分
t_start = 0
t_end = 100
u_start = (1, 0, 0)
sample_times = [15.0, 30.0, 60.0]

rk_vals_15 = []
fe_vals_15 = []
rk_vals_30 = []
fe_vals_30 = []
rk_vals_60 = []
fe_vals_60 = []
n_list = []

# 複数のステップ数で比較
for pow in range(14):
    n = 10000 * (2 ** pow)
    n_list.append(n)

    rk_samples = sample_lorenz("rk4", n, t_start, t_end, u_start, sample_times)
    fe_samples = sample_lorenz("fe",  n, t_start, t_end, u_start, sample_times)

    rk_vals_15.append(rk_samples[0])
    rk_vals_30.append(rk_samples[1])
    rk_vals_60.append(rk_samples[2])

    fe_vals_15.append(fe_samples[0])
    fe_vals_30.append(fe_samples[1])
    fe_vals_60.append(fe_samples[2])

rk_vals_15 = np.array(rk_vals_15)
fe_vals_15 = np.array(fe_vals_15)
rk_vals_30 = np.array(rk_vals_30)
fe_vals_30 = np.array(fe_vals_30)
rk_vals_60 = np.array(rk_vals_60)
fe_vals_60 = np.array(fe_vals_60)

# x軸をΔt（ステップ幅）に設定
delta_t_values = [(100 - 0) / n for n in n_list]

# プロット（x成分のみ、3つの時刻）
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(delta_t_values, rk_vals_15[:, 0], 'o-', label='RK4 (t=15)', linewidth=2, markersize=4)
ax.plot(delta_t_values, fe_vals_15[:, 0], 's--', label='Forward Euler (t=15)', linewidth=2, markersize=4)
ax.plot(delta_t_values, rk_vals_30[:, 0], '^-', label='RK4 (t=30)', linewidth=2, markersize=4)
ax.plot(delta_t_values, fe_vals_30[:, 0], 'v--', label='Forward Euler (t=30)', linewidth=2, markersize=4)
ax.plot(delta_t_values, rk_vals_60[:, 0], 'd-', label='RK4 (t=60)', linewidth=2, markersize=4)
ax.plot(delta_t_values, fe_vals_60[:, 0], 'x--', label='Forward Euler (t=60)', linewidth=2, markersize=4)
ax.set_xlabel('Step Width (Δt)', fontsize=12)
ax.set_ylabel('x Value', fontsize=12)
ax.set_title('Lorenz System x-component vs Step Width', fontsize=14)
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lorenz_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
