import numpy as np
import matplotlib.pyplot as plt

def f(u, t=0):
    return (u - 1) * u

def analytic_solution(t, u0):
    if u0 == 0:
        return 0.0
    if u0 == 1:
        return 1.0
    return u0 / (u0 + (1 - u0) * np.exp(t))

def update_runge_kutta4(delta_t, u1, t1, f=f):
    k1 = f(u1, t1)
    k2 = f(u1 + (delta_t / 2) * k1, t1 + (delta_t / 2))
    k3 = f(u1 + (delta_t / 2) * k2, t1 + (delta_t / 2))
    k4 = f(u1 + delta_t * k3, t1 + delta_t)
    return u1 + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def runge_kutta4_arc_length(n, t_start=0, t_end=1, u_start=1, return_trajectory=False):
    delta_t_default = np.float64((t_end - t_start) / n)
    delta_t = delta_t_default
    u = np.float64(u_start)
    t = np.float64(t_start)

    if return_trajectory:
        t_values = [t]
        u_values = [u]

    for i in range(1000000):
        u = update_runge_kutta4(delta_t, u, t, f)
        if abs(u) > 1e5:
            u = np.float64('inf')
            if return_trajectory:
                t_values.append(t)
                u_values.append(u)
            break

        delta_t = delta_t_default / np.sqrt(1 + (f(u, t))**2)
        t += delta_t

        if return_trajectory:
            t_values.append(t)
            u_values.append(u)

        if t > t_end:
            break

    if return_trajectory:
        return u, t, np.array(t_values), np.array(u_values)
    return u

def estimate_blowup_time_from_atan(t_vals, u_vals):
    """
    arctan(u) 空間で、最後の有限な2点から
    blow-up time の推定値 t_hat を返す。
    """
    finite_mask = np.isfinite(u_vals)
    t_finite = t_vals[finite_mask]
    u_finite = u_vals[finite_mask]

    if t_finite.size < 2:
        return None

    # 最後の2点
    t1, t2 = t_finite[-2], t_finite[-1]
    y1, y2 = np.arctan(u_finite[-2]), np.arctan(u_finite[-1])

    if t2 == t1:
        return None

    # 2点を通る直線 y = m t + b
    m = (y2 - y1) / (t2 - t1)
    if m == 0.0:
        return None

    b = y2 - m * t2

    # y = π/2 と交わる t を発散時刻の推定値とする
    t_hat = (0.5 * np.pi - b) / m

    return t_hat


# プロット用のデータを収集
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# ノーマルバージョン（上）
for u_start in np.arange(0, 2.1, 0.2):
    u_final, t_final, t_vals, u_vals = runge_kutta4_arc_length(
        10000, 0, 5, u_start, return_trajectory=True
    )
    ax1.plot(t_vals, u_vals, label=f'u(0) = {u_start:.1f}', linewidth=2)
    approx = u_vals[-1] if len(u_vals) > 0 else np.nan

    # 解析解
    analytic_vals = analytic_solution(t_vals, u_start)

    # arctan(u) 空間での誤差
    u_vals_atan        = np.arctan(u_vals)
    analytic_vals_atan = np.arctan(analytic_vals)

    abs_errors_atan     = np.abs(u_vals_atan - analytic_vals_atan)
    mean_abs_error_atan = np.mean(abs_errors_atan)
    max_abs_error_atan  = np.max(abs_errors_atan)

    msg = (
        f"u_start: {u_start:.1f}, final t: {t_final:.4f}, approx: {approx:.6f}, "
        f"mean |atan err|: {mean_abs_error_atan:.6e}, "
        f"max |atan err|: {max_abs_error_atan:.6e}"
    )

    # u_start > 1 の場合は blow-up 時刻の推定値と解析解を比較
    if u_start > 1.0:
        t_hat = estimate_blowup_time_from_atan(t_vals, u_vals)
        if t_hat is not None:
            # 解析的 blow-up 時刻
            t_blow = np.log(u_start / (u_start - 1.0))

            diff      = t_hat - t_blow          # 推定値 - 真値
            rel_error = diff / t_blow           # 相対誤差

            msg += (
                f", t_blow_exact: {t_blow:.6f}, "
                f"t_hat: {t_hat:.6f}, "
                f"diff: {diff:.3e}, "
                f"rel_error: {rel_error:.3e}"
            )
        else:
            msg += ", t_hat: N/A"

    print(msg)

ax1.set_xlabel('t', fontsize=14)
ax1.set_ylabel('u(t)', fontsize=14)
ax1.set_title('Normal version', fontsize=16)
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 7)
ax1.tick_params(axis='both', labelsize=12)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3)

# アークタンジェントバージョン（下）
for u_start in np.arange(0, 2.1, 0.2):
    _, t_final, t_vals, u_vals = runge_kutta4_arc_length(
        10000, 0, 5, u_start, return_trajectory=True
    )
    u_vals_atan = np.arctan(u_vals)
    ax2.plot(t_vals, u_vals_atan, label=f'u(0) = {u_start:.1f}', linewidth=2)

ax2.set_xlabel('t', fontsize=14)
ax2.set_ylabel('arctan(u(t))', fontsize=14)
ax2.set_title('Arctangent version', fontsize=16)
ax2.set_xlim(0, 5)
ax2.set_ylim(0, np.pi/2 * 1.05)
ax2.set_yticks([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
ax2.set_yticklabels(['0', 'π/6', 'π/4', 'π/3', 'π/2'], fontsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task6_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nグラフを 'task6_trajectory.png' として保存しました。")
