import numpy as np
import matplotlib.pyplot as plt

def f (u,t =0):
    return (u-1)*u

def update_runge_kutta4 (delta_t, u1, t1 ,f= f):
    k1 = f(u1, t1)
    k2 = f(u1 + (delta_t / 2) * k1, t1 + (delta_t / 2))
    k3 = f(u1 + (delta_t / 2) * k2, t1 + (delta_t / 2))
    k4 = f(u1 + delta_t * k3, t1 + delta_t)
    return u1 + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def runge_kutta4_arc_length (n,t_start = 0, t_end = 1, u_start = 1, return_trajectory=False):
    delta_t_default = np.float64((t_end - t_start) / n)
    delta_t = delta_t_default
    u = np.float64(u_start)
    t = np.float64(t_start)
    
    if return_trajectory:
        t_values = [t]
        u_values = [u]
    
    for i in range (1000000):
        u = update_runge_kutta4 (delta_t, u, t, f)
        if abs(u) > 1e5:
            u = np.float64('inf')
            if return_trajectory:
                t_values.append(t)
                u_values.append(u)
            break
        delta_t = delta_t_default / np.sqrt (1 + (f(u, t))**2)
        t += delta_t
        
        if return_trajectory:
            t_values.append(t)
            u_values.append(u)
            
        if t > t_end:
            break
    if return_trajectory:
        return u,t, np.array(t_values), np.array(u_values)
    return u


# プロット用のデータを収集
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# ノーマルバージョン（上）
for u_start in np.arange(0, 2.1, 0.2):
    _, t,t_vals, u_vals = runge_kutta4_arc_length(10000, 0, 5, u_start, return_trajectory=True)
    ax1.plot(t_vals, u_vals, label=f'u(0) = {u_start:.1f}', linewidth=2)
    approx = u_vals[-1] if len(u_vals) > 0 else np.nan
    print(f"u_start: {u_start:.1f},final t:{t}, approx: {approx}")

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
    _,t, t_vals, u_vals = runge_kutta4_arc_length(10000, 0, 5, u_start, return_trajectory=True)
    # アークタンジェント変換したデータをプロット
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
    