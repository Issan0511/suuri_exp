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

def forward_euler_lorenz(n, t_start=0, t_end=50, u_start=(1, 0, 0), return_trajectory=False):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.array(u_start, dtype=np.float64)
    t = np.float64(t_start)
    
    if return_trajectory:
        t_values = [t]
        trajectory = [u.copy()]
    
    for i in range(n):
        u = update_forward_euler_lorenz(delta_t, u, t, x_dot, y_dot, z_dot)
        t += delta_t
        
        if return_trajectory:
            t_values.append(t)
            trajectory.append(u.copy())
            
    if return_trajectory:
        return u, np.array(t_values), np.array(trajectory)
    return u

def runge_kutta4_lorenz (n,t_start = 0, t_end = 50, u_start = (1,0,0), return_trajectory=False):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.array(u_start, dtype=np.float64)
    t = np.float64(t_start)
    
    if return_trajectory:
        t_values = [t]
        trajectory = [u.copy()]
    
    for i in range (n):
        u = update_runge_kutta4_lorenz (delta_t, u, t, x_dot, y_dot, z_dot)
        t += delta_t
        
        if return_trajectory:
            t_values.append(t)
            trajectory.append(u.copy())
            
    if return_trajectory:
        return u, np.array(t_values), np.array(trajectory)
    return u


# パラメータ設定
n = 10000
t_start = 0
t_end = 100
u_start = (1, 0, 0)

print(f"初期条件: x={u_start[0]}, y={u_start[1]}, z={u_start[2]}")
print(f"時間範囲: t={t_start} から t={t_end}")
print(f"ステップ数: {n}")
print()

# ルンゲ・クッタ法で軌道を計算
print("ルンゲ・クッタ法で計算中...")
u_final_rk, t_values_rk, trajectory_rk = runge_kutta4_lorenz(
    n, t_start, t_end, u_start, return_trajectory=True
)
print(f"ルンゲ・クッタ法 最終状態: x={u_final_rk[0]:.4f}, y={u_final_rk[1]:.4f}, z={u_final_rk[2]:.4f}")

# 前進オイラー法で軌道を計算
print("前進オイラー法で計算中...")
u_final_euler, t_values_euler, trajectory_euler = forward_euler_lorenz(
    n, t_start, t_end, u_start, return_trajectory=True
)
print(f"前進オイラー法 最終状態: x={u_final_euler[0]:.4f}, y={u_final_euler[1]:.4f}, z={u_final_euler[2]:.4f}")
print()

# ルンゲ・クッタ法の3次元プロット
fig1 = plt.figure(figsize=(12, 9))
ax1 = fig1.add_subplot(111, projection='3d')

ax1.plot(trajectory_rk[:, 0], trajectory_rk[:, 1], trajectory_rk[:, 2], linewidth=0.5, color='blue', alpha=0.8)
ax1.scatter(trajectory_rk[0, 0], trajectory_rk[0, 1], trajectory_rk[0, 2], color='green', s=100, label='Start', marker='o')
ax1.scatter(trajectory_rk[-1, 0], trajectory_rk[-1, 1], trajectory_rk[-1, 2], color='red', s=100, label='End', marker='x')

ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Z', fontsize=12)
ax1.set_title('Lorenz Attractor - Runge-Kutta Method', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# ルンゲ・クッタ法のプロットを保存
fig1.savefig('lorenz_runge_kutta.png', dpi=300, bbox_inches='tight')
print("ルンゲ・クッタ法のプロットを 'lorenz_runge_kutta.png' に保存しました")

# 前進オイラー法の3次元プロット
fig2 = plt.figure(figsize=(12, 9))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.plot(trajectory_euler[:, 0], trajectory_euler[:, 1], trajectory_euler[:, 2], linewidth=0.5, color='red', alpha=0.8)
ax2.scatter(trajectory_euler[0, 0], trajectory_euler[0, 1], trajectory_euler[0, 2], color='green', s=100, label='Start', marker='o')
ax2.scatter(trajectory_euler[-1, 0], trajectory_euler[-1, 1], trajectory_euler[-1, 2], color='purple', s=100, label='End', marker='x')

ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_zlabel('Z', fontsize=12)
ax2.set_title('Lorenz Attractor - Forward Euler Method', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 前進オイラー法のプロットを保存
fig2.savefig('lorenz_forward_euler.png', dpi=300, bbox_inches='tight')
print("前進オイラー法のプロットを 'lorenz_forward_euler.png' に保存しました")

plt.tight_layout()
plt.show()

