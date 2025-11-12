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
n=100000
t_start = 0
t_end = 100

epsilons = [0.0, 0.1, 0.01, 0.001]
colors = ['blue', 'red', 'green', 'purple']

# 各εに対する軌道を計算
trajectories_rk = []
trajectories_fe = []
t_values_list = []

for epsilon in epsilons:
    x_start = 1.0 + epsilon
    u_start = (x_start, 0, 0)
    u_final_rk, t_values_rk, trajectory_rk = runge_kutta4_lorenz(n, t_start, t_end, u_start, return_trajectory=True)
    u_final_fe, t_values_fe, trajectory_fe = forward_euler_lorenz(n, t_start, t_end, u_start, return_trajectory=True)
    
    trajectories_rk.append(trajectory_rk)
    trajectories_fe.append(trajectory_fe)
    t_values_list.append(t_values_rk)

# グラフ作成: t ∈ [0, 50]
fig1 = plt.figure(figsize=(12, 6))

# Runge-Kutta法: x(t)
for i, epsilon in enumerate(epsilons):
    mask = t_values_list[i] <= 50
    plt.plot(t_values_list[i][mask], trajectories_rk[i][mask, 0], 
             color=colors[i], label=f'ε={epsilon}', alpha=0.8, linewidth=1.0)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Runge-Kutta: x(t) (t∈[0,50])')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lorenz_x_t0_50.png', dpi=150)
plt.show()

# グラフ作成: t ∈ [50, 100]
fig2 = plt.figure(figsize=(12, 6))

# Runge-Kutta法: x(t)
for i, epsilon in enumerate(epsilons):
    mask = t_values_list[i] > 50
    plt.plot(t_values_list[i][mask], trajectories_rk[i][mask, 0], 
             color=colors[i], label=f'ε={epsilon}', alpha=0.8, linewidth=1.0)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Runge-Kutta: x(t) (t∈[50,100])')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lorenz_x_t50_100.png', dpi=150)
plt.show()
    