import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
hbar = 1.0
m = 1.0
k = 1.0

N = 400
dx = 0.05
L = N * dx  # L = 20
dt = 0.001

# 空間格子点
x = np.zeros(N + 2)  # j = 0, 1, ..., N, N+1 (境界条件用)
for j in range(1, N + 1):
    x[j] = (j - 0.5) * dx - L / 2  # (8.78)

# 初期条件
R = np.zeros(N + 2)
I = np.zeros(N + 2)

# R_j^0 の設定 (8.83)
for j in range(1, N + 1):
    R[j] = (np.sqrt(2) / np.pi**0.25) * np.exp(-2 * (x[j] - 5)**2)

# 周期境界条件の適用 (8.79), (8.80)
R[0] = R[N]
R[N + 1] = R[1]

# I_j^0 の設定 (8.84)
for j in range(1, N + 1):
    laplacian_R = (R[j - 1] - 2 * R[j] + R[j + 1]) / dx**2
    I[j] = -dt * (-0.5 * laplacian_R + 0.5 * x[j]**2 * R[j])

# 周期境界条件の適用 (8.81), (8.82)
I[0] = I[N]
I[N + 1] = I[1]

# 結果を保存するリスト
results = {}
output_times_original = [1, 2, 3, 4, 5, 6, 7, 8]
output_steps_original = [int(t / dt) for t in output_times_original]

# 追加: π倍数での出力時刻
output_times_pi = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2]
output_steps_pi = [int(t / dt) for t in output_times_pi]
results_pi = {}

# Pの最大値を保存するリスト
P_max_history = []
t_history = []

# 時間発展
n_max = max(max(output_steps_original), max(output_steps_pi))

# t=0の保存
P0 = R[1:N+1]**2 + I[1:N+1]**2
results_pi[0] = P0.copy()
P_max_history.append(np.max(P0))
t_history.append(0)
print(f"t = 0.0000 (n = 0): 確率密度の総和 = {np.sum(P0) * dx:.6f}")

for n in range(n_max):
    # R^{n+1} の計算 (8.76)
    R_new = np.zeros(N + 2)
    for j in range(1, N + 1):
        laplacian_I = (I[j - 1] - 2 * I[j] + I[j + 1]) / dx**2
        R_new[j] = R[j] + dt * (-0.5 * laplacian_I + 0.5 * x[j]**2 * I[j])
    
    # 周期境界条件 (8.79), (8.80)
    R_new[0] = R_new[N]
    R_new[N + 1] = R_new[1]
    
    # I^{n+1} の計算 (8.77) - R^{n+1} を使用
    I_new = np.zeros(N + 2)
    for j in range(1, N + 1):
        laplacian_R_new = (R_new[j - 1] - 2 * R_new[j] + R_new[j + 1]) / dx**2
        I_new[j] = I[j] - dt * (-0.5 * laplacian_R_new + 0.5 * x[j]**2 * R_new[j])
    
    # 周期境界条件 (8.81), (8.82)
    I_new[0] = I_new[N]
    I_new[N + 1] = I_new[1]
    
    # 更新
    R = R_new
    I = I_new
    
    # 各ステップでPの最大値を保存
    step = n + 1
    P_current = R[1:N+1]**2 + I[1:N+1]**2
    P_max_history.append(np.max(P_current))
    t_history.append(step * dt)
    
    # 出力時刻で保存
    if step in output_steps_original:
        t = step * dt
        P = R[1:N+1]**2 + I[1:N+1]**2
        results[t] = P.copy()
        print(f"t = {t:.1f} (n = {step}): 確率密度の総和 = {np.sum(P) * dx:.6f}")
    
    if step in output_steps_pi:
        t_pi = output_times_pi[output_steps_pi.index(step)]
        P = R[1:N+1]**2 + I[1:N+1]**2
        results_pi[t_pi] = P.copy()
        print(f"t = {step * dt:.4f} (n = {step}): 確率密度の総和 = {np.sum(P) * dx:.6f}")

# 確率密度のプロット (t=1,2,3,4,5,6,7,8)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

x_plot = x[1:N+1]

for idx, t in enumerate(output_times_original):
    ax = axes[idx]
    P = results[t]
    ax.plot(x_plot, P, 'b-', linewidth=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x,t)$')
    ax.set_title(f'$t = {t}$')
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs_schrodinger/probability_density.png', dpi=150)
plt.show()

# 確率密度のプロット (t=0, π/2, π, 3π/2, 2π, 5π/2)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

pi_labels = ['0', r'\pi/2', r'\pi', r'3\pi/2', r'2\pi', r'5\pi/2']

for idx, (t, label) in enumerate(zip(output_times_pi, pi_labels)):
    ax = axes[idx]
    P = results_pi[t]
    ax.plot(x_plot, P, 'b-', linewidth=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x,t)$')
    ax.set_title(f'$t = {label}$')
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs_schrodinger/probability_density_pi.png', dpi=150)
plt.show()

# Pの最大値の時間発展プロット
plt.figure(figsize=(10, 6))
plt.plot(t_history, P_max_history, 'b-', linewidth=0.5)
plt.xlabel('$t$')
plt.ylabel('$\\max_x P(x,t)$')
plt.title('Time Evolution of Maximum Probability Density')
plt.grid(True, alpha=0.3)

# π倍数に縦線を追加
for i in range(1, 4):
    plt.axvline(x=i*np.pi, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.savefig('graphs_schrodinger/P_max_evolution.png', dpi=150)
plt.show()

# Pの最大値の最大と最小を出力
P_max_arr = np.array(P_max_history)
print(f"\n=== Pの最大値の統計 ===")
print(f"max(P_max) = {np.max(P_max_arr):.6f} (t = {t_history[np.argmax(P_max_arr)]:.4f})")
print(f"min(P_max) = {np.min(P_max_arr):.6f} (t = {t_history[np.argmin(P_max_arr)]:.4f})")

# Pの最大値の逆数の時間発展プロット
P_max_inv = 1.0 / P_max_arr
plt.figure(figsize=(10, 6))
plt.plot(t_history, P_max_inv, 'b-', linewidth=0.5)
plt.xlabel('$t$')
plt.ylabel('$1 / \\max_x P(x,t)$')
plt.title('Time Evolution of Inverse Maximum Probability Density')
plt.grid(True, alpha=0.3)

# π倍数に縦線を追加
for i in range(1, 4):
    plt.axvline(x=i*np.pi, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.savefig('graphs_schrodinger/P_max_inv_evolution.png', dpi=150)
plt.show()

# 時空間プロット (確率密度の等高線)
# 全時刻のデータを保存するために再計算
print("\n時空間プロットを作成中...")

# 初期条件の再設定
R = np.zeros(N + 2)
I = np.zeros(N + 2)

for j in range(1, N + 1):
    R[j] = (np.sqrt(2) / np.pi**0.25) * np.exp(-2 * (x[j] - 5)**2)

R[0] = R[N]
R[N + 1] = R[1]

for j in range(1, N + 1):
    laplacian_R = (R[j - 1] - 2 * R[j] + R[j + 1]) / dx**2
    I[j] = -dt * (-0.5 * laplacian_R + 0.5 * x[j]**2 * R[j])

I[0] = I[N]
I[N + 1] = I[1]

# 時空間データの保存
save_interval = 100  # 100ステップごとに保存
t_values = [0]
P_history = [R[1:N+1]**2 + I[1:N+1]**2]

for n in range(n_max):
    # R^{n+1} の計算
    R_new = np.zeros(N + 2)
    for j in range(1, N + 1):
        laplacian_I = (I[j - 1] - 2 * I[j] + I[j + 1]) / dx**2
        R_new[j] = R[j] + dt * (-0.5 * laplacian_I + 0.5 * x[j]**2 * I[j])
    
    R_new[0] = R_new[N]
    R_new[N + 1] = R_new[1]
    
    # I^{n+1} の計算
    I_new = np.zeros(N + 2)
    for j in range(1, N + 1):
        laplacian_R_new = (R_new[j - 1] - 2 * R_new[j] + R_new[j + 1]) / dx**2
        I_new[j] = I[j] - dt * (-0.5 * laplacian_R_new + 0.5 * x[j]**2 * R_new[j])
    
    I_new[0] = I_new[N]
    I_new[N + 1] = I_new[1]
    
    R = R_new
    I = I_new
    
    step = n + 1
    if step % save_interval == 0:
        t_values.append(step * dt)
        P_history.append(R[1:N+1]**2 + I[1:N+1]**2)

# 時空間データの準備
t_arr = np.array(t_values)
P_arr = np.array(P_history)
X, T = np.meshgrid(x_plot, t_arr)

# 等高線プロット
plt.figure(figsize=(10, 6))
levels = np.linspace(0, np.max(P_arr), 20)
plt.contour(X, T, P_arr, levels=levels, colors='black', linewidths=0.5)
plt.contourf(X, T, P_arr, levels=levels, cmap='hot')
plt.colorbar(label='$P(x,t) = |\\psi(x,t)|^2$')

# t=π, 2πに点線を追加
plt.axhline(y=np.pi, color='blue', linestyle='--', linewidth=1.5, label='$t=\\pi$')
plt.axhline(y=2*np.pi, color='cyan', linestyle='--', linewidth=1.5, label='$t=2\\pi$')
plt.legend(loc='upper right')

plt.xlabel('$x$')
plt.ylabel('$t$')
plt.title('Contour Plot of Probability Density $P(x,t)$')
plt.savefig('graphs_schrodinger/probability_density_contour.png', dpi=150)
plt.show()

print("\n計算完了")
