import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

def u0_fisher(x: float, b: float) -> float:
    """
    Fisher方程式の初期条件 (8.67):
    u0(x) = 1 / (1 + e^(bx-5))^2
    """
    return 1.0 / (1.0 + math.exp(b * x - 5.0)) ** 2

def init_u_fisher(N: int, dx: float, b: float) -> List[float]:
    """
    初期条件を離散化してu^0を作る。
    配列は ghost を含めて長さ N+2（index 0..N+1）。
    セル中心の座標 x=(j-1/2)dx で初期条件を評価する。
    """
    u = [0.0] * (N + 2)
    for j in range(1, N + 1):
        x = (j - 0.5) * dx
        u[j] = u0_fisher(x, b)
    return u

def apply_bc_fisher(u: List[float], N: int) -> None:
    """
    境界条件 (8.68)(8.69):
    u(0,t) = 1, u(L,t) = 0
    ghost cell で表現:
    u[0] = 1, u[N+1] = 0
    """
    u[0] = 1.0
    u[N + 1] = 0.0

def f_fisher(u_val: float) -> float:
    """
    反応項 f(u) = u(1-u)
    """
    return u_val * (1.0 - u_val)

def step_explicit_euler_fisher(u: List[float], c: float, dt: float, N: int) -> List[float]:
    """
    Fisher方程式に対するオイラー陽解法 (8.70):
    (u^{n+1}_j - u^n_j) / Δt = f(u^n_j) + (u^n_{j-1} - 2u^n_j + u^n_{j+1}) / Δx^2
    
    これを整理すると:
    u^{n+1}_j = u^n_j + Δt*f(u^n_j) + c*(u^n_{j-1} - 2u^n_j + u^n_{j+1})
    
    ここで c = Δt/(Δx^2)。
    """
    un = u[:]
    up = [0.0] * (N + 2)
    
    for j in range(1, N + 1):
        f_val = f_fisher(un[j])
        diffusion = c * (un[j-1] - 2.0*un[j] + un[j+1])
        up[j] = un[j] + dt * f_val + diffusion
    
    apply_bc_fisher(up, N)
    return up

def run_fisher_case(b: float, L: float, dx: float, dt: float, output_times: List[int]) -> dict:
    """
    Fisher方程式を1ケース実行する。
    
    Parameters:
    - b: 初期条件のパラメータ
    - L: 空間領域の長さ
    - dx: 空間刻み幅
    - dt: 時間刻み幅
    - output_times: 出力するステップ数のリスト
    
    Returns:
    - dict: {時刻: (x座標リスト, u値リスト)}
    """
    N = int(L / dx)
    c = dt / (dx * dx)
    
    # 初期条件
    u = init_u_fisher(N, dx, b)
    apply_bc_fisher(u, N)
    
    results = {}
    
    # t=0 (初期条件) を保存
    if 0 in output_times:
        x_vals = [(j - 0.5) * dx for j in range(1, N + 1)]
        u_vals = u[1:N+1]
        results[0] = (x_vals, u_vals[:])
    
    # 最大ステップ数
    max_steps = max(output_times) if output_times and max(output_times) > 0 else 0
    
    for n in range(max_steps):
        u = step_explicit_euler_fisher(u, c, dt, N)
        
        step = n + 1
        if step in output_times:
            t = step * dt
            # グラフ用にデータを保存
            x_vals = [(j - 0.5) * dx for j in range(1, N + 1)]
            u_vals = u[1:N+1]
            results[int(t)] = (x_vals, u_vals)
    
    return results

def main():
    """
    問題2: Fisher方程式をオイラー陽解法で解く
    
    条件:
    - L = 200
    - Δx = 0.05 (N = 4000)
    - Δt = 0.001
    - b = 0.25, 0.5, 1.0
    - 出力時刻: t = 10, 20, 30, 40 (n = 10000, 20000, 30000, 40000)
    """
    L = 200.0
    dx = 0.05
    dt = 0.001
    b_values = [0.25, 0.5, 1.0]
    output_times = [0, 10000, 20000, 30000, 40000]  # n の値 (0, 10000, 20000, 30000, 40000)
    
    # 各bについて計算
    all_results = {}
    for b in b_values:
        print(f"Computing for b = {b}...")
        results = run_fisher_case(b, L, dx, dt, output_times)
        all_results[b] = results
    
    # グラフを作成
    # 1. 各時刻について、異なるbの値を重ねて描画
    output_dir = Path("graphs_fisher")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    times = [0, 10, 20, 30, 40]
    colors = {'0.25': 'blue', '0.5': 'green', '1.0': 'red'}
    
    for t in times:
        plt.figure(figsize=(10, 6))
        for b in b_values:
            x_vals, u_vals = all_results[b][t]
            plt.plot(x_vals, u_vals, color=colors[str(b)], label=f'b = {b}', linewidth=1.5)
        
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Fisher Equation (t = {t})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 200)
        plt.ylim(-0.1, 1.1)
        
        plt.savefig(output_dir / f"fisher_t{t}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved graph for t = {t}")
    
    # 2. 各bについて、異なる時刻を重ねて描画
    for b in b_values:
        plt.figure(figsize=(10, 6))
        time_colors = {0: 'black', 10: 'blue', 20: 'green', 30: 'orange', 40: 'red'}
        
        for t in times:
            x_vals, u_vals = all_results[b][t]
            plt.plot(x_vals, u_vals, color=time_colors[t], label=f't = {t}', linewidth=1.5)
        
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Fisher Equation (b = {b})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 200)
        plt.ylim(-0.1, 1.1)
        
        plt.savefig(output_dir / f"fisher_b{str(b).replace('.', '_')}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved graph for b = {b}")

if __name__ == "__main__":
    main()
