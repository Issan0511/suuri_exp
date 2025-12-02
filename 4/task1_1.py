import numpy as np

N = 64
M = 10
T = 256
# 全試行で同じ初期条件：32番目だけ 1
s = np.zeros((M, N), dtype=int)
plot_MT = np.zeros((M, T), dtype=int) 
s[:, 31] = 1 
p = 0.7 

def prob_from_s_n_batch(s, n, p):
        # 6 通りすべての α を p から計算しておく
    # 順番は (s,n) = (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    vals = np.array([
        0.0,
        p**2,
        p**2*(2-p**2),
        p**2*(2-p**2),
        p**2*(p**3 - 2*p**2 - p + 3),
        p**2*(2-p)*(p**3 - 2*p**2 + 2)
    ])

    # s, n は (M, N) 配列
    index = s*3 + n          # (0,0,0,1,1,1)×3 + (0,1,2,0,1,2) → 0〜5 にマップ
    alpha = vals[index]      # 完全にブランチレスなベクトル演算
    return alpha

for t in range(T):
    right = np.roll(s, -1, axis=1)  # 右隣の状態
    left = np.roll(s, 1, axis=1)    # 左隣の状態
    n = right + left                # 隣接する 1 の数
    alpha = prob_from_s_n_batch(s, n, p)  # 各セルの α を計算
    rand = np.random.rand(M, N)    # 一様乱数を生成 
    s = (rand < alpha).astype(int)  # α と比較して次の状態を決定
    num_ones = np.sum(s, axis=1)  # 各試行の 1 の数をカウント
    plot_MT[:, t] = num_ones

# プロット
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for m in range(M):
    plt.plot(range(T), plot_MT[m, :], alpha=0.7, linewidth=0.8)

plt.xlabel('Time step (t)', fontsize=14)
plt.ylabel('Cells in state 1', fontsize=14)
plt.title(f'Evolution of Number of State-1 Cells (M={M} trials, N={N} cells, p={p})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1_1.png', dpi=300, bbox_inches='tight')
plt.show()

