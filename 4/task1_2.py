import numpy as np
import matplotlib.pyplot as plt

N = 64
M = 10000
T = 1024
p_values = [0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.66, 0.68, 0.7]

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

# 各 p の値に対してシミュレーションを実行
plt.figure(figsize=(12, 6))

for p in p_values:
    print(f'Computing for p = {p}...')
    
    # 初期化：全試行で同じ初期条件：32番目だけ 1
    s = np.zeros((M, N), dtype=int)
    s[:, 31] = 1
    
    # 各時刻の m(t) のサンプル平均を保存
    mean_m_t = np.zeros(T)
    
    for t in range(T):
        right = np.roll(s, -1, axis=1)  # 右隣の状態
        left = np.roll(s, 1, axis=1)    # 左隣の状態
        n = right + left                # 隣接する 1 の数
        alpha = prob_from_s_n_batch(s, n, p)  # 各セルの α を計算
        rand = np.random.rand(M, N)    # 一様乱数を生成 
        s = (rand < alpha).astype(int)  # α と比較して次の状態を決定
        
        # 各試行の 1 の数をカウントし、サンプル平均を計算
        num_ones = np.sum(s, axis=1)  # (M,) の配列
        mean_m_t[t] = np.mean(num_ones)
    
    # プロット
    plt.plot(range(T), mean_m_t, label=f'p = {p}', linewidth=2)
    print(f'  Final mean m(T-1) = {mean_m_t[-1]:.2f}')

plt.xlabel('Time step (t)', fontsize=14)
plt.ylabel('Mean of m(t)', fontsize=14)
plt.title(f'Sample Mean of m(t) for Different p Values (M={M} trials, N={N} cells)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save linear scale plot
plt.savefig('task1_2_sample_mean_linear.png', dpi=300, bbox_inches='tight')
print('\nSaved plot to task1_2_sample_mean_linear.png')

# Switch to log scale and save
plt.xscale('log')
plt.yscale('log')
plt.savefig('task1_2_sample_mean_log.png', dpi=300, bbox_inches='tight')
print('Saved plot to task1_2_sample_mean_log.png')

plt.show()



