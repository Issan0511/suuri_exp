import random
import time
import math

# ----------------------------
# 課題2：限定なし（全単純路DFS）
# ----------------------------
def shortest_path_branching(G, d, s, t):
    P = []
    best_length = float("inf")
    best_path = None
    node_count = 0

    def dfs(u, current_length):
        nonlocal best_length, best_path, node_count
        node_count += 1
        P.append(u)

        if u == t:
            if current_length < best_length:
                best_length = current_length
                best_path = P.copy()
        else:
            for v in G.get(u, []):
                if v not in P:
                    dfs(v, current_length + d[(u, v)])

        P.pop()

    dfs(s, 0.0)
    return best_path, best_length, node_count


# ----------------------------
# 課題4：限定あり（前処理LB + 下界値テスト）
#  - LB_k[u] = 「辺数 <= k」で u->t の最短歩道（単純路制約は無視）
#  - Bellman-Ford の各反復の dist を保存して作る（計算量 O(nm) のまま）
# ----------------------------
def precompute_lb_by_steps(G, d, t, n):
    """
    LB[k][u] = u->t の最短距離（歩道OK）ただし使用辺数 <= k
    k=0..n-1 を前処理で全部作る。
    """
    INF = float("inf")
    LB = [[INF] * n for _ in range(n)]  # LB[k][u]
    LB[0][t] = 0.0

    # k 回目 = 辺数 <= k の最短歩道
    for k in range(1, n):
        prev = LB[k - 1]
        cur = LB[k]

        # まず「何もしない（辺を増やさない）」を許す：cur[u] <= prev[u]
        for u in range(n):
            cur[u] = prev[u]

        # 1 辺伸ばす遷移：cur[u] <= d(u,v) + prev[v]
        # （u->t を求めたいのでこの形が自然）
        for u in range(n):
            for v in G.get(u, []):
                if prev[v] != INF:
                    cand = d[(u, v)] + prev[v]
                    if cand < cur[u]:
                        cur[u] = cand

    return LB


def greedy_initial_path(G, d, s, t, LB, n):
    """
    LB をガイドにして、とりあえずの実行可能単純路を作る（上界を早く得る用）
    失敗したら (None, inf)。
    """
    INF = float("inf")
    visited = [False] * n
    path = [s]
    visited[s] = True
    u = s
    length = 0.0

    while u != t:
        remaining = n - len(path)  # これから使える最大辺数（単純路なので）
        if remaining <= 0:
            return None, INF

        # 次候補（単純路制約のみ）
        cand = []
        for v in G.get(u, []):
            if not visited[v] and LB[remaining - 1][v] != INF:
                cand.append(v)

        if not cand:
            return None, INF

        # d(u,v) + 下界 の小さい順に選ぶ（雑だが速い）
        v = min(cand, key=lambda x: d[(u, x)] + LB[remaining - 1][x])

        length += d[(u, v)]
        path.append(v)
        visited[v] = True
        u = v

    return path, length


def shortest_path_branching_bb(G, d, s, t):
    """
    Branch & Bound 版（課題4）
    - LB[k][u] を前処理で作る（Bellman-Ford の途中結果）
    - 途中で remaining edges を使って強めの下界 LB[remaining][u] を参照
    - 下界値テスト：current_length + LB[remaining][u] >= best_length なら枝刈り
    - 探索順を d(u,v)+LB でソートして上界を早く更新しやすくする
    """
    n = len(G)
    INF = float("inf")

    LB = precompute_lb_by_steps(G, d, t, n)

    # 上界（incumbent）を早めに用意（貪欲で1本作る）
    best_path, best_length = greedy_initial_path(G, d, s, t, LB, n)
    if best_path is None:
        best_length = INF
        best_path = None

    P = []
    inP = [False] * n
    node_count = 0
    pruned_count = 0

    def dfs(u, current_length):
        nonlocal best_length, best_path, node_count, pruned_count
        node_count += 1
        P.append(u)
        inP[u] = True

        # 残りの最大辺数（単純路なので s->t 全体で <= n-1 辺）
        remaining = n - len(P)
        # u->t を「辺数 <= remaining」で行けないならここで終了
        if remaining < 0 or LB[remaining][u] == INF:
            pruned_count += 1
            inP[u] = False
            P.pop()
            return

        # 下界値テスト（限定操作）
        if current_length + LB[remaining][u] >= best_length:
            pruned_count += 1
            inP[u] = False
            P.pop()
            return

        if u == t:
            if current_length < best_length:
                best_length = current_length
                best_path = P.copy()
        else:
            # 探索順を工夫：見込みが良さそうな v から先に（上界更新を早める）
            cand = [v for v in G.get(u, []) if not inP[v]]
            # 次に 1 辺進むので remaining-1 の下界を使う
            if remaining - 1 >= 0:
                cand.sort(key=lambda v: d[(u, v)] + LB[remaining - 1][v])
            for v in cand:
                dfs(v, current_length + d[(u, v)])

        inP[u] = False
        P.pop()

    dfs(s, 0.0)
    return best_path, best_length, node_count, pruned_count


# ----------------------------
# グラフ生成（あなたのまま）
# ----------------------------
def generate_random_graph(n, m, weight_low=-10, weight_high=10):
    V = list(range(n))
    all_edges = [(i, j) for i in V for j in V if i != j]
    random.shuffle(all_edges)
    E = all_edges[:m]

    G = {i: [] for i in V}
    d = {}

    for (u, v) in E:
        G[u].append(v)
        d[(u, v)] = random.randint(weight_low, weight_high)

    return G, d


def mean_std(arr):
    mean = sum(arr) / len(arr)
    std = math.sqrt(sum((x - mean) ** 2 for x in arr) / len(arr))
    return mean, std


# ----------------------------
# 同一グラフで比較（これ大事：別々に回すと差がノイズになる）
# ----------------------------
def run_experiment_compare(n, m, trials=5):
    times_plain, nodes_plain = [], []
    times_bb, nodes_bb, prunes_bb = [], [], []

    for _ in range(trials):
        G, d = generate_random_graph(n, m)
        s = 0
        t = n - 1

        # s->t を必ず入れる（実行可能解の保証）
        if (s, t) not in d:
            G[s].append(t)
            d[(s, t)] = random.randint(-10, 10)

        # 限定なし
        st = time.perf_counter()
        _, _, nc_plain = shortest_path_branching(G, d, s, t)
        ed = time.perf_counter()
        times_plain.append((ed - st) * 1000)
        nodes_plain.append(nc_plain)

        # BB（限定あり）
        st = time.perf_counter()
        _, _, nc_bb, pr_bb = shortest_path_branching_bb(G, d, s, t)
        ed = time.perf_counter()
        times_bb.append((ed - st) * 1000)
        nodes_bb.append(nc_bb)
        prunes_bb.append(pr_bb)

    return (mean_std(times_plain), mean_std(nodes_plain),
            mean_std(times_bb), mean_std(nodes_bb), mean_std(prunes_bb))


# ----------------------------
# 実行
# ----------------------------
print("n | time_plain(ms)        | nodes_plain        || time_BB(ms)           | nodes_BB           | pruned_BB")
print("  | mean      std         | mean      std      || mean      std         | mean      std      | mean      std")
print("-" * 110)

for n in range(5, 16):
    m = n * (n - 1) // 2
    (tp_mean, tp_std), (np_mean, np_std), (tb_mean, tb_std), (nb_mean, nb_std), (pr_mean, pr_std) = run_experiment_compare(n, m)

    print(f"{n:2d} | "
          f"{tp_mean:11.3e} {tp_std:11.3e} | {np_mean:11.3e} {np_std:11.3e} || "
          f"{tb_mean:11.3e} {tb_std:11.3e} | {nb_mean:11.3e} {nb_std:11.3e} | "
          f"{pr_mean:11.3e} {pr_std:11.3e}")
