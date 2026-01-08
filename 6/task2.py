import random
import time
import math

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


def run_experiment(n, m, trials=5):
    times = []
    nodes = []

    for _ in range(trials):
        G, d = generate_random_graph(n, m)
        s = 0
        t = n - 1

        if (s, t) not in d:
            G[s].append(t)
            d[(s, t)] = random.randint(-10, 10)

        start = time.process_time()
        _, _, node_count = shortest_path_branching(G, d, s, t)
        end = time.process_time()

        times.append((end - start) * 1000)
        nodes.append(node_count)

    def mean_std(arr):
        mean = sum(arr) / len(arr)
        std = math.sqrt(sum((x - mean) ** 2 for x in arr) / len(arr))
        return mean, std

    return mean_std(times), mean_std(nodes)


# 実行（節点数5〜15で実験）
print("n | time_mean(ms) | time_std | nodes_mean | nodes_std")
print("-" * 55)

for n in range(5, 16):
    m = n * (n - 1) // 2
    (t_mean, t_std), (n_mean, n_std) = run_experiment(n, m)
    print(f"{n:2d} | {t_mean:12.2f} | {t_std:8.2f} | {n_mean:11.1f} | {n_std:9.1f}")
