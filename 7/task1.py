import argparse
from pathlib import Path
from typing import Literal, List, Tuple

import math
import matplotlib.pyplot as plt

BCType = Literal["dirichlet", "neumann"]
MethodType = Literal["euler", "cn"]

def u0_gaussian(x: float) -> float:
    """問題1の初期条件 u0(x) = 1/sqrt(2π) * exp(-1/2*(x-5)^2)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * (x - 5.0) ** 2)

def init_u_from_u0(N: int, dx: float, u0_func=u0_gaussian) -> List[float]:
    """
    u^0_j を (8.65) の自然な離散化で作る。
    配列は ghost を含めて長さ N+2（index 0..N+1）。
    """
    u = [0.0] * (N + 2)
    for j in range(1, N + 1):
        xL = (j - 1) * dx
        xR = j * dx
        u[j] = 0.5 * (u0_func(xL) + u0_func(xR))
    return u

def apply_bc(u: List[float], dx: float, bc: BCType,
             uL: float = 0.0, uR: float = 0.0,
             JL: float = 0.0, JR: float = 0.0) -> None:
    """
    ghost cell u[0], u[N+1] を境界条件で埋める。
    - Dirichlet: u[0]=uL, u[N+1]=uR
    - Neumann:   u[0]=u[1]-JL*dx, u[N+1]=u[N]+JR*dx
    """
    N = len(u) - 2
    if bc == "dirichlet":
        u[0] = uL
        u[N + 1] = uR
    elif bc == "neumann":
        u[0] = u[1] - JL * dx
        u[N + 1] = u[N] + JR * dx
    else:
        raise ValueError("bc must be 'dirichlet' or 'neumann'")

def step_euler_explicit(u: List[float], c: float,
                        dx: float, bc: str,
                        uL: float = 0.0, uR: float = 0.0,
                        JL: float = 0.0, JR: float = 0.0) -> List[float]:
    """
    オイラー陽解法 (8.17):
      u^{n+1}_j = u^n_j + c (u^n_{j-1} - 2u^n_j + u^n_{j+1})
    境界は各ステップで apply_bc する。
    """
    N = len(u) - 2
    un = u[:]  # 入力を壊さない
    apply_bc(un, dx, bc, uL=uL, uR=uR, JL=JL, JR=JR)

    up = un[:]
    for j in range(1, N + 1):
        up[j] = un[j] + c * (un[j - 1] - 2.0 * un[j] + un[j + 1])

    apply_bc(up, dx, bc, uL=uL, uR=uR, JL=JL, JR=JR)
    return up

def precompute_cn_lu(N: int, c: float, bc: BCType) -> Tuple[List[float], List[float]]:
    """
    クランク-ニコルソンの行列A（(8.38)(8.39)）に対する LU 用の
    alpha, beta を (8.46) の漸化式で前計算して返す。
    ここでは A は三重対角で、上/下対角は常に -c/2。
    Dirichlet: diag[1..N]=1+c
    Neumann:   diag[1]=diag[N]=1+c/2, それ以外は 1+c
    """
    if N < 2:
        raise ValueError("N must be >= 2 for this LU routine (課題のNは十分大きいはず)")

    # a_j (diag), b_j (upper), c_j (lower) for j=1..N in 1-index
    a = [0.0] * N
    b = [-c / 2.0] * (N - 1)
    lower = [-c / 2.0] * (N - 1)

    if bc == "dirichlet":
        for i in range(N):
            a[i] = 1.0 + c
    elif bc == "neumann":
        for i in range(N):
            a[i] = 1.0 + c
        a[0] = 1.0 + c / 2.0
        a[-1] = 1.0 + c / 2.0
    else:
        raise ValueError("bc must be 'dirichlet' or 'neumann'")

    alpha = [0.0] * N
    beta = [0.0] * (N - 1)

    alpha[0] = a[0]
    beta[0] = b[0] / alpha[0]
    for i in range(1, N - 1):
        alpha[i] = a[i] - lower[i - 1] * beta[i - 1]
        beta[i] = b[i] / alpha[i]
    alpha[N - 1] = a[N - 1] - lower[N - 2] * beta[N - 2]

    return alpha, beta

def solve_tridiag_with_precomputed_lu(z: List[float],
                                     alpha: List[float],
                                     beta: List[float],
                                     lower_const: float) -> List[float]:
    """
    LU 分解済み（alpha,beta）を使って Ax=z を解く。
    - 前進代入: (8.49)
    - 後退代入: (8.51)
    lower_const は下対角成分（ここでは -c/2）で一定なので引数にする。
    """
    N = len(z)
    y = [0.0] * N
    x = [0.0] * N

    # forward (Ly=z)
    y[0] = z[0] / alpha[0]
    for i in range(1, N):
        y[i] = (z[i] - lower_const * y[i - 1]) / alpha[i]

    # backward (Ux=y), note: diag(U)=1
    x[N - 1] = y[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = y[i] - beta[i] * x[i + 1]

    return x

def step_crank_nicolson(u: List[float], c: float,
                        dx: float, bc: BCType,
                        alpha: List[float], beta: List[float],
                        uL: float = 0.0, uR: float = 0.0,
                        JL: float = 0.0, JR: float = 0.0) -> List[float]:
    """
    クランク-ニコルソン (8.29) を、z (8.40)/(8.41) を作って Ax=z を LU で解く。
    alpha,beta は precompute_cn_lu で作ったものを渡す（各ステップで再計算しない）。
    """
    N = len(u) - 2
    un = u[:]
    apply_bc(un, dx, bc, uL=uL, uR=uR, JL=JL, JR=JR)

    # z は長さN（u[1..N]に対応）
    z = [0.0] * N
    if bc == "dirichlet":
        # (8.40)
        z[0] = (1.0 - c) * un[1] + c * uL + (c / 2.0) * un[2]
        for j in range(2, N):  # j=2..N-1
            z[j - 1] = (1.0 - c) * un[j] + (c / 2.0) * (un[j - 1] + un[j + 1])
        z[N - 1] = (1.0 - c) * un[N] + c * uR + (c / 2.0) * un[N - 1]
    elif bc == "neumann":
        # (8.41)
        z[0] = (1.0 - c / 2.0) * un[1] - c * JL * dx + (c / 2.0) * un[2]
        for j in range(2, N):  # j=2..N-1
            z[j - 1] = (1.0 - c) * un[j] + (c / 2.0) * (un[j - 1] + un[j + 1])
        z[N - 1] = (1.0 - c / 2.0) * un[N] + c * JR * dx + (c / 2.0) * un[N - 1]
    else:
        raise ValueError("bc must be 'dirichlet' or 'neumann'")

    # 下対角は常に -c/2
    x = solve_tridiag_with_precomputed_lu(z, alpha, beta, lower_const=-c / 2.0)

    up = un[:]
    for j in range(1, N + 1):
        up[j] = x[j - 1]

    apply_bc(up, dx, bc, uL=uL, uR=uR, JL=JL, JR=JR)
    return up

def x_for_plot(j: int, dx: float) -> float:
    """注意書き通り、セル中心の座標 x=(j-1/2)dx を返す。"""
    return (j - 0.5) * dx





def write_profile(filepath: Path, u: List[float], dx: float, t: float) -> None:
    """
    1時刻ぶんの u(x,t) をテキスト出力する。
    形式:
      # t=...
      # x u
      x u
      ...
    x はセル中心 x=(j-1/2)dx を使用（課題の注意書きに合わせる）。
    """
    N = len(u) - 2
    with filepath.open("w", encoding="utf-8") as f:
        f.write(f"# t={t:.2f}\n")
        f.write("# x u\n")
        for j in range(1, N + 1):
            x = (j - 0.5) * dx
            f.write(f"{x:.10f} {u[j]:.16e}\n")


def run_case(
    case_name: str,
    method: MethodType,
    bc: BCType,
    dt: float,
    dx: float,
    N: int,
    outdir: Path,
) -> dict:
    """
    1ケースを n=0..500 まで回し、n=100,200,300,400,500 (t=1..5) を出力する。
    結果を辞書形式で返す（グラフ作成用）。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    T = 500
    snap_steps = {100, 200, 300, 400, 500}

    c = dt / (dx * dx)

    # 初期条件（課題の注意書きに沿った離散化を使う前提）
    u = init_u_from_u0(N, dx, u0_func=u0_gaussian)

    # 課題1は uL=uR=0 or JL=JR=0 が指定
    uL = uR = 0.0
    JL = JR = 0.0

    # 初期時刻で境界を適用
    apply_bc(u, dx, bc, uL=uL, uR=uR, JL=JL, JR=JR)

    # CN の場合は LU を前計算
    alpha: List[float] = []
    beta: List[float] = []
    if method == "cn":
        alpha, beta = precompute_cn_lu(N, c, bc)

    # 結果を保存する辞書
    results = {}

    # 時間発展
    for n in range(T):
        if method == "euler":
            u = step_euler_explicit(u, c, dx, bc, uL=uL, uR=uR, JL=JL, JR=JR)
        elif method == "cn":
            u = step_crank_nicolson(u, c, dx, bc, alpha, beta, uL=uL, uR=uR, JL=JL, JR=JR)
        else:
            raise ValueError("method must be 'euler' or 'cn'")

        step = n + 1  # いまの u は n+1 に対応
        if step in snap_steps:
            t = step * dt  # t=1,2,3,4,5
            t_int = int(round(t))  # ファイル名用（1..5）
            # データをテキストファイルにも保存
            write_profile(outdir / f"{case_name}_t{t_int}.dat", u, dx, t)
            # グラフ用にデータを保存
            x_vals = [x_for_plot(j, dx) for j in range(1, N + 1)]
            u_vals = u[1:N+1]
            results[t_int] = (x_vals, u_vals)
    
    return results


def main() -> None:
    """
    課題1（問題1）の 4 ケースを計算して出力する main。
    条件：
      1) Euler + Dirichlet, dt=0.01, dx=0.5, N=20
      2) Euler + Neumann,  dt=0.01, dx=0.5, N=20
      3) CN    + Dirichlet, dt=0.01, dx=0.05, N=200
      4) CN    + Neumann,  dt=0.01, dx=0.05, N=200
    出力：
      t=1..5 (n=100..500) の u(x,t) のグラフ
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="out_problem1", help="出力先ディレクトリ")
    args = parser.parse_args()

    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)

    cases: List[Tuple[str, MethodType, BCType, float, float, int]] = [
        ("case1_euler_dirichlet", "euler", "dirichlet", 0.01, 0.5, 20),
        ("case2_euler_neumann",   "euler", "neumann",   0.01, 0.5, 20),
        ("case3_cn_dirichlet",    "cn",    "dirichlet", 0.01, 0.05, 200),
        ("case4_cn_neumann",      "cn",    "neumann",   0.01, 0.05, 200),
    ]

    # 各ケースを実行してデータを保存
    results_dict = {}
    for name, method, bc, dt, dx, N in cases:
        results = run_case(
            case_name=name,
            method=method,
            bc=bc,
            dt=dt,
            dx=dx,
            N=N,
            outdir=base / name,
        )
        results_dict[name] = results
    
    # Dirichlet条件とNeumann条件について、EulerとCNを重ね合わせたグラフを作成
    for bc_type in ["dirichlet", "neumann"]:
        euler_name = f"case1_euler_{bc_type}" if bc_type == "dirichlet" else f"case2_euler_{bc_type}"
        cn_name = f"case3_cn_{bc_type}" if bc_type == "dirichlet" else f"case4_cn_{bc_type}"
        
        euler_results = results_dict[euler_name]
        cn_results = results_dict[cn_name]
        
        # 各時刻についてグラフを作成
        for t in [1, 2, 3, 4, 5]:
            plt.figure(figsize=(10, 6))
            
            # Eulerを背面に描画
            x_euler, u_euler = euler_results[t]
            plt.plot(x_euler, u_euler, 'ro-', label=f'Euler ({bc_type.capitalize()})', linewidth=2, markersize=6, alpha=0.7)
            
            # CNを前面に描画
            x_cn, u_cn = cn_results[t]
            plt.plot(x_cn, u_cn, 'b-', label=f'CN ({bc_type.capitalize()})', linewidth=1.5)
            
            plt.xlabel('x')
            plt.ylabel('u')
            plt.title(f't = {t} ({bc_type.capitalize()} BC)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # グラフを保存
            graph_dir = base / f"graphs_{bc_type}"
            graph_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(graph_dir / f"comparison_t{t}.png", dpi=150, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    main()
