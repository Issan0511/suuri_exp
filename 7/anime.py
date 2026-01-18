from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_dat_series(folder: str, pattern: str = "*.dat"):
    files = sorted(Path(folder).glob(pattern))
    series = []
    times = []
    for fp in files:
        # 先頭2行はコメント想定 (# t=..., # x u)
        data = np.loadtxt(fp, comments="#")
        x = data[:, 0]
        u = data[:, 1]
        series.append(u)
        # ファイル先頭の "# t=..." から時刻を取るなら別途読む（簡略化してファイル名でも可）
        times.append(fp.name)
    return x, series, times

def animate_from_files(folder: str):
    x, series, times = load_dat_series(folder)

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, series[0])
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    title = ax.set_title(times[0])

    ax.set_ylim(min(map(np.min, series)), max(map(np.max, series)))

    def update(i):
        line.set_ydata(series[i])
        title.set_text(times[i])
        return line, title

    anim = FuncAnimation(fig, update, frames=len(series), interval=400, blit=False)
    plt.show()

if __name__ == "__main__":
    # 例: out_problem1/case1_euler_dirichlet の中の dat をアニメ化
    animate_from_files("out_problem1/case4_cn_neumann")
