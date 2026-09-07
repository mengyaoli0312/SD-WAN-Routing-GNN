import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from astropy.time import Time

from environment1 import SatelliteGraphEnv
from topology_builder import R_EARTH_KM
from matplotlib.lines import Line2D


# ==============================
# 1. 一些工具：画地球
# ==============================

def plot_earth(ax):
    u = np.linspace(0, 2 * np.pi, 60)   # 经度
    v = np.linspace(0, np.pi, 30)       # 纬度

    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        alpha=0.15,
        color="lightskyblue",
        linewidth=0,
    )


def plot_satellite_graph(
    G,
    title="Iridium + Ground Stations (ECEF)",
    show_isl_edges=True,
    show_sat_gnd_edges=True,
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 地球
    plot_earth(ax)

    # 缓存节点坐标
    pos = {
        n: np.array(d["pos_xyz"], dtype=float)
        for n, d in G.nodes(data=True)
    }

    # 画节点
    for n, d in G.nodes(data=True):
        r = pos[n]
        if d.get("kind") == "ground":
            ax.scatter(r[0], r[1], r[2], c="green", s=40)
            ax.text(r[0], r[1], r[2], n, fontsize=8)
        elif d.get("kind") == "sat":
            ax.scatter(r[0], r[1], r[2], c="red", s=10)

    # 画边
    for u, v, d in G.edges(data=True):
        kind = d.get("kind", "")
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]

        if kind == "ISL" and show_isl_edges:
            ax.plot(x, y, z, color="orange", linewidth=0.8, alpha=0.7)
        elif kind == "SAT-GND" and show_sat_gnd_edges:
            ax.plot(x, y, z, color="royalblue", linewidth=1.0, alpha=0.8)

    # 坐标轴 & 标题
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(title)

    # 图例
    legend_elems = [
        Line2D([0], [0], color="orange", lw=1, label="ISL"),
        Line2D([0], [0], color="royalblue", lw=1, label="SAT-GND"),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=6,
            label="Satellite",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=8,
            label="Ground station",
        ),
    ]
    ax.legend(handles=legend_elems, loc="upper left")

    # 尽量让比例 1:1:1
    max_range = 0.0
    for i in range(3):
        coord = np.array([p[i] for p in pos.values()])
        max_range = max(max_range, coord.max() - coord.min())
    max_range *= 0.6

    center = np.mean(np.array(list(pos.values())), axis=0)
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.show()


# ==============================
# 2. 构建指定时间的环境快照
# ==============================

def build_env_at_time(t_iso: str):
    # ⚠️ 这里不要用 astropy Time 对象，直接传字符串（跟训练脚本一致）
    env = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=[8, 32, 64],
        epoch=t_iso,          # ✅ 用字符串
        K_paths=4,
        sg_max_dist_km=2500.0,
        isl_k=4,
        isl_max_dist_km=4500.0,
    )
    env.verbose = False

    # ✅ 关键：reset() 才会真正调用 _build_new_graph() 给 self.G 赋值
    env.reset()

    return env


# ==============================
# 3. MAIN：生成 3D 模型
# ==============================

if __name__ == "__main__":
    # 单次快照时间（你可以改）
    t_snapshot = "2025-12-05 06:00:00"

    print(f"[INFO] Building Iridium snapshot at {t_snapshot} ...")
    env_snapshot = build_env_at_time(t_snapshot)
    G = env_snapshot.G

    print("=== Graph basic info ===")
    print("num_nodes:", G.number_of_nodes())
    print("num_edges:", G.number_of_edges())
    print(
        "num_sat_nodes:",
        sum(d.get("kind") == "sat" for _, d in G.nodes(data=True)),
    )
    print(
        "num_ground_nodes:",
        sum(d.get("kind") == "ground" for _, d in G.nodes(data=True)),
    )
    print(
        "num_ISL_edges:",
        sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "ISL"),
    )
    print(
        "num_SAT_GND_edges:",
        sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "SAT-GND"),
    )

    # 画 3D
    title = f"Iridium-66 + GS (ECEF) @ {t_snapshot} UTC"
    plot_satellite_graph(G, title=title)
