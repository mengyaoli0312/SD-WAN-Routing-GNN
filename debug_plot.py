import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

from environment1 import SatelliteGraphEnv
from topology_builder import R_EARTH_KM, compute_elevation  # 用统一版本的 compute_elevation


def plot_elevation_2d(rg, rs, gnd_name="GS", sat_name="SAT", elev_attr=None):
    """
    2D elevation-angle diagram (correct arc):
      - x: tangential direction (local horizon)
      - y: radial direction (up)
      - arc centered at GS, between horizon and GS->SAT
    """
    # ----- 1. build local 2D frame -----
    u_r = rg / np.linalg.norm(rg)        # local "up"
    v = rs - rg                          # GS -> SAT
    v_tan = v - np.dot(v, u_r) * u_r     # remove radial component

    # 防止 v_tan 退化
    if np.linalg.norm(v_tan) < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, u_r)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        v_tan = tmp - np.dot(tmp, u_r) * u_r

    u_t = v_tan / np.linalg.norm(v_tan)  # local horizon direction (+x)

    R = np.linalg.norm(rg)
    # GS at (0, R)
    P = np.array([0.0, R])
    # SAT projected into this 2D frame
    Sx = np.dot(rs, u_t)
    Sy = np.dot(rs, u_r)
    S = np.array([Sx, Sy])

    # ----- 2. elevation from 2D geometry (for显示用，再算一遍) -----
    dx = S[0] - P[0]
    dy = S[1] - P[1]
    elev_geom = np.rad2deg(np.arctan2(dy, dx))  # angle from horizon to GS->SAT

    # 如果有 attribute 的 elevation，可以顺便打印对比
    if elev_attr is not None:
        print(f"[DEBUG] elevation from edge attribute: {elev_attr:.3f} deg")
    print(f"[DEBUG] elevation from 2D geometry:      {elev_geom:.3f} deg")

    # ----- 3. start plotting -----
    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Earth (light blue cap)
    earth = Circle(
        (0.0, 0.0),
        R,
        edgecolor="#1E90FF",
        facecolor="#87CEFA",
        alpha=0.25,
    )
    ax.add_patch(earth)

    # GS
    ax.plot(P[0], P[1], "o", color="black")
    ax.text(P[0] - 0.08 * R, P[1] + 0.03 * R, gnd_name, fontsize=9)

    # local horizon (GS-centered)
    horizon_half = 0.5 * R
    ax.plot(
        [-horizon_half, horizon_half],
        [P[1], P[1]],
        color="black",
        linewidth=1,
    )
    ax.text(
        -0.1 * R,
        P[1] - 0.06 * R,
        "Local horizon",
        fontsize=9,
    )

    # SAT and line GS->SAT
    ax.plot(S[0], S[1], "o", color="black")
    ax.text(S[0] + 0.03 * R, S[1] + 0.03 * R, sat_name, fontsize=9)
    ax.plot([P[0], S[0]], [P[1], S[1]], color="black", linewidth=1.5)

    # ----- 4. elevation arc -----
    if elev_geom > 0:
        dist_PS = np.linalg.norm([dx, dy])
        arc_radius = 0.4 * min(dist_PS, 0.6 * R)  # small arc near GS

        theta1 = 0.0          # along +x (horizon)
        theta2 = elev_geom    # up to GS->SAT direction
        angle_arc = Arc(
            (P[0], P[1]),
            2 * arc_radius,
            2 * arc_radius,
            angle=0.0,
            theta1=theta1,
            theta2=theta2,
            color="black",
            linewidth=1,
        )
        ax.add_patch(angle_arc)

        # label near middle of arc
        theta_mid = np.deg2rad(elev_geom / 2.0)
        label_r = arc_radius * 1.3
        tx = P[0] + label_r * np.cos(theta_mid)
        ty = P[1] + label_r * np.sin(theta_mid) + 0.015 * R
        # 如果有 attribute elevation，就优先显示 attribute 的值
        label_elev = elev_attr if elev_attr is not None else elev_geom
        ax.text(tx, ty, f"{label_elev:.1f}°", fontsize=10)
    else:
        ax.text(
            P[0] + 0.3 * R,
            P[1] + 0.06 * R,
            f"{elev_geom:.1f}° (below horizon)",
            fontsize=10,
        )

    # ----- 5. tight view window -----
    y_min = R - 0.2 * R
    y_max = max(S[1], P[1]) + 0.4 * R
    x_half_span = max(abs(S[0]), horizon_half) * 1.1

    ax.set_xlim(-x_half_span, x_half_span)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("Tangential direction (km)")
    ax.set_ylabel("Radial direction (km)")
    title_elev = elev_attr if elev_attr is not None else elev_geom
    ax.set_title(
        f"Elevation angle: {gnd_name} → {sat_name} (elev≈{title_elev:.1f}°)"
    )

    plt.tight_layout()
    plt.show()


def main():
    # 1) 初始化环境（所有 SAT-GND 都用统一新逻辑）
    env = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=[8, 32, 64],
    )

    G = env.G

    # 2) 找所有 SAT-GND 边；可以按 elevation_deg 最高的来选
    sat_gnd_edges = []
    for u, v, d in G.edges(data=True):
        if d.get("kind") != "SAT-GND":
            continue

        # 边上的 elevation 属性（新逻辑一定有，如果没有就 None）
        elev_attr = d.get("elevation_deg", None)

        # 找 ground / sat 节点
        if G.nodes[u]["kind"] == "ground":
            gnd, sat = u, v
        else:
            gnd, sat = v, u

        sat_gnd_edges.append((gnd, sat, elev_attr))

    if not sat_gnd_edges:
        print("[ERROR] No SAT-GND edges in graph. Check topology_builder logic.")
        return

    # 3) 按仰角从大到小排序，优先用 elevation_deg；没有的话当成 -999
    sat_gnd_edges.sort(
        key=lambda x: (x[2] if x[2] is not None else -999.0),
        reverse=True,
    )

    gnd, sat, elev_attr = sat_gnd_edges[0]
    print(f"[INFO] Using link: {gnd} → {sat}, elev_attr={elev_attr}")

    rg = np.array(G.nodes[gnd]["pos_xyz"], dtype=float)
    rs = np.array(G.nodes[sat]["pos_xyz"], dtype=float)

    # 顺便用 compute_elevation 再算一遍，输出对比
    elev_recomputed = compute_elevation(rg, rs)
    print(f"[DEBUG] recomputed elevation (ECEF): {elev_recomputed:.3f} deg")

    # 4) 画 2D 仰角图
    plot_elevation_2d(rg, rs, gnd_name=gnd, sat_name=sat, elev_attr=elev_attr)


if __name__ == "__main__":
    main()
