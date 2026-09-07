import numpy as np
import matplotlib.pyplot as plt

from topology_builder import geodetic_to_ecef, R_EARTH_KM
from environment1 import SatelliteGraphEnv
from topology_builder import has_los_vec, compute_elevation

# ============================================================
#  Test 1: Check geodetic → ECEF consistency
# ============================================================

def check_ground_ecef(name, lat, lon, alt_km=0):
    print(f"=== Testing {name}")
    r = geodetic_to_ecef(lat, lon, alt_km)
    dist = np.linalg.norm(r)
    print("lat/lon:", lat, lon)
    print("ECEF xyz:", r)
    print("|r| =", dist, "km")
    print("expect:", R_EARTH_KM + alt_km)
    print("diff:", dist - (R_EARTH_KM + alt_km))
    print()


# ============================================================
#  Test 2: Compute elevation for ALL ground–sat links
#          and plot histogram
# ============================================================

def test_all_elevations(env):
    G = env.G

    ground_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "ground"]
    sat_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "sat"]

    elevations = []

    for g in ground_nodes:
        rg = np.array(G.nodes[g]["pos_xyz"], dtype=float)
        for s in sat_nodes:
            rs = np.array(G.nodes[s]["pos_xyz"], dtype=float)
            elev = compute_elevation(rg, rs)
            elevations.append(elev)

    elevations = np.array(elevations)

    print("\n========== Elevation Statistics ==========")
    print("Total links evaluated:", len(elevations))
    print("Visible (elev > 0°):  ", np.sum(elevations > 0))
    print("Percentage visible:    %.2f%%" % (np.mean(elevations > 0) * 100))
    print("Min elevation:         %.2f°" % np.min(elevations))
    print("Max elevation:         %.2f°" % np.max(elevations))
    print("==========================================")

    # ---- Plot histogram ----
    plt.figure(figsize=(7, 4))
    plt.hist(elevations, bins=40, color="skyblue", edgecolor="black")
    plt.title("Elevation angle distribution (all ground-sat links)")
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def debug_print_ground_visibility(env,
                                  elev_vis_threshold: float = 0.0,
                                  elev_select_threshold: float = 5.0):
    """
    对每个 ground station：
      1) 计算所有满足 LOS 且 elevation >= elev_vis_threshold 的“可见卫星全集”
      2) 标出哪些实际上被拓扑选中为 SAT-GND 边（elevation>=elev_select_threshold & top-K）

    参数：
      elev_vis_threshold:   用来定义“可见”的最低仰角（比如 0°）
      elev_select_threshold:和你 add_sat_ground_edges_with_elev 里 min_elev_deg 一致（比如 5°）
    """
    G = env.G

    ground_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "ground"]
    sat_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sat"]

    print("\n================ Ground visibility debug ================")
    print(f"Total ground nodes: {len(ground_nodes)}, sat nodes: {len(sat_nodes)}")
    print(f"Visible if elev >= {elev_vis_threshold}°, selectable if elev >= {elev_select_threshold}°")
    print("=========================================================\n")

    for g in ground_nodes:
        rg = np.array(G.nodes[g]["pos_xyz"], dtype=float)

        # 已经被图选中的 SAT-GND 邻居
        selected_sats = set(
            v if G.nodes[v]["kind"] == "sat" else u
            for u, v, d in G.edges(g, data=True)
            if d.get("kind") == "SAT-GND"
        )

        candidates = []  # (sat_name, elev_deg, has_los)

        for s in sat_nodes:
            rs = np.array(G.nodes[s]["pos_xyz"], dtype=float)

            # LOS 判断
            los = has_los_vec(rg, rs, R_earth_km=R_EARTH_KM, margin_km=0.0)

            elev = compute_elevation(rg, rs)

            if not los:
                continue
            if elev < elev_vis_threshold:
                continue

            candidates.append((s, elev, los))

        # 按仰角从大到小排序
        candidates.sort(key=lambda x: -x[1])

        print(f"--- Ground station: {g} ---")
        if not candidates:
            print("  No visible satellites (LOS & elev >= %.1f°)." % elev_vis_threshold)
            print()
            continue

        print("  Visible satellites (LOS & elev >= %.1f°): %d" %
              (elev_vis_threshold, len(candidates)))

        for s, elev, los in candidates:
            mark = ""
            if s in selected_sats and elev >= elev_select_threshold:
                mark = "  <-- SELECTED in topology"
            elif s in selected_sats:
                mark = "  <-- SELECTED (but elev < select_threshold?)"

            print("   - %-12s  elev = %6.2f°  LOS = %s%s" %
                  (s, elev, str(los), mark))

        print()

def print_ground_sat_links(G):
    """
    打印每个地面站所连接的所有卫星，并显示仰角 elevation。
    仅打印 kind="SAT-GND" 的边。
    """

    print("\n========== Ground Station → Satellite Links ==========")

    # 找出所有 ground 节点
    ground_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "ground"]

    for g in ground_nodes:
        links = []
        for u, v, d in G.edges(g, data=True):
            if d.get("kind") == "SAT-GND":
                sat = v if u == g else u
                elev = d.get("elevation_deg", None)
                dist = d.get("distance_km", None)
                delay = d.get("delay_ms", d.get("base_delay_ms", None))
                links.append((sat, elev, dist, delay))

        print(f"\n---- {g} ----")
        if not links:
            print("  (No SAT links)")
            continue

        # 按仰角从高到低排序
        links.sort(key=lambda x: -x[1])

        for sat, elev, dist, delay in links:
            print(f"  {sat:10s} | elev={elev:6.2f}° | dist={dist:8.1f} km | delay={delay:6.2f} ms")

    print("======================================================\n")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    # ---- Test 1: ECEF correctness ----
    tests = [
            ("Enschede", 52.2215, 6.8937, 0.0),
            ("Osnabrück", 52.2799, 8.0472, 0.0),

            ("New York City, USA", 40.7128, -74.0060, 0.0),
            ("San Francisco, USA", 37.7749, -122.4194, 0.0),
            ("Paris, France", 48.8566, 2.3522, 0.0),
            ("Moscow, Russia", 55.7558, 37.6173, 0.0),
            ("Tokyo, Japan", 35.6895, 139.6917, 0.0),
            ("Sydney, Australia", -33.8688, 151.2093, 0.0),
            ("Cape Town, South Africa", -33.9249, 18.4241, 0.0),
            ("Buenos Aires, Argentina", -34.6037, -58.3816, 0.0),
            ("Santiago, Chile", -33.4489, -70.6693, 0.0)
        ]

    for name, lat, lon, alt in tests:
        #print("=== Testing", name)
        check_ground_ecef(name, lat, lon, alt)

    # ---- Test 2: Elevation histogram ----
    print("\n=== Building satellite environment for elevation test...")
    env = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=[8, 32, 64],
    )
    
    print("\n\n==== CALLING print_ground_sat_links ====\n")
    print_ground_sat_links(env.G)

    debug_print_ground_visibility(
        env,
        elev_vis_threshold=0.0,   # 可见全集：仰角 >= 0°
        elev_select_threshold=10.0 # 和 add_sat_ground_edges_with_elev 里的 min_elev_deg 保持一致
    )

    test_all_elevations(env)


