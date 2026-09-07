from astropy.time import Time
from collections import Counter
import numpy as np
import os

from topology_builder import (
    build_satellite_graph,
    haversine_km,
    C_KM_PER_MS,
    V_FIBER_KM_PER_MS,
    R_EARTH_KM,
    has_los_vec,
    compute_elevation,
)
from environment1 import SatelliteGraphEnv


# ===================== 工具函数：路径分类 & 时延 / 可行性 =====================

def classify_path_for_test(env, path_idx):
    """
    分类一条路径:
      - pure_sat: 只经过 ISL / SAT-GND
      - pure_ground: 只经过 GROUND
      - hybrid: 同时有卫星和地面边

    注意：这里的 path_idx 是“节点 index 列表”，需要通过 env.index_node 映射回名字。
    """
    has_sat = False
    has_ground = False

    for u_idx, v_idx in zip(path_idx[:-1], path_idx[1:]):
        u_name = env.index_node[u_idx]
        v_name = env.index_node[v_idx]

        # 这里要用 v_name，不能用 index
        if v_name in env.G[u_name]:
            kind = env.G[u_name][v_name].get("kind", "ISL")
        else:
            # 找不到就当 ISL，防止 KeyError
            kind = "ISL"

        if kind in ("ISL", "SAT-GND"):
            has_sat = True
        elif kind == "GROUND":
            has_ground = True

    if has_sat and has_ground:
        return "hybrid"
    elif has_sat:
        return "pure_sat"
    else:
        return "pure_ground"


def compute_path_delay(env, path_idx):
    """
    根据 env.link_delay 计算一条路径的总时延（ms）。
    path_idx: [node_idx0, node_idx1, ...]
    """
    total_delay = 0.0
    for u_idx, v_idx in zip(path_idx[:-1], path_idx[1:]):
        edge_key = f"{u_idx}:{v_idx}"
        edge_pos = env.edgesDict.get(edge_key)
        if edge_pos is None:
            return None
        total_delay += float(env.link_delay[edge_pos])
    return total_delay


def path_is_feasible(env, state, path_idx, demand):
    """
    检查路径上所有边的剩余容量是否都 >= demand。
    path_idx: [node_idx0, node_idx1, ...]
    state: 当前 graph_state
    """
    if path_idx is None or path_idx == "GROUND":
        return False
    for u_idx, v_idx in zip(path_idx[:-1], path_idx[1:]):
        edge_key = f"{u_idx}:{v_idx}"
        pos = env.edgesDict.get(edge_key)
        if pos is None:
            return False
        if state[pos, 0] < demand - 1e-6:  # 容量不够
            return False
    return True


# ===================== 打印拓扑基本信息 =====================

def print_topology_summary(env):
    G = env.G
    print("\n=== [Level 3] Topology summary (from test) ===")
    print("num_nodes:", len(G.nodes()))
    print("num_edges:", len(G.edges()))
    num_sat_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "sat")
    num_gnd_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "ground")
    num_isl_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "ISL")
    num_sg_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "SAT-GND")
    num_ground_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "GROUND")
    print("num_sat_nodes:", num_sat_nodes)
    print("num_ground_nodes:", num_gnd_nodes)
    print("num_ISL_edges:", num_isl_edges)
    print("num_SAT_GND_edges:", num_sg_edges)
    print("num_GROUND_edges:", num_ground_edges)
    print("ground_fiber_delay_ms (ENS-OSN fiber):", G.graph.get("ground_fiber_delay_ms"))
    print()


# ===================== 打印当前 request 的所有候选路径 =====================

def print_paths_for_request(env, src_idx, dst_idx, state, demand):
    """
    打印当前 request 的所有候选路径（包括 GROUND），以及：
      - 路径节点名字
      - 路径类型 (pure_sat / pure_ground / hybrid / ground)
      - 路径几何时延 (sum(link_delay))
      - 是否带宽可行 (feasible / infeasible)
    """
    key = f"{src_idx}:{dst_idx}"
    paths = env.allPaths.get(key, [])
    src_name = env.index_node[src_idx]
    dst_name = env.index_node[dst_idx]

    print(f"  Candidate paths from {src_name} to {dst_name}:")
    for a, p in enumerate(paths):
        if p == "GROUND":
            ground_path = env._compute_ground_path(src_idx, dst_idx)
            if ground_path is None:
                print(f"    action {a}: GROUND")
                print("      ground path: NOT AVAILABLE")
                continue

            nodes = [env.index_node[i] for i in ground_path]
            delay = compute_path_delay(env, ground_path)
            feasible = path_is_feasible(env, state, ground_path, demand)

            print(f"    action {a}: GROUND")
            print(f"      ground path idx:   {ground_path}")
            print(f"      ground path nodes: {nodes}")
            print(f"      total delay (ms):  {delay:.4f}")
            print(f"      feasible:          {feasible}")
        else:
            nodes = [env.index_node[i] for i in p]
            ptype = classify_path_for_test(env, p)
            delay = compute_path_delay(env, p)
            feasible = path_is_feasible(env, state, p, demand)

            print(f"    action {a}: {p}")
            print(f"      nodes: {nodes}")
            print(f"      type:  {ptype}")
            print(f"      total delay (ms): {delay:.4f}")
            print(f"      feasible:         {feasible}")
    print()


# ===================== 打印本步实际使用到的边 =====================

def print_used_edges(env, old_state, new_state):
    """
    打印在这一步中实际使用了哪些边（根据容量变化判断），包括：
      - 边 (u, v) 名称
      - 使用前容量
      - 使用后容量
      - 本步消耗的带宽
    """
    delta = old_state[:, 0] - new_state[:, 0]
    used_positions = np.where(delta > 1e-6)[0]

    if len(used_positions) == 0:
        print("  No edge capacity was consumed in this step.\n")
        return

    print("  Edges used in this step:")
    for pos in used_positions:
        u_idx, v_idx = env.ordered_edges[pos]
        u_name = env.index_node[u_idx]
        v_name = env.index_node[v_idx]
        cap_before = old_state[pos, 0]
        cap_after = new_state[pos, 0]
        used_bw = delta[pos]
        print(f"    edge {pos}: ({u_name} - {v_name})")
        print(f"      capacity before: {cap_before:.4f} Mbps")
        print(f"      capacity after:  {cap_after:.4f} Mbps")
        print(f"      used in this step: {used_bw:.4f} Mbps")
    print()


# ===================== Level 3：跑一整个 episode 的详细测试 =====================

def run_one_episode(env):
    """
    Level 3 测试：
      - env.reset() 之后，会有多个 requests（env.requests）
      - 每个 request：
          * 打印候选路径信息
          * 选择：延迟最小 + 容量可行 的卫星路径；
            如果没有可行卫星路径，则尝试 GROUND；
            再没有就随便选一个交给 env 判 FAIL。
          * 调 env.make_step(...)
          * 打印 reward、本步使用的边、容量变化
          * 打印这个 request 的 delay 细分
    """
    state, demand, src_idx, dst_idx = env.reset()
    initial_state = state.copy()

    print_topology_summary(env)

    done = False

    while not done:
        print(f"=== Request {env.req_idx} / total {len(env.requests)} ===")
        src_name = env.index_node[src_idx]
        dst_name = env.index_node[dst_idx]
        print(f"  src={src_name}, dst={dst_name}, demand={demand}")

        # 打印所有候选路径
        print_paths_for_request(env, src_idx, dst_idx, state, demand)

        # ---------- 选 action：延迟最小 + 容量可行 ----------
        key = f"{src_idx}:{dst_idx}"
        paths = env.allPaths.get(key, [])
        best_action = None
        best_delay = None

        # 1) 优先在 SAT / hybrid 路径里选：可行 & 延迟最小
        for a, p in enumerate(paths):
            if p == "GROUND":
                continue
            if not path_is_feasible(env, state, p, demand):
                continue
            delay = compute_path_delay(env, p)
            if delay is None:
                continue
            if (best_delay is None) or (delay < best_delay):
                best_delay = delay
                best_action = a

        # 2) 再考虑 GROUND 作为 fallback
        ground_delay = None
        ground_action = None
        ground_path = None
        for a, p in enumerate(paths):
            if p == "GROUND":
                ground_action = a
                ground_path = env._compute_ground_path(src_idx, dst_idx)
                if ground_path is not None:
                    ground_delay = compute_path_delay(env, ground_path)
                break

        # 决策逻辑：
        if best_action is not None:
            chosen_action = best_action
            approx_delay = best_delay
        elif (
            ground_action is not None
            and ground_path is not None
            and path_is_feasible(env, state, ground_path, demand)
        ):
            chosen_action = ground_action
            approx_delay = ground_delay
        else:
            # 都不行：随便选一个（比如 GROUND 或 0），交给 env 去判 FAIL
            if ground_action is not None:
                chosen_action = ground_action
                approx_delay = ground_delay
            else:
                chosen_action = 0
                approx_delay = None

        if approx_delay is not None:
            print(f"  -> chosen action = {chosen_action}, approx delay = {approx_delay:.4f} ms")
        else:
            print(f"  -> chosen action = {chosen_action}, approx delay = N/A")

        # ---------- 执行一步 ----------
        old_state = state.copy()
        new_state, reward, done, new_demand, new_src_idx, new_dst_idx = env.make_step(
            state, chosen_action, demand, src_idx, dst_idx
        )

        print(f"  reward = {reward:.4f}, done = {done}")
        print_used_edges(env, old_state, new_state)

        # 打印本次 request 的 delay 细分（从 env.episode_logs 取最后一条）
        if env.episode_logs:
            rec = env.episode_logs[-1]
            ku = rec.get("ku_ms", 0.0)
            ka = rec.get("ka_ms", 0.0)
            mesh_sat = rec.get("mesh_sat_ms", 0.0)
            mesh_ground = rec.get("mesh_ground_ms", 0.0)
            congestion = rec.get("congestion_ms", 0.0)
            e2e = rec.get("e2e_delay_ms", 0.0)
            wet_src = rec.get("wet_src", 0.0)
            wet_dst = rec.get("wet_dst", 0.0)

            print("  --- Delay breakdown from env.episode_logs ---")
            print(
                f"  ku_ms          = {ku:.3f} ms, "
                f"ka_ms          = {ka:.3f} ms"
            )
            print(f"  gw_to_city_total_ms = {rec.get('gw_to_city_total_ms', 0.0):.3f} ms")

            # 如果这一条路径中有多次 Ka 落地，就把每个城市的贡献也列出来
            hops = rec.get("gw_to_city_hops", [])
            if hops:
                hop_str = ", ".join(
                    f"{h['city']}(+{h['gw_ms']:.3f} ms)" for h in hops
                )
                print(f"  gw_to_city_hops    = {hop_str}")
            print(
                f"wet_ping_src_ms = {wet_src:.3f} ms, "
                f"wet_ping_dst_ms = {wet_dst:.3f} ms"
            )
            print(
                f"  mesh_sat_ms    = {mesh_sat:.3f} ms, "
                f"mesh_ground_ms = {mesh_ground:.3f} ms"
            )
            print(
                f"  congestion_ms  = {congestion:.3f} ms"
            )
            print(f"  e2e_delay_ms   = {e2e:.3f} ms")
            print("  ---------------------------------------------\n")

        # 更新状态，准备处理下一个 request
        state = new_state
        demand = new_demand
        src_idx = new_src_idx
        dst_idx = new_dst_idx

    # episode 结束后，打印最后一个 request 的物理分解（mesh_sat / mesh_ground / ku / ka / congestion）
    if env.episode_logs:
        last_rec = env.episode_logs[-1]
        hops = rec.get("gw_to_city_hops", [])
        if hops:
            hop_str = ", ".join(f"{h['city']}(+{h['gw_ms']:.3f} ms)" for h in hops)
            print(f"  gw_to_city_hops: {hop_str}")
        print(
            "  delay_decompose: "
            f"ku={last_rec.get('ku_ms', 0.0):.3f}, "
            f"ka={last_rec.get('ka_ms', 0.0):.3f}, "
            f"gw_to_city_total={rec.get('gw_to_city_total_ms', 0.0):.3f}, "

            f"mesh_sat={last_rec.get('mesh_sat_ms', 0.0):.3f}, "
            f"mesh_ground={last_rec.get('mesh_ground_ms', 0.0):.3f}, "
            f"congestion={last_rec.get('congestion_ms', 0.0):.3f}, "
            f"e2e={last_rec.get('e2e_delay_ms', 0.0):.3f}"
            )

    # 整个 episode 结束后，可以统计总体平均利用率
    total_initial_cap = np.sum(initial_state[:, 0])
    total_final_cap = np.sum(state[:, 0])
    total_used = total_initial_cap - total_final_cap
    avg_util = total_used / (total_initial_cap + 1e-9)
    print("=== Episode summary (from run_one_episode) ===")
    print(f"  total capacity used: {total_used:.4f} Mbps")
    print(f"  average utilization over all edges: {avg_util:.6f}")
    print()


# ===================== Level 1：WETLINK → TOPOLOGY =====================

def test_level1_wetlink_in_topology():
    """
    Level 1：
      - 检查 WETLINK profile 是否正确注入到了拓扑的 ground 节点 / SAT-GND 边。
      - 统一打印 GS_ENS / GS_OSN / Paris 的 wet_xxx 属性。
    """
    tle_path = "starlink_latest.tle"
    t_str = "2025-11-28 03:00:00"

    wetlink_gs_map = {
        "GS_ENS": {
            "csv_path": "analysis_data_Enschede.csv",
            "site_name": "utwente",
        },
        "GS_OSN": {
            "csv_path": "analysis_data_Osnabrück.csv",
            "site_name": None,
        },
    }

    print("\n========== [Level 1] WETLINK → TOPOLOGY bridge ==========")

    G = build_satellite_graph(
        tle_path=tle_path,
        t_obs=t_str,
        max_sats=66,
        isl_k=4,
        isl_max_dist_km=2500.0,
        sg_max_dist_km=2000.0,
        wetlink_gs_map=wetlink_gs_map,
    )

    print("num_nodes:", G.number_of_nodes())
    print("num_edges:", G.number_of_edges())

    # 1) ENS / OSN / Paris 的 wetlink 属性（统一 6 行）
    for name in ["GS_ENS", "GS_OSN", "Paris, France"]:
        if name not in G.nodes:
            print(f"\n[WARN] {name} not in graph nodes.")
            continue
        nd = G.nodes[name]
        print(f"\nNode {name}:")
        print("  wet_ping_ms        =", nd.get("wet_ping_ms"))
        print("  wet_download_Mbps  =", nd.get("wet_download_Mbps"))
        print("  wet_temp           =", nd.get("wet_temp"))
        print("  wet_humidity       =", nd.get("wet_humidity"))
        print("  wet_rain           =", nd.get("wet_rain"))
        print("  wet_windspeed      =", nd.get("wet_windspeed"))

    # 2) 检查 SAT-GND 边的一致性（以 GS_ENS / GS_OSN 为例）
    for gs_name in ["GS_ENS", "GS_OSN"]:
        edges = []
        for u, v, d in G.edges(data=True):
            if d.get("kind") == "SAT-GND" and (u == gs_name or v == gs_name):
                edges.append((u, v, d))

        print(f"\n=== SAT-GND edges for {gs_name} ===")
        if not edges:
            print("  [WARN] no SAT-GND edges found for this ground node.")
            continue

        caps = [e[2].get("base_capacity_Mbps", 0.0) for e in edges]
        rains = [e[2].get("weather_rain", 0.0) for e in edges]
        pings = [e[2].get("wet_ping_ms", 0.0) for e in edges]

        for (u, v, d) in edges[:3]:  # 只打印前三条，避免刷屏
            sat = u if u != gs_name else v
            print(f"  edge {gs_name} -- {sat}:")
            print(f"    base_capacity_Mbps = {d.get('base_capacity_Mbps')}")
            print(f"    weather_rain       = {d.get('weather_rain')}")
            print(f"    wet_ping_ms        = {d.get('wet_ping_ms', 0.0)}")
            print(f"    base_delay_ms(mesh)= {d.get('base_delay_ms', 0.0)}")

        def check_range(name, values, tol=1e-3):
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                return
            diff = float(arr.max() - arr.min())
            print(f"  >> {name}: min={arr.min():.3f}, max={arr.max():.3f}, diff={diff:.6f}")

        print("  --- consistency summary ---")
        check_range("capacity_Mbps", caps)
        check_range("rain", rains)
        check_range("wet_ping_ms_on_edge", pings)

    print("========== [Level 1] DONE ==========\n")


# ===================== Level 2：SAT-GND LOS / elevation 可见性 =====================

def test_level2_sat_ground_visibility():
    """
    Level 2：
      - 在固定时间 t_obs 下，对所有 ground 节点做 LOS + elevation 检查。
      - elevation >= 0° 的都打印出来，并标出哪些真的被选进拓扑（存在 SAT-GND 边）。
    """
    tle_path = "starlink_latest.tle"
    t_obs = Time("2025-11-28 03:00:00", scale="utc")

    wetlink_gs_map = {
        "GS_ENS": {
            "csv_path": "analysis_data_Enschede.csv",
            "site_name": "utwente",
        },
        "GS_OSN": {
            "csv_path": "analysis_data_Osnabrück.csv",
            "site_name": None,
        },
    }

    # 和 env 一致的 ground_stations 配置可以从 environment1.SatelliteGraphEnv 的构造来看，
    # 这里只要保证城市名字相同即可；为了简单，直接让 build_satellite_graph 自己用默认的 city 集合。
    ground_stations = None

    print("\n========== [Level 2] SAT-GND visibility / elevation ==========")
    G = build_satellite_graph(
        tle_path=tle_path,
        t_obs=t_obs,
        max_sats=66,
        isl_k=4,
        isl_max_dist_km=5000.0,
        sg_max_dist_km=2000.0,
        wetlink_gs_map=wetlink_gs_map,
        ground_stations=ground_stations,
    )

    print("=== Graph basic info =\==")
    print("num_nodes:", len(G.nodes()))
    print("num_edges:", len(G.edges()))
    num_sat_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "sat")
    num_gnd_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "ground")
    num_sg_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "SAT-GND")
    print("num_sat_nodes:", num_sat_nodes)
    print("num_ground_nodes:", num_gnd_nodes)
    print("num_SAT_GND_edges:", num_sg_edges)
    print("ground_fiber_delay_ms (ENS-OSN fiber):", G.graph.get("ground_fiber_delay_ms"))

    sat_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sat"]
    gnd_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "ground"]

    print("\n================ Ground visibility debug ================")
    print(f"Total ground nodes: {len(gnd_nodes)}, sat nodes: {len(sat_nodes)}")
    print("Visible if elev >= 0.0°, selectable if elev >= 10.0°")
    print("=========================================================\n")

    for g in gnd_nodes:
        rg = np.asarray(G.nodes[g]["pos_xyz"], dtype=float)
        candidates = []  # (sat_name, elev, dist_km, in_graph)

        for s in sat_nodes:
            rs = np.asarray(G.nodes[s]["pos_xyz"], dtype=float)

            # LOS 检查
            if not has_los_vec(rg, rs, R_earth_km=R_EARTH_KM, margin_km=0.0):
                continue

            dist_km = float(np.linalg.norm(rs - rg))
            elev = compute_elevation(rg, rs)

            # 是否在图里真正有 SAT-GND 边
            in_graph = G.has_edge(g, s) and G[g][s].get("kind") == "SAT-GND"
            candidates.append((s, elev, dist_km, in_graph))

        print(f"--- Ground station: {g} ---")
        visible = [c for c in candidates if c[1] >= 0.0]
        if not visible:
            print("  No visible satellites (LOS & elev >= 0.0°).\n")
            continue

        # 按 elevation 从大到小排
        visible.sort(key=lambda x: -x[1])
        print(f"  Visible satellites (LOS & elev >= 0.0°): {len(visible)}")
        for s, elev, dist_km, in_graph in visible:
            mark = "<-- SELECTED in topology" if in_graph and elev >= 10.0 else ""
            print(
                f"   - {s:8s} elev = {elev:6.2f}°  dist = {dist_km:7.1f} km  {mark}"
            )
        print()

    print("========== [Level 2] DONE ==========\n")


# ===================== main：依次跑 Level 1 / Level 2 / Level 3 =====================

def main():
    # Level 1：WETLINK 注入检查
    test_level1_wetlink_in_topology()

    # Level 2：SAT-GND LOS / elevation 调试
    test_level2_sat_ground_visibility()

    # Level 3：环境联调 + 一整个 episode 的详细测试
    listofDemands = [8, 32, 64]
    env = SatelliteGraphEnv(
        tle_path="starlink_latest.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=listofDemands,
        K_paths=4,
    )

    print("\n========== [Level 3] Run one detailed episode ==========")
    run_one_episode(env)
    print("========== [Level 3] DONE ==========\n")


if __name__ == "__main__":
    main()
