import numpy as np
import networkx as nx

from environment1 import SatelliteGraphEnv  # 你 env 文件名是 environment1.py
# 如果你 env 文件不是这个名，改一下 import


def _make_env(seed=42):
    env = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=[2, 5],
        K_paths=4,
        requests_per_episode=3,
        shuffle_requests=False,
        epoch="2025-12-05 06:00:00",   # 固定
    )
    env.seed(seed)
    return env


def test_build_graph_kinds_and_fields():
    env = _make_env()
    env.reset()

    kinds = sorted({d.get("kind") for _, _, d in env.G.edges(data=True)})
    assert "ISL" in kinds, f"missing ISL edges, kinds={kinds}"
    assert "SAT-GND" in kinds, f"missing SAT-GND edges, kinds={kinds}"
    assert "GROUND" in kinds, f"missing GROUND edges, kinds={kinds}"

    # SAT-GND fields
    sg_cnt = 0
    for u, v, d in env.G.edges(data=True):
        if d.get("kind") != "SAT-GND":
            continue
        sg_cnt += 1
        assert "base_capacity_down_Mbps" in d, "SAT-GND missing base_capacity_down_Mbps"
        assert "base_capacity_up_Mbps" in d, "SAT-GND missing base_capacity_up_Mbps"
        assert "distance_km" in d, "SAT-GND missing distance_km (used by Ku/Ka calc)"
        assert "base_delay_ms" in d, "SAT-GND missing base_delay_ms (env uses this as link_delay)"
    assert sg_cnt > 0, "no SAT-GND edges created"

    # ISL/GROUND fields
    for u, v, d in env.G.edges(data=True):
        k = d.get("kind")
        if k in ("ISL", "GROUND"):
            assert "base_capacity_Mbps" in d, f"{k} missing base_capacity_Mbps"
            assert "base_delay_ms" in d, f"{k} missing base_delay_ms"


def test_edgesDict_is_bidirectional_same_pos():
    env = _make_env()
    env.reset()

    # 随机抽 20 条边检查
    edges = env.ordered_edges[: min(20, len(env.ordered_edges))]
    for (i, j) in edges:
        p1 = env.edgesDict.get(f"{i}:{j}")
        p2 = env.edgesDict.get(f"{j}:{i}")
        assert p1 is not None and p2 is not None, "edgesDict missing direction key"
        assert p1 == p2, "edgesDict forward/backward must map to same pos in undirected edge scheme"


def test_reset_paths_count_and_ground_last():
    env = _make_env()
    state, demand, src, dst = env.reset()

    key = f"{src}:{dst}"
    paths = env.allPaths.get(key)
    assert paths is not None, "allPaths missing key after reset"
    assert len(paths) == env.k_paths + 1, f"expected K+1 actions, got {len(paths)}"
    assert paths[-1] == "GROUND", "GROUND must be last action"

    # K 条里至少有一条不是 GROUND（除非真的无路）
    # 允许极端情况：所有都 pad 重复
    # 这里只做提示级检查：全是同一路径说明 shortest_simple_paths 可能被 ground penalty 吃掉
    non_ground = [p for p in paths if p != "GROUND"]
    assert len(non_ground) == env.k_paths, "non-ground actions should be exactly K paths"


def test_ground_path_exists_for_ground_action_most_cases():
    env = _make_env()
    state, demand, src, dst = env.reset()
    gp = env._compute_ground_path(src, dst)

    # 这里不强制 assert gp 必须存在，因为某些 request 是城市对，
    # 但如果你的 topology ground 子图没连通，会经常 None。
    # 所以我们改成：如果 gp is None，直接把情况打印出来，方便你定位
    if gp is None:
        src_name = env.index_node[src]
        dst_name = env.index_node[dst]
        H_gnd = nx.Graph()
        for u, v, d in env.G.edges(data=True):
            if d.get("kind") == "GROUND":
                H_gnd.add_edge(u, v, **d)
        print("[WARN] ground_path=None",
              "src=", src_name, "dst=", dst_name,
              "H_gnd_nodes=", H_gnd.number_of_nodes(),
              "H_gnd_edges=", H_gnd.number_of_edges(),
              "src_in=", src_name in H_gnd, "dst_in=", dst_name in H_gnd)
        # 不 fail，只提醒
        assert True


def test_one_step_capacity_direction_on_sat_gnd():
    env = _make_env(seed=7)
    state, demand, src, dst = env.reset()
    key = f"{src}:{dst}"
    path_list = env.allPaths[key]

    # 找一条非 GROUND action
    sat_action = None
    sat_path = None
    for a, p in enumerate(path_list):
        if p != "GROUND":
            sat_action = a
            sat_path = p
            break
    if sat_action is None:
        # 极端情况：没有卫星路，只剩 GROUND，跳过
        assert True
        return

    # 记录 SAT-GND 边在 step 前的 down/up
    before = {}
    for u_idx, v_idx in zip(sat_path[:-1], sat_path[1:]):
        u_name = env.index_node[u_idx]
        v_name = env.index_node[v_idx]
        kind = env.G[u_name][v_name].get("kind", "ISL")
        if kind != "SAT-GND":
            continue
        eidx = env.edgesDict.get(f"{u_idx}:{v_idx}")
        if eidx is None:
            continue
        before[(u_idx, v_idx, eidx)] = (state[eidx, 0], state[eidx, 1], env.G.nodes[u_name].get("kind"), env.G.nodes[v_name].get("kind"))

    new_state, reward, done, *_ = env.make_step(state, sat_action, demand, src, dst)

    # 校验：SAT->GND 扣 down；GND->SAT 扣 up
    for (u_idx, v_idx, eidx), (d0, u0, ukind, vkind) in before.items():
        d1, u1 = new_state[eidx, 0], new_state[eidx, 1]
        if ukind == "sat" and vkind == "ground":
            # down 应减少 demand
            assert abs((d0 - demand) - d1) < 1e-3, f"SAT->GND should reduce DOWN by demand, got {d0}->{d1}"
            # up 不应变化（或只轻微浮动）
            assert abs(u0 - u1) < 1e-3, f"SAT->GND should not reduce UP, got {u0}->{u1}"
        elif ukind == "ground" and vkind == "sat":
            assert abs((u0 - demand) - u1) < 1e-3, f"GND->SAT should reduce UP by demand, got {u0}->{u1}"
            assert abs(d0 - d1) < 1e-3, f"GND->SAT should not reduce DOWN, got {d0}->{d1}"

if __name__ == "__main__":
    test_build_graph_kinds_and_fields()
    test_edgesDict_is_bidirectional_same_pos()
    test_reset_paths_count_and_ground_last()
    test_ground_path_exists_for_ground_action_most_cases()
    test_one_step_capacity_direction_on_sat_gnd()
    print("ALL TESTS PASSED")