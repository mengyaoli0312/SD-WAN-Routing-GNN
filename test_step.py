import numpy as np
import random

from environment1 import SatelliteGraphEnv  # ← 你的环境文件名


def run_multi_episode_test(env, episodes=5):
    """
    多 episode 随机策略小测试：
    - 每个 episode 用随机 action 跑完整个请求序列；
    - 统计：
        * SAT / GROUND 边的平均利用率
        * 总 uplink / downlink 消耗
        * 各类路径类型 (pure_sat / hybrid / ground / unknown) 的数量 & 平均 e2e_delay
    """
    total_sat_util = 0.0
    total_ground_util = 0.0
    total_up_used = 0.0
    total_down_used = 0.0

    # 全局路径统计（跨 episode）
    global_path_counts = {"pure_sat": 0, "hybrid": 0, "ground": 0, "unknown": 0}
    global_path_delay_sum = {"pure_sat": 0.0, "hybrid": 0.0, "ground": 0.0, "unknown": 0.0}

    for ep in range(episodes):
        print(f"\n==================== Episode {ep} ====================")
        state, demand, src, dst = env.reset()

        done = False
        step = 0
        while not done:
            # 当前 request 的 key
            key = f"{env.current_src}:{env.current_dst}"
            path_list = env.allPaths.get(key, [])
            if not path_list:
                action = 0
            else:
                # 随机选一个 action（0 ~ num_actions-1）
                action = random.randint(0, len(path_list) - 1)

            next_state, reward, done, new_demand, new_src, new_dst = env.make_step(
                state, action, demand, src, dst
            )

            state = next_state
            demand = new_demand
            src = new_src
            dst = new_dst
            step += 1

        # ====== 1) Episode 结束后：统计边利用率 & up/down 使用量 ======
        base_cap_down = env.base_cap_down
        base_cap_up = env.base_cap_up

        rem_down = state[:, 0].astype(float)
        rem_up = state[:, 1].astype(float)

        used_down = base_cap_down - rem_down
        used_up = base_cap_up - rem_up

        used_down = np.nan_to_num(used_down, nan=0.0, posinf=0.0, neginf=0.0)
        used_up = np.nan_to_num(used_up, nan=0.0, posinf=0.0, neginf=0.0)

        used_down = np.clip(used_down, 0.0, None)
        used_up = np.clip(used_up, 0.0, None)

        # 区分 SAT / GROUND 边
        sat_edge_indices = []
        ground_edge_indices = []

        for eidx, (i, j) in enumerate(env.ordered_edges):
            u_name = env.index_node[i]
            v_name = env.index_node[j]
            kind = env.G[u_name][v_name].get("kind", "ISL")
            if kind in ("ISL", "SAT-GND"):
                sat_edge_indices.append(eidx)
            elif kind == "GROUND":
                ground_edge_indices.append(eidx)

        sat_edge_indices = np.array(sat_edge_indices, dtype=int) if sat_edge_indices else np.array([], dtype=int)
        ground_edge_indices = np.array(ground_edge_indices, dtype=int) if ground_edge_indices else np.array([], dtype=int)

        def safe_mean_util(used, base_cap):
            if used.size == 0 or base_cap.size == 0:
                return 0.0
            denom = np.where(base_cap > 0.0, base_cap, 1.0)
            util = used / denom
            util = np.clip(util, 0.0, 1.0)
            return float(util.mean())

        sat_util = safe_mean_util(used_down[sat_edge_indices], base_cap_down[sat_edge_indices])
        ground_util = safe_mean_util(used_down[ground_edge_indices], base_cap_down[ground_edge_indices])

        print(f"[Episode {ep}] SAT利用率={sat_util:.6f},  GROUND利用率={ground_util:.6f}")
        print(f"[Episode {ep}] Up使用={used_up.sum():.2f}, Down使用={used_down.sum():.2f}")

        total_sat_util += sat_util
        total_ground_util += ground_util
        total_up_used += float(used_up.sum())
        total_down_used += float(used_down.sum())

        # ====== 2) Episode 内部：按 path_type 统计 数量 & 平均 e2e_delay ======
        ep_counts = {"pure_sat": 0, "hybrid": 0, "ground": 0, "unknown": 0}
        ep_delay_sum = {"pure_sat": 0.0, "hybrid": 0.0, "ground": 0.0, "unknown": 0.0}

        for rec in env.episode_logs:
            pt = rec.get("path_type", "unknown")
            if pt not in ep_counts:
                pt = "unknown"
            ep_counts[pt] += 1
            ep_delay_sum[pt] += float(rec.get("e2e_delay_ms", 0.0))

        print(f"[Episode {ep}] 路径类型统计：")
        for pt in ["pure_sat", "hybrid", "ground", "unknown"]:
            cnt = ep_counts[pt]
            if cnt > 0:
                avg_delay = ep_delay_sum[pt] / cnt
                print(f"  - {pt:10s}: 数量={cnt:2d}, 平均e2e_delay={avg_delay:.3f} ms")
            else:
                print(f"  - {pt:10s}: 数量=0")

            # 汇总到全局统计
            global_path_counts[pt] += cnt
            global_path_delay_sum[pt] += ep_delay_sum[pt]

    # ====== 3) 全局 summary ======
    print("\n########## FINAL SUMMARY ##########")
    avg_sat_util = total_sat_util / episodes if episodes > 0 else 0.0
    avg_ground_util = total_ground_util / episodes if episodes > 0 else 0.0

    print(f"平均 SAT 利用率：{avg_sat_util:.6f}")
    print(f"平均 GROUND 利用率：{avg_ground_util:.6f}")
    print("---------------------------------------")
    print(f"总 uplink 消耗：{total_up_used:.2f}")
    print(f"总 downlink 消耗：{total_down_used:.2f}")

    if total_down_used > 0.0:
        ratio = total_up_used / total_down_used
    else:
        ratio = 0.0

    print("---------------------------------------")
    print(f"uplink : downlink 比例 = {total_up_used:.2f} : {total_down_used:.2f}")
    print(f"uplink/downlink = {ratio:.3f}")
    print("---------------------------------------")

    print("全局路径类型统计：")
    for pt in ["pure_sat", "hybrid", "ground", "unknown"]:
        cnt = global_path_counts[pt]
        if cnt > 0:
            avg_delay = global_path_delay_sum[pt] / cnt
            print(f"  - {pt:10s}: 总数={cnt:3d}, 平均e2e_delay={avg_delay:.3f} ms")
        else:
            print(f"  - {pt:10s}: 总数=0")

    print("#####################################")


if __name__ == "__main__":
    # 根据你的实际文件名/路径调整
    TLE_PATH = "iridium_66_main.tle"
    EN_CSV="analysis_data_Enschede.csv"
    OSN_CSV="analysis_data_Osnabrück.csv"

    listofDemands = [1, 2, 5, 10, 20, 50]

    env = SatelliteGraphEnv(
        tle_path=TLE_PATH,
        en_csv=EN_CSV,
        osn_csv=OSN_CSV,
        listofDemands=listofDemands,
        K_paths=4,
        requests_per_episode=10,
        shuffle_requests=True,
    )

    run_multi_episode_test(env, episodes=5)
