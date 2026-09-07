import json
from environment1 import SatelliteGraphEnv
from train_DQN import DQNAgent, hparams, listofDemands, checkpoint_dir, SEED
import numpy as np
from astropy.time import Time

# =========================
# Weather toggle (future use)
# =========================
ENABLE_WEATHER = False  # 现在关；将来测试雨衰直接改 True

# =========================
# Time-sweep settings (NEW)
# =========================
START_EPOCH_STR = "2025-12-05 06:00:00"   # 起始时间
DURATION_SEC = 3600                       # 一小时
STEP_SEC = 60                             # 每隔多少秒采样一次（建议 60）
# 如果想更细：STEP_SEC=10 (360个点)；更粗：STEP_SEC=300 (12个点)


def load_jsonl(path):
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reqs.append(json.loads(line))
    return reqs


def build_name_alias(env):
    def norm(s: str):
        return "".join(str(s).lower().split())

    alias = {}
    for name in env.G.nodes:
        alias[norm(name)] = name
    return alias, norm


def normalize_requests(reqs, env):
    alias, norm = build_name_alias(env)
    fixed, bad = [], []
    for i, r in enumerate(reqs):
        src_raw = r["src_name"]
        dst_raw = r["dst_name"]
        src = alias.get(norm(src_raw), None)
        dst = alias.get(norm(dst_raw), None)
        if src is None or dst is None:
            bad.append((i, src_raw, dst_raw))
            continue
        fixed.append(
            {
                "src_name": src,
                "dst_name": dst,
                "demand": float(r["demand"]),
                "ground_delay_ms": float(r.get("ground_delay_ms", 0.0)),
            }
        )
    return fixed, bad


def summarize_60req_in_3_groups(env, state_snaps):
    logs = getattr(env, "episode_logs", []) or []
    groups = [(1, 0, 19, 0, 20), (2, 20, 39, 20, 40), (3, 40, 59, 40, 60)]
    results = []

    satgnd_mask = np.zeros(len(env.ordered_edges), dtype=bool)
    for eidx, (i, j) in enumerate(env.ordered_edges):
        u = env.index_node[i]
        v = env.index_node[j]
        kind = str(env.G[u][v].get("kind", "")).upper()
        if ("SAT" in kind) and ("GND" in kind):
            satgnd_mask[eidx] = True

    den = getattr(env, "eff_cap_up", getattr(env, "base_cap_up", None))
    if den is None:
        raise RuntimeError("[FATAL] env missing eff_cap_up/base_cap_up for UL capacity")

    den = den.astype(float)
    den_safe = np.where(den > 0.0, den, 1.0)

    def util_summary_between(st_before, st_after):
        before_down = st_before[:, 0].astype(float)
        before_up   = st_before[:, 1].astype(float)
        after_down  = st_after[:, 0].astype(float)
        after_up    = st_after[:, 1].astype(float)

        used_down = np.clip(before_down - after_down, 0.0, None)
        used_up   = np.clip(before_up   - after_up,   0.0, None)

        util_down = used_down / den_safe
        util_up   = used_up   / den_safe

        util_down = np.clip(np.nan_to_num(util_down, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        util_up   = np.clip(np.nan_to_num(util_up,   nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # overall mean (两条独立方向池，但容量数值相同)
        util_any = (used_down + used_up) / (2.0 * den_safe)
        util_any = np.clip(np.nan_to_num(util_any, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        used_any_mask = (used_down > 0.0) | (used_up > 0.0)

        ul_mask = (used_up > 0.0) & satgnd_mask
        dl_mask = (used_down > 0.0) & satgnd_mask

        def safe_mean(arr, mask):
            return float(arr[mask].mean()) if np.any(mask) else 0.0

        return {
            "avg_util_used_edges": safe_mean(util_any, used_any_mask),
            "avg_util_upload_used_edges": safe_mean(util_up, ul_mask),
            "avg_util_download_used_edges": safe_mean(util_down, dl_mask),
            "n_used_edges": int(np.sum(used_any_mask)),
            "n_ul_edges": int(np.sum(ul_mask)),
            "n_dl_edges": int(np.sum(dl_mask)),
        }

    for gid, lo, hi, s0, s1 in groups:
        if s0 not in state_snaps or s1 not in state_snaps:
            raise RuntimeError(f"[FATAL] missing state snapshots: need {s0} and {s1}, got {sorted(state_snaps.keys())}")

        group_logs = [rec for rec in logs if (lo <= rec.get("req_id", -1) <= hi)]
        success_logs = [rec for rec in group_logs if rec.get("success")]

        delays = [float(rec.get("e2e_delay_ms", 0.0)) for rec in success_logs]
        mean_delay = (sum(delays) / len(delays)) if delays else 0.0

        util = util_summary_between(state_snaps[s0], state_snaps[s1])

        results.append({
            "group": gid,
            "req_range": (lo, hi),
            "mean_e2e_delay_ms": mean_delay,
            **util,
            "n_success": int(len(success_logs)),
            "n_total": int(hi - lo + 1),
        })

    util_total = util_summary_between(state_snaps[0], state_snaps[60])
    total_success = sum(r["n_success"] for r in results)
    total_count = 60
    total_mean_delay = (
        sum(r["mean_e2e_delay_ms"] * r["n_success"] for r in results) / total_success
        if total_success else 0.0
    )

    results.append({
        "group": "TOTAL",
        "req_range": (0, 59),
        "mean_e2e_delay_ms": total_mean_delay,
        **util_total,
        "n_success": int(total_success),
        "n_total": int(total_count),
    })

    return results


def _edge_util_after_two_dirs_equal_cap(env, st_after, overall_mode="mean"):
    den = getattr(env, "eff_cap_up", getattr(env, "base_cap_up", None))
    if den is None:
        raise RuntimeError("[FATAL] env missing eff_cap_up/base_cap_up for UL capacity")

    den = np.asarray(den, dtype=float)
    if den.ndim == 0:
        den = np.full(st_after.shape[0], float(den), dtype=float)

    den_safe = np.where(den > 0.0, den, 1.0)

    down_rem = st_after[:, 0].astype(float)
    up_rem   = st_after[:, 1].astype(float)

    used_down = np.clip(den - down_rem, 0.0, None)
    used_up   = np.clip(den - up_rem,   0.0, None)

    util_down = np.clip(np.nan_to_num(used_down / den_safe, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    util_up   = np.clip(np.nan_to_num(used_up   / den_safe, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if overall_mode == "max":
        util_any = np.maximum(util_up, util_down)
    else:
        util_any = 0.5 * (util_up + util_down)

    return util_any, util_up, util_down


def _infer_edges_from_state_delta(env, st_before, st_after, eps=1e-9):
    used_down = np.clip(st_before[:, 0].astype(float) - st_after[:, 0].astype(float), 0.0, None)
    used_up   = np.clip(st_before[:, 1].astype(float) - st_after[:, 1].astype(float), 0.0, None)

    used_mask = (used_down > eps) | (used_up > eps)
    idxs = np.where(used_mask)[0]

    util_any_all, util_up_all, util_down_all = _edge_util_after_two_dirs_equal_cap(
        env, st_after, overall_mode="mean"
    )

    edges = []
    for eidx in idxs.tolist():
        i, j = env.ordered_edges[eidx]
        u = env.index_node[i]
        v = env.index_node[j]
        edges.append({
            "u": u,
            "v": v,
            "edge_idx": int(eidx),
            "used_down": float(used_down[eidx]),
            "used_up": float(used_up[eidx]),
            "util_after": float(util_any_all[eidx]),
            "util_up_after": float(util_up_all[eidx]),
            "util_down_after": float(util_down_all[eidx]),
        })

    edges.sort(key=lambda x: x["edge_idx"])
    return edges


def _get_latest_rec(env, step: int):
    logs = getattr(env, "episode_logs", []) or []
    if not logs:
        return None
    for rec in reversed(logs):
        if rec.get("req_id") == step:
            return rec
    for rec in reversed(logs):
        if rec.get("step") == step:
            return rec
    return logs[-1]


def _try_set_weather(env, enable: bool):
    for attr in ["enable_weather", "weather", "weather_enabled", "use_weather"]:
        if hasattr(env, attr):
            try:
                setattr(env, attr, bool(enable))
            except Exception:
                pass


def _build_env_for_epoch(epoch_time: Time) -> SatelliteGraphEnv:
    """NEW: 每个时间点创建一个新 env，保证拓扑刷新"""
    env_eval = SatelliteGraphEnv(
        tle_path="iridium1209_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=listofDemands,
        K_paths=20,
        requests_per_episode=60,
        shuffle_requests=False,
        epoch=epoch_time,
    )
    _try_set_weather(env_eval, ENABLE_WEATHER)
    env_eval.seed(SEED)
    env_eval.generate_environment(0, listofDemands)
    if getattr(env_eval, "G", None) is None:
        env_eval._build_new_graph()
    return env_eval


def _percentile(xs, q):
    if not xs:
        return 0.0
    a = np.asarray(xs, dtype=float)
    return float(np.percentile(a, q))


def run_one_timepoint(epoch_time: Time, fixed_requests, out_prefix: str):
    """
    NEW: 跑一个时间点（一个 episode=60 requests）
    返回: summary(dict)
    同时写出逐 step log: f"{out_prefix}_details.jsonl"
    """
    env_eval = _build_env_for_epoch(epoch_time)

    env_eval.use_fixed_requests = True
    env_eval.fixed_requests = fixed_requests
    env_eval.fixed_shuffle = False
    env_eval.requests_per_episode = 60

    agent = DQNAgent(env_eval, hparams["batch_size"])
    agent.load_model(checkpoint_dir)

    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    if hasattr(agent, "eps"):
        agent.eps = 0.0
    for net_name in ["policy_net", "qnetwork", "q_net", "model", "net"]:
        net = getattr(agent, net_name, None)
        if net is not None and hasattr(net, "eval"):
            net.eval()

    state, demand, source, destination = env_eval.reset()
    done = False
    total_r = 0.0
    step = 0
    state_snaps = {0: np.copy(state)}

    detail_path = f"{out_prefix}_details.jsonl"
    with open(detail_path, "w", encoding="utf-8") as fout:
        while not done:
            action, approx_delay = agent.select_action(env_eval, state, demand, source, destination)
            new_state, reward, done, new_demand, new_source, new_destination = env_eval.make_step(
                state, action, demand, source, destination
            )

            edges_inferred = _infer_edges_from_state_delta(env_eval, state, new_state)

            if (step + 1) in (20, 40, 60):
                state_snaps[step + 1] = np.copy(new_state)

            rec = _get_latest_rec(env_eval, step)
            fout.write(json.dumps({
                "epoch_utc": epoch_time.isot,
                "step": step,
                "src_idx": int(source),
                "dst_idx": int(destination),
                "src_name": env_eval.index_node[int(source)],
                "dst_name": env_eval.index_node[int(destination)],
                "demand": float(demand),
                "action": int(action),
                "approx_delay_ms": None if approx_delay is None else float(approx_delay),
                "reward": float(reward),
                "done": bool(done),
                "enable_weather": bool(ENABLE_WEATHER),

                "success": None if rec is None else bool(rec.get("success")),
                "ground_delay_ms": None if rec is None else (None if rec.get("ground_delay_ms") is None else float(rec.get("ground_delay_ms"))),
                "e2e_delay_ms": None if rec is None else (None if rec.get("e2e_delay_ms") is None else float(rec.get("e2e_delay_ms"))),
                "ratio": None if rec is None else (None if rec.get("ratio") is None else float(rec.get("ratio"))),
                "improve_ms": None if rec is None else (None if rec.get("improve_ms") is None else float(rec.get("improve_ms"))),
                "hops": None if rec is None else rec.get("hops"),
                "path_nodes": None if rec is None else rec.get("path_nodes"),
                "edges": edges_inferred,
            }, ensure_ascii=False) + "\n")

            state = new_state
            demand, source, destination = new_demand, new_source, new_destination
            total_r += reward
            step += 1

    # 组内统计（你的原逻辑）
    groups = summarize_60req_in_3_groups(env_eval, state_snaps)

    # 额外：该时间点整体 mean / p95（更适合做时间序列）
    logs = getattr(env_eval, "episode_logs", []) or []
    e2es = [float(r.get("e2e_delay_ms")) for r in logs if r.get("success") and r.get("e2e_delay_ms") is not None]
    summary = {
        "epoch_utc": epoch_time.isot,
        "n_success": int(sum(1 for r in logs if r.get("success"))),
        "n_total": int(len(logs)) if logs else 60,
        "mean_e2e_delay_ms": float(np.mean(e2es)) if e2es else 0.0,
        "p95_e2e_delay_ms": _percentile(e2es, 95),
        "total_reward": float(total_r),
        "groups": groups,  # 保留你原来的三组输出
        "detail_file": detail_path,
    }
    return summary


def main():
    files = ["./fixed_reqs1.jsonl", "./fixed_reqs2.jsonl", "./fixed_reqs3.jsonl"]

    all_reqs = []
    for p in files:
        all_reqs.extend(load_jsonl(p))
    if len(all_reqs) != 60:
        raise SystemExit(f"[ERROR] 3个文件合起来应为60条，但现在是 {len(all_reqs)} 条。")

    # 先用起始 epoch 建一个 env 用于 normalize（只做一次）
    start_epoch = Time(START_EPOCH_STR, scale="utc")
    env_for_norm = _build_env_for_epoch(start_epoch)

    fixed, bad = normalize_requests(all_reqs, env_for_norm)
    if bad:
        print("[ERROR] node name not found for these lines:")
        for i, s, d in bad[:20]:
            print(f"  line {i}: {s!r} -> {d!r}")
        raise SystemExit(1)
    if len(fixed) != 60:
        raise SystemExit(f"[ERROR] normalize 后可用请求不是60条，而是 {len(fixed)} 条。")

    # 时间序列总结果（NEW）
    sweep_out = "time_sweep_summary.jsonl"
    with open(sweep_out, "w", encoding="utf-8") as fsum:
        n_points = int(DURATION_SEC // STEP_SEC) + 1
        for k in range(n_points):
            t = start_epoch + (k * STEP_SEC) * 1.0  # astropy 支持秒偏移
            # astropy 的 Time 加秒：用 TimeDelta 更严谨，但这里最小改动
            t = Time(start_epoch.unix + k * STEP_SEC, format="unix", scale="utc")

            tag = t.isot.replace(":", "").replace("-", "")
            out_prefix = f"test_results_{tag}"

            summary = run_one_timepoint(t, fixed, out_prefix)
            fsum.write(json.dumps(summary, ensure_ascii=False) + "\n")

            # 控制台打印一行便于观察
            print(
                f"[{summary['epoch_utc']}] "
                f"success={summary['n_success']}/{summary['n_total']} "
                f"mean={summary['mean_e2e_delay_ms']:.3f}ms "
                f"p95={summary['p95_e2e_delay_ms']:.3f}ms "
                f"reward={summary['total_reward']:.3f}"
            )

    print("saved ->", sweep_out)


if __name__ == "__main__":
    main()
