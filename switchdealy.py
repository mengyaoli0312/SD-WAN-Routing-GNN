# plot_delta_delay.py
import csv
import numpy as np

# ✅ 必须在 import pyplot 之前设置后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from environment1 import SatelliteGraphEnv


def path_delay_ms(env, path):
    """Compute total propagation delay (ms) for a path."""
    if path is None:
        return None

    if path == "GROUND":
        gp = env._compute_ground_path(env.current_src, env.current_dst)
        if gp is None:
            return float(env.G.graph.get("ground_fiber_delay_ms", 60.0))
        path = gp

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        eidx = env.edgesDict.get(f"{u}:{v}")
        if eidx is None:
            return None
        total += float(env.link_delay[eidx])
    return float(total)


def best_path_at_time(env, src_name, dst_name, demand, K=10):
    """Find the minimum-delay candidate path in current topology."""
    if src_name not in env.node_index or dst_name not in env.node_index:
        return None, None

    s = env.node_index[src_name]
    d = env.node_index[dst_name]

    env.current_src = s
    env.current_dst = d
    env.current_demand = float(demand)

    paths = env._compute_k_paths(s, d, int(K))
    paths.append("GROUND")

    best_p, best_delay = None, None
    for p in paths:
        delay = path_delay_ms(env, p)
        if delay is None:
            continue
        if best_delay is None or delay < best_delay:
            best_delay = delay
            best_p = p

    return best_p, best_delay


def run_switch_delta_delay(
    env,
    src_name,
    dst_name,
    demand,
    start_time_utc="2025-12-05 06:00:00",
    dt_sec=10,
    num_steps=360,
    K=10,
    outfile_timeseries="delta_delay_timeseries.csv",
    outfile_switch="delta_delay_switch_only.csv",
):
    t0 = Time(start_time_utc, scale="utc")

    timeseries_rows = []
    switch_rows = []

    prev_best_path = None
    prev_best_delay = None

    for step in range(int(num_steps)):
        env.epoch = t0 + (step * dt_sec) / 86400.0
        env._build_new_graph()

        best_p, best_delay = best_path_at_time(env, src_name, dst_name, demand, K=K)

        if best_p is None or best_delay is None:
            path_type = "none"
        elif best_p == "GROUND":
            path_type = "ground"
        else:
            path_type = env.classify_path(best_p)

        switched = 0
        delta_delay = ""

        if (
            prev_best_path is not None
            and best_p is not None
            and prev_best_delay is not None
            and best_p != prev_best_path
        ):
            switched = 1
            delta_delay_val = float(best_delay - prev_best_delay)
            delta_delay = delta_delay_val

            switch_rows.append({
                "epoch": str(env.epoch),
                "delta_delay_ms": delta_delay_val,
            })

        timeseries_rows.append({
            "step": step,
            "epoch": str(env.epoch),
            "delay_ms": "" if best_delay is None else float(best_delay),
            "path_type": path_type,
            "switched": switched,
            "delta_delay_ms": delta_delay,
        })

        prev_best_path = best_p
        prev_best_delay = best_delay

    # 写 CSV：time series
    with open(outfile_timeseries, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "epoch", "delay_ms", "path_type", "switched", "delta_delay_ms"],
        )
        writer.writeheader()
        writer.writerows(timeseries_rows)

    # 写 CSV：switch only
    with open(outfile_switch, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "delta_delay_ms"])
        writer.writeheader()
        writer.writerows(switch_rows)

    print(f"[DATA] time-series saved -> {outfile_timeseries} ({len(timeseries_rows)} rows)")
    print(f"[DATA] switch-only saved -> {outfile_switch} ({len(switch_rows)} rows)")

    return timeseries_rows, switch_rows


def plot_cdf(deltas_ms, title, out_png, out_pdf):
    if len(deltas_ms) == 0:
        print("[WARN] No switch events -> deltas list is empty. No plot generated.")
        return

    x = np.sort(np.array(deltas_ms, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    med = float(np.median(x))

    plt.figure(figsize=(5.5, 4))
    plt.plot(x, y, linewidth=2, label="Empirical CDF")
    plt.axvline(med, linestyle="--", linewidth=1.5, label=f"Median = {med:.2f} ms")
    plt.xlabel("Δ Delay at Path Switch (ms)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print(f"[PLOT] saved: {out_png}")
    print(f"[PLOT] saved: {out_pdf}")


def main():
    SRC_NAME = "New York City, USA"
    DST_NAME = "Paris, France"
    DEMAND = 8.0

    env = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=[2, 5, 8],
        K_paths=15,
        requests_per_episode=1,
        shuffle_requests=False,
        epoch="2025-12-05 06:00:00",
        enable_weather=False,
    )
    env.seed(37)
    env.generate_environment(0, [2, 5, 8])

    timeseries, switch_rows = run_switch_delta_delay(
        env,
        src_name=SRC_NAME,
        dst_name=DST_NAME,
        demand=DEMAND,
        start_time_utc="2025-12-05 06:00:00",
        dt_sec=10,
        num_steps=360,
        K=15,
    )

    # ✅ 从 switch_rows 提取 deltas_ms
    deltas_ms = [float(r["delta_delay_ms"]) for r in switch_rows if r.get("delta_delay_ms") is not None]

    print(f"[DONE] steps={len(timeseries)} switch_events={len(deltas_ms)}")
    if deltas_ms:
        print(
            f"[STATS] mean={np.mean(deltas_ms):.3f} ms | "
            f"median={np.median(deltas_ms):.3f} ms | "
            f"min={np.min(deltas_ms):.3f} ms | max={np.max(deltas_ms):.3f} ms"
        )

    plot_cdf(
        deltas_ms,
        title="Empirical CDF of ΔDelay at Path Switch",
        out_png="delta_delay_switch_cdf.png",
        out_pdf="delta_delay_switch_cdf.pdf",
    )


if __name__ == "__main__":
    main()
