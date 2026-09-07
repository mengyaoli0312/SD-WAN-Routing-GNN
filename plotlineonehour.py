import os
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl


mpl.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def parse_epoch_utc(s: str) -> datetime:
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc)


def read_jsonl(path):
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 跳过明显不是 JSON 的行
            if not line.startswith("{"):
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 脏行直接跳过
                continue


def load_delays_from_detail(detail_path: str) -> np.ndarray:
    delays = []
    for rec in read_jsonl(detail_path):
        if rec.get("success") is False:
            continue
        d = rec.get("e2e_delay_ms")
        if d is None:
            continue
        if isinstance(d, (int, float)) and np.isfinite(d) and d >= 0:
            delays.append(float(d))
    return np.asarray(delays, dtype=float)


def quantiles(x: np.ndarray):
    mean = float(np.mean(x))
    p05, p25, p75, p95 = np.percentile(x, [5, 25, 75, 95])
    return mean, float(p05), float(p25), float(p75), float(p95)


def main():
    # ====== 你喜欢的“写死参数”风格 ======
    summary = "results/resultonehour/time_sweep_summary.jsonl"
    detail_dir = "results/resultonehour"
    start = "2025-12-05T06:00:00.000"
    out = "one_hour_delay_0600.png"
    title = "E2E Delay over Time (Mean + Variability from detail files)"
    # ===================================

    start_dt = parse_epoch_utc(start)
    end_dt = start_dt + timedelta(hours=1)

    rows = []
    for rec in read_jsonl(summary):
        ts = rec.get("epoch_utc")
        if not ts:
            continue

        t = parse_epoch_utc(ts)
        if not (start_dt <= t < end_dt):
            continue

        detail_file = rec.get("detail_file")
        if not detail_file:
            continue

        # detail_file 可能是 "/xxx_details.jsonl" 或 "xxx_details.jsonl"
        detail_name = os.path.basename(detail_file)
        detail_path = os.path.join(detail_dir, detail_name)
        if not os.path.exists(detail_path):
            continue

        delays = load_delays_from_detail(detail_path)
        if delays.size == 0:
            continue

        mean, p05, p25, p75, p95 = quantiles(delays)
        rows.append((t, mean, p25, p75, p05, p95))

    rows.sort(key=lambda r: r[0])

    if not rows:
        raise RuntimeError(
            "No valid records found in the 1-hour window. "
            "Check `start`, `summary`, and `detail_dir`."
        )

    times, mean, p25, p75, p05, p95 = map(np.array, zip(*rows))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, mean, marker="o", label="Mean")
    ax.fill_between(times, p25, p75, alpha=0.25, label="P25–P75 (IQR)")
    ax.fill_between(times, p05, p95, alpha=0.15, label="P05–P95 (Whiskers)")

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("E2E Delay (ms)")

    # 每 10 分钟一个刻度（你示例图差不多就是这种节奏）
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
