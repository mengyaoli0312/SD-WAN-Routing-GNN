import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SUMMARY_JSONL = "out_20231224/summary_yao.jsonl"
DETAIL_DIR = "out_20231224"

import matplotlib as mpl
import matplotlib.ticker as mticker

import matplotlib.dates as mdates


mpl.rcParams.update({
    "font.size": 20,          # 全局基础字号
    "axes.titlesize": 24,     # 标题
    "axes.labelsize": 20,     # x/y label
    "xtick.labelsize": 18,    # x 轴刻度
    "ytick.labelsize": 20,    # y 轴刻度
    "legend.fontsize": 20,    # 图例
    "lines.linewidth": 2.0,   # 线宽（论文里很重要）
    "lines.markersize": 6,    # marker 大小
})

def safe_read_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []

    if text[0] == "[":
        try:
            arr = json.loads(text)
            return arr if isinstance(arr, list) else [arr]
        except json.JSONDecodeError:
            pass

    rows, buf = [], ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]

        buf = s if not buf else (buf + " " + s)

        try:
            obj = json.loads(buf)
            rows.append(obj)
            buf = ""
        except json.JSONDecodeError:
            continue

    if buf:
        print(f"[WARN] leftover truncated JSON in {path} (ignored)")

    return rows


# =========================
# 1) 从 detail 里提取样本（每个 epoch 一个样本集合）
# =========================

def samples_delay(detail_path):
    vals = []
    for r in safe_read_json_records(detail_path):
        if r.get("success") is True and r.get("e2e_delay_ms") is not None:
            vals.append(float(r["e2e_delay_ms"]))
    return vals


def upload_util_path_from_record(r):
    if r.get("success") is not True:
        return None
    edges = r.get("edges") or []
    ups = []
    for e in edges:
        used_up = e.get("used_up", 0.0) or 0.0
        v = e.get("util_up_after", None)
        if v is None:
            continue
        # 关键：只算真的走 upload 的边
        if float(used_up) > 0:
            ups.append(float(v))
    if not ups:
        return None
    return float(np.mean(ups))



def samples_upload_util(detail_path):
    vals = []
    for r in safe_read_json_records(detail_path):
        v = upload_util_path_from_record(r)
        if v is not None:
            vals.append(v)
    return vals


# =========================
# 2) 通用：把“样本列表”变成 mean+分位数
# =========================

def summarize_samples(a):
    a = np.asarray(a, dtype=float)
    return dict(
        mean=float(a.mean()),
        p05=float(np.percentile(a, 5)),
        p25=float(np.percentile(a, 25)),
        p50=float(np.percentile(a, 50)),
        p75=float(np.percentile(a, 75)),
        p95=float(np.percentile(a, 95)),
        n=int(a.size),
    )


def build_time_stats_from_detail(summary_path, sampler_fn):
    rows = safe_read_json_records(summary_path)
    if not rows:
        raise RuntimeError(f"summary 读不到任何记录：{summary_path}")

    df_sum = pd.DataFrame(rows)
    if "epoch_utc" not in df_sum.columns:
        raise RuntimeError("summary 里缺少 epoch_utc 字段")

    df_sum["t"] = pd.to_datetime(df_sum["epoch_utc"], utc=True, errors="coerce")
    df_sum = df_sum.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    out = []
    for _, row in df_sum.iterrows():
        detail_file = row.get("detail_file")
        detail_path = None
        if isinstance(detail_file, str) and detail_file.strip():
            detail_path = DETAIL_DIR + detail_file.strip()
        

        rec = {"t": row["t"], "detail_file": detail_path,
               "mean": np.nan, "p05": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p95": np.nan, "n": 0}

        if detail_path:
            try:
                vals = sampler_fn(detail_path)
                if len(vals) > 0:
                    rec.update(summarize_samples(vals))
            except FileNotFoundError:
                print(f"[WARN] detail file not found: {detail_path}")

        out.append(rec)
    return pd.DataFrame(out).sort_values("t").reset_index(drop=True)


# =========================
# 3) 绘图：mean + 分位带
# =========================

def plot_mean_with_bands(df, y_label, title, out_png, y_lim=None):
    plt.figure(figsize=(12,6))  # 双栏跨两列常用尺寸
    plt.plot(df["t"], df["mean"], marker="o", linewidth=2.0, markersize=6, label="Mean")

    has_iqr = df["p25"].notna().any() and df["p75"].notna().any()
    has_whisk = df["p05"].notna().any() and df["p95"].notna().any()

    if has_iqr:
        plt.fill_between(df["t"], df["p25"], df["p75"], alpha=0.25, label="P25–P75 (IQR)")
    if has_whisk:
        plt.fill_between(df["t"], df["p05"], df["p95"], alpha=0.12, label="P05–P95 (Whiskers)")

    plt.xlabel("Time (UTC)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    ax = plt.gca()

    # 每 4 个点显示一个刻度/label（按数据点索引）
    step = 4
    ax.set_xticks(df["t"].iloc[::step])

    # 时间显示格式（可按需改）
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # 避免重叠
    ax.tick_params(axis="x", labelrotation=30)

    if y_lim is not None:
        plt.ylim(*y_lim)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[OK] saved: {out_png}")

# =========================
# 4) 生成两张图
# =========================

df_delay = build_time_stats_from_detail(SUMMARY_JSONL, samples_delay)
print("[INFO][DELAY] points =", len(df_delay), "detail available =", df_delay["n"].gt(0).sum(), "/", len(df_delay))
plot_mean_with_bands(
    df_delay,
    y_label="E2E Delay (ms)",
    title="E2E Delay over Time (Mean + Variability from detail files)",
    out_png="figure_delay.png",
)

df_up = build_time_stats_from_detail(SUMMARY_JSONL, samples_upload_util)
print("[INFO][UPLOAD] points =", len(df_up), "detail available =", df_up["n"].gt(0).sum(), "/", len(df_up))
plot_mean_with_bands(
    df_up,
    y_label="Upload Utilization (0–1)",
    title="Upload Utilization over Time (Path mean + variability)",
    out_png="figure_upload_util.png",
    y_lim=(0.0, 1.0),
)

def avg_util_pos_path_from_record(r, util_key="util_after", threshold=0.0):
    """
    计算一条记录的“路径平均 utilization”（只统计 utilization > threshold 的边）
    util_key 可选: util_after / util_up_after / util_down_after
    """
    if r.get("success") is not True:
        return None
    edges = r.get("edges") or []
    vals = []
    for e in edges:
        v = e.get(util_key, None)
        if v is None:
            continue
        v = float(v)
        if v > threshold:
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))

def load_avg_util_pos_from_detail(detail_path, util_key="util_after", threshold=0.0):
    vals = []
    for r in safe_read_json_records(detail_path):
        v = avg_util_pos_path_from_record(r, util_key=util_key, threshold=threshold)
        if v is not None:
            vals.append(v)
    return vals

def build_time_stats_avg_util_pos(summary_path, util_key="util_after", threshold=0.0, detail_dir="out_20231224"):
    rows = safe_read_json_records(summary_path)
    if not rows:
        raise RuntimeError(f"summary 读不到任何记录：{summary_path}")

    df_sum = pd.DataFrame(rows)
    if "epoch_utc" not in df_sum.columns:
        raise RuntimeError("summary 里缺少 epoch_utc 字段")

    df_sum["t"] = pd.to_datetime(df_sum["epoch_utc"], utc=True, errors="coerce")
    df_sum = df_sum.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    out = []
    for _, row in df_sum.iterrows():
        detail = row.get("detail_file", "")
        detail_path = detail_dir + detail.strip() if isinstance(detail, str) and detail.strip() else None

        mean = np.nan
        p05 = p25 = p50 = p75 = p95 = np.nan
        n = 0

        if detail_path:
            try:
                vals = load_avg_util_pos_from_detail(detail_path, util_key=util_key, threshold=threshold)
                if len(vals) > 0:
                    a = np.asarray(vals, dtype=float)
                    n = int(a.size)
                    mean = float(a.mean())
                    p05 = float(np.percentile(a, 5))
                    p25 = float(np.percentile(a, 25))
                    p50 = float(np.percentile(a, 50))
                    p75 = float(np.percentile(a, 75))
                    p95 = float(np.percentile(a, 95))
                    
            except FileNotFoundError:
                print(f"[WARN] detail file not found: {detail_path}")

        out.append({
            "t": row["t"],
            "mean": mean,
            "p05": p05, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
            "n": n,
            "detail_file": detail_path
        })

    return pd.DataFrame(out).sort_values("t").reset_index(drop=True)

df_avgutil = build_time_stats_avg_util_pos(
    SUMMARY_JSONL,
    util_key="util_after",     # 想看 uplink 就改成 util_up_after；downlink 就 util_down_after
    threshold=0.0              # 只统计 >0
)

df_delay.to_csv("delay_stats_for_plot.csv", index=False)
df_up.to_csv("upload_util_stats_for_plot.csv", index=False)
df_avgutil.to_csv("avg_util_stats_for_plot.csv", index=False)

print("[OK] plot data exported:")
print("  adelay_stats_for_plot.csv")
print("  aupload_util_stats_for_plot.csv")
print("  avg_util_stats_for_plot.csv")


print("[INFO][AVG_UTIL_POS] points =", len(df_avgutil),
      "detail available =", df_avgutil["p25"].notna().sum(), "/", len(df_avgutil))

plot_mean_with_bands(
    df_avgutil,
    y_label="Avg Utilization over used links (0–1)",
    title="Average Link Utilization (only util>0 links) over Time",
    out_png="avg_util_pos.png",
    y_lim=(0.0, 1.0),
)
