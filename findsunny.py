import pandas as pd
import numpy as np

CSV_PATH = "analysis_data_Enschede.csv"          # 改成你的文件名
SITE_FILTER = None             # 例如 "utwente"；不筛选就 None
TIME_COL = "timestamp_start"   # 你CSV里就是这个
RAIN_COL = "rain"              # 你CSV里就是这个

# 可选：如果你想按本地时区判断“自然日”
# 你的数据看起来像本地时间字符串（无时区），一般保持 None 就行
TZ = None  # 例如 "Europe/Rome" / "UTC"

def build_hour_table(df: pd.DataFrame):
    # 解析时间
    t = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.copy()
    df["_t"] = t
    df = df.dropna(subset=["_t"])

    # 时区处理（可选）
    if TZ:
        if df["_t"].dt.tz is None:
            df["_t"] = df["_t"].dt.tz_localize(TZ)
        else:
            df["_t"] = df["_t"].dt.tz_convert(TZ)

    # rain 数值化
    df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce")

    # 小时桶 & 日期
    df["_hour_ts"] = df["_t"].dt.floor("H")
    df["_date"] = df["_t"].dt.date
    df["_hour"] = df["_t"].dt.hour

    # 每小时聚合：只要该小时出现过 rain>0，就认为该小时“下雨”
    g = df.groupby(["_date", "_hour_ts"], as_index=False).agg(
        hour=("_hour", "first"),
        rain_max=("_rain", "max"),
        rain_nonnull=("_rain", lambda x: x.notna().sum()),
        n=(" _rain".strip(), "size")  # 不影响运行
    )
    g = df.groupby(["_date", "_hour_ts"], as_index=False).agg(
        hour=("_hour", "first"),
        rain_max=("_rain", "max"),
        rain_nonnull=("_rain", lambda x: x.notna().sum()),
        count=("_rain", "size"),
    )

    g["is_clear_hour"] = (g["rain_max"].fillna(0) == 0)
    return g

def find_full_clear_days(hour_df: pd.DataFrame):
    rows = []
    for day, d in hour_df.groupby("_date"):
        hours = sorted(set(d["hour"].tolist()))
        missing = [h for h in range(24) if h not in hours]

        has_24 = (len(hours) == 24)
        all_clear = bool(has_24 and d["is_clear_hour"].all())

        rain_hours = sorted(set(d.loc[~d["is_clear_hour"], "hour"].tolist()))

        rows.append({
            "date": str(day),
            "has_24_hours": has_24,
            "all_clear_24h": all_clear,
            "hours_present": ",".join(map(str, hours)),
            "missing_hours": ",".join(map(str, missing)),
            "rain_hours": ",".join(map(str, rain_hours)),
        })

    res = pd.DataFrame(rows).sort_values(
        ["all_clear_24h", "has_24_hours", "date"],
        ascending=[False, False, True]
    )
    return res

def find_rolling_clear_windows(df: pd.DataFrame):
    """
    可选增强：找任意“连续24小时窗口”全为晴（rain==0）
    不要求对齐自然日。
    """
    df = df.copy()
    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_t"])
    df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce").fillna(0)

    # 转小时粒度：每小时 rain_max
    df["_hour_ts"] = df["_t"].dt.floor("H")
    h = df.groupby("_hour_ts", as_index=False).agg(rain_max=("_rain", "max"))
    h = h.sort_values("_hour_ts").reset_index(drop=True)

    # 补齐缺失小时（很重要）：没数据的小时我们标记为 NaN（不能算作覆盖）
    full_range = pd.date_range(h["_hour_ts"].min(), h["_hour_ts"].max(), freq="H")
    h = h.set_index("_hour_ts").reindex(full_range)
    h.index.name = "_hour_ts"
    h = h.reset_index()

    # 规则：窗口内必须 24 小时都有观测（非NaN），并且 rain_max 全为0
    is_obs = h["rain_max"].notna().astype(int)
    is_clear = (h["rain_max"].fillna(1) == 0).astype(int)

    # rolling sum
    obs_24 = is_obs.rolling(24).sum()
    clear_24 = is_clear.rolling(24).sum()

    ok = (obs_24 == 24) & (clear_24 == 24)
    windows = []
    idxs = np.where(ok.values)[0]
    for i in idxs:
        end = h.loc[i, "_hour_ts"]
        start = h.loc[i-23, "_hour_ts"]
        windows.append({"start": str(start), "end": str(end)})

    return pd.DataFrame(windows)

def main():
    df = pd.read_csv(CSV_PATH)

    if SITE_FILTER:
        df = df[df["site_name"] == SITE_FILTER].copy()

    # 1) 自然日 24小时全晴
    hour_df = build_hour_table(df)
    daily = find_full_clear_days(hour_df)

    print("\n=== Full-day check (00-23) ===")
    good = daily[(daily["has_24_hours"]) & (daily["all_clear_24h"])]
    if len(good) == 0:
        print("No full 24-hour clear day found.")
    else:
        print("Found full 24-hour clear days:")
        print(good[["date"]].to_string(index=False))

    # 打印最接近的几天（缺小时最少）
    daily["n_missing"] = daily["missing_hours"].apply(lambda s: 0 if (pd.isna(s) or s == "") else len(s.split(",")))
    near = daily.sort_values(["n_missing", "date"]).head(10)
    print("\n=== Closest days (missing hours fewest) top10 ===")
    print(near[["date", "n_missing", "missing_hours", "rain_hours"]].to_string(index=False))

    daily.to_csv("daily_24h_clear_check.csv", index=False)
    print("\n[OK] saved: daily_24h_clear_check.csv")

    # 2) 可选：滚动连续24小时窗口
    windows = find_rolling_clear_windows(df)
    windows.to_csv("rolling_24h_clear_windows.csv", index=False)
    print(f"[OK] saved: rolling_24h_clear_windows.csv  (windows={len(windows)})")

if __name__ == "__main__":
    main()
