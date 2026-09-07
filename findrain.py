import pandas as pd
import numpy as np

CSV_PATH = "analysis_data_Osnabrück.csv"          # 改成你的csv
SITE_FILTER = None             # 例如 "utwente"；不筛选就 None
TIME_COL = "timestamp_start"
RAIN_COL = "rain"

# 如果你要按UTC日(24Z)切分：TZ="UTC"
# 如果你要按本地日切分：TZ=None（默认）或者填你的本地时区
TZ = None  # 例如 "UTC" / "Europe/Rome"

def build_hour_table(df: pd.DataFrame):
    df = df.copy()
    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_t"])

    if TZ:
        if df["_t"].dt.tz is None:
            df["_t"] = df["_t"].dt.tz_localize(TZ)
        else:
            df["_t"] = df["_t"].dt.tz_convert(TZ)

    df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce")

    df["_hour_ts"] = df["_t"].dt.floor("H")
    df["_date"] = df["_t"].dt.date
    df["_hour"] = df["_t"].dt.hour

    # 每小时聚合：只要该小时出现过 rain>0，就认为该小时下雨
    hour_df = df.groupby(["_date", "_hour_ts"], as_index=False).agg(
        hour=("_hour", "first"),
        rain_max=("_rain", "max"),
        rain_nonnull=("_rain", lambda x: x.notna().sum()),
        count=("_rain", "size"),
    )

    # 下雨小时：该小时 rain_max > 0
    hour_df["is_rain_hour"] = (hour_df["rain_max"] > 0)
    return hour_df

def find_full_rain_days(hour_df: pd.DataFrame):
    rows = []
    for day, d in hour_df.groupby("_date"):
        hours = sorted(set(d["hour"].tolist()))
        missing = [h for h in range(24) if h not in hours]

        has_24 = (len(hours) == 24)
        all_rain = bool(has_24 and d["is_rain_hour"].all())

        # 哪些小时不是雨（rain<=0 或 NaN聚合后不可能NaN，但以防）
        non_rain_hours = sorted(set(d.loc[~d["is_rain_hour"], "hour"].tolist()))

        rows.append({
            "date": str(day),
            "has_24_hours": has_24,
            "all_rain_24h": all_rain,
            "missing_hours": ",".join(map(str, missing)),
            "non_rain_hours": ",".join(map(str, non_rain_hours)),
        })

    res = pd.DataFrame(rows).sort_values(
        ["all_rain_24h", "has_24_hours", "date"],
        ascending=[False, False, True]
    )
    return res

def find_rolling_rain_windows(df: pd.DataFrame):
    """
    找任意连续24小时窗口：24小时都有观测 + 每小时 rain_max>0
    不要求对齐自然日。
    """
    df = df.copy()
    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_t"])
    df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce")

    df["_hour_ts"] = df["_t"].dt.floor("H")
    h = df.groupby("_hour_ts", as_index=False).agg(rain_max=("_rain", "max"))
    h = h.sort_values("_hour_ts").reset_index(drop=True)

    full_range = pd.date_range(h["_hour_ts"].min(), h["_hour_ts"].max(), freq="H")
    h = h.set_index("_hour_ts").reindex(full_range)
    h.index.name = "_hour_ts"
    h = h.reset_index()

    is_obs = h["rain_max"].notna().astype(int)
    is_rain = (h["rain_max"] > 0).astype(int)  # NaN会变0

    obs_24 = is_obs.rolling(24).sum()
    rain_24 = is_rain.rolling(24).sum()

    ok = (obs_24 == 24) & (rain_24 == 24)

    windows = []
    idxs = np.where(ok.values)[0]
    for i in idxs:
        end = h.loc[i, "_hour_ts"]
        start = h.loc[i - 23, "_hour_ts"]
        windows.append({"start": str(start), "end": str(end)})

    return pd.DataFrame(windows)

def main():
    df = pd.read_csv(CSV_PATH)

    if SITE_FILTER:
        df = df[df["site_name"] == SITE_FILTER].copy()

    hour_df = build_hour_table(df)
    daily = find_full_rain_days(hour_df)

    print("\n=== Full-day check (00-23) ===")
    good = daily[(daily["has_24_hours"]) & (daily["all_rain_24h"])]
    if len(good) == 0:
        print("No full 24-hour ALL-RAIN day found (rain>0 every hour).")
    else:
        print("Found full 24-hour ALL-RAIN days:")
        print(good[["date"]].to_string(index=False))

    # 最接近：缺小时最少，同时 non_rain_hours 最少
    def cnt_list(s):
        if pd.isna(s) or s == "":
            return 0
        return len(str(s).split(","))

    daily["n_missing"] = daily["missing_hours"].apply(cnt_list)
    daily["n_non_rain"] = daily["non_rain_hours"].apply(cnt_list)

    near = daily.sort_values(["n_missing", "n_non_rain", "date"]).head(10)
    print("\n=== Closest days top10 (missing fewest, non-rain fewest) ===")
    print(near[["date", "n_missing", "n_non_rain", "missing_hours", "non_rain_hours"]].to_string(index=False))

    daily.to_csv("daily_24h_all_rain_check.csv", index=False)
    print("\n[OK] saved: daily_24h_all_rain_check.csv")

    windows = find_rolling_rain_windows(df)
    windows.to_csv("rolling_24h_all_rain_windows.csv", index=False)
    print(f"[OK] saved: rolling_24h_all_rain_windows.csv  (windows={len(windows)})")

if __name__ == "__main__":
    main()
