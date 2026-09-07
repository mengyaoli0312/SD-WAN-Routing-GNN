# =========================
# USER CONFIG (只改这里)
# =========================

import os
import json
import pandas as pd
import numpy as np
from astropy.time import Time

from environment1 import SatelliteGraphEnv
from train_DQN import DQNAgent, hparams, listofDemands, checkpoint_dir, SEED

CSV_PATH = "analysis_data_Enschede.csv"

TARGET_DATE = "2023-10-26"        # 固定哪一天 (YYYY-MM-DD)
SITE_FILTER = None                # 例如 "utwente"，不筛选就 None

TIMEZONE = "UTC"          # CSV 时间的时区；如果 CSV 本身是 UTC，可设 None 或 "UTC"
PICK_CLEAR_SKY = False  #False means rain , true means sunny.
Mean_method = True

# 固定 60 个请求（必须合起来正好 60 条）
FIXED_REQ_FILES = [
    "./fixed_reqs1.jsonl",
    "./fixed_reqs2.jsonl",
    "./fixed_reqs3.jsonl",
]

OUT_DIR = "mean_out_rain"          # 输出目录名

# 每小时选 rain>0 的策略
REQUIRE_RAIN_GT_ZERO = True       # True = 必须 rain>0；False = 允许 fallback

TIME_COL = "timestamp_start"
UPLOAD_COL = "upload"
DOWNLOAD_COL = "download"
PING_COL = "ping_avg"
RAIN_COL = "rain"
SITE_COL = "site_name"

ENABLE_WEATHER = False  # 你现在不测雨衰就 False


# =========================
# Helpers
# =========================
def _try_set_weather(env, enable: bool):
    """兼容不同 env 字段名，尽量打开/关闭天气模块"""
    for attr in ["enable_weather", "weather", "weather_enabled", "use_weather"]:
        if hasattr(env, attr):
            try:
                setattr(env, attr, bool(enable))
            except Exception:
                pass


def _build_env_for_epoch(epoch_time: Time) -> SatelliteGraphEnv:
    """每个时间点创建一个新 env，保证拓扑刷新"""
    env_eval = SatelliteGraphEnv(
        tle_path="iridium1209_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv=CSV_PATH,
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


def load_jsonl(path: str):
    """
    读取 jsonl 文件：每行一个 JSON object
    这里用于读取“固定 60 个 requests”（和 CSV 没关系）
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"[ERROR] JSON parse failed in {path} line {ln}: {e}")
    return items


def build_name_alias(env):
    def norm(s: str):
        return "".join(str(s).lower().split())
    alias = {}
    for name in env.G.nodes:
        alias[norm(name)] = name
    return alias, norm


def normalize_requests(reqs, env):
    """把 jsonl 里的 src/dst name 映射到 env 节点名"""
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
def apply_measurement_to_env(env_eval: SatelliteGraphEnv, row: pd.Series):
    """
    把 picked 的观测值注入 env，让 env 用它来计算 ground bandwidth/delay
    这里是“兼容式写法”：优先调用 env 内部方法；否则写属性供你在 env 内部读取。
    """
    meas = {
        "timestamp_start": str(row.get(TIME_COL)),
        "upload": None if pd.isna(row.get(UPLOAD_COL)) else float(row.get(UPLOAD_COL)),
        "download": None if pd.isna(row.get(DOWNLOAD_COL)) else float(row.get(DOWNLOAD_COL)),
        "ping_avg": None if pd.isna(row.get(PING_COL)) else float(row.get(PING_COL)),
        "rain": None if pd.isna(row.get(RAIN_COL)) else float(row.get(RAIN_COL)),
    }

    # 1) 如果 env 有明确接口（你可以把这里改成你真实的方法名）
    for fn_name in ["set_measurement", "apply_measurement", "set_ground_measurement", "update_from_measurement"]:
        fn = getattr(env_eval, fn_name, None)
        if callable(fn):
            fn(meas)
            print(f"[INFO] applied measurement via env.{fn_name}():", meas)
            return

    # 2) fallback：写成属性（你需要在 env 的 make_step / 计算容量时读取这些属性）
    env_eval.current_measurement = meas
    env_eval.meas_upload = meas["upload"]
    env_eval.meas_download = meas["download"]
    env_eval.meas_ping_avg = meas["ping_avg"]
    env_eval.meas_rain = meas["rain"]
    env_eval.meas_timestamp_start = meas["timestamp_start"]
    print("[WARN] env has no set_measurement-like method. Measurement stored in env_eval.current_measurement:", meas)
    print("[DBG] meas injected:", env_eval.meas_upload, env_eval.meas_download, env_eval.meas_ping_avg, env_eval.meas_rain)
    print(f"[MEAS][OK] injected via attributes: {env_eval.current_measurement}")
    assert hasattr(env_eval, "current_measurement") and env_eval.current_measurement["upload"] == meas["upload"]

def trimmed_mean(x: pd.Series, min_keep: int = 3) -> float:
    """
    IQR-trimmed mean with safety fallback.
    - If trimmed set is empty or too small, fallback to median.
    """
    x = x.dropna()
    if len(x) == 0:
        return np.nan

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    # x_trimmed = x[(x >= q1) & (x <= q3)]
    x_trimmed = x[(x >= q1)]

    # fallback 1: trimmed 为空
    if len(x_trimmed) == 0:
        return float(x.median())

    # fallback 2: trimmed 太少，不稳定
    if len(x_trimmed) < min_keep:
        return float(x.median())

    return float(x_trimmed.mean())

def avg_rows_per_hour(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    平均方式（每小时聚合一行）：
    - 先解析时间 _t，并生成 _hour
    - upload/download/ping/rain 转数值
    - 字段完整：_rain/_up/_down/_ping 不为 NaN
    - PICK_CLEAR_SKY=False: 只保留 rain>0 的记录再做小时均值
      PICK_CLEAR_SKY=True : 只保留 rain==0 的记录再做小时均值（isclose）
      若某小时过滤后为空：fallback 用该小时“字段完整”的所有记录做均值（可按需关掉）
    返回：每小时一行，包含 hour、count、以及均值字段（upload/download/ping/rain）
    """
    df = day_df.copy()
    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_t"]).copy()

    df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce")
    df["_up"]   = pd.to_numeric(df[UPLOAD_COL], errors="coerce")
    df["_down"] = pd.to_numeric(df[DOWNLOAD_COL], errors="coerce")
    df["_ping"] = pd.to_numeric(df[PING_COL], errors="coerce")
    df["_hour"] = df["_t"].dt.hour

    required_notna = ["_rain", "_up", "_down", "_ping"]
    df = df.dropna(subset=required_notna).copy()

    # === 删除“同一天同一小时内又晴又雨”的整小时数据 ===
    df["_date"] = df["_t"].dt.date

    clear_mask = np.isclose(df["_rain"].astype(float), 0.0, atol=1e-12)
    rain_mask  = df["_rain"] > 0

    mix_keys = (
        df.assign(_is_clear=clear_mask, _is_rain=rain_mask)
        .groupby(["_date", "_hour"])[["_is_clear", "_is_rain"]]
        .any()
    )

    bad_keys = mix_keys.index[mix_keys["_is_clear"] & mix_keys["_is_rain"]]

    if len(bad_keys) > 0:
        bad_df = (
            pd.DataFrame(bad_keys.tolist(), columns=["_date", "_hour"])
        )
        df = (
            df.merge(bad_df, on=["_date", "_hour"], how="left", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
            .copy()
        )


    out_rows = []
    for h in range(24):
        hh = df[df["_hour"] == h].copy()
        if hh.empty:
            continue
        rain_flag = 1
        # 先按模式过滤
        if PICK_CLEAR_SKY:
            cand = hh[np.isclose(hh["_rain"].astype(float), 0.0, atol=1e-12)].copy()
            reason = "clear_sky_rain==0_hourly_mean"
            rain_flag = 1
        else:
            cand = hh[hh["_rain"] > 5].copy()
            reason = "rain_mode_rain>0_hourly_mean"
            rain_flag =  0
        # 若该小时过滤后为空：fallback 用该小时完整字段的所有记录做均值
        if cand.empty:
            cand = hh
            reason += "_fallback_use_all_complete_in_hour"
        row = {
            "hour": h,
            "count": int(len(cand)),
            # 代表时间：该小时最早一条（也可改成整点）
            TIME_COL: cand["_t"].min(),
            "rain": float(cand["_rain"].mean()),
            "upload": trimmed_mean(cand["_up"]),
            "download": trimmed_mean(cand["_down"]),
            "ping_avg": trimmed_mean(cand["_ping"]),
            # 可选：方差/极值，做特征更稳
            "rain_max": float(cand["_rain"].max()),
            "upload_min": float(cand["_up"].min()),
            "upload_max": float(cand["_up"].max()),
            "reason": reason,
            "picked_from_no_rain": 0,
        }
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
        out = out.sort_values(["hour", TIME_COL]).reset_index(drop=True)
    return out



# def avg_rows_per_hour(day_df: pd.DataFrame, weather: str) -> pd.DataFrame:
#     """
#     同一小时内按每条记录分流：
#       - weather="sunny": 该小时内仅用 rain==0 的条目求平均
#       - weather="rainy": 该小时内仅用 rain>0 的条目求平均
#     不做 fallback：某小时无样本 -> 输出 NaN，count=0（仍保留该小时行）。
#     """
#     assert weather in ("sunny", "rainy")

#     df = day_df.copy()
#     df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
#     df = df.dropna(subset=["_t"]).copy()

#     # 转成数值
#     df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce")
#     df["_up"]   = pd.to_numeric(df[UPLOAD_COL], errors="coerce")
#     df["_down"] = pd.to_numeric(df[DOWNLOAD_COL], errors="coerce")
#     df["_ping"] = pd.to_numeric(df[PING_COL], errors="coerce")

#     # 字段完整才参与平均（否则会污染均值）
#     df = df.dropna(subset=["_rain", "_up", "_down", "_ping"]).copy()
#     df["_hour"] = df["_t"].dt.hour

#     rows = []
#     for h in range(24):
#         hh = df[df["_hour"] == h].copy()

#         if weather == "sunny":
#             # 每条记录 rain==0 才算晴天
#             cand = hh[np.isclose(hh["_rain"].astype(float), 0.0, atol=1e-12)].copy()
#             reason = "sunny_by_entry(rain==0)_hourly_mean"
#         else:
#             # 每条记录 rain>0 才算雨天
#             cand = hh[hh["_rain"] > 0].copy()
#             reason = "rainy_by_entry(rain>0)_hourly_mean"

#         if cand.empty:
#             rows.append({
#                 "hour": h,
#                 "count": 0,
#                 TIME_COL: pd.NaT,
#                 UPLOAD_COL: np.nan,
#                 DOWNLOAD_COL: np.nan,
#                 PING_COL: np.nan,
#                 RAIN_COL: np.nan,
#                 "reason": reason + "_NO_SAMPLES",
#             })
#             continue

#         rows.append({
#             "hour": h,
#             "count": int(len(cand)),
#             TIME_COL: cand["_t"].min(),  # 代表时间：该组最早一条
#             UPLOAD_COL: float(cand["_up"].mean()),
#             DOWNLOAD_COL: float(cand["_down"].mean()),
#             PING_COL: float(cand["_ping"].mean()),
#             RAIN_COL: float(cand["_rain"].mean()),
#             "reason": reason,
#         })

#     out = pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)
#     out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
#     return out



def pick_one_rain_row_per_hour(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    PICK_CLEAR_SKY=False:
      - 每小时 rain>0
      - 过滤字段完整
      - 按 rain desc, time asc 排序
      - 取前5条 -> 在这5条里选 upload 最小 的那条（time 最早打破并列）

    PICK_CLEAR_SKY=True:
      - 每小时 rain==0
      - 过滤字段完整
      - 选 upload 最大 的那条（time 最早打破并列）

    字段完整：upload/download/ping/rain 都能转为数且不为 NaN（timestamp_start 由 _t 保证）
    """
    df = day_df.copy()
    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_t"]).copy()

    df["_rain"] = pd.to_numeric(df[RAIN_COL], errors="coerce")
    df["_up"]   = pd.to_numeric(df[UPLOAD_COL], errors="coerce")
    df["_down"] = pd.to_numeric(df[DOWNLOAD_COL], errors="coerce")
    df["_ping"] = pd.to_numeric(df[PING_COL], errors="coerce")
    df["_hour"] = df["_t"].dt.hour

    required_notna = ["_rain", "_up", "_down", "_ping"]

    def complete(d: pd.DataFrame) -> pd.DataFrame:
        return d.dropna(subset=required_notna)

    picked_rows = []

    for h in range(24):
        hh = df[df["_hour"] == h].copy()
        if hh.empty:
            continue

        if PICK_CLEAR_SKY:
            # 晴天：rain==0（用 isclose 防浮点误差）
            cand = hh[np.isclose(hh["_rain"].astype(float), 0.0, atol=1e-12)].copy()
            cand = complete(cand)

            if cand.empty:
                # 没有完全晴天且字段完整的数据：你可以选择报错或 fallback
                # 这里给一个 fallback：选 rain 最小 + upload 最大（字段完整）
                cand2 = complete(hh).sort_values(["_rain", "_up", "_t"], ascending=[True, False, True])
                if cand2.empty:
                    # 实在没有完整字段：直接挑 upload 最大(可能有缺失)兜底
                    cand2 = hh.sort_values(["_up", "_t"], ascending=[False, True])
                    row = cand2.iloc[0].to_dict()
                    row["hour"] = h
                    row["picked_from_no_rain"] = 1
                    row["reason"] = "clear_sky_none_complete_fallback_max_upload_even_incomplete"
                    picked_rows.append(row)
                    continue

                row = cand2.iloc[0].to_dict()
                row["hour"] = h
                row["picked_from_no_rain"] = 1
                row["reason"] = "clear_sky_none_complete_fallback_min_rain_then_max_upload"
                picked_rows.append(row)
                continue

            # 晴天：upload 最大优先，并列时间最早
            cand = cand.sort_values(["_up", "_t"], ascending=[False, True])
            row = cand.iloc[0].to_dict()
            row["hour"] = h
            row["picked_from_no_rain"] = 0
            row["reason"] = "clear_sky_rain==0_pick_max_upload_complete"
            picked_rows.append(row)

        else:
            # 雨天：rain>0
            cand = hh[hh["_rain"] > 0].copy()
            cand = complete(cand)

            if cand.empty:
                # 没有 rain>0 且字段完整：这里按你的原逻辑可标记并兜底
                # 兜底：从所有完整字段里挑 rain 最大（可能为0）
                cand2 = complete(hh).sort_values(["_rain", "_t"], ascending=[False, True])
                if cand2.empty:
                    # 实在没有完整字段：直接挑 rain 最大兜底
                    cand2 = hh.sort_values(["_rain", "_t"], ascending=[False, True])
                    row = cand2.iloc[0].to_dict()
                    row["hour"] = h
                    row["picked_from_no_rain"] = 1
                    row["reason"] = "rain_mode_none_complete_fallback_max_rain_even_incomplete"
                    picked_rows.append(row)
                    continue

                row = cand2.iloc[0].to_dict()
                row["hour"] = h
                row["picked_from_no_rain"] = 1
                row["reason"] = "rain_mode_none_rain>0_complete_fallback_max_rain_complete"
                picked_rows.append(row)
                continue

            # 先按 rain 最大排序，取“字段完整”的前5条
            cand = cand.sort_values(["_rain", "_t"], ascending=[False, True])
            top5 = cand.head(5).copy()

            # 在 top5 里选 upload 最小；并列时间最早
            top5 = top5.sort_values(["_up", "_t"], ascending=[True, True])
            row = top5.iloc[0].to_dict()
            row["hour"] = h
            row["picked_from_no_rain"] = 0
            row["reason"] = f"rain_mode_top5_by_rain_then_pick_min_upload_complete(n_top={len(top5)})"
            picked_rows.append(row)

    picked_df = pd.DataFrame(picked_rows)
    if not picked_df.empty:
        picked_df["_t"] = pd.to_datetime(picked_df[TIME_COL], errors="coerce")
        picked_df = picked_df.sort_values(["hour", "_t"]).reset_index(drop=True)

    return picked_df



def to_utc_isot(ts: pd.Timestamp, tz: str | None) -> str:
    """把 pandas Timestamp 变成 UTC 的 'YYYY-MM-DD HH:MM:SS'，供 astropy Time 使用。"""
    if tz:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        ts_utc = ts.tz_convert("UTC")
        return ts_utc.strftime("%Y-%m-%d %H:%M:%S")
    else:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.strftime("%Y-%m-%d %H:%M:%S")


def _percentile(xs, q):
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def run_one_timepoint(epoch_time: Time, fixed_requests, out_prefix: str, meas_row=None):
    """
    跑一个时间点：一个 episode=60 requests
    输出：{out_prefix}_details.jsonl
    返回：summary dict
    """
    env_eval = _build_env_for_epoch(epoch_time)
    pre = {
    "ground_fiber_delay_ms": getattr(env_eval, "ground_fiber_delay_ms", None),
    "base_cap_up_mean": float(np.mean(getattr(env_eval, "base_cap_up", [0]))) if hasattr(env_eval, "base_cap_up") else None,
    "eff_cap_up_mean": float(np.mean(getattr(env_eval, "eff_cap_up", [0]))) if hasattr(env_eval, "eff_cap_up") else None,
    }

    apply_measurement_to_env(env_eval, meas_row)

    # 注入后快照（注：如果 env 只有在 reset()/build graph 后才应用 meas，这里可能不变）
    post = {
        "ground_fiber_delay_ms": getattr(env_eval, "ground_fiber_delay_ms", None),
        "base_cap_up_mean": float(np.mean(getattr(env_eval, "base_cap_up", [0]))) if hasattr(env_eval, "base_cap_up") else None,
        "eff_cap_up_mean": float(np.mean(getattr(env_eval, "eff_cap_up", [0]))) if hasattr(env_eval, "eff_cap_up") else None,
    }

    print("[MEAS][CHECK] pre =", pre)
    print("[MEAS][CHECK] post=", post)
    print("[MEAS][CHECK] stored measurement =", getattr(env_eval, "current_measurement", None))

    # 3) 固定请求
    env_eval.use_fixed_requests = True
    env_eval.fixed_requests = fixed_requests
    env_eval.fixed_shuffle = False
    env_eval.requests_per_episode = 60

    agent = DQNAgent(env_eval, hparams["batch_size"])
    agent.load_model(checkpoint_dir)

    # greedy
    for attr in ["epsilon", "eps"]:
        if hasattr(agent, attr):
            try:
                setattr(agent, attr, 0.0)
            except Exception:
                pass

    for net_name in ["policy_net", "qnetwork", "q_net", "model", "net"]:
        net = getattr(agent, net_name, None)
        if net is not None and hasattr(net, "eval"):
            net.eval()

    state, demand, source, destination = env_eval.reset()
    done = False
    total_r = 0.0
    step = 0

    detail_path = f"{out_prefix}_details.jsonl"
    with open(detail_path, "w", encoding="utf-8") as fout:
        while not done:
            action, approx_delay = agent.select_action(env_eval, state, demand, source, destination)
            new_state, reward, done, new_demand, new_source, new_destination = env_eval.make_step(
                state, action, demand, source, destination
            )
            step_avg_utils = []
            step_avg_up_utils = []

            edges_inferred = _infer_edges_from_state_delta(env_eval, state, new_state)

            rec = _get_latest_rec(env_eval, step)
            if edges_inferred:
                util_any_vals = [e["util_after"] for e in edges_inferred]
                util_up_vals  = [e["util_up_after"] for e in edges_inferred if e.get("used_up", 0.0) > 1e-9]
                step_avg_util_any = float(np.mean(util_any_vals)) if util_any_vals else None
                step_avg_util_up  = float(np.mean(util_up_vals))  if util_up_vals  else None
            else:
                step_avg_util_any = None
                step_avg_util_up  = None

            if step_avg_util_any is not None:
                step_avg_utils.append(step_avg_util_any)
            if step_avg_util_up is not None:
                step_avg_up_utils.append(step_avg_util_up)

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
                "step_avg_util": step_avg_util_any,
                "step_avg_upload_util": step_avg_util_up,
            }, ensure_ascii=False) + "\n")

            state = new_state
            demand, source, destination = new_demand, new_source, new_destination
            total_r += float(reward)
            step += 1

    logs = getattr(env_eval, "episode_logs", []) or []
    e2es = [float(r.get("e2e_delay_ms")) for r in logs
            if r.get("success") and r.get("e2e_delay_ms") is not None]

    detail_rel = os.path.relpath(detail_path, OUT_DIR).replace("\\", "/")
    if not detail_rel.startswith("/"):
        detail_rel = "/" + detail_rel

    summary = {
        "epoch_utc": epoch_time.isot,
        "n_success": int(sum(1 for r in logs if r.get("success"))),
        "n_total": int(len(logs)) if logs else 60,
        "mean_e2e_delay_ms": float(np.mean(e2es)) if e2es else 0.0,
        "p95_e2e_delay_ms": _percentile(e2es, 95),
        "total_reward": float(total_r),
        "detail_file": detail_rel,
        "avg_utilization": float(np.mean(step_avg_utils)) if step_avg_utils else 0.0,
        "avg_upload_utilization": float(np.mean(step_avg_up_utils)) if step_avg_up_utils else 0.0,
    }
    return summary


def _edge_utils_after(env, st_after, mode="mean"):
    """
    返回:
      util_any: 每条边的综合利用率
      util_up : 每条边 uplink 利用率
      util_down: 每条边 downlink 利用率
    注意：分母分别用 eff_cap_up / eff_cap_down（或 base_cap_*）
    """
    den_up = getattr(env, "eff_cap_up", getattr(env, "base_cap_up", None))
    den_dn = getattr(env, "eff_cap_down", getattr(env, "base_cap_down", None))
    if den_up is None or den_dn is None:
        raise RuntimeError("[FATAL] env missing eff_cap_up/eff_cap_down (or base_cap_*)")

    den_up = np.asarray(den_up, dtype=float)
    den_dn = np.asarray(den_dn, dtype=float)

    up_rem   = st_after[:, 1].astype(float)
    down_rem = st_after[:, 0].astype(float)

    used_up   = np.clip(den_up - up_rem,   0.0, None)
    used_down = np.clip(den_dn - down_rem, 0.0, None)

    den_up_safe = np.where(den_up > 0.0, den_up, 1.0)
    den_dn_safe = np.where(den_dn > 0.0, den_dn, 1.0)

    util_up   = np.clip(np.nan_to_num(used_up / den_up_safe,   nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    util_down = np.clip(np.nan_to_num(used_down / den_dn_safe, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if mode == "max":
        util_any = np.maximum(util_up, util_down)
    else:
        util_any = 0.5 * (util_up + util_down)

    return util_any, util_up, util_down



def _infer_edges_from_state_delta(env, st_before, st_after, eps=1e-9):
    used_down = np.clip(st_before[:, 0].astype(float) - st_after[:, 0].astype(float), 0.0, None)
    used_up   = np.clip(st_before[:, 1].astype(float) - st_after[:, 1].astype(float), 0.0, None)

    used_mask = (used_down > eps) | (used_up > eps)
    idxs = np.where(used_mask)[0]

    util_any_all, util_up_all, util_down_all = _edge_utils_after(env, st_after, mode="mean")

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
        if "SAT" not in u or "SAT" not in v:
            print(int(eidx),u,v,float(used_down[eidx]),float(used_up[eidx]),float(util_up_all[eidx]),float(util_down_all[eidx]))

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


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== 1) 读固定60条请求 =====
    all_reqs = []
    for p in FIXED_REQ_FILES:
        all_reqs.extend(load_jsonl(p))
    if len(all_reqs) != 60:
        raise SystemExit(f"[ERROR] FIXED_REQ_FILES 合计必须是60条，现在是 {len(all_reqs)}")

    dummy_epoch = Time("2025-12-05 00:00:00", scale="utc")
    env_for_norm = _build_env_for_epoch(dummy_epoch)

    fixed_reqs, bad = normalize_requests(all_reqs, env_for_norm)
    if bad:
        print("[ERROR] node name not found for these lines:")
        for i, s, d in bad[:50]:
            print(f"  line {i}: {s!r} -> {d!r}")
        raise SystemExit(1)
    if len(fixed_reqs) != 60:
        raise SystemExit(f"[ERROR] normalize 后可用请求不是60条，而是 {len(fixed_reqs)}")

    # ===== 2) 读CSV + 过滤日期/site + 取所需列 =====
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"[ERROR] CSV_PATH not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print("[INFO] CSV loaded:", CSV_PATH, "rows=", len(df))
    print("[INFO] CSV columns:", list(df.columns))

    if SITE_FILTER is not None and SITE_COL in df.columns:
        df = df[df[SITE_COL] == SITE_FILTER].copy()
        print("[INFO] after site filter:", SITE_FILTER, "rows=", len(df))

    if TIME_COL not in df.columns:
        raise SystemExit(f"[ERROR] TIME_COL not in CSV: {TIME_COL}")

    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["_t"]).copy()
    df["_date"] = df["_t"].dt.date.astype(str)

    day_df = df.copy() # df[df["_date"] == TARGET_DATE].copy()
    print("[INFO] target date rows:", TARGET_DATE, "rows=", len(day_df))
    if day_df.empty:
        # 给你一个提示：看看 CSV 里到底有哪些日期
        sample_dates = df["_date"].value_counts().head(10)
        print("[HINT] top dates in CSV:\n", sample_dates.to_string())
        raise SystemExit(f"[ERROR] {TARGET_DATE} 在 CSV 中没有数据（可能是时区/日期切分不一致）")

    needed = [TIME_COL, UPLOAD_COL, DOWNLOAD_COL, PING_COL, RAIN_COL]
    missing_cols = [c for c in needed if c not in day_df.columns]
    if missing_cols:
        raise SystemExit(f"[ERROR] CSV 缺少列: {missing_cols}；请检查列名映射")

    day_df = day_df[needed].copy()

    # # ===== 3) 每小时挑 1 条 rain>0（或fallback）=====
    # if Mean_method:
    #     picked = avg_rows_per_hour(day_df)
    # else:
    #     picked = pick_one_rain_row_per_hour(day_df)
    # if picked.empty:
    #     raise SystemExit("[ERROR] picked is empty: 这一天按小时没有任何可用记录（时间列解析失败？）")

    # # 保存挑选结果
    # pick_csv = os.path.join(OUT_DIR, "selected_24h_samples.csv")
    # picked_out_cols = [
    #     "hour", TIME_COL, UPLOAD_COL, DOWNLOAD_COL, PING_COL, RAIN_COL,
    #     "picked_from_no_rain", "reason"
    # ]
    # picked[picked_out_cols].to_csv(pick_csv, index=False)
    # print("[OK] saved picked samples ->", pick_csv)
    # print("[INFO] picked hours:", sorted(picked["hour"].astype(int).unique().tolist()))

    # hours = sorted(picked["hour"].astype(int).unique().tolist())
    # missing_hours = [h for h in range(24) if h not in hours]
    # if missing_hours:
    #     raise SystemExit(f"[ERROR] {TARGET_DATE} 缺少小时数据：{missing_hours}（picked 不是24条）")


    # # 严格模式：如果有小时没有 rain>0，直接报错退出
    # if REQUIRE_RAIN_GT_ZERO:
    #     bad_hours = picked[picked.get("picked_from_no_rain", 0) == 1]
    #     if not bad_hours.empty:
    #         print("[ERROR] 以下小时没有 rain>0 的记录（严格模式 REQUIRE_RAIN_GT_ZERO=True）：")
    #         print(bad_hours[["hour", TIME_COL, RAIN_COL, "reason"]].to_string(index=False))
    #         raise SystemExit(1)
    day_df = day_df[needed].copy()

    hourly_df = avg_rows_per_hour(day_df)

    if hourly_df.empty:
        raise SystemExit("[ERROR] hourly_df is empty: 这一天按小时没有任何可用记录")

    # （可选）保存一下
    hourly_csv = os.path.join(OUT_DIR, "hourly_24h.csv")
    hourly_df.to_csv(hourly_csv, index=False)
    print("[OK] saved hourly df ->", hourly_csv)
    print("[INFO] hours:", sorted(hourly_df["hour"].astype(int).unique().tolist()))
    # ===== 4) 跑 24 个时间点（每个时间点跑 60 req）=====    
    summary_path = os.path.join(OUT_DIR, "summary_yao.jsonl")
    BASE_EPOCH_STR = "2025-12-05 00:00:00"

    # 构造 hour -> row 映射（hourly_df 是你的小时级结果表）
    hourly_map = {
        int(r["hour"]): r
        for _, r in hourly_df.iterrows()
    }

    with open(summary_path, "w", encoding="utf-8") as fsum:
        base_epoch = Time(BASE_EPOCH_STR, scale="utc")

        for hour in range(24):
            epoch_time = Time(
                base_epoch.unix + hour * 3600,
                format="unix",
                scale="utc"
            )

            tag = epoch_time.isot.replace(":", "").replace("-", "")
            out_prefix = os.path.join(OUT_DIR, f"hour{hour:02d}_{tag}")

            row = hourly_map.get(hour)  # 可能不存在

            summary = run_one_timepoint(
                epoch_time,
                fixed_reqs,
                out_prefix,
                meas_row=row,   # 没有就传 None
            )

            summary.update({
                "target_date": TARGET_DATE,
                "hour": hour,
                "timestart_raw": None if row is None else str(row.get(TIME_COL)),
                "upload": None if row is None or pd.isna(row.get(UPLOAD_COL)) else float(row.get(UPLOAD_COL)),
                "download": None if row is None or pd.isna(row.get(DOWNLOAD_COL)) else float(row.get(DOWNLOAD_COL)),
                "ping_avg": None if row is None or pd.isna(row.get(PING_COL)) else float(row.get(PING_COL)),
                "rain": None if row is None or pd.isna(row.get(RAIN_COL)) else float(row.get(RAIN_COL)),
                # 👇 picked 相关字段彻底删除
            })

            fsum.write(json.dumps(summary, ensure_ascii=False) + "\n")

            print(
                f"[hour={summary['hour']:02d} epoch_utc={summary['epoch_utc']}] "
                f"rain={summary['rain']} "
                f"success={summary['n_success']}/{summary['n_total']} "
                f"mean={summary['mean_e2e_delay_ms']:.3f}ms "
                f"p95={summary['p95_e2e_delay_ms']:.3f}ms "
                f"reward={summary['total_reward']:.3f} "
                f"detail={summary['detail_file']}"
            )

    print("[DONE] summary ->", summary_path)
    print("[DONE] details ->", OUT_DIR, "(files like hour00_..._details.jsonl)")


if __name__ == "__main__":
    main()
