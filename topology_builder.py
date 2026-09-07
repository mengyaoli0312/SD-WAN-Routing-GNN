import numpy as np
import networkx as nx
from tle_loader import load_tle_file
import pandas as pd
from astropy.time import Time

C_KM_PER_MS = 299792.458 / 1000.0  # km/ms ≈ 299.8

V_FIBER_KM_PER_MS = (2.0 / 3.0) * C_KM_PER_MS  # km/ms

R_EARTH_KM = 6371.0

def has_los_horizon_test(rg, rs, R_earth_km=R_EARTH_KM):
    rg = np.asarray(rg, float)
    rs = np.asarray(rs, float)
    Rg = np.linalg.norm(rg)
    Rs = np.linalg.norm(rs)

    # 两个半径向量的夹角
    cos_gamma = np.dot(rg, rs) / (Rg * Rs)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    # 卫星相对地平线的最大可见中心角
    # 对球体：cos(gamma_horizon) = R_earth / Rs
    cos_gh = R_earth_km / Rs
    cos_gh = np.clip(cos_gh, -1.0, 1.0)
    gamma_h = np.arccos(cos_gh)

    return gamma <= gamma_h

def add_sat_ground_edges_with_elev(
    G,
    min_elev_deg: float = 10.0,
    max_per_ground: int = 3,
    c_km_per_ms: float = C_KM_PER_MS,
    sg_max_dist_km: float | None = None,
    gs_profiles: dict[str, dict] | None = None,
):
    """
    Add SAT-GND edges to graph G based on LOS + elevation threshold.
    Rules:
    1) Ground/sat coordinates are all in G.nodes[*]["pos_xyz"], unit km
    2) First check if has_los_vec(...) penetrates the ground.
    3) Then check if elevation >= min_elev_deg
    4) For each ground, only keep the first max_per_ground edges sorted by elevation from highest to lowest.
    Edge attributes:
    kind = "SAT-GND"
    elevation_deg
    distance_km
    delay_ms (vacuum propagation delay)
    """
    ground_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "ground"]
    sat_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sat"]

    num_added = 0

    for g in ground_nodes:
        rg = np.asarray(G.nodes[g]["pos_xyz"], dtype=float)

        candidates = []  # (sat, elev, dist_km)
        for s in sat_nodes:
            rs = np.asarray(G.nodes[s]["pos_xyz"], dtype=float)

            # --- 直接用仰角判断是否可见 ---
            elev = compute_elevation(rg, rs)
            dist_km = float(np.linalg.norm(rs - rg))

            # 自己的 LOS 检查：仰角必须 > 0（或者 > 一个很小的负数做容错）
            if elev <= 0.0:
                continue

            # 仍然保留 sg_max_dist_km 限制
            if (sg_max_dist_km is not None) and (dist_km > sg_max_dist_km):
                continue

            # （可选调试）看看 has_los_vec 和 elevation 是否打架
            # los_flag = has_los_vec(rg, rs, R_earth_km=R_EARTH_KM, margin_km=0.0)
            # if not los_flag:
            #     # 只打印，不再用它做筛选
            #     print(
            #         f"[WARN][LOS-MISMATCH] ground={g}, sat={s}, "
            #         f"elev={elev:.2f}°, dist={dist_km:.1f} km, has_los_vec=False"
            #     )

            candidates.append((s, elev, dist_km))

        # Sort by elevation from highest to lowest
        candidates.sort(key=lambda x: -x[1])

        # 4) elev >= min_elev_deg
        main_selected = [c for c in candidates if c[1] >= min_elev_deg]

        if not main_selected:
            # 这里有两种情况：
            # 1) candidates 非空，但都 < min_elev_deg → fallback 到最高仰角
            # 2) candidates 本身就是空的 → 这个 ground 在当前时刻根本没有任何可见卫星
            if len(candidates) == 0:
                print(
                    f"[WARN] Ground {g} has NO sats with elev>0 & dist<=sg_max_dist_km "
                    f"(min_elev={min_elev_deg}°). Skip SAT-GND for this ground."
                )
                # 不给这个 ground 建 SAT-GND 边，只能走地面光纤
                continue
            else:
                # 有可见卫星，但都没达标 min_elev，就选仰角最高的那颗
                best_sat = candidates[0]
                print(
                    f"[WARN] Ground {g} has no sats with elev >= {min_elev_deg}°. "
                    f"Fallback to best-elev sat {best_sat[0]} elev={best_sat[1]:.2f}°"
                )
                selected = [best_sat]
        else:
            selected = main_selected[:max_per_ground]

        for s, elev, dist_km in selected:
            # WETLINK profile
            prof = gs_profiles.get(g) if gs_profiles is not None else None

            geom_delay_ms = dist_km / c_km_per_ms  # for debug

            if prof is not None:
                if ("upload_Mbps" not in prof) or ("download_Mbps" not in prof):
                    raise RuntimeError(
                        f"[FATAL] gs_profiles[{g}] missing upload/download keys. "
                        f"keys={list(prof.keys())}"
                    )

                cap_dl = float(prof["download_Mbps"])
                cap_ul = float(prof["upload_Mbps"])
                temp = prof.get("temp", 10.0)
                hum = prof.get("humidity", 80.0)
                rain = prof.get("rain", 0.0)
                wind = prof.get("windspeed", 1.0)
                wet_ping = prof.get("ping_ms", 40.0)

                cap_penalty_dl = float(prof.get("capacity_penalty_dl_Mbps", 0.0))
                cap_penalty_ul = float(prof.get("capacity_penalty_ul_Mbps", 0.0))
                dl_degrade     = float(prof.get("download_degradation", 0.0))
                ul_degrade     = float(prof.get("upload_degradation", 0.0))
            else:
                cap_dl = 200.0
                cap_ul = 15.0

                temp = hum = rain = wind = 0.0
                wet_ping = 40.0

                cap_penalty_dl = 0.0
                cap_penalty_ul = 0.0

                dl_degrade = 0.0
                ul_degrade = 0.0
            # Key point: SAT-GND do not adds delay within the mesh.
            base_delay_ms = 0.0

            G.add_edge(
                g,
                s,
                kind="SAT-GND",
                elevation_deg=float(elev),
                distance_km=float(dist_km),
                delay_ms=float(geom_delay_ms),      # check debug
                base_delay_ms=float(base_delay_ms), # = 0，no delay here
                base_capacity_down_Mbps=float(cap_dl),
                base_capacity_up_Mbps=float(cap_ul),
                weather_temp=float(temp),
                weather_humidity=float(hum),
                weather_rain=float(rain),
                weather_windspeed=float(wind),
                wet_ping_ms=float(wet_ping),
                wet_capacity_penalty_dl_Mbps=float(cap_penalty_dl),
                wet_capacity_penalty_ul_Mbps=float(cap_penalty_ul),
                wet_download_degradation=float(dl_degrade),
                wet_upload_degradation=float(ul_degrade),
            )
            num_added += 1

    print(
        f"[INFO] SAT-GND edges built ONLY via unified new logic: "
        f"min_elev={min_elev_deg}°, max_per_ground={max_per_ground}, total={num_added}"
    )

def compute_elevation(rg: np.ndarray, rs: np.ndarray) -> float:
    """
    Compute elevation angle (deg) of satellite (rs) as seen from ground (rg),
    both given in ECEF (km).

    rg: ground station ECEF position (km)
    rs: satellite ECEF position (km)
    return: elevation in degrees ( >0 means above local horizon )
    """
    rg = np.asarray(rg, dtype=float)
    rs = np.asarray(rs, dtype=float)

    u_r = rg / np.linalg.norm(rg)   # local "up" direction
    v = rs - rg                     # ground -> satellite
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return 90.0

    cos_zenith = np.dot(v, u_r) / (v_norm * np.linalg.norm(u_r))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)

    zenith_rad = np.arccos(cos_zenith)
    zenith_deg = np.rad2deg(zenith_rad)
    elev = 90.0 - zenith_deg
    return float(elev)


def eci_to_ecef(r_eci: np.ndarray, t: Time) -> np.ndarray:
    """
    Transforms the inertial frame ECI coordinates to the Earth-Fixed frame ECEF coordinates (unit: km).
    The IAU-76/FK5 style GMST approximation formula is used here, which is accurate enough for our routing and LoS determination.
    Parameters:
    r_eci : A NumPy vector of shape (3,) in km
    t : astropy.time.Time, UTC time
    Returns:
    r_ecef : A NumPy vector of shape (3,) in km
    """
    jd = float(t.jd)
    # Julian centuries since J2000.0
    T = (jd - 2451545.0) / 36525.0
    # GMST in degrees (IAU-76/FK5)
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T ** 3) / 38710000.0
    )
    gmst_rad = np.deg2rad(gmst_deg % 360.0)
    cosg = np.cos(gmst_rad)
    sing = np.sin(gmst_rad)
    x, y, z = r_eci
    x_ecef = cosg * x + sing * y
    y_ecef = -sing * x + cosg * y
    z_ecef = z
    return np.array([x_ecef, y_ecef, z_ecef], dtype=float)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """
    Convert geodetic coordinates (lat, lon, altitude) to ECEF XYZ, in km.

    lat_deg: latitude in degrees
    lon_deg: longitude in degrees
    alt_km: altitude above Earth surface in km
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    r = (R_EARTH_KM + alt_km)

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.array([x, y, z], dtype=float)

def haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg, R_earth_km=6371.0):
    """
    The distance (km) between two points on the Earth's surface on a great circle is consistent with the logic of haversine in MATLAB.。
    """
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R_earth_km * c

def has_los_vec(r1: np.ndarray,
                r2: np.ndarray,
                R_earth_km: float = R_EARTH_KM,
                margin_km: float = 0.0) -> bool:
    """
    Determine if a straight line segment from r1 to r2 "crosses the ground".
    r1, r2: 3D vectors (km), in the same coordinate system (approximately ECI/Geocentric coordinates).
    Approach:
    - Line segment: r(t) = r1 + t (r2 - r1), t ∈ [0,1]
    CAN NOT BE USE !!!!!!!!! ITS WRONG SYSTEM
    """
    d = r2 - r1  # 方向向量
    # t* 是原点到直线的投影参数，限制在 [0,1] 范围内
    denom = np.dot(d, d) + 1e-9
    t_star = -np.dot(r1, d) / denom
    t_star = np.clip(t_star, 0.0, 1.0)

    r_closest = r1 + t_star * d
    dist_center = np.linalg.norm(r_closest)

    return dist_center >= (R_earth_km + margin_km)


def sample_wetlink_profile(csv_path: str,
                           site_name: str | None = None,
                           rng: np.random.Generator | None = None,
                           t_obs: str | None = None):
    """
    Randomly sample one row from the specified csv_path,
    then subtract physical RTT baseline (LEO Ku/Ka + Frankfurt-to-server fiber)
    to extract congestion + processing delay.

    现在同时处理 download & upload：
    - download_Mbps：下行吞吐
    - upload_Mbps：上行吞吐
    """
    if rng is None:
        rng = np.random.default_rng()

    df = pd.read_csv(csv_path)

    # Site filter
    if t_obs is None:
        if site_name is not None and "site_name" in df.columns:
            df = df[df["site_name"] == site_name].copy()

        if df.empty:
            print(f"[WARN] Empty after site_name={site_name}, using full CSV.")
            df = pd.read_csv(csv_path)
    else:
        csv_path = "mean_out_rain/selected_24h_samples.csv"
        df = pd.read_csv(csv_path)
        df = df[df["timestamp_start"].str.slice(11, 13) == f"{t_obs.ymdhms.hour:02d}"].copy()

    # Cleaning
    df_clean = df.copy()
    if "humidity" in df_clean.columns:
        df_clean = df_clean[df_clean["humidity"].notna() & (df_clean["humidity"] >= 0)]
    if "temp" in df_clean.columns:
        df_clean = df_clean[df_clean["temp"].notna() & (df_clean["temp"] > -50)]
    if "windspeed" in df_clean.columns:
        df_clean = df_clean[df_clean["windspeed"].notna() & (df_clean["windspeed"] >= 0)]

    if "ping_avg" in df_clean.columns:
        df_clean = df_clean[df_clean["ping_avg"].notna() & (df_clean["ping_avg"] > 0)]
    if "download" in df_clean.columns:
        df_clean = df_clean[df_clean["download"].notna() & (df_clean["download"] > 0)]
    if "upload" in df_clean.columns:
        df_clean = df_clean[df_clean["upload"].notna() & (df_clean["upload"] > 0)]

    if df_clean.empty:
        print(f"[WARN] Cleaning removed all rows, reverting.")
        df_clean = df

    if df_clean.empty:
        print("[ERROR] Still empty—returning default.")
        return {
            "download_Mbps": 200.0,
            "upload_Mbps":   50.0,
            "ping_ms": 40.0,
            "temp": 10.0,
            "humidity": 80.0,
            "rain": 0.0,
            "windspeed": 1.0,
            "capacity_penalty_dl_Mbps": 0.0,
            "capacity_penalty_ul_Mbps": 0.0,
            "download_degradation": 0.0,
            "upload_degradation": 0.0,
        }

    # 抽样一行（你也可以用重试机制，这里先保持简单）

    idx = rng.integers(0, len(df_clean))
    row = df_clean.iloc[idx]

    # ---- ping ----
    if "ping_avg" in df_clean.columns:
        raw_ping = float(row["ping_avg"])
        if (not np.isfinite(raw_ping)) or (raw_ping <= 0):
            raw_ping = float(df_clean["ping_avg"].median())
    else:
        raw_ping = 60.0

    # ---- download ----
    if "download" in df_clean.columns:
        raw_dl = float(row["download"])
        if (not np.isfinite(raw_dl)) or (raw_dl <= 0):
            raw_dl = float(df_clean["download"].median())
        download_Mbps = raw_dl / 1e6
    else:
        download_Mbps = 200.0

    # ---- upload ----
    if "upload" in df_clean.columns:
        raw_ul = float(row["upload"])
        if (not np.isfinite(raw_ul)) or (raw_ul <= 0):
            raw_ul = float(df_clean["upload"].median())
        upload_Mbps = raw_ul / 1e6
    else:
        # 没有 upload 列就用一个保守值 / 或者等于 download
        upload_Mbps = min(download_Mbps, 50.0)

    # ---- subtract baseline RTT C ----
    if "Enschede" in csv_path or "ENS" in csv_path:
        C = 20.4  # ms
    elif "Osnabr" in csv_path or "OSN" in csv_path:
        C = 20.3 # ms
    else:
        C = 20.4

    ping_corrected = max(0.0, raw_ping - C)

    # ---- capacity penalty & degradation (上下行分别算) ----
    C_max_dl = 200.0  # 你原来的假设下行最大吞吐
    C_max_ul = 20.0   # 举例：上行典型上限可以比下行小很多，你可以改成别的值

    capacity_penalty_dl = max(0.0, C_max_dl - download_Mbps)
    capacity_penalty_ul = max(0.0, C_max_ul - upload_Mbps)

    download_degradation = 1.0 - download_Mbps / C_max_dl
    upload_degradation   = 1.0 - upload_Mbps   / C_max_ul

    download_degradation = min(max(download_degradation, 0.0), 1.0)
    upload_degradation   = min(max(upload_degradation,   0.0), 1.0)

    profile = {
        "download_Mbps": float(download_Mbps),
        "upload_Mbps":   float(upload_Mbps),

        "ping_ms":       float(ping_corrected),
        "temp":          float(row["temp"]) if "temp" in df_clean.columns and np.isfinite(row["temp"]) else 10.0,
        "humidity":      float(row["humidity"]) if "humidity" in df_clean.columns and np.isfinite(row["humidity"]) else 80.0,
        "rain":          float(row["rain"]) if "rain" in df_clean.columns and np.isfinite(row["rain"]) else 0.0,
        "windspeed":     float(row["windspeed"]) if "windspeed" in df_clean.columns and np.isfinite(row["windspeed"]) else 1.0,

        "capacity_penalty_dl_Mbps": capacity_penalty_dl,
        "capacity_penalty_ul_Mbps": capacity_penalty_ul,
        "download_degradation":  download_degradation,
        "upload_degradation":    upload_degradation,
    }

    return profile

def build_satellite_graph(
    tle_path: str,
    t_obs,
    max_sats: int = 66,
    isl_k: int = 4,
    isl_max_dist_km: float = 4500.0,
    ground_stations=None,
    sg_max_dist_km: float = 4000.0,
    wetlink_gs_map: dict[str, dict] | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Construct a topology G containing:
    - Starlink satellite nodes + ISL
    - Several ground city/GS nodes
    - SAT-GND edges (each city is connected to at least one satellite)
    - GROUND fiber optic edges (between any two cities)
    Common attributes in the graph:
    - node: kind = "sat" / "ground"
    - edge.kind = "ISL" / "SAT-GND" / "GROUND"
    - edge.base_delay_ms, edge.base_capacity_Mbps
    """
    # --- 0) Time processing ---
    if isinstance(t_obs, Time):
        t = t_obs
    else:
        t = Time(t_obs, scale="utc")

    if rng is None:
        rng = np.random.default_rng()

    # ground station
    if ground_stations is None:
        ground_stations = {
            "GS_ENS": (52.2215, 6.8937, 0.0),  # Enschede
            "GS_OSN": (52.2799, 8.0472, 0.0),  # Osnabrück
            "San Francisco, USA":      (37.7749,  -122.4194, 0.0),
            "New York City, USA":      (40.7128,   -74.0060, 0.0),
            "Paris, France":           (48.8566,     2.3522, 0.0),
            "Moscow, Russia":          (55.7558,    37.6173, 0.0),
            "Cape Town, South Africa": (-33.9249,   18.4241, 0.0),
            "Tokyo, Japan":            (35.6762,   139.6503, 0.0),
            "Sydney, Australia":       (-33.8688,  151.2093, 0.0),
            "Buenos Aires, Argentina": (-34.6037, -58.3816, 0.0),
            "Santiago, Chile": (-33.4489, -70.6693, 0.0),
        }

    
    G = nx.Graph()

    # ===================== 1) Construct satellite nodes =====================
    sats = load_tle_file(tle_path, max_sats=max_sats)

    sat_pos: dict[str, np.ndarray] = {}  # Satellite ECI coordinates
    def xyz_to_latlon(r: np.ndarray):
        """
        Simply convert the geocentric coordinates to approximate (lat, lon) values ​​for visualization/output purposes only; high precision is not required.
        """
        x, y, z = r
        r_norm = np.linalg.norm(r) + 1e-12
        lat = np.arcsin(z / r_norm)
        lon = np.arctan2(y, x)
        return np.rad2deg(lat), np.rad2deg(lon)

    for sat in sats:
        # Calculate the satellite's ECI coordinates based on the astropy time t.
        r_eci = sat.position_eci(t)
        if r_eci is None:
            # This satellite is not available on the current t_obs (possibly due to decay or being too far from the epoch), so we will skip it.
            continue

        # 2) A unified transformation to the Earth-Fixed Frame (ECEF) coordinate system is required to ensure a consistent coordinate system with ground nodes.
        r_ecef = eci_to_ecef(r_eci, t)
        sat_name = f"SAT_{sat.satnum}"

        # 3) The (lat, lon) and height are approximated using ECEF coordinates and are for visualization/recording purposes only.
        lat_deg, lon_deg = xyz_to_latlon(r_ecef)
        alt_km = np.linalg.norm(r_ecef) - R_EARTH_KM

        G.add_node(
            sat_name,
            kind="sat",
            lat=float(lat_deg),
            lon=float(lon_deg),
            alt_km=float(alt_km),
            pos_xyz=r_ecef,
        )
        sat_pos[sat_name] = r_ecef

    # ===================== 2) Constructing ground nodes =====================
    def ground_to_xyz(lat_deg, lon_deg, alt_km):
        R_E = R_EARTH_KM + alt_km
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        x = R_E * np.cos(lat) * np.cos(lon)
        y = R_E * np.cos(lat) * np.sin(lon)
        z = R_E * np.sin(lat)
        return np.array([x, y, z], dtype=float)

    gs_pos: dict[str, np.ndarray] = {}
    for name, (lat, lon, alt_km) in ground_stations.items():
        p = ground_to_xyz(lat, lon, alt_km)
        gs_pos[name] = p
        G.add_node(
            name,
            kind="ground",
            lat=lat,
            lon=lon,
            alt_km=alt_km,
            pos_xyz=p,
        )

    # ===================== 3) Generate WETLINK configuration for each ground station. =====================
    gs_profiles: dict[str, dict] = {}

    if wetlink_gs_map is not None and len(wetlink_gs_map) > 0:
        # 3.1 ENS/OSN, etc.: Sampling directly according to the configuration file
        for gs_name, cfg in wetlink_gs_map.items():
            csv_path = cfg.get("csv_path")
            site_name = cfg.get("site_name")
            if csv_path is None:
                continue
            profile = sample_wetlink_profile(
                csv_path=csv_path,
                site_name=site_name,
                rng=rng,
                t_obs=t_obs
            )
            gs_profiles[gs_name] = profile

        # 3.2 ther cities: Randomly select one from these CSV files to create a profile.
        cfg_list = [cfg for cfg in wetlink_gs_map.values() if cfg.get("csv_path") is not None]
        for gs_name in ground_stations.keys():
            if gs_name in gs_profiles:
                continue  # ENS/OSN already has a profile.
            if not cfg_list:
                # Defense: If wetlink_gs_map is empty, use the default profile.
                gs_profiles[gs_name] = {
                    "download_Mbps": 200.0,
                    "ping_ms": 40.0,
                    "temp": 10.0,
                    "humidity": 80.0,
                    "rain": 0.0,
                    "windspeed": 1.0,
                }
                continue

            cfg = cfg_list[rng.integers(0, len(cfg_list))]
            csv_path = cfg["csv_path"]
            site_name = cfg.get("site_name")
            profile = sample_wetlink_profile(
                csv_path=csv_path,
                site_name=site_name,
                rng=rng,
                t_obs=t_obs
            )
            gs_profiles[gs_name] = profile
    else:
        # No WETLINK data at all: giving all grounds a uniform default profile
        for gs_name in ground_stations.keys():
            gs_profiles[gs_name] = {
                "download_Mbps": 200.0,
                "ping_ms": 40.0,
                "temp": 10.0,
                "humidity": 80.0,
                "rain": 0.0,
                "windspeed": 1.0,
            }
            
    # === 3.x) give WETLINK profile to each ground node ===
    for gs_name, prof in gs_profiles.items():
        if gs_name not in G.nodes:
            continue
        G.nodes[gs_name]["wet_ping_ms"] = float(prof.get("ping_ms", 40.0))
        G.nodes[gs_name]["wet_download_Mbps"] = float(prof.get("download_Mbps", 200.0))
        G.nodes[gs_name]["wet_temp"] = float(prof.get("temp", 10.0))
        G.nodes[gs_name]["wet_humidity"] = float(prof.get("humidity", 80.0))
        G.nodes[gs_name]["wet_rain"] = float(prof.get("rain", 0.0))
        G.nodes[gs_name]["wet_windspeed"] = float(prof.get("windspeed", 1.0))

    # ===================== 4) Building ISL (Satellite-to-Satellite) =====================
    sat_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "sat"]
    isl_los_margin_km = 50.0

    for u in sat_nodes:
        ru = sat_pos[u]
        # Find the nearest isl_k neighboring satellites
        dists = []
        for v in sat_nodes:
            if u == v:
                continue
            rv = sat_pos[v]
            dist = float(np.linalg.norm(ru - rv))
            dists.append((v, dist))

        dists.sort(key=lambda x: x[1])

        for v, dist in dists[:isl_k]:
            rv = sat_pos[v]
            if G.has_edge(u, v):
                continue
            # The line segment [ru -> rv] does not cross the ground.
            if dist > isl_max_dist_km:
                continue

            delay_ms = dist/ C_KM_PER_MS
            G.add_edge(
                u,
                v,
                kind="ISL",
                length_km=float(dist),
                base_delay_ms=float(delay_ms),
                base_capacity_Mbps=10000.0,
                weather_temp=0.0,
                weather_humidity=0.0,
                weather_rain=0.0,
                weather_windspeed=0.0,
            )

    # ===================== 5) Constructing SAT-GND (Satellite-Ground) =====================
    # Main logic: LOS + elevation + first K ground planes
    add_sat_ground_edges_with_elev(
        G,
        min_elev_deg=10.0,          # elevation min
        max_per_ground=5,           # maximum number of satellites each ground station can connect
        c_km_per_ms=C_KM_PER_MS,
        sg_max_dist_km=sg_max_dist_km,  # distance
        gs_profiles=gs_profiles,    #  WETLINK profile 
    )

    # ===================== 5.x) DEBUG: print SAT-GND neighbors per ground =====================
    for g, d in G.nodes(data=True):
        if d.get("kind") != "ground":
            continue
        sats = []
        for nbr in G.neighbors(g):
            ed = G[g][nbr]
            if ed.get("kind") == "SAT-GND":
                sats.append((
                    nbr,
                    ed.get("elevation_deg"),
                    ed.get("distance_km")
                ))
        sats.sort(key=lambda x: -x[1] if x[1] is not None else -999)

        print(f"\n[DEBUG] Ground {g} has {len(sats)} SAT-GND edges:")
        for (sat, elev, dist) in sats:
            print(f"  -> {sat}: elev={elev:.2f}°, dist={dist:.1f} km")

    # ===================== 6) Building a GROUND fiber edge=====================
    POP_NODES = {"GS_ENS", "GS_OSN"}
    k_pop_links = 3
    pop_fixed_neighbors = None  # 也可以做成 dict: {"GS_ENS":[...], "GS_OSN":[...]}

    all_ground = [n for n, d in G.nodes(data=True) if d.get("kind") == "ground"]
    pop_nodes  = [n for n in all_ground if n in POP_NODES]
    city_nodes = [n for n in all_ground if n not in POP_NODES]

    def add_ground_edge(u, v, dist_km, cap=10000.0):
        delay_ms = 0.018 * dist_km + 20.0
        # 防御：不要覆盖非 ground 的边
        if G.has_edge(u, v) and G[u][v].get("kind") != "GROUND":
            return
        G.add_edge(
            u, v,
            kind="GROUND",
            length_km=float(dist_km),
            base_delay_ms=float(delay_ms),
            base_capacity_Mbps=float(cap),
            weather_temp=0.0,
            weather_humidity=0.0,
            weather_rain=0.0,
            weather_windspeed=0.0,
        )

    # (a) 城市全连接
    for i in range(len(city_nodes)):
        for j in range(i + 1, len(city_nodes)):
            u, v = city_nodes[i], city_nodes[j]
            lat1, lon1 = float(G.nodes[u]["lat"]), float(G.nodes[u]["lon"])
            lat2, lon2 = float(G.nodes[v]["lat"]), float(G.nodes[v]["lon"])
            dist_km = haversine_km(lat1, lon1, lat2, lon2, R_EARTH_KM)
            add_ground_edge(u, v, dist_km, cap=1000.0)

    # (b) POP 只连最近 k 个城市
    for pop in pop_nodes:
        pop_lat = float(G.nodes[pop]["lat"])
        pop_lon = float(G.nodes[pop]["lon"])

        if pop_fixed_neighbors is not None:
            cand_cities = [c for c in pop_fixed_neighbors if c in city_nodes]
        else:
            dlist = []
            for c in city_nodes:
                latc, lonc = float(G.nodes[c]["lat"]), float(G.nodes[c]["lon"])
                dkm = haversine_km(pop_lat, pop_lon, latc, lonc, R_EARTH_KM)
                dlist.append((c, dkm))
            dlist.sort(key=lambda x: x[1])
            cand_cities = [c for c, _ in dlist[:k_pop_links]]

        for c in cand_cities:
            latc, lonc = float(G.nodes[c]["lat"]), float(G.nodes[c]["lon"])
            dist_km = haversine_km(pop_lat, pop_lon, latc, lonc, R_EARTH_KM)
            add_ground_edge(pop, c, dist_km, cap=10000.0)

    # (c) ENS-OSN 直连 + 写入 graph baseline（用同一个 delay 模型！）
    if ("GS_ENS" in G.nodes) and ("GS_OSN" in G.nodes):
        lat1, lon1 = float(G.nodes["GS_ENS"]["lat"]), float(G.nodes["GS_ENS"]["lon"])
        lat2, lon2 = float(G.nodes["GS_OSN"]["lat"]), float(G.nodes["GS_OSN"]["lon"])
        dist_km = haversine_km(lat1, lon1, lat2, lon2, R_EARTH_KM)

        delay_ms = 0.018 * dist_km + 20.0
        add_ground_edge("GS_ENS", "GS_OSN", dist_km, cap=10000.0)

        G.graph["ground_dist_km"] = float(dist_km)
        G.graph["ground_fiber_delay_ms"] = float(delay_ms)   # ✅ 和边一致

    num_ground_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "GROUND")
    print(f"[INFO] Scheme-2 GROUND edges: cities={len(city_nodes)}, pops={pop_nodes}, total_ground_edges={num_ground_edges}")
    print("[DBG] unique edge kinds:", sorted({d.get("kind") for _,_,d in G.edges(data=True)}))
    print("[DBG] num GROUND edges:", sum(1 for _,_,d in G.edges(data=True) if d.get("kind")=="GROUND"))
    print("[DBG] num SAT-GND edges:", sum(1 for _,_,d in G.edges(data=True) if d.get("kind")=="SAT-GND"))
    print("[DBG] num ISL edges:", sum(1 for _,_,d in G.edges(data=True) if d.get("kind")=="ISL"))
    
    return G


