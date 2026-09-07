import networkx as nx
import numpy as np
import random
from topology_builder import build_satellite_graph, haversine_km, has_los_vec, R_EARTH_KM
from astropy.time import Time
import json
import logging

C_KM_PER_MS = 299792.458 / 1000.0  # light speed (km/ms)
V_FIBER_KM_PER_MS = (2.0 / 3.0) * C_KM_PER_MS  # fiber ≈ 2/3 c
class SatelliteGraphEnv(object):
    """
    Starlink+WETLINK Environment:
    - Each episode calls `build_satellite_graph(...)` to generate a new satellite + dual ground station topology G.
    - Edges in G contain:
    * kind: "ISL" / "SAT-GND"
    * base_delay_ms
    * base_capacity_Mbps
    * weather_*
    - State: `graph_state[e,0] = remaining_down_capacity
    graph_state[e,1] = remaining_up_capacity
    graph_state[e,2] = used_flag
    - Action: Select 1 path from K candidate satellite paths + 1 ground path
    """
    print("[DEBUG] SatelliteGraphEnv __init__ from NEW FILE is used")
    def __init__(
        self,
        tle_path: str,
        en_csv: str,
        osn_csv: str,
        listofDemands,
        K_paths: int = 4,
        sg_max_dist_km: float = 4000.0,
        isl_k: int = 4,
        epoch=None,
        isl_max_dist_km: float = 4500.0,
        requests_per_episode: int = 20,     # 每个 episode 多少个 request
        shuffle_requests: bool = True,      # `` 是否打乱每轮 request 顺序
        enable_weather: bool = False,
    ):
        super().__init__()
        
        self.tle_path = tle_path
        self.en_csv = en_csv
        self.osn_csv = osn_csv

        self.listofDemands = listofDemands
        self.max_demand = max(listofDemands)
        self.epoch = epoch

        # Source and destination: Fixed CPE
        self.src_gs = "GS_ENS"
        self.dst_gs = "GS_OSN"
        # self.long_pairs = [
        #     ("Cape Town, South Africa", "Buenos Aires, Argentina"),
        #     ("Cape Town, South Africa", "Santiago, Chile"),

        #     # Moscow 出发
        #     ("Moscow, Russia", "Buenos Aires, Argentina"),
        #     ("Moscow, Russia", "Cape Town, South Africa"),
        #     ("Moscow, Russia", "Santiago, Chile"),
        #     ("Moscow, Russia", "Sydney, Australia"),
        #     ("Moscow, Russia", "Tokyo, Japan"),

        #     # New York 出发
        #     ("New York City, USA", "Buenos Aires, Argentina"),
        #     ("New York City, USA", "Cape Town, South Africa"),
        #     ("New York City, USA", "Moscow, Russia"),
        #     ("New York City, USA", "Paris, France"),
        #     ("New York City, USA", "San Francisco, USA"),
        #     ("New York City, USA", "Santiago, Chile"),
        #     ("New York City, USA", "Sydney, Australia"),
        #     ("New York City, USA", "Tokyo, Japan"),

        #     # Paris 出发
        #     ("Paris, France", "Buenos Aires, Argentina"),
        #     ("Paris, France", "Cape Town, South Africa"),
        #     ("Paris, France", "Santiago, Chile"),
        #     ("Paris, France", "Sydney, Australia"),
        #     ("Paris, France", "Tokyo, Japan"),

        #     # San Francisco 出发
        #     ("San Francisco, USA", "Buenos Aires, Argentina"),
        #     ("San Francisco, USA", "Cape Town, South Africa"),
        #     ("San Francisco, USA", "Moscow, Russia"),
        #     ("San Francisco, USA", "Paris, France"),
        #     ("San Francisco, USA", "Santiago, Chile"),
        #     ("San Francisco, USA", "Sydney, Australia"),
        #     ("San Francisco, USA", "Tokyo, Japan"),

        #     # Sydney 出发
        #     ("Sydney, Australia", "Buenos Aires, Argentina"),
        #     ("Sydney, Australia", "Cape Town, South Africa"),
        #     ("Sydney, Australia", "Santiago, Chile"),

        #     # Tokyo 出发
        #     ("Tokyo, Japan", "Buenos Aires, Argentina"),
        #     ("Tokyo, Japan", "Cape Town, South Africa"),
        #     ("Tokyo, Japan", "Santiago, Chile"),
        #     ("Tokyo, Japan", "Sydney, Australia"),
        # ]

        
        self.k_paths = K_paths
        self.k_paths = int(K_paths)
        self.sg_max_dist_km = sg_max_dist_km
        self.isl_k = isl_k
        self.isl_max_dist_km = isl_max_dist_km

        self.tle_start_time = Time("2025-12-05 00:00:00", scale="utc")
        self.tle_end_time   = Time("2025-12-05 12:00:00", scale="utc")
        # RNG（给 WETLINK 抽样用）
        self._rng = np.random.default_rng()

        # 占位：下面 reset() / _build_graph_and_features() 会真正填充
        self.G = None
        self.node_index = {}
        self.index_node = {}
        self.numNodes = 0
        self.ordered_edges = []
        self.numEdges = 0
        self.edgesDict = {}


        self.graph_state = None
        self.first = None
        self.second = None
        self.firstTrueSize = 0

        self.link_delay = None
        self.link_weather = None
        self.link_type_onehot = None
        self.between_feature = None
        self.max_delay = 1.0
        self.max_capacity = 1.0
        self.weather_mean = 0.0
        self.weather_std = 1.0

        self.allPaths = {}

        self.step_count = 0
        self.max_steps_per_episode = 20

        

        self.current_src = None
        self.current_dst = None
        self.current_demand = None

        # 初始化一次（也可以等第一次 reset 时再建图）
        self.wetlink_gs_map = {
            "GS_ENS": {
                "csv_path": en_csv,
                "site_name": "utwente",  # 如果 Enschede CSV 里 site_name 是这个
            },
            "GS_OSN": {
                "csv_path": osn_csv,
                "site_name": None,       # 如果 Osnabrück 没有 site_name 列就设 None
            },
        }
        self.long_pairs = []
        # 业务端点的最小大圆距离（km）
        self.request_min_dist_km = 3500

        self.max_sats = 66

        # 地面链路吞吐：写死一个较大值，后面 reward 会权衡 delay vs capacity
        self.ground_throughput = 1000.0

        self.city_coords = {
            "New York City, USA": (40.7128, -74.0060, 0.0),
            "San Francisco, USA": (37.7749, -122.4194, 0.0),
            "Paris, France": (48.8566, 2.3522, 0.0),
            "Moscow, Russia": (55.7558, 37.6173, 0.0),
            "Tokyo, Japan": (35.6895, 139.6917, 0.0),
            "Sydney, Australia": (-33.8688, 151.2093, 0.0),
            "Cape Town, South Africa": (-33.9249, 18.4241, 0.0),
            "Buenos Aires, Argentina":(-34.6037, -58.3816, 0.0),
            "Santiago, Chile": (-33.4489, -70.6693, 0.0)
        }

        # 2) ENS/OSN 作为 Starlink POP
        self.pop_coords = {
            "GS_ENS": (52.2215, 6.8937, 0.0),
            "GS_OSN": (52.2799, 8.0472, 0.0),
        }

        self.city_to_gw = {
        "New York City, USA": {
            "gw_name": "GW_Beekmantown_US",
            "gw_lat": 44.77087,
            "gw_lon": -73.49208,
            "ground_ow_delay_ms": 20.8118,   # distance ≈ 451 km
        },
        "San Francisco, USA": {
            "gw_name": "GW_Arbuckle_US",
            "gw_lat": 39.01750,
            "gw_lon": -122.05778,
            "ground_ow_delay_ms": 20.2551,   # distance ≈ 141.7 km
        },
        "Paris, France": {
            "gw_name": "GW_Villenave_FR",
            "gw_lat": 44.77998,
            "gw_lon": -0.56726,
            "ground_ow_delay_ms": 20.9085,   # distance ≈ 504.7 km
        },
        "Moscow, Russia": {
            "gw_name": "GW_Usingen_DE",
            "gw_lat": 50.33694,
            "gw_lon": 8.52917,
            "ground_ow_delay_ms": 23.6324,  # distance ≈ 2018 km
        },
        "Tokyo, Japan": {
            "gw_name": "GW_Hitachinaka_JP",
            "gw_lat": 36.39661,
            "gw_lon": 140.53468,
            "ground_ow_delay_ms": 20.1966,   # distance ≈ 109.2 km
        },
        "Sydney, Australia": {
            "gw_name": "GW_Boorowa_AU",
            "gw_lat": -34.43661,
            "gw_lon": 148.71634,
            "ground_ow_delay_ms": 20.4284,   # distance ≈ 238 km
        },
        "Cape Town, South Africa": {
            "gw_name": "GW_FaldaDelCarmen_AR",
            "gw_lat": -31.58598,
            "gw_lon": -64.45868,
            "ground_ow_delay_ms": 33.545,  # distance ≈ 7525 km
        },
        "Buenos Aires, Argentina": {
            "gw_name": "GW_FaldaDelCarmen_AR",
            "gw_lat": -31.58598,
            "gw_lon": -64.45868,
            "ground_ow_delay_ms": 21.1844,   # distance ≈ 658 km
        },
        "Santiago, Chile": {
            "gw_name": "GW_Noviciado_CL",
            "gw_lat": -33.40253,
            "gw_lon": -70.85530,
            "ground_ow_delay_ms": 20.0324,   # distance ≈ 18 km
    },
}

        self.requests_per_episode = requests_per_episode   # ← 用参数
        self.shuffle_requests = shuffle_requests           # ← 用参数
        self.requests = []               # 当前 episode 的 request 列表
        self.req_idx = 0                 # 当前是第几个 request
        self.base_capacity = None   # 每条边的初始总容量
        self.total_delay = 0.0      # 本 episode 已累积的 e2e delay 之和
        self.verbose = True          # 打印开关，想关掉 log 就设 False
        self.episode_logs = []       # 每个 episode 的 request 记录列表
        self.access_delay = {}
        # 初始化一次图（也可以等第一次 reset 时再建图）

        self.delay_norm = 30.0          # 原本你就有一个常数，这里显式写出来
        self.lambda_path_util   = 0.5
        self.lambda_global_util = 5.0
        self.lambda_sat_bonus   = 0.5
        self.enable_weather = bool(enable_weather)
        self.fixed_requests = None
        self.fixed_shuffle = False
        self.use_fixed_requests: bool = False

        self.log = logging.getLogger("sat_env")
        if not self.log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(h)
        # 默认 INFO：terminal 不爆；需要细查时你再临时改 DEBUG
        self.log.setLevel(logging.INFO)

    def _p(self, *args, **kwargs):
    # 控制所有原 print
        if getattr(self, "verbose", False) and self.log.level <= logging.DEBUG:
            print(*args, **kwargs)
        
    def set_fixed_requests(self, requests, shuffle=False):
        """
        requests: list[dict]
        每个 dict 至少包含：
            src_name, dst_name, demand, ground_delay_ms
        """
        if requests is None:
            self.fixed_requests = None
            self.fixed_shuffle = False
            self.use_fixed_requests = False
            return

        self.fixed_requests = list(requests)
        self.fixed_shuffle = bool(shuffle)
        self.use_fixed_requests = True

    def load_requests_jsonl(self, path, shuffle=False):
        """
        从 JSONL 文件读取固定 requests，然后调用 set_fixed_requests()
        JSONL: 每行一个 dict
        """
        reqs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                reqs.append(json.loads(line))

        # ✅ 融合点：load 只是“读文件”，真正存到 env 里由 set_fixed_requests 做
        self.set_fixed_requests(reqs, shuffle=shuffle)
        return reqs


    def _build_new_graph(self):
        """随机一个时间, 根据 TLE + WETLINK 构建新的 NetworkX 图."""
        # 随机时间
        if self.epoch is not None:
            t_obs = self.epoch
        else:
            alpha = float(self._rng.random())
            t_obs = self.tle_start_time + (self.tle_end_time - self.tle_start_time) * alpha

        # 两个气象站分别使用各自的 CSV + site_name
        

        ground_stations = {}
        ground_stations.update(self.pop_coords)
        ground_stations.update(self.city_coords)

        self.G = build_satellite_graph(
            t_obs=t_obs,                # 你现在用的观测时间
            tle_path=self.tle_path,
            max_sats=self.max_sats,
            ground_stations=ground_stations, # ← 新增这一行
            isl_max_dist_km=self.isl_max_dist_km,
            sg_max_dist_km=self.sg_max_dist_km,
            wetlink_gs_map=self.wetlink_gs_map,
        )
        self.access_delay = {}
        for n, d in self.G.nodes(data=True):
            if d.get("kind") == "ground":
                self.access_delay[n] = float(d.get("access_delay_ms", 0.0))

        print("=== Graph basic info ===")
        print("num_nodes:", len(self.G.nodes()))
        print("num_edges:", len(self.G.edges()))
        num_sat_nodes = sum(1 for _, d in self.G.nodes(data=True) if d.get("kind") == "sat")
        num_gnd_nodes = sum(1 for _, d in self.G.nodes(data=True) if d.get("kind") == "ground")
        num_isl_edges = sum(1 for _, _, d in self.G.edges(data=True) if d.get("kind") == "ISL")
        num_sg_edges  = sum(1 for _, _, d in self.G.edges(data=True) if d.get("kind") == "SAT-GND")
        print("num_sat_nodes:", num_sat_nodes)
        print("num_ground_nodes:", num_gnd_nodes)
        print("num_ISL_edges:", num_isl_edges)
        print("num_SAT_GND_edges:", num_sg_edges)
        print("ground_fiber_delay_ms (ENS-OSN fiber):", self.G.graph.get("ground_fiber_delay_ms"))

        # ==== 调试：打印一些 pos_xyz 和 LOS 检查 ====
        if self.verbose:
            # 1) 打印一个卫星节点和一个地面节点的 pos_xyz
            sat_nodes = [n for n, d in self.G.nodes(data=True) if d.get("kind") == "sat"]
            gnd_nodes = [n for n, d in self.G.nodes(data=True) if d.get("kind") == "ground"]

            if sat_nodes and gnd_nodes:
                sat = sat_nodes[0]
                gnd = gnd_nodes[0]
                rs = np.array(self.G.nodes[sat]["pos_xyz"], dtype=float)
                rg = np.array(self.G.nodes[gnd]["pos_xyz"], dtype=float)

                print("\n[DEBUG] Example SAT / GND pos_xyz (ECEF, km):")
                print(f"  {sat}: {rs}, |rs|={np.linalg.norm(rs):.2f} km")
                print(f"  {gnd}: {rg}, |rg|={np.linalg.norm(rg):.2f} km")

                # 2) 用 has_los_vec 再做一次 LOS 检查
                los_ok = has_los_vec(rg, rs, R_earth_km=R_EARTH_KM, margin_km=0.0)
                print(f"  LOS between {gnd} and {sat}: {los_ok}")

                # 3) 手动算一下这个卫星–地面连线离地心的最小距离，验证“不穿地”
                d = rs - rg
                denom = np.dot(d, d) + 1e-9
                t_star = -np.dot(rg, d) / denom
                t_star = np.clip(t_star, 0.0, 1.0)
                r_closest = rg + t_star * d
                dist_center = np.linalg.norm(r_closest)
                print(f"  min distance to Earth center along segment: {dist_center:.2f} km")
                print(f"  Earth radius: {R_EARTH_KM:.2f} km\n")

                
        # 2.5) 从图属性里拿地面光纤时延（你之前在 topology_builder 里已经加了）
        self.ground_delay = self.G.graph.get("ground_fiber_delay_ms", 60.0)

        self.node_index = {n: i for i, n in enumerate(self.G.nodes())}
        self.index_node = {i: n for n, i in self.node_index.items()}
        self.numNodes = len(self.G.nodes())

        # 4) 边重编号，生成 ordered_edges = [(i,j), ...]，i,j 是 int index
        some_edges = [
            tuple(sorted((self.node_index[u], self.node_index[v])))
            for u, v in self.G.edges()
        ]
        self.ordered_edges = sorted(some_edges)
        self.numEdges = len(self.ordered_edges)

        self.edgesDict = {}
        for pos, (i, j) in enumerate(self.ordered_edges):
            self.edgesDict[f"{i}:{j}"] = pos
            self.edgesDict[f"{j}:{i}"] = pos

        # 5) 分配特征数组
        # graph_state[:,0] = remaining_down
        # graph_state[:,1] = remaining_up
        # graph_state[:,2] = used_flag
        self.graph_state = np.zeros((self.numEdges, 3), dtype=np.float32)
        self.first = []
        self.second = []

        self.link_delay = np.zeros(self.numEdges, dtype=np.float32)
        self.link_weather = np.zeros(self.numEdges, dtype=np.float32)
        self.link_type_onehot = np.zeros((self.numEdges, 3), dtype=np.float32)
        self.between_feature = np.zeros(self.numEdges, dtype=np.float32)
        self.link_cap_penalty = np.zeros(self.numEdges, dtype=np.float32)
        self.link_download_deg = np.zeros(self.numEdges, dtype=np.float32)
        self.link_upload_deg   = np.zeros(self.numEdges, dtype=np.float32)

        # 方向性容量：down / up
        self.base_cap_down = np.zeros(self.numEdges, dtype=np.float32)
        self.base_cap_up   = np.zeros(self.numEdges, dtype=np.float32)

        # 6) 填 edgesDict + 初始容量 + 基础 delay + weather + type
        for pos, (i, j) in enumerate(self.ordered_edges):
            u = self.index_node[i]
            v = self.index_node[j]
            edata = self.G.get_edge_data(u, v)

            delay = edata.get("base_delay_ms", 50.0)
            kind = edata.get("kind", None)
            kind = str(kind).upper() if kind is not None else "UNKNOWN"

            # === 根据 kind 拿容量 ===
            cap_dl = None
            cap_ul = None

            if kind == "SAT-GND":
                # 这两项必须存在，否则说明 build_satellite_graph 没注入
                if ("base_capacity_down_Mbps" not in edata) or ("base_capacity_up_Mbps" not in edata):
                    raise RuntimeError(
                        f"[FATAL] SAT-GND edge missing UL/DL capacity: {u}<->{v}, keys={list(edata.keys())}"
                    )
                cap_dl = float(edata["base_capacity_down_Mbps"])
                cap_ul = float(edata["base_capacity_up_Mbps"])
                #cap_ul = max(cap_ul, 10.0)  # MIN_UL

            elif kind == "ISL":
                cap = float(edata.get("base_capacity_Mbps", 10000.0))
                cap_dl = cap
                cap_ul = cap

            else:
                # ground / fiber / city / pop 等边，统一走这里
                cap = float(edata.get("base_capacity_Mbps", 10000.0))
                cap_dl = cap
                cap_ul = cap

            if kind == "UNKNOWN":
                print("[WARN] edge kind missing:", u, v, "edata_keys=", list(edata.keys()))

            # 最后防御
            if cap_dl is None or cap_ul is None:
                raise RuntimeError(f"[FATAL] cap no`t assigned for edge {u}<->{v}, kind={kind}, keys={list(edata.keys())}")

            if kind == "SAT-GND" and (u == "San Francisco, USA" and v == "SAT_63" or u == "SAT_63" and v == "San Francisco, USA"):
                print("[CHECK] SAT-GND edge", u, "<->", v)
                print("  base_capacity_down_Mbps =", edata.get("base_capacity_down_Mbps", None))
                print("  base_capacity_up_Mbps   =", edata.get("base_capacity_up_Mbps", None))
                print("  (after MIN_UL) cap_ul   =", cap_ul)
                print("  wet_upload_degradation  =", edata.get("wet_upload_degradation", None))
                print("  wet_capacity_penalty_ul =", edata.get("wet_capacity_penalty_ul_Mbps", None))

            # 存到 base_cap_* 和 graph_state
            self.base_cap_down[pos] = float(cap_dl)
            self.base_cap_up[pos]   = float(cap_ul)

            self.graph_state[pos, 0] = float(cap_dl)   # remaining_down
            self.graph_state[pos, 1] = float(cap_ul)   # remaining_up
            self.graph_state[pos, 2] = 0.0             # used_flag（先全 0）

            # 你现在 link_cap_penalty 是一维数组：先存 DL penalty（最常用）
            self.link_cap_penalty[pos]  = float(edata.get("wet_capacity_penalty_dl_Mbps", 0.0))
            self.link_download_deg[pos] = float(edata.get("wet_download_degradation", 0.0))
            self.link_upload_deg[pos]   = float(edata.get("wet_upload_degradation", 0.0))
            self.link_weather[pos]      = float(edata.get("weather_rain", 0.0))

            self.link_delay[pos] = float(delay)

            # link_type_onehot: [ground, ISL, sat-gnd]
            if kind == "ISL":
                self.link_type_onehot[pos] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif kind == "SAT-GND":
                self.link_type_onehot[pos] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                self.link_type_onehot[pos] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        if self.enable_weather:
            base_down = self.base_cap_down.astype(float)
            base_up   = self.base_cap_up.astype(float)

            dl_deg = np.nan_to_num(self.link_download_deg.astype(float), nan=0.0)
            ul_deg = np.nan_to_num(self.link_upload_deg.astype(float), nan=0.0)

            down_eff = base_down * np.clip(1.0 - dl_deg, 0.0, 1.0)
            up_eff   = base_up   * np.clip(1.0 - ul_deg, 0.0, 1.0)

            # 给个下限（可调），防止极端天气直接变 0
            down_floor = 0.5 * base_down
            up_floor   = 0.5 * base_up
            down_eff = np.maximum(down_eff, down_floor)
            up_eff   = np.maximum(up_eff,   up_floor)

            self.graph_state[:, 0] = down_eff
            self.graph_state[:, 1] = up_eff
        else:
            # 训练时：不做雨衰，容量=原始容量（但 wet 元数据仍然保留在 link_* 数组）
            self.graph_state[:, 0] = self.base_cap_down.copy()
            self.graph_state[:, 1] = self.base_cap_up.copy()

        # 利用率那边就用 down 作为 reference（简单一点）
        self.eff_cap_down = self.graph_state[:, 0].astype(np.float32).copy()
        self.eff_cap_up   = self.graph_state[:, 1].astype(np.float32).copy()

        self.base_capacity = self.base_cap_down.copy()
        self.max_delay = 200
        self.max_capacity = float(np.max(self.base_cap_down)) if self.numEdges > 0 else 1.0

        # weather_scalar 可以基于 rain/其他气象组合，现在简单用雨量
        self.weather_mean = float(np.mean(self.link_weather)) if self.numEdges > 0 else 0.0
        self.weather_std = float(np.std(self.link_weather) + 1e-6)

        # 7) 计算 edge betweenness，作为 GNN 边特征
        if self.verbose:
            bet_dict = nx.edge_betweenness_centrality(self.G, normalized=True)
        else:
            bet_dict = None

        for pos, (i, j) in enumerate(self.ordered_edges):
            if bet_dict is None:
                self.between_feature[pos] = 0.0
            else:
                u = self.index_node[i]
                v = self.index_node[j]
                b = bet_dict.get((u, v), bet_dict.get((v, u), 0.0))
                self.between_feature[pos] = float(b)

        # 8) 构造 edge-edge 邻接 (first, second)：两条边如果共享一个节点，就互为邻居
        edge_to_pos = { (i, j): p for p, (i, j) in enumerate(self.ordered_edges) }
        for p, (i, j) in enumerate(self.ordered_edges):
            # 以 i 为中心的邻边
            for (m, n) in self._neighbour_edges(i):
                if (m, n) == (i, j):
                    continue
                q = edge_to_pos.get(tuple(sorted((m, n))))
                if q is not None:
                    self.first.append(p)
                    self.second.append(q)
            # 以 j 为中心的邻边
            for (m, n) in self._neighbour_edges(j):
                if (m, n) == (i, j):
                    continue
                q = edge_to_pos.get(tuple(sorted((m, n))))
                if q is not None:
                    self.first.append(p)
                    self.second.append(q)

        self.first = np.array(self.first, dtype=np.int32)
        self.second = np.array(self.second, dtype=np.int32)
        self.firstTrueSize = len(self.first)

        # 9) 预计算 src,dst 之间的 K 条卫星路径 + GROUND
        self.initial_state = np.copy(self.graph_state)

        # 10) 统计所有“可作为业务端点”的长城市对（排除 GS_ENS / GS_OSN）
        ground_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("kind") == "ground" and n not in ("GS_ENS", "GS_OSN")
        ]

        long_pairs = []
        for i in range(len(ground_nodes)):
            for j in range(i + 1, len(ground_nodes)):
                u = ground_nodes[i]
                v = ground_nodes[j]
                nu = self.G.nodes[u]
                nv = self.G.nodes[v]
                lat1, lon1 = nu.get("lat"), nu.get("lon")
                lat2, lon2 = nv.get("lat"), nv.get("lon")
                if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                    continue
                dist_km = haversine_km(lat1, lon1, lat2, lon2) 
                if dist_km >= self.request_min_dist_km:
                    long_pairs.append((u, v))

        if not long_pairs:
            print(
                f"[WARN] _build_new_graph: no long city pairs (>{self.request_min_dist_km} km) "
                f"found among ground nodes. Fallback to ENS-OSN only."
            )

        self.long_pairs = long_pairs

        # === 额外：给每条边写一个 hybrid_weight，用于“混合卫星+地面”的路径搜索 ===
        self.ground_penalty_factor = 5.0  # 你可以调大/调小，比如 3~10 都可以试

        for u, v, d in self.G.edges(data=True):
            if "length_km" in d:
                base_len = d["length_km"]
            elif "distance_km" in d:
                base_len = d["distance_km"]
            else:
                base_len = 1.0

            kind = d.get("kind", "")
            kind_u = str(kind).upper()

            if kind_u not in ("ISL", "SAT-GND"):
                d["hybrid_weight"] = float(base_len * self.ground_penalty_factor)
            else:
                d["hybrid_weight"] = float(base_len)
        # --- DBG: ground edges sanity check ---
        gnd_edges = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if str(d.get("kind","")).upper() not in ("ISL", "SAT-GND")
        ]
        if len(gnd_edges) == 0:
            print("[BUG][GROUND] No ground-like edges in graph! _compute_ground_path() will always fail.")
        else:
            print(f"[DBG][GROUND] num_ground_like_edges={len(gnd_edges)}")

        print("[DBG] unique edge kinds:", sorted({d.get("kind") for _,_,d in self.G.edges(data=True)}))
        print("[DBG] num GROUND edges:", sum(1 for _,_,d in self.G.edges(data=True) if str(d.get("kind","")).upper() not in ("ISL","SAT-GND")))
              
        for u, v, d in self.G.edges(data=True):
            if d.get("kind") == "SAT-GND":
                print("[SANITY] one SAT-GND edge wet fields:",
                    "pen_dl=", d.get("wet_capacity_penalty_dl_Mbps"),
                    "pen_ul=", d.get("wet_capacity_penalty_ul_Mbps"),
                    "dl_deg=", d.get("wet_download_degradation"),
                    "ul_deg=", d.get("wet_upload_degradation"))
                break


    def _neighbour_edges(self, node_idx: int):
        node_name = self.index_node[node_idx]
        for u, v in self.G.edges(node_name):
            iu = self.node_index[u]
            iv = self.node_index[v]
            yield tuple(sorted((iu, iv)))

    def _compute_k_paths(self, src_idx: int, dst_idx: int, k: int):
        """
        只用卫星相关边 (ISL + SAT-GND) 计算 src->dst 的 K 条候选卫星路径。
        如果 src/dst 在卫星子图 H 里不存在，或者没有路，返回 []。
        """
        src_name = self.index_node[src_idx]
        dst_name = self.index_node[dst_idx]

        paths = []

        try:
            # 在整张 G 上，用 hybrid_weight 做权重
            gen = nx.shortest_simple_paths(
                self.G,
                source=src_name,
                target=dst_name,
                weight="hybrid_weight"
            )

            for path_nodes in gen:
                # 判断这条路径里是否使用了至少一个 SAT-GND 或 ISL 边
                has_sat_edge = False
                for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                    kind = self.G[u][v].get("kind", "ISL")
                    if kind in ("ISL", "SAT-GND"):
                        has_sat_edge = True
                        break

                # 如果一条路径完全是 GROUND 边，就把它留给纯 GROUND action，用不到这里
                if not has_sat_edge:
                    continue

                # 转成 index 列表
                path_idx = [self.node_index[n] for n in path_nodes]
                paths.append(path_idx)

                if len(paths) >= k:
                    break

        except nx.NetworkXNoPath:
            pass

        return paths
    
    def _compute_ground_path(self, src_idx: int, dst_idx: int):
        src_name = self.index_node[src_idx]
        dst_name = self.index_node[dst_idx]

        H_gnd = nx.Graph()
        for u, v, d in self.G.edges(data=True):
            kind = d.get("kind", "")
            kind_u = str(kind).upper()

            # ✅ 只要不是卫星相关边，就当作“地面边”
            if kind_u not in ("ISL", "SAT-GND"):
                # 确保有权重字段
                if "base_delay_ms" not in d:
                    d["base_delay_ms"] = float(self.G.graph.get("ground_fiber_delay_ms", 60.0))
                H_gnd.add_edge(u, v, **d)

        # ✅ debug 要放这里
        if self.verbose:
            print(f"[DBG][GROUND] H_gnd nodes={H_gnd.number_of_nodes()}, edges={H_gnd.number_of_edges()}")
            print(f"[DBG][GROUND] src/dst in H_gnd? src_in={src_name in H_gnd}, dst_in={dst_name in H_gnd}")

        if not (src_name in H_gnd and dst_name in H_gnd):
            return None

        try:
            path_nodes = nx.shortest_path(H_gnd, source=src_name, target=dst_name, weight="base_delay_ms")
        except nx.NetworkXNoPath:
            return None

        return [self.node_index[n] for n in path_nodes]

    # ---- 给 DQN / GNN 用的接口 ----
    
    def label_action_path(self, path):
        if path == "GROUND" or path is None:
            return "ground"

        return self.classify_path(path)
        
    def classify_path(self, path_idx):
        """
        输入：path_idx = [node_idx, ...]
        输出："sat" / "ground" / "hybrid"
        规则：
        - sat     : 只包含 ISL / SAT-GND / GND-SAT
        - ground  : 只包含 GROUND
        - hybrid  : 同时包含 GROUND + (ISL 或 SAT-GND)
        """
        has_ground_edge = False
        has_sat_edge = False

        for u, v in zip(path_idx[:-1], path_idx[1:]):
            u_name = self.index_node[u]
            v_name = self.index_node[v]
            kind = self.G[u_name][v_name].get("kind", "")
            kind_u = str(kind).upper()

            if kind_u in ("ISL", "SAT-GND"):
                has_sat_edge = True
            else:
                has_ground_edge = True
        # —— 按你给的严格定义判定 ——
        if has_ground_edge and has_sat_edge:
            return "hybrid"
        elif has_sat_edge:
            return "sat"
        else:
            # 只剩下纯 GROUND
            return "ground"

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        np.random.seed(seed)
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        self._rng = np.random.default_rng(seed)
        return [seed]

    def _ensure_ground_delay_ms(self, r: dict):
        """防御式补齐 ground_delay_ms：fixed/random 都能用"""
        if "ground_delay_ms" in r and r["ground_delay_ms"] is not None:
            return

        src_data = self.G.nodes.get(r["src_name"], {})
        dst_data = self.G.nodes.get(r["dst_name"], {})
        lat1, lon1 = src_data.get("lat"), src_data.get("lon")
        lat2, lon2 = dst_data.get("lat"), dst_data.get("lon")

        if None not in (lat1, lon1, lat2, lon2):
            dist_km = haversine_km(lat1, lon1, lat2, lon2)
            r["ground_delay_ms"] = float(0.018 * dist_km + 20.0)
        else:
            r["ground_delay_ms"] = float(self.G.graph.get("ground_fiber_delay_ms", 60.0))


    def reset(self):
        """
        每次 reset:
        1) 重建拓扑 (TLE + WETLINK)
        2) 生成本 episode 的多个 request
        3) 准备第一个 request 的 K 条卫星路径 + GROUND
        """
        self.step_count = 0
        self.req_idx = 0
        self.requests = []
        self.total_delay = 0.0
        self.episode_logs = []

        # 1) 重建图
        self._build_new_graph()
        self.max_delay = float(np.nanmax(self.link_delay))
        if not np.isfinite(self.max_delay) or self.max_delay <= 0:
            self.max_delay = 1.0

        self.log.info("[EP] start | weather=%s reqs=%d shuffle=%s fixed=%s",
              self.enable_weather, self.requests_per_episode,
              self.shuffle_requests, self.use_fixed_requests)

        # 2) 生成多个 request
        if self.use_fixed_requests and self.fixed_requests:
            self.requests = [dict(r) for r in self.fixed_requests]
            if self.fixed_shuffle and len(self.requests) > 1:
                random.shuffle(self.requests)
        else:
            self.requests = []
            for _ in range(self.requests_per_episode):
                # 2.0) 选 src/dst
                if not self.long_pairs:
                    src_name = self.src_gs
                    dst_name = self.dst_gs
                else:
                    src_name, dst_name = random.choice(self.long_pairs)

                # 2.1) 选 demand
                demand = random.choice(self.listofDemands)

                # 2.2) baseline ground_delay_ms
                src_data = self.G.nodes.get(src_name, {})
                dst_data = self.G.nodes.get(dst_name, {})
                lat1, lon1 = src_data.get("lat"), src_data.get("lon")
                lat2, lon2 = dst_data.get("lat"), dst_data.get("lon")

                if None not in (lat1, lon1, lat2, lon2):
                    dist_km = haversine_km(lat1, lon1, lat2, lon2)
                    ground_delay_ms = 0.018 * dist_km + 20.0
                else:
                    ground_delay_ms = float(self.G.graph.get("ground_fiber_delay_ms", 60.0))

                self.requests.append({
                    "src_name": src_name,
                    "dst_name": dst_name,
                    "demand": demand,
                    "ground_delay_ms": float(ground_delay_ms),
                })

            if self.shuffle_requests and len(self.requests) > 1:
                random.shuffle(self.requests)

        # 2.4) fixed/random 都统一补 ground_delay_ms（防御式）
        for r in self.requests:
            self._ensure_ground_delay_ms(r)

        # ✅ 关键：下面这段必须在 for-loop 外面，只执行一次（准备第一个 request）
        if not self.requests:
            raise RuntimeError("[reset] requests is empty. Check long_pairs / fixed_requests.")

        demands = [r["demand"] for r in self.requests] if self.requests else [0]
        self.log.info("[EP] requests=%d | demand(min/mean/max)=(%.1f/%.1f/%.1f)",
                    len(self.requests), float(np.min(demands)),
                    float(np.mean(demands)), float(np.max(demands)))
        first_req = self.requests[0]
        src_name = first_req["src_name"]
        dst_name = first_req["dst_name"]
        demand   = first_req["demand"]
        ground_delay_ms = float(first_req["ground_delay_ms"])

        src_idx = self.node_index[src_name]
        dst_idx = self.node_index[dst_name]

        self.current_ground_delay = ground_delay_ms
        self.ground_delay = ground_delay_ms

        print(f"[Env] request0 baseline fiber delay={self.current_ground_delay:.3f} ms "
            f"(src={src_name}, dst={dst_name})")

        paths = self._compute_k_paths(src_idx, dst_idx, self.k_paths)
        paths.append("GROUND")

        labels = [self.label_action_path(p) for p in paths]
        key = f"{src_idx}:{dst_idx}"
        self.allPaths = {key: paths}
        self.allPathLabels = {key: labels}
        cnt = {"sat":0,"hybrid":0,"ground":0}
        for lab in labels: cnt[lab] = cnt.get(lab,0)+1
        self.log.info("[REQ0] %s->%s demand=%.1f | actions=%d (sat=%d hyb=%d gnd=%d)",
                    src_name, dst_name, float(demand), len(paths),
                    cnt.get("sat",0), cnt.get("hybrid",0), cnt.get("ground",0))
        print("[RESET] actions with labels:")
        for a, p in enumerate(paths):
            lab = labels[a]
            if p == "GROUND":
                print(f"  action {a}: GROUND  label={lab}")
            else:
                names = [self.index_node[i] for i in p]
                print(f"  action {a}: label={lab}, hops={len(p)-1}")
                print("    ", " -> ".join(names))

        self.current_src = src_idx
        self.current_dst = dst_idx
        self.current_demand = demand

        print(f"[RESET] request0: src={src_name}, dst={dst_name}, demand={demand}, "
            f"num_actions={len(paths)}, has_sat={any(p!='GROUND' for p in paths)}")

        # === DBG: 初始容量 sanity check ===
        down0 = self.graph_state[:, 0]
        up0   = self.graph_state[:, 1]
        print(f"[DBG][RESET-CAP] down(min/mean/max)=({down0.min():.3f}/{down0.mean():.3f}/{down0.max():.3f}), "
            f"up(min/mean/max)=({up0.min():.3f}/{up0.mean():.3f}/{up0.max():.3f})")

        z_down = int(np.sum(down0 <= 1e-9))
        z_up   = int(np.sum(up0   <= 1e-9))
        print(f"[DBG][RESET-CAP] zero_down_edges={z_down}/{len(down0)}, zero_up_edges={z_up}/{len(up0)}")

        gp = self._compute_ground_path(src_idx, dst_idx)
        if gp is None:
            print(f"[WARN][RESET] ground_path=None for {src_name}->{dst_name}. "
                f"GROUND action will be baseline-delay-only, not a real path.")

        return self.graph_state.copy(), self.current_demand, self.current_src, self.current_dst

    def _debug_why_infeasible(self, state, path_idx, demand, tag=""):
        for u_idx, v_idx in zip(path_idx[:-1], path_idx[1:]):
            pos = self.edgesDict.get(f"{u_idx}:{v_idx}")
            u_name = self.index_node[u_idx]
            v_name = self.index_node[v_idx]
            if pos is None:
                print(f"[DBG][{tag}] missing edge {u_name}->{v_name}")
                return
            kind = self.G[u_name][v_name].get("kind", "ISL")
            if kind == "SAT-GND":
                is_u_sat = (self.G.nodes[u_name].get("kind") == "sat")
                is_v_sat = (self.G.nodes[v_name].get("kind") == "sat")

                if is_u_sat and (not is_v_sat):
                    cap_col = 0   # SAT -> ground 用 down
                elif (not is_u_sat) and is_v_sat:
                    cap_col = 1   # ground -> SAT 用 up
                else:
                    cap_col = 0   # 防御
            else:
                cap_col = 0
            rem = float(state[pos, cap_col])
            if rem < demand - 1e-6:
                print(f"[DBG][{tag}] infeasible at {u_name}->{v_name} kind={kind} "
                    f"cap_col={cap_col} rem={rem:.3f} < demand={demand}")
                return
        print(f"[DBG][{tag}] path feasible (unexpected)")

    def _path_is_feasible(self, state, path_nodes_idx, demand):
        """不扣容量，仅检查这条路径上每条边容量是否足够"""
        for u_idx, v_idx in zip(path_nodes_idx[:-1], path_nodes_idx[1:]):
            eidx = self.edgesDict.get(f"{u_idx}:{v_idx}")
            if eidx is None:
                return False

            u_name = self.index_node[u_idx]
            v_name = self.index_node[v_idx]
            kind = self.G[u_name][v_name].get("kind", "ISL")

            # 跟你 make_step/_apply_path_transaction 的 down/up 规则一致
            if kind == "SAT-GND":
                is_u_sat = (self.G.nodes[u_name].get("kind") == "sat")
                is_v_sat = (self.G.nodes[v_name].get("kind") == "sat")
                if is_u_sat and (not is_v_sat):
                    cap_col = 0
                elif (not is_u_sat) and is_v_sat:
                    cap_col = 1
                else:
                    cap_col = 0
            else:
                cap_col = 0

            if float(state[eidx, cap_col]) < float(demand):
                return False

        return True


    def _select_feasible_route(self, state, src_idx, dst_idx, pathList, action, demand):
        """
        返回一个优先级列表：[(mode, cand), ...]
        规则：优先 sat（从 action 开始换），sat 全不可行才 ground
        """
        # 约定：GROUND 放在最后一个（你上游已经这么做了）
        ground_idx = None
        for i, p in enumerate(pathList):
            if p == "GROUND":
                ground_idx = i
                break

        sat_indices = [i for i in range(len(pathList)) if i != ground_idx]

        # 把 sat 的尝试顺序排成：action, action+1, ..., wraparound
        if action in sat_indices:
            start_pos = sat_indices.index(action)
            ordered_sat = sat_indices[start_pos:] + sat_indices[:start_pos]
        else:
            ordered_sat = sat_indices  # action 选到 ground 或越界（越界上游已处理）

        candidates = []

        # 1) 先加可行 sat（按优先级）
        for i in ordered_sat:
            p = pathList[i]
            if p == "GROUND":
                continue
            if isinstance(p, (list, tuple)) and self._path_is_feasible(state, list(p), demand):
                candidates.append(("SAT", p))

        # 2) sat 全不可行，才允许 ground
        if not candidates:
            candidates.append(("GROUND", "GROUND"))

        if not candidates:
            print(f"[DEBUG] No feasible SAT. key={src_idx}:{dst_idx} demand={demand} "
                f"num_paths={len(pathList)} action={action} "
                f"types={[type(x).__name__ for x in pathList[:min(5,len(pathList))]]}")

        return candidates

    def _apply_path_transaction(self, state, path_nodes_idx, demand):
        """
        对一条路径做“先检查后扣减”的事务：
        - 若任何边容量不足：不修改 state，返回 (False, None, None)
        - 若全部足够：返回 (True, new_state, path_edges_idx)
        """
        new_state = np.copy(state)
        path_edges_idx = []

        for u_idx, v_idx in zip(path_nodes_idx[:-1], path_nodes_idx[1:]):
            edge_key = f"{u_idx}:{v_idx}"
            edge_idx = self.edgesDict.get(edge_key)
            if edge_idx is None:
                return False, None, None

            u_name = self.index_node[u_idx]
            v_name = self.index_node[v_idx]
            kind = self.G[u_name][v_name].get("kind", "ISL")

            # 和 path_is_feasible 完全一致的方向选择
            if kind == "SAT-GND":
                is_u_sat = (self.G.nodes[u_name].get("kind") == "sat")
                is_v_sat = (self.G.nodes[v_name].get("kind") == "sat")
                if is_u_sat and (not is_v_sat):
                    cap_col = 0  # sat->ground down
                elif (not is_u_sat) and is_v_sat:
                    cap_col = 1  # ground->sat up
                else:
                    cap_col = 0
            else:
                cap_col = 0

            rem_before = float(new_state[edge_idx, cap_col])
            if rem_before < float(demand) - 1e-6:
                return False, None, None

            # 扣减 + 标记
            new_state[edge_idx, cap_col] = rem_before - float(demand)
            new_state[edge_idx, 2] = 1.0
            path_edges_idx.append(edge_idx)

        return True, new_state, path_edges_idx

    def _build_residual_graph(self, state, demand, sat_only=False):
        H = nx.Graph()

        for u, v, d in self.G.edges(data=True):
            kind_u = str(d.get("kind","")).upper()

            # 如果只要卫星相关边
            if sat_only and kind_u not in ("ISL", "SAT-GND"):
                continue

            iu = self.node_index[u]
            iv = self.node_index[v]
            pos = self.edgesDict.get(f"{iu}:{iv}")
            if pos is None:
                continue

            # --- 判断这条边在当前 demand 下是否可用 ---
            if kind_u == "SAT-GND":
                is_u_sat = (self.G.nodes[u].get("kind") == "sat")
                is_v_sat = (self.G.nodes[v].get("kind") == "sat")
                # sat->ground 用 down；ground->sat 用 up
                if is_u_sat and (not is_v_sat):
                    ok = (state[pos,0] >= demand - 1e-6)
                elif (not is_u_sat) and is_v_sat:
                    ok = (state[pos,1] >= demand - 1e-6)
                else:
                    ok = (min(state[pos,0], state[pos,1]) >= demand - 1e-6)
            else:
                ok = (state[pos,0] >= demand - 1e-6)

            if not ok:
                continue

            # 保留权重字段（你现在用 hybrid_weight 做搜索）
            w = float(d.get("hybrid_weight", 1.0))
            H.add_edge(u, v, **d)
            H[u][v]["hybrid_weight"] = w

        return H


    def _reroute_sat(self, state, src_idx, dst_idx, demand):
        src = self.index_node[src_idx]
        dst = self.index_node[dst_idx]

        H = self._build_residual_graph(state, demand, sat_only=True)
        if src not in H or dst not in H:
            return None

        try:
            nodes = nx.shortest_path(H, src, dst, weight="hybrid_weight")
            return [self.node_index[n] for n in nodes]
        except nx.NetworkXNoPath:
            return None

    def _reroute_sat_k(self, state, src_idx, dst_idx, demand, k=20):
        """
        在 residual 的 sat-only 图上找最多 k 条候选路径（都满足容量约束）。
        返回: list[list[int]]  (每条路径是 node_idx 列表)
        """
        src = self.index_node[src_idx]
        dst = self.index_node[dst_idx]

        H = self._build_residual_graph(state, demand, sat_only=True)
        if src not in H or dst not in H:
            return []

        paths = []
        try:
            gen = nx.shortest_simple_paths(H, source=src, target=dst, weight="hybrid_weight")
            for nodes in gen:
                paths.append([self.node_index[n] for n in nodes])
                if len(paths) >= int(k):
                    break
                if len(nodes) > 14:
                    continue
        except nx.NetworkXNoPath:
            return []

        return paths
    

    def _reroute_hybrid_k(self, state, src_idx, dst_idx, demand, k=20):
        src = self.index_node[src_idx]
        dst = self.index_node[dst_idx]

        H = self._build_residual_graph(state, demand, sat_only=False)  # ✅ 允许 ground 边
        if src not in H or dst not in H:
            return []

        paths = []
        try:
            gen = nx.shortest_simple_paths(H, source=src, target=dst, weight="hybrid_weight")
            for nodes in gen:
                # 可选：过滤掉纯 ground（如果你仍希望留给 GROUND action）
                has_sat_edge = False
                for u, v in zip(nodes[:-1], nodes[1:]):
                    kind = str(self.G[u][v].get("kind", "")).upper()
                    if kind in ("ISL", "SAT-GND"):
                        has_sat_edge = True
                        break
                if not has_sat_edge:
                    continue

                paths.append([self.node_index[n] for n in nodes])
                if len(paths) >= int(k):
                    break
        except nx.NetworkXNoPath:
            return []
        return paths



    def generate_environment(self, graph_topology=None, listofDemands=None):
        """
        为了兼容原来的 train_DQN.py：
            env.generate_environment(G, listofDemands)
        这里 graph_topology 会被忽略，真正的拓扑在 reset() 里按 TLE+WETLINK 生成。
        """
        if listofDemands is not None:
            self.listofDemands = listofDemands
            self.max_demand = max(listofDemands)
        return None

    def make_step(self, state, action, demand, source, destination):
        """
        新版 step：
        - 先在候选路径列表中找一条“容量足够”的路（按 action 指定的优先级）
        - 找到后再扣容量；如果 SAT 下所有候选路都容量不够，才算 fail
        """
        
        min_cap = float("inf")
        success = False
        mode = None
        route_nodes_idx = None

        e2e_delay = 0.0          # ✅ 必须先定义
        e2e_thput = 0.0
        path_edges_idx = []
        path_nodes_idx = []
        self.step_count += 1
        new_state = np.copy(state)

        # 用 env 内部当前 request 为准
        src_idx = self.current_src
        dst_idx = self.current_dst
        demand  = self.current_demand

        key = f"{src_idx}:{dst_idx}"
        key = f"{src_idx}:{dst_idx}"
        pathList = list(self.allPaths.get(key, [])) 
        if not pathList or action < 0 or action >= len(pathList):
            # 没有可用路径或动作越界，直接失败
            return new_state, -10.0, True, demand, src_idx, dst_idx

        original_choice = pathList[action]

# ---------- 1) 先在 pathList 中选择一条“容量足够”的路 ----------
        candidates = self._select_feasible_route(
            state=state,
            src_idx=src_idx,
            dst_idx=dst_idx,
            pathList=pathList,
            action=action,
            demand=demand,
        )

        labels = getattr(self, "allPathLabels", {}).get(key, None)
        original_choice = pathList[action]
        chosen_label = labels[action] if labels is not None else self.label_action_path(original_choice)

        # ---------- 2) 逐条尝试：事务扣减成功的第一条就执行 ----------
        success = False
        mode = None
        route_nodes_idx = None
        path_nodes_idx = []
        path_edges_idx = []
        new_state = np.copy(state)  # 默认不变

        for (m, cand) in candidates:
            if cand == "GROUND":
                # baseline ground：永远成功，不扣容量
                success = True
                mode = "GROUND"
                route_nodes_idx = "GROUND"
                path_nodes_idx = []
                path_edges_idx = []
                new_state = np.copy(state)
                break

            if m == "GROUND" and isinstance(cand, (list, tuple)):
                # 真实 ground path：也走事务扣减（如果你希望 ground 也受容量约束）
                ok, ns, eidxs = self._apply_path_transaction(state, list(cand), demand)
                if ok:
                    success = True
                    mode = "GROUND"
                    route_nodes_idx = list(cand)
                    path_nodes_idx = list(cand)
                    path_edges_idx = eidxs
                    new_state = ns
                    break
                else:
                    continue

            # SAT / hybrid 候选
            ok, ns, eidxs = self._apply_path_transaction(state, list(cand), demand)
            if ok:
                success = True
                mode = "SAT"
                route_nodes_idx = list(cand)
                path_nodes_idx = list(cand)
                path_edges_idx = eidxs
                new_state = ns
                break

        # ---------- 2.1) executed_label ----------
        if not success:
            executed_label = "none"
        elif route_nodes_idx == "GROUND":
            executed_label = "ground"
        else:
            executed_label = self.label_action_path(route_nodes_idx)

            e2e_delay = 0.0
        min_cap = float("inf")

        if success and (route_nodes_idx not in (None, "GROUND")) and path_nodes_idx:
            for u_idx, v_idx in zip(path_nodes_idx[:-1], path_nodes_idx[1:]):
                edge_idx = self.edgesDict.get(f"{u_idx}:{v_idx}")
                if edge_idx is None:
                    success = False
                    break

                u_name = self.index_node[u_idx]
                v_name = self.index_node[v_idx]
                kind = self.G[u_name][v_name].get("kind", "ISL")

                # ✅ 决定使用 down/up（和 _apply_path_transaction 一致）
                if kind == "SAT-GND":
                    is_u_sat = (self.G.nodes[u_name].get("kind") == "sat")
                    is_v_sat = (self.G.nodes[v_name].get("kind") == "sat")
                    if is_u_sat and (not is_v_sat):
                        cap_col = 0
                    elif (not is_u_sat) and is_v_sat:
                        cap_col = 1
                    else:
                        cap_col = 0
                else:
                    cap_col = 0

                # ✅ delay 只累加
                e2e_delay += float(self.link_delay[edge_idx])

                # ✅ min_cap 读事务扣完后的 new_state
                min_cap = min(min_cap, float(new_state[edge_idx, cap_col]))

            if success and np.isfinite(min_cap) and min_cap != float("inf"):
                e2e_thput = max(0.0, min(float(demand), float(min_cap)))
            else:
                e2e_thput = 0.0
        else:
            # baseline ground 或失败
            e2e_thput = 0.0

            if success:
                e2e_thput = max(0.0, min(min_cap, float(demand)))
            else:
                e2e_thput = 0.0

        # ========== 2) 计算 mesh_sat / mesh_ground + “按落地城市” gw_to_city_total_ms ==========
        mesh_sat_ms = 0.0
        mesh_ground_ms = 0.0
        gw_to_city_total_ms = 0.0
        gw_to_city_hops = []

        # 仅用于 Ku/Ka：记录第一个/最后一个 SAT-GND 边
        first_sg = None
        last_sg = None

        if path_nodes_idx:
            for u_idx, v_idx in zip(path_nodes_idx[:-1], path_nodes_idx[1:]):
                u_name = self.index_node[u_idx]
                v_name = self.index_node[v_idx]
                kind = self.G[u_name][v_name].get("kind", "ISL")

                eidx = self.edgesDict.get(f"{u_idx}:{v_idx}")
                if eidx is None:
                    continue

                # --- mesh delay 拆分 ---
                if kind in ("ISL", "SAT-GND"):
                    mesh_sat_ms += float(self.link_delay[eidx])
                else:
                    mesh_ground_ms += float(self.link_delay[eidx])

                # --- Ku/Ka：仅记录 first/last SAT-GND（不在这里算 Ku/Ka，先记下来） ---
                if kind == "SAT-GND":
                    if first_sg is None:
                        first_sg = (u_name, v_name)
                    last_sg = (u_name, v_name)   # ✅ 关键：每次都更新

                    # --- gw_to_city_total_ms：只统计 sat -> ground 的落地点（只在第2段算一次） ---
                    is_u_sat = (self.G.nodes[u_name].get("kind") == "sat")
                    is_v_sat = (self.G.nodes[v_name].get("kind") == "sat")
                    if is_u_sat and (not is_v_sat) and (v_name in self.city_to_gw):
                        gw_ms = float(self.city_to_gw[v_name].get("ground_ow_delay_ms", 0.0))
                        if gw_ms > 0.0:
                            gw_to_city_total_ms += gw_ms
                            gw_to_city_hops.append({"city": v_name, "gw_ms": gw_ms})

        # -------- 3) 计算 Ku / Ka（只与 SAT-GND 几何有关；不再动 gw_to_city_total_ms）--------
        ku_ms = 0.0
        ka_ms = 0.0

        if (mode != "GROUND") and path_nodes_idx:
            if first_sg is not None:
                u_name, v_name = first_sg
                dist_km = float(self.G[u_name][v_name].get("distance_km", 0.0))
                ku_ms = dist_km / C_KM_PER_MS

            if last_sg is not None:
                u_name, v_name = last_sg
                dist_km = float(self.G[u_name][v_name].get("distance_km", 0.0))
                ka_ms = dist_km / C_KM_PER_MS

        # -------- 4) 拥塞 + 处理延迟（由 WETLINK RTT 剩余部分估计）--------
        src_name = self.index_node[src_idx]
        dst_name = self.index_node[dst_idx]

        wet_src = float(self.G.nodes[src_name].get("wet_ping_ms", 0.0))
        wet_dst = float(self.G.nodes[dst_name].get("wet_ping_ms", 0.0))
        congestion_ms = wet_dst if (wet_dst > 0.0) else 0.0
        if congestion_ms < 0.0:
            congestion_ms = 0.0



        # 对纯 GROUND 路径：完全不用 Starlink，只保留地面光纤几何延迟
        if mode == "GROUND":
            ku_ms = 0.0
            ka_ms = 0.0
            gw_to_city_total_ms = 0.0
            congestion_ms = 0.0

            if not path_edges_idx:
                mesh_ground_ms = float(self.ground_delay)

            e2e_delay = mesh_ground_ms
        else:
            phys_total_ms = (
                ku_ms
                + ka_ms
                + gw_to_city_total_ms
                + mesh_sat_ms
                + mesh_ground_ms
            )
            e2e_delay = phys_total_ms + congestion_ms

        if success:
            self.total_delay += e2e_delay

        # -------- 5) 计算全图每条边的利用率 util_per_edge --------
        if self.base_capacity is not None:
            base_down = self.base_cap_down.astype(float)
            base_up   = self.base_cap_up.astype(float)

            cur_down = new_state[:, 0].astype(float)
            cur_up   = new_state[:, 1].astype(float)

            used_down = np.clip(base_down - cur_down, 0.0, None)
            used_up   = np.clip(base_up   - cur_up,   0.0, None)

            # ✅ 分母用“有效容量”（weather on 时会变小；weather off 时等于 base）
            den_down = getattr(self, "eff_cap_down", self.base_cap_down).astype(float)
            den_up   = getattr(self, "eff_cap_up",   self.base_cap_up).astype(float)

            util_down = used_down / np.where(den_down > 0.0, den_down, 1.0)
            util_up   = used_up   / np.where(den_up   > 0.0, den_up,   1.0)

            util_per_edge = np.maximum(util_down, util_up)
            util_per_edge = np.clip(util_per_edge, 0.0, 1.0)
            util_per_edge = np.nan_to_num(util_per_edge, nan=0.0, posinf=1.0, neginf=0.0)

            avg_util_global = float(util_per_edge.mean())

            if path_edges_idx:
                avg_util_path = float(np.mean([util_per_edge[e] for e in path_edges_idx]))
            else:
                avg_util_path = 0.0
        else:
            avg_util_global = 0.0
            avg_util_path = 0.0
            util_per_edge = None

        # -------- 5.5) 如果这次 request 失败，打印是哪条边炸了 --------
        if not success:
            print("[DEBUG] make_step: success=False TRIGGERED")
            print(f"[DEBUG] req_idx={self.req_idx}, src_idx={src_idx}, dst_idx={dst_idx}, demand={demand}")
            print(f"[DEBUG] mode={mode}, original_choice={original_choice}")
            print(f"[DEBUG] path_edges_idx={path_edges_idx}")

            if self.base_capacity is not None and path_edges_idx:
                for eidx in path_edges_idx:
                    i_node, j_node = self.ordered_edges[eidx]
                    u_name = self.index_node[i_node]
                    v_name = self.index_node[j_node]

                    cap   = float(self.base_capacity[eidx])
                    rem   = float(new_state[eidx, 0])
                    util  = float(util_per_edge[eidx])

                    print(
                        f"  edge {u_name} -> {v_name} "
                        f"(idx={eidx}): util={util:.3f}, cap={cap:.1f}, rem={rem:.1f}"
                    )

        # -------- 6) reward --------
        if not success:
            # 失败：不用给到 -20 这么狠，不然 loss 会很炸
            reward = -2.0
            done = True
        else:
            # 当前 request 的 ground baseline delay（在 reset / 切换 request 时已经设置好）
            ground_delay_ms = float(getattr(self, "current_ground_delay", self.ground_delay))

            # 保护一下，防止 ground_delay_ms 是 0 或 NaN
            if not np.isfinite(ground_delay_ms) or ground_delay_ms <= 0.0:
                ground_delay_ms = float(self.ground_delay) if np.isfinite(self.ground_delay) and self.ground_delay > 0 else 100.0

            # 1) 和 ground 比，快多少就是多少（单位大概每 100ms 提供 1 分奖励）
            rel_improve = (ground_delay_ms - e2e_delay) / 100.0

            # 2) 轻微惩罚全局平均利用率，避免把所有流量压到同一条边
            global_util = avg_util_global if avg_util_global is not None else 0.0
            util_penalty = 0.5 * global_util

            reward_raw = rel_improve - util_penalty

            # 3) 做一个 clip，防止 outlier 把 TD target 炸飞
            reward = float(np.clip(reward_raw, -2.0, 2.0))
            done = False



        # -------- 7) log --------
        src_name = self.index_node[src_idx]
        dst_name = self.index_node[dst_idx]
        path_node_names = [self.index_node[i] for i in path_nodes_idx]

        if mode == "GROUND":
            path_type = "ground"
        else:
            path_type = self.classify_path(path_nodes_idx) if path_nodes_idx else "unknown"

        log_record = {
            "req_id": self.req_idx,
            "success": bool(success),
            "src": src_name,
            "dst": dst_name,
            "demand": float(demand),
            "log_action_type": "GROUND" if original_choice == "GROUND" else "SAT",   # agent 选的动作类型
            "exec_action_type": "GROUND" if mode == "GROUND" else "SAT",            # 实际执行的类型（含 fallback）
            "action_label": chosen_label,          # ← agent 选的 action 的标签：sat/hybrid/ground
            "executed_label": executed_label, 
            "path_nodes": path_node_names,
            "path_type": path_type,
            "e2e_delay_ms": float(e2e_delay),
            "avg_util_path": float(avg_util_path),
            "avg_util_global": float(avg_util_global),

            "mesh_sat_ms": float(mesh_sat_ms),
            "mesh_ground_ms": float(mesh_ground_ms),
            "ku_ms": float(ku_ms),
            "ka_ms": float(ka_ms),
            "gw_to_city_total_ms": float(gw_to_city_total_ms),
            "gw_to_city_hops": gw_to_city_hops,
            "wet_ping_src_ms": float(wet_src),
            "wet_ping_dst_ms": float(wet_dst),
            "congestion_ms": float(congestion_ms),
        }

        recon = (
            log_record["ku_ms"]
            + log_record["ka_ms"]
            + log_record["gw_to_city_total_ms"]
            + log_record["mesh_sat_ms"]
            + log_record["mesh_ground_ms"]
            + log_record["congestion_ms"]
        )
        if abs(recon - log_record["e2e_delay_ms"]) > 1e-3:
            print(
                f"[WARN][Req {self.req_idx}] delay decomposition mismatch: "
                f"reconstructed={recon:.3f} ms, "
                f"e2e_delay={log_record['e2e_delay_ms']:.3f} ms"
            )

        if util_per_edge is not None and path_edges_idx:
            edge_list = []
            for eidx in path_edges_idx:
                i, j = self.ordered_edges[eidx]
                edge_list.append({
                    "u": self.index_node[i],
                    "v": self.index_node[j],
                    "edge_idx": int(eidx),
                    "util_after": float(util_per_edge[eidx]),
                })
            log_record["edges"] = edge_list

        self.episode_logs.append(log_record)

        # -------- 8) 切换下一个 request 或结束 episode --------
        self.req_idx += 1

        if self.req_idx >= len(self.requests) or not success:
            done = True
        else:
            next_req = self.requests[self.req_idx]
            src_name_next = next_req["src_name"]
            dst_name_next = next_req["dst_name"]
            demand_next   = next_req["demand"]
            ground_delay_next = float(next_req["ground_delay_ms"])

            src_idx_next = self.node_index[src_name_next]
            dst_idx_next = self.node_index[dst_name_next]

            paths_next = self._compute_k_paths(src_idx_next, dst_idx_next, self.k_paths)
            paths_next.append("GROUND")

            key_next = f"{src_idx_next}:{dst_idx_next}"
            labels_next = [self.label_action_path(p) for p in paths_next]
            self.allPaths = {key_next: paths_next}
            self.allPathLabels = {key_next: labels_next}

            self.current_src = src_idx_next
            self.current_dst = dst_idx_next
            self.current_demand = demand_next
            self.current_ground_delay = ground_delay_next
            self.ground_delay = ground_delay_next

        # -------- 9) Episode summary --------
        if done and self.verbose:
            print("=== Episode summary ===")
            for rec in self.episode_logs:
                print(f"[req {rec['req_id']:02d}] "
                    f"{'OK' if rec['success'] else 'FAIL'} "
                    f"{rec['src']} -> {rec['dst']} "
                    f"(demand={rec['demand']})")
                print(f"  action = {rec.get('exec_action_type', rec.get('log_action_type', 'NA'))}, "
                    f"path = {rec['path_nodes']}")
                print(
                    f"  e2e_delay = {rec['e2e_delay_ms']:.3f} ms "
                    f"(Ku={rec.get('ku_ms', 0.0):.3f}, "
                    f"Ka={rec.get('ka_ms', 0.0):.3f}, "
                    f"wet_src={rec.get('wet_ping_src_ms', 0.0):.3f}, "
                    f"wet_dst={rec.get('wet_ping_dst_ms', 0.0):.3f}, "
                    f"gw→city_total={rec.get('gw_to_city_total_ms', 0.0):.3f}, "
                    f"mesh_sat={rec.get('mesh_sat_ms', 0.0):.3f}, "
                    f"mesh_ground={rec.get('mesh_ground_ms', 0.0):.3f}, "
                    f"congestion={rec.get('congestion_ms', 0.0):.3f})"
                )
                print(f"  chosen_action = {rec.get('log_action_type','NA')}, "
                    f"executed_action = {rec.get('exec_action_type','NA')}, "
                    f"chosen_label={rec.get('action_label')}, executed_label={rec.get('executed_label')}")
                print(f"  path = {rec['path_nodes']}")
                hops = rec.get("gw_to_city_hops", [])
                if hops:
                    hop_str = ", ".join(
                        f"{h['city']}(+{h['gw_ms']:.3f} ms)" for h in hops
                    )
                    print(f"  gw→city hops: {hop_str}")
                print(
                    f"  path_avg_util = {rec['avg_util_path']:.3f}, "
                    f"global_avg_util = {rec['avg_util_global']:.3f}"
                )
            print("=== End of episode ===")

        # 把“是否使用”的标记清零（只清掉 used_flag 这一列）
        new_state[:, 2] = 0.0
        new_source = self.current_src
        new_destination = self.current_dst
        new_demand = self.current_demand

        return new_state, reward, done, new_demand, new_source, new_destination



if __name__ == "__main__":
    """
    小测试：
    - 构建一个 env（requests_per_episode=3）
    - reset 一次
    - 选一条 SAT/hybrid 路径（不是 GROUND）
    - 打印这条路径上 SAT-GND 边的容量 before/after
    """

    # TODO: 这里改成你自己的实际路径
    tle_path="iridium_66_main.tle"
    en_csv="analysis_data_Enschede.csv"
    osn_csv="analysis_data_Osnabrück.csv"

    # 需求列表随便给几个，比如 50/100/200 Mbps
    list_of_demands = [2,5]



    env = SatelliteGraphEnv(
        tle_path=tle_path,
        en_csv=en_csv,
        osn_csv=osn_csv,
        listofDemands=list_of_demands,
        K_paths=4,
        requests_per_episode=3,
        shuffle_requests=False,
    )

    # 固定一下随机种子，方便重现
    env.seed(42)

    env.load_requests_jsonl("fixed_reqs.jsonl", shuffle=False)

    # ====== 1) reset 一次，生成拓扑和第一个 request ======
    state, demand, src_idx, dst_idx = env.reset()

    print("\n==== After reset() ====")
    print(f"current request: src_idx={src_idx} ({env.index_node[src_idx]}), "
          f"dst_idx={dst_idx} ({env.index_node[dst_idx]}), "
          f"demand={demand}")

    key = f"{src_idx}:{dst_idx}"
    path_list = env.allPaths.get(key, [])
    print(f"num candidate actions = {len(path_list)}")
    for a, p in enumerate(path_list):
        if p == "GROUND":
            print(f"  action {a}: GROUND")
        else:
            names = [env.index_node[i] for i in p]
            print(f"  action {a}: SAT/hybrid path, hops={len(p)}")
            print("    ", " -> ".join(names))

    # ====== 2) 选一条 SAT/hybrid 的 action（不是 GROUND） ======
    sat_action = None
    for a, p in enumerate(path_list):
        if p != "GROUND":
            sat_action = a
            break

    if sat_action is None:
        print("[TEST] 没有任何 SAT/hybrid 路径可用，只剩 GROUND。直接退出测试。")
        exit(0)

    chosen_path = path_list[sat_action]
    chosen_names = [env.index_node[i] for i in chosen_path]
    print(f"\n[TEST] 选择 action={sat_action} 这条路径：")
    print("  ", " -> ".join(chosen_names))

    # ====== 3) 在调用 make_step 之前，找出这条路径上的 SAT-GND 边，打印容量 ======
    sat_gnd_edges = []

    for u_idx, v_idx in zip(chosen_path[:-1], chosen_path[1:]):
        edge_key = f"{u_idx}:{v_idx}"
        edge_idx = env.edgesDict.get(edge_key)
        if edge_idx is None:
            print(f"  [WARN] 边 {edge_key} 在 edgesDict 里找不到，跳过")
            continue

        u_name = env.index_node[u_idx]
        v_name = env.index_node[v_idx]
        kind = env.G[u_name][v_name].get("kind", "ISL")

        if kind == "SAT-GND":
            sat_gnd_edges.append((edge_idx, u_idx, v_idx, u_name, v_name))

    if not sat_gnd_edges:
        print("[TEST] 这条路径上居然没有 SAT-GND 边（纯 ISL?），也可以作为 sanity check。")
    else:
        print("\n[BEFORE make_step] SAT-GND edges on chosen path:")
        for edge_idx, u_idx, v_idx, u_name, v_name in sat_gnd_edges:
            print(
                f"  edge_idx={edge_idx}, {u_name} -> {v_name}, "
                f"base_down={env.base_cap_down[edge_idx]:.1f}, "
                f"base_up={env.base_cap_up[edge_idx]:.1f}, "
                f"state_down={state[edge_idx,0]:.1f}, "
                f"state_up={state[edge_idx,1]:.1f}"
            )

    # ====== 4) 调用 make_step，一步走完这个 request ======
    print("\n[TEST] 调用 make_step() ...\n")
    new_state, reward, done, new_demand, new_src, new_dst = env.make_step(
        state=state,
        action=sat_action,
        demand=demand,
        source=src_idx,
        destination=dst_idx,
    )

    print(f"[RESULT] reward={reward:.4f}, done={done}")
    print(f"next request: src_idx={new_src} ({env.index_node.get(new_src, 'N/A')}), "
          f"dst_idx={new_dst} ({env.index_node.get(new_dst, 'N/A')}), "
          f"demand={new_demand}")

    # ====== 5) 再看一次同样的 SAT-GND 边容量，确认 up/down 是否扣对 ======
    if sat_gnd_edges:
        print("\n[AFTER make_step] SAT-GND edges on chosen path:")
        for edge_idx, u_idx, v_idx, u_name, v_name in sat_gnd_edges:
            print(
                f"  edge_idx={edge_idx}, {u_name} -> {v_name}, "
                f"base_down={env.base_cap_down[edge_idx]:.1f}, "
                f"base_up={env.base_cap_up[edge_idx]:.1f}, "
                f"state_down(before)={state[edge_idx,0]:.1f}, "
                f"state_up(before)={state[edge_idx,1]:.1f}, "
                f"state_down(after)={new_state[edge_idx,0]:.1f}, "
                f"state_up(after)={new_state[edge_idx,1]:.1f}"
            )

    # ====== 6) 打印一下这次 request 的 log 记录 ======
    if env.episode_logs:
        print("\n[LOG] last request record:")
        last = env.episode_logs[-1]
        for k, v in last.items():
            if k == "edges":
                print("  edges:")
                for e in v:
                    print("    ", e)
            else:
                print(f"  {k}: {v}")
    else:
        print("[LOG] episode_logs 为空？说明 make_step 里可能没 append log_record。")
