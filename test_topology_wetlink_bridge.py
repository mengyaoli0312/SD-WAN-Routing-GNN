from topology_builder import build_satellite_graph
from topology_builder import sample_wetlink_profile

def test_wetlink_in_graph():
    G = build_satellite_graph(
        tle_path="starlink_latest.tle",
        t_obs="2025-11-28 03:00:00",
        max_sats=10,
        isl_k=2,
        sg_max_dist_km=1500.0,
        wetlink_gs_map={
            "GS_ENS": {"csv_path": "analysis_data_Enschede.csv", "site_name": "utwente"},
            "GS_OSN": {"csv_path": "analysis_data_Osnabrück.csv", "site_name": None},
        },
    )

    for gs in ["GS_ENS", "GS_OSN"]:
        print(gs, "wet_ping_ms =", G.nodes[gs].get("wet_ping_ms"),
                  "wet_download_Mbps =", G.nodes[gs].get("wet_download_Mbps"))
        assert "wet_ping_ms" in G.nodes[gs]
        assert "wet_download_Mbps" in G.nodes[gs]

def test_sat_gnd_consistency_for_gs(G, gs_name):
    edges = []
    for u, v, d in G.edges(data=True):
        if d.get("kind") == "SAT-GND" and (u == gs_name or v == gs_name):
            edges.append((u, v, d))

    caps = [e[2]["base_capacity_Mbps"] for e in edges]
    rains = [e[2]["weather_rain"] for e in edges]
    pings = [e[2]["wet_ping_ms"] for e in edges]  # 如果你在 edge 上也存了，可检查

    print(gs_name, "capacity range:", min(caps), max(caps))
    # 差异应该很小（浮点误差级别）

if __name__ == "__main__":
    test_sat_gnd_consistency_for_gs()
    test_wetlink_in_graph()
