# ============================================================
# SLOW — 7 días — Heatmap ACUMULATIVO (aguas arriba) — día 3
# ROBUSTO: nodes/edges pueden tener i/j como int o como "Node_X"
# 1) Lee Excel, primer Home_in por agente y día (time_slot en minutos)
# 2) Replica a 7 días si hace falta
# 3) Simula potencia local SLOW por nodo y hora
# 4) Construye árbol aguas arriba desde root (Dijkstra)
# 5) Dibuja la red (antes del heatmap)
# 6) Heatmap acumulativo: cada nodo suma lo de su subárbol
# ============================================================

import math
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ Constantes ------------------
MJ_TO_KWH = 1.0 / 3.6
P_SLOW_KW = 3.7

SOC_THRESHOLD = 0.50
SOC_TARGET = 0.80

DAY_NAMES_7 = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_TO_SHOW_INDEX = 2  # 0-based: 0 Mo, 1 Tu, 2 We

# ===================== Helpers =====================
def to_node_label(x):
    """Convierte 0 -> 'Node_0' y deja 'Node_0' tal cual."""
    s = str(x).strip()
    return s if s.startswith("Node_") else f"Node_{s}"

def parse_time_slot_to_minutes(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series.astype(str).str.slice(0, 5), format="%H:%M", errors="coerce")
    return (ts.dt.hour * 60 + ts.dt.minute).astype("float")

def ancestors_closure(parent: dict, root: str, targets: list[str]) -> list[str]:
    """
    Devuelve el conjunto de nodos que incluye:
      - root
      - todos los targets
      - y todos sus ancestros hasta root (para no romper la propagación)
    """
    keep = set([root])
    for t in targets:
        u = t
        while u is not None and u not in keep:
            keep.add(u)
            u = parent.get(u, None)
        # si u llega a None, ese target no está conectado al root en el árbol
    return list(keep)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ===================== 1) Marcadores por agente y día =====================
def compute_agents_home_markers_by_day(xlsx_path: str, sheet_name: str | int | None = 0, day_filter: str | None = None):
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    required = {"time_slot", "agent", "archetype", "todo", "mjkm", "node", "day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Disponibles: {list(df.columns)}")

    df = df[df["archetype"] == "PC_electric"].copy()
    if day_filter is not None:
        df = df[df["day"] == day_filter].copy()

    df["mjkm"] = df["mjkm"].astype(str).str.replace(",", ".", regex=False)
    df["mjkm"] = pd.to_numeric(df["mjkm"], errors="coerce").fillna(0.0)

    df["time_slot_min"] = parse_time_slot_to_minutes(df["time_slot"])
    df = df.dropna(subset=["time_slot_min"]).copy()
    df["time_slot_min"] = df["time_slot_min"].astype(int)

    home_in = df[df["todo"] == "Home_in"].copy()
    home_in = home_in.sort_values(["agent", "day", "time_slot_min"], ascending=[True, True, True])

    first_home_in = (
        home_in.groupby(["agent", "day"], as_index=False)
        .first()[["agent", "day", "time_slot_min", "node", "mjkm"]]
        .rename(columns={"time_slot_min": "start_min", "mjkm": "mj_consumed_at_marker"})
    )

    first_home_in["start_hour"] = (first_home_in["start_min"] // 60).astype(int)
    first_home_in["node_label"] = first_home_in["node"].apply(to_node_label)
    first_home_in["e_daily_kwh"] = first_home_in["mj_consumed_at_marker"] * MJ_TO_KWH

    meta = {
        "n_agents_pc_electric": int(df["agent"].nunique()),
        "n_agent_day_with_home_in": int(len(first_home_in)),
        "days_present": sorted(first_home_in["day"].unique().tolist()),
    }
    return first_home_in[["agent", "day", "start_min", "start_hour", "node_label", "e_daily_kwh"]], meta

def ensure_7_days(markers: pd.DataFrame, day_names: list[str]) -> pd.DataFrame:
    present = sorted(markers["day"].unique().tolist())
    if len(present) >= len(day_names):
        return markers.copy()
    pattern_day = present[0]
    base = markers[markers["day"] == pattern_day].copy()
    return pd.concat([base.assign(day=d) for d in day_names], ignore_index=True)

# ===================== 2) Batería + SoC inicial =====================
def assign_battery_and_initial_soc(
    agents_markers: pd.DataFrame,
    soc_min: float = 0.50,
    soc_max: float = 0.80,
    batt_mean_kwh: float = 60.0,
    batt_sd_kwh: float = 10.0,
    batt_min_kwh: float = 30.0,
    batt_max_kwh: float = 100.0,
    seed: int | None = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = agents_markers.copy()

    batt = rng.normal(loc=batt_mean_kwh, scale=batt_sd_kwh, size=len(out))
    batt = np.clip(batt, batt_min_kwh, batt_max_kwh)
    out["battery_kwh"] = batt

    out["soc0"] = rng.uniform(low=soc_min, high=soc_max, size=len(out))
    return out

# ===================== 3) Simulación SLOW local por nodo =====================
def simulate_fleet_soc_rule_by_node_slow(
    agents_day: pd.DataFrame,
    day_names: list[str],
    soc_threshold: float = SOC_THRESHOLD,
    soc_target: float = SOC_TARGET,
    p_slow_kw: float = P_SLOW_KW,
):
    day_to_index = {d: i for i, d in enumerate(day_names)}
    n_days = len(day_names)
    horizon = n_days * 24

    df = agents_day.copy()
    df = df[df["day"].isin(day_to_index)].copy()
    df["day_index"] = df["day"].map(day_to_index).astype(int)

    agents = df["agent"].unique().tolist()
    soc0_map = df.groupby("agent")["soc0"].first().to_dict()
    batt_map = df.groupby("agent")["battery_kwh"].first().to_dict()
    soc_state = {a: float(soc0_map.get(a, 0.65)) for a in agents}

    fleet_slow = {}  # node_label -> np.array(horizon)
    n_charge_events_slow = 0

    df = df.sort_values(["agent", "day_index", "start_min"])

    for a, sub in df.groupby("agent", sort=False):
        batt = float(batt_map[a])
        soc = float(soc_state[a])

        for _, row in sub.iterrows():
            d = int(row["day_index"])
            start_min = int(row["start_min"])
            node = str(row["node_label"])
            e_daily = float(row["e_daily_kwh"])

            t_arr = d * 24 + (start_min // 60)
            if t_arr >= horizon:
                continue

            soc -= (e_daily / batt)
            soc = max(soc, 0.0)

            if soc < soc_threshold:
                e_need = max(0.0, (soc_target - soc) * batt)
                if e_need <= 0:
                    continue

                if node not in fleet_slow:
                    fleet_slow[node] = np.zeros(horizon, dtype=float)

                n_charge_events_slow += 1
                remaining = e_need
                for t in range(t_arr, horizon):
                    if remaining <= 0:
                        break
                    e_can = p_slow_kw * 1.0  # kWh/h
                    e_this = min(remaining, e_can)
                    fleet_slow[node][t] += e_this  # kWh/h == kW
                    remaining -= e_this

                soc = soc_target

        soc_state[a] = soc

    rows = []
    for node in sorted(fleet_slow.keys()):
        slow = fleet_slow[node]
        for t in range(horizon):
            rows.append((t, t // 24, t % 24, day_names[t // 24], node, slow[t]))

    fleet_node_ts = pd.DataFrame(rows, columns=["t", "day_index", "hour", "day", "node_label", "P_slow_kW"])

    node_summary = (
        fleet_node_ts.groupby("node_label", as_index=False)
        .agg(kWh_total_slow=("P_slow_kW", "sum"),
             peak_slow_kW=("P_slow_kW", "max"))
    )

    meta = {
        "n_days": n_days,
        "horizon_hours": horizon,
        "n_agents": int(len(agents)),
        "n_charge_events_slow": int(n_charge_events_slow),
        "n_nodes_with_charging": int(fleet_node_ts["node_label"].nunique()),
    }
    return fleet_node_ts, node_summary, meta

# ===================== 4) Grafo ponderado + árbol aguas arriba =====================
def build_weighted_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    nodes: debe tener lon/lat y alguna id (i o node_label)
    edges: columnas i, j (pueden ser int o "Node_X")
    """
    nodes = nodes.copy()

    # Descubrir columna id en nodes
    if "node_label" not in nodes.columns:
        if "i" in nodes.columns:
            nodes["node_label"] = nodes["i"].apply(to_node_label)
        else:
            raise ValueError("nodes debe tener columna 'i' o 'node_label'")

    coord_map = dict(zip(nodes["node_label"], zip(nodes["lon"], nodes["lat"])))

    adj = defaultdict(list)
    for _, r in edges.iterrows():
        a = to_node_label(r["i"])
        b = to_node_label(r["j"])
        if a in coord_map and b in coord_map:
            lon1, lat1 = coord_map[a]
            lon2, lat2 = coord_map[b]
            w = haversine_km(lon1, lat1, lon2, lat2)
        else:
            w = 0.0
        adj[a].append((b, w))
        adj[b].append((a, w))

    return adj, coord_map

def dijkstra_tree(adj, root: str):
    dist = {root: 0.0}
    parent = {root: None}
    pq = [(0.0, root)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, None):
            continue
        for v, w in adj.get(u, []):
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent

def build_children_and_postorder(parent: dict, root: str):
    children = defaultdict(list)
    for node, p in parent.items():
        if p is None:
            continue
        children[p].append(node)

    post = []
    stack = [(root, 0)]
    while stack:
        u, state = stack.pop()
        if state == 0:
            stack.append((u, 1))
            for ch in children.get(u, []):
                stack.append((ch, 0))
        else:
            post.append(u)
    return children, post

# ===================== 5) Plot de red =====================
def plot_network(coord_map: dict, edges: pd.DataFrame, root: str, dist: dict, charging_nodes: set[str], parent: dict):
    plt.figure(figsize=(9, 9))

    # edges
    for _, r in edges.iterrows():
        a = to_node_label(r["i"])
        b = to_node_label(r["j"])
        if a in coord_map and b in coord_map:
            x1, y1 = coord_map[a]
            x2, y2 = coord_map[b]
            plt.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.6)

    # nodes colored by dist
    nodes_list = list(coord_map.keys())
    xs = [coord_map[n][0] for n in nodes_list]
    ys = [coord_map[n][1] for n in nodes_list]
    dvals = np.array([dist.get(n, np.nan) for n in nodes_list], dtype=float)

    sc = plt.scatter(xs, ys, c=dvals, s=25, zorder=3)
    cb = plt.colorbar(sc)
    cb.set_label("Distancia al root (km)")

    # root
    if root in coord_map:
        rx, ry = coord_map[root]
        plt.scatter([rx], [ry], s=140, marker="*", zorder=6)
        plt.text(rx, ry, f"  {root}", fontsize=10, va="center", zorder=7)

    # charging highlight
    ch = [n for n in charging_nodes if n in coord_map]
    if ch:
        cx = [coord_map[n][0] for n in ch]
        cy = [coord_map[n][1] for n in ch]
        plt.scatter(cx, cy, s=90, facecolors="none", edgecolors="black", linewidths=1.5, zorder=7)

    # arrows parent->child to show upstream/downstream orientation
    for child, p in parent.items():
        if p is None:
            continue
        if p in coord_map and child in coord_map:
            x1, y1 = coord_map[p]
            x2, y2 = coord_map[child]
            plt.annotate(
                "",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.35),
                zorder=2
            )

    plt.title("Red: edges + nodos (color=distancia al root). Círculo negro=nodos con carga. Flechas=árbol desde root")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# ===================== 6) Matriz local/acumulativa y heatmap =====================
def make_local_matrix_day(fleet_node_ts: pd.DataFrame, ordered_buses: list[str], day_index: int):
    df = fleet_node_ts[fleet_node_ts["day_index"] == day_index].copy()
    agg = df.groupby(["node_label", "hour"], as_index=False)["P_slow_kW"].sum()
    piv = agg.pivot(index="node_label", columns="hour", values="P_slow_kW").fillna(0.0)

    for h in range(24):
        if h not in piv.columns:
            piv[h] = 0.0
    piv = piv[list(range(24))]

    piv = piv.reindex(ordered_buses).fillna(0.0)
    return piv.to_numpy(dtype=float), piv.index.tolist()

def accumulate_upstream_subtree(M_local: np.ndarray, y_labels: list[str], root: str, parent: dict):
    idx = {n: i for i, n in enumerate(y_labels)}
    children, post = build_children_and_postorder(parent, root=root)

    M_cum = M_local.copy()
    for u in post:  # hijos -> padre
        if u not in idx:
            continue
        ui = idx[u]
        for ch in children.get(u, []):
            if ch not in idx:
                continue
            ci = idx[ch]
            M_cum[ui, :] += M_cum[ci, :]
    return M_cum

def plot_heatmap(M: np.ndarray, y_labels: list[str], title: str, cbar_label: str, max_buses_labels: int = 60):
    plt.figure(figsize=(12, max(6, 0.18 * len(y_labels))))
    im = plt.imshow(M, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, label=cbar_label)

    plt.title(title)
    plt.xlabel("Hora del día")
    plt.ylabel("Bus (ordenado por distancia desde root)")
    plt.xticks(ticks=list(range(24)), labels=[f"{h:02d}" for h in range(24)])

    n = len(y_labels)
    if n <= max_buses_labels:
        yticks = np.arange(n)
        ytxt = [lab.replace("Node_", "") for lab in y_labels]
    else:
        step = int(np.ceil(n / max_buses_labels))
        yticks = np.arange(0, n, step)
        ytxt = [y_labels[i].replace("Node_", "") for i in yticks]

    plt.yticks(ticks=yticks, labels=ytxt)
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # --------- INPUTS ----------
    xlsx = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_vehicle_quantified_24.xlsx"

    study_case = "Kanaleneiland"
    path = rf"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\{study_case}\population"
    nodes_csv = rf"{path}\node_data_{study_case}.csv"
    edges_csv = rf"{path}\{study_case}_line_data.csv"

    root = "Node_67"  # <- ajusta

    # --------- Cargar red ----------
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    # --------- Leer markers ----------
    markers, meta_read = compute_agents_home_markers_by_day(xlsx, sheet_name=0, day_filter=None)
    print("META READ:", meta_read)
    print(markers.head())

    # --------- Forzar 7 días ----------
    markers_7d = ensure_7_days(markers, DAY_NAMES_7)
    print("Días tras ensure_7_days:", sorted(markers_7d["day"].unique().tolist()))
    print("agent-day únicos:", markers_7d[["agent", "day"]].drop_duplicates().shape[0])

    # --------- Asignar batería y SoC ----------
    agents_day = assign_battery_and_initial_soc(markers_7d, seed=123)

    # --------- Simular 7 días SLOW local por nodo ----------
    fleet_node_ts, node_summary, meta_sim = simulate_fleet_soc_rule_by_node_slow(
        agents_day=agents_day,
        day_names=DAY_NAMES_7,
        soc_threshold=SOC_THRESHOLD,
        soc_target=SOC_TARGET,
        p_slow_kw=P_SLOW_KW
    )
    print("META SIM:", meta_sim)
    print(node_summary.sort_values("kWh_total_slow", ascending=False).head(10))
    print("Suma P_slow_kW por día_index:")
    print(fleet_node_ts.groupby("day_index")["P_slow_kW"].sum())

    # --------- Construir árbol desde root ----------
    adj, coord_map = build_weighted_graph(nodes, edges)
    dist, parent = dijkstra_tree(adj, root=root)

    # --------- Nodos con carga (toda la semana) ----------
    charging_nodes = set(fleet_node_ts.loc[fleet_node_ts["P_slow_kW"] > 0, "node_label"].unique().tolist())

    charging_reachable = [n for n in charging_nodes if n in dist]

    # 🔴 ESTA es la corrección: incluir ancestros/intermedios
    universe_nodes = ancestors_closure(parent=parent, root=root, targets=charging_reachable)

    print("Nodos con carga:", len(charging_nodes))
    print("Nodos con carga alcanzables desde root:", len(charging_reachable), "/", len(charging_nodes))

    # --------- Mostrar red antes del heatmap ----------
    plot_network(coord_map, edges, root, dist, charging_nodes, parent)

    # --------- Universo y orden eje Y (solo nodos con carga alcanzables + root) ----------
    # orden por distancia desde root
    universe = sorted(universe_nodes, key=lambda n: (dist.get(n, float("inf")), n))

    # --------- Matriz local día 3 ----------
    M_local, y_labels = make_local_matrix_day(
        fleet_node_ts=fleet_node_ts,
        ordered_buses=universe,
        day_index=DAY_TO_SHOW_INDEX
    )
    print("LOCAL día 3 -> max:", float(M_local.max()), "sum:", float(M_local.sum()))

    # --------- Matriz acumulativa aguas arriba (subárbol) día 3 ----------
    M_cum = accumulate_upstream_subtree(M_local, y_labels, root=root, parent=parent)
    print("CUM día 3 -> max:", float(M_cum.max()), "sum:", float(M_cum.sum()))

    # Sanity: root cum vs suma local (en el universo)
    if root in y_labels:
        ri = y_labels.index(root)
        print("Sanity (hora 20): root_cum=", float(M_cum[ri, 20]), " sum_local=", float(M_local[:, 20].sum()))

    # --------- Heatmap acumulativo ----------
    plot_heatmap(
        M_cum, y_labels,
        title=f"Heatmap SLOW ACUMULATIVO (aguas arriba) — día 3 ({DAY_NAMES_7[DAY_TO_SHOW_INDEX]}) — raíz {root}",
        cbar_label="Potencia acumulada (subárbol) [kW]"
    )
