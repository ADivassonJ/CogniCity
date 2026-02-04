import os
import math
import glob
import heapq
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
S_FOLDERS = [f"s{i}" for i in range(5)]

# ROOT por caso (los que ya fijaste; añade los nuevos cuando quieras)
root_map = {
    "Annelinn": "Node_43",
    "Aradas": "Node_20",
    "Kanaleneiland": "Node_67",
    # "OtroCaso": "Node_XX",
    # "OtroCaso2": "Node_YY",
}

SAT_THRESHOLD_PCT = 100.0     # saturación
DEMAND_SCALE = 1.0            # si pDemand está en kW, pon 1/1000 para MW

OUT_DIR_NAME = "heatmaps_line_saturation"
SAVE_FIGS = False


# ============================================================
# DESKTOP AUTO
# ============================================================
def candidate_desktops():
    home = Path.home()
    cands = [
        home / "Desktop",
        home / "Escritorio",
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "Escritorio",
    ]
    return [p for p in cands if p.exists() and p.is_dir()]

def find_files_on_desktop(pattern: str):
    hits = []
    for d in candidate_desktops():
        hits += list(d.glob(pattern))
    return hits

def pick_best_excel(case_name: str):
    """
    Busca en el escritorio un Excel que contenga el case_name y tenga hojas pDemand y pPmax_line.
    Prioriza nombres que contengan 'datos_extraidos' si existen.
    """
    excels = []
    for d in candidate_desktops():
        excels += list(d.glob(f"*{case_name}*.xls*"))

    # preferencia: datos_extraidos primero
    excels.sort(key=lambda p: (0 if "datos_extraidos" in p.name.lower() else 1, len(p.name)))

    for x in excels:
        try:
            xls = pd.ExcelFile(x)
            if "pDemand" in xls.sheet_names and "pPmax_line" in xls.sheet_names:
                return x
        except Exception:
            continue

    return None

def pick_best_nodes_csv(case_name: str):
    # típico: node_data_<case>.csv
    candidates = []
    for d in candidate_desktops():
        candidates += list(d.glob(f"*node*data*{case_name}*.csv"))
        candidates += list(d.glob(f"*{case_name}*node*data*.csv"))
        candidates += list(d.glob(f"*node_data_{case_name}.csv"))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0] if candidates else None

def pick_best_edges_csv(case_name: str):
    # típico: <case>_line_data.csv o Aradas_line_data.csv
    candidates = []
    for d in candidate_desktops():
        candidates += list(d.glob(f"*{case_name}*line*data*.csv"))
        candidates += list(d.glob(f"*line*data*{case_name}*.csv"))
        candidates += list(d.glob(f"*{case_name}_line_data.csv"))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0] if candidates else None


# ============================================================
# HELPERS RED / DIJKSTRA
# ============================================================
def to_node_label(x):
    s = str(x).strip()
    return s if s.startswith("Node_") else f"Node_{s}"

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_weighted_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    nodes = nodes.copy()
    if "i" in nodes.columns:
        nodes["node_label"] = nodes["i"].apply(to_node_label)
    elif "node_label" not in nodes.columns:
        raise ValueError("nodes debe tener columna 'i' o 'node_label'")

    # soporta lat/lon o lon/lat
    if "lon" in nodes.columns and "lat" in nodes.columns:
        coord_map = dict(zip(nodes["node_label"], zip(nodes["lon"], nodes["lat"])))
    elif "longitude" in nodes.columns and "latitude" in nodes.columns:
        coord_map = dict(zip(nodes["node_label"], zip(nodes["longitude"], nodes["latitude"])))
    else:
        coord_map = {}

    adj = defaultdict(list)
    for _, r in edges.iterrows():
        a = to_node_label(r["i"])
        b = to_node_label(r["j"])
        if a in coord_map and b in coord_map:
            lon1, lat1 = coord_map[a]
            lon2, lat2 = coord_map[b]
            w = haversine_km(lon1, lat1, lon2, lat2)
        else:
            w = 1.0
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

def edge_key(a, b):
    return frozenset([to_node_label(a), to_node_label(b)])


# ============================================================
# INPUTS: Pmax y Demanda
# ============================================================
def load_pmax_map(pPmax_line_df: pd.DataFrame):
    pmax = {}
    for _, r in pPmax_line_df.iterrows():
        a = to_node_label(r["level_0"])
        b = to_node_label(r["level_1"])
        pmax[edge_key(a, b)] = float(r["values"])
    return pmax

def load_hourly_demand(pDemand_df: pd.DataFrame):
    df = pDemand_df.copy()
    df["Buses"] = df["Buses"].astype(str).map(to_node_label)
    df["values"] = pd.to_numeric(df["values"], errors="coerce").fillna(0.0) * float(DEMAND_SCALE)

    g = df.groupby(["Time", "Buses"])["values"].sum()
    hours = sorted(df["Time"].unique().tolist())

    demand_by_time = {}
    for t in hours:
        if t in g.index.get_level_values(0):
            s = g.loc[t]
            demand_by_time[t] = s.to_dict()
        else:
            demand_by_time[t] = {}
    return hours, demand_by_time


# ============================================================
# CÁLCULO: flujo radial + saturación por línea y hora
# ============================================================
def compute_line_flows_and_saturation(hours, demand_by_time, universe, parent, root, pmax_map):
    children, post = build_children_and_postorder(parent, root=root)

    # lista de líneas orientadas (parent -> child)
    oriented = []
    for u in universe:
        p = parent.get(u, None)
        if p is None:
            continue
        oriented.append((p, u))

    line_names = [f"{p}->{u}" for p, u in oriented]
    sat = np.full((len(oriented), len(hours)), np.nan, dtype=float)

    anomalies = {t: [] for t in hours}

    for hi, t in enumerate(hours):
        dmap = demand_by_time.get(t, {})
        node_d = {n: float(dmap.get(n, 0.0)) for n in universe}

        subtree = {n: node_d[n] for n in universe}
        for u in post:
            for ch in children.get(u, []):
                if ch in subtree:
                    subtree[u] += subtree[ch]

        for li, (p, u) in enumerate(oriented):
            flow = float(subtree.get(u, 0.0))
            pmax = pmax_map.get(edge_key(p, u), None)
            if pmax is None or pmax <= 0:
                continue

            pct = 100.0 * flow / float(pmax)
            sat[li, hi] = min(100.0, pct)

            if pct >= SAT_THRESHOLD_PCT:
                anomalies[t].append((f"{p}->{u}", pct, flow, float(pmax)))

    anomalies = {t: v for t, v in anomalies.items() if len(v) > 0}
    return line_names, sat, anomalies


# ============================================================
# PLOT: panel 5 heatmaps + 1 colorbar
# ============================================================
def plot_heatmap_panel_5(mats: dict, s_order: list[str], title: str, vmin=0.0, vmax=100.0):
    fig_height = max(7, 0.18 * max(mats[s]["M"].shape[0] for s in s_order))
    fig = plt.figure(figsize=(24 * 0.22 * 5 + 0.6, fig_height))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.15)

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax = fig.add_subplot(gs[0, 5])

    mappable = None
    for ax, s in zip(axes, s_order):
        M = mats[s]["M"]
        y_labels = mats[s]["y_labels"]

        im = ax.imshow(
            M,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        mappable = im
        ax.set_title(s, fontsize=11)
        ax.set_xlim(-0.5, 23.5)
        even_hours = np.arange(0, 24, 2)
        ax.set_xticks(even_hours)
        ax.set_xticklabels([str(h) for h in even_hours], fontsize=9)
        ax.set_xlabel("Hour")

        if s == s_order[0]:
            ax.set_ylabel("Line (parent->child)")
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=6)
        else:
            ax.set_yticks([])

    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Line saturation [%] (capped at 100)")
    fig.suptitle(title, fontsize=13)
    plt.show()
    return fig

# ============================================================
# RESUMEN: saturación promedio por sistema y por hora
# ============================================================
def mean_saturation_overall(satM: np.ndarray) -> float:
    """Promedio global (líneas x horas), ignorando NaN."""
    x = np.asarray(satM, dtype=float)
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))

def mean_saturation_by_hour(satM: np.ndarray) -> np.ndarray:
    """Vector de 24 elementos: promedio por hora (promedio sobre líneas), ignorando NaN."""
    x = np.asarray(satM, dtype=float)
    # axis=0 => promedio sobre líneas
    return np.nanmean(x, axis=0)

def mean_saturation_by_line(satM: np.ndarray) -> np.ndarray:
    """Promedio por línea (promedio sobre horas), ignorando NaN."""
    x = np.asarray(satM, dtype=float)
    # axis=1 => promedio sobre horas
    return np.nanmean(x, axis=1)

def sat_stats_overall(satM: np.ndarray) -> dict:
    """Devuelve avg, min y max global (líneas x horas), ignorando NaN."""
    x = np.asarray(satM, dtype=float)

    if np.all(np.isnan(x)):
        return {
            "avg": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "avg": float(np.nanmean(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
    }

# ============================================================
# MAIN por caso
# ============================================================
def run_one_case(case_name: str, out_dir: Path):
    root = root_map.get(case_name)
    if not root:
        return

    nodes_csv = pick_best_nodes_csv(case_name)
    edges_csv = pick_best_edges_csv(case_name)
    excel_path = pick_best_excel(case_name)

    if not nodes_csv or not edges_csv or not excel_path:
        return

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    adj, _ = build_weighted_graph(nodes, edges)
    dist, parent = dijkstra_tree(adj, root=root)
    universe = sorted(dist.keys(), key=lambda n: (dist[n], n))

    xls = pd.ExcelFile(excel_path)
    pDemand = pd.read_excel(xls, "pDemand")
    pPmax_line = pd.read_excel(xls, "pPmax_line")

    hours, demand_by_time = load_hourly_demand(pDemand)
    pmax_map = load_pmax_map(pPmax_line)

    mats = {}
    text_anoms = {}

    # >>> PROMEDIOS: iremos acumulando un resumen por sistema
    avg_rows = []  # lista de dicts para DataFrame

    for s in S_FOLDERS:
        line_names, satM, anomalies = compute_line_flows_and_saturation(
            hours=hours,
            demand_by_time=demand_by_time,
            universe=universe,
            parent=parent,
            root=root,
            pmax_map=pmax_map,
        )
        mats[s] = {"M": satM, "y_labels": line_names}
        text_anoms[s] = anomalies

        # >>> PROMEDIOS
        avg_global = mean_saturation_overall(satM)
        stats_rows = []

    for s in S_FOLDERS:
        line_names, satM, anomalies = compute_line_flows_and_saturation(
            hours=hours,
            demand_by_time=demand_by_time,
            universe=universe,
            parent=parent,
            root=root,
            pmax_map=pmax_map,
        )
        mats[s] = {"M": satM, "y_labels": line_names}
        text_anoms[s] = anomalies

        # >>> avg / min / max
        stats = sat_stats_overall(satM)

        stats_rows.append({
            "case": case_name,
            "root": root,
            "system": s,
            "avg_saturation_pct": stats["avg"],
            "min_saturation_pct": stats["min"],
            "max_saturation_pct": stats["max"],
        })

    # >>> imprime tabla de promedios por sistema para este caso
    stats_df = pd.DataFrame(stats_rows).sort_values(["case", "system"])

    print("\n" + "-" * 90)
    print(f"RESUMEN SATURACIÓN | CASO: {case_name} | ROOT: {root}")
    print(
        stats_df[
            ["system", "avg_saturation_pct", "min_saturation_pct", "max_saturation_pct"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:7.2f}"
        )
    )
    print("-" * 90)

    # >>> (Opcional) guarda CSV por caso
    if SAVE_FIGS:
        avg_out = out_dir / f"avg_line_saturation_{case_name}_root{root}.csv"
        avg_df.to_csv(avg_out, index=False)

    # TEXTO anomalías (como ya lo tenías)
    anomalies = text_anoms["s0"]
    if anomalies:
        print("\n" + "=" * 80)
        print(f"CASO: {case_name} | ROOT: {root}")
        print(f"Inputs: {excel_path.name} | Nodes: {nodes_csv.name} | Lines: {edges_csv.name}")
        print("=" * 80)

        for t in sorted(anomalies.keys()):
            hits = sorted(anomalies[t], key=lambda x: x[1], reverse=True)
            print(f"\nAnomalía detectada a las {int(t):02d}:00 (línea(s) saturada(s) >=100%)")
            for (lname, pct, flow, pmax) in hits[:8]:
                print(f"  - {lname}: {pct:.1f}%  (P={flow:.3f} MW, Pmax={pmax:.3f} MW)")

    # HEATMAPS
    title = f"{case_name} — Saturación líneas [%] (cap 100) — root {root}"
    fig = plot_heatmap_panel_5(mats=mats, s_order=S_FOLDERS, title=title, vmin=0.0, vmax=100.0)

    if SAVE_FIGS:
        out_path = out_dir / f"panel_line_saturation_{case_name}_root{root}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

def main():
    # Salida en escritorio
    out_dir = None
    desks = candidate_desktops()
    if not desks:
        raise RuntimeError("No encuentro Desktop/Escritorio en rutas típicas.")
    out_dir = desks[0] / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Casos a evaluar: los que estén en root_map (así no inventamos)
    # (Si tienes 5 study_cases, añade los 2 que faltan al root_map y listo)
    for case_name in list(root_map.keys()):
        run_one_case(case_name, out_dir)

    # Guardados
    # (No imprimimos nada extra si no hay anomalías; las figuras sí se guardan.)

if __name__ == "__main__":
    main()
