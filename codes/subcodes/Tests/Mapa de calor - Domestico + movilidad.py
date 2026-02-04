# ============================================================
# LINE SATURATION HEATMAPS + STATS (avg/min/max/incidents)
# + MOBILITY SLOW-CHARGING DEMAND (from schedule_vehicle_quantified_24)
#
# For each study_case:
#   - load network + Dijkstra tree from root_map
#   - load base demand + line Pmax from Excel (Desktop autodetect)
# For each scenario s0..s4:
#   - compute mobility charging power by node/hour (7 days), take day 3 (We)
#   - add mobility demand to base pDemand, compute line flows & saturation
#   - plot 5-heatmap panel per study_case + print stats and incidents
# ============================================================

import os
import glob
import math
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
STUDY_CASES = ["Annelinn", "Aradas", "Kanaleneiland"]

root_map = {
    "Annelinn": "Node_43",
    "Aradas": "Node_20",
    "Kanaleneiland": "Node_67",
}

# --- Base demand Excel + lines Pmax (from Desktop autodetect)
SAT_THRESHOLD_PCT = 100.0

# If pDemand is in kW -> MW: set 1/1000.
# If pDemand already MW: set 1.0
DEMAND_SCALE_BASE = 1.0

# Mobility slow charging output is in kW (per hour bin). Convert to MW:
MOB_KW_TO_MW = 1.0 / 1000.0

# --- Mobility schedule input (results folder)
BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

# Mobility simulation parameters (as in your script 2)
DAY_NAMES_7 = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_TO_SHOW_INDEX = 2  # day 3 -> "We"

MJ_TO_KWH = 1.0 / 3.6
P_SLOW_KW = 3.7
SOC_THRESHOLD = 0.50
SOC_TARGET = 0.80

# --- Output
OUT_DIR_NAME = "heatmaps_line_saturation_plus_mobility"
SAVE_FIGS = False


# ============================================================
# DESKTOP AUTO (for base Excel + network CSVs)
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

def pick_best_excel(case_name: str):
    excels = []
    for d in candidate_desktops():
        excels += list(d.glob(f"*{case_name}*.xls*"))
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
    candidates = []
    for d in candidate_desktops():
        candidates += list(d.glob(f"*node*data*{case_name}*.csv"))
        candidates += list(d.glob(f"*{case_name}*node*data*.csv"))
        candidates += list(d.glob(f"*node_data_{case_name}.csv"))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0] if candidates else None

def pick_best_edges_csv(case_name: str):
    candidates = []
    for d in candidate_desktops():
        candidates += list(d.glob(f"*{case_name}*line*data*.csv"))
        candidates += list(d.glob(f"*line*data*{case_name}*.csv"))
        candidates += list(d.glob(f"*{case_name}_line_data.csv"))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0] if candidates else None


# ============================================================
# HELPERS: labels + graph + Dijkstra
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
# INPUTS: base demand + Pmax
# ============================================================
def load_pmax_map(pPmax_line_df: pd.DataFrame):
    pmax = {}
    for _, r in pPmax_line_df.iterrows():
        a = to_node_label(r["level_0"])
        b = to_node_label(r["level_1"])
        pmax[edge_key(a, b)] = float(r["values"])
    return pmax

def load_hourly_demand_base(pDemand_df: pd.DataFrame):
    df = pDemand_df.copy()
    df["Buses"] = df["Buses"].astype(str).map(to_node_label)
    df["values"] = pd.to_numeric(df["values"], errors="coerce").fillna(0.0) * float(DEMAND_SCALE_BASE)

    g = df.groupby(["Time", "Buses"])["values"].sum()

    # We want 0..23 always for saturation heatmaps
    hours = list(range(24))
    demand_by_time = {}
    for t in hours:
        if t in g.index.get_level_values(0):
            s = g.loc[t]
            demand_by_time[t] = s.to_dict()
        else:
            demand_by_time[t] = {}
    return hours, demand_by_time


# ============================================================
# MOBILITY: schedule -> slow charging demand by node/hour (kW)
# ============================================================
def parse_time_slot_to_minutes(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series.astype(str).str.slice(0, 5), format="%H:%M", errors="coerce")
    return (ts.dt.hour * 60 + ts.dt.minute).astype("float")

def find_mobility_excel(base_path: str, s_folder: str, study_case: str) -> str:
    folder = os.path.join(base_path, s_folder, study_case)
    pattern = os.path.join(folder, f"*{study_case}_schedule_vehicle_quantified_24*.xlsx")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No mobility Excel found in: {folder}\nPattern: {pattern}")
    files = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)
    if len(files) > 1:
        print(f"[WARN] Multiple mobility Excels in {folder}. Using most recent:\n  {files[0]}")
    return files[0]

def compute_agents_home_markers_by_day(xlsx_path: str, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    required = {"time_slot", "agent", "archetype", "todo", "mjkm", "node", "day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{xlsx_path}] Missing columns: {missing}. Available: {list(df.columns)}")

    df = df[df["archetype"] == "PC_electric"].copy()

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

    return first_home_in[["agent", "day", "start_min", "start_hour", "node_label", "e_daily_kwh"]]

def ensure_7_days(markers: pd.DataFrame, day_names: list[str]) -> pd.DataFrame:
    present = sorted(markers["day"].unique().tolist())
    if len(present) >= len(day_names):
        return markers.copy()
    pattern_day = present[0]
    base = markers[markers["day"] == pattern_day].copy()
    return pd.concat([base.assign(day=d) for d in day_names], ignore_index=True)

def assign_battery_and_initial_soc(
    agents_markers: pd.DataFrame,
    soc_min=0.50, soc_max=0.80,
    batt_mean_kwh=60.0, batt_sd_kwh=10.0,
    batt_min_kwh=30.0, batt_max_kwh=100.0,
    seed=123,
):
    rng = np.random.default_rng(seed)
    out = agents_markers.copy()

    batt = rng.normal(loc=batt_mean_kwh, scale=batt_sd_kwh, size=len(out))
    batt = np.clip(batt, batt_min_kwh, batt_max_kwh)
    out["battery_kwh"] = batt

    out["soc0"] = rng.uniform(low=soc_min, high=soc_max, size=len(out))
    return out

def simulate_fleet_soc_rule_by_node_slow(
    agents_day: pd.DataFrame,
    day_names: list[str],
    soc_threshold: float,
    soc_target: float,
    p_slow_kw: float,
) -> pd.DataFrame:
    day_to_index = {d: i for i, d in enumerate(day_names)}
    horizon = len(day_names) * 24

    df = agents_day.copy()
    df = df[df["day"].isin(day_to_index)].copy()
    df["day_index"] = df["day"].map(day_to_index).astype(int)

    agents = df["agent"].unique().tolist()

    soc0_map = df.groupby("agent")["soc0"].first().to_dict()
    batt_map = df.groupby("agent")["battery_kwh"].first().to_dict()
    soc_state = {a: float(soc0_map.get(a, 0.65)) for a in agents}

    fleet_slow = {}

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

                remaining = e_need
                for t in range(t_arr, horizon):
                    if remaining <= 0:
                        break
                    e_can = p_slow_kw * 1.0
                    e_this = min(remaining, e_can)
                    fleet_slow[node][t] += e_this
                    remaining -= e_this

                soc = soc_target

        soc_state[a] = soc

    rows = []
    for node in sorted(fleet_slow.keys()):
        slow = fleet_slow[node]
        for t in range(horizon):
            rows.append((t, t // 24, t % 24, day_names[t // 24], node, slow[t]))

    return pd.DataFrame(rows, columns=["t", "day_index", "hour", "day", "node_label", "P_slow_kW"])

def mobility_demand_by_hour_MW(base_path: str, s_folder: str, study_case: str, day_index: int) -> dict:
    """
    Returns dict[hour] -> dict[node_label] = mobility MW demand.
    Day is selected by day_index (0..6).
    """
    xlsx_path = find_mobility_excel(base_path, s_folder, study_case)
    markers = compute_agents_home_markers_by_day(xlsx_path, sheet_name=0)
    markers_7d = ensure_7_days(markers, DAY_NAMES_7)
    agents_day = assign_battery_and_initial_soc(markers_7d, seed=123)

    fleet_node_ts = simulate_fleet_soc_rule_by_node_slow(
        agents_day=agents_day,
        day_names=DAY_NAMES_7,
        soc_threshold=SOC_THRESHOLD,
        soc_target=SOC_TARGET,
        p_slow_kw=P_SLOW_KW,
    )

    df_day = fleet_node_ts[fleet_node_ts["day_index"] == int(day_index)].copy()
    if df_day.empty:
        return {h: {} for h in range(24)}

    agg = df_day.groupby(["hour", "node_label"], as_index=False)["P_slow_kW"].sum()

    out = {h: {} for h in range(24)}
    for _, r in agg.iterrows():
        h = int(r["hour"])
        node = str(r["node_label"])
        mw = float(r["P_slow_kW"]) * MOB_KW_TO_MW
        out[h][node] = out[h].get(node, 0.0) + mw

    return out


# ============================================================
# SATURATION: flow radial + saturation by line/hour
# ============================================================
def compute_line_flows_and_saturation(hours, demand_by_time, universe, parent, root, pmax_map):
    children, post = build_children_and_postorder(parent, root=root)

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
# STATS + PLOT
# ============================================================
def sat_stats_overall(satM: np.ndarray) -> dict:
    x = np.asarray(satM, dtype=float)
    if np.all(np.isnan(x)):
        return {"avg": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "avg": float(np.nanmean(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
    }

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
# CASE RUNNER
# ============================================================
def add_demands(base_by_time: dict, mob_by_time: dict) -> dict:
    out = {}
    for h in range(24):
        db = base_by_time.get(h, {})
        dm = mob_by_time.get(h, {})
        keys = set(db.keys()) | set(dm.keys())
        out[h] = {k: float(db.get(k, 0.0)) + float(dm.get(k, 0.0)) for k in keys}
    return out

def run_one_case(case_name: str, out_dir: Path):
    root = root_map.get(case_name)
    if not root:
        print(f"[WARN] case '{case_name}' not in root_map. Skipping.")
        return

    nodes_csv = pick_best_nodes_csv(case_name)
    edges_csv = pick_best_edges_csv(case_name)
    excel_path = pick_best_excel(case_name)

    if not nodes_csv or not edges_csv or not excel_path:
        print(f"[WARN] Missing inputs for '{case_name}' (nodes/edges/excel). Skipping.")
        return

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    adj, _ = build_weighted_graph(nodes, edges)
    dist, parent = dijkstra_tree(adj, root=root)
    universe = sorted(dist.keys(), key=lambda n: (dist[n], n))

    xls = pd.ExcelFile(excel_path)
    pDemand = pd.read_excel(xls, "pDemand")
    pPmax_line = pd.read_excel(xls, "pPmax_line")

    hours, demand_by_time_base = load_hourly_demand_base(pDemand)
    pmax_map = load_pmax_map(pPmax_line)

    mats = {}
    anomalies_by_s = {}
    stats_rows = []

    for s in S_FOLDERS:
        # Mobility demand (MW) for this scenario and selected day
        try:
            mob_by_time = mobility_demand_by_hour_MW(
                base_path=BASE_PATH,
                s_folder=s,
                study_case=case_name,
                day_index=DAY_TO_SHOW_INDEX,
            )
        except Exception as e:
            print(f"[WARN] Mobility failed for {case_name}/{s}: {e}")
            mob_by_time = {h: {} for h in range(24)}

        demand_total = add_demands(demand_by_time_base, mob_by_time)

        line_names, satM, anomalies = compute_line_flows_and_saturation(
            hours=hours,
            demand_by_time=demand_total,
            universe=universe,
            parent=parent,
            root=root,
            pmax_map=pmax_map,
        )

        mats[s] = {"M": satM, "y_labels": line_names}
        anomalies_by_s[s] = anomalies

        st = sat_stats_overall(satM)
        stats_rows.append({
            "case": case_name,
            "root": root,
            "system": s,
            "avg_saturation_pct": st["avg"],
            "min_saturation_pct": st["min"],
            "max_saturation_pct": st["max"],
            "incidents_hours_count": len(anomalies),
        })

    stats_df = pd.DataFrame(stats_rows).sort_values(["case", "system"])

    print("\n" + "-" * 120)
    print(f"RESUMEN SATURACIÓN (pDemand + movilidad slow-charging) | CASO: {case_name} | ROOT: {root} | Day: {DAY_NAMES_7[DAY_TO_SHOW_INDEX]}")
    print(f"Base Excel: {excel_path.name} | Nodes: {nodes_csv.name} | Lines: {edges_csv.name}")
    print(f"Mobility: BASE_PATH='{BASE_PATH}' | P_slow={P_SLOW_KW} kW | SoC thr={SOC_THRESHOLD} -> target={SOC_TARGET}")
    print("-" * 120)
    print(
        stats_df[
            ["system", "avg_saturation_pct", "min_saturation_pct", "max_saturation_pct", "incidents_hours_count"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:7.2f}"
        )
    )
    print("-" * 120)

    # Incidents
    any_inc = any(len(anomalies_by_s[s]) > 0 for s in S_FOLDERS)
    if any_inc:
        print("\n" + "=" * 120)
        print(f"INCIDENTES (pct >= {SAT_THRESHOLD_PCT:.1f}%) | CASO: {case_name} | ROOT: {root} | Day: {DAY_NAMES_7[DAY_TO_SHOW_INDEX]}")
        print("=" * 120)
        for s in S_FOLDERS:
            an = anomalies_by_s[s]
            if not an:
                continue
            print(f"\n--- {s} ---")
            for t in sorted(an.keys()):
                hits = sorted(an[t], key=lambda x: x[1], reverse=True)
                print(f"{int(t):02d}:00 -> {len(hits)} línea(s) (top 8)")
                for (lname, pct, flow, pmax) in hits[:8]:
                    print(f"  - {lname}: {pct:.1f}%  (P={flow:.3f} MW, Pmax={pmax:.3f} MW)")

    # Heatmaps
    title = f"{case_name} — Saturación líneas [%] (cap 100) — root {root} — + movilidad ({DAY_NAMES_7[DAY_TO_SHOW_INDEX]})"
    fig = plot_heatmap_panel_5(mats=mats, s_order=S_FOLDERS, title=title, vmin=0.0, vmax=100.0)

    if SAVE_FIGS:
        out_path = out_dir / f"panel_line_saturation_plus_mob_{case_name}_day{DAY_TO_SHOW_INDEX+1}_root{root}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

        stats_out = out_dir / f"stats_line_saturation_plus_mob_{case_name}_day{DAY_TO_SHOW_INDEX+1}_root{root}.csv"
        stats_df.to_csv(stats_out, index=False)
        print("Saved:", out_path)
        print("Saved:", stats_out)


def main():
    desks = candidate_desktops()
    if not desks:
        raise RuntimeError("No encuentro Desktop/Escritorio en rutas típicas.")
    out_dir = desks[0] / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_name in STUDY_CASES:
        run_one_case(case_name, out_dir)


if __name__ == "__main__":
    main()
