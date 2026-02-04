# ============================================================
# CUMULATIVE SLOW-CHARGING HEATMAP (upstream aggregation) — 7 days — show day 3
# Comparative panel: 5 heatmaps (s0..s4) IN A SINGLE FIGURE
# -> ONE shared colorbar (gradient legend) per study_case
#
# ROOT per study case (as specified):
#   - Annelinn: Node_43
#   - Aradas: Node_20
#   - Kanaleneiland: Node_67
# ============================================================

import os
import glob
import math
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"
S_FOLDERS = [f"s{i}" for i in range(5)]
STUDY_CASES = ["Annelinn", "Aradas", "Kanaleneiland"]

# ✅ Correct ROOT per study case (do NOT assume same root for all)
root_map = {
    "Annelinn": "Node_43",
    "Aradas": "Node_20",
    "Kanaleneiland": "Node_67",
}

DAY_NAMES_7 = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_TO_SHOW_INDEX = 2  # day 3 = "We"

MJ_TO_KWH = 1.0 / 3.6
P_SLOW_KW = 3.7
SOC_THRESHOLD = 0.50
SOC_TARGET = 0.80

SHOW_NETWORK_ONCE = True
SAVE_FIGS = True
OUT_DIR = Path.home() / "Desktop" / "heatmaps_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== Helpers =====================
def to_node_label(x):
    """Normalize node identifiers to 'Node_<id>' format."""
    s = str(x).strip()
    return s if s.startswith("Node_") else f"Node_{s}"

def parse_time_slot_to_minutes(series: pd.Series) -> pd.Series:
    """
    Parse HH:MM strings (first 5 chars) into minutes since midnight.
    Returns float dtype with NaNs for unparsable values (later dropped).
    """
    ts = pd.to_datetime(series.astype(str).str.slice(0, 5), format="%H:%M", errors="coerce")
    return (ts.dt.hour * 60 + ts.dt.minute).astype("float")

def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) used as edge weight in the network graph."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_mobility_excel(base_path: str, s_folder: str, study_case: str) -> str:
    """
    Locate the mobility schedule Excel for a given (scenario folder, study_case).
    Uses a filename pattern and picks the most recently modified file if multiple match.
    """
    folder = os.path.join(base_path, s_folder, study_case)
    pattern = os.path.join(folder, f"*{study_case}_schedule_vehicle_quantified_24*.xlsx")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No Excel (quantified_24) found in: {folder}\nPattern: {pattern}")
    files = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)
    if len(files) > 1:
        print(f"[WARN] Multiple Excels in {folder}. Using most recent:\n  {files[0]}")
    return files[0]

# ===================== 1) Home-arrival markers per agent and day =====================
def compute_agents_home_markers_by_day(xlsx_path: str, sheet_name=0) -> pd.DataFrame:
    """
    Read mobility schedule and extract, for each (agent, day), the FIRST 'Home_in' event.
    This defines:
      - arrival time (minutes + hour bin),
      - the node where the agent arrives,
      - energy consumed at that marker (MJ -> kWh).
    Only considers archetype == 'PC_electric'.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    required = {"time_slot", "agent", "archetype", "todo", "mjkm", "node", "day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{xlsx_path}] Missing columns: {missing}. Available: {list(df.columns)}")

    # Filter to electric passenger cars only
    df = df[df["archetype"] == "PC_electric"].copy()

    # Robust conversion of numeric strings that may use comma decimals
    df["mjkm"] = df["mjkm"].astype(str).str.replace(",", ".", regex=False)
    df["mjkm"] = pd.to_numeric(df["mjkm"], errors="coerce").fillna(0.0)

    # Time parsing to integer minutes since midnight
    df["time_slot_min"] = parse_time_slot_to_minutes(df["time_slot"])
    df = df.dropna(subset=["time_slot_min"]).copy()
    df["time_slot_min"] = df["time_slot_min"].astype(int)

    # First home arrival per agent/day
    home_in = df[df["todo"] == "Home_in"].copy()
    home_in = home_in.sort_values(["agent", "day", "time_slot_min"], ascending=[True, True, True])

    first_home_in = (
        home_in.groupby(["agent", "day"], as_index=False)
        .first()[["agent", "day", "time_slot_min", "node", "mjkm"]]
        .rename(columns={"time_slot_min": "start_min", "mjkm": "mj_consumed_at_marker"})
    )

    # Hour bin for hourly simulation
    first_home_in["start_hour"] = (first_home_in["start_min"] // 60).astype(int)
    first_home_in["node_label"] = first_home_in["node"].apply(to_node_label)

    # Energy unit conversion MJ -> kWh
    first_home_in["e_daily_kwh"] = first_home_in["mj_consumed_at_marker"] * MJ_TO_KWH

    return first_home_in[["agent", "day", "start_min", "start_hour", "node_label", "e_daily_kwh"]]

def ensure_7_days(markers: pd.DataFrame, day_names: list[str]) -> pd.DataFrame:
    """
    If the schedule does not contain a full 7-day week, replicate a representative day
    (the first day present) across the full set of day_names.
    """
    present = sorted(markers["day"].unique().tolist())
    if len(present) >= len(day_names):
        return markers.copy()
    pattern_day = present[0]
    base = markers[markers["day"] == pattern_day].copy()
    return pd.concat([base.assign(day=d) for d in day_names], ignore_index=True)

# ===================== 2) Battery capacity + initial SoC =====================
def assign_battery_and_initial_soc(
    agents_markers: pd.DataFrame,
    soc_min=0.50, soc_max=0.80,
    batt_mean_kwh=60.0, batt_sd_kwh=10.0,
    batt_min_kwh=30.0, batt_max_kwh=100.0,
    seed=123,
):
    """
    Assign per-row (agent/day marker) battery parameters:
      - battery_kwh ~ Normal(mean, sd), clipped
      - initial soc0 ~ Uniform(soc_min, soc_max)

    NOTE: The same agent may appear multiple days; downstream code uses the FIRST
          soc0/battery_kwh per agent for the full simulation horizon.
    """
    rng = np.random.default_rng(seed)
    out = agents_markers.copy()

    batt = rng.normal(loc=batt_mean_kwh, scale=batt_sd_kwh, size=len(out))
    batt = np.clip(batt, batt_min_kwh, batt_max_kwh)
    out["battery_kwh"] = batt

    out["soc0"] = rng.uniform(low=soc_min, high=soc_max, size=len(out))
    return out

# ===================== 3) Local SLOW charging simulation (node/hour) =====================
def simulate_fleet_soc_rule_by_node_slow(
    agents_day: pd.DataFrame,
    day_names: list[str],
    soc_threshold: float,
    soc_target: float,
    p_slow_kw: float,
) -> pd.DataFrame:
    """
    For each agent, iterate across their home-arrival markers (by day order).
    At each arrival:
      1) Decrease SoC by energy consumed (e_daily_kwh / battery_kwh)
      2) If SoC < soc_threshold, charge from arrival hour onward at constant power p_slow_kw
         until reaching soc_target.

    Outputs an hourly time series (7*24 rows per node) of 'P_slow_kW' (kWh/h).
    """
    day_to_index = {d: i for i, d in enumerate(day_names)}
    horizon = len(day_names) * 24

    df = agents_day.copy()
    df = df[df["day"].isin(day_to_index)].copy()
    df["day_index"] = df["day"].map(day_to_index).astype(int)

    agents = df["agent"].unique().tolist()

    # Use first values per agent (consistent across multiple days)
    soc0_map = df.groupby("agent")["soc0"].first().to_dict()
    batt_map = df.groupby("agent")["battery_kwh"].first().to_dict()
    soc_state = {a: float(soc0_map.get(a, 0.65)) for a in agents}

    # Node -> hourly kWh array (length horizon)
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

            # Arrival hour index in global horizon
            t_arr = d * 24 + (start_min // 60)
            if t_arr >= horizon:
                continue

            # Energy consumption reduces SoC
            soc -= (e_daily / batt)
            soc = max(soc, 0.0)

            # Rule-based charging trigger
            if soc < soc_threshold:
                e_need = max(0.0, (soc_target - soc) * batt)
                if e_need <= 0:
                    continue

                if node not in fleet_slow:
                    fleet_slow[node] = np.zeros(horizon, dtype=float)

                # Charge forward in time at constant power until full target reached
                remaining = e_need
                for t in range(t_arr, horizon):
                    if remaining <= 0:
                        break
                    e_can = p_slow_kw * 1.0  # kWh that can be delivered in 1 hour
                    e_this = min(remaining, e_can)
                    fleet_slow[node][t] += e_this
                    remaining -= e_this

                # After charging, set SoC to target (no partial target retained)
                soc = soc_target

        soc_state[a] = soc

    # Flatten into long-form DataFrame: (t, day_index, hour, day, node_label, P_slow_kW)
    rows = []
    for node in sorted(fleet_slow.keys()):
        slow = fleet_slow[node]
        for t in range(horizon):
            rows.append((t, t // 24, t % 24, day_names[t // 24], node, slow[t]))

    return pd.DataFrame(rows, columns=["t", "day_index", "hour", "day", "node_label", "P_slow_kW"])

# ===================== 4) Network -> Dijkstra shortest-path tree =====================
def build_weighted_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    Build an undirected adjacency list with edge weights = geographic distance (km).
    Requires nodes to have (lon, lat) and a node identifier column:
      - if 'node_label' exists, use it
      - else if 'i' exists, convert with to_node_label
    """
    nodes = nodes.copy()
    if "node_label" not in nodes.columns:
        if "i" in nodes.columns:
            nodes["node_label"] = nodes["i"].apply(to_node_label)
        else:
            raise ValueError("nodes must have either 'i' or 'node_label'")

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
            # Fallback if coordinates are missing; keeps graph connected but weightless
            w = 0.0
        adj[a].append((b, w))
        adj[b].append((a, w))

    return adj, coord_map

def dijkstra_tree(adj, root: str):
    """
    Standard Dijkstra from root.
    Returns:
      - dist[node] = shortest path distance (km) from root
      - parent[node] = predecessor on shortest path tree
    """
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
    """
    Convert parent pointers to children lists and compute a post-order traversal.
    Post-order is used to aggregate descendant contributions upward (bottom-up).
    """
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

def ancestors_closure(parent: dict, root: str, targets: list[str]) -> list[str]:
    """
    Utility to get the set of nodes required to connect each target to the root
    (i.e., all ancestors on the parent chain). Not used in main flow currently.
    """
    keep = set([root])
    for t in targets:
        u = t
        while u is not None and u not in keep:
            keep.add(u)
            u = parent.get(u, None)
    return list(keep)

# ===================== 5) Network plot (optional) =====================
def plot_network(coord_map: dict, edges: pd.DataFrame, root: str, dist: dict,
                 charging_nodes: set[str], parent: dict, title: str):
    """
    Diagnostic plot:
      - draw all edges as lines
      - color nodes by distance-to-root (km)
      - mark root with a star
      - circle nodes that have any charging event across scenarios
      - draw directed arrows following the Dijkstra tree (parent pointers)
    """
    plt.figure(figsize=(9, 9))

    # Draw edges
    for _, r in edges.iterrows():
        a = to_node_label(r["i"])
        b = to_node_label(r["j"])
        if a in coord_map and b in coord_map:
            x1, y1 = coord_map[a]
            x2, y2 = coord_map[b]
            plt.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.6)

    # Scatter nodes colored by dist to root
    nodes_list = list(coord_map.keys())
    xs = [coord_map[n][0] for n in nodes_list]
    ys = [coord_map[n][1] for n in nodes_list]
    dvals = np.array([dist.get(n, np.nan) for n in nodes_list], dtype=float)
    sc = plt.scatter(xs, ys, c=dvals, s=25, zorder=3)
    cb = plt.colorbar(sc)
    cb.set_label("Distance to root (km)")

    # Root marker
    if root in coord_map:
        rx, ry = coord_map[root]
        plt.scatter([rx], [ry], s=140, marker="*", zorder=6)
        plt.text(rx, ry, f"  {root}", fontsize=10, va="center", zorder=7)

    # Mark nodes where charging happens (union across scenarios)
    ch = [n for n in charging_nodes if n in coord_map]
    if ch:
        cx = [coord_map[n][0] for n in ch]
        cy = [coord_map[n][1] for n in ch]
        plt.scatter(cx, cy, s=90, facecolors="none", edgecolors="black", linewidths=1.5, zorder=7)

    # Draw arrows for the Dijkstra tree (parent -> child)
    for child, p in parent.items():
        if p is None:
            continue
        if p in coord_map and child in coord_map:
            x1, y1 = coord_map[p]
            x2, y2 = coord_map[child]
            plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.35),
                         zorder=2)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# ===================== 6) Local + cumulative matrices =====================
def make_local_matrix_day(fleet_node_ts: pd.DataFrame, ordered_buses: list[str], day_index: int):
    """
    Build the node-by-hour matrix for a given day:
      - rows: ordered_buses (the full reachable set from root, ordered by dist)
      - cols: 0..23 hours
      - value: sum of P_slow_kW for that node/hour (local, not accumulated)
    """
    df = fleet_node_ts[fleet_node_ts["day_index"] == day_index].copy()
    agg = df.groupby(["node_label", "hour"], as_index=False)["P_slow_kW"].sum()
    piv = agg.pivot(index="node_label", columns="hour", values="P_slow_kW").fillna(0.0)

    # Ensure all 24 hours exist as columns
    for h in range(24):
        if h not in piv.columns:
            piv[h] = 0.0
    piv = piv[list(range(24))]

    # Reindex to the full bus universe (includes nodes with zero charging)
    piv = piv.reindex(ordered_buses).fillna(0.0)
    return piv.to_numpy(dtype=float), piv.index.tolist()

def accumulate_upstream_subtree(M_local: np.ndarray, y_labels: list[str], root: str, parent: dict):
    """
    Accumulate power "upstream" along the Dijkstra tree:
      - For each node u, M_cum[u,:] = M_local[u,:] + sum_{child in subtree(u)} M_cum[child,:]

    Implementation detail:
      - Build children lists from parent pointers
      - Post-order traversal ensures children are processed before parents (bottom-up)
    """
    idx = {n: i for i, n in enumerate(y_labels)}
    children, post = build_children_and_postorder(parent, root=root)
    M_cum = M_local.copy()
    for u in post:
        if u not in idx:
            continue
        ui = idx[u]
        for ch in children.get(u, []):
            if ch not in idx:
                continue
            ci = idx[ch]
            M_cum[ui, :] += M_cum[ci, :]
    return M_cum

# ===================== 7) Panel with 5 heatmaps (s0..s4) + 1 shared colorbar =====================
def plot_heatmap_panel_5(mats: dict, s_order: list[str], title: str, vmin: float, vmax: float):
    """
    Plot 5 heatmaps (s0..s4) in a single row plus ONE shared colorbar.
    The colorbar axis is explicitly placed so its height matches the heatmaps.
    """
    max_buses = max(mats[s]["M"].shape[0] for s in s_order)

    # Figure sizing heuristics based on number of buses
    fig_height = max(8, 0.22 * max_buses)
    one_w = 24 * 0.22
    cbar_w = 0.35  # colorbar column width

    fig = plt.figure(figsize=(one_w * 5 + cbar_w, fig_height))

    # GridSpec: 5 heatmaps + 1 colorbar column
    gs = fig.add_gridspec(
        nrows=1,
        ncols=6,
        width_ratios=[1, 1, 1, 1, 1, 0.05],
        wspace=0.15
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax = fig.add_subplot(gs[0, 5])  # dedicated axis for the colorbar

    mappable = None

    for ax, s in zip(axes, s_order):
        M = mats[s]["M"]
        y_labels = mats[s]["y_labels"]

        im = ax.imshow(
            M,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        mappable = im

        ax.set_title(s, fontsize=11)

        # X axis: hours 0..23 (tick every 2 hours)
        ax.set_xlim(-0.5, 23.5)
        even_hours = np.arange(0, 24, 2)
        ax.set_xticks(even_hours)
        ax.set_xticklabels([str(h) for h in even_hours], fontsize=9)

        ax.set_xlabel("Time")

        # Only left-most subplot shows y tick labels (buses) for readability
        if s == s_order[0]:
            ax.set_ylabel("Bus")
            ax.set_ylim(-0.5, len(y_labels) - 0.5)
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels([lab.replace("Node_", "") for lab in y_labels], fontsize=7)
        else:
            ax.set_yticks([])

    # Shared colorbar spanning the heatmaps' height
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Accumulated power (subtree) [kW]")

    fig.suptitle(title, fontsize=13)
    plt.show()

    return fig


# ===================== MAIN per study_case =====================
def run_one_study_case(study_case: str):
    """
    For a given study case:
      1) Load network (nodes/edges), build weighted graph, compute Dijkstra tree from root.
      2) For each scenario s0..s4:
         - load mobility Excel
         - extract home-arrival markers
         - assign battery + initial SoC
         - simulate slow charging rule across 7 days (hourly)
         - build local matrix for day 3 and accumulate over subtree (upstream)
      3) Optionally plot the network and save it.
      4) Plot a 5-heatmap panel with one shared colorbar and save it.
    """
    root = root_map[study_case]  # ✅ root depends on study_case

    # --- Load network ---
    net_path = rf"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\{study_case}\population"
    nodes_csv = rf"{net_path}\node_data_{study_case}.csv"
    edges_csv = rf"{net_path}\{study_case}_line_data.csv"
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    adj, coord_map = build_weighted_graph(nodes, edges)
    dist, parent = dijkstra_tree(adj, root=root)

    # Full set of reachable buses from root (includes non-charging nodes)
    universe = sorted(dist.keys(), key=lambda n: (dist[n], n))

    mats = {}
    vmax_local = 0.0
    charging_union = set()  # union of charging nodes across scenarios (for network plot)

    for s in S_FOLDERS:
        # --- Load mobility + build agent markers ---
        xlsx_path = find_mobility_excel(BASE_PATH, s, study_case)
        markers = compute_agents_home_markers_by_day(xlsx_path, sheet_name=0)
        markers_7d = ensure_7_days(markers, DAY_NAMES_7)

        # --- Assign stochastic battery parameters (seed fixed for reproducibility) ---
        agents_day = assign_battery_and_initial_soc(markers_7d, seed=123)

        # --- Simulate slow charging time series per node (7 days hourly) ---
        fleet_node_ts = simulate_fleet_soc_rule_by_node_slow(
            agents_day=agents_day,
            day_names=DAY_NAMES_7,
            soc_threshold=SOC_THRESHOLD,
            soc_target=SOC_TARGET,
            p_slow_kw=P_SLOW_KW,
        )

        # Nodes with any positive charging in this scenario
        charging_nodes = set(fleet_node_ts.loc[fleet_node_ts["P_slow_kW"] > 0, "node_label"].unique().tolist())
        charging_union |= charging_nodes

        # --- Build day-specific matrices ---
        M_local, y_labels = make_local_matrix_day(fleet_node_ts, universe, day_index=DAY_TO_SHOW_INDEX)

        # Accumulate power "upstream" (i.e., sum descendants in Dijkstra tree)
        M_cum = accumulate_upstream_subtree(M_local, y_labels, root=root, parent=parent)

        mats[s] = {"M": M_cum, "y_labels": y_labels, "xlsx": xlsx_path}
        vmax_local = max(vmax_local, float(M_cum.max()) if M_cum.size else 0.0)

        print(f"{study_case}/{s} root={root} -> buses={len(y_labels)} vmax={float(M_cum.max()):.3f}")

    # --- Optional: plot the network once per study_case ---
    if SHOW_NETWORK_ONCE:
        net_title = f"NETWORK — {study_case} — root {root} (rings=bus with any charging; arrows=Dijkstra tree)"
        plot_network(coord_map, edges, root, dist, charging_union, parent, net_title)
        if SAVE_FIGS:
            out_net = OUT_DIR / f"network_{study_case}_root{root}.png"
            plt.gcf().savefig(out_net, dpi=200, bbox_inches="tight")
            print("Saved:", out_net)

    # --- Heatmap panel title (kept blank in your code) ---
    panel_title = f" "
    fig = plot_heatmap_panel_5(
        mats=mats,
        s_order=S_FOLDERS,
        title=panel_title,
        vmin=0.0,
        vmax=vmax_local if vmax_local > 0 else 1.0,
    )

    # --- Save panel ---
    if SAVE_FIGS:
        out_panel = OUT_DIR / f"panel_heatmaps_{study_case}_day3_root{root}.png"
        fig.savefig(out_panel, dpi=200, bbox_inches="tight")
        print("Saved:", out_panel)

if __name__ == "__main__":
    for study_case in STUDY_CASES:
        print("\n===================================================")
        print("STUDY CASE:", study_case, "| ROOT:", root_map[study_case])
        print("===================================================")
        run_one_study_case(study_case)
