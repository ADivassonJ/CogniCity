"""
Standalone: plot network (nodes+edges) on a map and highlight "substation candidates".

Loads:
- node_data_{study_case}.csv with columns: lat, lon, i   (i can be 0 or "Node_0")
- {study_case}_line_data.csv with columns: i, j          (same)

Does:
1) normalize node ids to strings like "Node_###"
2) build undirected graph
3) compute degree + (if available) networkx centralities
4) plot edges/nodes on OSM basemap (if contextily+pyproj installed), else lon/lat
5) mark Top-K candidate hub nodes

Requirements:
- pandas, numpy, matplotlib
Optional:
- networkx (better hub detection)
- contextily, pyproj (OSM basemap)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER CONFIG
# -----------------------------
study_case = "Aradas"
path = Path(r"C:/Users/asier.divasson/Documents/GitHub/CogniCity/data") / study_case / "population"

nodes_csv = path / f"node_data_{study_case}.csv"
edges_csv = path / f"{study_case}_line_data.csv"

TOP_K_CANDIDATES = 10

# -----------------------------
# Helpers
# -----------------------------
def to_node_label(x) -> str:
    """Normalize any id to 'Node_<int>' if possible, else keep string with Node_ prefix."""
    s = str(x).strip()
    if s.startswith("Node_"):
        return s
    # try numeric -> Node_int
    try:
        # handle "0.0" too
        v = int(float(s))
        return f"Node_{v}"
    except Exception:
        # fallback: keep as Node_<raw>
        return f"Node_{s}"

# -----------------------------
# Load data
# -----------------------------
nodes = pd.read_csv(nodes_csv)
edges = pd.read_csv(edges_csv)

required_nodes_cols = {"lat", "lon", "i"}
required_edges_cols = {"i", "j"}
if not required_nodes_cols.issubset(nodes.columns):
    raise ValueError(f"nodes file must contain {required_nodes_cols}, got {set(nodes.columns)}")
if not required_edges_cols.issubset(edges.columns):
    raise ValueError(f"edges file must contain {required_edges_cols}, got {set(edges.columns)}")

nodes = nodes.dropna(subset=["lat", "lon", "i"]).copy()
edges = edges.dropna(subset=["i", "j"]).copy()

# Normalize node ids to labels "Node_k"
nodes["node"] = nodes["i"].apply(to_node_label)
edges["u"] = edges["i"].apply(to_node_label)
edges["v"] = edges["j"].apply(to_node_label)

# Build coordinate map (node_label -> (lon,lat))
coord = dict(zip(nodes["node"], zip(nodes["lon"].astype(float), nodes["lat"].astype(float))))

# Keep only edges whose endpoints exist in coord
mask = edges["u"].isin(coord.keys()) & edges["v"].isin(coord.keys())
edges = edges.loc[mask].copy()

# -----------------------------
# Build graph
# -----------------------------
# Adjacency + degree
adj = {n: [] for n in coord.keys()}
for u, v in zip(edges["u"].values, edges["v"].values):
    adj[u].append(v)
    adj[v].append(u)

deg = {n: len(adj[n]) for n in adj.keys()}

# -----------------------------
# Candidate scoring
# -----------------------------
# Try networkx for betweenness/closeness; fallback to degree-only
centrality_available = True
try:
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(list(coord.keys()))
    G.add_edges_from(list(zip(edges["u"].values, edges["v"].values)))

    n_nodes = G.number_of_nodes()
    # betweenness: exact for small graphs, approximate for large
    if n_nodes <= 800:
        betw = nx.betweenness_centrality(G, normalized=True)
    else:
        k = min(200, n_nodes)
        betw = nx.betweenness_centrality(G, normalized=True, k=k, seed=42)

    close = nx.closeness_centrality(G)

    # normalized degree
    deg_max = max(deg.values()) if deg else 1
    deg_norm = {n: (deg[n] / deg_max if deg_max > 0 else 0.0) for n in coord.keys()}

    # Score: tweak weights as needed
    score = {
        n: (2.0 * deg_norm.get(n, 0.0)) + (3.0 * betw.get(n, 0.0)) + (0.5 * close.get(n, 0.0))
        for n in coord.keys()
    }

except Exception:
    centrality_available = False
    score = {n: float(deg.get(n, 0)) for n in coord.keys()}

top_candidates = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_CANDIDATES]
cand_nodes = [n for n, _ in top_candidates]

print("\nTop candidate nodes (hubs / possible substation locations):")
for n, sc in top_candidates:
    print(f"  {n}: score={sc:.6f}, degree={deg.get(n,0)}")

if not centrality_available:
    print("\nNote: networkx not available (or failed). Using degree-only ranking.")
    print("Install networkx for better results: pip install networkx")

# -----------------------------
# Plot on basemap if available
# -----------------------------
use_basemap = True
try:
    import contextily as cx
    from pyproj import Transformer
except Exception:
    use_basemap = False

lon = nodes["lon"].astype(float).to_numpy()
lat = nodes["lat"].astype(float).to_numpy()

if use_basemap:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    X, Y = transformer.transform(lon, lat)
    nodes_plot = nodes.copy()
    nodes_plot["x"] = X
    nodes_plot["y"] = Y
    coord_plot = dict(zip(nodes_plot["node"], zip(nodes_plot["x"], nodes_plot["y"])))
else:
    nodes_plot = nodes.copy()
    nodes_plot["x"] = lon
    nodes_plot["y"] = lat
    coord_plot = dict(zip(nodes_plot["node"], zip(nodes_plot["x"], nodes_plot["y"])))

fig, ax = plt.subplots(figsize=(11, 11))

# edges
for u, v in zip(edges["u"].values, edges["v"].values):
    x1, y1 = coord_plot[u]
    x2, y2 = coord_plot[v]
    ax.plot([x1, x2], [y1, y2], linewidth=1.1, alpha=0.65)

# nodes
ax.scatter(nodes_plot["x"], nodes_plot["y"], s=14, alpha=0.85, zorder=3)

# candidates
cand_xy = np.array([coord_plot[n] for n in cand_nodes], dtype=float)
ax.scatter(cand_xy[:, 0], cand_xy[:, 1], s=180, marker="s", zorder=5)

for n in cand_nodes:
    x, y = coord_plot[n]
    ax.text(x, y, f"  {n}", fontsize=9, va="center", zorder=6)

# basemap
if use_basemap:
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
else:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

ax.set_title(
    f"Network map — {study_case} | Top-{TOP_K_CANDIDATES} hub candidates marked"
    + (" | OSM basemap" if use_basemap else " | (no basemap: install contextily+pyproj)"),
    pad=12
)
plt.tight_layout()
plt.show()
