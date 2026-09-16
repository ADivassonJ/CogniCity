import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, box
from matplotlib.patches import Patch

# --------------------------------------------------------
# A) Cargar datos y FILTRAR solo PC_electric
# --------------------------------------------------------
study_case = "Kanaleneiland"  # Cambia a "Kanaleneiland" para el otro caso de estudio
scenario = "s0"                 # ajusta a "S0" si tu carpeta está en mayúsculas
archetype_folder = "PC_electric"  # carpeta del arquetipo dentro del escenario

if study_case == "Kanaleneiland":
    colors = {
        "s4": "#d0e0e3",
        "q1": "#a2c4c9",
        "s2": "#76a5af",
        "s1": "#45818e",
        "max": "#134f5c",
        "base": "#434343",
    }
elif study_case == "Annelinn":
        colors = {
            "s4": "#ead1dc",
            "q1": "#d5a6bd",
            "s2": "#c27ba0",
            "s1": "#a64d79",
            "max": "#741b47",
            "base": "#434343",
        }
elif study_case == "Aradas":
        colors = {
            "s4": "#ffe599",
            "q1": "#ffd966",
            "s2": "#f1c232",
            "s1": "#bf9000",
            "max": "#7f6000",
            "base": "#434343",
        }

project_root = Path(r"C:/Users/asier.divasson/Documents/GitHub/CogniCity")

# Estructura: data/results/<scenario>/<archetype>/<archivo>
results_dir = project_root / "results" / scenario

# Nombre del archivo: <study_case>_schedule_vehicle.xlsx
excel_filename = f"{study_case}_schedule_vehicle_quantified_24.xlsx"
excel_path = results_dir / excel_filename

df = pd.read_excel(excel_path)

# 🔴 FILTRO CLAVE: solo vehículos eléctricos
df = df[df["archetype"] == "PC_electric"].copy()

# --------------------------------------------------------
# B) Seleccionar último time_slot por agent y day
# --------------------------------------------------------
ts = pd.to_datetime(df["time_slot"], format="%H:%M", errors="coerce")
df["time_slot_minutes"] = ts.dt.hour * 60 + ts.dt.minute

df_last_per_agent = (
    df.sort_values(
        by=["agent", "day", "time_slot_minutes"],
        ascending=[True, True, False]
    )
    .groupby(["agent", "day"], as_index=False)
    .first()
    .drop(columns=["time_slot_minutes"])
)

# --------------------------------------------------------
# C) MJ (campo mjkm) -> float
#    (aunque se llame mjkm, aquí son MJ)
# --------------------------------------------------------
df_last_per_agent["mjkm"] = (
    df_last_per_agent["mjkm"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df_last_per_agent["mjkm"] = pd.to_numeric(
    df_last_per_agent["mjkm"], errors="coerce"
).fillna(0.0)

# --------------------------------------------------------
# D) Suma por node en MJ y conversión a kWh
# --------------------------------------------------------
df_mj_by_node = (
    df_last_per_agent
    .groupby("node", as_index=False)["mjkm"]
    .sum()
    .rename(columns={"mjkm": "MJ_total"})
)

# Conversión MJ -> kWh (1 kWh = 3.6 MJ)
df_mj_by_node["kWh_total"] = df_mj_by_node["MJ_total"] / 3.6

total_kwh = df_mj_by_node["kWh_total"].sum()

# Diccionario node -> kWh_total
# NOTA: el "node" del Excel de la agenda ES el osm_id del edificio/POI
# (así que kwh_map ya está indexado por osm_id, listo para el flujo
#  osm_id -> lat/lon (pop_building.parquet) -> malla)
kwh_map = dict(zip(df_mj_by_node["node"], df_mj_by_node["kWh_total"]))

# --------------------------------------------------------
# 1) Leer nodos y edges
# --------------------------------------------------------
path = f"C:/Users/asier.divasson/Documents/GitHub/CogniCity/data/{scenario}/{study_case}/population"
nodes = pd.read_csv(f"{path}/node_data_{study_case}.csv")   # lat, lon, i
edges = pd.read_csv(f"{path}/{study_case}_line_data.csv")   # i, j

# --------------------------------------------------------
# 2) Definir polígono (lat, lon) -> (lon, lat)
# --------------------------------------------------------

if study_case == "Kanaleneiland":
    #Kanaleneiland
    boundary_latlon = [(52.07893, 5.08148),
                       (52.08112, 5.09062),
                       (52.06719, 5.10358),
                       (52.06656, 5.10748),
                       (52.06553, 5.10796),
                       (52.06595, 5.10379),
                       (52.06308, 5.09525),
                       (52.07663, 5.08268),
                       (52.07893, 5.08148),]
elif study_case == "Ogg in Al":
    #Ogg in Al
    boundary_latlon =   [(52.09245, 5.08429),
                         (52.08747, 5.09429),
                         (52.08267, 5.08766),
                         (52.08337, 5.08549),
                         (52.08551, 5.08354),
                         (52.09003, 5.08270),
                         (52.09245, 5.08429),]

boundary_lonlat = [(lon, lat) for (lat, lon) in boundary_latlon]
boundary_poly = Polygon(boundary_lonlat)

# --------------------------------------------------------
# 3) Malla de celdas de 100x100 m, agregando por la ubicación
#    real de cada osm_id (pop_building.parquet), no por Voronoi
#    ni por la posición aproximada de los nodos de la red.
# --------------------------------------------------------
CRS_WGS84 = "EPSG:4326"
CELL_SIZE = 100.0  # metros

def utm_epsg_for(lon, lat):
    """Calcula el EPSG UTM (metros) correspondiente a un punto lat/lon."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

# CRS métrico local, calculado a partir del centroide del boundary
_centroid = boundary_poly.centroid
CRS_METRIC = utm_epsg_for(_centroid.x, _centroid.y)

# Boundary en CRS métrico (para construir la malla en metros reales)
boundary_gdf_wgs = gpd.GeoDataFrame(geometry=[boundary_poly], crs=CRS_WGS84)
boundary_gdf_m = boundary_gdf_wgs.to_crs(CRS_METRIC)
boundary_poly_m = boundary_gdf_m.geometry.iloc[0]

# --------------------------------------------------------
# 3.1) Labels Node_X (los sigue usando el esquema de red BFS, más abajo)
# --------------------------------------------------------
def to_node_label(x):
    s = str(x)
    return s if s.startswith("Node_") else f"Node_{s}"

nodes = nodes.copy()
nodes["node_label"] = nodes["i"].apply(to_node_label)

# --------------------------------------------------------
# 3.2) Flujo Node_X (agenda) -> node en pop_building.parquet -> osm_id/lat/lon -> malla
#      pop_building.parquet trae, por EDIFICIO (osm_id), a qué "node" de la
#      red pertenece. El kWh de kwh_map está por "node" (no por edificio),
#      así que se reparte a partes iguales entre los edificios de ese node.
# --------------------------------------------------------
pop_building_path = f"{path}/pop_building.parquet"
pop_building = pd.read_parquet(pop_building_path)
print(f"[debug] pop_building.parquet: {len(pop_building)} filas, columnas = {list(pop_building.columns)}")

# Ajusta estos nombres si en tu parquet las columnas se llaman distinto
OSM_ID_COL = "osm_id"   # identificador del edificio (solo para referencia/plot)
NODE_COL = "node"       # columna que enlaza el edificio con el Node_X de la agenda
LAT_COL = "lat"
LON_COL = "lon"

def normalize_id(x):
    """'43', 43, 43.0 -> '43'. Evita que un mismo id no cruce por venir
    como texto en un lado y como número (con '.0') en el otro."""
    if pd.isna(x):
        return None
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(x).strip()

pop_building = (
    pop_building[[OSM_ID_COL, NODE_COL, LAT_COL, LON_COL]]
    .dropna(subset=[LAT_COL, LON_COL, NODE_COL])
    .rename(columns={OSM_ID_COL: "osm_id", NODE_COL: "node", LAT_COL: "lat", LON_COL: "lon"})
    .copy()
)
pop_building["node"] = pop_building["node"].apply(normalize_id)
pop_building["osm_id"] = pop_building["osm_id"].astype(str)

# node -> kWh (kwh_map, sección D: "node" de la agenda, formato "Node_X")
kwh_by_node = pd.DataFrame({
    "node": [normalize_id(k) for k in kwh_map.keys()],
    "kWh_total": list(kwh_map.values()),
})
print(f"[debug] nodes con consumo (kwh_map): {len(kwh_by_node)}")
print(f"[debug] ejemplo node agenda: {kwh_by_node['node'].head(3).tolist()}")
print(f"[debug] ejemplo node pop_building: {pop_building['node'].head(3).tolist()}")

# node -> todos los edificios (osm_id/lat/lon) que cuelgan de ese node
poi_df = kwh_by_node.merge(pop_building, on="node", how="inner")

nodes_sin_match = kwh_by_node.loc[~kwh_by_node["node"].isin(pop_building["node"]), "node"].tolist()
print(f"[debug] edificios encontrados: {len(poi_df)} (para {poi_df['node'].nunique()} / {len(kwh_by_node)} nodes)")
if nodes_sin_match:
    print(f"Aviso: estos nodes de la agenda no aparecen en pop_building.parquet: {nodes_sin_match}")

# Repartir el kWh de cada node a partes iguales entre sus edificios,
# para que la suma total siga siendo el consumo real (no se multiplica)
edificios_por_node = poi_df.groupby("node")["node"].transform("count")
poi_df["kWh_total"] = poi_df["kWh_total"] / edificios_por_node

poi_gdf = gpd.GeoDataFrame(
    poi_df,
    geometry=gpd.points_from_xy(poi_df["lon"], poi_df["lat"]),
    crs=CRS_WGS84,
).to_crs(CRS_METRIC)

# --------------------------------------------------------
# 3.3) Construir la malla regular de 100x100 m que cubre el boundary
# --------------------------------------------------------
minx, miny, maxx, maxy = boundary_poly_m.bounds
minx = np.floor(minx / CELL_SIZE) * CELL_SIZE
miny = np.floor(miny / CELL_SIZE) * CELL_SIZE
maxx = np.ceil(maxx / CELL_SIZE) * CELL_SIZE
maxy = np.ceil(maxy / CELL_SIZE) * CELL_SIZE

x_edges = np.arange(minx, maxx + CELL_SIZE, CELL_SIZE)
y_edges = np.arange(miny, maxy + CELL_SIZE, CELL_SIZE)

cells = []
for x0 in x_edges[:-1]:
    for y0 in y_edges[:-1]:
        cell = box(x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE)
        if cell.intersects(boundary_poly_m):
            cells.append(cell)

grid_gdf = gpd.GeoDataFrame(geometry=cells, crs=CRS_METRIC)
grid_gdf["cell_id"] = grid_gdf.index

# Recortar cada celda al contorno del caso de estudio (para que el dibujo
# no se salga del boundary)
grid_gdf["geometry"] = grid_gdf.geometry.intersection(boundary_poly_m)
grid_gdf = grid_gdf[~grid_gdf.geometry.is_empty].copy()

# --------------------------------------------------------
# 3.4) Agregar kWh de los osm_id (edificios/POI) dentro de cada celda
# --------------------------------------------------------
join = gpd.sjoin(
    poi_gdf[[OSM_ID_COL, "kWh_total", "geometry"]],
    grid_gdf[["cell_id", "geometry"]],
    predicate="within",
    how="left",
)
kwh_by_cell = join.groupby("cell_id")["kWh_total"].sum()

grid_gdf["kWh_total"] = grid_gdf["cell_id"].map(kwh_by_cell).fillna(0.0)

# ------------------------------
# Valores de la muestra (reales), en base a las celdas de la malla
vals_pos = grid_gdf.loc[grid_gdf["kWh_total"] > 0, "kWh_total"].to_numpy()
print(f"[debug] celdas con kWh > 0: {len(vals_pos)} / {len(grid_gdf)}")

v_sample_min = float(vals_pos.min())  # mínimo positivo de la muestra (celdas)
v_sample_max = float(vals_pos.max())  # máximo de la muestra (celdas)

def hex_to_rgb01(hex_color):
    """Convierte #RRGGBB a tupla RGB en [0,1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

# --------------------------------------------------------
# Gradiente simple: blanco (0) -> color del caso de estudio (máximo)
# --------------------------------------------------------
COLOR_ZERO = hex_to_rgb01("#ffffff")
COLOR_MAX = hex_to_rgb01(colors.get("max", "#434343"))

def _lerp_rgb(c1, c2, t):
    t = float(np.clip(t, 0.0, 1.0))
    return (
        c1[0] + t * (c2[0] - c1[0]),
        c1[1] + t * (c2[1] - c1[1]),
        c1[2] + t * (c2[2] - c1[2]),
    )

def color_from_value(v):
    """Gradiente lineal simple: 0 -> blanco, v_sample_max -> color del caso."""
    if (v is None) or (not np.isfinite(v)) or v <= 0:
        return COLOR_ZERO
    t = v / v_sample_max
    return _lerp_rgb(COLOR_ZERO, COLOR_MAX, t)

# --------------------------------------------------------
# 4) Dibujar: solo la malla 100x100 m (sin red eléctrica ni mapa base)
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))

grid_gdf["face_color"] = grid_gdf["kWh_total"].apply(color_from_value)
grid_gdf.plot(
    ax=ax,
    color=grid_gdf["face_color"].tolist(),
    edgecolor="#cccccc",
    linewidth=0.5,
    zorder=2,
)

boundary_gdf_m.boundary.plot(ax=ax, color=colors.get("max", "grey"), linewidth=1.4, zorder=5)

# Solo los puntos (osm_id) que caen DENTRO del boundary mostrado en el mapa
poi_dentro = poi_gdf[poi_gdf.within(boundary_poly_m)]
print(f"[debug] puntos dentro del boundary: {len(poi_dentro)} / {len(poi_gdf)}")

POINTS_ALPHA = 0.4  # 0 = invisible, 1 = opaco. Baja este valor para más transparencia

ax.scatter(
    poi_dentro.geometry.x, poi_dentro.geometry.y,
    s=10, color="#1f1f1f", edgecolor="white", linewidth=0.3,
    alpha=POINTS_ALPHA,
    zorder=6, label=f"POI (osm_id), n={len(poi_dentro)}",
)
ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title(f"{study_case} — malla {int(CELL_SIZE)}x{int(CELL_SIZE)} m — kWh agregado por ubicación del osm_id")

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --------------------------------------------------------
# 5) Leyenda: gradiente simple dentro de la figura
# --------------------------------------------------------
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "grad_blanco_color", ["#ffffff", colors.get("max", "#434343")]
)

cax = inset_axes(plt.gca(), width="5%", height="30%", loc='lower left', borderpad=2.0)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=v_sample_max))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical')

ticks = [0, v_sample_max / 2, v_sample_max]
ticklabels = ['0 kWh', f'{v_sample_max/2:.1f} kWh', f'Max={v_sample_max:.1f} kWh']

cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)

from collections import deque, defaultdict

# --------------------------------------------------------
# RECONSTRUIR TOPOLOGÍA DE LA RED (árbol BFS)
# --------------------------------------------------------
root = "Node_20"

# Grafo no dirigido a partir de edges
adj = defaultdict(list)
for _, row in edges.iterrows():
    a = to_node_label(row["i"])
    b = to_node_label(row["j"])
    adj[a].append(b)
    adj[b].append(a)

# BFS para obtener padre e hijos
parent = {root: None}
children = defaultdict(list)
order = []

q = deque([root])
while q:
    u = q.popleft()
    order.append(u)
    for v in adj[u]:
        if v not in parent:
            parent[v] = u
            children[u].append(v)
            q.append(v)


# ========================================================
# EXTRA (sustituye la “segunda figura GIS”):
# Esquema tipo “perfil” de la red, de izquierda (Node_43) a derecha,
# donde la ALTURA de cada nodo es el kWh acumulado.
#
# Requisitos/Asunciones:
# - Ya has calculado: kwh_cum (dict: "Node_X" -> kWh acumulado), root="Node_43"
# - Ya existen: nodes (DataFrame con lon, lat, node_label), edges (i,j),
#               to_node_label (función), children/parent/order (del BFS que hicimos)
# ========================================================

import math
from matplotlib.ticker import MaxNLocator

# ---------- 1) Utilidades: haversine (km) usando lat/lon ----------
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------- 2) Mapa de coordenadas por nodo ----------
coord_map = dict(zip(nodes["node_label"], zip(nodes["lon"], nodes["lat"])))

# ---------- 3) Longitud de cada arista del árbol BFS (padre->hijo) ----------
edge_len = {}
for child, p in parent.items():
    if p is None:
        continue
    if (p in coord_map) and (child in coord_map):
        (lon1, lat1) = coord_map[p]
        (lon2, lat2) = coord_map[child]
        edge_len[(p, child)] = haversine_km(lon1, lat1, lon2, lat2)
    else:
        edge_len[(p, child)] = 0.0

# ---------- 4) Distancia acumulada desde la raíz (x-axis), siguiendo el árbol ----------
dist_from_root = {root: 0.0}
for u in order:
    if u not in dist_from_root:
        continue
    for ch in children.get(u, []):
        dist_from_root[ch] = dist_from_root[u] + edge_len.get((u, ch), 0.0)

reachable_nodes = list(dist_from_root.keys())

# --------------------------------------------------------
# 5) CONSTRUIR kWh ACUMULADO POR TOPOLOGÍA (subárbol)
# --------------------------------------------------------
kwh_map_complete = {lab: float(kwh_map.get(lab, 0.0)) for lab in parent.keys()}
kwh_cum = {lab: kwh_map_complete.get(lab, 0.0) for lab in parent.keys()}

for u in reversed(order):
    for ch in children.get(u, []):
        kwh_cum[u] += kwh_cum.get(ch, 0.0)

# ========================================================
# 6) ESQUEMA DE RED (perfil)
# ========================================================
plt.figure(figsize=(12, 5))

# 6.1 Aristas
for ch, p in parent.items():
    if p is None:
        continue
    if p not in dist_from_root or ch not in dist_from_root:
        continue

    x1 = dist_from_root[p]
    y1 = float(kwh_cum.get(p, 0.0))
    x2 = dist_from_root[ch]
    y2 = float(kwh_cum.get(ch, 0.0))

    plt.plot([x1, x2], [y1, y2], linewidth=1.5)

# 6.2 Nodos
xs = [dist_from_root[n] for n in reachable_nodes]
ys = [float(kwh_cum.get(n, 0.0)) for n in reachable_nodes]
plt.scatter(xs, ys, s=20, zorder=3)

# 6.3 Etiquetas sin solape (sin eliminar ninguna)
ax = plt.gca()

texts = []
for n in reachable_nodes:
    x = dist_from_root[n]
    y = float(kwh_cum.get(n, 0.0))
    n_short = n.replace("Node_", "")

    t = ax.text(
        x, y,
        n_short,
        fontsize=8,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.15"),
        zorder=4
    )
    texts.append(t)

plt.draw()
renderer = plt.gcf().canvas.get_renderer()

def _bbox_tuple(text_obj):
    bb = text_obj.get_window_extent(renderer=renderer)
    return [bb.x0, bb.y0, bb.x1, bb.y1]

def _overlap(b1, b2, pad=2.0):
    return not (b1[2] + pad < b2[0] or b2[2] + pad < b1[0] or
                b1[3] + pad < b2[1] or b2[3] + pad < b1[1])

ref_disp = []
for t in texts:
    x_data, y_data = t.get_position()
    x_disp, y_disp = ax.transData.transform((x_data, y_data))
    ref_disp.append([x_disp, y_disp])

'''max_iter = 20000
step_y = 0.5
step_x = 0.5
pad = 1.35
pull = 0.06'''

max_iter = 200
step_y = 0.5
step_x = 0.5
pad = 1.35
pull = 0.06

pos_disp = [p.copy() for p in ref_disp]

for _ in range(max_iter):
    moved = False

    for i, t in enumerate(texts):
        x_data, y_data = ax.transData.inverted().transform(pos_disp[i])
        t.set_position((x_data, y_data))

    plt.draw()
    renderer = plt.gcf().canvas.get_renderer()
    bboxes = [_bbox_tuple(t) for t in texts]

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            b1, b2 = bboxes[i], bboxes[j]
            if not _overlap(b1, b2, pad=pad):
                continue

            c1x = (b1[0] + b1[2]) * 0.5
            c1y = (b1[1] + b1[3]) * 0.5
            c2x = (b2[0] + b2[2]) * 0.5
            c2y = (b2[1] + b2[3]) * 0.5

            sign_y = 1.0 if (c1y <= c2y) else -1.0
            sign_x = 1.0 if (c1x <= c2x) else -1.0

            pos_disp[i][1] -= sign_y * step_y
            pos_disp[j][1] += sign_y * step_y

            if step_x != 0.0:
                pos_disp[i][0] -= sign_x * step_x
                pos_disp[j][0] += sign_x * step_x

            moved = True

    for i in range(len(texts)):
        pos_disp[i][0] += (ref_disp[i][0] - pos_disp[i][0]) * pull
        pos_disp[i][1] += (ref_disp[i][1] - pos_disp[i][1]) * pull

    if not moved:
        break

for i, t in enumerate(texts):
    x_data, y_data = ax.transData.inverted().transform(pos_disp[i])
    t.set_position((x_data, y_data))

plt.draw()

plt.title(f"Esquema de red (raíz {root} a la izquierda) — Altura = kWh acumulado")
plt.xlabel("Distancia acumulada desde la raíz [km]")
plt.ylabel("kWh acumulado (subárbol) [kWh]")
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.tight_layout()
plt.show()