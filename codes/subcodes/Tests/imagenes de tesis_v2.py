import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram
from matplotlib.patches import Patch
from collections import deque, defaultdict
import math
from matplotlib.ticker import MaxNLocator

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path(r"C:/Users/asier.divasson/Documents/GitHub/CogniCity")
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

STUDY_CASES = ["Aradas", "Annelinn", "Kanaleneiland"]
S_FOLDERS = [f"s{i}" for i in range(5)]
ARCHETYPE = "PC_electric"

# -----------------------------
# Funciones auxiliares
# -----------------------------
def to_node_label(x):
    s = str(x)
    return s if s.startswith("Node_") else f"Node_{s}"

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2],16)/255 for i in (0,2,4))

def _lerp_rgb(c1, c2, t):
    t = float(np.clip(t,0.0,1.0))
    return tuple(c1[i] + t*(c2[i]-c1[i]) for i in range(3))

def color_from_value(v, v_sample_min, vmin, vmax, v_sample_max, vals_pos):
    COLOR_ZERO = hex_to_rgb01("#ffffff")
    COLOR_MIN  = hex_to_rgb01("#ffffff")
    COLOR_20   = hex_to_rgb01("#a2c4c9")
    COLOR_80   = hex_to_rgb01("#0c343d")
    COLOR_MAX  = hex_to_rgb01("#000000")
    if (v is None) or (not np.isfinite(v)) or (v <= 0):
        return COLOR_ZERO
    a,b,c,d = v_sample_min, vmin, vmax, v_sample_max
    if d <= a:
        return COLOR_20
    if v <= b:
        t = (v-a)/(b-a) if (b-a)>0 else 0.0
        return _lerp_rgb(COLOR_MIN, COLOR_20, t)
    elif v <= c:
        t = (v-b)/(c-b) if (c-b)>0 else 0.0
        return _lerp_rgb(COLOR_20, COLOR_80, t)
    else:
        t = (v-c)/(d-c) if (d-c)>0 else 0.0
        return _lerp_rgb(COLOR_80, COLOR_MAX, t)

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
for study_case in STUDY_CASES:
    for scenario in S_FOLDERS:
        print(f"Procesando {study_case} | {scenario}...")

        # 1) Leer Excel
        results_dir = RESULTS_ROOT / scenario / study_case
        excel_path = results_dir / f"{study_case}_schedule_vehicle_quantified_24.xlsx"
        if not excel_path.exists():
            print(f"  ⚠ Excel no encontrado: {excel_path}")
            continue

        df = pd.read_excel(excel_path)
        df = df[df["archetype"]==ARCHETYPE].copy()

        # Último time_slot por agent+day
        ts = pd.to_datetime(df["time_slot"], errors="coerce")
        df["time_slot_minutes"] = ts.dt.hour*60 + ts.dt.minute
        df_last = df.sort_values(["agent","day","time_slot_minutes"], ascending=[True,True,False])\
                    .groupby(["agent","day"], as_index=False).first().drop(columns=["time_slot_minutes"])

        # MJ a float
        df_last["mjkm"] = pd.to_numeric(df_last["mjkm"].astype(str).str.replace(",",".",regex=False), errors="coerce").fillna(0.0)

        # Suma por node
        df_mj_by_node = df_last.groupby("node", as_index=False)["mjkm"].sum().rename(columns={"mjkm":"MJ_total"})
        df_mj_by_node["kWh_total"] = df_mj_by_node["MJ_total"]/3.6
        kwh_map = dict(zip(df_mj_by_node["node"], df_mj_by_node["kWh_total"]))

        # 2) Leer nodos y edges
        path = DATA_ROOT / scenario / study_case / "population"
        nodes_csv = path / f"node_data_{study_case}.csv"
        edges_csv = path / f"{study_case}_line_data.csv"
        if not nodes_csv.exists() or not edges_csv.exists():
            print(f"  ⚠ CSVs de nodos o edges no encontrados.")
            continue

        nodes = pd.read_csv(nodes_csv)
        edges = pd.read_csv(edges_csv)
        nodes["node_label"] = nodes["i"].apply(to_node_label)
        coords = nodes[["lon","lat"]].to_numpy()
        tree = cKDTree(coords)

        # 3) Polígono de boundary (ejemplo Aradas, ajustar según estudio)
        if study_case=="Aradas":
            boundary_latlon = [(40.6260277,-8.6691095),(40.6242125,-8.666836), ...]  # mantener completo
        # elif study_case=="Annelinn": ...
        # elif study_case=="Kanaleneiland": ...

        boundary_lonlat = [(lon,lat) for lat,lon in boundary_latlon]
        boundary_poly = Polygon(boundary_lonlat)

        # 4) Voronoi
        points_geom = [Point(row["lon"], row["lat"]) for _, row in nodes.iterrows()]
        multi_points = MultiPoint(points_geom)
        voronoi_multi = voronoi_diagram(multi_points, envelope=boundary_poly, edges=False)

        # 5) Escala de colores
        vals_pos = df_mj_by_node.loc[df_mj_by_node["kWh_total"]>0,"kWh_total"].to_numpy()
        if vals_pos.size:
            v_sample_min = float(vals_pos.min())
            v_sample_max = float(vals_pos.max())
            vmin = float(np.percentile(vals_pos,25))
            vmax = float(np.percentile(vals_pos,75))
        else:
            v_sample_min = v_sample_max = vmin = vmax = 0.0

        # 6) Dibujar Voronoi + red
        plt.figure(figsize=(8,8))
        for poly in voronoi_multi.geoms:
            clipped = poly.intersection(boundary_poly)
            if clipped.is_empty: continue
            c = clipped.centroid
            _, idx = tree.query([c.x, c.y], k=1)
            node_label = nodes.iloc[idx]["node_label"]
            v = kwh_map.get(node_label,0.0)
            face = color_from_value(v,v_sample_min,vmin,vmax,v_sample_max,vals_pos)
            x,y = clipped.exterior.xy
            if v<=0:
                plt.fill(x,y,facecolor=hex_to_rgb01("#ffffff"),edgecolor="#0c343d",hatch="/////",linewidth=0.0,zorder=1)
            else:
                plt.fill(x,y,facecolor=face,edgecolor="none",zorder=1)
            plt.plot(x,y,linestyle="-",color="#cccccc",linewidth=1.2,dashes=(4,7),zorder=2)

        bx,by = boundary_poly.exterior.xy
        plt.plot(bx,by,color="#0c343d",linewidth=1.2,zorder=5)

        node_dict = {row["i"]:(row["lon"],row["lat"]) for _, row in nodes.iterrows()}
        for _, row in edges.iterrows():
            n1,n2 = row["i"], row["j"]
            if n1 in node_dict and n2 in node_dict:
                x1,y1 = node_dict[n1]
                x2,y2 = node_dict[n2]
                plt.plot([x1,x2],[y1,y2],color="#000000",linewidth=1.0,zorder=6)

        plt.scatter(nodes["lon"],nodes["lat"],s=25,color="#000000",zorder=7)
        for node_label,(x,y) in zip(nodes["node_label"], zip(nodes["lon"], nodes["lat"])):
            plt.text(x,y,node_label,ha="center",va="center",fontsize=8,color="black",
                     zorder=10,bbox=dict(facecolor="white",edgecolor="none",alpha=0.7,boxstyle="round,pad=0.2"))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
        plt.tight_layout()
        plt.show()