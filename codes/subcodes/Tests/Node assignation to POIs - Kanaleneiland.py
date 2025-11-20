import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram

# --------------------------------------------------------
# 1. Leer datos
# --------------------------------------------------------
path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population"

nodes = pd.read_csv(f"{path}\\node_data_Kanaleneiland.csv")     # columnas: lat, lon, i
edges = pd.read_csv(f"{path}\\Kanaleneiland_data.csv")          # columnas: i, j

# --------------------------------------------------------
# 2. Definir polígono de Kanaleneiland (lat, lon -> lon, lat)
# --------------------------------------------------------
boundary_latlon = [
    (52.07904398, 5.081736117),
    (52.07624318, 5.08308264),
    (52.06046958, 5.09756737),
    (52.06021839, 5.097758556),
    (52.06008988, 5.11164107),
    (52.06328398, 5.113065093),
    (52.06860149, 5.111588679),
    (52.07642504, 5.109425399),
    (52.07861645, 5.108711591),
    (52.08034774, 5.107271173),
    (52.08592257, 5.097037869),
    (52.08498639, 5.096460351),
    (52.08309467, 5.094751129),
    (52.0803543,  5.087985518),
    (52.07904398, 5.081736117),
]

# Convertir a (lon, lat)
boundary_lonlat = [(lon, lat) for (lat, lon) in boundary_latlon]
boundary_poly = Polygon(boundary_lonlat)

# --------------------------------------------------------
# 3. Crear Voronoi (recortado al polígono)
# --------------------------------------------------------
points_geom = [Point(row["lon"], row["lat"]) for _, row in nodes.iterrows()]
multi_points = MultiPoint(points_geom)

voronoi_multi = voronoi_diagram(multi_points, envelope=boundary_poly, edges=False)

buildings = pd.read_parquet(f"{path}\\pop_building.parquet")     # columnas: lat, lon, node, ...

# 3) KDTree con nodos
node_coords = np.column_stack([nodes["lon"].values, nodes["lat"].values])
tree = cKDTree(node_coords)

# 4) Coordenadas de edificios
building_coords = np.column_stack([buildings["lon"].values, buildings["lat"].values])

# 5) Buscar nodo más cercano
dist, idx = tree.query(building_coords, k=1)

# 6) Crear columna con el nodo asignado por distancias
nearest_nodes = nodes.iloc[idx]["i"].values

# 7) Determinar si cada edificio está dentro del boundary
inside = [
    boundary_poly.contains(Point(lon, lat))
    for lon, lat in building_coords
]

# 8) Asignar nodo si está dentro, 'unknown' si está fuera
buildings["node"] = [
    nearest_nodes[i] if inside[i] else "unknown"
    for i in range(len(buildings))
]

# 9) Guardar resultado
output_path = f"{path}\\pop_building_with_nodes.parquet"
buildings.to_parquet(output_path, index=False)

# Contar ocurrencias de cada nodo
counts = buildings["node"].value_counts()

print("Archivo guardado en:", output_path)