import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram

# --------------------------------------------------------
# 1. Leer datos
# --------------------------------------------------------
path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Aradas\population"

nodes = pd.read_excel(f"{path}\\electric_system_Aradas.xlsx")     # columnas: lat, lon, i
edges = pd.read_excel(f"{path}\\electric_system_Aradas.xlsx")          # columnas: i, j

# --------------------------------------------------------
# 2. Definir polígono de Kanaleneiland (lat, lon -> lon, lat)
# --------------------------------------------------------
boundary_latlon = 

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