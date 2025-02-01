import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import numpy as np

# Descargar el grafo de calles de una ubicación específica
place_name = "Manhattan, New York, USA"
graph = ox.graph_from_place(place_name, network_type='drive')
gdf_streets = ox.graph_to_gdfs(graph, nodes=False)

# Obtener los límites del área
minx, miny, maxx, maxy = gdf_streets.total_bounds

# Generar puntos aleatorios dentro de los límites del área
num_points = 10
x_coords = np.random.uniform(minx, maxx, num_points)
y_coords = np.random.uniform(miny, maxy, num_points)
points = [Point(x, y) for x, y in zip(x_coords, y_coords)]

# Obtener las coordenadas de los puntos
coords = [(point.x, point.y) for point in points]

# Calcular el diagrama de Voronoi
vor = Voronoi(coords)

# Crear una lista para los polígonos de Voronoi
polygons = []

# Iterar sobre cada región de Voronoi
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        polygon = Polygon([vor.vertices[i] for i in region])
        polygons.append(polygon)

# Crear un GeoDataFrame a partir de los polígonos de Voronoi
gdf_voronoi = gpd.GeoDataFrame(geometry=polygons)

# Crear una figura y un eje
fig, ax = plt.subplots(figsize=(10, 10))

# Graficar las calles
gdf_streets.plot(ax=ax, linewidth=1, edgecolor='black')

# Graficar los polígonos de Voronoi
gdf_voronoi.plot(ax=ax, color='none', edgecolor='red')

# Graficar los puntos
gpd.GeoDataFrame(geometry=points).plot(ax=ax, color='blue')

# Mostrar el gráfico
plt.show()
