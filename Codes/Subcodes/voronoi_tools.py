import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MplPolygon
import geopandas as gpd
from random import random
from Subcodes.territory_geometries import Kanaleneiland

def voronoi_net(data_path, area):
    # Cargar datos desde el archivo Excel
    file_path = f"{data_path}/{area}_bus_data.xlsx"
    data = pd.read_excel(file_path)

    # Reemplaza 'lat' y 'long' con los nombres reales de las columnas
    id_column = data.columns[0]  # Suponemos que es la primera columna
    latitude_column = 'lat'
    longitude_column = 'long'

    # Extraer las coordenadas y los nombres
    names = data[id_column].values
    coordinates = data[[longitude_column, latitude_column]].dropna().values

    # Calcular el punto medio y la desviación estándar
    center = coordinates.mean(axis=0)
    std_dev = coordinates.std(axis=0)

    # Crear 9 puntos adicionales
    factor = 5  # Multiplicador de la desviación estándar
    offsets = [
        (0, 0),  # Centro
        (0, factor),  # Norte
        (0, -factor),  # Sur
        (factor, 0),  # Este
        (-factor, 0),  # Oeste
        (factor, factor),  # Noreste
        (-factor, factor),  # Noroeste
        (factor, -factor),  # Sureste
        (-factor, -factor),  # Suroeste
    ]
    extra_points = np.array([center + offset * std_dev for offset in offsets])

    # Combinar puntos originales con puntos adicionales
    all_points = np.vstack([coordinates, extra_points])

    # Crear el diagrama de Voronoi
    vor = Voronoi(all_points)

    if area == 'Kanaleneiland':
        geometry = Kanaleneiland
    else:
        print(f'Los limites de la gemoetria para el area "{area}" no han sido especificados en el archivo territory_geometries')
    
    polygon = Polygon(geometry)

    # Función para recortar las celdas de Voronoi al polígono
    def clip_voronoi_cells(vor, polygon, coordinates, names):
        regions = []
        for i, region_idx in enumerate(vor.point_region[:len(coordinates)]):  # Solo puntos originales
            vertices = vor.regions[region_idx]
            if -1 in vertices or not vertices:
                continue  # Ignorar regiones infinitas o vacías
            region_polygon = Polygon([vor.vertices[i] for i in vertices])
            clipped_polygon = region_polygon.intersection(polygon)
            if not clipped_polygon.is_empty:
                regions.append({'name': names[i], 'polygon': clipped_polygon})
        return regions

    # Recortar las regiones del Voronoi al polígono
    clipped_regions = clip_voronoi_cells(vor, polygon, coordinates, names)

    # Convertir a GeoDataFrame
    gdf = gpd.GeoDataFrame(clipped_regions, columns=['name', 'polygon'], geometry='polygon')

    # Guardar en un archivo GeoJSON para reutilizar
    gdf.to_file(f"{data_path}/{area}_voronoi.geojson", driver='GeoJSON')
    '''
    # Graficar las regiones con colores aleatorios
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, row in gdf.iterrows():
        color = (random(), random(), random())  # Color aleatorio
        patch = MplPolygon(list(row['polygon'].exterior.coords), facecolor=color, edgecolor='black', alpha=0.6)
        ax.add_patch(patch)
    
    # Graficar el polígono
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=1.5)
    ax.set_xlim(polygon.bounds[0] - 0.01, polygon.bounds[2] + 0.01)
    ax.set_ylim(polygon.bounds[1] - 0.01, polygon.bounds[3] + 0.01)
    ax.set_title("Voronoi Areas Clipped to Polygon")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.axis('equal')
    plt.show()
    '''
    return gdf

import geopandas as gpd
from shapely.geometry import Point

def find_voronoi_region(data_path, area, query_coords):
    """
    Encuentra a qué región Voronoi pertenece una coordenada.

    Parameters:
    -----------
    geojson_file : str
        Ruta al archivo GeoJSON que contiene las áreas Voronoi recortadas.
    query_coords : tuple
        Coordenadas del punto de consulta en formato (longitud, latitud).

    Returns:
    --------
    str
        Nombre del área correspondiente si el punto está en una región,
        o un mensaje indicando que no está dentro de ninguna región.
    """
    # Cargar el GeoJSON
    try:
        gdf = gpd.read_file(f"{data_path}/{area}_voronoi.geojson")
    except Exception as e:
        gdf = voronoi_net(data_path, area)
    # Crear un objeto Point a partir de las coordenadas de consulta
    query_coords_inverted = query_coords[::-1] 
    query_point = Point(query_coords_inverted)

    # Verificar en qué área cae el punto
    for _, row in gdf.iterrows():
        if row['geometry'].contains(query_point):
            return row['name']
    
    return None
