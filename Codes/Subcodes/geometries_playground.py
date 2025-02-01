#   V1.0.0  ->  Se toma la version V31 y se busca dividir todo el codigo en sub-funciones.

import re
import ast
import osmnx as ox
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon

def get_center(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        return centroid.y, centroid.x
    elif isinstance(geom, Point):
        return geom.y, geom.x
    return None, None

def get_df_buildings(area):
    try:
        print(f"Retrieving building data for {area}.")
        edificios = ox.features_from_place(area, tags={'building': True})
        building_ID = edificios.index.get_level_values('osmid').tolist()
        values = {'osmid': building_ID, 'coord': []}
        names = ['osmid', 'coord']
        for geom in edificios['geometry']:
            centroid = geom.centroid
            values['coord'].append((centroid.y, centroid.x))
        variables = ['building', 'amenity', 'geometry']
        for vari in variables:
            if vari in edificios.columns:
                data_array = edificios[vari].tolist()
                names.append(vari)
                values[vari] = data_array
        df_buildings = pd.DataFrame(values, columns=names)
        print("[Completed]")
        return df_buildings
    except Exception as e:
        print("Error retrieving buildings data:", e)
        return pd.DataFrame()

def plot_voronoi_on_map(vor, area):
    """Genera un gráfico interactivo del diagrama de Voronoi sobre un mapa."""
    fig = go.Figure()
    
    # Agregar bordes de las regiones de Voronoi
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            fig.add_trace(go.Scattergeo(
                lon=[p[0] for p in polygon],
                lat=[p[1] for p in polygon],
                mode='lines',
                line=dict(width=1, color='olive'),
                fill='toself'
            ))
    
    fig.update_geos(fitbounds="locations")
    fig.update_layout(title=f"Diagrama de Voronoi sobre {area}")
    fig.show()

def parse_coordinates(coord_string):
    # Usar una expresión regular para extraer los números
    if isinstance(coord_string, str):
        match = re.match(r'\((\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)\)', coord_string)
        if match:
            x, y = map(float, match.groups())  # Convertir a números flotantes
            return (x, y)
        else:
            raise ValueError("El formato de entrada no es válido. Debe ser '(x,y)' con números.")
    else:
        return coord_string

def parse_and_swap_coordinates(coord):
    if isinstance(coord, str):
        try:
            # Convertir el string en una tupla real
            coord = ast.literal_eval(coord)
        except (ValueError, SyntaxError):
            raise ValueError(f"Formato inválido para coordenadas: {coord}")
    if isinstance(coord, tuple) and len(coord) == 2:
        # Invertir latitud y longitud
        return (coord[1], coord[0])
    raise ValueError(f"Coordenadas no válidas: {coord}")
