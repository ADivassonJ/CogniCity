import os
import sys
import math
import random
import folium
import itertools
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt 
from folium.plugins import AntPath
from geopy.distance import geodesic
from collections import defaultdict
from datetime import datetime, timedelta



def load_filter_sort_reset(filepath):
    """
    Summary: 
       Load an Excel file, filter the rows where 'stat' is 'inactive' and return the DataFrame.
    Args:
       filepath (Path): path to the file that wants to be readed
    Returns:
       df: Readed dfs
    """
    df = pd.read_excel(filepath)
    df = df[df['state'] != 'inactive']
    return df




def main_td():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    
    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    file_management = system_management[['file_1', 'file_2', 'pre']]

    # Paso 2: Bucle sobre filas del mini DF
    for index, row in file_management.iterrows():
        
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        
        paths[file_2] = file_1 / file_2
        
        if not paths[file_2].exists():
            if row['pre'] == 'y':
                print(f"[Error] Critical file not detected:")
                print(f"{paths[file_2]}")
                print(f"Please solve the mentioned issue and reestart the model.")
                sys.exit()
            else:
                os.makedirs(paths[file_2], exist_ok=True)
                
    networks = ['drive', 'walk']
    networks_map = {}   
    for net_type in networks:           
        networks_map[net_type + "_map"] = ox.load_graphml(paths['maps'] / (net_type + '.graphml'))
    
    level_1_results = pd.read_excel(f"{paths['results']}/{study_area}_level_1.xlsx")
    level_2_results = pd.read_excel(f"{paths['results']}/{study_area}_level_2.xlsx")
    
    pop_citizen = pd.read_excel(f"{paths['population']}/pop_citizen.xlsx")
    
    pop_building = pd.read_excel(f"{paths['population']}/pop_building.xlsx")
    pop_transport = pd.read_excel(f"{paths['population']}/pop_transport.xlsx")
    
    pop_archetypes_transport = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    # Agrupamos los resultados por familias
    level2_families = level_2_results.groupby(['family'])
    # Pasamos por los datos de cada una de ellas
    for f_name, family in level2_families:
        # Inicializamos distime_matrix de la familia
        distime_matrix = pd.DataFrame()
        # Logramos los vehiculos asignados a cada miembro familiar
        avail_vehicles = pop_transport[pop_transport['family'] == f_name]
        # Agrupamos las actividades familiares por sus miembros
        level2_citizens = family.groupby(['agent'])     
        # Pasamos por todos los agentes de la familia
        for c_name, c_route in level2_citizens:
            # Sacamos los datos especificos de este familiar
            citizen = pop_citizen[pop_citizen['name'] == c_name]
            # Creamos una lista en la que la route del agente sea más facil de usar
            clean_route = [(c_route['osm_id'].unique()[i], c_route['osm_id'].unique()[i+1]) for i in range(len(c_route['osm_id'].unique()) - 1)]
            # Calculamos la distime_matrix del agente
            agent_distime_matrix = score_calculation(networks_map, avail_vehicles, clean_route, pop_archetypes_transport, citizen, pop_building)
            # Lo sumamos al total
            distime_matrix = pd.concat([distime_matrix, agent_distime_matrix], ignore_index=True)
        
        input(distime_matrix)


def score_calculation(networks_map, avail_vehicles, clean_route, pop_archetypes_transport, citizen, pop_building):
    # Inicializamos el df de output
    distime_matrix = pd.DataFrame()
    # Evaluamos cada opcion de vehiculo
    for _, vehicle in avail_vehicles.iterrows():
        # Evaluamos este vehiculo en toda la rout (para ello vamos trip por trip)
        for poi_A,poi_B in clean_route:
            # Sacamos los datos del archetype con el que estamos trabajando
            vehicle_archdata = pop_archetypes_transport[pop_archetypes_transport['name']==vehicle['archetype']].iloc[0]
            new_row = distime_matrix_calculation(networks_map, pop_building, vehicle_archdata, poi_A, poi_B)
            
            
            
            input(f"poi_A: {poi_A}, poi_B: {poi_B}")
        
            # Creamos los nuevos datos
            new_row = {
                'vehicle': vehicle['name'],
                'walk_time': 0,
                'travel_time': 0,
                'waiting_time': 0,
                'costs': 0,
                'beneficts': 0,
                'emissions': 0,
            }
            # La añadimos a la matriz de resultados
            distime_matrix = pd.concat([distime_matrix, pd.DataFrame([new_row])], ignore_index=True)
        
        
def distime_matrix_calculation(networks_map, pop_building, vehicle_archdata, poi_A, poi_B):
    # Inicializamos el trip con el punto de inicio
    trip = [poi_A]
    # Miramos si el vehicle, por como es su archetype, tiene P1
    if vehicle_archdata['P1'] != 0:
        # Si lo tiene, añade la ubicación valida más cercana
        trip.append(find_closest(pop_building, poi_A, vehicle_archdata['P1s']))
    # Miramos si el vehicle, por como es su archetype, tiene P2
    if vehicle_archdata['P2'] != 0:
        # Si lo tiene, añade la ubicación valida más cercana
        trip.append(find_closest_p2(networks_map, pop_building, poi_A, vehicle_archdata['P2s']))
    # Sumamos la última ubicación del trip
    trip.append(poi_B)    
    # Evaluamos si el trip no requiere paradas
    if len(trip) == 2:
        1 == 1
        
        
        

        
def find_closest_p2(networks_map, pop_building, poi, P_servicetype):
    # El agente, busca el P1 en base a la ubicacion en la que este el vehiculo actualmente
    # Solo necesita calcular el P2, en verdad, que quedará guardado despues.
    
    
    # Sacamos las coordenadas del poi a analizar
    lat_poi, lon_poi = pop_building[pop_building['osm_id'] == poi][['lat', 'lon']].iloc[0]
    feasible_P = pop_building[pop_building['archetype'] == P_servicetype].copy()
    # En caso de no encontrar servicios
    if feasible_P.empty:
        input("Cosorro")
    
    
    
    
    # Elimina los print/input si aún están en tu función
    feasible_P['distance'] = feasible_P.apply(
        lambda row: distancia_haversine(lat_poi, lon_poi, row['lat'], row['lon']), axis=1)
    # Ordenamos feasible_P por aquellos con menor distancia
    input(feasible_P)
    
    # Encontrar el nodo más cercano al POI
    poi_node = ox.distance.nearest_nodes(networks_map['walk_map'], lon_poi, lat_poi)



    
    data = []



    # Iterar sobre cada fila en df2 para encontrar el edificio más cercano
    for row in feasible_P.iterrows():
        lat2, lon2 = row.lat, row.lon
        
        # Calcular distancia euclidiana
        euc_dist = distancia_haversine(lat_poi, lon_poi, lat2, lon2)
        if euc_dist > max_distance:  # Filtrar por distancia máxima
            continue

        try:
            # Encuentra el nodo más cercano en G
            nodo_df2 = ox.distance.nearest_nodes(G, lon2, lat2)

            # Encuentra la ruta más corta entre nodo_df1 y nodo_df2
            route = ox.shortest_path(G, nodo_df1, nodo_df2, weight='length')
            distance = nx.path_weight(G, route, weight="length")
            if distance > max_distance:
                continue
            # Inicializar rise y fall
            rise, fall = 0, 0

            # Calcular desniveles a lo largo de la ruta
            for n, node in enumerate(route):
                actual_elevation = G.nodes[node].get('elevation', 0)
                if n > 0:  # Comparar con el nodo anterior
                    last_elevation = G.nodes[route[n - 1]].get('elevation', 0)
                    rise += max(actual_elevation - last_elevation, 0)
                    fall += max(last_elevation - actual_elevation, 0)

            # Obtener tipo de edificio
            building_type = getattr(row, 'building_type_name', "N/A")

            # Agregar los datos al listado
            data.append({
                'building_type': building_type,
                'osmid': getattr(row, 'osm_id', None),
                'coord': (lat2, lon2),
                'distance': distance,
                'rise': rise,
                'fall': fall,
            })
        except Exception as e:
            print(f"Error al procesar ({lat2}, {lon2}): {e}")
            continue  # Si ocurre un error, continuar con la siguiente iteración

    # Verificar si se recopilaron datos
    if not data:
        return
    
    # Convertir la lista de datos a un DataFrame antes de exportar
    data_df = pd.DataFrame(data)
    
    # Ordenar por distancia
    if 'distance' in data_df.columns:
        data_df = data_df.sort_values(by='distance', ascending=True)
    
    return data_df['osm_id'].iloc[0]

def distancia_haversine(lat1, lon1, lat2, lon2):
    # Convertir las coordenadas de grados a radianes
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Diferencias de latitud y longitud
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Fórmula de Haversine
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Radio de la Tierra en metros
    radio_tierra = 6371000  # en metros
    
    # Distancia final
    distancia = radio_tierra * c
    
    return distancia    



def add_walk_public(avail_vehicles, c_route, pop_archetypes_transport, citizen):
    """_summary_

    Args:
        avail_vehicles (df): 
        citizen (df): 
    """
    
    
    
    
    
    
    
    # Inicializamos el df de resultados
    avail_transport = pd.DataFrame()
    # Empezamos añadiendo 'walk' como opcion
    new_row = {
        'name': 'walk',
        'archetype': None,
        'family': None,
        'ubication': None,
        'v': citizen['walk_speed'],
        'Ekm': None,
        'enkm': None, 
        'Emin': None,
        'COkm': None,
        'SoC': None,
    }
    # Lo añadimos el df de resultados
    avail_transport = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    # Repetimos proceso añadiendo 'public' como opcion
    name = 'conb_public'
    new_row = {
        'name': name,
        'archetype': None,
        'family': None,
        'ubication': obtain_P1P2(name, citizen, pop_archetypes_transport, c_route),
        'v': 3,
        'Ekm': 3,
        'enkm': None, 
        'Emin': None,
        'COkm': 3,
        'SoC': None,
    }
    # Lo añadimos el df de resultados
    avail_transport = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    
    return avail_transport
    

def find_closest(pop_building, poi_A, vehicle_archdata):
    
    return 1

# Ejecución
if __name__ == '__main__':
    main_td()
    
    
    
    