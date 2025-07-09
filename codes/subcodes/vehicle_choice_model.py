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
    pop_family = pd.read_excel(f"{paths['population']}/pop_family.xlsx")
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
    # Evaluamos cada opcion de vehiculo
    for _, vehicle in avail_vehicles.iterrows():
        # Inicializamos el df de output
        distime_matrix_vehicle = pd.DataFrame()
        # Sacamos los datos del archetype con el que estamos trabajando
        vehicle_archdata = pop_archetypes_transport[pop_archetypes_transport['name']==vehicle['archetype']].iloc[0]
        # Evaluamos este vehiculo en toda la rout (para ello vamos trip por trip)
        for poi_A, poi_B in clean_route:
            
            distime_matrix_vehicle = distime_matrix_calculation(distime_matrix_vehicle, avail_vehicles, networks_map, pop_building, vehicle_archdata, poi_A, poi_B, vehicle)
 
def find_last(distime_matrix_vehicle, avail_vehicles, vehicle_name):
    # Si no existe ningun valor previo, estará en el hogar
    if distime_matrix_vehicle.empty:
        print(f"distime_matrix_vehicle is empty")
        osm_id = avail_vehicles[avail_vehicles['name'] == vehicle_name]['ubication'].iloc[0]
    else:
        # Si existe algún valor previo, lo busco y añado el último osmid como ubicación de parada
        osm_id = distime_matrix_vehicle[distime_matrix_vehicle['name'] == vehicle_name]['osm_id'].iloc[-1]
    return osm_id
                                
def distime_matrix_calculation(distime_matrix_vehicle, avail_vehicles, networks_map, pop_building, vehicle_archdata, poi_A, poi_B, vehicle):
    # Inicializamos el trip con el punto de inicio
    trip = [poi_A]
    # Miramos si el vehicle, por como es su archetype, tiene P1
    if vehicle_archdata['P1'] != 0:
        trip.append(find_last(distime_matrix_vehicle, avail_vehicles, vehicle['name']))
    # Miramos si el vehicle, por como es su archetype, tiene P2
    if vehicle_archdata['P2'] != 0:
        # Si lo tiene, añade la ubicación valida más cercana
        osm_id_new, km_new = find_closest_p2(networks_map, pop_building, poi_A, vehicle_archdata['P2s'])
        trip.append(osm_id_new)
    # Sumamos la última ubicación del trip
    trip.append(poi_B)    
    # Evaluamos si el trip no requiere paradas
    if len(trip) != 2:
        P_existance = True
        for step in range(len(trip)):
            if step == 0 or step == 2:
                P_P = False
            elif step == 1:
                P_P = True
            else:
                continue
            
            
            # Sacamos las coordenadas 
            lat1, lon1 = pop_building[pop_building['osm_id'] == trip[step]][['lat', 'lon']].values[0]
            lat2, lon2 = pop_building[pop_building['osm_id'] == trip[step+1]][['lat', 'lon']].values[0]
            # Calculamos las distacias  
            km = distancia_network(networks_map, lat1, lon1, lat2, lon2, P_existance, P_P = False)
            # Creamos los nuevos datos
            new_row = {
                'vehicle': vehicle['name'],
                'osm_id': trip[step],
                'walk_time': 0 if P_P else km*1, # en vez de 1 meter la velocidad MINIMA del agente involucrado
                'travel_time': 0 if not P_P else km*vehicle['speed'],
                'waiting_time': 0,
                'costs': 0 if not P_P else km*vehicle['Ekm'],
                'beneficts': 0 if not P_P else km*vehicle['Emin'],
                'emissions': 0 if not P_P else km*vehicle['COkm'],
                'SoC': 0 if not P_P else km*vehicle['SoC'],
            }

            input(new_row)
        
        

        
def find_closest_p2(networks_map, pop_building, poi, P_servicetype):
    # El agente, busca el P1 en base a la ubicacion en la que este el vehiculo actualmente
    # Solo necesita calcular el P2, en verdad, que quedará guardado despues.
    
    
    # Sacamos las coordenadas del poi a analizar
    lat_poi, lon_poi = pop_building[pop_building['osm_id'] == poi][['lat', 'lon']].iloc[0]
    feasible_P = pop_building[pop_building['archetype'] == P_servicetype].copy()
    # En caso de no encontrar servicios
    if feasible_P.empty:
        input(f"El caso de estudio analizado no cuenta con '{P_servicetype}', se tendrá que realizar algúna aproximación ...")
    # Añade una columna 'distance' donde se muestra el valor (orientativo) de haversine
    feasible_P['distance'] = feasible_P.apply(
        lambda row: distancia_haversine(lat_poi, lon_poi, row['lat'], row['lon']), axis=1)
    # Ordenamos feasible_P por aquellos con menor distancia
    feasible_P = feasible_P.sort_values(by='distance', ascending=True)
    # Encontrar el nodo más cercano al POI
    poi_node = ox.distance.nearest_nodes(networks_map['walk_map'], lon_poi, lat_poi)
    # Crear la columna 'km' si no existe
    if 'km' not in feasible_P.columns:
        feasible_P['km'] = None
    # Iterar sobre cada fila en df2 para encontrar el edificio más cercano
    for index, row in feasible_P.head().iterrows():
        # Sacamos las coordenadas del nuevo punto a ha analizar
        lat2, lon2 = row['lat'], row['lon']
        try:
            # Encuentra el nodo más cercano en el mapa
            P_node = ox.distance.nearest_nodes(networks_map['drive_map'], lon2, lat2)
            # Encuentra la ruta más corta entre poi_node y P_node
            route = ox.shortest_path(networks_map['walk_map'], poi_node, P_node, weight='length')
            distance = nx.path_weight(networks_map['walk_map'], route, weight="length")
            # Guardar el valor de distancia en la fila correspondiente
            feasible_P.at[index, 'km'] = distance/1000
        except Exception as e:
            print(f"Error al procesar ({lat2}, {lon2}): {e}")
            continue
    # Ordenamos feasible_P por aquellos con menor distancia
    feasible_P = feasible_P.sort_values(by='km', ascending=True)
    # Devolvemos los datos del 'osm_id' y distancia en 'km' del mejor caso (menor valor en 'km')
    return feasible_P['osm_id'].iloc[0], feasible_P['km'].iloc[0]

def distancia_network(networks_map, lat1, lon1, lat2, lon2, P_existance=False, P_P = False):
    # Encontrar el nodo más cercano
    node_1 = ox.distance.nearest_nodes(networks_map['walk_map'], lon1, lat1)
    # Si tenemos P_existance significa que tenemos algún tipo de parada en el trip
    if P_existance:
        network_map = 'drive_map'
    else:
        network_map = 'walk_map'
    # Encontramos el segundo nodo (dependiente de que que red queramos)
    node_2 = ox.distance.nearest_nodes(networks_map[network_map], lon2, lat2) 
    # Si tenemos P_P significa que la interacción es entre las dos P (P1P2)
    if P_P:
        network_map_2 = 'drive_map'
    else:
        network_map_2 = 'walk_map'
    # Encuentra la ruta más corta entre poi_node y P_node
    route = ox.shortest_path(networks_map[network_map_2], node_1, node_2, weight='length')
    distance = nx.path_weight(networks_map[network_map_2], route, weight="length")/1000
    # Devuelve los datos de distancia en km
    return distance

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
    
    
    
    