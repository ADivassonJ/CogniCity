import os
import sys
import random
import itertools
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from collections import defaultdict

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

# Función de distancia haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

# Función que crea la matriz de distancias de cada familia
def StoW_matrix_creation(family_df, SG_relationship_unique, todolistaction):
    # Separar los dos grupos
    list_type_0 = family_df[family_df[f'{todolistaction}_type'] == 0]
    list_type_not_0 = family_df[family_df[f'{todolistaction}_type'] != 0]

    # Hacer merge para traer lat y lon
    list_type_0 = list_type_0.merge(SG_relationship_unique[['osm_id', 'lat', 'lon']], left_on=todolistaction, right_on='osm_id', how='left')
    list_type_not_0 = list_type_not_0.merge(SG_relationship_unique[['osm_id', 'lat', 'lon']], left_on=todolistaction, right_on='osm_id', how='left')

    # Si alguno está vacío, devolvemos None
    if list_type_0.empty:
        return None
    if list_type_not_0.empty:
#        print(f'Familia {family_df["family"].iloc[0]} no tiene responsables!!!!')
        return None

    # Crear combinaciones y calcular distancias
    rows = []
    for idx_0, row_0 in list_type_0.iterrows():
        for idx_n0, row_n0 in list_type_not_0.iterrows():
            if pd.notnull(row_0['lat']) and pd.notnull(row_0['lon']) and pd.notnull(row_n0['lat']) and pd.notnull(row_n0['lon']):
                distance = haversine(row_0['lat'], row_0['lon'], row_n0['lat'], row_n0['lon'])
                rows.append({
                    'family': family_df["family"].iloc[0],
                    'id_type_0': row_0['name'],
                    'id_type_not_0': row_n0['name'],
                    'distance_km': distance
                })

    # Crear DataFrame de resultados
    StoW_matrix = pd.DataFrame(rows)
    return StoW_matrix

# Función que asigna el responsable más cercano a cada dependiente en una familia
def assign_responsable(family_df, SG_relationship_unique, todolist_family):
    
    
    StoW_matrix = StoW_matrix_creation(family_df, SG_relationship_unique, todolist_family)

    # Si no hay datos (familia vacía), saltamos
    if StoW_matrix is None or StoW_matrix.empty:
        return None

    # Para cada id_type_0, encontrar el id_type_not_0 más cercano
    idx_min = StoW_matrix.groupby('id_type_0')['distance_km'].idxmin()
    df_min_distances = StoW_matrix.loc[idx_min].reset_index(drop=True)

    return df_min_distances

def choice_modeling(df_priv_vehicles, df_citizens, route, SG_relationship, transport_archetypes, networks_map, citizen, df_family_result):
    acutime_matrix = acutime_matrix_creation(df_priv_vehicles, df_citizens, route, SG_relationship, transport_archetypes, networks_map, citizen, df_family_result)
    #willinness_matrix = willinness_calculation(acutime_matrix)

    #seleccionar de forma estadistica cual elegir, en base a willinness_matrix que sera como:
    # archetype    wilinness
    # walk         1.2
    # E_car        5.2
    # E_micro      2.3
    #        ...

def acutime_matrix_creation(df_priv_vehicles, df_citizens, route, SG_relationship, transport_archetypes, networks_map, citizen, df_family_result):
    
    acutime_matrix = pd.DataFrame(columns=['archetype', 't_walk', 't_travel', 't_wait', 'cost', 'benefict', 'CO2'])
    
    family_name = df_family_result.iloc[0]['family']
    priv_vehicle_names = df_priv_vehicles.loc[df_priv_vehicles['family'] == family_name, 'name']
    
    # como gestionamos los publicos?
    print(route)
    print(priv_vehicle_names.to_list())
    
    for priv_vehicle in priv_vehicle_names:
        
        vehicle_archetype = df_priv_vehicles.loc[df_priv_vehicles['name'] == priv_vehicle, 'archetype'].values[0]
        value = transport_archetypes.loc[transport_archetypes['name'] == vehicle_archetype, 'P_1'].values[0]      
        
        print(f"")
        print(f"### {priv_vehicle}: {vehicle_archetype}")
        
        if value == 1:
            #### sumar los intermedios a route
            route_methods = ['walk']*(len(route)-1)
            
        route_methods = ['walk', 'drive', 'walk', 'drive','drive','drive']   # si tienes tres POIs, tendrias dos methods,, si pillas coche, seria walk entre poi home y poi P_1, 
                                                    # drive entre P_1 y P_2, walk entre P_2 y WoS
        
        distances = defaultdict(float)
        
        # Analisis para cada vehiculo privado
        for idx in range(len(route)):
            if idx+1 == len(route):
                break
            # Coordenadas de origen y destino (lat, lon)
            lat1, lon1 = SG_relationship.loc[SG_relationship['osm_id'] == route[idx], ['lat', 'lon']].values[0]
            lat2, lon2 = SG_relationship.loc[SG_relationship['osm_id'] == route[idx+1], ['lat', 'lon']].values[0]

            graph = networks_map[f"{route_methods[idx]}_map"]
                
            # Encontrar los nodos más cercanos en el grafo
            orig_node = ox.distance.nearest_nodes(graph, X=lon1, Y=lat1)
            dest_node = ox.distance.nearest_nodes(graph, X=lon2, Y=lat2)

            # Calcular la ruta más corta en distancia
            shot_route = nx.shortest_path(graph, orig_node, dest_node, weight='length') # en metros??????????????????????????????????????

            # Calcular la longitud total de la ruta (en metros)
            route_length = nx.path_weight(graph, shot_route, weight='length')
            
            distances[route_methods[idx]] += route_length
        
        walk_time = (distances['walk']/dependant_min_speed(df_citizens, df_family_result, citizen))/60
        transport_time = (distances['drive']/df_priv_vehicles.loc[df_priv_vehicles['name'] == priv_vehicle, 'v'].values[0])/60
        
        print(f'walk_time: {walk_time} mins')
        print(f'transport_time: {transport_time} mins')
        
        wait_time = 0
        beneficts = 0
        
        energy_consumed = (distances['drive']*df_priv_vehicles.loc[df_priv_vehicles['name'] == priv_vehicle, 'enkm'].values[0])/1000
        costs = (distances['drive']*df_priv_vehicles.loc[df_priv_vehicles['name'] == priv_vehicle, 'Ekm'].values[0])/1000
        CO2_emission = (distances['drive']*df_priv_vehicles.loc[df_priv_vehicles['name'] == priv_vehicle, 'COkm'].values[0])/1000
        
        print(f'energy_consumed: {energy_consumed} kw')
        print(f'costs: {costs} €')
        print(f'CO2_emission: {CO2_emission} ton')
        
        # costes y blablabla
        
    input()    
           
def dependant_min_speed(df_citizens, df_family_result, citizen):
    # df_family_results:
    #      family    id_type_0     id_type_not_0   distance_km
    #   0  family_0  citizen_2     citizen_1       0.784348
    #   1  family_0  citizen_3     citizen_1       0.125682
    # citizen: citizen_1
    
    dependants = df_family_result.loc[df_family_result['id_type_not_0'] == citizen, 'id_type_0'].values  
    
    walk_speeds = []
    
    for dep in dependants:
      to_add = df_citizens.loc[df_citizens['name'] == dep, 'walk_speed'].values
      walk_speeds.append(to_add)
    
    return min(walk_speeds)[0]          
           
# Función principal
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
    
    
    df_citizens = pd.read_excel(f"{paths['population']}/pop_citizen.xlsx")
    df_priv_vehicles = pd.read_excel(f"{paths['population']}/pop_transport.xlsx")

    citizen_archetypes = load_filter_sort_reset(paths['archetypes'] / 'pop_archetypes_citizen.xlsx')
    family_archetypes = load_filter_sort_reset(paths['archetypes'] / 'pop_archetypes_family.xlsx')
    transport_archetypes = load_filter_sort_reset(paths['archetypes'] / 'pop_archetypes_transport.xlsx')
    
    networks = ['drive', 'walk']
    networks_map = {}   
    for net_type in networks:           
        networks_map[net_type + "_map"] = ox.load_graphml(paths['maps'] / (net_type + '.graphml'))
    
    SG_relationship = pd.read_excel(f"{paths['maps']}/SG_relationship.xlsx")
    
    ##############################################################################
    
    
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')
    results = pd.DataFrame(columns=['agent', 'route'])
    
    # Recorrer cada familia
    for family_name in df_citizens['family'].unique():
        family_df = df_citizens[df_citizens['family'] == family_name]
        
        todolist_family = pd.DataFrame(columns=['agent', 'todo', 'osm_id', 'todo_type', 'opening', 'closing', 'fixed?', 'time2spend', 'in', 'out', 'conmu_time'])
        responsability_matrix = pd.DataFrame(columns=['helper', 'dependent', 'geo_dist', 'time_dist', 'soc_dist', 'score'])
        
        for idx_f_df, row_f_df in family_df.iterrows():
### WoS
            osm_id = row_f_df['WoS']
            fixed = row_f_df['WoS_action_type'] != 1
            
            if fixed:
                fixed_word = 'Service'
            else:
                fixed_word = 'WoS'	
                
            opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
            closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
            time2spend = int(row_f_df['WoS_time'])
            in_h = opening
            out_h = opening + time2spend
            
            rew_row ={
                'agent': row_f_df['name'],
                'todo': 'WoS', 
                'osm_id': osm_id, 
                'todo_type': row_f_df['WoS_type'], 
                'opening': opening, 
                'closing': closing, 
                'fixed?': fixed, 
                'time2spend': time2spend, 
                'in': in_h, 
                'out': out_h,
                'conmu_time': int(row_f_df['conmu_time'])
            }
            
            todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)           
            
            
# Dutties          
            for _ in range(row_f_df['Dutties_amount']):
                # aqui habrá que meter más tema estadistico aun
                dutties_ids_options = SG_relationship[SG_relationship['service_group'] == 'entertainment']['osm_id'].tolist()
                
                osm_id = random.choice(dutties_ids_options)
                fixed = False
                
                fixed_word = 'Service'	
                    
                opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
                closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
                time2spend = int(row_f_df['Dutties_time']) # aqui habrá que meter más tema estadistico aun y ver por tipo y tal
                
                #mirar donde estaba antes
                filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
                
                if not filtered.empty:
                    in_h = max(filtered['out']) + filtered['conmu_time'].iloc[0]
                else: # si resulta que hoy no trabajaba
                    in_h = opening
                
                out_h = in_h + time2spend
                
                rew_row ={
                    'agent': row_f_df['name'],
                    'todo': 'Dutties', 
                    'osm_id': osm_id, 
                    'todo_type': row_f_df['Dutties_type'], 
                    'opening': opening, 
                    'closing': closing, 
                    'fixed?': fixed, 
                    'time2spend': time2spend, 
                    'in': in_h, 
                    'out': out_h,
                    'conmu_time': int(row_f_df['conmu_time'])
                }
                
            todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)
            
### entertainment
            # aqui habrá que meter más tema estadistico aun
            entertainment_ids_options = SG_relationship[SG_relationship['service_group'] == 'entertainment']['osm_id'].tolist()
                
            osm_id = random.choice(entertainment_ids_options)
            
            fixed = False
            fixed_word = 'Service'	
                    
            opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
            closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
            
            time2spend = 0 # aqui habrá que meter más tema estadistico aun y ver por tipo y tal
                
            #mirar donde estaba antes
            filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
                
            if not filtered.empty:
                in_h = max(filtered['out']) + filtered['conmu_time'].iloc[0]
            else: # si resulta que hoy no trabajaba
                in_h = opening
                
            out_h = closing
            
            rew_row ={
                'agent': row_f_df['name'],
                'todo': 'Entertainment', 
                'osm_id': osm_id, 
                'todo_type': row_f_df['Entertainment_type'], 
                'opening': opening, 
                'closing': closing, 
                'fixed?': fixed, 
                'time2spend': time2spend, 
                'in': in_h, 
                'out': out_h,
                'conmu_time': int(row_f_df['conmu_time'])
            }
            
            todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)   
        
        # DataFrame con todo_type == 0 (independientes)
        helpers = todolist_family[todolist_family["todo_type"] == 0].add_suffix('_h')
        # DataFrame con todo_type > 0 (dependientes)
        dependents = todolist_family[todolist_family["todo_type"] > 0].add_suffix('_d')
        
        # Producto cartesiano (todas las combinaciones posibles)
        df_combinado = helpers.merge(dependents, how='cross')
        
        for idx_df_conb, row_df_conb in df_combinado.iterrows():
            lat_h, lon_h = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_h'], ['lat', 'lon']].values[0]
            lat_d, lon_d = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_d'], ['lat', 'lon']].values[0]
            
            geo_dist = haversine(lat_h, lon_h, lat_d, lon_d)
            time_dist = row_df_conb['in_d'] - row_df_conb['in_h']
            soc_dist = 1 # Aqui habrá que poner algo estadistico o algo
            score = geo_dist + abs(time_dist) + soc_dist
            
            new_row = {
                'helper': row_df_conb['agent_h'],
                'osm_id_h': row_df_conb['osm_id_h'],
                'dependent': row_df_conb['agent_d'],
                'osm_id_d': row_df_conb['osm_id_d'],
                'geo_dist': geo_dist,
                'time_dist': time_dist,
                'soc_dist': soc_dist,
                'score': score
            }

            responsability_matrix = pd.concat([responsability_matrix, pd.DataFrame([new_row])], ignore_index=True)  
        
        input(responsability_matrix)
        
        top_scores_df = responsability_matrix.loc[
            responsability_matrix.groupby(['dependent', 'osm_id_d'])['score'].idxmin()
        ].reset_index(drop=True)

        input(top_scores_df)
            
        
        
        
        
        
        
        
        
        
        
        df_family_result = assign_responsable(family_df, SG_relationship_unique, todolist_family)
        
        input(df_family_result)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        df_family_result = assign_responsable(family_df, SG_relationship_unique)
        if df_family_result is not None:
            results = pd.concat([results, df_family_result], ignore_index=True)
        else:
            continue
            
        for _, family_member in family_df.iterrows():
            id_type_not_0_list = df_family_result['id_type_not_0'].tolist() if df_family_result is not None else []
            # familiares que cuentan con alguna persona dependiente
            if family_member['name'] in id_type_not_0_list:
                related_ids = df_family_result[df_family_result['id_type_not_0'] == family_member['name']]['id_type_0'].tolist()
                # Buscamos los WoS relacionados al id_type_0
                related_wos = df_citizens[df_citizens['name'].isin(related_ids)]['WoS'].tolist()
                route = [family_member['home']] + related_wos + [family_member['WoS']]
                # aqui el agente mira que transporte mode tomar
                choice_modeling(df_priv_vehicles, df_citizens, route, SG_relationship, transport_archetypes, networks_map, family_member['name'], df_family_result)

    # Unir todos los resultados
    if not results.empty:
        final_df = results
    else:
        print('No se encontró ningún dependiente con responsable.')
        final_df = None
    
    return final_df    
 
# Ejecución
if __name__ == '__main__':
    main_td()
