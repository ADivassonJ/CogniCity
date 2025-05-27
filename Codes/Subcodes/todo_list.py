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
import folium
from folium.plugins import AntPath

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


def plot_agent_route_on_map(todolist_df, sg_relationship_df, agent_name, save_path="recorrido.html"):
    """
    Genera un mapa folium con el recorrido de un agente, mostrando nombre del 'todo' en orden cronológico,
    y flechas para indicar la dirección del recorrido.

    Args:
        todolist_df (pd.DataFrame): DataFrame con ['agent', 'todo', 'osm_id', 'in', 'out']
        sg_relationship_df (pd.DataFrame): DataFrame con ['osm_id', 'lat', 'lon']
        agent_name (str): Nombre del agente
        save_path (str): Ruta de archivo HTML a guardar
    """
    # Filtrar tareas del agente
    agent_tasks = todolist_df[todolist_df['agent'] == agent_name].copy()
    if agent_tasks.empty:
        raise ValueError(f"No se encontraron tareas para el agente '{agent_name}'")

    # Ordenar cronológicamente
    agent_tasks.sort_values(by='in', inplace=True)

    # Unir con coordenadas
    route = pd.merge(agent_tasks, sg_relationship_df, on='osm_id', how='left')
    if route[['lat', 'lon']].isnull().any().any():
        raise ValueError("Faltan coordenadas para algunas tareas del agente.")

    # Centrar mapa en la primera ubicación
    start_location = [route.iloc[0]['lat'], route.iloc[0]['lon']]
    route_map = folium.Map(location=start_location, zoom_start=13)

    # Agregar puntos con etiquetas
    points = []
    for _, row in route.iterrows():
        label = row['todo']
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=label,
            tooltip=label,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(route_map)
        points.append([row['lat'], row['lon']])

    # Agregar flechas de dirección
    if len(points) >= 2:
        AntPath(points, color='blue', weight=4, delay=1000).add_to(route_map)

    # Guardar el mapa
    route_map.save(save_path)
    print(f"Mapa guardado en: {save_path}")

def responsability_matrix_creation(todolist_family, SG_relationship_unique):
    responsability_matrix = pd.DataFrame(columns=['helper', 'dependent', 'geo_dist', 'time_dist', 'soc_dist', 'score'])
    # DataFrame con todo_type > 0 (dependientes con trips que requieren asistencia)
    dependents = todolist_family[todolist_family["todo_type"] > 0].add_suffix('_d')
    # DataFrame con todo_type == 0 (independientes)
    helpers = todolist_family[todolist_family["todo_type"] == 0].add_suffix('_h')
    # Eliminamos los independientes pero no capaces de ayudar (aquellos que en WoS son dependientes)
    # 1. Encontrar los agentes que cumplen con las condiciones
    agents_with_wos = todolist_family[
        (todolist_family['todo'] == 'WoS') & 
        (todolist_family['todo_type'] != 0)
    ]['agent'].unique()
    # 2. Filtrar el DataFrame helpers para eliminar las filas con esos agentes
    helpers = helpers[~helpers['agent_h'].isin(agents_with_wos)].reset_index(drop=True)
    
    # Producto cartesiano (todas las combinaciones posibles)
    df_combinado = helpers.merge(dependents, how='cross')
        
    for idx_df_conb, row_df_conb in df_combinado.iterrows():
        lat_h, lon_h = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_h'], ['lat', 'lon']].values[0]
        lat_d, lon_d = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_d'], ['lat', 'lon']].values[0]
            
        geo_dist = haversine(lat_h, lon_h, lat_d, lon_d)
        time_dist = row_df_conb['in_d'] - row_df_conb['in_h'] # si esto da negativo, el agente helper tiene más tiempo para gestionar al dependant
        soc_dist = 1 # Aqui habrá que poner algo estadistico o algo
        score = geo_dist + abs(time_dist)/100 + soc_dist
            
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
    
    if not responsability_matrix.empty:
        responsability_matrix = responsability_matrix.loc[
            responsability_matrix.groupby(['dependent', 'osm_id_d'])['score'].idxmin()
        ].reset_index(drop=True)
    else:
        print(f'family XX has no responsables.')
    
    return responsability_matrix

def todolist_family_initialization(SG_relationship, family_df):
    todolist_family = pd.DataFrame(columns=['agent', 'todo', 'osm_id', 'todo_type', 'opening', 'closing', 'fixed?', 'time2spend', 'in', 'out', 'conmu_time'])
        
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
            
            if in_h < closing:
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
            else:
                print(f"{row_f_df['name']} was not able to fullfill 'Dutties' at {in_h}.")
            
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
        
        if in_h < closing:    
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
        else:
            print(f"{row_f_df['name']} was not able to fullfill 'Entertainment' at {in_h}.")
        
        ###home in
        # in_home
        row_in_home = filtered[filtered['out'] == max(filtered['out'])]
        in_home_time = row_in_home['out'].iloc[0] + row_in_home['conmu_time'].iloc[0]
        rew_row ={
            'agent': row_f_df['name'],
            'todo': 'Home', 
            'osm_id': row_f_df['home'], 
            'todo_type': row_f_df['WoS_type'], 
            'opening': 0, 
            'closing': float('inf'), 
            'fixed?': False, 
            'time2spend': 0, 
            'in': in_home_time, 
            'out': float('inf'),
            'conmu_time': int(row_f_df['conmu_time'])
        }   
        todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True) 
        
        
    if todolist_family['agent'].nunique() == 1:
        todolist_family['todo_type'] = 0
    
    return todolist_family

def todolist_family_creation(df_citizens, SG_relationship):
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')
    results = pd.DataFrame(columns=['agent', 'route'])
    
    # Recorrer cada familia
    for family_name in df_citizens['family'].unique():
        family_df = df_citizens[df_citizens['family'] == family_name]
        
        todolist_family = todolist_family_initialization(SG_relationship, family_df)  
        if max(todolist_family['todo_type']) > 0:
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique)
            todolist_family = home_trips_adding(family_df, todolist_family) 
            
            
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique)
            
             

        input(todolist_family)
        # y luego meter antes y despues casa, con resta o suma para ver cuando salen o llegan

def home_trips_adding(family_df, todolist_family):
    
    for _, row_agent in family_df.iterrows():
        filtered = todolist_family[todolist_family['agent'] == row_agent['name']]
        row_out_home = filtered[filtered['in'] == min(filtered['in'])]
        out_home_time = row_out_home['in'].iloc[0] - row_out_home['conmu_time'].iloc[0]
        
        # out_home
        rew_row ={
            'agent': row_agent['name'],
            'todo': 'Home', 
            'osm_id': row_agent['home'], 
            'todo_type': 0, 
            'opening': 0, 
            'closing': float('inf'), 
            'fixed?': False, 
            'time2spend': 0, 
            'in': 0, 
            'out': out_home_time,
            'conmu_time': int(row_agent['conmu_time'])
        }   
        todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)    
    todolist_family = todolist_family.sort_values(by='in', ascending=True).reset_index(drop=True) 
    
    return todolist_family
    
def todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique): 

    resties2cover = resties2cover_creation(todolist_family, responsability_matrix)
    
    matrix2cover = matrix2cover_creation(todolist_family, resties2cover)
        
    new_todolist_family = new_todolist_family_creation(matrix2cover)
    
    
    
    todolist_family_adapted = todolist_family[~todolist_family.isin(matrix2cover.to_dict(orient='list')).all(axis=1)]

    up2adapt = todolist_family_adapted[todolist_family_adapted['in'] < min(matrix2cover['in'])] 
    down2adapt = todolist_family_adapted[todolist_family_adapted['in'] > min(matrix2cover['in'])] 
    # obtenemos los agentes que se han visto afectados
    agents_affected = matrix2cover['agent'].unique()
    # guardamos las rutas de aquellos que no han tenido ningun trip afectado
    schedule2add = todolist_family[~todolist_family['agent'].isin(agents_affected)]   
    
    for agent in agents_affected:
        schedule2adapt = up2adapt[up2adapt['agent'] == agent].sort_values(by='in', ascending=False)
        schedule2consider = new_todolist_family[new_todolist_family['agent'] == agent]
        fist_in_time = min(schedule2consider['in'])
        
        for _, row_s2a in schedule2adapt.iterrows():
            out_time = fist_in_time - int(row_s2a['conmu_time'])
            
            if out_time < row_s2a['opening']:
                print(f"{row_s2a['agent']} no ha podido realizar {row_s2a['todo']}")
                print(f'creo que esto no deberia poder pasar')
                continue
            
            if row_s2a['time2spend'] == 0:
                in_time = row_s2a['opening']                                                           ###### Aqui habria que mirar el paso anterior, si existe un paso anterior 
                                                                                                       ###### mira su salida y sumale el tiempo de traslado para el in de este.
                                                                                                       ###### Si este in es mayor que el out calculado, que salte error
            else:
                in_time = out_time - row_s2a['time2spend']
            #actualizamos 'fist_in_time' para que si hay otra vualta del for, mire este in y le reste el recorrido
            fist_in_time = row_s2a['in']
            
            rew_row ={
                'agent': row_s2a['agent'],
                'todo': row_s2a['todo'], 
                'osm_id': row_s2a['osm_id'], 
                'todo_type': row_s2a['todo_type'], 
                'opening': row_s2a['opening'], 
                'closing': row_s2a['closing'], 
                'fixed?': row_s2a['fixed?'], 
                'time2spend': row_s2a['time2spend'], 
                'in': in_time, 
                'out': out_time,
                'conmu_time': int(row_s2a['conmu_time'])
            }   
            new_todolist_family = pd.concat([new_todolist_family, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
            
    print('hemos llegado')
    input(new_todolist_family)
            
            
    input(schedule2adapt)
    
    print(up2adapt)
    print(down2add)
    input(new_todolist_family)
    
    #### addschedule2add que son los agentes que en este intento no se han tocado
        
def new_todolist_family_creation(matrix2cover):
    new_todolist_family = pd.DataFrame()
    
    if matrix2cover['fixed?'].any():
        first_true_index = matrix2cover.index[matrix2cover['fixed?'] == True][0]
        in_time = matrix2cover['opening'].iloc[first_true_index] + first_true_index*matrix2cover['conmu_time'].iloc[0]
    else:
       in_time = matrix2cover['opening'].iloc[0]
    
    out_time = min([in_time + matrix2cover['time2spend'].iloc[0], matrix2cover['closing'].iloc[0]])
    
    # tras acompañar, helper va a su hubicacion
    rew_row ={
        'agent': matrix2cover['agent'].iloc[0],
        'todo': matrix2cover['todo'].iloc[0], 
        'osm_id': matrix2cover['osm_id'].iloc[0], 
        'todo_type': 0, 
        'opening': matrix2cover['opening'].iloc[0], 
        'closing': matrix2cover['closing'].iloc[0], 
        'fixed?': matrix2cover['fixed?'].iloc[0], 
        'time2spend': matrix2cover['time2spend'].iloc[0], 
        'in': in_time, 
        'out': out_time,
        'conmu_time': int(matrix2cover['conmu_time'].iloc[0])
    }
    helper_row = pd.DataFrame([rew_row])
    new_todolist_family = pd.concat([new_todolist_family, helper_row], ignore_index=True) 

    grouped = matrix2cover.iloc[1:].groupby('osm_id')
    next_in_time = in_time
    temporal_schedule = pd.DataFrame()
    for _, group in grouped:
        for idx_group, row_group in group.iterrows():
            # If row_group is a Series of type object (not usual, but interpreted)
            if isinstance(row_group, object) and not isinstance(row_group, pd.Series):
                row_group = pd.DataFrame([row_group])  # Convert to DataFrame if needed
            if int(idx_group) == int(first_true_index):
                in_time = row_group['opening']  # row_group is a Series, no need for iloc
            else:
                filtered = new_todolist_family[new_todolist_family['todo'] == 'accompaniment']
                in_time = min(filtered['in']) - int(row_group['conmu_time'])
            
            out_time = min([in_time + row_group['time2spend'], row_group['closing']])
            
            rew_row ={
                'agent': row_group['agent'],
                'todo': row_group['todo'], 
                'osm_id': row_group['osm_id'], 
                'todo_type': 0, 
                'opening': row_group['opening'], 
                'closing': row_group['closing'], 
                'fixed?': row_group['fixed?'], 
                'time2spend': row_group['time2spend'], 
                'in': in_time, 
                'out': out_time,
                'conmu_time': int(row_group['conmu_time'])
            }
            temporal_schedule = pd.concat([temporal_schedule, pd.DataFrame([rew_row])], ignore_index=True) 

        # la flota les siguen
        for _, row_ntf in new_todolist_family.iterrows():
            rew_row ={
                'agent': row_ntf['agent'],
                'todo': f"accompaniment", 
                'osm_id': group['osm_id'].iloc[0], 
                'todo_type': 0, 
                'opening': group['opening'].iloc[0], 
                'closing': group['closing'].iloc[0], 
                'fixed?': False, 
                'time2spend': 0, 
                'in': in_time, 
                'out': in_time,
                'conmu_time': int(group['conmu_time'].iloc[0])
            }
            new_todolist_family = pd.concat([new_todolist_family, pd.DataFrame([rew_row])], ignore_index=True) 
        
        new_todolist_family = pd.concat([new_todolist_family, temporal_schedule], ignore_index=True).sort_values(by='in', ascending=True).drop_duplicates().reset_index(drop=True)    
    return new_todolist_family
    
def resties2cover_creation(todolist_family, responsability_matrix):
    # 1. Filtrar tareas que no sean de tipo 0 y con el menor 'in'
    min_in = todolist_family.loc[todolist_family['todo_type'] != 0, 'in'].min()
    target_task = todolist_family[(todolist_family['todo_type'] != 0) & (todolist_family['in'] == min_in)]
    # 2. Tomar los valores relevantes del POI y agente dependiente (si hay una única fila)
    poi_id = target_task.iloc[0]['osm_id']
    # Buscar en la matriz de responsabilidades por POI dependiente
    dependencies = responsability_matrix[responsability_matrix['osm_id_d'] == poi_id]
    helper = dependencies.iloc[0]['helper']
    osm_id_h = dependencies.iloc[0]['osm_id_h']
    # Filtrar por helper y osm_id_h
    resties2cover = responsability_matrix[
        (responsability_matrix['helper'] == helper) &
        (responsability_matrix['osm_id_h'] == osm_id_h)
    ].sort_values('score').reset_index(drop=True) ##### Igual mejor del reves, ordenar de peor a mejor
    
    return resties2cover
        
        
def matrix2cover_creation(todolist_family, resties2cover):
    matrix2cover = pd.DataFrame()

    agent = resties2cover['helper'].iloc[0]
    osm_id = resties2cover['osm_id_h'].iloc[0]
        
    rew_row = todolist_family[(todolist_family['agent'] == agent) & (todolist_family['osm_id'] == osm_id)]
    matrix2cover = pd.concat([matrix2cover, rew_row], ignore_index=True)

    for _, row_r2c in resties2cover.iterrows():
        rew_row = todolist_family[(todolist_family['agent'] == row_r2c['dependent']) & (todolist_family['osm_id'] == row_r2c['osm_id_d'])]
        matrix2cover = pd.concat([matrix2cover, rew_row], ignore_index=True)
    
    return matrix2cover   
    
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
    print(f'docs readed')
    
    todolist_family_creation(df_citizens, SG_relationship)
        
 
# Ejecución
if __name__ == '__main__':
    main_td()
