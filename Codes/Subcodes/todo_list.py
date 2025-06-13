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
import pandas as pd
import folium
from folium.plugins import AntPath
import matplotlib.pyplot as plt 

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

def plot_agents_in_split_map(todolist_df, sg_relationship_df, save_path="recorridos_agentes.html"):
    agentes = todolist_df['agent'].unique()
    num_agentes = len(agentes)

    # Generar colores únicos
    cmap = plt.cm.get_cmap('tab10', num_agentes)
    color_map = {
        agent: "#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in cmap(i)[:3]))
        for i, agent in enumerate(agentes)
    }

    # Crear directorio temporal para guardar mapas individuales
    temp_dir = "temp_maps"
    os.makedirs(temp_dir, exist_ok=True)
    map_files = []

    for agent in agentes:
        agent_tasks = todolist_df[todolist_df['agent'] == agent].copy()
        agent_tasks.sort_values(by='in', inplace=True)
        route = pd.merge(agent_tasks, sg_relationship_df, on='osm_id', how='left')

        if route[['lat', 'lon']].isnull().any().any():
            raise ValueError(f"Faltan coordenadas para el agente '{agent}'.")

        # Centro inicial del mapa
        start_latlon = [route.iloc[0]['lat'], route.iloc[0]['lon']]
        mapa = folium.Map(location=start_latlon, zoom_start=13)
        puntos = []

        for _, row in route.iterrows():
            punto = [row['lat'], row['lon']]
            puntos.append(punto)
            label = f"{row['todo']}"
            folium.Marker(
                location=punto,
                popup=label,
                tooltip=label,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(mapa)

        if len(puntos) >= 2:
            folium.PolyLine(
                puntos,
                color=color_map[agent],
                weight=6,
                opacity=0.8
            ).add_to(mapa)

        # Guardar mapa individual
        map_file = os.path.join(temp_dir, f"{agent}.html")
        mapa.save(map_file)
        map_files.append((agent, map_file))

    # Crear HTML combinado
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Recorridos por agente</title></head><body>")
        f.write("<style>iframe { width: 48%; height: 400px; display: inline-block; margin: 1%; }</style>")
        f.write("<h1 style='text-align:center;'>Recorridos de agentes</h1>")

        for agent, map_file in map_files:
            f.write(f"<h3>{agent}</h3>")
            f.write(f"<iframe src='{map_file}'></iframe>")

        f.write("</body></html>")

    print(f"Pantalla partida guardada en: {save_path}")

def responsability_matrix_creation(todolist_family, SG_relationship_unique):   
    responsability_matrix = pd.DataFrame()
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
        if row_df_conb['in_h'] == 0:
            continue
        lat_h, lon_h = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_h'], ['lat', 'lon']].values[0]
        lat_d, lon_d = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_d'], ['lat', 'lon']].values[0]
        
        geo_dist = haversine(lat_h, lon_h, lat_d, lon_d)
        time_dist = row_df_conb['in_d'] - row_df_conb['in_h'] # si esto da negativo, el agente helper tiene más tiempo para gestionar al dependant
        soc_dist = 1 # Aqui habrá que poner algo estadistico o algo
        score = geo_dist + abs(time_dist)/100 + soc_dist
        
        
        try: 
            h_schedule = todolist_family[(todolist_family['agent'] == row_df_conb['agent_h']) & (todolist_family['out'] <= row_df_conb['in_h'])]              
            print('h_schedule')
            print(h_schedule)
            print('todolist_family')
            print(todolist_family)
            h_pre_step = h_schedule[h_schedule['out'] == max(h_schedule['out'])]
        except Exception:
            input('sa petao')
        
        d_schedule = todolist_family[(todolist_family['agent'] == row_df_conb['agent_d']) & (todolist_family['out'] <= row_df_conb['in_d'])]              
        d_pre_step = d_schedule[d_schedule['out'] == max(d_schedule['out'])]
        
        new_row = {
            'helper': row_df_conb['agent_h'],
            'dependent': row_df_conb['agent_d'],
            'osm_id_h0': h_pre_step['osm_id'].iloc[0],
            'osm_id_h1': row_df_conb['osm_id_h'],
            'osm_id_d0': d_pre_step['osm_id'].iloc[0],
            'osm_id_d1': row_df_conb['osm_id_d'],
            'geo_dist': geo_dist,
            'time_dist': time_dist,
            'soc_dist': soc_dist,
            'score': score,
            'out_h': h_pre_step['out'].iloc[0],
            'in_h': row_df_conb['in_h'],
            'out_d': d_pre_step['out'].iloc[0],
            'in_d': row_df_conb['in_d']
        }

        responsability_matrix = pd.concat([responsability_matrix, pd.DataFrame([new_row])], ignore_index=True)

    if not responsability_matrix.empty:
        responsability_matrix = responsability_matrix.loc[
            responsability_matrix.groupby(['dependent', 'osm_id_d1'])['score'].idxmin()
        ].reset_index(drop=True)
        # Agrupar por 'out_h' y sumar los valores de 'score'
        grouped = responsability_matrix.groupby('out_h')['score'].sum()
        # Encontrar el grupo con menor suma
        min_group = grouped.idxmin()
        # Filtrar el DataFrame original para quedarte solo con ese grupo
        responsability_matrix = responsability_matrix[responsability_matrix['out_h'] == min_group]
    else:
        print(f'family XX has no responsables.')
    
    ## Nos deshacemos de los casos de un helper ayuda al mismo dependant para multiples tareas (evitamos futuros conflictos)
    # Agrupamos por las columnas 'helper' y 'dependent' y obtenemos el índice del menor 'score'
    idx_min_scores = responsability_matrix.groupby(['helper', 'dependent'])['score'].idxmin()
    # Seleccionamos solo esas filas
    responsability_matrix = responsability_matrix.loc[idx_min_scores].reset_index(drop=True)
    
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
        out_h = (in_h + time2spend) if time2spend != 0 else closing
            
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
                
            out_h = (in_h + time2spend) if time2spend != 0 else closing
            
            if in_h < closing and out_h <= closing:
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
                
        out_h = (in_h + time2spend) if time2spend != 0 else closing
        
        if in_h < closing and out_h <= closing:    
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
        
        filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
        
        ### home in
        # in_home
        in_home_time = max(filtered['out']) + filtered['conmu_time'].iloc[0]
        
        rew_row ={
            'agent': row_f_df['name'],
            'todo': 'Home_in', 
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
        
        todolist_family = home_out_trips_adding(family_df, todolist_family)
        
        while max(todolist_family['todo_type']) > 0:
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique)            
            
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique)
            print('todolist_family')
            print(todolist_family)
            input('#' * 80)
        # plot_agents_in_split_map(todolist_family, SG_relationship, save_path="recorridos_todos.html")

def home_out_trips_adding(family_df, todolist_family):
    
    for _, row_agent in family_df.iterrows():
        filtered = todolist_family[todolist_family['agent'] == row_agent['name']]
        row_out_home = filtered[filtered['in'] == min(filtered['in'])]
        out_home_time = row_out_home['in'].iloc[0] - row_out_home['conmu_time'].iloc[0]
        
        # out_home
        rew_row ={
            'agent': row_agent['name'],
            'todo': 'Home_out', 
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

    matrix2cover, prev_matrix2cover = matrix2cover_creation(todolist_family, responsability_matrix)
    
    new_todolist_family = route_creation(matrix2cover, prev_matrix2cover)
    
    todolist_family = new_todolist_family_adaptation(todolist_family, matrix2cover, new_todolist_family, prev_matrix2cover)
    return todolist_family

def df_division(agent_todo, agent_schedule):
    # Paso 1: Añadir el índice como columna para no perderlo al hacer merge
    agent_todo_with_index = agent_todo.reset_index()  # El índice se guarda en la columna 'index'
    # Paso 2: Merge con agent_schedule
    merged = agent_todo_with_index.merge(agent_schedule, on=['todo', 'osm_id'], how='inner')
    # Paso 3: Obtener los índices originales de agent_todo que coincidieron
    matching_indices = merged['index']  # Esta es la columna que guarda el índice original
    # Paso 4: Calcular índice mínimo y máximo
    min_index = matching_indices.min()
    max_index = matching_indices.max()
    # Paso 5: Crear los dos nuevos DataFrames a partir de agent_todo original
    df_before_min = agent_todo[agent_todo.index < min_index]
    df_after_max = agent_todo[agent_todo.index > max_index]
        
    print('df_before_min')
    print(df_before_min)
    print('agent_schedule')
    print(agent_schedule)
    print('df_after_max')
    print(df_after_max)
    
    return df_before_min, df_after_max


def new_todolist_family_adaptation(todolist_family, matrix2cover, new_todolist_family, prev_matrix2cover):
    
    # Inicializamos el df de suma de cosas
    new_new_list = pd.DataFrame()
    # Pasamos por todos los agentes del schedule
    for agent in todolist_family['agent'].unique():
        # Si el agente no ha sido modificado
        if not agent in new_todolist_family['agent'].to_list():
            # Obtenemos los datos del agente
            agent_todo = todolist_family[todolist_family['agent'] == agent]
            ################################################################################### Mirar la matrix de dependencias
            
            # Se mantienen lo previo
            new_new_list = pd.concat([new_new_list, agent_todo], ignore_index=True).sort_values(by='in', ascending=True)
            continue
        
        # Sacamos los datos especificos del agente modificado
        agent_schedule = new_todolist_family[new_todolist_family['agent'] == agent].reset_index(drop=True)
        # Sacamos el 'todo' del agente
        agent_todo = todolist_family[todolist_family['agent'] == agent].reset_index(drop=True)
        
        # Dividimos el df agent_todo en superior e inferior (sin coincidencias con agent_schedule)
        df_before_min, df_after_max = df_division(agent_todo, agent_schedule)
        
        # La previa no se necesita mayor modificacion, por lo que se copia
        new_new_list = pd.concat([new_new_list, df_before_min], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
        # Se agrega la parte modificada previamente del agente
        new_new_list = pd.concat([new_new_list, agent_schedule], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
        # semodifica el df df_after_max para actualizar los tiempos
        df_after_max = time_adding(df_after_max, max(new_new_list['out']))
        # Y despues se suma
        new_new_list = pd.concat([new_new_list, df_after_max], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)

    return new_new_list

def time_adding(df_after_max, last_out):
    df_after_max_adapted = pd.DataFrame()
    
    for _, df_a_row in df_after_max.iterrows():
        new_in = last_out + int(df_a_row['conmu_time'])
        new_out = (new_in + df_a_row['time2spend']) if df_a_row['time2spend'] != 0 else df_a_row['closing'] # puede causar problemas si df_a_row['closing'] es inf
        print('df_a_row')
        print(df_a_row)
        if new_in == float('inf'):
            print('df_after_max')
            print(df_after_max)
            print(f"new_in es infinito")
            print(f"int(df_a_row['conmu_time']): {int(df_a_row['conmu_time'])}")
            input(f"last_out: {last_out}")
        if df_a_row['closing'] < new_in or df_a_row['closing'] < new_out:
            print(f"After adaptation, {df_a_row['agent']} was not able to fullfill '{df_a_row['todo']}' at {df_a_row['in']}.")
            continue
        if df_a_row['todo'] == 'Delivery' or df_a_row['todo'] == 'Collect': # df_a_row['todo'] == 'Waiting'???
            new_out == new_in
            
        rew_row ={
            'agent': df_a_row['agent'],
            'todo': df_a_row['todo'], 
            'osm_id': df_a_row['osm_id'], 
            'todo_type': df_a_row['todo_type'], 
            'opening': df_a_row['opening'], 
            'closing': df_a_row['closing'], # Issue 16
            'fixed?': df_a_row['fixed?'], 
            'time2spend': df_a_row['time2spend'], 
            'in': new_in, 
            'out': new_out,
            'conmu_time': int(df_a_row['conmu_time'])
        }   
        df_after_max_adapted = pd.concat([df_after_max_adapted, pd.DataFrame([rew_row])], ignore_index=True)
        last_out = new_out
    return df_after_max_adapted

def sort_route(osm_ids, helper):
    # Esta funcion deberia devolver el df ordenado con los verdaderos siempor de out
    # recuerda que el helper siempre debe ser el primero

    dependants = osm_ids[osm_ids['osm_id'] != helper['osm_id'].iloc[0]].copy()
    helper = osm_ids[osm_ids['osm_id'] == helper['osm_id'].iloc[0]].copy()    
                       
    # Detectar si la columna 'in' o 'out' está presente
    target_col = 'in' if 'in' in dependants.columns else 'out'
    
    current_max = 0
    for d_idx, d_row in dependants.iterrows():
        current_max = max([d_row['conmu_time'], current_max])
        dependants.loc[d_idx, 'conmu_time'] = current_max
    
    if target_col == 'in':
        # Aplicar la operación
        dependants.loc[:, target_col] = dependants[target_col] - dependants['conmu_time'] * dependants.index
        if not dependants.empty:
            helper.at[helper.index[0], target_col] = (dependants[target_col].max() + helper['conmu_time'].iloc[0])
        combined_df = pd.concat([dependants, helper], ignore_index=True)
        combined_df = combined_df.sort_values(by=target_col, ascending=False).reset_index(drop=True)
    else:
        # Aplicar la operación
        dependants.loc[:, target_col] = dependants[target_col] + dependants['conmu_time'] * dependants.index
        if not dependants.empty:
            helper.at[helper.index[0], target_col] = (dependants[target_col].min() - helper['conmu_time'].iloc[0])
        combined_df = pd.concat([dependants, helper], ignore_index=True)
        combined_df = combined_df.sort_values(by=target_col, ascending=True).reset_index(drop=True)
    print('osm_ids')
    print(osm_ids)
    print('combined_df')
    input(combined_df)
    
    return combined_df

def agent_collection(new_new_list, prev_matrix2cover, matrix2cover, helper):
    ## Creación de ruta de recogida
    # DataFrame con datos de outs
    out_osm_ids = pd.DataFrame(columns=['osm_id', 'out', 'conmu_time'])
    # Agrupamos para crear ruta de recogida
    osm_id_groups = prev_matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor maximo de out en el grupo que tenga time2spend != 0 (quién condiciona)
        filtered = oi_group[oi_group['time2spend']!=0]
        # Asignamos tiempo de conmutación del grupo
        group_conmu_time = oi_group['conmu_time'].max()
        # Asignamos tiempo de salida del grupo
        if filtered.empty:
            filtered = matrix2cover[matrix2cover['fixed?'] == True]
            if filtered.empty:
                group_out_time = oi_group['out'].min()
            else:
                group_out_time = filtered['in'].max() - group_conmu_time*len(filtered)
        else:
            group_out_time = filtered['out'].max()
        # Añadir nueva fila de datos
        rew_row ={ 
            'osm_id': name_group,
            'out': group_out_time,
            'conmu_time': group_conmu_time
        }   
        # Suma a dataframe
        out_osm_ids = pd.concat([out_osm_ids, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='out', ascending=False).reset_index(drop=True)
    
    # Crear la ruta ordenada
    sorted_route = sort_route(out_osm_ids, helper)
    
    ## Crear el nuevo schedule (parte de recogida de agentes)
    # Iteramos todos los osm_id de salida
    for _, name_group in sorted_route.iterrows():
        # Sacamos el grupo relativo al trip actual
        group = osm_id_groups.get_group(name_group['osm_id'])
        # Sacamos los valores a asignar para este grupo
        group_out_time = name_group['out']
        group_conmu_time = name_group['conmu_time']
        # Miramos los agenets que ya estan en movimiento
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'todo': 'Collect', 
                'osm_id': name_group['osm_id'], 
                'todo_type': 0, 
                'opening': 0, 
                'closing': float('inf'), 
                'fixed?': False, 
                'time2spend': 0, 
                'in': group_out_time, 
                'out': group_out_time,
                'conmu_time': group_conmu_time
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        # Despues agentes que se mueven por primera vez
        for _, agent in group.iterrows():
            # En caso de que el agente tenga que estar un tiempo especifico, de espera o se haya cerrado el servicio en el que estan, tendrá que esperar (sin poder hacer nada más)
            if (group_out_time > agent['closing']) or (group_out_time > agent['out'] and agent['time2spend'] != 0):
                ## Calculamos el tiempo de espera
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'todo': f'Waiting collection', 
                    'osm_id': agent['osm_id'],  # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related 
                    'fixed?': agent['fixed?'], 
                    'time2spend': group_out_time - agent['out'], 
                    'in': agent['out'], 
                    'out': group_out_time,
                    'conmu_time': group_conmu_time
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)

            # Actualización del caso original del agente
            rew_row ={
                'agent': agent['agent'],
                'todo': f"{agent['todo']}", 
                'osm_id': agent['osm_id'], 
                'todo_type': agent['todo_type'], 
                'opening': agent['opening'], 
                'closing': agent['closing'], 
                'fixed?': agent['fixed?'], 
                'time2spend': agent['time2spend'], 
                'in': agent['in'], 
                'out': group_out_time,
                'conmu_time': group_conmu_time
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    return new_new_list

def agent_delivery(new_new_list, new_list, matrix2cover, helper):
    ## Creación de ruta de recogida
    # DataFrame con datos de ins
    in_osm_ids = pd.DataFrame(columns=['osm_id', 'in', 'conmu_time'])
    # Agrupamos para crear ruta de entrega
    osm_id_groups = matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor minimo de in en el grupo que tenga fixed? == True (quién condiciona)
        filtered = oi_group[oi_group['fixed?'] == True]
        # Asignamos tiempo de conmutación del grupo
        group_conmu_time = oi_group['conmu_time'].max()
        # Asignamos tiempo de llegada del grupo
        if filtered.empty: # No tiene más condiciones, porque si es fixed? tendra un time2spend seguro, no hace falta comprobar
            group_in_time = oi_group['in'].max()
        else:
            group_in_time = filtered['in'].min()
        # Añadir nueva fila de datos
        rew_row ={ 
            'osm_id': name_group,
            'in': group_in_time,
            'conmu_time': group_conmu_time
        }   
        # Suma a dataframe
        in_osm_ids = pd.concat([in_osm_ids, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=False).reset_index(drop=True)
    
    # Crear la ruta ordenada
    sorted_route = sort_route(in_osm_ids, helper)
    
    print(sorted_route)
    
    ## Crear el nuevo schedule (parte de recogida de agentes)
    # Iteramos todos los osm_id de salida
    for _, name_group in sorted_route.iterrows():
        # Sacamos el grupo relativo al trip actual
        group = osm_id_groups.get_group(name_group['osm_id'])
        # Sacamos los valores a asignar para este grupo
        group_in_time = name_group['in']
        group_conmu_time = name_group['conmu_time']
        # Miramos los agentes que ya estan en movimiento (si estan presentes en new_new_list, son de otro ciclo, porque new_new_list empieza limpio)
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'todo': 'Delivery', 
                'osm_id': name_group['osm_id'], 
                'todo_type': 0, 
                'opening': 0, 
                'closing': float('inf'), 
                'fixed?': False, 
                'time2spend': 0, 
                'in': group_in_time, 
                'out': group_in_time,
                'conmu_time': group_conmu_time
            }
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        # Despues agentes que se mueven por primera vez
        for indxesss, agent in group.iterrows():
            # En caso de que el agente tenga que estar un tiempo especifico de espera en el servicio en el que estan (sin poder hacer nada más)
            if (group_in_time < agent['opening']) or (group_in_time < agent['in'] and agent['fixed?'] == True):
                # Calculamos el tiempo de espera
                waiting_time = max([agent['in'], agent['opening']]) - group_in_time # Tecnicamente [agent['in'], agent['opening']] deberian ser iguales si es fix, pero bue
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'todo': f'Waiting opening', 
                    'osm_id': agent['osm_id'], # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related
                    'fixed?': agent['fixed?'], 
                    'time2spend': waiting_time, 
                    'in': group_in_time, 
                    'out': agent['in'],
                    'conmu_time': group_conmu_time
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                
                new_out = (agent['in'] + agent['time2spend']) if agent['time2spend'] != 0 else agent['closing']
                
                if new_out <= agent['closing']:
                    rew_row ={
                        'agent': agent['agent'],
                        'todo': f"{agent['todo']}", 
                        'osm_id': agent['osm_id'], 
                        'todo_type': 0, 
                        'opening': agent['opening'], 
                        'closing': agent['closing'], 
                        'fixed?': agent['fixed?'], 
                        'time2spend': agent['time2spend'], 
                        'in': agent['in'], 
                        'out': new_out,
                        'conmu_time': group_conmu_time
                    }   
                    # Suma a dataframe
                    new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                else:
                    print(f"Due to 'Accompany' {agent['agent']} was not able to fullfill '{agent['todo']}' at {agent['in']}.")
            else:
                # Actualización del caso original del agente
                new_out = (group_in_time + agent['time2spend']) if agent['time2spend'] != 0 else agent['closing']
                print(f"para {agent['agent']} se calcula group_in_time ({group_in_time}):")
                print(agent)
                
                if agent['todo'] in ['Collect', 'Delivery']:
                    new_out = group_in_time
                elif new_out > agent['closing'] and agent['fixed?'] == False:
                    non_active_time = new_out - agent['closing'] 
                    new_out = agent['closing']
                    print(f"Due to 'Accompany' {agent['agent']} lost {non_active_time} minutes of'{agent['todo']}'.")
                elif new_out > agent['closing'] and agent['fixed?'] == True:
                    print(f"Due to 'Accompany' {agent['agent']} was not able to fullfill '{agent['todo']}' at {agent['in']}.")
                    continue
                    
                rew_row ={
                    'agent': agent['agent'],
                    'todo': agent['todo'], 
                    'osm_id': agent['osm_id'], 
                    'todo_type': 0, 
                    'opening': agent['opening'], 
                    'closing': agent['closing'], 
                    'fixed?': agent['fixed?'], 
                    'time2spend': agent['time2spend'], 
                    'in': group_in_time, 
                    'out': new_out,
                    'conmu_time': group_conmu_time
                }   
                print('name_group')
                print(name_group)
                print('rew_row')
                input(rew_row)
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                
    return new_new_list

def route_creation(matrix2cover, prev_matrix2cover): 
    ## Dividir los datos
    # DataFrame para almacenar resultados si vas a ir agregando algo más adelante
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed?','time2spend','in','out','conmu_time']
    new_list = pd.DataFrame(columns=columns)
    new_new_list = pd.DataFrame(columns=columns)
    # Filtrar dependientes con todo_type distinto de 0
    dependants_1 = matrix2cover[matrix2cover['todo_type'] != 0]
    # Filtrar en el DataFrame anterior aquellos agentes que están en dependants_1
    dependants_0 = prev_matrix2cover[prev_matrix2cover['agent'].isin(dependants_1['agent'])]
    # Filtrar helper con todo_type igual a 0
    helper_1 = matrix2cover[matrix2cover['todo_type'] == 0]
    # Conseguir los datos especificos del helper
    helper_0 = prev_matrix2cover[prev_matrix2cover['agent'].isin(helper_1['agent'])] # Issue 16
    helper_1 = matrix2cover[matrix2cover['agent'] == helper_0['agent'].iloc[0]]
    
    new_new_list = agent_collection(new_new_list, prev_matrix2cover, matrix2cover, helper_0)
    new_list = pd.concat([new_list, new_new_list], ignore_index=True)
    new_new_list = pd.DataFrame(columns=columns)
    new_new_list = agent_delivery(new_new_list, prev_matrix2cover, matrix2cover, helper_1)
    new_list = pd.concat([new_list, new_new_list], ignore_index=True)
    
    print('new_list')
    print(new_list)
    
    return new_list
  
def matrix2cover_creation(todolist_family, resties2cover):
    matrix2cover = pd.DataFrame()
    prev_matrix2cover = pd.DataFrame()
    
    rew_row = todolist_family[(todolist_family['agent'] == resties2cover['helper'].iloc[0]) & (todolist_family['osm_id'] == resties2cover['osm_id_h1'].iloc[0]) & (todolist_family['in'] == resties2cover['in_h'].iloc[0])]
    matrix2cover = pd.concat([matrix2cover, rew_row], ignore_index=True)
    rew_row = todolist_family[(todolist_family['agent'] == resties2cover['helper'].iloc[0]) & (todolist_family['osm_id'] == resties2cover['osm_id_h0'].iloc[0]) & (todolist_family['out'] == resties2cover['out_h'].iloc[0])]
    prev_matrix2cover = pd.concat([prev_matrix2cover, rew_row], ignore_index=True)
    for _, row_r2c in resties2cover.iterrows():
        rew_row = todolist_family[(todolist_family['agent'] == row_r2c['dependent']) & (todolist_family['osm_id'] == row_r2c['osm_id_d1']) & (todolist_family['in'] == row_r2c['in_d'])]
        matrix2cover = pd.concat([matrix2cover, rew_row], ignore_index=True)
        rew_row = todolist_family[(todolist_family['agent'] == row_r2c['dependent']) & (todolist_family['osm_id'] == row_r2c['osm_id_d0']) & (todolist_family['out'] == row_r2c['out_d'])]
        prev_matrix2cover = pd.concat([prev_matrix2cover, rew_row], ignore_index=True)
    return matrix2cover, prev_matrix2cover  
    
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
