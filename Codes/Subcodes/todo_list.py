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
        
        ### home in
        # in_home
        row_in_home = filtered[filtered['out'] == max(filtered['out'])]
        in_home_time = row_in_home['out'].iloc[0] + row_in_home['conmu_time'].iloc[0]
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
        todolist_family = home_trips_adding(family_df, todolist_family)
        
        while max(todolist_family['todo_type']) > 0:
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique)            
            
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique)

        # plot_agents_in_split_map(todolist_family, SG_relationship, save_path="recorridos_todos.html")

def home_trips_adding(family_df, todolist_family):
    
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
   
def new_todolist_family_adaptation(todolist_family, matrix2cover, new_todolist_family, prev_matrix2cover):
    # Inicializamos el df de suma de cosas
    new_new_list = pd.DataFrame()
    # Pasamos por todos los agentes del schedule
    for agent in todolist_family['agent'].unique():
        # Si el agente no ha sido modificado
        if not agent in new_todolist_family['agent'].to_list():
            # Obtenemos los datos del agente
            agent_todo = todolist_family[todolist_family['agent'] == agent]
            # Se mantienen lo previo
            new_new_list = pd.concat([new_new_list, agent_todo], ignore_index=True).sort_values(by='in', ascending=True)
            continue
        # Sacamos los datos especificos del agente modificado
        agent_schedule = new_todolist_family[new_todolist_family['agent'] == agent].reset_index(drop=True)
        # Sacamos el 'todo' del agente
        agent_todo = todolist_family[todolist_family['agent'] == agent].reset_index(drop=True)
        
        filtered_df1 = agent_todo.merge(agent_schedule[['todo', 'osm_id']], on=['todo', 'osm_id'], how='inner')
        
        filtered_idx = agent_todo.merge(
            filtered_df1[['todo', 'osm_id']], 
            on=['todo', 'osm_id'], 
            how='inner'
        ).index
        
        first_idx = filtered_idx.min()
        last_idx = filtered_idx.max()
        # Logramos las partes previa y posterior al cambio
        prev_df1 = agent_todo.loc[:first_idx - 1]
        post_df1 = agent_todo.loc[last_idx + 1:]
        # La previa no se necesita mayor modificacion, por lo que se copia
        new_new_list = pd.concat([new_new_list, prev_df1], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
        # Si la seccion posterior no concluye el recorrido, se actualiza
        if max(agent_schedule['out']) != float('inf'):
            post_df1 = time_adding(post_df1, max(agent_schedule['out']))
        # Y despues se suma
        new_new_list = pd.concat([new_new_list, post_df1], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
        # Finalmente se agrega la parte modificada previamente del agente
        new_new_list = pd.concat([new_new_list, agent_schedule], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
    print('new_new_list:')
    input(new_new_list)
    return new_new_list

def time_adding(post_df1, last_out):
    post_df1_adapted = pd.DataFrame()
    for _, pd_row in post_df1.iterrows():
        new_in = last_out + int(pd_row['conmu_time'])
        new_out = (new_in + pd_row['time2spend']) if pd_row['time2spend'] != 0 else pd_row['closing']
        
        if pd_row['closing'] < new_in or pd_row['closing'] < new_out:
            print(f"After adaptation, {pd_row['agent']} was not able to fullfill '{pd_row['todo']}' at {pd_row['in']}.")
            input(post_df1)
            continue
        
        rew_row ={
            'agent': pd_row['agent'],
            'todo': pd_row['todo'], 
            'osm_id': pd_row['osm_id'], 
            'todo_type': pd_row['todo_type'], 
            'opening': pd_row['opening'], 
            'closing': pd_row['closing'], # Issue 16
            'fixed?': pd_row['fixed?'], 
            'time2spend': pd_row['time2spend'], 
            'in': new_in, 
            'out': new_out,
            'conmu_time': int(pd_row['conmu_time'])
        }   
        post_df1_adapted = pd.concat([post_df1_adapted, pd.DataFrame([rew_row])], ignore_index=True)
        last_out = new_out
    return post_df1_adapted

def sort_route(osm_ids, helper):
    # Esta funcion deberia devolver el df ordenado con los verdaderos siempor de out
    # recuerda que el helper siempre debe ser el primero

    dependants = osm_ids[osm_ids['osm_id'] != helper['osm_id'].iloc[0]].copy()
    helper = osm_ids[osm_ids['osm_id'] == helper['osm_id'].iloc[0]].copy()
    
    group_conmu_time = max(osm_ids['conmu_time'])
    
    # Detectar si la columna 'in' o 'out' está presente
    target_col = 'in' if 'in' in dependants.columns else 'out'

    # Aplicar la operación
    dependants.loc[:, target_col] = dependants[target_col] - group_conmu_time * dependants.index
    
    if not dependants.empty:
        helper.at[helper.index[0], target_col] = (dependants[target_col].iloc[0] + group_conmu_time)    
    
    combined_df = pd.concat([dependants, helper], ignore_index=True)
    combined_df = combined_df.sort_values(by=target_col, ascending=False).reset_index(drop=True)
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
                'todo': 'Accompany', 
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
                # Calculamos el tiempo de espera
                waiting_time = group_out_time - agent['out']
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'todo': f'Waiting', 
                    'osm_id': agent['osm_id'], 
                    'todo_type': 0, 
                    'opening': agent['opening'], 
                    'closing': agent['closing'], 
                    'fixed?': agent['fixed?'], 
                    'time2spend': waiting_time, 
                    'in': agent['out'], 
                    'out': group_out_time,
                    'conmu_time': group_conmu_time
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
            # Actualización del caso original del agente
            rew_row ={
                'agent': agent['agent'],
                'todo': agent['todo'], 
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
        if filtered.empty:
            group_in_time = oi_group['in'].min()
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
    
    ## Crear el nuevo schedule (parte de recogida de agentes)
    # Iteramos todos los osm_id de salida
    for _, name_group in sorted_route.iterrows():
        # Sacamos el grupo relativo al trip actual
        group = osm_id_groups.get_group(name_group['osm_id'])
        # Sacamos los valores a asignar para este grupo
        group_in_time = name_group['in']
        group_conmu_time = name_group['conmu_time']
        # Miramos los agenets que ya estan en movimiento
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'todo': 'Accompany', 
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
        for _, agent in group.iterrows():
            # En caso de que el agente tenga que estar un tiempo especifico, de espera o se haya cerrado el servicio en el que estan, tendrá que esperar (sin poder hacer nada más)
            if (group_in_time < agent['opening']) or (group_in_time < agent['in'] and agent['fixed?'] == True):
                # Calculamos el tiempo de espera
                waiting_time = agent['in'] - group_in_time
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'todo': f'Waiting', 
                    'osm_id': agent['osm_id'], 
                    'todo_type': 0, 
                    'opening': agent['opening'], 
                    'closing': agent['closing'], 
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
                        'todo': agent['todo'], 
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
                if new_out > agent['closing'] and agent['fixed?'] == False:
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
