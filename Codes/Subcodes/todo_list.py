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
        todolist_family = home_trips_adding(family_df, todolist_family)
        
        while max(todolist_family['todo_type']) > 0:
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique)            
            
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique)

        plot_agents_in_split_map(todolist_family, SG_relationship, save_path="recorridos_todos.html")
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

    matrix2cover, prev_matrix2cover = matrix2cover_creation(todolist_family, responsability_matrix)

    new_todolist_family, flag_to_jump = new_todolist_family_creation(matrix2cover, prev_matrix2cover)
    
    todolist_family = new_todolist_family_adaptation(todolist_family, matrix2cover, new_todolist_family, prev_matrix2cover, flag_to_jump)
    
    return todolist_family
   
def new_todolist_family_adaptation(todolist_family, matrix2cover, new_todolist_family, prev_matrix2cover, flag_to_jump):
    
    if flag_to_jump:
        todolist_family_adapted = todolist_family[~todolist_family.isin(matrix2cover.to_dict(orient='list')).all(axis=1)]
    else:
        todolist_family_adapted = todolist_family[~todolist_family.isin(prev_matrix2cover.to_dict(orient='list')).all(axis=1)]
        
        # Crear una máscara booleana para las filas que serán eliminadas
        mask = todolist_family.isin(prev_matrix2cover.to_dict(orient='list')).all(axis=1)

        # Guardar los índices de las filas eliminadas
        eliminated_indices = todolist_family[mask].index.tolist()
        
        # Eliminar las filas coincidentes
        todolist_family_adapted = todolist_family[~mask]
        
        # Paso 1: Obtener los índices eliminados y los agentes afectados
        eliminated_rows = todolist_family.loc[eliminated_indices]
        affected_agents = eliminated_rows['agent'].unique()

        # Paso 2: Crear una máscara para identificar filas con el mismo agente y con índice mayor a alguno eliminado
        rows_to_remove = []

        for agent in affected_agents:
            # Obtener todos los índices eliminados para este agente
            agent_deleted_indices = eliminated_rows[eliminated_rows['agent'] == agent].index

            # Para cada índice eliminado, encontrar en el DataFrame adaptado los índices mayores con el mismo agente
            for idx in agent_deleted_indices:
                agent_rows = todolist_family_adapted[
                    (todolist_family_adapted['agent'] == agent) & 
                    (todolist_family_adapted.index > idx)
                ]
                rows_to_remove.extend(agent_rows.index.tolist())

        # Eliminar duplicados por si algún índice fue agregado varias veces
        rows_to_remove = list(set(rows_to_remove))

        # Paso 3: Guardar esas filas en un nuevo DataFrame
        extra_rows_by_agents = todolist_family_adapted.loc[rows_to_remove]

        # Paso 4: Borrar esas filas de todolist_family_adapted
        todolist_family_adapted = todolist_family_adapted.drop(index=rows_to_remove)
        
        new_todolist_family = pd.concat([todolist_family_adapted, new_todolist_family], ignore_index=True).sort_values(by='in', ascending=True)
        
        agents_schedules2adapt = extra_rows_by_agents.groupby('agent')
 
        for agent_sch, agent_schedule in agents_schedules2adapt:
            new_todolist_family_agent = new_todolist_family[new_todolist_family['agent'] == agent_sch]
            last_agent_trip = new_todolist_family_agent[new_todolist_family_agent['out'] == max(new_todolist_family_agent['out'])]
            in_time = last_agent_trip['out'].iloc[0] + last_agent_trip['conmu_time'].iloc[0]
            
            for _, agent_trip in agent_schedule.iterrows():
                rew_row ={
                    'agent': agent_trip['agent'],
                    'todo': agent_trip['todo'], 
                    'osm_id': agent_trip['osm_id'], 
                    'todo_type': 0, 
                    'opening': agent_trip['opening'], 
                    'closing': agent_trip['closing'], 
                    'fixed?': agent_trip['fixed?'], 
                    'time2spend': agent_trip['time2spend'], 
                    'in': in_time, 
                    'out': agent_trip['out'],
                    'conmu_time': agent_trip['conmu_time']
                } 
                new_todolist_family = pd.concat([new_todolist_family, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    
        # en este caso, le metemos new_todolist_family a todolist_family_adapted, y luego tendremos que trabajar los matrix2cover
    
    up2adapt = todolist_family_adapted[todolist_family_adapted['in'] < min(matrix2cover['in'])] 
    down2adapt = todolist_family_adapted[todolist_family_adapted['in'] > min(matrix2cover['in'])] 
    # obtenemos los agentes que se han visto afectados
    agents_affected = matrix2cover['agent'].unique()
    # guardamos las rutas de aquellos que no han tenido ningun trip afectado
    schedule2add = todolist_family[~todolist_family['agent'].isin(agents_affected)]   
    
    for agent in agents_affected:
        schedule2consider = new_todolist_family[new_todolist_family['agent'] == agent]
        
        # parte superior de la matriz a adaptar
        schedule2adapt = up2adapt[up2adapt['agent'] == agent].sort_values(by='in', ascending=False)
        fist_in_time = min(schedule2consider['in'])
        conmu_time = int(schedule2consider['conmu_time'].iloc[0])
        for idx_s2a, row_s2a in schedule2adapt.iterrows():
                
            out_time = fist_in_time - conmu_time
            
            if out_time < row_s2a['opening']:
                print(f"{row_s2a['agent']} ha tenido un error al realizar {row_s2a['todo']} por ser out_time '{out_time}'")
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
                'conmu_time': conmu_time
            }   
            new_todolist_family = pd.concat([new_todolist_family, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
            conmu_time = int(row_s2a['conmu_time'])
        
        # parte INFERIOR de la matriz a adaptar
        schedule2adapt = down2adapt[down2adapt['agent'] == agent].sort_values(by='out', ascending=True)
        last_out_time = max(schedule2consider['out'])
        conmu_time = int(schedule2consider['conmu_time'].iloc[0])
        for idx_s2a, row_s2a in schedule2adapt.iterrows():
            in_time = last_out_time + conmu_time
            
            if in_time > row_s2a['closing']:
                print(f"(Due to addaptation) {row_s2a['agent']} was not able to fullfill '{row_s2a['todo']}' at {row_s2a['in']}")
                continue
            
            if row_s2a['time2spend'] == 0:
                out_time = row_s2a['closing']                                                          ###### Aqui habria que mirar el paso siguiente, si existe un paso siguiente 
                                                                                                       ###### mira si es fixed. Si es Fixed mirar su llegada y restarle el
                                                                                                       ###### tiempo de traslado para el out de este.
                                                                                                       ###### Si no es fixed, no tengo ni idea de que se podría hacer, sinceramente
            else:
                out_time = in_time + row_s2a['time2spend']
            #actualizamos 'fist_in_time' para que si hay otra vualta del for, mire este in y le reste el recorrido
            last_out_time = out_time
            
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
                'conmu_time': conmu_time
            }   
            new_todolist_family = pd.concat([new_todolist_family, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
            conmu_time = int(row_s2a['conmu_time'])
    
    # añadimos los agentes no afectados
    new_todolist_family = pd.concat([new_todolist_family, schedule2add], ignore_index=True).sort_values(by='in', ascending=True)           
    
    return new_todolist_family

def gestiona_esto_que_no_se_como_hacerlo_socorro(matrix2cover, new_todolist_family, prev_matrix2cover): 
    
    # DataFrame para almacenar resultados si vas a ir agregando algo más adelante
    new_new_list = pd.DataFrame()
    # Filtrar dependientes con todo_type distinto de 0
    dependants_1 = matrix2cover[matrix2cover['todo_type'] != 0]
    # Filtrar en el DataFrame anterior aquellos agentes que están en dependants_1
    dependants_0 = prev_matrix2cover[prev_matrix2cover['agent'].isin(dependants_1['agent'])]

    # Agrupar por 'out'
    grouped = dependants_0.groupby('out')
    # Ordenar los grupos por el valor de 'out' (clave del grupo)
    dependents2cover = dict(sorted(grouped, key=lambda x: x[0]))
    
    helper = prev_matrix2cover[prev_matrix2cover['todo_type'] == 0]
    
    out_time = next(iter(dependents2cover)) - helper['conmu_time'].iloc[0]
    
    rew_row ={
        'agent': helper['agent'].iloc[0],
        'todo': helper['todo'].iloc[0], 
        'osm_id': helper['osm_id'].iloc[0], 
        'todo_type': 0, 
        'opening': helper['opening'].iloc[0], 
        'closing': helper['closing'].iloc[0], 
        'fixed?': helper['fixed?'].iloc[0], 
        'time2spend': helper['time2spend'].iloc[0], 
        'in': helper['in'].iloc[0], 
        'out': out_time,
        'conmu_time': helper['conmu_time'].iloc[0]
    }   
    
    new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    prev_out_time = out_time
    
    for _, d2r_group in dependents2cover.items():
        out_time = prev_out_time + max(d2r_group['conmu_time'])
        prev_out_time = out_time
        
        for agent in new_new_list['agent'].unique():
            rew_row ={
                'agent': agent,
                'todo': f"accompaniment", 
                'osm_id': d2r_group['osm_id'].iloc[0], 
                'todo_type': 0, 
                'opening': d2r_group['opening'].iloc[0], 
                'closing': d2r_group['closing'].iloc[0], 
                'fixed?': d2r_group['fixed?'].iloc[0], 
                'time2spend': 0, 
                'in': out_time, 
                'out': out_time,
                'conmu_time': d2r_group['conmu_time'].iloc[0]
            } 
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        
        d2r_group['out'] = out_time
        new_new_list = pd.concat([new_new_list, d2r_group], ignore_index=True).sort_values(by='in', ascending=True)
        
    return new_new_list


def agent_transfer(matrix2cover, new_todolist_family, prev_matrix2cover):
    if matrix2cover['fixed?'].any():
        first_true_index = matrix2cover.index[matrix2cover['fixed?'] == True][0]
        in_time = matrix2cover['opening'].iloc[first_true_index] + first_true_index*matrix2cover['conmu_time'].iloc[0]
    else:                                                                               #El principal problema esta aqui, que hacer si no necesito acudir a ninguna hora en concreto?
        new_todolist_family = gestiona_esto_que_no_se_como_hacerlo_socorro(matrix2cover, new_todolist_family, prev_matrix2cover)
        return new_todolist_family, False
    
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

    # recoridos de las flotas
    grouped = matrix2cover.iloc[1:].groupby('osm_id')
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
    
    return new_todolist_family, True

def new_todolist_family_creation(matrix2cover, prev_matrix2cover):
    new_todolist_family = pd.DataFrame()
    
    # agent transfer 
    new_todolist_family, flag_to_jump = agent_transfer(matrix2cover, new_todolist_family, prev_matrix2cover)
    
    return new_todolist_family, flag_to_jump
    
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
