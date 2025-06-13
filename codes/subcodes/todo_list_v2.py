import os
import sys
import random
import folium
import itertools
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt 
from folium.plugins import AntPath
from geopy.distance import geodesic
from collections import defaultdict
from datetime import datetime, timedelta

def todolist_family_initialization(SG_relationship, family_df, activities): # esta funcion necesita tambien relacion de familias y arquetipos
    # No puede trabajarse sin ningun tipo de actividades asignadas
    if activities == []:
        activities = ['WoS', 'Dutties', 'Entertainment']
    # Lista para sumar la salida y vuelta al hogar
    home_travels = ['Home_in', 'Home_out']
    activities = activities + home_travels
    # Inicializamos el df en el que meteremos los schedules
    todolist_family = pd.DataFrame()
    # Pasamos por cada agente que constitulle la familia 
    for idx_f_df, row_f_df in family_df.iterrows():               
        
        for activity in activities: 
            try:
                activity_amount = row_f_df[f'{activity}_amount']
            except Exception:
                activity_amount = 1
            
            for _ in range(activity_amount):
                
                try:
                    # En caso de que el agente cuente ya con un edificio especifico para realizar la accion acude a él
                    osm_id = row_f_df[activity.split('_')[0]]
                except Exception:
                    # En caso de que el agente NO cuente con un edificio especifico para realizar la accion
                    # Elegimos, según el tipo de actividad que lista de edificios pueden ser validos
                    # [Aqui habrá que meter una funcion de verdad, que valore en base a estadistica]
                    available_options = SG_relationship[SG_relationship['service_group'] == activity]['osm_id'].tolist()
                    # Elegimos uno aleatorio del grupo de validos
                    osm_id = random.choice(available_options)    
                try:
                    # En caso de que el agente tenga una hora especifica de accceso y salida
                    fixed = row_f_df[f'{activity}_fixed'] != 1
                    if fixed:
                        # Si tienen algún tipo de dependencia, no van a ir a trabajar, si no a estudiar, por lo que usan el horario de servicio
                        fixed_word = 'Service'
                    else:
                        # En caso de no depender es probable que vayan a trabajar, por lo que entrar en WoS
                        fixed_word = 'WoS'
                except Exception:
                    # En caso de que el agente NO tenga una hora especifica de accceso y salida
                    fixed = False
                    fixed_word = 'Service'
                ## Sacamos los datos relevantes
                opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
                closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
                try:
                    # En caso de que el agente tenga un tiempo requerido de actividad
                    time2spend = int(row_f_df[f'{activity}_time'])
                except Exception:
                    # En caso de que el agente NO tenga un tiempo requerido de actividad
                    time2spend = 0
                # El caso mayoritario de 'todo' para las acciones
                todo_type = row_f_df[f'{activity}_type'] if not activity in home_travels else row_f_df['WoS_type']
                # En caso de la primera acción
                if activity == activities[0]:
                    in_time = opening
                    out_time = (in_time + time2spend) if time2spend != 0 else closing
                # En caso de la última acción
                elif activity == activities[-1]:
                    filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
                    in_time = 0
                    row_out = filtered[filtered['in'] == min(filtered['in'])]
                    out_time = row_out['in'].iloc[0] - row_out['conmu_time'].iloc[0] # Mira al tiempo de entrada de la primera acción y le resta el tiempo de conmutación
                    todo_type = 0 # No necesitan que nadie les acompañe, porqueempiezan el día ahí
                # El resto de acciones
                else:
                    try:
                        filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
                    except:
                        filtered = pd.DataFrame()
                    if not filtered.empty:
                        in_time = max(filtered['out']) + filtered['conmu_time'].iloc[0]
                    else: # si resulta que hoy no trabajaba
                        in_time = opening
                    out_time = (in_time + time2spend) if time2spend != 0 else closing
                
                # Creamos la nueva fila en caso de que se pueda realizar la accion
                if in_time < closing and out_time <= closing:
                    rew_row ={
                        'agent': row_f_df['name'],
                        'todo': activity, 
                        'osm_id': osm_id, 
                        'todo_type': todo_type, 
                        'opening': opening, 
                        'closing': closing, 
                        'fixed?': fixed, 
                        'time2spend': time2spend, 
                        'in': in_time, 
                        'out': out_time,
                        'conmu_time': int(row_f_df['conmu_time'])
                    }
                    # La añadimos    
                    todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)
                else:
                    print(f"{row_f_df['name']} was not able to fullfill '{activity}' at {in_time}.")
                    print(f"They were trying to go at {in_time} until {out_time} but '{osm_id}' it closes at {closing}")
            
    # En caso de haber solo un agente en la familia, le transformamos en autosufuciente    
    if todolist_family['agent'].nunique() == 1:
        todolist_family['todo_type'] = 0
    
    todolist_family = todolist_family.sort_values(by='in', ascending=True).reset_index(drop=True)
    
    return todolist_family

def todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique):
    # Calculamos los apartados sobre los que actuar dentro de 'todolist_family'
    matrix2cover, prev_matrix2cover = matrix2cover_creation(todolist_family, responsability_matrix)
    # Tomamos los apartados sobre los que actuar y los adaptamos a las necesidades de los dependants
    new_todolist_family = route_creation(matrix2cover, prev_matrix2cover)
    # Adaptar el resto del schedule a las modificaciones realizadas
    todolist_family = new_todolist_family_adaptation(todolist_family, matrix2cover, new_todolist_family, prev_matrix2cover)
    return todolist_family

def route_creation(matrix2cover, prev_matrix2cover): 
    ## Dividir los datos
    # DataFrame para almacenar resultados si vas a ir agregando algo más adelante
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmu_time']
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
    ## Ejecutamos las actividades de recogida y entrega de agentes
    # Recogida
    new_new_list = agent_collection(new_new_list, prev_matrix2cover, matrix2cover, helper_0)
    new_list = pd.concat([new_list, new_new_list], ignore_index=True)
    
    new_new_list = pd.DataFrame(columns=columns)
    
    # Entrega
    new_new_list = agent_delivery(new_new_list, prev_matrix2cover, matrix2cover, helper_1)
    new_list = pd.concat([new_list, new_new_list], ignore_index=True)
    return new_list



def matrix2cover_creation(todolist_family, responsability_matrix):
    matrix2cover_rows = []
    prev_matrix2cover_rows = []

    # First row from helper (assumes first row contains this info)
    r0 = responsability_matrix.iloc[0]

    # Add current helper task
    current_helper_row = todolist_family[(todolist_family['agent'] == r0['helper']) & (todolist_family['osm_id'] == r0['osm_id_h1']) & (todolist_family['in'] == r0['in_h'])]
    matrix2cover_rows.append(current_helper_row)

    # Add previous helper task
    prev_helper_row = todolist_family[(todolist_family['agent'] == r0['helper']) & (todolist_family['osm_id'] == r0['osm_id_h0']) & (todolist_family['out'] == r0['out_h'])]
    prev_matrix2cover_rows.append(prev_helper_row)

    # Process dependents
    for _, row in responsability_matrix.iterrows():
        current_dependent_row = todolist_family[(todolist_family['agent'] == row['dependent']) & (todolist_family['osm_id'] == row['osm_id_d1']) & (todolist_family['in'] == row['in_d'])]
        matrix2cover_rows.append(current_dependent_row)

        prev_dependent_row = todolist_family[(todolist_family['agent'] == row['dependent']) & (todolist_family['osm_id'] == row['osm_id_d0']) &(todolist_family['out'] == row['out_d'])]
        prev_matrix2cover_rows.append(prev_dependent_row)

    # Concatenate once for efficiency
    matrix2cover = pd.concat(matrix2cover_rows, ignore_index=True)
    prev_matrix2cover = pd.concat(prev_matrix2cover_rows, ignore_index=True)

    return matrix2cover, prev_matrix2cover
  

def todolist_family_creation(df_citizens, SG_relationship):
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')
    results = pd.DataFrame()
    df_citizens_families = df_citizens.groupby('family')
    # Recorrer cada familia
    for family_name, family_df in df_citizens_families:
        
        # Creamos una lista de tareas con sus recorridos para cada agente de forma independiente
        todolist_family = todolist_family_initialization(SG_relationship, family_df, [])  # Aqui el '[]' se debe meter los servicios a analizar (pueden ser 'WoS', 'Dutties' y/o 'Entertainment')
                                                                                          # Deberiamos leerlo del doc de system management
        # Evaluamos todolist_family para observar si existen agentes con dependencias
        while max(todolist_family['todo_type']) > 0:
            # En caso de existir dependencias, se asignan responsables
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique)            
            # Tras esto, la matriz todo de esta familia es adaptada a las responsabilidades asignadas
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique)
            
            input('#' * 80)
        # plot_agents_in_split_map(todolist_family, SG_relationship, save_path="recorridos_todos.html")

# Función de distancia haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c   

def responsability_matrix_creation(todolist_family, SG_relationship_unique):   
    # Creamos la matriz de los resultados
    responsability_matrix = pd.DataFrame()
    # DataFrame con todo_type > 0 (dependientes con trips que requieren asistencia)
    dependents = todolist_family[todolist_family["todo_type"] > 0].add_suffix('_d')
    # DataFrame con todo_type == 0 (independientes)
    helpers = todolist_family[todolist_family["todo_type"] == 0].add_suffix('_h')
    # Eliminamos los independientes pero no capaces de ayudar (aquellos que en WoS son dependientes)
    agents_with_wos = todolist_family[(todolist_family['todo'] == 'WoS') & (todolist_family['todo_type'] != 0)]['agent'].unique()
    helpers = helpers[~helpers['agent_h'].isin(agents_with_wos)].reset_index(drop=True)
    # Producto cartesiano (todas las combinaciones posibles)
    df_combinado = helpers.merge(dependents, how='cross')
    # Calculamos todas las convinaciones
    for idx_df_conb, row_df_conb in df_combinado.iterrows():
        # Si la entrada es 0, sera la actividad de Home_out, por lo que lo ignoramos, no se plantean actividades previas a esta
        if row_df_conb['in_h'] == 0:
            continue
        # Sacamos las latitudes y longitudes de las posiciones de helper y dependant
        lat_h, lon_h = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_h'], ['lat', 'lon']].values[0]
        lat_d, lon_d = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_d'], ['lat', 'lon']].values[0]
        ## Sacamos los valores de las distancias
        # Distancia geografica
        geo_dist = haversine(lat_h, lon_h, lat_d, lon_d)
        # Distancia temporal
        time_dist = row_df_conb['in_d'] - row_df_conb['in_h'] # si esto da negativo, el agente helper tiene más tiempo para gestionar al dependant
        # Distancia social
        soc_dist = 1 # Aqui habrá que poner algo estadistico o algo
        # Calculamos la puntuación
        score = geo_dist + abs(time_dist)/100 + soc_dist # quizas cada uno entre el maximo?
        # Sacamos los schedule y step previo del helper
        try: 
            h_schedule = todolist_family[(todolist_family['agent'] == row_df_conb['agent_h']) & (todolist_family['out'] <= row_df_conb['in_h'])]            
            h_pre_step = h_schedule[h_schedule['out'] == max(h_schedule['out'])]
        except Exception:
            input('sa petao')
        
        # Sacamos los schedule y step previo del dependant
        d_schedule = todolist_family[(todolist_family['agent'] == row_df_conb['agent_d']) & (todolist_family['out'] <= row_df_conb['in_d'])] 
        d_pre_step = d_schedule[d_schedule['out'] == max(d_schedule['out'])]
        # Creamos la nueva fila
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
        # La añadimos a la matriz de responsabilidad
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
    
    