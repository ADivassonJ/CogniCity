import os
import sys
import random
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def create_family_level_1_schedule(pop_building, family_df, activities):
    """
      Summary: Crea la version inicial de los schedules de cada familia (level 1), 
    donde los agentes pueden realizar las actividades tal y como les apetezca, 
    independiente de si son o no capaces de hacerlo sin ayuda.

    Args:
        pop_building (DataFrame): Describe la poblacion de EDIFICIOS disponible, junto a sus caracteristicas.
        family_df (DataFrame): Describe las caracteristicas de cada agente CIUDADANO participe en una familia especifica.
        activities (list): Describe las actividades diarias a acometer por los agentes.

    Returns:
        todolist_family (DataFrame): Descripcion de level 1 del daily schedule de los agentes.
    """
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
        # Vamos actividad por actividad            
        for activity in activities:
            # Miramos si tenemos alguna cantidad de esta tarea a realizar (solo existe si la cantidad es != 1)
            try:
                activity_amount = row_f_df[f'{activity}_amount']
            except Exception:
                activity_amount = 1
            # Hacemos un loop para realizar la suma de tareas la X cantidad de veces necesaria
            for _ in range(int(activity_amount)):
                try:
                    # En caso de que el agente cuente ya con un edificio especifico para realizar la accion acude a él
                    osm_id = row_f_df[activity.split('_')[0]]
                except Exception:
                    # En caso de que el agente NO cuente con un edificio especifico para realizar la accion
                    # Elegimos, según el tipo de actividad que lista de edificios pueden ser validos
                    available_options = pop_building[pop_building['archetype'] == activity]['osm_id'].tolist() # ISSUE 33
                    # Elegimos uno aleatorio del grupo de validos
                    osm_id = random.choice(available_options)
                try:
                    # Si el agente tiene una hora de accion especifica fixed True, si no False
                    fixed = True if row_f_df[f'{activity}_fixed'] == 1 else False
                except Exception:
                    # En caso de que el agente NO tenga una hora especifica de accceso y salida
                    fixed = False
                # Si la actividad es el curro fixed sera WoS, si no Service 
                fixed_word = 'WoS' if activity == 'WoS' else 'Service'
                # Los casos de work y home son distintos. En el documento a referenciar tienen etiquetas distintas a su nombre de actividad
                if activity == 'WoS':
                    activity_re = 'work'
                elif activity in ['Home_in', 'Home_out']:
                    activity_re = 'Home'
                else:
                    activity_re = activity
                # Buscamos las horas de apertura y cierre del servicio/WoS
                opening = pop_building[(pop_building['osm_id'] == osm_id) & (pop_building['archetype'] == activity_re)][f'{fixed_word}_opening'].iloc[0]
                closing = pop_building[(pop_building['osm_id'] == osm_id) & (pop_building['archetype'] == activity_re)][f'{fixed_word}_closing'].iloc[0]
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
                    out_time = row_out['in'].iloc[0] - row_out['conmutime'].iloc[0] # Mira al tiempo de entrada de la primera acción y le resta el tiempo de conmutación
                    todo_type = 0 # No necesitan que nadie les acompañe, porque empiezan el día ahí
                # El resto de acciones
                else:
                    try:
                        filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
                    except:
                        filtered = pd.DataFrame()
                    if not filtered.empty:
                        in_time = max(filtered['out']) + filtered['conmutime'].iloc[0]
                    else: # si resulta que hoy no trabajaba
                        in_time = opening
                    out_time = (in_time + time2spend) if time2spend != 0 else closing
                
                # Creamos la nueva fila en caso de que se pueda realizar la accion
                if in_time < closing and out_time <= closing:
                    rew_row ={
                        'agent': row_f_df['name'],
                        'archetype': row_f_df['archetype'],
                        'todo': activity, 
                        'osm_id': osm_id, 
                        'todo_type': todo_type, 
                        'opening': opening, 
                        'closing': closing, 
                        'fixed': fixed, 
                        'time2spend': time2spend, 
                        'in': in_time, 
                        'out': out_time,
                        'conmutime': int(row_f_df['conmutime']),
                        'family': row_f_df['family'],
                        'family_archetype': row_f_df['family_archetype'],
                    }
                    # La añadimos    
                    todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)
                else:
                    print(f"{row_f_df['name']} was not able to fullfill '{activity}' at {in_time}.")
                    print(f"They were trying to go at {in_time} until {out_time} but '{osm_id}' it closes at {closing}")
                    
    ## En caso de que la familia cuente con dependientes pero no con helpers, se da por hecho que estos son saciados de algún modo por algún otro agente externo a la familia
    dependent = todolist_family[todolist_family['todo_type'] != 0]['agent'].unique()
    helpers = todolist_family[(todolist_family['todo_type'] == 0) & (~todolist_family['agent'].isin(dependent))]['agent'].unique()
    if (len(helpers) == 0) and (len(dependent) != 0):
        print(f"The family '{family_df['family'].iloc[0]}' has no responsables for its dependants. For LEVEL 2 analisys, we will consider their need somehow fulfilled, but it is an aproximation.")
        todolist_family['todo_type'] = 0
    # Ordenamos la schedule por hora de 'in' 
    todolist_family = todolist_family.sort_values(by='in', ascending=True).reset_index(drop=True)
    # Devolvemos el df de salida
    return todolist_family

def time_adding(df_after_max, last_out):
    # Inicializamos el df de resultados
    df_after_max_adapted = pd.DataFrame()
    # Actuamos sobre cada linea del df de acciones posteriores a las modificadas
    for _, df_a_row in df_after_max.iterrows():
        # Se calcula la nueva hora de entrada
        new_in = last_out + int(df_a_row['conmutime'])
        # Se calcula la nueva hora de salida
        new_out = (new_in + df_a_row['time2spend']) if df_a_row['time2spend'] != 0 else df_a_row['closing']
        if df_a_row['todo'] in ['Delivery', 'Collect']:
            new_out = new_in
        # En caso de detectarse alguna incompativilidad horaria, el agente no realiza la acción
        if df_a_row['closing'] < new_in or df_a_row['closing'] < new_out:
            print(f"After adaptation, {df_a_row['agent']} was not able to fullfill '{df_a_row['todo']}'.")
            print(f"Opening: {df_a_row['opening']} Closing: {df_a_row['closing']}.")
            print(f"new_in: {new_in} new_out: {new_out}.")
            continue
        # Se crea la nueva fila
        rew_row ={
            'agent': df_a_row['agent'],
            'archetype': df_a_row['archetype'],
            'todo': df_a_row['todo'], 
            'osm_id': df_a_row['osm_id'], 
            'todo_type': df_a_row['todo_type'], 
            'opening': df_a_row['opening'], 
            'closing': df_a_row['closing'], # Issue 16
            'fixed': df_a_row['fixed'], 
            'time2spend': df_a_row['time2spend'], 
            'in': new_in, 
            'out': new_out,
            'conmutime': int(df_a_row['conmutime']),
            'family': df_a_row['family'],
            'family_archetype': df_a_row['family_archetype'],
        }
        # Se añade la fila al df
        df_after_max_adapted = pd.concat([df_after_max_adapted, pd.DataFrame([rew_row])], ignore_index=True)
        # Actualizamos para la siguiente iteracción
        last_out = new_out
    return df_after_max_adapted

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
    return df_before_min, df_after_max

def new_todolist_family_adaptation(todolist_family, new_todolist_family): 
    # Inicializamos el df de resultados
    new_list = pd.DataFrame()
    # Pasamos por todos los agentes del schedule
    for agent in todolist_family['agent'].unique():
        # Inicializamos el df temporal resultados especificos para cada agente
        new_new_list = pd.DataFrame()
        # Si el agente no ha sido modificado
        if not agent in new_todolist_family['agent'].to_list():
            # Obtenemos los datos del agente
            agent_todo = todolist_family[todolist_family['agent'] == agent]
            # Se mantienen lo previo
            new_list = pd.concat([new_list, agent_todo], ignore_index=True).sort_values(by='in', ascending=True)
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
        # Miramos si existe algo posterior
        if not df_after_max.empty:
            # Se modifica el df df_after_max para actualizar los tiempos
            df_after_max = time_adding(df_after_max, max(new_new_list['out']))
        # Y despues se suma
        new_new_list = pd.concat([new_new_list, df_after_max], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
        new_list = pd.concat([new_list, new_new_list], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
    return new_list

def todolist_family_adaptation(responsability_matrix, todolist_family):
    # Calculamos los apartados sobre los que actuar dentro de 'todolist_family'
    matrix2cover, prev_matrix2cover = matrix2cover_creation(todolist_family, responsability_matrix)
    # Tomamos los apartados sobre los que actuar y los adaptamos a las necesidades de los dependants
    new_todolist_family = route_creation(matrix2cover, prev_matrix2cover)
    # Adaptar el resto del schedule a las modificaciones realizadas
    todolist_family = new_todolist_family_adaptation(todolist_family, new_todolist_family)
    return todolist_family

def route_creation(matrix2cover, prev_matrix2cover): 
    ## Dividir los datos
    # DataFrame para almacenar resultados si vas a ir agregando algo más adelante
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmutime']
    new_list = pd.DataFrame(columns=columns)
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
    new_new_list = agent_collection(prev_matrix2cover, matrix2cover, helper_0)
    new_list = pd.concat([new_list, new_new_list], ignore_index=True)
    # Entrega
    new_new_list = agent_delivery(prev_matrix2cover, matrix2cover, helper_1, new_list)
    new_list = pd.concat([new_list, new_new_list], ignore_index=True)
    return new_list

def sort_route(osm_ids, helper):
    # Esta funcion deberia devolver el df ordenado con los verdaderos siempor de out
    # recuerda que el helper siempre debe ser el primero

    dependants = osm_ids[osm_ids['osm_id'] != helper['osm_id'].iloc[0]].copy().reset_index(drop=True)  
    helper = osm_ids[osm_ids['osm_id'] == helper['osm_id'].iloc[0]].copy().reset_index(drop=True)
                       
    # Detectar si la columna 'in' o 'out' está presente
    target_col = 'in' if 'in' in dependants.columns else 'out'
    
    ascending = True
    if not dependants.empty:
        current_max = max(helper['conmutime'])
        
        for d_idx, d_row in dependants.iterrows():
            current_max = max([d_row['conmutime'], current_max])
            dependants.loc[d_idx, 'conmutime'] = current_max
            
        if target_col == 'in':
            # Aplicar la operación
            dependants.loc[:, target_col] = dependants[target_col].iloc[0] - dependants['conmutime'] * dependants.index
            if not dependants.empty:
                helper.at[helper.index[0], target_col] = (dependants[target_col].max() + helper['conmutime'].iloc[0])
            ascending = False
        else:
            # Aplicar la operación
            dependants.loc[:, target_col] = dependants[target_col].iloc[0] + dependants['conmutime'] * dependants.index
            if not dependants.empty:
                helper.at[helper.index[0], target_col] = (dependants[target_col].min() - helper['conmutime'].iloc[0])
            ascending = True
    
    combined_df = pd.concat([dependants, helper], ignore_index=True)
    combined_df = combined_df.sort_values(by=target_col, ascending=ascending).reset_index(drop=True)
    
    return combined_df

def agent_collection(prev_matrix2cover, matrix2cover, helper):
    # Inicializamos el df de los resultados
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmutime']
    new_new_list = pd.DataFrame(columns=columns)
    ## Creación de ruta de recogida
    # DataFrame con datos de outs
    out_osm_ids = pd.DataFrame(columns=['osm_id', 'out', 'conmutime'])
    # Agrupamos para crear ruta de recogida
    osm_id_groups = prev_matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor maximo de out en el grupo que tenga time2spend != 0 (quién condiciona)
        filtered = oi_group[oi_group['time2spend']!=0]
        # Asignamos tiempo de conmutación del grupo
        group_conmutime = oi_group['conmutime'].max()
        # Asignamos tiempo de salida del grupo
        if filtered.empty:
            filtered = matrix2cover[matrix2cover['fixed'] == True]
            if filtered.empty:
                group_out_time = oi_group['out'].min() # - group_conmutime*len(filtered) # El probelma es que lo he añadido pero len(filtered) es 0 porque se supone que si entra aqui es .empty
            else:
                group_out_time = filtered['in'].max() - group_conmutime*len(filtered)
        else:
            group_out_time = filtered['out'].max()
        # Añadir nueva fila de datos
        rew_row ={ 
            'osm_id': name_group,
            'out': group_out_time,
            'conmutime': group_conmutime
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
        group_conmutime = name_group['conmutime']
        # Miramos los agenets que ya estan en movimiento
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            agent_data = new_new_list[new_new_list['agent'] == p_agent].iloc[0]
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'archetype': agent_data['archetype'],
                'todo': 'Collect', 
                'osm_id': name_group['osm_id'], 
                'todo_type': 0, 
                'opening': group_out_time, 
                'closing': group_out_time, 
                'fixed': False, 
                'time2spend': 0, 
                'in': group_out_time, 
                'out': group_out_time,
                'conmutime': group_conmutime,
                'family': agent_data['family'],
                'family_archetype': agent_data['family_archetype'],
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        # Despues agentes que se mueven por primera vez
        for _, agent in group.iterrows():
            # Si el agente acaba la actividad y aun no le vienen a recoger
            if (group_out_time > agent['closing']) or (group_out_time > agent['out'] and agent['time2spend'] != 0):
                ## Calculamos el tiempo de espera
                waiting_time = group_out_time - min([agent['out'], agent['closing']])
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'archetype': agent['archetype'],
                    'todo': f'Waiting collection', 
                    'osm_id': agent['osm_id'],  # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related 
                    'fixed': agent['fixed'], 
                    'time2spend': waiting_time, 
                    'in': agent['out'], 
                    'out': group_out_time,
                    'conmutime': agent['conmutime'],
                    'family': agent['family'],
                    'family_archetype': agent['family_archetype'],
                }
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                new_out = agent['out']
                # En caso de que esta espera sea por parte del helper
                if agent['agent'] in helper['agent'].to_list():
                    rew_row ={
                        'agent': agent['agent'],
                        'archetype': agent['archetype'],
                        'todo': f'Collect', 
                        'osm_id': agent['osm_id'],  # Issue 17
                        'todo_type': 0, 
                        'opening': group_out_time,               # Es una accion not-place-related, pero sí time-related
                        'closing': group_out_time,    # Es una accion not-place-related, pero sí time-related 
                        'fixed': False, 
                        'time2spend': 0, 
                        'in': group_out_time, 
                        'out': group_out_time,
                        'conmutime': group_conmutime,
                        'family': agent['family'],
                        'family_archetype': agent['family_archetype'],
                    }   
                    # Suma a dataframe
                    new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
            else: 
                new_out = group_out_time
            # Actualización del caso original del agente
            rew_row ={
                'agent': agent['agent'],
                'archetype': agent['archetype'],
                'todo': agent['todo'], 
                'osm_id': agent['osm_id'], 
                'todo_type': agent['todo_type'], 
                'opening': agent['opening'], 
                'closing': agent['closing'], 
                'fixed': agent['fixed'], 
                'time2spend': agent['time2spend'], 
                'in': agent['in'], 
                'out': new_out,
                'conmutime': agent['conmutime'],
                'family': agent['family'],
                'family_archetype': agent['family_archetype'],
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    return new_new_list

def agent_delivery(prev_matrix2cover, matrix2cover, helper, agent_collection):   
    # Inicializamos el df de los resultados
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmutime']
    new_new_list = pd.DataFrame(columns=columns)
    ## Creación de ruta de recogida
    # DataFrame con datos de ins
    in_osm_ids = pd.DataFrame(columns=['osm_id', 'in', 'conmutime'])
    # Agrupamos para crear ruta de entrega
    osm_id_groups = matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor minimo de in en el grupo que tenga fixed == True (quién condiciona)
        filtered = oi_group[oi_group['fixed'] == True]
        # Asignamos tiempo de conmutación del grupo
        group_conmutime = oi_group['conmutime'].max()
        ## Asignamos tiempo de llegada del grupo
        # En caso de NO HABER ningún agente condicionante
        if filtered.empty: # No tiene más condiciones, porque si es fixed tendra un time2spend seguro, no hace falta comprobar
            max_out = agent_collection['out'].max()
            agents_row = agent_collection[agent_collection['out'] == max_out]
            max_in = agents_row['in'].max()           
            agents_row = agents_row[agents_row['in'] == max_in]
            group_in_time = agents_row['out'].iloc[0] + agents_row['conmutime'].iloc[0]
        else:
            # Tomamos como hora de entrada la del agente condicionante con hora más tenprana
            # Este será el último en ser entregado, asi te aseguras de que todos llegan antes de la hora, ninguno tarde.
            group_in_time = filtered['in'].min()
        # Añadir nueva fila de datos
        rew_row ={ 
            'osm_id': name_group,
            'in': group_in_time,
            'conmutime': group_conmutime
        }   
        # Suma a dataframe
        in_osm_ids = pd.concat([in_osm_ids, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=False).reset_index(drop=True)
    
    # Crear la ruta ordenada
    sorted_route = sort_route(in_osm_ids, helper)
    
    ## Crear el nuevo schedule (parte de recogida de agentes)
    # Iteramos todos los osm_id de llegada
    for _, name_group in sorted_route.iterrows():
        # Sacamos el grupo relativo al trip actual
        group = osm_id_groups.get_group(name_group['osm_id'])
        # Sacamos los valores a asignar para este grupo
        group_in_time = name_group['in']
        group_conmutime = name_group['conmutime']
        # Miramos los agentes que ya estan en movimiento (si estan presentes en new_new_list, son de otro ciclo, porque new_new_list empieza limpio)
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            agent_data = new_new_list[new_new_list['agent'] == p_agent].iloc[0]
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'archetype': agent_data['archetype'],
                'todo': 'Delivery', 
                'osm_id': name_group['osm_id'], 
                'todo_type': 0, 
                'opening': group_in_time, 
                'closing': group_in_time, 
                'fixed': False, 
                'time2spend': 0, 
                'in': group_in_time, 
                'out': group_in_time,
                'conmutime': group_conmutime,
                'family': agent_data['family'],
                'family_archetype': agent_data['family_archetype'],
            }
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        # Despues agentes que se mueven por primera vez
        for _, agent in group.iterrows():
            # Si el agente llega antes de la apertura
            if (group_in_time < agent['opening']) or (group_in_time < agent['in'] and agent['fixed'] == True):
                # Calculamos el tiempo de espera
                waiting_time = max([agent['in'], agent['opening']]) - group_in_time # Tecnicamente [agent['in'], agent['opening']] deberian ser iguales si es fix, pero bue
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'archetype': agent['archetype'],
                    'todo': f'Waiting opening', 
                    'osm_id': agent['osm_id'], # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related
                    'fixed': agent['fixed'], 
                    'time2spend': waiting_time, 
                    'in': group_in_time, 
                    'out': agent['in'],
                    'conmutime': agent['conmutime'],
                    'family': agent['family'],
                    'family_archetype': agent['family_archetype'],
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                in_time = agent['in']
            else:
                # El caso de entrada es un poco distinto del de salida. Mientras que la salida se hubica entre dos actividades,
                # la entrada esta entre 
                in_time = group_in_time
            
            new_out = (agent['in'] + agent['time2spend']) if agent['time2spend'] != 0 else agent['closing']
                
            rew_row ={
                'agent': agent['agent'],
                'archetype': agent['archetype'],
                'todo': agent['todo'], 
                'osm_id': agent['osm_id'], 
                'todo_type': 0, 
                'opening': agent['opening'], 
                'closing': agent['closing'], 
                'fixed': agent['fixed'], 
                'time2spend': agent['time2spend'], 
                'in': in_time, 
                'out': new_out,
                'conmutime': agent['conmutime'],
                'family': agent['family'],
                'family_archetype': agent['family_archetype'],
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    
    # Paso 1: Filtrar la fila deseada (como ya haces)
    filtered_nnl = new_new_list[new_new_list['todo'] == 'Delivery'] 
    filtered_nnl = filtered_nnl.copy()
    filtered_nnl['out'] = pd.to_numeric(filtered_nnl['out'], errors='coerce')
    filtered_nnl = filtered_nnl.loc[filtered_nnl.groupby(['agent'])['out'].idxmax()]
    filtered_nnl = filtered_nnl[filtered_nnl['agent'] == helper['agent'].iloc[0]]
    # Asegurarse de que solo hay una fila
    if len(filtered_nnl) > 1:
        print(f"filtered_nnl no tiene solo una fila:")
        print('filtered_nnl:')
        input(filtered_nnl)
    if len(filtered_nnl) == 0:
        return new_new_list
    # Paso 2: Buscar la fila idéntica en new_new_list
    target_row = filtered_nnl.iloc[0]  # Convertir a Series
    match_idx = (new_new_list == target_row).all(axis=1)
    # Paso 3: Obtener el índice y modificar conmutime
    idx = new_new_list[match_idx].index[0]
    
    new_new_list.at[idx, 'conmutime'] = matrix2cover.loc[
        matrix2cover['agent'] == helper['agent'].iloc[0], 'conmutime'
    ].iloc[0]

    return new_new_list

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
  

def todolist_family_creation(df_citizens, pop_building, system_management):
    """
    Summary: Esta funcion crea las daily schedule de los agentes, tanto de level 1 como de level 2

    Args:
        df_citizens (DataFrame): Describe la poblacion de CIUDADANOS disponible, junto a sus caracteristicas
        pop_building (DataFrame): Describe la poblacion de EDIFIOS disponible, junto a sus caracteristicas
        system_management (DataFrame): Describe caracteristicas principales del sistema

    Returns:
        level_1_results (DataFrame): daily schedule (level 1) de ciudadanos
        level_2_results (DataFrame): daily schedule (level 2) de ciudadanos
    """
    
    # Simplificamos la población de edificios, para tener datos basicos
    pop_building_unique = pop_building.drop_duplicates(subset='osm_id')
    # Inicializamos los df de resultados
    level_1_schedule = pd.DataFrame()
    level_2_schedule = pd.DataFrame()
    # Agrupamos la poblacion de ciudadanos en familias
    df_citizens_families = df_citizens.groupby('family')
    # Recorremos cada familia
    for family_name, family_df in tqdm(df_citizens_families, desc="Procesando familias"):
        # Sacamos la lista de actividades del archivo system_management
        activities = system_management['activities'].tolist()
        # Filtrar elementos que no sean NaN
        activities = [a for a in activities if pd.notna(a)]  
        
        ## LEVEL 1
        # Creamos una lista de tareas con sus recorridos para cada agente de forma independiente
        family_level_1_schedule = create_family_level_1_schedule(pop_building, family_df, activities)
        # Sumamos los datos a la lista de resultados de level 1
        level_1_schedule = pd.concat([level_1_schedule, family_level_1_schedule], ignore_index=True).reset_index(drop=True)
        
        ## LEVEL 2
        # Evaluamos todolist_family para observar si existen agentes con dependencias
        family_level_2_schedule = create_family_level_2_schedule(pop_building_unique, family_level_1_schedule)
        # Sumamos los datos a la lista de resultados de level 1
        level_2_schedule = pd.concat([level_2_schedule, family_level_2_schedule], ignore_index=True).reset_index(drop=True)
        
    # Devolvemos los resultados
    return level_1_schedule, level_2_schedule

def create_family_level_2_schedule(pop_building, family_level_1_schedule):
    # Inicializamos el df de resultados
    family_level_2_schedule = pd.DataFrame()
    # Identificamos las actividades que requieren asistencia
    dependents = family_level_1_schedule[family_level_1_schedule['todo_type'] != 0]
    # Logramos los datos de los independents
    independents = family_level_1_schedule[
        (family_level_1_schedule['todo_type'] == 0) & 
        (~family_level_1_schedule['agent'].isin(dependents['agent'])) &
        (family_level_1_schedule['in'] != 0)
    ]
    # Si no hay dependents en la familia, level 1 y level 2 serán iguales
    if dependents.empty:
        return family_level_1_schedule
    
    # Calculamos las responsabilidades
    responsability_matrix = create_responsability_matrix(dependents, independents, pop_building, family_level_1_schedule)
    
    
    
    ### Simplificamos los df de trabajo
    ## Quitamos los non-helpin independents 
    # Sacamos los schedule de los independents non-helpers
    new_rows, independents_schedule =  non_helpers(independents, family_level_1_schedule, responsability_matrix)
    # Como no van a molestar, los copiamos en el final
    family_level_2_schedule = pd.concat([family_level_2_schedule, new_rows], ignore_index=True).reset_index(drop=True)
    
    """ No se si me convence, igual es mejor hacerlo al final, rollo cubrir huecos en comparacion de las actividades realizadas en level 1 con las de level 2
    ## Quitamos las secciones unnafected
    new_rows, affected_actions = get_unaffected_actions(family_level_1_schedule, responsability_matrix)
    # Como no van a molestar, los copiamos en el final
    family_level_2_schedule = pd.concat([family_level_2_schedule, new_rows], ignore_index=True).reset_index(drop=True)
    """
    
    all_new_schedules = pd.DataFrame()
    
    # Agrupamos por ['agent_x', 'todo_x', 'osm_id_x'] para ver si el mismo agente recoge a mas de una persona
    grouped_responsability_matrix = responsability_matrix.groupby(['agent_x', 'todo_x', 'osm_id_x'])
    # Paso 1: Crear un DataFrame con un valor 'out_x' representativo por grupo
    group_keys_with_out = responsability_matrix.groupby(['agent_x', 'todo_x', 'osm_id_x'])['out_x'].first().reset_index()
    # Paso 2: Ordenar esos grupos por 'out_x'
    group_keys_with_out_sorted = group_keys_with_out.sort_values('out_x')
    # Paso 3: Recorrer los grupos en ese orden
    for _, row in group_keys_with_out_sorted.iterrows():
        # Recuperamos los datos 
        key = (row['agent_x'], row['todo_x'], row['osm_id_x'])
        df_grm = grouped_responsability_matrix.get_group(key)
        # Creamos el df 'data' que nos da info relevante de los agentes
        data = data_creation(df_grm)
        # Inicializamos el df de actividad previa
        previous_actions = pd.DataFrame()
        # Miramos los agentes
        for _, row_data in data.iterrows():
            # Sacamos su previa actividad
            new_row, _ = get_previous_action(family_level_1_schedule, row_data['agent'], row_data['in'])
            # La guardamos
            previous_actions = pd.concat([previous_actions, pd.DataFrame([new_row])], ignore_index=True).reset_index(drop=True)
        # Sacamos los nombres 
        helper_name= df_grm['agent_x'].unique()
        # Sacamos el data del prev
        prev_data = prev_data_creation(previous_actions, helper_name)
        # Adaptamos el schedule de los agentes implicados
        new_schedule = schedule_adaptation(prev_data, data, family_level_1_schedule)

        all_new_schedules = pd.concat([all_new_schedules, new_schedule], ignore_index=True).sort_values(by=['in','out']).reset_index(drop=True)
    
    adapted_schedule = schedules_compatibilisation(all_new_schedules, family_level_1_schedule)   
    
    
    
    return family_level_2_schedule

def schedules_compatibilisation(all_new_schedules, family_level_1_schedule):
    
    groups = all_new_schedules.groupby(['agent', 'todo', 'osm_id'])
    
    rows2add = pd.DataFrame()
    rows2delete = pd.DataFrame()
    
    for name, group in groups:
        # Si no existen duplicidades
        if len(group) == 1:
            continue
        
        rows2delete = pd.concat([rows2delete, group], ignore_index=False)
        
        
        new_row = group.iloc[0].copy()
        new_row['in'] = group['in'].max()
        new_row['out'] = group['out'].min()
        
        rows2add = pd.concat([rows2add, pd.DataFrame([new_row])], ignore_index=True)
        
    # Paso 1: Eliminar las filas de all_new_schedules usando los índices de rows2delete
    updated_df = all_new_schedules.drop(rows2delete.index)

    # Paso 2: Agregar las nuevas filas
    updated_df = pd.concat([updated_df, rows2add], ignore_index=True).sort_values(by=['in','out']).reset_index(drop=True)
    
    # Paso 1: Filtramos las columnas clave
    cols_to_check = ['agent', 'todo', 'osm_id']

    # Paso 2: Hacemos una comparación para encontrar las filas de family_level_1_schedule que no están en updated_df
    mask = ~family_level_1_schedule[cols_to_check].apply(tuple, axis=1).isin(
        updated_df[cols_to_check].apply(tuple, axis=1)
    )

    # Paso 3: Obtenemos solo las filas que faltan
    missing_rows = family_level_1_schedule[mask]

    # Paso 4: Añadimos esas filas a updated_df
    final_df = pd.concat([updated_df, missing_rows], ignore_index=True).sort_values(by=['in','out']).reset_index(drop=True)
    
    
    print('all_new_schedules:')
    input(final_df)

def prev_data_creation(previous_actions, helper_name):
    # Sacamos los datos del helper
    new_row_data = previous_actions[previous_actions['agent'].isin(helper_name)].iloc[0]
    # Creamos el df de los resultados
    prev_data = [{
        'agent': new_row_data['agent'],
        'type': 'helper',
        'fixed': new_row_data['fixed'],
        'time2spend': new_row_data['time2spend'],
        'in': new_row_data['in'],
        'out': new_row_data['out'],
        'osm_id': new_row_data['osm_id'],
        'todo': new_row_data['todo'],
        'conmutime': new_row_data['conmutime'],
    }]
    # Convertimos prev_data en df
    prev_data = pd.DataFrame(prev_data)
    # Sacamos los datos del helper
    new_row_data = previous_actions[~previous_actions['agent'].isin(helper_name)]
    # Sumamos todos los datos
    for _, row in new_row_data.iterrows():
        new_row = [{
            'agent': row['agent'],
            'type': 'dependent',
            'fixed': row['fixed'],
            'time2spend': row['time2spend'],
            'in': row['in'],
            'out': row['out'],
            'osm_id': row['osm_id'],
            'todo': row['todo'],
            'conmutime': row['conmutime'],
        }]
        prev_data = pd.concat([prev_data, pd.DataFrame(new_row)], ignore_index=True).reset_index(drop=True)
    return prev_data


def data_creation(df_grm):
    # Sacamos la lista de los nombres de los agentes participantes en esta accion
    data_names = np.unique(np.concatenate([df_grm['agent_x'].unique(), df_grm['agent_y'].unique()]))
    # Inicializamos el df de data
    data = pd.DataFrame()
    # Creamos el df 'data' 
    for agent in data_names:
        if agent in df_grm['agent_x'].unique():
            new_row = [{
                'agent': df_grm['agent_x'].iloc[0],
                'type': 'helper',
                'fixed': df_grm['fixed_x'].iloc[0],
                'time2spend': df_grm['time2spend_x'].iloc[0],
                'in': df_grm['in_x'].iloc[0],
                'out': df_grm['out_x'].iloc[0],
                'osm_id': df_grm['osm_id_x'].iloc[0],
                'todo': df_grm['todo_x'].iloc[0],
                'conmutime': df_grm['conmutime_x'].iloc[0],
            }]
            data = pd.concat([data, pd.DataFrame(new_row)], ignore_index=True).reset_index(drop=True)
        else:
            info_row = df_grm[df_grm['agent_y']==agent]
            new_row = [{
                'agent': info_row['agent_y'].iloc[0],
                'type': 'dependent',
                'fixed': info_row['fixed_y'].iloc[0],
                'time2spend': info_row['time2spend_y'].iloc[0],
                'in': info_row['in_y'].iloc[0],
                'out': info_row['out_y'].iloc[0],
                'osm_id': info_row['osm_id_y'].iloc[0],
                'todo': info_row['todo_y'].iloc[0],
                'conmutime': info_row['conmutime_y'].iloc[0],
            }]
            data = pd.concat([data, pd.DataFrame(new_row)], ignore_index=True).reset_index(drop=True)
    # Devolvemos data como resultado
    return data

def schedule_adaptation(prev_data, data, family_level_1_schedule):
    # Sacamos las condiciones que afectan en el orden de uso y condiciones de las actividades
    c_condition = (prev_data['time2spend'] != 0).any()
    d_condition = (data['fixed']).any()
    # Si se cumple el caso cd01
    if not c_condition and d_condition:
        # Realizamos el delivery de los agentes por parte del helper
        delivery_schedule, condition_time = dc_action(data, family_level_1_schedule, 'Delivery')
        # Realizamos el collection de los agentes por parte del helper
        collection_schedule,_ = dc_action(prev_data, family_level_1_schedule, 'Collection', condition_time)
    # Si se cumple otro caso
    else:
        # Realizamos el delivery de los agentes por parte del helper
        collection_schedule, condition_time = dc_action(prev_data, family_level_1_schedule, 'Collection')
        # Realizamos el collection de los agentes por parte del helper
        delivery_schedule,_ = dc_action(data, family_level_1_schedule, 'Delivery', condition_time)
    # Sumamos ambas matrices
    new_schedule = pd.concat([collection_schedule, delivery_schedule], ignore_index=True).sort_values(by=['in','out']).reset_index(drop=True)
    # Devolvemos el resultado
    return new_schedule

def dc_action(data, family_level_1_schedule, action, condition_time=0):
    # Ordenamos la lista y asignamos conmutimes adecuados a las interacciones por grupo
    data = comnutime_assignment(data, action)
    # Adaptamos los 'in' o 'out' para que los agentes llegen a tiempo (o antes de tiempo, pero no tarde)
    data, condition_time = action_times_calculation(data, action, condition_time)    
    # Adaptamos family_level_1_schedule para las necesidades de Delivery
    delivery_schedule = new_schedule_creation(data, family_level_1_schedule, action)
    return delivery_schedule, condition_time

def action_times_calculation(data, l_action, condition_time):
    """
      Summary: Calcula en un df simplificado (data), los tiempos de 'in' o 'out' para 'Delivery' o 
    'Collection', en base a las necesidades de las actuaciones de los agentes, es decir, si cuentan
    con actividades time-related. Busca optimizar los tiempos de espera, asegurando siempre que sea
    posible que ningún agente inclumpla las normas de fixed y time2spend.
    
    Args:
        data (DataFrame): Presenta datos relevantes de los agentes implicados en la actuacion a realizar
        l_action (str): Describe la acción que se realiza, es decir, 'Delivery' o 'Collection'.
        condition_time (int): Define el tiempo en el que se ejecuta la accion contraria a action (si 
      action es 'Collection', define el momento en el que se debe realizar el primer 'in' de Delivery).

    Returns:
        results (DataFrame): Similar a data, pero con los tiempos de la accion adapatdos a las necesidades
      del conjunto de agentes participantes.
        condition_time (int): Minuto del que depende el conjunto para posteriores actividades (ej. si la 
      entrega de los agentes debe empezar a las 400 sera 400-conmutime, para que la accion de collection 
      sepa cuando debe haberse completado).
    """
    
    # Copiamos el l_action en action 'just in case' que luego 'jugamos' un poco con ello
    action = l_action
    # Evaluamos el tipo de accion a acometer y asignamos el 'word' en base a esto
    if action == 'Delivery':
        word = 'in'
    else:
        word = 'out'
    # Sacamos los datos de helper
    helper_row = data[data['type'] == 'helper'].iloc[0].copy()
    # El resto del DataFrame, sin la fila 'helper' y lo ordenamos
    other_rows = data[data['type'] != 'helper'].sort_values(by=word, ascending=True).reset_index(drop=True)
    # Ordenamos el df distinto dependiendo de si es entrega o recoleccion
    if action == 'Delivery':
        analis_rows = pd.concat([other_rows, pd.DataFrame([helper_row])], ignore_index=True).reset_index(drop=True)
    else: 
        analis_rows = pd.concat([pd.DataFrame([helper_row]), other_rows], ignore_index=True).reset_index(drop=True)
    # En caso de que existan osm_id compartidos, agrupamos por osm_id
    POI_groups = analis_rows.groupby(by='osm_id')
    # Inicializamos unos df de trabajo
    new_analis_rows = pd.DataFrame()
    rest_rows = pd.DataFrame()
    # Pasamos por cada grupo en orden 
    for osm_id in analis_rows['osm_id'].unique():
        # Sacamos los datos de cada grupo
        group = POI_groups.get_group(osm_id).reset_index(drop=True)
        # Casos distintos para cada actuacion
        if action == 'Delivery':
            # Sacamos la fila más cohartante del grupo y las demás
            cohart = group[group['fixed']].reset_index(drop=True)
            # Si no hay cohartantes pues copiamos group
            if cohart.empty:
                cohart = group
            # De los cohartantes, sacamos el valor más cohartante
            row = cohart.iloc[0]
            new_row = group[group['agent']!=row['agent']]            
        else:
            # Sacamos la fila más cohartante del grupo y las demás
            cohart = group[group['time2spend']!=0].reset_index(drop=True)
            # Si no hay cohartantes pues copiamos group
            if cohart.empty:
                cohart = group
            # De los cohartantes, sacamos el valor más cohartante
            row = cohart.iloc[-1]
            new_row = group[group['agent']!=row['agent']]
        # Sumamos las filas a los df de trabajo
        new_analis_rows = pd.concat([new_analis_rows, pd.DataFrame([row])], ignore_index=True).reset_index(drop=True)
        rest_rows = pd.concat([rest_rows, new_row], ignore_index=True).reset_index(drop=True)
    # Actualizamos el df 'analis_rows'
    analis_rows = new_analis_rows
    # Si tenemos algún tiempo de condicion
    if condition_time != 0:
        # Asignamos valores de restriccion para que el posterior algoritmo los considere
        analis_rows['fixed'] = True
        analis_rows['time2spend'] = 1
        # El funcionamiento de la actuacion se invierte 
        if action == 'Delivery':
            # Copiamos el valor más restrictivo
            analis_rows[word] = analis_rows[word].apply(lambda x: max(x, condition_time))
            action = 'Collection'
        else:
            # Copiamos el valor más restrictivo
            analis_rows[word] = analis_rows[word].apply(lambda x: min(x, condition_time))
            action = 'Delivery'    
    # Pasamos por las filas para crear una columna de tiempo real de la actuación
    for idx, row in analis_rows.iterrows():
        if idx == 0:
            analis_rows.at[idx, f'r{word}'] = row[word]
        else:
            analis_rows.at[idx, f'r{word}'] = last_rword + row['conmutime']
        last_rword = analis_rows.at[idx, f'r{word}']      
    # Por comodidad, lo ponemos como int
    analis_rows[f'r{word}'] = analis_rows[f'r{word}'].astype(int)
    # Sacamos la diferencia de lo real y de lo que debería ser originalmente
    analis_rows['diff'] = analis_rows[f'r{word}'] - analis_rows[word]
    # Actuamos diferente dependiendo de la accion
    if action == 'Delivery':
        # Filtramos el df a aquellos que pueden provocar una dependencia
        filtered_rows = analis_rows[analis_rows['fixed']]
        # Buscamos el valor que coharta al resto
        act_diff = filtered_rows['diff'].max()
        # Evaluamos la condicion
        condition = act_diff > 0
    else:
        # Filtramos el df a aquellos que pueden provocar una dependencia
        filtered_rows = analis_rows[analis_rows['time2spend']!=0]
        # Buscamos el valor que coharta al resto
        act_diff = analis_rows['diff'].min()
        # Evaluamos la condicion
        condition = act_diff < 0
    # Si se da la condición, es necesario actualizar las horas de actuacion
    if condition:
        analis_rows[f'r{word}'] = analis_rows[f'r{word}'] - act_diff
    # Sustituimos word por f'r{word}' donde exista
    analis_rows[word] = analis_rows[f'r{word}'].combine_first(analis_rows[word])
    # Eliminamos la columna f'r{word}' si ya no la necesitamos
    analis_rows = analis_rows.drop(columns=[f'r{word}', 'diff']).reset_index(drop=True)
    # Calculamos el 'condition_time' de un posible caso posterior
    if action == 'Delivery':        
        condition_time = analis_rows[word].iloc[0] - analis_rows['conmutime'].iloc[0]
    else:
        condition_time = analis_rows[word].iloc[-1] + analis_rows['conmutime'].iloc[-1]
    # Inicializamos el df de resultados
    results = analis_rows
    # Añadimos, con los mismos tiempos, los rows previamente ignorados por compartir osm_id
    for _, row in analis_rows.iterrows():
        # Sacamos los coincidentes
        rows2add = rest_rows[rest_rows['osm_id'] == row['osm_id']]
        # Les cambiamos el action time
        rows2add[word] = row[word]
        # Los añadimos al df de resultados
        results = pd.concat([results, rows2add], ignore_index=True).reset_index(drop=True)
    # Ordenamos los resultados
    results = results.sort_values(by=[word], ascending=True).reset_index(drop=True)
    # Devolvemos el df de salida
    return results, condition_time


def new_schedule_creation(data, family_level_1_schedule, action):
    """
      Summary: Modifica, en base al df 'data' los datos originales del schedule, para adaptarlos
    a las necesidades especificadas en el df en cuestion, añadiendo las acciones intermedias como 
    'Delivery' o 'Collect' y 'Waiting' del tipo que sea.

    Args:
      data (DataFrame): Presenta datos relevantes de los agentes implicados en la actuacion a realizar
      family_level_1_schedule (DataFrame): Descripcion de los daily schedule de los agentes, de level 
    1, es decir, independientes entre sí, sin interacciones familiares.
      action (str):  Describe la acción que se realiza, es decir, 'Delivery' o 'Collection'.

    Returns:
      new_schedule (DataFrame): Describe las acciones a realizar por los agentes para cada actuacion
    """
    
    # Inicializamos unos datos especificos para cada escenario
    if action == 'Delivery':
        word = 'in'
        wordnt = 'out'
        order = False
    else:
        word = 'out'
        wordnt = 'in'
        order = True
    # Ordenamos los datos
    data = data.sort_values(by=word, ascending=order).reset_index(drop=True)
    # Inicializamos el df de resultados
    new_schedule = pd.DataFrame()
    # Pasamos por todas las lineas de data (en orden)
    for idx, row in data.iterrows():
        # Sacamos los valores originales para dicha accion
        new_row = family_level_1_schedule[(family_level_1_schedule['agent'] == row['agent']) &
                                          (family_level_1_schedule['osm_id'] == row['osm_id']) &
                                          (family_level_1_schedule['todo'] == row['todo'])].iloc[0]
        ## Actualizamos algunos datos
        new_row['todo_type'] = 0 # Como ya hemos saciado la necesidad, ya no es dependiente
        new_row[word] = row[word] 
        new_row['conmutime'] = row['conmutime']
        # Diferentes casos
        if action == 'Delivery':
            condition = row[word] < new_row['opening']
            new_time = new_row['opening']
            wait_action = 'opening'
        else:
            condition = row[word] > new_row['closing']
            new_time = new_row['closing']
            wait_action = 'collection'
        # Si se cumple la condicion significa que el agente requerira de esperar
        if condition:
            # Actualizamos el word de la accion principal
            new_row[word] = new_time
            # Cambiamos los datos para ajustarlo al row de espera
            alt_new_row = new_row.copy()
            alt_new_row['todo'] = f'Waiting {wait_action}'
            alt_new_row['time2spend'] = abs(row[word]-new_time)
            alt_new_row[word] = row[word]
            alt_new_row[wordnt] = new_time
            # Añadimos la accion original modificada
            new_schedule = pd.concat([new_schedule, pd.DataFrame([alt_new_row])], ignore_index=True).reset_index(drop=True)
        # Añadimos la accion original modificada
        new_schedule = pd.concat([new_schedule, pd.DataFrame([new_row])], ignore_index=True).reset_index(drop=True)    
        # Miramos los agentes que ya estan en la flota
        agents_acting = new_schedule['agent'].unique()
        # Pasamos por cada uno de ellos
        for agent in agents_acting:
            # Copiamos los datos
            alt_new_row = new_row.copy()
            rox = new_schedule[new_schedule['agent']==agent].iloc[0].copy()
            # Si es el mismo evaluado en este loop, nos saltamos este paso
            if agent == new_row['agent'] or rox['osm_id'] == new_row['osm_id']:
                continue
            ## Actualizamos algunos datos
            alt_new_row['agent'] = rox['agent']
            alt_new_row['archetype'] = rox['archetype']
            alt_new_row['todo'] = action
            alt_new_row[word] = row[word]
            alt_new_row[wordnt] = alt_new_row[word]
            alt_new_row['time2spend'] = 0 # Estan en operaciones de asistencia, por lo que es no-time related
            # Añadimos la accion original modificada
            new_schedule = pd.concat([new_schedule, pd.DataFrame([alt_new_row])], ignore_index=True).reset_index(drop=True)
    # Añadimos el acto original al final, para evitar fallos entre wait y action (delivery o collection)
    new_schedule = new_schedule.sort_values(by=['in', 'out'], ascending=True).reset_index(drop=True)
    # Devolvemos el df de resultados
    return new_schedule

def delivery_schedule_creation(data, family_level_1_schedule):
    # Inicializamos el df de resultados
    delivery_df = pd.DataFrame()
    # Pasamos por todas las filas de data
    for idx, row in data.iterrows():
        # Sacamos los valores originales para dicha accion
        new_row = family_level_1_schedule[(family_level_1_schedule['agent'] == row['agent']) &
                                          (family_level_1_schedule['osm_id'] == row['osm_id']) &
                                          (family_level_1_schedule['todo'] == row['todo'])].iloc[0]
        # Miramos los agentes que quedan (estos son los que realizan la entrega)
        for agent in data['agent'].iloc[idx+1:]:
            # Sacamos la info de cada agente
            agent_info = family_level_1_schedule[family_level_1_schedule['agent'] == agent].iloc[0]
            # Modificamos sus lineas
            new_action = new_row.copy()
            new_action['agent'] = agent
            new_action['archetype'] = agent_info['archetype']
            new_action['todo'] = 'Delivery'
            new_action['todo_type'] = 0
            new_action['fixed'] = True
            new_action['time2spend'] = 0
            new_action['in'] = row['in']
            new_action['out'] = row['in']
            new_action['conmutime'] = row['conmutime']
            # La añadimos
            delivery_df = pd.concat([delivery_df, pd.DataFrame([new_action])], ignore_index=True).reset_index(drop=True)
        # En caso de que el agente llegue antes de tiempo, deberá esperar
        if row['in'] < new_row['opening']:
            # Modificamos la linea de acción para realizar la espera
            new_action = new_row.copy()
            new_action['todo'] = 'Waiting opening'
            new_action['todo_type'] = 0
            new_action['fixed'] = True
            new_action['time2spend'] = new_row['opening'] - row['in']
            new_action['in'] = row['in']
            new_action['out'] = new_row['opening']
            new_action['conmutime'] = row['conmutime']
            # La añadimos
            delivery_df = pd.concat([delivery_df, pd.DataFrame([new_action])], ignore_index=True).reset_index(drop=True)
        # Modificamos la acción a su nuevo in
        new_row['todo_type'] = 0
        new_row['in'] = max([row['in'], new_row['opening']])
        new_row['out'] = row['in'] + row['time2spend'] if new_row['todo'] != 'Entertainment' else new_row['closing']
        new_row['conmutime'] = row['conmutime']
        # Lo añadimos
        delivery_df = pd.concat([delivery_df, pd.DataFrame([new_row])], ignore_index=True).reset_index(drop=True)
    # Devolvemos los resultados
    return delivery_df

def in_times_calculation(data):
    # El resto del DataFrame, sin la fila 'helper' y lo ordenamos descendente
    other_rows = data[data['type'] != 'helper'].sort_values(by='in', ascending=False).reset_index(drop=True)
    # Inicializamos el df de resultados
    df = pd.DataFrame()
    # Pasamos por todas las filas de datos relativos a dependents
    for idx, row in other_rows.iterrows():
        # Asignamos el valor real de 'in'
        in_v = row['in']
        # Si es el primer caso (el que deberia provocar dependencias en el resto)
        if idx == 0:
            # Mantenemos igual el in
            rin_v = in_v
        else:
            # Adaptamos a lo que antes teniamos menos el tiempo de conmutacion
            rin_v = last_in_v - row['conmutime']
        # Actualizamos el valor historico de rin
        last_in_v = rin_v
        # Calculamos la diferrencia entre intencio y real
        diff_v = in_v - rin_v
        # Creamos la nueva linea
        new_row = [{
            'agent': row['agent'],
            'in': in_v,
            'rin': rin_v,
            'diff': diff_v,
        }]
        # Añadimos la nueva fila
        df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True).reset_index(drop=True)
    # Recalculamos el in de cada accion
    df['rin'] = df['rin'] + df['diff'].min() if df['diff'].min() < 0 else 0
    # Hacemos merge para incorporar la columna 'rin' desde df al DataFrame other_rows
    other_rows_updated = other_rows.merge(df[['agent', 'in', 'rin']], on=['agent', 'in'], how='left')
    # Sustituimos 'in' por 'rin' donde exista
    other_rows_updated['in'] = other_rows_updated['rin'].combine_first(other_rows_updated['in'])
    # Eliminamos la columna 'rin' si ya no la necesitamos
    other_rows_updated = other_rows_updated.drop(columns=['rin']).sort_values(by='in', ascending=True).reset_index(drop=True)
    # Sacamos los datos de helper
    helper_row = data[data['type'] == 'helper'].copy()
    # Actualizamos su in (deberá ser despues de llevar a todos los agentes)
    helper_row['in'] = other_rows_updated['in'].max() + helper_row['conmutime']
    # Lo añadimos el df de salida
    data = pd.concat([other_rows_updated, helper_row], ignore_index=True).reset_index(drop=True)
    # Devolvemos el df de salida
    return data
    
    
def comnutime_assignment(data, action):
    ## Ordenamos para poder trabajar
    # Separar la fila que contiene 'helper'
    helper_row = data[data['type'] == 'helper']
    # El resto del DataFrame, sin la fila 'helper'
    other_rows = data[data['type'] != 'helper']
    # Ordenamos distinto dependiendo de la accion a realizar
    if action == 'Delivery':
        other_rows.sort_values(by='in', ascending=True)
    else:
        other_rows.sort_values(by='out', ascending=False)
    # Concatenar: primero el resto, luego la fila 'helper'
    data_sorted = pd.concat([other_rows,helper_row], ignore_index=True).reset_index(drop=True)
    
    ## Asignamos los tiempos de conmutacion maximo por recorrido
    for idx in range(len(data_sorted)):
        max_from_idx = data_sorted['conmutime'].iloc[idx:].max()
        data_sorted.loc[idx, 'conmutime'] = max_from_idx
    
    # Devolvemos el df con los conmutime bien
    return data_sorted    


def non_helpers(independents, family_level_1_schedule, responsability_matrix):   
    # Sacamos los nombres de los independents
    independents_names = independents['agent'].unique()
    # Filtramos family_level_1_schedule
    independents_schedule = family_level_1_schedule[family_level_1_schedule['agent'].isin(independents_names)]
    # Guardamos los datos en el df de resultado en dicho caso
    new_rows =  independents_schedule[~independents_schedule['agent'].isin(responsability_matrix['agent_x'].unique())]
    # Actualizamos 'independents_schedule' eliminando el agente non-helper
    independents_schedule = independents_schedule[independents_schedule['agent'].isin(responsability_matrix['agent_x'].unique())]
    # Devolvemos las nuevas filas y 'independents_schedule' actualizado
    return new_rows, independents_schedule

def get_unaffected_actions(family_level_1_schedule, responsability_matrix):
    # Inicializamos los df de resultados
    all_unaffected_actions = pd.DataFrame()
    all_affected_actions = pd.DataFrame()
    # Sacamos los nombres de los independent agent
    indep_agents = responsability_matrix['agent_x'].unique()
    # Sacamos los nombres de los dependent agent
    depend_agents = responsability_matrix['agent_y'].unique()
    
    for indep in indep_agents:
        # Sacamos las reposnability matrix del agente de analisis
        helper_responsabilities = responsability_matrix[responsability_matrix['agent_x'] == indep]
        # Sacamos tambien la schedule del agente
        helper_schedule = family_level_1_schedule[family_level_1_schedule['agent'] == indep]
        # Encontrar el valor mínimo de out_x
        min_out_x = helper_responsabilities['out_x'].min()
        # Crear nueva lista con el que tiene ese valor mínimo y reseteamos el indice (just in case)
        first_help = helper_responsabilities[helper_responsabilities['out_x'] == min_out_x].reset_index(drop=True)
        prev_action, idx = get_previous_action(helper_schedule, first_help['agent_x'].iloc[0], first_help['in_x'].iloc[0])
        # Evaluamos si la actividad previa sera o no afectada (si ningún agente que requiere asistencia en esta primera accion es tipo fixed
        # el helper puede ayudar a los dependents cuando sea necesario, es decir, que los dependents pueden esperar a que el helper acabe su
        # actividad previa).
        if not first_help['fixed_y'].any() and prev_action['todo'] != 'Entertainment':
            # Si el previo es idx, el actual será idx+1
            idx += 1
        # Retiramos la accion anterior, porque igaul tiene que salir antes de lo que estubiese haciendo (se observará en la siguiente uncion, no en esta)
        unaffected_actions = helper_schedule.loc[:idx-1].copy()
        # Lo sumamos al total
        all_unaffected_actions = pd.concat([all_unaffected_actions, unaffected_actions], ignore_index=True).reset_index(drop=True)
        # Sacamos las lineas afectadas por las asistencias (tecnicamente, la .iloc[0] aun no sabemos si esta o no afectada)
        affected_actions = helper_schedule.loc[idx:].copy()   
        # Lo sumamos al total
        all_affected_actions = pd.concat([all_affected_actions, affected_actions], ignore_index=True).reset_index(drop=True)
        
    # Devolvemos los datos que no seran y los que si serán afectados por las actividades de asistencia
    return all_unaffected_actions, all_affected_actions
    

    





def create_responsability_matrix(dependents, independents, pop_building, family_level_1_schedule):
    # Inicializamos el df de mejores opciones porposible helper
    best_helpers_activity = pd.DataFrame()
    # Producto cartesiano (todas las combinaciones posibles)
    cartesian = independents.merge(dependents, how='cross')
    for idx_car, row_car in cartesian.iterrows():
        # Distances calculation
        cartesian.loc[idx_car, 'time_dist'] = time_dist_calculate(row_car)
        cartesian.loc[idx_car, 'geo_dist']  = geo_dist_calculate(row_car, pop_building, family_level_1_schedule)
        cartesian.loc[idx_car, 'soc_dist']  = soc_dist_calculate(row_car)
        # Score calculation
        cartesian.loc[idx_car, 'score']  = score_calculate(cartesian.loc[idx_car])
    # Agrupamos por agente dependiente y actividad ('todo_y' y 'osm_id_y' al mismo tiempo para evitar problemas)
    cartesian_filtered = cartesian.groupby(['agent_y', 'todo_y', 'osm_id_y'])
    
    # Inicializamos el condicionante para reiterar el modelo
    repeat = True
    # Mientras no se detecte el modelo bien constituido
    while repeat:
        # Evitamos volver en caso de que el sistema no detecte cambios en 'repeat'
        repeat = False
        # Con esto sacamos el valor minimo para cada accion (best score)
        cartesian_filtered = cartesian.loc[cartesian.groupby(['agent_y', 'todo_y', 'osm_id_y'])['score'].idxmin()].sort_values(by=['in_y', 'in_x']).reset_index(drop=True)
        
        ## Verificamos que no haya una misma accion de un agente afectada (su lo hubiera el agente se quedaria esperando desde la entrega asta la siguiente recogida, sin hacer ninguna accion)
        # Agrupamos por actividad de cada independent
        best_helpers_activity = cartesian_filtered.groupby(['agent_x', 'todo_x', 'osm_id_x'])
        # Miramos grupo a grupo
        for cart_n, cart in best_helpers_activity:
            # Si un independent ayuda dos veces a un mismo dependent, significa que se da el caso de la espera mazo larga
            if cart['agent_y'].duplicated().any():
                print(F"\nSe ha detectado que una misma actividad tenia más de una actividad para el mismo agente y se ha cambiado")
                # Identificar los índices de los valores mínimos por cada grupo duplicado de 'agent_y'
                idx_to_drop = cart.loc[cart.duplicated('agent_y', keep=False)].groupby('agent_y')['in_y'].idxmin()
                # Eliminar esos índices del DataFrame
                cart = cart.drop(index=idx_to_drop)
                # Le pedimos que repita el ciclo
                repeat = True
                # Eliminamos el caso del 'cartesian' para no volver a sufrir el mismo problema
                cartesian = pd.concat([cartesian, cart]).drop_duplicates(keep=False).reset_index(drop=True)
                break
    # Devolvemos la matrix
    return cartesian_filtered

def time_dist_calculate(row_car):
    return abs(row_car['in_x'] - row_car['in_y'])

def geo_dist_calculate(row_car, pop_building, family_level_1_schedule):
    ## Trip 0
    # Previous osm_id data gathering
    prev_row_h, prev_idxh = get_previous_action(family_level_1_schedule, row_car['agent_x'], row_car['in_x'])
    prev_row_d, prev_idxd = get_previous_action(family_level_1_schedule, row_car['agent_y'], row_car['in_y'])
    
    # Data gathering
    lat_h, lon_h = pop_building.loc[pop_building['osm_id'] == prev_row_h['osm_id'], ['lat', 'lon']].values[0]
    lat_d, lon_d = pop_building.loc[pop_building['osm_id'] == prev_row_d['osm_id'], ['lat', 'lon']].values[0]
    # Trip distance calculation
    trip_0 = haversine(lat_h, lon_h, lat_d, lon_d)
    
    ## Trip 1
    # Data gathering
    lat_h, lon_h = pop_building.loc[pop_building['osm_id'] == row_car['osm_id_x'], ['lat', 'lon']].values[0]
    lat_d, lon_d = pop_building.loc[pop_building['osm_id'] == row_car['osm_id_y'], ['lat', 'lon']].values[0]
    # Trip distance calculation
    trip_1 = haversine(lat_h, lon_h, lat_d, lon_d)
    
    return trip_0+trip_1

def soc_dist_calculate(row_car):
    return 1

def score_calculate(row_car):
    return (row_car['time_dist']/1000)*10 + row_car['geo_dist'] + row_car['soc_dist'] #le ponemos 10 como heuristica

def get_previous_action(family_level_1_schedule, data_agent, data_in):
    """
    Devuelve la acción previa más reciente realizada por un agente
    antes de un tiempo específico (`data_in`).

    Parámetros:
        family_level_1_schedule (pd.DataFrame): tabla de acciones.
        data_agent (str o int): ID o nombre del agente.
        data_in (int o float): tiempo actual de análisis.

    Retorna:
        previous_action (pd.DataFrame): la última acción previa.
        index (int): índice de esa acción en el DataFrame original.
    """
    # Filtramos las acciones del agente
    agent_actions = family_level_1_schedule[family_level_1_schedule['agent'] == data_agent]
    # Filtramos acciones previas a 'data_in'
    previous_actions = agent_actions[agent_actions['in'] < data_in]
    # Si no hay acciones previas, devolvemos None
    if previous_actions.empty:
        return None, None
    # Tomamos la acción previa más reciente (con mayor 'in')
    idx = previous_actions['in'].idxmax()
    previous_action = previous_actions.loc[[idx]]
    return previous_action.iloc[0], idx


# Función de distancia haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c   

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
    
    pop_building = pd.read_excel(f"{paths['population']}/pop_building.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    level_1_results, level_2_results = todolist_family_creation(df_citizens, pop_building, system_management)
    
    level_1_results.to_excel(f"{paths['results']}/{study_area}_level_1.xlsx", index=False)
    level_2_results.to_excel(f"{paths['results']}/{study_area}_level_2.xlsx", index=False)

# Ejecución
if __name__ == '__main__':
    main_td()
    
    