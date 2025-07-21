import os
import sys
import random
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def family_level_1_schedule(pop_building, family_df, activities):
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
    level_1_results = pd.DataFrame()
    level_2_results = pd.DataFrame()
    # Agrupamos la poblacion de ciudadanos en familias
    df_citizens_families = df_citizens.groupby('family')
    # Recorremos cada familia
    for family_name, family_df in tqdm(df_citizens_families, desc="Procesando familias"):
        # Sacamos la lista de actividades del archivo system_management
        activities = system_management['activities'].tolist()
        # Filtrar elementos que no sean NaN
        activities = [a for a in activities if pd.notna(a)]  
        # Creamos una lista de tareas con sus recorridos para cada agente de forma independiente
        todolist_family = family_level_1_schedule(pop_building, family_df, activities)
        # Actualizamos el 'todolist_family_original' para guardarlo (en proximas actividades 'todolist_family' se actualizará).
        todolist_family_original = todolist_family
        # Sumamos los datos a la lista de resultados de level 1
        level_1_results = pd.concat([level_1_results, todolist_family], ignore_index=True).reset_index(drop=True)
        # Inicializamos historial de actuaciones no compatibles
        not_feasible = pd.DataFrame(columns=['agent_h', 'agent_d', 'todo_d'])
        # Evaluamos todolist_family para observar si existen agentes con dependencias
        while max(todolist_family['todo_type']) > 0:
            # En caso de existir dependencias, se asignan responsables
            responsability_matrix, not_feasible = responsability_matrix_creation(todolist_family, pop_building_unique, todolist_family_original, not_feasible)
            if responsability_matrix.empty:
                print(f"Se ha ejecutado el 'truquito'.")
                print(f"Basicamente, no hay agentes helper disponibles para asistir en estos escenarios.")
                print(f"No deberia haber pasado esto ...")
                # Encuentra índices donde 'todo_type' es distinto de 0
                mask = todolist_family['todo_type'] != 0
                # Si hay alguno, reemplaza el primero por 0
                if mask.any():
                    first_index = todolist_family[mask].index[0]
                    todolist_family.at[first_index, 'todo_type'] = 0
                continue
            # Tras esto, la matriz todo de esta familia es adaptada a las responsabilidades asignadas
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family)
        # Sumamos los datos a la lista de resultados de level 1
        level_2_results = pd.concat([level_2_results, todolist_family], ignore_index=True).reset_index(drop=True)
    # Devolvemos los resultados
    return level_1_results, level_2_results

# Función de distancia haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c   

def resp_score_calculation(todolist_family, pop_building_unique, df_combinado, dependent):
    # Creamos la matriz de los resultados
    responsability_matrix = pd.DataFrame()
    # Calculamos todas las convinaciones
    for idx_df_conb, row_df_conb in df_combinado.iterrows():
        # Si la entrada es 0, sera la actividad de Home_out, por lo que lo ignoramos, no se plantean actividades previas a esta
        if row_df_conb['in_h'] == 0:
            continue
        # Sacamos las latitudes y longitudes de las posiciones de helper y dependant
        lat_h, lon_h = pop_building_unique.loc[pop_building_unique['osm_id'] == row_df_conb['osm_id_h'], ['lat', 'lon']].values[0]
        lat_d, lon_d = pop_building_unique.loc[pop_building_unique['osm_id'] == row_df_conb['osm_id_d'], ['lat', 'lon']].values[0]
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
            'todo_h0':h_pre_step['todo'].iloc[0],
            'todo_h1':row_df_conb['todo_h'],
            'todo_d0': d_pre_step['todo'].iloc[0],
            'todo_d1': row_df_conb['todo_d'],
            'osm_id_h0': h_pre_step['osm_id'].iloc[0],
            'osm_id_d0': d_pre_step['osm_id'].iloc[0],
            'osm_id_h1': row_df_conb['osm_id_h'],            
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
    responsability_matrix = responsability_matrix.loc[responsability_matrix.groupby(['dependent'])['score'].idxmin()].reset_index(drop=True)
    
    ## Nos aseguramos que las acciona a acometer se realizan por el mismo agente
    # Logramos el helper del dependent principal
    helper = responsability_matrix[responsability_matrix['dependent'] == dependent]['helper'].iloc[0]
    # Quitamos las acciones no realizadas por este helper
    responsability_matrix = responsability_matrix[responsability_matrix['helper'] == helper]
    
    return responsability_matrix

def responsability_matrix_creation(todolist_family, pop_building_unique, todolist_family_original, not_feasible):
    # Retiramos los agentes 
    correction = 0
    df_combinado = pd.DataFrame()
    
    while df_combinado.empty:
        # Actualizamos el multiplicador
        correction += 1        
        # Encuentra los agentes con alguna dependencia
        dependents = todolist_family[todolist_family['todo_type'] != 0].add_suffix('_d')
        # Saca la primera
        first_not0 = dependents.iloc[0]
        # Sacamos los valores relevantes
        dependent = first_not0['agent_d']
        in_d = first_not0['in_d']
        
        # Encuentra los agentes con independientes
        helpers = todolist_family[(todolist_family["todo_type"] == 0) & (~todolist_family['agent'].isin(dependents['agent_d']))].add_suffix('_h')
        # Eliminamos los valores de acciones por actividades previas
        valores_excluir = ['Collect', 'Waiting collection', 'Waiting opening', 'Delivery']  # Esto puede ser problematico
        helpers = helpers[~helpers['todo_h'].isin(valores_excluir)]

        
        test = pd.DataFrame()
        for helper in helpers['agent_h'].unique():
            new_row = {
                'agent_h': helper,
                'agent_d': first_not0['agent_d'],
                'todo_d': first_not0['todo_d']
            }
            test = pd.concat([test, pd.DataFrame([new_row])], ignore_index=True)
            
        # Elimina filas de test que ya están en not_feasible_renamed
        test_filtrado = test.merge(
            not_feasible[['agent_h', 'agent_d', 'todo_d']],
            on=['agent_h', 'agent_d', 'todo_d'],
            how='left',
            indicator=True
        ).query('_merge == "left_only"').drop(columns=['_merge'])
        
        if test_filtrado.empty:
            print(f"El agente '{first_not0['agent_d']} no podrá realizar nunca '{first_not0['todo_d']}'.")
            return pd.DataFrame(), not_feasible
        
        # Ahora miramos los siguientes por ver si existe la opcion de compatibilizar el recorrido
        rest_not0 = todolist_family[todolist_family['todo_type'] != 0].iloc[1:].reset_index(drop=True).add_suffix('_d')
        # Retiramos del grupo el agente ya analizado
        rest_not0 = rest_not0[rest_not0['agent_d'] != dependent]

        rest_not0['in_d'] = pd.to_numeric(rest_not0['in_d'], errors='coerce').astype('Int64')
        
        # Nos quedamos con el primero de cada nuevo dependant    
        rest_not0 = rest_not0.loc[rest_not0.groupby(['agent_d'])['in_d'].idxmin()].reset_index(drop=True)

        # Retiramos los casos que superen la hora de diferencia entre la entrada de uno y del otro
        deleted_rest_not0 = rest_not0[(rest_not0['in_d'] - in_d) > 30*correction]
        rest_not0 = rest_not0[(rest_not0['in_d'] - in_d) <= 30*correction] # Issue 22
        # Repetir la fila para que tenga el mismo número de filas que helper
        feasible_not0 = pd.concat([pd.DataFrame([first_not0]), rest_not0], ignore_index=True)
        # Producto cartesiano (todas las combinaciones posibles)
        df_combinado = helpers.merge(feasible_not0, how='cross')
        
        # Filtramos en base a las rutas previamente provadas y no funcionales
        df_combinado = df_combinado.merge(not_feasible, on=['agent_h', 'agent_d', 'todo_d'], how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

        if df_combinado.empty:
            print(f"SE HA REALIZADO UN NUEVO LOOP POR PROBLEMAS EN LA ASIGNACION (correction: {correction})")
    
    # Calculamos todas las convinaciones
    responsability_matrix = resp_score_calculation(todolist_family, pop_building_unique, df_combinado, dependent)
    
    for depend in deleted_rest_not0['agent_d'].unique():
        if depend in responsability_matrix['dependent'].unique():
            continue
        new_row = {
            'agent_h': responsability_matrix['helper'].iloc[0],
            'agent_d': depend,
            'todo_d': deleted_rest_not0[deleted_rest_not0['agent_d'] == depend]['todo_d'].iloc[0]
        }
        
        not_feasible = pd.concat([not_feasible, pd.DataFrame([new_row])], ignore_index=True)
    
    return responsability_matrix, not_feasible

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
    
    