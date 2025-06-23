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
                        'fixed': fixed, 
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

def now_and_next(agent_todo, idx_a_m):
    # Especificamos las actuaciones sobre las que no se realizan calculos
    valores_excluir = ['Collect', 'Waiting collection', 'Waiting opening', 'Delivery']
    # En caso de que la actividad de analisis sea del cierre del dia o excluida
    if (agent_todo['out'][idx_a_m] == float('inf')) or (agent_todo['todo'][idx_a_m] in valores_excluir):
        return None, None
    # Sacamos el valor actual del osm_id
    current_id = agent_todo['osm_id'][idx_a_m]
    # Inicializamos el sumatorio para el indice
    i = 1
    # Mientras la fila que queremos analizar este entre alguna de las actividades excluidas que sume al sumatorio
    while (agent_todo['todo'][idx_a_m + i] in valores_excluir):
        i += 1
        if i >= len(agent_todo):
            print("agent_todo")
            print(agent_todo)
            print("i")
            input(i)
    # Obtenemos el id del siguiente paso
    next_id = agent_todo['osm_id'][idx_a_m + i]
    return current_id, next_id
    
def time_adding(df_after_max, last_out, responsability_matrix_history):
    # Inicializamos el df de resultados
    df_after_max_adapted = pd.DataFrame()
    # Actuamos sobre cada linea del df de acciones posteriores a las modificadas
    for idx_a_m, df_a_row in df_after_max.iterrows():
        # Se calcula la nueva hora de entrada
        new_in = last_out + int(df_a_row['conmu_time'])
        # Se calcula la nueva hora de salida
        new_out = (new_in + df_a_row['time2spend']) if df_a_row['time2spend'] != 0 else df_a_row['closing']
        if df_a_row['todo'] in ['Delivery', 'Collect']:
            new_out = new_in
        # En caso de detectarse alguna incompativilidad horaria, el agente no realiza la acción
        if df_a_row['closing'] < new_in or df_a_row['closing'] < new_out:
            print(f"After adaptation, {df_a_row['agent']} was not able to fullfill '{df_a_row['todo']}' at {new_in}.")
            print(f"new_in: {new_in} and new_out: {new_out}")
            print(f"new_in = {last_out} + {int(df_a_row['conmu_time'])}")
            continue
        # Se crea la nueva fila
        rew_row ={
            'agent': df_a_row['agent'],
            'todo': df_a_row['todo'], 
            'osm_id': df_a_row['osm_id'], 
            'todo_type': df_a_row['todo_type'], 
            'opening': df_a_row['opening'], 
            'closing': df_a_row['closing'], # Issue 16
            'fixed': df_a_row['fixed'], 
            'time2spend': df_a_row['time2spend'], 
            'in': new_in, 
            'out': new_out,
            'conmu_time': int(df_a_row['conmu_time'])
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

def manage_agent_dependences(agent, todolist_family, responsability_matrix_history):
    # Buscamos en el historial si el agente ha tenido alguna modificación
    mask_h = (responsability_matrix_history['dependent'] == agent)
    mask_d = (responsability_matrix_history['helper'] == agent)
    filtrado = responsability_matrix_history[mask_h | mask_d]
    # Sacamos el primero (el que afectará en cascada al resto de acciones)
    fila_menor_in = filtrado.loc[filtrado['in_h'].idxmin()]
    # Sacamos la iteración en la que ha ocurrido
    iteration = fila_menor_in['iter']
    
    # Si el agente ha sido modificado en iteraciones anteriores
    if (agent in responsability_matrix_history['dependent']) or (agent in responsability_matrix_history['helper']):
        # Obtenemos los datos del agente
        agent_todo = todolist_family[todolist_family['agent'] == agent].reset_index()
        # Vamos fila por fila dentro de su schedule
        for idx_a_t, row_a_t in agent_todo.iterrows():
            # Sacamos su osm_id actual y el del next step
            now,next = now_and_next(agent_todo, idx_a_t)
            if not [now,next] == [None, None]:
                # Compara si el par [now, next] (en cualquier orden) coincide con las columnas h o d
                mask_h = ((responsability_matrix_history['osm_id_h0'] == now) & (responsability_matrix_history['osm_id_h1'] == next))
                mask_d = ((responsability_matrix_history['osm_id_d0'] == now) & (responsability_matrix_history['osm_id_d1'] == next))
                filtrado = responsability_matrix_history[mask_h | mask_d]
                print('row_a_t')
                print(row_a_t)
                print('now,next')
                print(now,next)
                print('filtrado')
                input(filtrado)
                # si una actividaqd no se puede realizar y pasa a la siguiente, la siguiente debe vorrarse de agent_todo, si no dará error
            else:
                input('None None')
                # Si es el último, por lo que no hay next o otra condicion, se adapta la entrada de la acción
    # Si el agente NO ha sido modificado en iteraciones anteriores
    else:
        # Obtenemos los datos del agente
        agent_todo = todolist_family[todolist_family['agent'] == agent].reset_index()
    return agent_todo

def new_todolist_family_adaptation(todolist_family, new_todolist_family, responsability_matrix_history): 
    # Inicializamos el df de resultados
    new_list = pd.DataFrame()
    # Pasamos por todos los agentes del schedule
    for agent in todolist_family['agent'].unique():
        # Inicializamos el df temporal resultados especificos para cada agente
        new_new_list = pd.DataFrame()
        # Si el agente no ha sido modificado en esta iteración
        if not agent in new_todolist_family['agent'].to_list():
            new_new_list = manage_agent_dependences(agent, todolist_family, responsability_matrix_history)
            new_list = pd.concat([new_list, new_new_list], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
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
            df_after_max = time_adding(df_after_max, max(new_new_list['out']), responsability_matrix_history)
        # Y despues se suma
        new_new_list = pd.concat([new_new_list, df_after_max], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
        new_list = pd.concat([new_list, new_new_list], ignore_index=True).sort_values(by='in', ascending=True).reset_index(drop=True)
    return new_list

def todolist_family_adaptation(responsability_matrix, todolist_family, responsability_matrix_history):
    # Calculamos los apartados sobre los que actuar dentro de 'todolist_family'
    matrix2cover, prev_matrix2cover = matrix2cover_creation(todolist_family, responsability_matrix)
    # Tomamos los apartados sobre los que actuar y los adaptamos a las necesidades de los dependants
    new_todolist_family = route_creation(matrix2cover, prev_matrix2cover)
    # Adaptar el resto del schedule a las modificaciones realizadas
    todolist_family = new_todolist_family_adaptation(todolist_family, new_todolist_family, responsability_matrix_history)
    return todolist_family

def route_creation(matrix2cover, prev_matrix2cover): 
    ## Dividir los datos
    # DataFrame para almacenar resultados si vas a ir agregando algo más adelante
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmu_time']
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
    return combined_df

def agent_collection(prev_matrix2cover, matrix2cover, helper):
    # Inicializamos el df de los resultados
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmu_time']
    new_new_list = pd.DataFrame(columns=columns)
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
            filtered = matrix2cover[matrix2cover['fixed'] == True]
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
                'fixed': False, 
                'time2spend': 0, 
                'in': group_out_time, 
                'out': group_out_time,
                'conmu_time': group_conmu_time
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
                    'todo': f'Waiting collection', 
                    'osm_id': agent['osm_id'],  # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related 
                    'fixed': agent['fixed'], 
                    'time2spend': waiting_time, 
                    'in': agent['out'], 
                    'out': group_out_time,
                    'conmu_time': agent['conmu_time']
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                new_out = agent['out']
            else: 
                new_out = group_out_time
            # Actualización del caso original del agente
            rew_row ={
                'agent': agent['agent'],
                'todo': agent['todo'], 
                'osm_id': agent['osm_id'], 
                'todo_type': agent['todo_type'], 
                'opening': agent['opening'], 
                'closing': agent['closing'], 
                'fixed': agent['fixed'], 
                'time2spend': agent['time2spend'], 
                'in': agent['in'], 
                'out': new_out,
                'conmu_time': agent['conmu_time']
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    return new_new_list

def agent_delivery(prev_matrix2cover, matrix2cover, helper, agent_collection):
    # Inicializamos el df de los resultados
    columns=['agent','todo','osm_id','todo_type','opening','closing','fixed','time2spend','in','out','conmu_time']
    new_new_list = pd.DataFrame(columns=columns)
    ## Creación de ruta de recogida
    # DataFrame con datos de ins
    in_osm_ids = pd.DataFrame(columns=['osm_id', 'in', 'conmu_time'])
    # Agrupamos para crear ruta de entrega
    osm_id_groups = matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor minimo de in en el grupo que tenga fixed == True (quién condiciona)
        filtered = oi_group[oi_group['fixed'] == True]
        # Asignamos tiempo de conmutación del grupo
        group_conmu_time = oi_group['conmu_time'].max()
        ## Asignamos tiempo de llegada del grupo
        # En caso de NO HABER ningún agente condicionante
        if filtered.empty: # No tiene más condiciones, porque si es fixed tendra un time2spend seguro, no hace falta comprobar
            max_out = agent_collection['out'].max()
            agents_row = agent_collection[agent_collection['out'] == max_out]
            max_in = agents_row['in'].max()           
            agents_row = agents_row[agents_row['in'] == max_in]
            group_in_time = agents_row['out'].iloc[0] + agents_row['conmu_time'].iloc[0]
        else:
            # Tomamos como hora de entrada la del agente condicionante con hora más tenprana
            # Este será el último en ser entregado, asi te aseguras de que todos llegan antes de la hora, ninguno tarde.
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
    # Iteramos todos los osm_id de llegada
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
                'fixed': False, 
                'time2spend': 0, 
                'in': group_in_time, 
                'out': group_in_time,
                'conmu_time': group_conmu_time
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
                    'todo': f'Waiting opening', 
                    'osm_id': agent['osm_id'], # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related
                    'fixed': agent['fixed'], 
                    'time2spend': waiting_time, 
                    'in': group_in_time, 
                    'out': agent['in'],
                    'conmu_time': agent['conmu_time']
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
                'todo': agent['todo'], 
                'osm_id': agent['osm_id'], 
                'todo_type': 0, 
                'opening': agent['opening'], 
                'closing': agent['closing'], 
                'fixed': agent['fixed'], 
                'time2spend': agent['time2spend'], 
                'in': in_time, 
                'out': new_out,
                'conmu_time': agent['conmu_time']
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)     
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
  

def todolist_family_creation(df_citizens, SG_relationship):
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')
    results = pd.DataFrame()
    df_citizens_families = df_citizens.groupby('family')
    # Recorrer cada familia
    for family_name, family_df in df_citizens_families:
        # Inicializamos el dataframe para guardar el historial de la matriz de responsabilidades. Esta es por familia, por lo que se reinicia en cada bucle.
        responsability_matrix_history = pd.DataFrame()
        # Creamos una lista de tareas con sus recorridos para cada agente de forma independiente
        todolist_family = todolist_family_initialization(SG_relationship, family_df, [])  # Aqui el '[]' se debe meter los servicios a analizar (pueden ser 'WoS', 'Dutties' y/o 'Entertainment')
                                                                                          # Deberiamos leerlo del doc de system management
        todolist_family_original = todolist_family
        print('todolist_family:')
        print(todolist_family)
        input('#' * 80)
        iter = 0
        # Evaluamos todolist_family para observar si existen agentes con dependencias
        while max(todolist_family['todo_type']) > 0:
            # En caso de existir dependencias, se asignan responsables
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique, todolist_family_original, responsability_matrix_history, iter)
            responsability_matrix_history = pd.concat([responsability_matrix_history, responsability_matrix], ignore_index=True)
            print('responsability_matrix_history')
            print(responsability_matrix_history)
            # Tras esto, la matriz todo de esta familia es adaptada a las responsabilidades asignadas
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, responsability_matrix_history)
            print('todolist_family')
            print(todolist_family)
            input('#' * 80)
            iter += 1
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

def responsability_matrix_creation(todolist_family, SG_relationship_unique, todolist_family_original, responsability_matrix_history, iter):   
    # Creamos la matriz de los resultados
    responsability_matrix = pd.DataFrame()
    # DataFrame con todo_type > 0 (dependientes con trips que requieren asistencia)
    dependents = todolist_family[todolist_family["todo_type"] > 0].add_suffix('_d')
    # DataFrame con todo_type == 0 (independientes)
    helpers = todolist_family[todolist_family["todo_type"] == 0].add_suffix('_h')
    # Eliminamos los independientes pero no capaces de ayudar (aquellos que en WoS son dependientes)
    agents_with_wos = todolist_family_original[(todolist_family_original['todo'] == 'WoS') & (todolist_family_original['todo_type'] != 0)]['agent'].unique()
    helpers = helpers[~helpers['agent_h'].isin(agents_with_wos)].reset_index(drop=True)
    # Producto cartesiano (todas las combinaciones posibles)
    df_combinado = helpers.merge(dependents, how='cross')
    # Eliminamos los valores de acciones por actividades previas
    valores_excluir = ['Collect', 'Waiting collection', 'Waiting opening', 'Delivery']  # Esto puede ser problematico
    df_combinado = df_combinado[~df_combinado['todo_h'].isin(valores_excluir) & ~df_combinado['todo_d'].isin(valores_excluir)]
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
            'in_d': row_df_conb['in_d'], 
            'iter': iter
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
        input(f'action has no responsables available.')
    
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
    
    