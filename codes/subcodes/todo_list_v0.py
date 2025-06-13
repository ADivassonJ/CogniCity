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

def todolist_family_initialization(SG_relationship, family_df, activities):
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
                if activity == activities[0]:
                    in_time = opening
                    out_time = (in_time + time2spend) if time2spend != 0 else closing
                elif activity == activities[-1]:
                    filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
                    in_time = 0
                    row_out = filtered[filtered['in'] == min(filtered['in'])]
                    out_time = row_out['in'].iloc[0] - row_out['conmu_time'].iloc[0]
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
                        'todo_type': row_f_df[f'{activity}_type'] if not activity in home_travels else row_f_df['WoS_type'], 
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


def todolist_family_creation(df_citizens, SG_relationship):
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')
    results = pd.DataFrame(columns=['agent', 'route'])
    df_citizens_families = df_citizens.groupby('family')
    # Recorrer cada familia
    for family_name, family_df in df_citizens_families:
        
        # Creamos una lista de tareas con sus recorridos para cada agente de forma independiente
        todolist_family = todolist_family_initialization(SG_relationship, family_df)  
        
        results = pd.concat([results, todolist_family], ignore_index=True)
        
        '''
        # Evaluamos todolist_family para observar si existen agentes con dependencias
        while max(todolist_family['todo_type']) > 0:
            responsability_matrix = responsability_matrix_creation(todolist_family, SG_relationship_unique)            
            
            todolist_family = todolist_family_adaptation(responsability_matrix, todolist_family, SG_relationship_unique)
            print('todolist_family')
            print(todolist_family)
            input('#' * 80)
        # plot_agents_in_split_map(todolist_family, SG_relationship, save_path="recorridos_todos.html")
        '''
    results.to_excel(r'C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\todo_results.xlsx', index=False)

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
    
    