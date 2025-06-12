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


def todolist_family_initialization(SG_relationship, family_df):
    # Inicializamos el df en el que meteremos los schedules
    todolist_family = pd.DataFrame(columns=['agent', 'todo', 'osm_id', 'todo_type', 'opening', 'closing', 'fixed?', 'time2spend', 'in', 'out', 'conmu_time'])
    # Pasamos por cada agente que constitulle la familia 
    for idx_f_df, row_f_df in family_df.iterrows():
        ### WoS
        osm_id = row_f_df['WoS']
        fixed = row_f_df['WoS_action_type'] != 1
        # Aqui seleccionamos si los agentes se fijan en el horario de servicio (usuario) o de labor (trabajador)
        if fixed:
            # Si tienen algún tipo de dependencia, no van a ir a trabajar, si no a estudiar, por lo que usan el horario de servicio
            fixed_word = 'Service'
        else:
            # En caso de no depender es probable que vayan a trabajar, por lo que entrar en WoS
            fixed_word = 'WoS'	
        
        ## Sacamos los datos relevantes        
        opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
        closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
        time2spend = int(row_f_df['WoS_time'])
        in_h = opening
        # En caso de tener un time2spend, hace in + este tiempo, si no todo lo que pueda
        out_h = (in_h + time2spend) if time2spend != 0 else closing
        # Creamos la nueva fila en caso de que se pueda realizar la accion
        if in_h < closing and out_h <= closing:  
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
            # La añadimos
            todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True) 
        else:
            print(f"{row_f_df['name']} was not able to fullfill 'WoS' at {in_h}.")          
                
        ### Dutties          
        for _ in range(row_f_df['Dutties_amount']):
            # Elegimos, según el tipo de actividad (en este caso Dutties) que lista de edificios pueden ser validos
            # [Aqui habrá que meter una funcion de verdad, que valore en base a estadistica]
            dutties_ids_options = SG_relationship[SG_relationship['service_group'] == 'entertainment']['osm_id'].tolist()
            # Elegimos uno aleatorio del grupo de validos
            osm_id = random.choice(dutties_ids_options)
            ## En principio, suponemos los Dutties como NO FIXED
            # Aqui tendriamos que meter tema de inicio a horas especificas y tal (ej. un extraescolar no empieza a las 13:42)
            fixed = False
            # Como es una accion de tipo usuario, siempre es Service
            fixed_word = 'Service'	
            ## Sacamos los datos relevantes
            opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
            closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
            time2spend = int(row_f_df['Dutties_time']) # aqui habrá que meter más tema estadistico aun y ver por tipo y tal
            # Miramos la posición previa del agente
            filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
            if not filtered.empty:
                in_h = max(filtered['out']) + filtered['conmu_time'].iloc[0]
            else: # si resulta que hoy no trabajaba
                in_h = opening
            # En caso de tener un time2spend, hace in + este tiempo, si no todo lo que pueda
            out_h = (in_h + time2spend) if time2spend != 0 else closing
            # Creamos la nueva fila en caso de que se pueda realizar la accion
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
                # La añadimos    
                todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)
            else:
                print(f"{row_f_df['name']} was not able to fullfill 'Dutties' at {in_h}.")
                print(f"they were trying to go at {in_h} until {out_h} but '{osm_id}' it closes at {closing}")
            
        ### entertainment
        # Elegimos, según el tipo de actividad (en este caso Dutties) que lista de edificios pueden ser validos
        # [Aqui habrá que meter una funcion de verdad, que valore en base a estadistica]
        entertainment_ids_options = SG_relationship[SG_relationship['service_group'] == 'entertainment']['osm_id'].tolist()
        # Elegimos uno aleatorio del grupo de validos        
        osm_id = random.choice(entertainment_ids_options)
        ## En principio, suponemos los Dutties como NO FIXED
        # Aqui tendriamos que meter tema de inicio a horas especificas y tal (ej. un extraescolar no empieza a las 13:42)    
        fixed = False
        fixed_word = 'Service'	
        ## Sacamos los datos relevantes            
        opening = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_opening'].values[0]
        closing = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, f'{fixed_word}_closing'].values[0]
        time2spend = 0 # Al ser una actividad de ocio, no requiere invertir tiempo en ella, aunque lo desee      
        # Miramos la posición previa del agente
        filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
        if not filtered.empty:
            in_h = max(filtered['out']) + filtered['conmu_time'].iloc[0]
        else: # si resulta que hoy no trabajaba
            in_h = opening
        # En caso de tener un time2spend, hace in + este tiempo, si no todo lo que pueda
        out_h = (in_h + time2spend) if time2spend != 0 else closing
        # Creamos la nueva fila en caso de que se pueda realizar la accion
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
            # La añadimos
            todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True)
        else:
            print(f"{row_f_df['name']} was not able to fullfill 'Entertainment' at {in_h}.")
        
        ### home in
        ## Sacamos los datos relevantes            
        opening = 0
        closing = float('inf')
        # Miramos la posición previa del agente
        filtered = todolist_family[todolist_family['agent'] == row_f_df['name']]
        if not filtered.empty:
            in_h = max(filtered['out']) + filtered['conmu_time'].iloc[0]
        else: # si resulta que hoy no trabajaba
            in_h = opening
        # Creamos la nueva fila en caso de que se pueda realizar la accion
        if in_h < closing and out_h <= closing: 
            rew_row ={
                'agent': row_f_df['name'],
                'todo': 'Home_in', 
                'osm_id': row_f_df['home'], 
                'todo_type': row_f_df['WoS_type'], 
                'opening': 0, 
                'closing': float('inf'), 
                'fixed?': False, 
                'time2spend': 0, 
                'in': in_h, 
                'out': float('inf'),
                'conmu_time': int(row_f_df['conmu_time'])
            } 
            # La añadimos  
            todolist_family = pd.concat([todolist_family, pd.DataFrame([rew_row])], ignore_index=True) 
        else:
            print(f"{row_f_df['name']} was not able to fullfill 'Home_in' at {in_h}.")
    # En caso de haber solo un agente en la familia, le transformamos en autosufuciente    
    if todolist_family['agent'].nunique() == 1:
        todolist_family['todo_type'] = 0
    # Añadimos la salidad del hogar, del principio del día
    todolist_family = home_out_trips_adding(family_df, todolist_family)
    
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
        
        input(todolist_family)
        
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
    
    