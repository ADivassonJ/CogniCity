# === Instalación automática de dependencias ===================================
import importlib.util
import subprocess
import sys
import os
import sys
import pandas as pd
import geopandas as gpd
import random
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from shapely.geometry import Point
try:
    from todo_list import todolist_family_creation
    from vehicle_choice_model import vehicle_choice_model
except Exception as e:
    from subcodes.todo_list import todolist_family_creation
    from subcodes.vehicle_choice_model import vehicle_choice_model

def delete_used_files(file_type, paths):
    # We search for used files
    for file in paths['results'].glob(f'*_{file_type}.xlsx'):
        # We analyze docs names
        parts = file.stem.split('_')
        # If file is not f'{use_case}_{day}_{file_type}.xlsx' style, we ignore it
        if parts[-1].lower() == "todolist" and len(parts) != 3:
            continue
        elif parts[-2] == "schedule" and len(parts) != 4:
            continue
        # We delete used file
        file.unlink()
        
def docs_convining(final_doc_done, file_type, study_area, paths):
    
    final_doc = []
       
    if not final_doc_done:
        print(f"All weekdays modelated: Generating {study_area}_{file_type}.xlsx ...")
        for file in paths['results'].glob(f'*_{file_type}.xlsx'):
            #aqui los sumamos todos los acabados en _{file_type} (kind == '{file_type}') y creamos una nueva columna ('day') que especifique su valor de parts[-2]
            current = pd.read_excel(file)
            parts = file.stem.split('_')
            current = current.copy()
            if file_type == "todolist":
                current['day'] = parts[-2]
            else:
                current['day'] = parts[-3]
            final_doc.append(current)
        # Path to convined results
        final_file_type_path = os.path.join(paths['results'], f"{study_area}_{file_type}.xlsx")
        # Creamos df con los datos compilados
        df = pd.concat(final_doc, ignore_index=True)        
        df.to_excel(final_file_type_path, index=False)

def check_current_data(days, paths):
    
    found_files = {'schedule_citizen': set(),
                   'schedule_vehicle': set(),
                   'todolist': set()
                }
    
    files_done = {'schedule_citizen': False,
                  'schedule_vehicle': False,
                  'todolist': False
                }
    
    days_missing = {}
    
    for file in paths['results'].glob('*.xlsx'):
        name = file.stem  # sin extensión
        parts = name.split('_')
        if len(parts) == 2:
            kind = parts[-1].lower()
            files_done[kind] = True
            continue
        elif len(parts) == 4:
            # Formato: studyarea_day_kind
            day = parts[-3]
            kind = f"{parts[-2]}_{parts[-1].lower()}"
        elif len(parts) == 3:
            # Formato: studyarea_day_kind
            day = parts[-2]
            kind = parts[-1].lower()
        else:
            continue

        if day not in days:
            continue
        found_files[kind].add(day)

    # Faltantes por tipo
    missing_schedule = days - found_files['schedule_citizen']
    missing_vehicles = days - found_files['schedule_vehicle']

    days_missing['todolist'] = days - found_files['todolist']
    # Faltantes en general (en al menos uno)
    days_missing['schedule'] = missing_schedule | missing_vehicles
    
    return files_done, days_missing

def Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations):
    
    print('#'*20, ' System running ','#'*20)

    days = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}

    days = {'Mo'}
    
    files_done, days_missing = check_current_data(days, paths)
    
    #################### TODOLIST ################
    
    if days_missing['todolist'] and (not files_done['todolist']):
        for day in days_missing['todolist']:
            todolist_family_creation(study_area, agent_populations['citizen'], agent_populations['building'], system_management, paths, day, pop_archetypes['citizen'], pop_archetypes['building'])
    
    #################### SCHEDULES ################
    # In case of having days to model
    if days_missing['schedule'] and ((not files_done['schedule_citizen']) or (not files_done['schedule_vehicle'])):
        # We act on each different day
        for day in days_missing['schedule']:
            # Input reading
            todolist = pd.read_excel(f"{paths['results']}/{study_area}_{day}_todolist.xlsx")
            # Vehicle Choice Modeling
            vehicle_choice_model(todolist, agent_populations, paths, study_area, pop_archetypes, networks_map, day)
        
    files_done, days_missing = check_current_data(days, paths)

    print('#'*20, ' Simulation Completed ','#'*20)

    ## Todolist
    # Convining docs
    docs_convining(files_done['todolist'], 'todolist', study_area, paths)
    # Delete used files
    delete_used_files('todolist', paths)
    
    ## Vehicles
    # Convining docs
    docs_convining(files_done['schedule_vehicle'], 'schedule_vehicle', study_area, paths)
    # Delete used files
    delete_used_files('schedule_vehicle', paths)
    
    ## Schedule
    # Convining docs
    docs_convining(files_done['schedule_citizen'], 'schedule_citizen', study_area, paths)
    # Delete used files
    delete_used_files('schedule_citizen', paths)


        
        
        
        
if __name__ == '__main__':
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
    
    
    df_citizens = pd.read_parquet(f"{paths['population']}/pop_citizen.parquet")

    citizen_archetypes = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_citizen.xlsx")

    networks = ['drive', 'walk']
    networks_map = {}   
    for net_type in networks:           
        networks_map[net_type + "_map"] = ox.load_graphml(paths['maps'] / (net_type + '.graphml'))
    
    agent_populations = {}
    agent_populations['citizen'] = pd.read_parquet(f"{paths['population']}/pop_citizen.parquet")
    agent_populations['family'] = pd.read_parquet(f"{paths['population']}/pop_family.parquet")
    agent_populations['building'] = pd.read_parquet(f"{paths['population']}/pop_building.parquet")
    agent_populations['transport'] = pd.read_parquet(f"{paths['population']}/pop_transport.parquet")
    
    pop_archetypes = {}
    pop_archetypes['transport'] = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    pop_archetypes['citizen'] = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_citizen.xlsx")
    pop_archetypes['building'] = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_building.xlsx")
    
    ##############################################################################################################
    print(f'docs readed')
    
    Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations)