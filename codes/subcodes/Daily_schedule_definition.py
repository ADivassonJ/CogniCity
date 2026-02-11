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
       
    if final_doc_done:
        print(f"All weekdays modelated: Generating {study_area}_{file_type}.xlsx ...")
        for file in paths['results'].glob(f'{study_area}_*_{file_type}.xlsx'):
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

def check_current_data(study_area, days, paths):
    
    files_names = ['schedule_citizen', 'schedule_vehicle', 'todolist']

    files_done = {}

    for file in files_names:
        files_done[file] = (Path(paths['results']) / f"{study_area}_{file}.xlsx").exists()
    
    if all(files_done.values()):
        return files_done, {}, True
    
    todolist_done_days = {}
    
    for day in days:
        todolist_done_days[day] = (Path(paths['results']) / f"{study_area}_{day}_todolist.xlsx").exists()

    if all(todolist_done_days.values()):
        files_done['todolist'] = True
    
    schedule_citizen_done_days = {}
    schedule_vehicle_done_days = {}
    
    for day in days:
        schedule_citizen_done_days[day] = (Path(paths['results']) / f"{study_area}_{day}_schedule_citizen.xlsx").exists()
        schedule_vehicle_done_days[day] = (Path(paths['results']) / f"{study_area}_{day}_schedule_vehicle.xlsx").exists()

    # Combinar diccionarios: False tiene prioridad sobre True
    combined_done_days = {day: schedule_citizen_done_days[day] and schedule_vehicle_done_days[day] 
                        for day in days}

    # Ahora, si todos son True, marcamos los archivos como completos
    if all(combined_done_days.values()):
        files_done['schedule_citizen'] = True
        files_done['schedule_vehicle'] = True
        return files_done, {}, False


    # Lista de días que faltan (es decir, que son False)
    days_missing = [day for day, done in combined_done_days.items() if not done]
    
    return files_done, days_missing, False

def Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations, WP3_active):
    
    print('#'*20, ' System running ','#'*20)

    days = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}

    days = {'Mo'}
    
    files_done, days_missing, already_done = check_current_data(study_area, days, paths)

    if not already_done:
        #################### TODOLIST ################
        if not files_done['todolist']:
            for day in days_missing:
                todolist_family_creation(study_area, agent_populations['citizen'], agent_populations['building'], system_management, paths, day, pop_archetypes['citizen'], pop_archetypes['building'])
            days_missing = days
        #################### SCHEDULES ################
        # In case of having days to model
        for day in days_missing:
            # Input reading
            todolist = pd.read_excel(f"{paths['results']}/{study_area}_{day}_todolist.xlsx")
            # Vehicle Choice Modeling
            vehicle_choice_model(todolist, agent_populations, paths, study_area, pop_archetypes, networks_map, day, WP3_active)
            
        files_done, days_missing, already_done = check_current_data(study_area, days, paths)

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
        return False
    else:
        print(f"    [NOTE] Data related to {study_area} was already created. Please, if you want to generate new data, delete the current one from:")
        print(f"    {paths['results']}")
        print('#'*20, ' Simulation Completed ','#'*20)
        return True


        
        
        
        
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

    WP3_active = False
    
    Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations, WP3_active)