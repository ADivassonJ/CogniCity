# === Estándar de Python =======================================================
from __future__ import annotations

# === Instalación automática de dependencias ===================================
import importlib.util
import subprocess
import sys

modules = [
    "folium",
    "geopandas",
    "matplotlib",
    "numpy",
    "osmnx",
    "pandas",
    "pyproj",
    "haversine",
    "scipy",
    "shapely",
    "pyarrow",
    "fastparquet",
    "tqdm", 
    "scikit-learn",
    "openpyxl",
    "numpy",
]

def install_if_missing(package):
    """Instala automáticamente un paquete si no está disponible."""
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for mod in modules:
    install_if_missing(mod)


import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from subcodes.initialization import Archetype_documentation_initialization, Geodata_initialization, Synthetic_population_initialization
from subcodes.todo_list import todolist_family_creation
from subcodes.vehicle_choice_model import vehicle_choice_model
pd.set_option('mode.chained_assignment', 'raise')  # Convierte el warning en error

### Main
def main():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent
    paths['system'] = paths['main'] / 'system'
    paths['desktop'] = Path.home() / "Desktop"
    
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
            elif row['pre'] == 'p':
                user_is_stupid = True
                while user_is_stupid:    
                    response = input(f"Data for the case study '{study_area}' was not found.\nDo you want to copy data from standar scenario or do you want to create your own? [Y/N]\n")
                    if response == 'Y':
                        user_is_stupid = False
                        shutil.copytree(paths['base_scenario'], paths[file_2])
                    elif response == 'N':
                        user_is_stupid = False
                        os.makedirs(paths[file_2], exist_ok=True)
                    else:
                        print(f"Your response was not valid, please respond Y (yes) or N (no).")
            else:
                os.makedirs(paths[file_2], exist_ok=True)
    

    print('#'*20, ' System initialization ','#'*20)
    # Archetype documentation initialization
    pop_archetypes, stats = Archetype_documentation_initialization(paths)
    # Geodata initialization
    agent_populations, networks_map = Geodata_initialization(study_area, paths, pop_archetypes)
    # Synthetic population initialization
    agent_populations = Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats, paths, study_area)
    print('#'*20, ' Initialization finalized ','#'*20)

    # pop_error_printing(agent_populations['citizen'], agent_populations['family'], pop_archetypes['citizen'], pop_archetypes['family'])
    
    try:
        level_1_results = pd.read_excel(f"{paths['results']}/{study_area}_todolist.xlsx")
        print(f"{study_area}_todolist.xlsx loaded")
    except Exception as e:
        print(f"Creating {study_area}_todolist.xlsx")
        level_1_results = todolist_family_creation(study_area, agent_populations['citizen'], agent_populations['building'], system_management, paths)
    
    try:
        vehicles_actions = pd.read_excel(f"{paths['results']}/{study_area}_vehicles_actions.xlsx")
        new_level1_schedules = pd.read_excel(f"{paths['results']}/{study_area}_new_level_1.xlsx")
        print(f"{study_area}_vehicles_actions.xlsx and {study_area}_new_level_1.xlsx loaded")
    except Exception as e:
        print(f"Creating {study_area}_vehicles_actions.xlsx and {study_area}_new_level_1.xlsx")
        vehicles_actions, new_level1_schedules = vehicle_choice_model(level_1_results, agent_populations['transport'], agent_populations['citizen'], paths, study_area, pop_archetypes['transport'], agent_populations['building'], networks_map)
    
def pop_error_printing(df_citizens, df_families, citizen_archetypes, family_archetypes):
    # Suponiendo que df_citizens y df_families ya están definidos
    df_final_stats_citizens = df_citizens['archetype'].value_counts().reset_index()
    df_final_stats_citizens.columns = ['archetype', 'count']

    # Para df_families
    df_final_stats_families = df_families['archetype'].value_counts().reset_index()
    df_final_stats_families.columns = ['archetype', 'count']

    # Usar directamente los valores de citizen_archetypes y family_archetypes
    df_final_stats_citizen_archetypes = citizen_archetypes[['name', 'presence']].copy()
    df_final_stats_citizen_archetypes.columns = ['name', 'count']

    df_final_stats_family_archetypes = family_archetypes[['name', 'presence']].copy()
    df_final_stats_family_archetypes.columns = ['name', 'count']

    # Combinar df_final_stats_families con df_final_stats_family_archetypes
    merged_families = df_final_stats_families.merge(df_final_stats_family_archetypes, left_on='archetype', right_on='name', how='outer', suffixes=('_families', '_family_archetypes')).drop(columns=['name'])
    merged_families.fillna(0, inplace=True)
    merged_families['rate_families'] = merged_families['count_families'] / merged_families['count_families'].sum()
    merged_families['rate_family_archetypes'] = merged_families['count_family_archetypes'] / merged_families['count_family_archetypes'].sum()
    merged_families['rate_difference'] = abs((merged_families['rate_families'] - merged_families['rate_family_archetypes'])/ merged_families['rate_family_archetypes']*100)

    # Combinar df_final_stats_citizens con df_final_stats_citizen_archetypes
    merged_citizens = df_final_stats_citizens.merge(df_final_stats_citizen_archetypes, left_on='archetype', right_on='name', how='outer', suffixes=('_citizens', '_citizen_archetypes')).drop(columns=['name'])
    merged_citizens.fillna(0, inplace=True)
    merged_citizens['rate_citizens'] = merged_citizens['count_citizens'] / merged_citizens['count_citizens'].sum()
    merged_citizens['rate_citizen_archetypes'] = merged_citizens['count_citizen_archetypes'] / merged_citizens['count_citizen_archetypes'].sum()
    merged_citizens['rate_difference'] = abs((merged_citizens['rate_citizens'] - merged_citizens['rate_citizen_archetypes'])/merged_citizens['rate_citizen_archetypes']*100)

    # Mostrar el promedio de rate_difference
    print("    Abs error on citizens:", round(merged_citizens['rate_difference'].mean(), 4), "%")
    print("    Abs error on families:", round(merged_families['rate_difference'].mean(), 4), "%")
    

if __name__ == '__main__':
    main()

