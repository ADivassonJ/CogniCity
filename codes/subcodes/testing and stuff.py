import os
import sys
import random
import folium
import itertools
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt 
from folium.plugins import AntPath
from geopy.distance import geodesic
from collections import defaultdict
from datetime import datetime, timedelta



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
    
    level_1_results = pd.read_excel(f"{paths['results']}/{study_area}_level_1.xlsx")
    level_2_results = pd.read_excel(f"{paths['results']}/{study_area}_level_2.xlsx")
    
    pop_citizen = pd.read_excel(f"{paths['population']}/pop_citizen.xlsx")
    
    pop_building = pd.read_excel(f"{paths['population']}/pop_building.xlsx")
    pop_transport = pd.read_excel(f"{paths['population']}/pop_transport.xlsx")
    
    pop_archetypes_transport = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    
    level2_families = level_2_results.groupby(['family'])
    
    for f_name, family in level2_families:
        avail_vehicles = pop_transport[pop_transport['family'] == f_name]
        
        print(f"{f_name} has the following vehicles available")
        print(avail_vehicles)
        
        level2_citizens = family.groupby(['agent'])
        
        for c_name, c_route in level2_citizens:
            
            citizen = pop_citizen[pop_citizen['name'] == c_name]
            
            avail_transport = add_walk_public(avail_vehicles, c_route['osm_id'].unique(), pop_archetypes_transport, citizen)
            
            score_calculation()


def score_calculation(pop_building, avail_transport):
    
    for _, transport in avail_transport.iterrows():
        # Filtrar solo los que son 'private_transportation'
        p1p2_options = pop_building[pop_building['archetype'] == 'charging_station']



def add_walk_public(avail_vehicles, c_route, pop_archetypes_transport, citizen):
    """_summary_

    Args:
        avail_vehicles (df): 
        citizen (df): 
    """
    
    
    
    
    
    
    
    # Inicializamos el df de resultados
    avail_transport = pd.DataFrame()
    # Empezamos añadiendo 'walk' como opcion
    new_row = {
        'name': 'walk',
        'archetype': None,
        'family': None,
        'ubication': None,
        'v': citizen['walk_speed'],
        'Ekm': None,
        'enkm': None, 
        'Emin': None,
        'COkm': None,
        'SoC': None,
    }
    # Lo añadimos el df de resultados
    avail_transport = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    # Repetimos proceso añadiendo 'public' como opcion
    name = 'conb_public'
    new_row = {
        'name': name,
        'archetype': None,
        'family': None,
        'ubication': obtain_P1P2(name, citizen, pop_archetypes_transport, c_route),
        'v': 3,
        'Ekm': 3,
        'enkm': None, 
        'Emin': None,
        'COkm': 3,
        'SoC': None,
    }
    # Lo añadimos el df de resultados
    avail_transport = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    
    return avail_transport
    
    
    
def obtain_P1P2(archetype, citizen, transport_archetypes, c_route):
    """_summary_

    Args:
        archetype (_type_): _description_
        citizen (_type_): _description_
    """
    
    
    P1s = transport_archetypes[transport_archetypes['name'] == archetype]['P1s']
    P2s = transport_archetypes[transport_archetypes['name'] == archetype]['P2s']
    
    find_closest(c_route, P1s)
    find_closest(c_route, P2s)
    

def find_closest(c_route, P1s):
    
    input(f"osm_id_0:\n {c_route}")

# Ejecución
if __name__ == '__main__':
    main_td()
    
    
    
    