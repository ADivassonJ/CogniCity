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
    level_2_results = pd.read_excel(f"{paths['results']}/{study_area}_level_2_v2.xlsx")
    
    df_citizens = pd.read_excel(f"{paths['population']}/pop_citizen.xlsx")
    
    pop_building = pd.read_excel(f"{paths['population']}/pop_building.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    testdf = level_2_results
    
    testdf['tot_time'] = testdf['out'] - testdf['in']
    
    '''testdf['archetype'] = testdf['agent'].apply(lambda name: find_group(name, df_citizens, 'archetype'))
    testdf['family'] = testdf['agent'].apply(lambda name: find_group(name, df_citizens, 'family'))
    testdf['family_archetype'] = testdf['agent'].apply(lambda name: find_group(name, df_citizens, 'family_archetype'))'''

    media_por_arch_ciudadano = (testdf.groupby(['archetype', 'todo'])['tot_time'].mean().reset_index())
    media_por_arch_ciudadano = media_por_arch_ciudadano.pivot(index='archetype', columns='todo', values='tot_time')

    print('media_por_arch_ciudadano:')
    print(media_por_arch_ciudadano)
    
    
    media_por_arch_familiar = (testdf.groupby(['family_archetype', 'todo'])['tot_time'].mean().reset_index())
    media_por_arch_familiar = media_por_arch_familiar.pivot(index='family_archetype', columns='todo', values='tot_time')

    
    print('media_por_arch_familiar:')
    print(media_por_arch_familiar)
    
    

def find_group(name, df_families, row_out):
    for idx, row in df_families.iterrows():
        if name == row['name']:
            return row[row_out]
    return None


# Ejecución
if __name__ == '__main__':
    main_td()
    
    
    
    