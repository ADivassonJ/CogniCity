# === Importaciones estándar ===================================================
import importlib
import subprocess
import sys

def ensure_package(package_name, import_name=None):
    """
    package_name → nombre para pip
    import_name → nombre real del módulo (si difiere)
    """
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Lista de paquetes
packages = [
    ("geopandas", None),
    ("osmnx", None),
    ("folium", None),
    ("pyproj", None),
    ("shapely", None),
    ("scipy", None),
    ("scikit-learn", "sklearn"),  # pip vs import
    ("tqdm", None),
    ("haversine", None),
    ("pyarrow", None),
    ("fastparquet", None),
    ("openpyxl", None),
    ("matplotlib", None),
]

for package_name, import_name in packages:
    ensure_package(package_name, import_name)

import os
import sys
import pandas as pd

# === Librerías externas =======================================================
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pyproj
from haversine import haversine
from scipy import stats
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from sklearn.cluster import KMeans

# === Módulos locales ==========================================================
from subcodes.Documents_initialisation import Documents_initialisation
from subcodes.Daily_schedule_definition import Daily_schedule_definition
from subcodes.results_clean import build_quantified_outputs_per_excel
from subcodes.results_scenario import build_daily_total_stats_from_constructed_outputs

def CogniCity(population: int, study_area: str, WP3_active: bool, scenario: str = None):
    paths, system_management, pop_archetypes, agent_populations, networks_map = Documents_initialisation(population, study_area, scenario)
    
    already_done = Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations, WP3_active)

    if not already_done:
        build_quantified_outputs_per_excel(paths=paths, study_area=study_area)

        build_daily_total_stats_from_constructed_outputs(
            paths=paths,
            study_area=study_area,
        )


### Main
def main():
    # Input

       
    population = 360
    study_area = 'Kanaleneiland'
        
    
    '''
    population = 10000
    study_area = 'Aradas'
    '''   
          
    '''
    population = 27000
    study_area = 'Annelinn'
    '''


    WP3_active = False
    scenario = "s0"


    CogniCity(population, study_area, WP3_active, scenario)
    #CogniCity(population, study_area, WP3_active, scenario)



if __name__ == '__main__':
    main()

