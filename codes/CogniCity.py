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

def CogniCity(population: int, study_area: str, WP3_active: bool, scenario: str = None, days: dict = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}):
    """
    Orchestrates the CogniCity simulation workflow, from environment initialization 
    to data quantification and statistical analysis.
    
    Args:
        population (int): Total number of agents in the simulation.
        study_area (str): Geographic or administrative boundary for the study.
        WP3_active (bool): Flag to enable/disable Work Package 3 modules.
        scenario (str, optional): Specific simulation setup or climate scenario.
    """

    # Initialize core components: file paths, management systems, and agent networks
    paths, system_management, pop_archetypes, agent_populations, networks_map = Documents_initialisation(population, study_area, scenario)
    
    # Define and execute the daily activity schedules for the agent population
    # Returns a boolean indicating if the simulation results are already quantified and basic stats calculated
    already_done = Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations, WP3_active, days)

    # If quantification and basic stats need to be calculated, the following functions are called
    if not already_done:
        # Generate quantified Excel reports from raw simulation data
        build_quantified_outputs_per_excel(paths=paths, study_area=study_area)

        # Calculate all basic stats from the simulation
        build_daily_total_stats_from_constructed_outputs(
            paths=paths,
            study_area=study_area,
        )


### Main
def main():
    # Input

    WP3_active = False
    scenarios = ["s0", "s1", "s2", "s3", "s4"]
    scenarios = ["paper"]
    study_areas = ['Annelinn', 'Aradas', 'Kanaleneiland']
    days = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}
    days = {'Mo'}
    reduction = 100


    '''population = 27000//reduction
    study_area = 'Annelinn'

    for scenario in scenarios:
        CogniCity(population, study_area, WP3_active, scenario, days)

    population = 10000//reduction
    study_area = 'Aradas'

    for scenario in scenarios:
        CogniCity(population, study_area, WP3_active, scenario, days)'''

    population = 16000//reduction
    study_area = 'Kanaleneiland'

    for scenario in scenarios:
        CogniCity(population, study_area, WP3_active, scenario, days)



if __name__ == '__main__':
    main()

