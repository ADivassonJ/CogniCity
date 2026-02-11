# === Importaciones estándar ===================================================
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

def CogniCity(population: int, study_area: str, WP3_active: bool):
    paths, system_management, pop_archetypes, agent_populations, networks_map = Documents_initialisation(population, study_area)
    
    Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations, WP3_active)

    build_quantified_outputs_per_excel(paths=paths, study_area=study_area)

    build_daily_total_stats_from_constructed_outputs(
        paths=paths,
        study_area=study_area,
    )


### Main
def main():
    # Input

    '''    
    population = 260
    study_area = 'Kanaleneiland'
    '''    

    
    population = 200
    study_area = 'Aradas'

 
          
    population = 280
    study_area = 'Annelinn'
           

    WP3_active = True
    
    CogniCity(population, study_area, WP3_active)



if __name__ == '__main__':
    main()

