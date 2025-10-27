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

### Main
def main():
    # Input
    population = 200
    study_area = 'Kanaleneiland'
    
    paths, system_management, pop_archetypes, agent_populations, networks_map = Documents_initialisation(population, study_area)
    
    Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations)

if __name__ == '__main__':
    main()

