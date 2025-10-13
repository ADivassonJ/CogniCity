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
import pandas as pd
from subcodes.Documents_initialisation import Documents_initialisation
from subcodes.Daily_schedule_definition import Daily_schedule_definition


### Main
def main():
    # Input
    population = 1000
    study_area = 'Kanaleneiland'
    
    paths, system_management, pop_archetypes, agent_populations, networks_map = Documents_initialisation(population, study_area)
    
    Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations)

if __name__ == '__main__':
    main()

