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
    population = 45000
    study_area = 'Kanaleneiland'
    
    
    paths, system_management, pop_archetypes, agent_populations, networks_map = Documents_initialisation(population, study_area)

    # pop_error_printing(agent_populations['citizen'], agent_populations['family'], pop_archetypes['citizen'], pop_archetypes['family'])
    
    Daily_schedule_definition(study_area, paths, system_management, pop_archetypes, networks_map, agent_populations)


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

