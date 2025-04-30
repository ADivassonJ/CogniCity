import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from Subcodes.initialization import Archetype_documentation_initialization, Geodata_initialization, Synthetic_population_initialization
pd.set_option('mode.chained_assignment', 'raise')  # Convierte el warning en error

### Main
def main():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    main_path = Path(__file__).resolve().parent.parent
    subcodes_path = main_path / 'Subcodes'
    archetypes_path = main_path / 'Archetypes'
    data_path = main_path / 'Data'
    results_path = main_path / 'Results'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    print('#'*20, ' System initialization ','#'*20)
    # Archetype documentation initialization
    citizen_archetypes, family_archetypes, s_archetypes, cond_archetypes = Archetype_documentation_initialization(main_path, archetypes_path)
    # Geodata initialization
    SG_relationship, networks_map = Geodata_initialization(study_area, data_path)
    # Synthetic population initialization
    df_citizens, df_families = Synthetic_population_initialization(citizen_archetypes, family_archetypes, population, cond_archetypes, data_path, SG_relationship, study_area)
    print('#'*20, ' Initialization finalized ','#'*20)

    pop_error_printing(df_citizens, df_families, citizen_archetypes, family_archetypes)
    



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

