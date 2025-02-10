import os
import numpy as np
import pandas as pd
from pathlib import Path

# Función para cargar, filtrar, ordenar y reiniciar índices
def load_filter_sort_reset(filepath):
    try:
        df = pd.read_excel(filepath)
        df = df[df['state'] != 'inactive']  # Filtrar filas donde 'state' no sea 'inactive'
        return df
    except Exception as e:
        print(f"{filepath.name} not found or error loading:", e)
        return None  # Retorna None si hay un error

def random_arch(archetype_to_analyze):
    if archetype_to_analyze is not None and not archetype_to_analyze.empty:
        # Normalizar la columna 'presence' para que sea una distribución de probabilidades
        presence_probabilities = archetype_to_analyze['presence'] / archetype_to_analyze['presence'].sum()
        # Seleccionar un 'name' aleatorio según las probabilidades de 'presence'
        random_name = np.random.choice(archetype_to_analyze['name'], p=presence_probabilities)
        return random_name
    else:
        print(f'Some error ocurred when reading a df in function random_arch.')
        return

############ Basic data ############
population = 45000
priority_homes = False
####################################

study_area = 'Kanaleneiland'
main_path = Path(__file__).resolve().parent.parent 
subcodes_path = main_path / 'Subcodes'
archetypes_path = main_path / 'Archetypes'
data_path = main_path / 'Data'

# Cargar y procesar cada DataFrame
a_archetypes = load_filter_sort_reset(archetypes_path / 'a_archetypes.xlsx')
h_archetypes = load_filter_sort_reset(archetypes_path / 'h_archetypes.xlsx')
s_archetypes = load_filter_sort_reset(archetypes_path / 's_archetypes.xlsx')

if priority_homes:
    archetype_to_analyze = h_archetypes
    archetype_to_fill = a_archetypes
else: 
    archetype_to_analyze = a_archetypes
    archetype_to_fill = h_archetypes

if archetype_to_analyze is not None:
    # Calcular el porcentaje de 'presence'
    total_presence = archetype_to_analyze['presence'].sum()  # Sumar toda la columna 'presence'
    if total_presence > 0:  # Evitar división por cero
        archetype_to_analyze['presence_percentage'] = archetype_to_analyze['presence'] / total_presence
    else:
        archetype_to_analyze['presence_percentage'] = 0  # Si la suma es 0, asignar 0%

    # Multiplicar 'presence_percentage' por population, redondear y convertir a entero
    archetype_to_analyze['population'] = (archetype_to_analyze['presence_percentage'] * population).round().astype(int)
    
    df_distribution = archetype_to_analyze[['name', 'population']].copy()
    
    # Mostrar las primeras filas para verificar
    print(df_distribution)
    
    counter = 0
    
    while total_presence > counter:
        arch_to_fill = random_arch(archetype_to_fill)
        # Filtrar la fila
        row = archetype_to_fill[archetype_to_fill['name'] == arch_to_fill]
        columns_to_keep = [col for col in row.columns if 'archetype' in col.lower()]
        row_filtered = row[columns_to_keep]
        # Tansponer la fila (convertir las columnas en filas y viceversa)
        transposed_row = row_filtered.T  # .T transponde el DataFrame
        transposed_row = transposed_row.reset_index()  # Reiniciar el índice para que sea una columna
        transposed_row.columns = ['name', 'participants']
        # Unimos los dataframes por la columna 'name' para compararlos
        merged_df = pd.merge(df_distribution, transposed_row, on='name', how='left')
        # Recorrer todas las filas del dataframe combinado
        for index, row in merged_df.iterrows():
            # Verificamos si la columna 'participants' no es NaN
            if pd.notna(row['participants']):
                # Si el valor en 'participants' es menor o igual al valor en 'population', lo restamos
                if row['participants'] <= row['population']:
                    df_distribution.loc[df_distribution['name'] == row['name'], 'population'] -= row['participants']
                    counter = 0
                else:
                    counter += 1
                    print(f'{total_presence} > {counter}')
            # Si la columna 'population' está vacía, la dejamos como NaN
            if pd.isna(row['population']):
                df_distribution.loc[df_distribution['name'] == row['name'], 'population'] = np.nan
        
    print(df_distribution) 
    input()    

