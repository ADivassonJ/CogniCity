import os
from pathlib import Path
import pandas as pd

# Función para cargar, filtrar, ordenar y reiniciar índices
def load_filter_sort_reset(filepath):
    try:
        df = pd.read_excel(filepath)
        df = df[df['state'] != 'inactive']  # Filtrar filas donde 'state' no sea 'inactive'
        df = df.sort_values(by='presence', ascending=False)  # Ordenar por 'presence' de mayor a menor
        df = df.reset_index(drop=True)  # Reiniciar índices y eliminar la columna original de índices
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

############ Basic data ############
population = 100
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
    
    while not df_distribution.empty:
        random_arch(archetype_to_fill)

