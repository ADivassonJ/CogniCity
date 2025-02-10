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

############ Basic data ############
population = 100
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

if a_archetypes is not None:
    # Calcular el porcentaje de 'presence'
    total_presence = a_archetypes['presence'].sum()  # Sumar toda la columna 'presence'
    if total_presence > 0:  # Evitar división por cero
        a_archetypes['presence_percentage'] = a_archetypes['presence'] / total_presence
    else:
        a_archetypes['presence_percentage'] = 0  # Si la suma es 0, asignar 0%

    # Multiplicar 'presence_percentage' por population, redondear y convertir a entero
    a_archetypes['population'] = (a_archetypes['presence_percentage'] * population).round().astype(int)

    # Mostrar las primeras filas para verificar
    print(a_archetypes.head())

