import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

def load_filter_sort_reset(filepath):
    """
    Carga un archivo Excel, filtra las filas donde 'state' sea 'inactive'
    y retorna el DataFrame.
    """
    try:
        df = pd.read_excel(filepath)
        df = df[df['state'] != 'inactive']
        return df
    except Exception as e:
        print(f"{filepath.name} not found or error loading: {e}")
        return None

def random_arch(df):
    """
    Selecciona aleatoriamente un 'name' del DataFrame basado en la 
    distribución de probabilidades calculada a partir de la columna 'presence'.
    """
    if df is not None and not df.empty:
        presence_probabilities = df['presence'] / df['presence'].sum()
        random_name = np.random.choice(df['name'], p=presence_probabilities)
        return random_name
    else:
        print("Error en random_arch: DataFrame vacío o None.")
        return None

def compute_presence_distribution(df, population):
    """
    Calcula el porcentaje de 'presence' y asigna a cada fila la población 
    (redondeada) proporcional a ese porcentaje.
    
    Retorna:
      - Un DataFrame con las columnas ['name', 'population'].
      - El total de 'presence' para usarlo en posteriores iteraciones.
    """
    total_presence = df['presence'].sum()
    if total_presence > 0:
        df['presence_percentage'] = df['presence'] / total_presence
    else:
        df['presence_percentage'] = 0
    df['population'] = (df['presence_percentage'] * population).round().astype(int)
    return df[['name', 'population']].copy(), total_presence

def process_arch_to_fill(archetype_df, arch_name, df_distribution):
    """
    Para un 'arch_name' seleccionado, filtra la fila correspondiente en el DataFrame
    'archetype_df', conserva solo las columnas que contengan la palabra 'archetype',
    transpone el resultado y lo une con 'df_distribution' para poder compararlo.
    
    Retorna el DataFrame mergeado.
    """
    row = archetype_df[archetype_df['name'] == arch_name]
    # Seleccionar columnas que contengan "archetype" (sin importar mayúsculas/minúsculas)
    columns_to_keep = [col for col in row.columns if 'archetype' in col.lower()]
    row_filtered = row[columns_to_keep]
    # Transponer y reformatear el DataFrame
    transposed = row_filtered.T.reset_index()
    transposed.columns = ['name', 'participants']
    # Unir con df_distribution para comparar los valores
    merged_df = pd.merge(df_distribution, transposed, on='name', how='left')
    return merged_df

def update_distribution(archetype_to_fill, df_distribution, total_presence):
    """
    Actualiza el DataFrame de distribución ('df_distribution') usando los datos
    del DataFrame 'archetype_to_fill'. Se itera mientras que la cantidad de intentos 
    (counter) sea menor que 'total_presence', reiniciando el contador cuando se logra
    descontar participantes.

    Durante la ejecución se muestra un mensaje dinámico que indica
    "Distributing the population" con puntos que van de 1 a 3.
    """
    counter = 0
    dot_count = 1  # Número inicial de puntos

    while counter < total_presence:
        # Construir y mostrar el mensaje con la cantidad de puntos correspondiente.
        message = f"\rDistributing the population{'.' * dot_count}   "
        sys.stdout.write(message)
        sys.stdout.flush()
        dot_count = dot_count % 3 + 1  # Ciclo de 1 a 3 puntos
        time.sleep(0.3)  # Ajusta el retardo según lo que prefieras

        arch_to_fill = random_arch(archetype_to_fill)
        if arch_to_fill is None:
            print("\nNo se pudo seleccionar un archetype para rellenar.")
            break

        merged_df = process_arch_to_fill(archetype_to_fill, arch_to_fill, df_distribution)

        for idx, row in merged_df.iterrows():
            if pd.notna(row['participants']):
                if row['participants'] <= row['population']:
                    # Actualiza la población restante para ese archetype
                    df_distribution.loc[df_distribution['name'] == row['name'], 'population'] -= row['participants']
                    counter = 0  # Reiniciamos el contador al aplicar una actualización
                else:
                    counter += 1
            # Si la columna 'population' es NaN se deja sin cambios
            if pd.isna(row['population']):
                df_distribution.loc[df_distribution['name'] == row['name'], 'population'] = np.nan
        
        # Salir si se excede el número de intentos
        if counter >= total_presence:
            print("\n    [DONE]")
            break

    # Limpiar la línea del mensaje al finalizar.
    sys.stdout.write("\r" + " " * 50 + "\r")
    return df_distribution

def main():
    # Configuración básica
    population = 45000
    priority_homes = False
    study_area = 'Kanaleneiland'
    
    # Definir rutas relativas a partir de __file__
    main_path = Path(__file__).resolve().parent.parent
    subcodes_path = main_path / 'Subcodes'
    archetypes_path = main_path / 'Archetypes'
    data_path = main_path / 'Data'
    
    # Cargar los DataFrames de los archetypes
    a_archetypes = load_filter_sort_reset(archetypes_path / 'a_archetypes.xlsx')
    h_archetypes = load_filter_sort_reset(archetypes_path / 'h_archetypes.xlsx')
    s_archetypes = load_filter_sort_reset(archetypes_path / 's_archetypes.xlsx')
    
    # Seleccionar los DataFrames según la prioridad definida
    if priority_homes:
        archetype_to_analyze = h_archetypes
        archetype_to_fill = a_archetypes
    else:
        archetype_to_analyze = a_archetypes
        archetype_to_fill = h_archetypes
    
    if archetype_to_analyze is not None:
        # Calcular la distribución inicial de población
        df_distribution, total_presence = compute_presence_distribution(archetype_to_analyze, population)
        
        # Actualizar la distribución según los participantes de los archetypes
        df_distribution = update_distribution(archetype_to_fill, df_distribution, total_presence)
        
        print("Distribución final:")
        print(df_distribution)
#        input("Presione Enter para finalizar...")
    else:
        print("Error: No se pudo cargar el DataFrame para analizar.")

if __name__ == '__main__':
    main()

