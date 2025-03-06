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

def update_distribution(archetype_to_fill, df_distribution, total_presence, cond_archetypes):
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
        
        # Obtenermos un df con el numero actual de individuos por distribuir en hogares y el tipo de hogar elegido en esta iteracion (random) lo que consume de cada
        merged_df = process_arch_to_fill(archetype_to_fill, arch_to_fill, df_distribution)

        print(merged_df)

        for idx, row in merged_df.iterrows():
            if pd.notna(row['participants']):
                valor = row['participants']
                if isinstance(valor, str) and valor.strip().lower() == 'nan':  # Caso 'NaN'
                    merged_df.at[idx, 'participants'] = 0
                elif isinstance(valor, str) and valor.strip() == '*':  # Caso '*'
                    row_2 = cond_archetypes[(cond_archetypes['item_1'] == arch_to_fill) & (cond_archetypes['item_2'] == row['name'])]
                    merged_df.at[idx, 'participants'] = round(np.random.normal(row_2['mu'], row_2['sigma'])[0])
                else:
                    try:
                        merged_df.at[idx, 'participants'] = int(valor)  # Convertir a int si es posible
                    except ValueError:
                        pass  # Si no se puede convertir, deja el valor tal como está
                
                if isinstance(valor, str):
                    merged_df.at[idx, 'participants'] = float(merged_df.at[idx, 'participants'])
                
                if merged_df.at[idx, 'participants'] <= row['population']:
                    # Actualiza la población restante para ese archetype
                    df_distribution.loc[df_distribution['name'] == row['name'], 'population'] -= merged_df.at[idx, 'participants']
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


# Función para detectar '*' y agregar las filas correspondientes
def add_matches_to_cond_archetypes(cond_archetypes, df, name_column='name'):
    for col in df.columns:
        # Saltar la columna 'name' ya que no es relevante para la búsqueda del '*'
        if col == name_column:
            continue
        
        # Iterar por cada celda de la columna
        for index, value in df[col].items():
            # Convertir el valor a string
            value_str = str(value)
            # Verificar si la celda contiene un '*'
            if '*' in value_str:
                # Obtener el valor de 'name' y la columna actual, manejando posibles valores NaN
                name_value = df.loc[index, name_column] if pd.notna(df.loc[index, name_column]) else "Unknown"
                # Añadir al DataFrame cond_archetypes
                cond_archetypes.loc[len(cond_archetypes)] = [name_value, col, None, None]
    return cond_archetypes

def main():
    # Configuración básica
    population = 45
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
    
    try:
        cond_archetypes = pd.read_excel(archetypes_path / 'cond_archetypes.xlsx')
        if cond_archetypes.isnull().sum().sum() != 0:
            input(f'{archetypes_path}/cond_archetypes has one or more values empty, please include all μ and σ for each detected scenario')
            sys.exit()
    except Exception:
        # Crear un DataFrame vacío para almacenar los resultados
        cond_archetypes = pd.DataFrame(columns=['item_1', 'item_2', 'mu', 'sigma'])
        
        # Llamar a la función para ambos DataFrames
        cond_archetypes = add_matches_to_cond_archetypes(cond_archetypes, a_archetypes)
        cond_archetypes = add_matches_to_cond_archetypes(cond_archetypes, h_archetypes)
        cond_archetypes.to_excel(archetypes_path/'cond_archetypes.xlsx', index=False)
        
        input(f'{archetypes_path}/cond_archetypes has no information, please include all μ and σ for each detected scenario')
        sys.exit()
    
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
        df_distribution = update_distribution(archetype_to_fill, df_distribution, total_presence, cond_archetypes)
        
        print("Distribución final:")
        print(df_distribution)
#        input("Presione Enter para finalizar...")
    else:
        print("Error: No se pudo cargar el DataFrame para analizar.")

if __name__ == '__main__':
    main()

