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

def is_it_any_archetype(archetype_to_fill, df_distribution): 
    # Lista donde se almacenarán los valores de 'name' si se detecta un 0 o NaN
    names_with_zero_or_nan = []

    # Iterar sobre las filas del DataFrame
    for index, row in df_distribution.iterrows():
        # Si hay algún 0 o NaN en las columnas numéricas, guarda el valor de 'name'
        if (row[1:] == 0).any() or row[1:].isna().any():
            names_with_zero_or_nan.append(row['name'])

    if names_with_zero_or_nan == []:
        return archetype_to_fill
    # Filtrar nombres de columnas donde los valores sean distintos de NaN o 0
    df_simplificado = archetype_to_fill[['name']].copy()
    df_simplificado['archetypes'] = archetype_to_fill.iloc[:, 4:].apply(
        lambda row: [col for col, val in zip(archetype_to_fill.columns[4:], row) if pd.notna(val) and val != 0],
        axis=1)
    # Lista donde almacenar los valores de 'name' de df_simplificado si existen en names_with_zero_or_nan
    result_names = []

    # Iterar sobre las filas de df_simplificado
    for index, row in df_simplificado.iterrows():
        # Verificar si alguno de los valores de 'archetypes' está en names_with_zero_or_nan
        if any(name in row['archetypes'] for name in names_with_zero_or_nan):
            result_names.append(row['name'])
    # Eliminar filas en archetype_to_fill donde el valor en la columna "name" está en result_names
    archetype_to_fill = archetype_to_fill[~archetype_to_fill['name'].isin(result_names)]
    archetype_to_fill = archetype_to_fill.reset_index(drop=True)
    # Ver el DataFrame resultante
    return archetype_to_fill

def update_distribution(archetype_to_fill, df_distribution, total_presence, cond_archetypes):
    df_homes = pd.DataFrame(columns=['name', 'archetype', 'description', 'members'])
    df_part_citizens = pd.DataFrame(columns=['name', 'archetype', 'description'])
    df_citizens = pd.DataFrame(columns=df_part_citizens.columns)
    over_stat = 0
    
    print('Distributing the population... (it might take a while)')
    
    while 1==1:
        archetype_to_fill = is_it_any_archetype(archetype_to_fill, df_distribution)
        if archetype_to_fill.empty:
            break
        
        arch_to_fill = random_arch(archetype_to_fill)
        
        # Obtenermos un df con el numero actual de individuos por distribuir en hogares y el tipo de hogar elegido en esta iteracion (random) lo que consume de cada
        merged_df = process_arch_to_fill(archetype_to_fill, arch_to_fill, df_distribution)

        for idx in merged_df.index:
            row = merged_df.loc[idx]
            if pd.notna(row['participants']):
                valor = row['participants']
                if isinstance(valor, str) and valor.strip().lower() == 'nan':  # Caso 'NaN'
                    merged_df.at[idx, 'participants'] = 0
                elif isinstance(valor, str) and valor.strip() == '*':  # Caso '*'
                    row_2 = cond_archetypes[(cond_archetypes['item_1'] == arch_to_fill) & (cond_archetypes['item_2'] == row['name'])]
                    counter = 0
                    while counter < 5:
                        stat_value = round(np.random.normal(row_2['mu'], row_2['sigma'])[0])
                        if stat_value < int(row_2['min']) or stat_value > int(row_2['max']):
                            counter =+ 1
                        else:
                            break
                    if counter == 5:
                        print('A problem with statistical values has been detected:')
                        print(row_2)
                        print('Please, solve the issue and run the code again.')
                        sys.exit()
                    merged_df.at[idx, 'participants'] = stat_value
                else:
                    try:
                        merged_df.at[idx, 'participants'] = int(valor)  # Convertir a int si es posible
                    except ValueError:
                        pass  # Si no se puede convertir, deja el valor tal como está

                if merged_df.at[idx, 'participants'] <= row['population']:
                    for _ in range(int(merged_df.at[idx, 'participants'])):
                        new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': 'Cool guy'}
                        df_part_citizens.loc[len(df_part_citizens)] = new_row
                    over_stat = 0
                    flag = False
                else:
                    df_part_citizens = df_part_citizens.drop(df_part_citizens.index)                
                    # Obtener la fila donde es igual al valor dado
                    fila = archetype_to_fill.loc[archetype_to_fill["name"] == arch_to_fill]

                    # Verificar si en esa fila hay algún '*'
                    if not (fila.isin(['*']).any().any()):  
                        archetype_to_fill = archetype_to_fill[archetype_to_fill["name"] != arch_to_fill].reset_index(drop=True)
                    flag = True
                    over_stat += 1
                    break
            if pd.isna(row['population']):
                df_distribution.loc[df_distribution['name'] == row['name'], 'population'] = np.nan
        # 4 por poner algo, si hay 4 fallos deribado de selecciones estadisticas deja de intentearlo (es decir
        # ha intentado crear familias y no ha podido porque los valores de las curvas normales no le han dado
        # los que podia, y esto ha pasado 5 veces, por eso sale).
        if over_stat > 4:
            break
        if flag:
            continue
        # Actualiza la población restante para ese archetype
        mask = df_distribution['population'].notna() & merged_df['participants'].notna()
        df_distribution.loc[mask, 'population'] = df_distribution.loc[mask, 'population'] - merged_df.loc[mask, 'participants']
   
        df_citizens = pd.concat([df_citizens, df_part_citizens], ignore_index=True)
        new_row_2 = {'name': f'home_{len(df_homes)}', 'archetype': arch_to_fill, 'description': 'Cool family', 'members': df_part_citizens['name'].tolist()}
        if not len(df_part_citizens['name'].tolist()) == 0:
            df_homes.loc[len(df_homes)] = new_row_2
        df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
        
    print("\n    [DONE]")

    return df_distribution, df_citizens, df_homes


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
                cond_archetypes.loc[len(cond_archetypes)] = [name_value, col, None, None, None, None]
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
            print(f'{archetypes_path}/cond_archetypes has one or more values empty,')
            print('please include all μ, σ, max and min for each detected scenario and run the code again.')
            sys.exit()
    except Exception:
        # Crear un DataFrame vacío para almacenar los resultados
        cond_archetypes = pd.DataFrame(columns=['item_1', 'item_2', 'mu', 'sigma', 'min', 'max'])
        
        # Llamar a la función para ambos DataFrames
        cond_archetypes = add_matches_to_cond_archetypes(cond_archetypes, a_archetypes)
        cond_archetypes = add_matches_to_cond_archetypes(cond_archetypes, h_archetypes)
        cond_archetypes.to_excel(archetypes_path/'cond_archetypes.xlsx', index=False)
        
        print(f'{archetypes_path}/cond_archetypes has no information,')
        print('please include all μ, σ, max and min for each detected scenario and run the code again.')
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
        df_distribution, df_citizens, df_homes = update_distribution(archetype_to_fill, df_distribution, total_presence, cond_archetypes)
        
        print("Distribución final:")
        print(df_distribution)
        print(df_homes)
#        input("Presione Enter para finalizar...")
    else:
        print("Error: No se pudo cargar el DataFrame para analizar.")

if __name__ == '__main__':
    main()

