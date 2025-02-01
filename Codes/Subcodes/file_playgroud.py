#   V1.0.0  ->  Se toma la version V31 y se busca dividir todo el codigo en sub-funciones.

import os
import pandas as pd

def df_a_excel(directory, df, nombre):
    try:
        path = os.path.join(directory, f'{nombre}.xlsx')
        df.to_excel(path, index=False)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving to Excel:")
        print(e)

def data_gather(df, agent_name):
    try:
        # Filtrar las filas donde 'agent_name' coincide y seleccionar la última
        row = df.loc[df['agent_name'] == agent_name].iloc[-1]
    except IndexError:
        # Manejo del caso donde no se encuentra el agente
        print(f"No se encontró el agente con nombre: {agent_name}")
        return None
    # Obtener todos los nombres de las columnas del DataFrame
    variables = df.columns
    # Crear un nuevo DataFrame con una sola fila a partir de los datos del agente
    data_agent = pd.DataFrame([{var: row[var] for var in variables}])
    return data_agent

def save_dataframes(directory, df_group, df_names):
    """Guarda los DataFrames en archivos Excel."""
    for df_name, df in zip(df_names, df_group):
        df_a_excel(directory, df, df_name)
