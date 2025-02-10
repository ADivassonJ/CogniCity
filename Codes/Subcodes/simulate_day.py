#   V1.0.0  ->  Se toma la version V31 y se busca dividir todo el codigo en sub-funciones.

import os
import re
import math
import random
import pandas as pd
from tqdm import tqdm
from geopy.distance import great_circle
from Subcodes.geometries_playground import parse_coordinates

def simulate_day(df_actions, df_citizens, df_buildings, moving_agents, day, steps_captured, main_path):
    """Simula un día completo."""
    for time in range(steps_captured):
        if day == 0 and time == 0:
            continue
        instant = time / steps_captured * 24
        hour = f"{int(instant):02d}:{int((instant - int(instant)) * 60):02d}"
        print('-' * 20, 'time:', hour, '-' * 20)
        
        # Procesamiento de agentes en movimiento
        df_actions, moving_agents = process_moving_agents(df_actions, moving_agents, time, day)

        # Procesamiento de ciudadanos
        df_actions, moving_agents = process_agents(df_actions, df_citizens, df_buildings, moving_agents, time, day, steps_captured, main_path)
        
    return df_actions, moving_agents

def process_moving_agents(df_actions, moving_agents, time, day):
    """Procesa los agentes en movimiento y actualiza el DataFrame de acciones."""
    for agent in tqdm(moving_agents['agent_name'], desc="Preprocessing: ", bar_format="{desc}{percentage:3.4f}%"):
        data = moving_agents.loc[moving_agents['agent_name'] == agent].iloc[0]
        time_data, day_data = data['time'], data['day']
        
        if time_data == time and day_data == day:
            df_actions = pd.concat([df_actions, data.to_frame().T], ignore_index=True)
            moving_agents.drop(index=data.name, inplace=True)
        elif time_data < time and day_data <= day:
            raise ValueError(f"ERROR: time_data < time for agent {agent} (time: {time}, day: {day})")
    
    return df_actions, moving_agents

def process_agents(df_actions, df_citizens, df_buildings, moving_agents, time, day, steps_captured, main_path):
    """Procesa los ciudadanos y decide sus movimientos."""
    for agent in tqdm(df_citizens['agent_name'], desc="Processing: ", bar_format="{desc}{percentage:3.4f}%"):
        if agent not in moving_agents['agent_name'].values:
            new_data_agent = data_gather(df_actions, agent)
            decision, intention = behaviour_module(time, day, new_data_agent, df_citizens, steps_captured, main_path)
            
            if
                # Asegúrate de que new_data_agent sea un DataFrame antes de concatenar
                if isinstance(new_data_agent, pd.Series):
                    new_data_agent = new_data_agent.to_frame().T
                
                df_actions = pd.concat([df_actions, new_data_agent], ignore_index=True)
                new_data, flag = next_move(df_buildings, df_citizens, new_data_agent, intention, steps_captured)
                
                if flag != 1:
                    moving_agents = pd.concat([moving_agents, pd.DataFrame([new_data])], ignore_index=True)
                    # calcular consumos para SoC (si aplica)
    return df_actions, moving_agents

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

def next_move(df_buildings, df_citizens, data, intention, steps_captured):
    flag = 0
    building_name = data.loc[0, 'osmid']
    agent_name = data.loc[0, 'agent_name']
    
    # Store building coordinates in a dictionary for quick access
    building_coords = dict(zip(df_buildings['osmid'], df_buildings['coord']))
    
    building_coord = building_coords[building_name]
    new_building = building_name
    
    # Create a set with all possible buildings to go
    available_options = set()
    
    if intention in ['Living', 'Working']:
        new_building = df_citizens.loc[df_citizens['agent_name'] == agent_name, intention].iloc[0]
    
    # Filter buildings based on intention and distance
    for idx, row in df_buildings.iterrows():
        if intention in row['services']:
            building_dest = building_coords[row['osmid']]
            building_coord = parse_coordinates(building_coord)
            building_dest = parse_coordinates(building_dest)
            distance = great_circle(building_coord, building_dest).meters                                   # Modificar a Djikska
            max_distance = 100000                                                                           # Adjust based on archetype
            if distance < max_distance:
                available_options.add(row['osmid'])
    
    # Remove current position from available destinations
    available_options.discard(building_name)
    
    if available_options:
        while new_building == building_name: 
            new_building = random.choice(list(available_options))
        
        new_building_coord = building_coords[new_building]
        building_coord = parse_coordinates(building_coord)
        new_building_coord = parse_coordinates(new_building_coord)
        distance = great_circle(building_coord, new_building_coord).kilometers
        speed = 1 #km/h                 # Porque si                                     #########################
        travel_time = distance/speed
        steps2jump = math.ceil(travel_time/(24/steps_captured))      # cada Km es un step si son 24h es una hora por km  ######################### AQUI hay que implementar coches y tal
        
        new_data = data.copy()
        new_data.loc[0, 'osmid'] = new_building
        new_data.loc[0, 'service'] = intention
        if new_data.loc[0, 'time'] + steps2jump >= steps_captured:
            new_data.loc[0, 'day'] += int((new_data.loc[0, 'time'] + steps2jump) / steps_captured)
            new_data.loc[0, 'time'] = (new_data.loc[0, 'time'] + steps2jump) % steps_captured
        else:
            new_data.loc[0, 'time'] += steps2jump
        new_data.loc[0, 'action'] = 'in'
        row = df_buildings[df_buildings['osmid']==new_building]
        new_data.loc[0, 'bus'] = row['bus'].iloc[0]
    else:
        flag = 1
    
    return new_data.loc[0], flag

def behaviour_module(time, day, actions_data, df_citizens, steps_captured, main_path):
    # Take data from 'df_actions'
    agent_name = actions_data['agent_name'].iloc[0]  # Extracting the single agent name
    service_data = actions_data['service'].iloc[0]
    # Take data from 'df_citizens'
    citizen_data = df_citizens.loc[df_citizens['agent_name'] == agent_name].iloc[0]
    link_to = citizen_data['link_to']
    archetype_data = citizen_data['archetype']
    # Use behaviour-afecting variables to evaluate next action
    decision, intention  = behaviour_core(link_to, day, time, archetype_data, service_data, steps_captured, main_path)
    #if decision == "move" and time in [0, 1, 2, 3 ,4, 5, 6]:
        #print(f"move from {service_data} to {intention}")
    return decision, intention

def behaviour_core(link_to, day, time, archetype_data, service_data, steps_captured, main_path):
    # Calcular la hora a partir de los pasos y el tiempo
    instant = time / steps_captured * 24
    hour = f"{int(instant):02d}:{int((instant - int(instant)) * 60):02d}"
    
    etiquetas, valores = archetype_probability_movement(archetype_data, hour, main_path)   
    
    # Seleccionar una etiqueta de forma estadística
    intention = random.choices(etiquetas, weights=valores)[0]

    if intention == service_data and intention in ['Living', 'Working', 'Healthcare', 'Education']:
        decision = 'stay'
    else:
        decision = 'move'
    
    return decision, intention

def archetype_probability_movement(archetype, hour, main_path):
    # Directorio donde se encuentran los archivos CSV
    behavior_model = f'{main_path}/Data/Behaviour Models'
    ruta_archivo = os.path.join(behavior_model, archetype + ".csv")
    
    # Cargar el archivo CSV
    df_behaviour = pd.read_csv(ruta_archivo)
    df_valores = df_behaviour[df_behaviour['Time'] == hour] 
    
    # Verificar si df_fila_XX no está vacío
    if df_valores.empty:
        print(f"No se encontró la hora {hour} en el Excel de {archetype}.")
        print(f"Recuerda que 6:00 y 06:00 no se detectan igual...")
        return None, None  # O maneja el error de otra manera

    # Obtener las etiquetas de las columnas
    tags = df_valores.columns.tolist()[1:]
    values = df_valores.values.flatten().tolist()[1:]
    
    return tags, values