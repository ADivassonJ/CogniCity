# v11   ->  se a integrado el dato 'link_to' al string de datos de output.
#           los ciudadanos estan vinculados a EV aleatorios, de momento, todos estan vinculados al menos a uno.
# v12   ->  se le ha añadido la función 'data_gather', para limpiar un poco el codigo.
# v13   ->  se suma la funcion 'initialize_SoC()'
#           ciclo day
# v14   ->  modificación del sistema de gestion de información: se dividen los datos en cuatro df; edificion, ciudadanos, EV y acciones.
#           del mismo modo, los output se generan a partir de estos datos se dan mediante otro df
# v15   ->  arreglar la variable in_out (ahora llamada action)

import os
import math
import random
import osmnx as ox
import pandas as pd
from geopy.distance import great_circle
from shapely.geometry import Point, Polygon, MultiPolygon

def get_center(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        return centroid.y, centroid.x
    elif isinstance(geom, Point):
        return geom.y, geom.x
    return None, None

def get_df_buildings(area):
    try:
        print(f"    Retrieving building data for {area}...")
        edificios = ox.features_from_place(area, tags={'building': True})
        building_ID = edificios.index.get_level_values('osmid').tolist()

        values = {'osmid': building_ID, 'coord': []}
        names = ['osmid', 'coord']

        for geom in edificios['geometry']:
            resultado = get_center(geom)
            values['coord'].append(resultado)

        variables = ['building', 'amenity', 'geometry']
        for vari in variables:
            if vari in edificios.columns:
                data_array = edificios[vari].tolist()
                names.append(vari)
                values[vari] = data_array

        df_buildings = pd.DataFrame(values, columns=names)
        print("    [Completed]")
        return df_buildings
    except Exception as e:
        print(f"Error retrieving buildings data:")
        print(e)
        return pd.DataFrame()

def building_has_service(directory, df_buildings, osmid, service_type):
    try:
        to_study = df_buildings.loc[df_buildings['osmid'] == osmid, 'amenity']
        if pd.isna(to_study.iloc[0]) or to_study.iloc[0] == 'yes':
            to_study = df_buildings.loc[df_buildings['osmid'] == osmid, 'building']
            if pd.isna(to_study.iloc[0]):
                print("NO DATA FROM BUILDING")
                return False

        to_study_str = str(to_study.iloc[0])
        if to_study_str == 'yes':
            return random.choice([True, False])
        else:
            filepath = os.path.join(directory, 'building characterization.csv')
            df_chara = pd.read_csv(filepath)
            mask = df_chara.iloc[:, 0].str.contains(to_study_str, na=False)
            if mask.any():
                in_type = df_chara.loc[mask, service_type].iloc[0]
                return in_type == 'x'
            return False
    except Exception as e:
        print(f"Error saving to Excel:")
        print(e)
        return False

# Modifies df_building adding a new column which describes the services available in the building
def buildind_to_services(directory, type_services, df_buildings):
    print("    Tagging all buildings to their services.")
    # Create a new dictionary to store services for each building
    building_services = {}
    # Test every building on df_buildings
    for index, row in df_buildings.iterrows():
        available = []
        # Test every type of service
        for service in type_services:
            # Does the building has the service available?
            flag = building_has_service(directory, df_buildings, row['osmid'], service)
            # If the building has the service available
            if flag:
                available.append(service)
        # Add services list to the dictionary
        building_services[index] = available
    # Assign the dictionary to the 'services' column of df_buildings
    df_buildings['services'] = pd.Series(building_services)
    print("    [Completed]")

# Creates and Excel out of any given DataFrame
def df_a_excel(directory, df, nombre):
    try:
        path = os.path.join(directory, f'{nombre}.xlsx')
        df.to_excel(path, index=False)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving to Excel:")
        print(e)

def initialize_population(df_buildings, population, EV_percentage):
    print("    Initializing population")
    # Initialize some parameters
    day = 0
    time = 0
    living_buildings = []
    working_buildings = []
    citizen_names = [f'citizen_{i}' for i in range(population)]
    vehicles_names = [f'elecvehi_{i}' for i in range(int(population * EV_percentage))]
    df_actions = pd.DataFrame(columns=['osmid', 'day', 'time', 'agent_name', 'action', 'agent_type', 'archetype'])
    df_citizens = pd.DataFrame(columns=['agent_name', 'archetype', 'link_to', 'vehicle', 'Living', 'Working'])
    df_vehicles = pd.DataFrame(columns=['agent_name', 'archetype', 'SoC'])
    # Create lists for all living and all working-related buildings
    for building in df_buildings.iterrows():
        if 'Living' in building[1]['services']:
            living_buildings.append(building[1]['osmid'])
        if 'Working' in building[1]['services']:
            working_buildings.append(building[1]['osmid'])
    # Add information to df_citizens and df_actions
    for citizen in citizen_names:
        # Selection of variables (depending on statistical model)
        archetype = random.choice(['abcd', 'efgh', 'ijkl'])
        living_on = random.choice(living_buildings) ## Habría que poner una condicion de que el edificio no este saturado de personas ya
        working_on = random.choice(working_buildings)
        vehicle_on_property = random.choice(vehicles_names)
        # Add all data from new citizen to df_citizens
        new_row = {'agent_name': citizen, 'archetype': archetype, 'link_to': None, 'vehicle': vehicle_on_property, 'Living': living_on, 'Working': working_on}
        df_citizens = pd.concat([df_citizens, pd.DataFrame([new_row])], ignore_index=True)
        # Add all data from new citizen to df_actions
        new_row = {'osmid': random.choice([living_on, working_on]), 'day': day, 'time': time, 'agent_name': citizen, 'action': 'in', 'agent_type': 'citizen', 'archetype': archetype}
        df_actions = pd.concat([df_actions, pd.DataFrame([new_row])], ignore_index=True)
    # Add information to df_vehicles
    for vehicle in vehicles_names:
        # Selection of variables (depending on statistical model)
        archetype = random.choice(['012', '345', '6789'])
        # Asign State of Charge (SoC) of vehicles
        SoC = 0 ## Por empezar con algo
        # Add all data from new vehicles to df_vehicles
        new_row = {'agent_name': vehicle, 'archetype': archetype, 'SoC': SoC}
        df_vehicles = pd.concat([df_vehicles, pd.DataFrame([new_row])], ignore_index=True)
        # Add all data from new vehicles to df_actions
        new_row = {'osmid': random.choice(df_buildings['osmid']), 'day': day, 'time': time, 'agent_name': vehicle, 'action': 'in', 'agent_type': 'vehicle', 'archetype': archetype}
        df_actions = pd.concat([df_actions, pd.DataFrame([new_row])], ignore_index=True)
    print("    [Completed]")
    return df_actions, df_citizens, df_vehicles

def initialization(area, population,EV_percentage, directory, type_services):
    print("Initializing sistem")
    # Scan selected area and obtain needed data from OpenStreetMap
    df_buildings = get_df_buildings(area)
    if df_buildings.empty:
        print("[ERROR] No data retrieved from the city.", "-"*40)
        return
    # Add to data labels of available services per building
    buildind_to_services(directory, type_services, df_buildings)
    
    df_actions, df_citizens, df_vehicles = initialize_population(df_buildings, population, EV_percentage)
    print("[Completed]")
    return df_buildings, df_actions, df_citizens, df_vehicles

def next_move(df_buildings, data, intention):
    flag = 0
    building_name = data.loc[0, 'osmid']
    building_coord = df_buildings.loc[df_buildings['osmid'] == int(building_name), 'coord'].iloc[0]
    new_building = building_name
    # Create a list with all possible buildings to go
    available_options = []
    for building in df_buildings.iterrows():
        if intention in building[1]['services']:
            available_options.append(building[1]['osmid'])
    if available_options:
        while new_building == building_name: # while the new building and the previous one are the same, search for a new building
            new_building = random.choice(available_options) # choose a random building name within the section of buildings that meet the necessary tag
            # ADD distance condition
        
        new_building_coord = df_buildings.loc[df_buildings['osmid'] == int(new_building), 'coord'].iloc[0]
        distance = great_circle(building_coord, new_building_coord).kilometers
        travel_time = math.ceil((distance / 3) * 12)  # Assuming 3 km/hr speed for travel time calculation

        if travel_time < 1: # It might happen that the distance is too short so time is evaluated as 0
            travel_time = 1

        # Copy the data to avoid SettingWithCopyWarning
        new_data = data.copy()
        new_data.loc[0, 'osmid'] = int(new_building)
        if new_data.loc[0, 'time'] + travel_time >= 4:
            new_data.loc[0, 'day'] += int((new_data.loc[0, 'time'] + travel_time) / 4)
            new_data.loc[0, 'time'] = (new_data.loc[0, 'time'] + travel_time) % 4  # Update time
        else:
            new_data.loc[0, 'time'] += travel_time
        new_data.loc[0, 'action'] = 'in'
    else:
        flag = 1
        
    return new_data.loc[0], flag
        
def behaviour_module(actions_data, df_citizens):
    # Take data from 'df_actions'
    agent_name = actions_data['agent_name'].iloc[0]  # Extracting the single agent name
    day_data = actions_data['day']
    time_data = actions_data['time']
    # Take data from 'df_citizens'
    citizen_data = df_citizens.loc[df_citizens['agent_name'] == agent_name].iloc[0]
    link_to = citizen_data['link_to']
    archetype_data = citizen_data['archetype']
    # Use behaviour-afecting variables to evaluate next action
    decision, intention  = behaviour_core(link_to, day_data, time_data, archetype_data)
    return decision, intention

def behaviour_core(link_to, day_data, time_data, archetype_data):
    decision = random.choice(['move', 'stay'])
    intention = random.choice(['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment'])
    return decision, intention

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

if __name__ == "__main__":
#   Definition of main variables
    area = "Otxarkoaga"
    population_citizens = 10
    EV_percentage = 0.3
    directory = 'C:/Users/asier.divasson/Downloads'
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']
#   Initialization of stuff 
    moving_agents = pd.DataFrame(columns=['agent_name', 'agent_type', 'osmid', 'link_to', 'day', 'time', 'archetype', 'SoC'])
    df_buildings, df_actions, df_citizens, df_vehicles = initialization(area, population_citizens,EV_percentage, directory, type_services)
    time = 1 # Due to initialization has been made in time 0 day 0 (01/01/20XX, 00.00)
    day = 0
#   Simulation loop
    while day < 2:
        print('#'*20, 'day:', day, '#'*20)
        while time < 4:
            print('-'*20, 'time:', time, '-'*20)
            for agent in moving_agents['agent_name']:
                data = moving_agents.loc[moving_agents['agent_name'] == agent].iloc[0]
                time_data = data['time']
                day_data = data['day']
                if time_data == time and day_data == day:
                    df_actions = pd.concat([df_actions, data.to_frame().T], ignore_index=True)
                    moving_agents = moving_agents[moving_agents['agent_name'] != agent]
                elif time_data < time and day_data <= day:
                    print('-'*20,'ERROR: time_data < time','-'*20)
                    print(agent, 'time:', time, 'day:', day)
                    print(moving_agents)
            for agent in df_citizens['agent_name']:
                if agent not in moving_agents['agent_name'].values:
                    new_data_agent = data_gather(df_actions, agent)
                    decision, intention = behaviour_module(new_data_agent, df_citizens)
    #               print(decision, intention)
                    if decision == 'move':
                        new_data_agent['day'] = day
                        new_data_agent['time'] = time
                        new_data_agent['action'] = 'out'
                        df_actions = pd.concat([df_actions, new_data_agent], ignore_index=True)
                        new_data, flag = next_move(df_buildings, new_data_agent, intention)
                        if flag != 1:
                            moving_agents = pd.concat([moving_agents, pd.DataFrame([new_data])], ignore_index=True)
                            # calcular consumos para SoC
            time += 1
        day += 1
        time = 0
                    
    print(moving_agents)
    
    # Print all data from DataFrames to Excels
    df_names = ['df_buildings', 'df_actions', 'df_citizens', 'df_vehicles']
    df_group = [df_buildings, df_actions, df_citizens, df_vehicles]
    for df in range(len(df_names)):
        df_a_excel(directory, df_group[df], df_names[df])