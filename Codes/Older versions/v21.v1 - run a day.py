# v11   ->  se a integrado el dato 'link_to' al string de datos de output.
#           los ciudadanos estan vinculados a EV aleatorios, de momento, todos estan vinculados al menos a uno.
# v12   ->  se le ha añadido la función 'data_gather', para limpiar un poco el codigo.
# v13   ->  se suma la funcion 'initialize_SoC()'
#           ciclo day
# v14   ->  modificación del sistema de gestion de información: se dividen los datos en cuatro df; edificion, ciudadanos, EV y acciones.
#           del mismo modo, los output se generan a partir de estos datos se dan mediante otro df
# v15   ->  arreglar la variable in_out (ahora llamada action)
# v16   ->  se agregan componentes para facilitar la comprobación del sistema
# v17   ->  mejoras de actos de comportamiento
# v18   ->  se aplican cambios a df_buildings tras su creacion para eliminar edificios inutiles del esfuerzo computacional
# v19   ->  se incorpora la función 'add_bus_relation', que mediante mapeo voonori, asigna a cada edificio su bus más cercano
# v20   ->  se optimiza el codigo mediante Chat GPT para agilizar computación

import os
import math
import random
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from joblib import Parallel, delayed
from geopy.distance import great_circle
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt

# Configurar pandas para mostrar todas las filas
pd.set_option('display.max_rows', None)

def get_center(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        return centroid.y, centroid.x
    elif isinstance(geom, Point):
        return geom.y, geom.x
    return None, None

def get_df_buildings(area):
    try:
        print(f"Retrieving building data for {area}.")
        edificios = ox.features_from_place(area, tags={'building': True, 'amenity': True})
        building_ID = edificios.index.get_level_values('osmid').tolist()
        values = {'osmid': building_ID, 'coord': []}
        names = ['osmid', 'coord']
        for geom in edificios['geometry']:
            centroid = geom.centroid
            values['coord'].append((centroid.y, centroid.x))
        variables = ['building', 'amenity', 'geometry']
        for vari in variables:
            if vari in edificios.columns:
                data_array = edificios[vari].tolist()
                names.append(vari)
                values[vari] = data_array
        df_buildings = pd.DataFrame(values, columns=names)
        print("[Completed]")
        return df_buildings
    except Exception as e:
        print("Error retrieving buildings data:", e)
        return pd.DataFrame()

def building_has_service(directory, df_buildings, osmid, service_type):
    try:
        # Preprocess the DataFrame to create a dictionary for quicker access
        building_info = df_buildings.set_index('osmid')[['amenity', 'building']].to_dict()
        to_study_amenity = building_info['amenity'].get(osmid)
        to_study_building = building_info['building'].get(osmid)
        if pd.isna(to_study_amenity) or to_study_amenity == 'yes':
            to_study = to_study_building
            if pd.isna(to_study):
                print("NO DATA FROM BUILDING")
                return False
        else:
            to_study = to_study_amenity
        if to_study == 'yes':
            return random.choice([True, False])
        else:
            filepath = os.path.join(directory, 'building characterization.csv')
            df_chara = pd.read_csv(filepath)
            mask = df_chara.iloc[:, 0].str.contains(to_study, na=False)
            if mask.any():
                in_type = df_chara.loc[mask, service_type].iloc[0]
                return in_type == 'x'
            return False
    except Exception as e:
        print(f"Error checking building service:")
        print(e)
        return False

def add_bus_relation(df_buildings, area, directory):
    print("    Tagging all buildings to their buses.")   
    # Create a new dictionary to store bus relationships for each building
    building_bus = {}
    # Definir las coordenadas de los puntos
    filepath = os.path.join(directory, f"{area}_bus_data.xlsx")
    df_buses = pd.read_excel(filepath)
    latitudes = df_buses['lat'].values
    longitudes = df_buses['long'].values
    coordenadas = np.column_stack((longitudes, latitudes))
    
    # Calcular el diagrama de Voronoi
    vor = Voronoi(coordenadas)
    
    # Obtener el mapa de la zona
    G = ox.graph_from_place(area, network_type='all')
    G_parking = ox.graph_from_place(area, network_type={'amenity': 'parking'})
    fig, ax = plt.subplots(figsize=(10, 10))
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='olive', edge_linewidth=0.8, show=False, close=False)
    G_parking.plot(ax=ax, color='red', markersize=5, alpha=0.7)
    # Dibujar el diagrama de Voronoi encima del mapa
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='blue', line_width=1)
    
    # Se recalcula porque el anterior intercambio longitud con latitud para que el grafico se muestre
    coordenadas = np.column_stack((latitudes, longitudes))
    vor = Voronoi(coordenadas)
    
    # Calcular las áreas de las regiones
    areas = []
    for region_index in tqdm(range(len(coordenadas)), desc="       ", bar_format="{desc}{percentage:3.0f}%"):
        indices_puntos_region = vor.regions[vor.point_region[region_index]]
        if -1 not in indices_puntos_region:
            region = vor.vertices[indices_puntos_region]
            areas.append(abs(np.cross(region[:-1], region[1:]).sum()) / 2)
        else:
            areas.append(np.inf)  # Región infinita
    
    # Supongamos que tienes otras coordenadas
    otras_coordenadas = np.array(df_buildings['coord'])
    # Determinar a qué región corresponden las otras coordenadas
    for i, coord in enumerate(otras_coordenadas):
        area = vor.point_region[np.argmin(np.linalg.norm(vor.points - coord, axis=1))]
        bus_name = 'bus_'+str(area)
        building_bus[i] = bus_name
    # Assign the dictionary to the 'bus' column of df_buildings
    df_buildings['bus'] = pd.Series(building_bus)
    
    
    
    # Mostrar el gráfico
    plt.show()

def initialize_population_parallel(df_buildings, population, EV_percentage):
    def initialize_single_citizen(df_buildings, living_buildings, working_buildings, vehicles_names, index):
        citizen_name = f'citizen_{index}'
        archetype = random.choice(['abcd', 'efgh', 'ijkl'])
        living_on = random.choice(living_buildings)
        working_on = random.choice(working_buildings)
        vehicle_on_property = random.choice(vehicles_names)
        return {'agent_name': citizen_name, 'archetype': archetype, 'link_to': None, 'vehicle': vehicle_on_property, 'Living': living_on, 'Working': working_on}

    living_buildings = df_buildings[df_buildings['services'].apply(lambda x: 'Living' in x)]['osmid'].tolist()
    working_buildings = df_buildings[df_buildings['services'].apply(lambda x: 'Working' in x)]['osmid'].tolist()
    vehicles_names = [f'elecvehi_{i}' for i in range(int(population * EV_percentage))]
    citizens_data = Parallel(n_jobs=-1)(delayed(initialize_single_citizen)(df_buildings, living_buildings, working_buildings, vehicles_names, index) for index in range(population))

    citizens_df = pd.DataFrame(citizens_data)
    vehicles_df = pd.DataFrame([{'agent_name': vehicle, 'archetype': random.choice(['012', '345', '6789']), 'SoC': 0} for vehicle in vehicles_names])

    return citizens_df, vehicles_df

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
    for citizen in tqdm(citizen_names, desc="       citizens: ", bar_format="{desc}{percentage:3.0f}%"):
        # Selection of variables (depending on statistical model)
        archetype = random.choice(['abcd', 'efgh', 'ijkl'])
        living_on = random.choice(living_buildings) ## Habría que poner una condicion de que el edificio no este saturado de personas ya
        working_on = random.choice(working_buildings)
        vehicle_on_property = random.choice(vehicles_names)
        # Add all data from new citizen to df_citizens
        new_row = {'agent_name': citizen, 'archetype': archetype, 'link_to': None, 'vehicle': vehicle_on_property, 'Living': living_on, 'Working': working_on}
        df_citizens = pd.concat([df_citizens, pd.DataFrame([new_row])], ignore_index=True)
        # Add all data from new citizen to df_actions
        osmid = random.choice([living_on, working_on])
        bus = df_buildings.loc[df_buildings['osmid'] == osmid, 'bus'].iloc[0]
        new_row = {'osmid': osmid, 'day': day, 'time': time, 'agent_name': citizen, 'action': 'in', 'agent_type': 'citizen', 'archetype': archetype, 'bus': bus}
        df_actions = pd.concat([df_actions, pd.DataFrame([new_row])], ignore_index=True)
    # Add information to df_vehicles
    for vehicle in tqdm(vehicles_names, desc="       vehicles: ", bar_format="{desc}{percentage:3.0f}%"):
        # Selection of variables (depending on statistical model)
        archetype = random.choice(['012', '345', '6789'])
        # Asign State of Charge (SoC) of vehicles
        SoC = 0 ## Por empezar con algo
        # Add all data from new vehicles to df_vehicles
        new_row = {'agent_name': vehicle, 'archetype': archetype, 'SoC': SoC}
        df_vehicles = pd.concat([df_vehicles, pd.DataFrame([new_row])], ignore_index=True)
        # Add all data from new vehicles to df_actions
        osmid = random.choice(df_actions['osmid'])
        bus = df_buildings.loc[df_buildings['osmid'] == osmid, 'bus'].iloc[0]
        new_row = {'osmid': osmid, 'day': day, 'time': time, 'agent_name': vehicle, 'action': 'in', 'agent_type': 'vehicle', 'archetype': archetype, 'bus': bus}
        df_actions = pd.concat([df_actions, pd.DataFrame([new_row])], ignore_index=True)

    return df_actions, df_citizens, df_vehicles

# Modifies df_building adding a new column which describes the services available in the building
def buildind_to_services(directory, type_services, df_buildings):
    print("    Tagging all buildings to their services.")
    
    # Create a new dictionary to store services for each building
    building_services = {}
    # Test every building on df_buildings
    for index, row in tqdm(df_buildings.iterrows(), total=len(df_buildings),  desc="       ", bar_format="{desc}{percentage:3.0f}%"):
        available = []
        # Test every type of service
        for service in type_services:
            # Does the building has the service available?
            flag = building_has_service(directory, df_buildings, row['osmid'], service)
            # If the building has the service available
            if flag:
                available.append(service)
        if available != []:
            # Add services list to the dictionary
            building_services[index] = available
        else:
            # Add services list to the dictionary
            building_services[index] = 'x'
    # Assign the dictionary to the 'services' column of df_buildings
    df_buildings['services'] = pd.Series(building_services)

# Creates and Excel out of any given DataFrame
def df_a_excel(directory, df, nombre):
    try:
        path = os.path.join(directory, f'{nombre}.xlsx')
        df.to_excel(path, index=False)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving to Excel:")
        print(e)

def initialization(area, population,EV_percentage, directory, type_services):
    print("Initializing sistem")
    
    # Scan selected area and obtain needed data from OpenStreetMap
    df_buildings = get_df_buildings(area)
    if df_buildings.empty:
        print("[ERROR] No data retrieved from the city.", "-"*40)
        return
    # Add to data labels of available services per building 
    buildind_to_services(directory, type_services, df_buildings)
    # Delete all buildings that offer no services
    df_buildings.drop(df_buildings[df_buildings['services'] == 'x'].index, inplace=True)
    df_buildings.reset_index(drop=True, inplace=True)
    # Add into 'df_buildings' relation to buses
    add_bus_relation(df_buildings, area, directory)
    # Initialize population
    df_actions, df_citizens, df_vehicles = initialize_population(df_buildings, population, EV_percentage)
    
    print("[Completed]")
    return df_buildings, df_actions, df_citizens, df_vehicles

def next_move(df_buildings, df_citizens, data, intention):
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
        if new_building == building_name:
            return None, 1  
    
    # Filter buildings based on intention and distance
    for idx, row in df_buildings.iterrows():
        if intention in row['services']:
            building_dest = building_coords[row['osmid']]
            distance = great_circle(building_coord, building_dest).meters
            max_distance = 100000 # Adjust based on archetype
            if distance < max_distance:
                available_options.add(row['osmid'])
    
    # Remove current position from available destinations
    available_options.discard(building_name)
    
    if available_options:
        while new_building == building_name: 
            new_building = random.choice(list(available_options))
        
        new_building_coord = building_coords[new_building]
        distance = great_circle(building_coord, new_building_coord).kilometers
        travel_time = math.ceil((distance / 3) * 12)
        if travel_time < 1:
            travel_time = 1
        
        new_data = data.copy()
        new_data.loc[0, 'osmid'] = new_building
        if new_data.loc[0, 'time'] + travel_time >= 4:
            new_data.loc[0, 'day'] += int((new_data.loc[0, 'time'] + travel_time) / steps_captured)
            new_data.loc[0, 'time'] = (new_data.loc[0, 'time'] + travel_time) % steps_captured
        else:
            new_data.loc[0, 'time'] += travel_time
        new_data.loc[0, 'action'] = 'in'
    else:
        flag = 1
    
    return new_data.loc[0], flag

def behaviour_module(time, day, actions_data, df_citizens):
    # Take data from 'df_actions'
    agent_name = actions_data['agent_name'].iloc[0]  # Extracting the single agent name
    # Take data from 'df_citizens'
    citizen_data = df_citizens.loc[df_citizens['agent_name'] == agent_name].iloc[0]
    link_to = citizen_data['link_to']
    archetype_data = citizen_data['archetype']
    # Use behaviour-afecting variables to evaluate next action
    decision, intention  = behaviour_core(link_to, day, time, archetype_data)
    return decision, intention

def behaviour_core(link_to, day, time, archetype_data):
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
    ##################  SIMULATION EVALUATION STUFF  ################## 
    area = "Kanaleneiland"
    population_citizens = 1000
    steps_captured = 4
    days_captured = 2
    ###################################################################
#   Definition of main variables
    EV_percentage = 0.3
    directory = 'C:/Users/asier.divasson/Downloads'
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']
#   Initialization of stuff 
    moving_agents = pd.DataFrame(columns=['agent_name', 'agent_type', 'osmid', 'link_to', 'day', 'time', 'archetype', 'SoC'])
    # Inicialización de la población en paralelo
    df_buildings, df_actions, df_citizens, df_vehicles = initialization(area, population_citizens, EV_percentage, directory, type_services)
    df_citizens, df_vehicles = initialize_population_parallel(df_buildings, population_citizens, EV_percentage)
    time = 1 # Due to initialization has been made in time 0 day 0 (01/01/20XX, 00.00)
    day = 0
#   Simulation loop
    while day < days_captured:
        print('#'*22, 'day:', day+1, '#'*22)
        while time < steps_captured:
            instant = time/steps_captured*24
            hour = f"{int(instant):02d}:{int((instant-int(instant))*60):02d}"
            print('-'*20, 'time:', hour, '-'*20)
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
            for agent in tqdm(df_citizens['agent_name'], desc="Processing: ", bar_format="{desc}{percentage:3.0f}%"):
                if agent not in moving_agents['agent_name'].values:
                    new_data_agent = data_gather(df_actions, agent)
                    decision, intention = behaviour_module(time, day, new_data_agent, df_citizens)
    #               print(decision, intention)
                    if decision == 'move':
                        new_data_agent['day'] = day
                        new_data_agent['time'] = time
                        new_data_agent['action'] = 'out'
                        df_actions = pd.concat([df_actions, new_data_agent], ignore_index=True)
                        new_data, flag = next_move(df_buildings, df_citizens,  new_data_agent, intention)
                        if flag != 1:
                            moving_agents = pd.concat([moving_agents, pd.DataFrame([new_data])], ignore_index=True)
                            # calcular consumos para SoC
            time += 1
        day += 1
        time = 0

    # Print all data from DataFrames to Excels
    df_names = ['df_buildings', 'df_actions', 'df_citizens', 'df_vehicles', 'moving_agents']
    df_group = [df_buildings, df_actions, df_citizens, df_vehicles, pd.DataFrame(moving_agents)]
    for df in range(len(df_names)):
        df_a_excel(directory, df_group[df], df_names[df])
