#   initialization 1.0.0  ->  Se toma la version V31 y se busca dividir todo el codigo en sub-funciones.

import os
import random
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm
from scipy.spatial import Voronoi
from Subcodes.simulate_day import archetype_probability_movement
from Subcodes.voronoi_tools import find_voronoi_region

def add_bus_relation(df_buildings, area, data_path):
    print("    Tagging all buildings to their buses.")       
    return assign_buildings_to_buses(df_buildings, data_path, area)

def assign_buildings_to_buses(df_buildings, data_path, area):
    """Asigna cada edificio a una región del diagrama de Voronoi y determina su bus correspondiente."""
    building_bus = {}
    otras_coordenadas = np.array(df_buildings['coord'])
    for i, coord in enumerate(otras_coordenadas):
        print(f'holi')
        bus_name = find_voronoi_region(data_path, area, coord)
        building_bus[i] = bus_name
    df_buildings['bus'] = pd.Series(building_bus)
    
    return df_buildings

def assign_families_to_citizens(df_citizens, max_family_size=5):
    """Asigna familias a los ciudadanos que viven en el mismo edificio."""
    df_citizens['family'] = None  # Creamos la columna para almacenar la familia
    family_counter = 0  # Contador global de familias

    # Agrupamos por edificio de vivienda
    grouped = df_citizens.groupby('Living')

    for name, group in grouped:
        citizens = group.index.tolist()
        random.shuffle(citizens)  # Mezclamos a los ciudadanos del edificio

        # Asignamos ciudadanos a familias
        for i in range(0, len(citizens), max_family_size):
            family_id = f"family_{family_counter}"
            family_counter += 1
            # Asignar familia a cada ciudadano en el rango correspondiente
            for citizen_index in citizens[i:i + max_family_size]:
                df_citizens.at[citizen_index, 'family'] = family_id

    return df_citizens

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

def building_to_services(directory, type_services, df_buildings):
    print("    Tagging all buildings to their services.")
    
    # Crear un diccionario para almacenar los servicios disponibles por edificio
    building_services = {}

    # Iterar sobre cada edificio en df_buildings
    for index, row in tqdm(df_buildings.iterrows(), total=len(df_buildings), desc="       ", bar_format="{desc}{percentage:3.4f}%"):
        available_services = check_available_services(directory, type_services, df_buildings, row['osmid'])
        building_services[index] = available_services if available_services else 'x'
    
    # Asignar los servicios al DataFrame
    df_buildings['services'] = pd.Series(building_services)
    return df_buildings

def check_available_services(directory, type_services, df_buildings, building_osmid):
    """Verifica qué servicios están disponibles para un edificio dado."""
    available_services = []
    for service in type_services:
        if building_has_service(directory, df_buildings, building_osmid, service):
            available_services.append(service)
    return available_services

def generate_agent_names(population, EV_percentage):
    """Genera nombres para ciudadanos y vehículos eléctricos."""
    citizen_names = [f'citizen_{i}' for i in range(population)]
    vehicles_names = [f'elecvehi_{i}' for i in range(int(population * EV_percentage))]
    return citizen_names, vehicles_names

def get_building_lists(df_buildings):
    """Crea listas de edificios donde se vive y se trabaja."""
    living_buildings = []
    working_buildings = []
    for building in df_buildings.iterrows():
        if 'Living' in building[1]['services']:
            living_buildings.append(building[1]['osmid'])
        if 'Working' in building[1]['services']:
            working_buildings.append(building[1]['osmid'])
    return living_buildings, working_buildings

def get_df_buildings(area):
    try:
        print(f"Retrieving building data for {area}.")
        edificios = ox.features_from_place(area, tags={'building': True})
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

def initialization(area, population_citizens, EV_percentage, main_path, type_services):
    print("Initializing sistem")
    main_path = str(main_path)
    data_path = f'{main_path}/Data/{area}'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    try:
        df_buildings = pd.read_csv(f'{data_path}/df_buildings.csv')
        print('df_buildings.csv readed.')
    except Exception as e:
        # Scan selected area and obtain needed data from OpenStreetMap
        df_buildings = get_df_buildings(area)
        # Add to data labels of available services per building 
        building_to_services(data_path, type_services, df_buildings)
        # Delete all buildings that offer no services
        df_buildings.drop(df_buildings[df_buildings['services'] == 'x'].index, inplace=True)
        df_buildings.reset_index(drop=True, inplace=True)
        # Add into 'df_buildings' relation to buses
        add_bus_relation(df_buildings, area, data_path)
        
        df_buildings.to_csv(f'{data_path}/df_buildings.csv', index=False)
    # Initialize population
    df_actions, df_citizens, df_vehicles, moving_agents = initialize_population(df_buildings, population_citizens, EV_percentage, main_path)
    
    print("[Completed]")
    return df_buildings, df_actions, df_citizens, df_vehicles, moving_agents

def initialize_dataframes():
    """Inicializa los DataFrames vacíos."""
    df_actions = pd.DataFrame(columns=['osmid', 'service', 'day', 'time', 'agent_name', 'action', 'agent_type', 'archetype', 'bus'])
    df_citizens = pd.DataFrame(columns=['agent_name', 'archetype', 'link_to', 'vehicle', 'Living', 'Working'])
    df_vehicles = pd.DataFrame(columns=['agent_name', 'archetype', 'SoC'])
    moving_agents = pd.DataFrame(columns=['agent_name', 'agent_type', 'osmid', 'link_to', 'day', 'time', 'archetype', 'SoC'])
    return df_actions, df_citizens, df_vehicles, moving_agents

def initialize_population(df_buildings, population, EV_percentage, main_path):
    print("    Initializing population")
    
    day, time = 0, 0
    living_buildings, working_buildings = get_building_lists(df_buildings)
    citizen_names, vehicles_names = generate_agent_names(population, EV_percentage)
    
    df_actions, df_citizens, df_vehicles, moving_agents = initialize_dataframes()

    # Procesar ciudadanos
    df_citizens, df_actions = process_citizens(citizen_names, living_buildings, working_buildings, vehicles_names, df_buildings, day, time, df_citizens, df_actions, main_path)

    # Procesar vehículos
    df_vehicles, df_actions = process_vehicles(vehicles_names, df_actions, df_buildings, day, time, df_vehicles, df_citizens)

    return df_actions, df_citizens, df_vehicles, moving_agents

def load_bus_data(directory, area):
    """Carga los datos de buses desde un archivo Excel."""
    filepath = os.path.join(directory, f"{area}_bus_data.xlsx")
    return pd.read_excel(filepath)

def process_citizens(citizen_names, living_buildings, working_buildings, vehicles_names, df_buildings, day, time, df_citizens, df_actions, main_path):
    """Asigna datos a los ciudadanos y actualiza los DataFrames."""
    citizens_data = []
    actions_data = []

    for citizen in tqdm(citizen_names, desc="       citizens: ", bar_format="{desc}{percentage:3.4f}%"):
        archetype = random.choices(['Worker', 'Student', 'Other'], [1, 0, 0])[0]
        living_on = random.choice(living_buildings)  # Aquí podría añadirse la lógica de saturación.
        working_on = random.choice(working_buildings)

        # Añadir ciudadano
        citizens_data.append({'agent_name': citizen, 'archetype': archetype, 'link_to': None, 'Living': living_on, 'Working': working_on})

        # Añadir acción
        etiquetas, valores = archetype_probability_movement(archetype, "00:00", main_path)
        intention = random.choices(etiquetas, weights=valores)[0]       
        osmid = living_on if intention == 'Living' else working_on
        bus = df_buildings.loc[df_buildings['osmid'] == osmid, 'bus'].iloc[0]
        actions_data.append({'osmid': osmid, 'service': intention, 'day': day, 'time': time, 
                             'agent_name': citizen, 'action': 'in', 'agent_type': 'citizen', 
                             'archetype': archetype, 'bus': bus})

    df_citizens = pd.DataFrame(citizens_data)
    df_citizens = assign_families_to_citizens(df_citizens)

    df_actions = pd.concat([df_actions, pd.DataFrame(actions_data)], ignore_index=True)
    
    return df_citizens, df_actions

def process_vehicles(vehicles_names, df_actions, df_buildings, day, time, df_vehicles, df_citizens):
    """Asigna datos a los vehículos eléctricos y actualiza los DataFrames."""
    vehicles_data = []
    actions_data = []

    # Obtener la lista de familias de los ciudadanos
    families = df_citizens['family'].unique()

    for vehicle in tqdm(vehicles_names, desc="       vehicles: ", bar_format="{desc}{percentage:3.4f}%"):
        archetype = random.choice(['012', '345', '6789'])
        SoC = 0  # State of Charge inicial

        # Asignar una familia aleatoria al vehículo
        family = random.choice(families)                                                #### en base estadistica por arquetipo!!!!

        # Añadir vehículo
        vehicles_data.append({'agent_name': vehicle, 'archetype': archetype, 'SoC': SoC, 'family': family})

        # Añadir acción
        osmid = random.choice(df_actions['osmid'])
        bus = df_buildings.loc[df_buildings['osmid'] == osmid, 'bus'].iloc[0]
        actions_data.append({'osmid': osmid, 'service': 'NaN', 'day': day, 'time': time, 
                             'agent_name': vehicle, 'action': 'in', 'agent_type': 'vehicle', 
                             'archetype': archetype, 'bus': bus})

    df_vehicles = pd.DataFrame(vehicles_data)
    df_actions = pd.concat([df_actions, pd.DataFrame(actions_data)], ignore_index=True)
    
    return df_vehicles, df_actions





