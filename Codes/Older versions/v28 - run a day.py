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
# [lost] v21   ->  se suma la funcion de creación de mapas de calor [FUNCION PERDIDA]
# v22   ->  primeros pasos para la incorporacion del modulo de comportamiento. Actualmente behaviour core hace cosas, pero no completa su toma de decisiones
# v23   ->  se a logrado incorporar el comportamiento asignado en los scv para dar a los agentes comportamientos más realistas
# v24   ->  heatmap
# v25   ->  he mejorado un poco lo del modulo de comportamiento, pero los valores estadisticos y los obtenidos por simulacion no cuadrán
# v26   ->  limpieza del codigo general
# v27   ->  inicializacion de ubicaciones en base estadistica
# v28   ->  Familias implementadas (tras asignar agentes a edificios, se toman estos y, por cada edificio, se agrupan en grupos 
#           de maximo 5 (porque sí). Cada uno de estos grupos constituye una 'familia' o unidad convivencial)

import os
import math
import random
import numpy as np
import osmnx as ox
import pandas as pd
import folium
from folium.plugins import HeatMap
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from joblib import Parallel, delayed
from geopy.distance import great_circle
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def load_bus_data(directory, area):
    """Carga los datos de buses desde un archivo Excel."""
    filepath = os.path.join(directory, f"{area}_bus_data.xlsx")
    return pd.read_excel(filepath)

def calculate_areas(vor):
    """Calcula las áreas de las regiones del diagrama de Voronoi."""
    areas = []
    for region_index in tqdm(range(len(vor.points)), desc="       ", bar_format="{desc}{percentage:3.4f}%"):
        indices_puntos_region = vor.regions[vor.point_region[region_index]]
        if -1 not in indices_puntos_region:
            region = vor.vertices[indices_puntos_region]
            areas.append(abs(np.cross(region[:-1], region[1:]).sum()) / 2)
        else:
            areas.append(np.inf)  # Región infinita
    return areas

def assign_buildings_to_buses(df_buildings, vor):
    """Asigna cada edificio a una región del diagrama de Voronoi y determina su bus correspondiente."""
    building_bus = {}
    otras_coordenadas = np.array(df_buildings['coord'])
    for i, coord in enumerate(otras_coordenadas):
        region_index = np.argmin(np.linalg.norm(vor.points - coord, axis=1))
        bus_name = f'bus_{region_index}'
        building_bus[i] = bus_name
    df_buildings['bus'] = pd.Series(building_bus)
    return df_buildings

def plot_voronoi_on_map(vor, area):
    """Genera un gráfico interactivo del diagrama de Voronoi sobre un mapa."""
    fig = go.Figure()
    
    # Agregar bordes de las regiones de Voronoi
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            fig.add_trace(go.Scattergeo(
                lon=[p[0] for p in polygon],
                lat=[p[1] for p in polygon],
                mode='lines',
                line=dict(width=1, color='olive'),
                fill='toself'
            ))
    
    fig.update_geos(fitbounds="locations")
    fig.update_layout(title=f"Diagrama de Voronoi sobre {area}")
    fig.show()

def add_bus_relation(df_buildings, area, directory):
    print("    Tagging all buildings to their buses.")
    
    # Cargar datos de buses y calcular el diagrama de Voronoi
    df_buses = load_bus_data(directory, area)
    coordenadas = np.column_stack((df_buses['long'], df_buses['lat']))
    vor = Voronoi(coordenadas)
    
    # Calcular áreas (si es necesario)
    areas = calculate_areas(vor)
    
    # Asignar edificios a buses
    df_buildings = assign_buildings_to_buses(df_buildings, vor)
    
    # Generar el gráfico interactivo
#    plot_voronoi_on_map(vor, area)
    
    return df_buildings

def initialize_population(df_buildings, population, EV_percentage):
    print("    Initializing population")
    
    day, time = 0, 0
    living_buildings, working_buildings = get_building_lists(df_buildings)
    citizen_names, vehicles_names = generate_agent_names(population, EV_percentage)
    
    df_actions, df_citizens, df_vehicles, moving_agents = initialize_dataframes()

    # Procesar ciudadanos
    df_citizens, df_actions = process_citizens(citizen_names, living_buildings, working_buildings, vehicles_names, df_buildings, day, time, df_citizens, df_actions)

    # Procesar vehículos
    df_vehicles, df_actions = process_vehicles(vehicles_names, df_actions, df_buildings, day, time, df_vehicles, df_citizens)

    return df_actions, df_citizens, df_vehicles, moving_agents

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

def generate_agent_names(population, EV_percentage):
    """Genera nombres para ciudadanos y vehículos eléctricos."""
    citizen_names = [f'citizen_{i}' for i in range(population)]
    vehicles_names = [f'elecvehi_{i}' for i in range(int(population * EV_percentage))]
    return citizen_names, vehicles_names

def initialize_dataframes():
    """Inicializa los DataFrames vacíos."""
    df_actions = pd.DataFrame(columns=['osmid', 'service', 'day', 'time', 'agent_name', 'action', 'agent_type', 'archetype', 'bus'])
    df_citizens = pd.DataFrame(columns=['agent_name', 'archetype', 'link_to', 'vehicle', 'Living', 'Working'])
    df_vehicles = pd.DataFrame(columns=['agent_name', 'archetype', 'SoC'])
    moving_agents = pd.DataFrame(columns=['agent_name', 'agent_type', 'osmid', 'link_to', 'day', 'time', 'archetype', 'SoC'])
    return df_actions, df_citizens, df_vehicles, moving_agents

def process_citizens(citizen_names, living_buildings, working_buildings, vehicles_names, df_buildings, day, time, df_citizens, df_actions):
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
        etiquetas, valores = archetype_probability_movement(archetype, "00:00")
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

# Modifies df_building adding a new column which describes the services available in the building
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
    building_to_services(directory, type_services, df_buildings)
    # Delete all buildings that offer no services
    df_buildings.drop(df_buildings[df_buildings['services'] == 'x'].index, inplace=True)
    df_buildings.reset_index(drop=True, inplace=True)
    # Add into 'df_buildings' relation to buses
    add_bus_relation(df_buildings, area, directory)
    # Initialize population
    df_actions, df_citizens, df_vehicles, moving_agents = initialize_population(df_buildings, population, EV_percentage)
    
    print("[Completed]")
    return df_buildings, df_actions, df_citizens, df_vehicles, moving_agents

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
    else:
        flag = 1
    
    return new_data.loc[0], flag

def behaviour_module(time, day, actions_data, df_citizens):
    # Take data from 'df_actions'
    agent_name = actions_data['agent_name'].iloc[0]  # Extracting the single agent name
    service_data = actions_data['service'].iloc[0]
    # Take data from 'df_citizens'
    citizen_data = df_citizens.loc[df_citizens['agent_name'] == agent_name].iloc[0]
    link_to = citizen_data['link_to']
    archetype_data = citizen_data['archetype']
    # Use behaviour-afecting variables to evaluate next action
    decision, intention  = behaviour_core(link_to, day, time, archetype_data, service_data)
    #if decision == "move" and time in [0, 1, 2, 3 ,4, 5, 6]:
        #print(f"move from {service_data} to {intention}")
    return decision, intention

def behaviour_core(link_to, day, time, archetype_data, service_data):
    # Calcular la hora a partir de los pasos y el tiempo
    instant = time / steps_captured * 24
    hour = f"{int(instant):02d}:{int((instant - int(instant)) * 60):02d}"
    
    etiquetas, valores = archetype_probability_movement(archetype_data, hour)   
    
    # Seleccionar una etiqueta de forma estadística
    intention = random.choices(etiquetas, weights=valores)[0]

    if intention == service_data and intention in ['Living', 'Working', 'Healthcare', 'Education']:
        decision = 'stay'
    else:
        decision = 'move'
    
    return decision, intention

def archetype_probability_movement(archetype, hour):
    # Directorio donde se encuentran los archivos CSV
    directorio_base = r"C:\Users\asier.divasson\Desktop\V2G-QUETS - Code\Behaviour Models"
    ruta_archivo = os.path.join(directorio_base, archetype + ".csv")
    
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

def create_heatmap(df_actions, df_buildings, area, directory):
    # Obtener las coordenadas de los edificios
    building_coords = dict(zip(df_buildings['osmid'], df_buildings['coord']))

    # Agrupar las acciones por 'time' y 'osmid'
    grouped_actions = df_actions.groupby(['time', 'osmid']).size().reset_index(name='count')

    # Iterar sobre cada hora para crear un mapa de calor
    for hour in grouped_actions['time'].unique():
        # Filtrar las acciones para la hora actual
        hour_actions = grouped_actions[grouped_actions['time'] == hour]

        # Crear una lista de coordenadas y sus respectivos conteos para la hora actual
        heatmap_data = []
        for _, row in hour_actions.iterrows():
            osmid = row['osmid']
            count = row['count']
            coord = building_coords.get(osmid)
            if coord:
                # Coord debe ser una tupla (lat, lon)
                heatmap_data.append([coord[0], coord[1], count])

        # Verificar si hay datos para el mapa de calor
        if not heatmap_data:
            print(f"No hay datos suficientes para generar el mapa de calor para la hora {hour}.")
            continue

        # Crear el DataFrame
        heatmap_df = pd.DataFrame(heatmap_data, columns=['lat', 'lon', 'count'])

        # Obtener el centro del área
        center_lat, center_lon = heatmap_df['lat'].mean(), heatmap_df['lon'].mean()

        # Crear el mapa
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Agregar el mapa de calor
        heat_data = [[row['lat'], row['lon'], row['count']] for index, row in heatmap_df.iterrows()]
        HeatMap(heat_data).add_to(m)

        # Guardar el mapa en un archivo HTML
        output_file = os.path.join(directory, f'{area}_heatmap_hour_{hour}.html')
        m.save(output_file)
        print(f"Mapa de calor para la hora {hour} guardado en {output_file}")

if __name__ == "__main__":
    ##################  SIMULATION EVALUATION STUFF  ################## 
    area = "Abando"
    population_citizens = 200
    steps_captured = 2
    days_captured = 1
    EV_percentage = 0.3
    ###################################################################
#   Definition of main variables
    directory = 'C:/Users/asier.divasson/Downloads'
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']
#   Initialization of stuff 
    df_buildings, df_actions, df_citizens, df_vehicles, moving_agents = initialization(area, population_citizens, EV_percentage, directory, type_services)
    time = 1 # Due to initialization has been made in time 0 day 0 (01/01/20XX, 00.00)
    day = 0
#   Simulation loop
    while day < days_captured:
        print('#'*22, 'day:', day+1, '#'*22)
        while time < steps_captured:
            instant = time/steps_captured*24
            hour = f"{int(instant):02d}:{int((instant-int(instant))*60):02d}"
            print('-'*20, 'time:', hour, '-'*20)
            for agent in tqdm(moving_agents['agent_name'], desc="Preprocessing: ", bar_format="{desc}{percentage:3.4f}%"):
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
            for agent in tqdm(df_citizens['agent_name'], desc="Processing: ", bar_format="{desc}{percentage:3.4f}%"):
                if agent not in moving_agents['agent_name'].values:
                    new_data_agent = data_gather(df_actions, agent)
                    decision, intention = behaviour_module(time, day, new_data_agent, df_citizens)
    #               print(decision, intention)
                    if decision == 'move':
                        new_data_agent['day'] = day
                        new_data_agent['time'] = time
                        new_data_agent['action'] = 'out'
                        df_actions = pd.concat([df_actions, new_data_agent], ignore_index=True)
                        new_data, flag = next_move(df_buildings, df_citizens,  new_data_agent, intention, steps_captured)
                        if flag != 1:
                            moving_agents = pd.concat([moving_agents, pd.DataFrame([new_data])], ignore_index=True)
                            # calcular consumos para SoC
            time += 1
        day += 1
        time = 0

    # Create Heatmap for better user experience
#    create_heatmap(df_actions, df_buildings, area, directory)

    # Print all data from DataFrames to Excels
    df_names = ['df_buildings', 'df_actions', 'df_citizens', 'df_vehicles', 'moving_agents']
    df_group = [df_buildings, df_actions, df_citizens, df_vehicles, moving_agents]
    for df in range(len(df_names)):
        df_a_excel(directory, df_group[df], df_names[df])
