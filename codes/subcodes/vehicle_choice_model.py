import os
import sys
from tqdm import tqdm
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
from pathlib import Path
from itertools import groupby
import matplotlib.pyplot as plt
import networkx as nx



def vehicle_choice_model(level_1_results, level_2_results, pop_transport, pop_citizen, paths, study_area, pop_archetypes_transport, pop_building, networks_map):
    try:
        past_dist_calculations = pd.read_excel(f"{paths['study_area']}/past_dist_calculations.xlsx")
    except Exception:
        past_dist_calculations = pd.DataFrame()
    
    # Agrupamos los resultados por familias
    level1_families = level_1_results.groupby(['family'])
    level2_families = level_2_results.groupby(['family'])
    transport_families = pop_transport.groupby(['family'])
    
    # Inicializamos la memoria de trips ya evaluados
    eval_trips = []
    
    # Inicializamos dfs de resultados
    new_level2_schedules = pd.DataFrame()
    vehicles_actions = pd.DataFrame()
    # Pasamos por los datos de cada una de las familias
    for f_name, family in tqdm(level2_families, desc="Procesando familias"):
        # Hacemos try pues cabe la posibilidad de que la familia no cuente con vehiculos, por lo que el get_group no pencontrará nada
        try:
            # Logramos los vehiculos asignados a cada miembro familiar
            avail_vehicles = transport_families.get_group(f_name)
        except Exception as e:
            avail_vehicles = pd.DataFrame()
        # Sacamos el schedule de level1 tambien
        level1_schedule = level1_families.get_group(f_name)
        # Sacamos los nombres de los agentes independientes
        independents = get_independents(level1_schedule)
        # Obtenemos el orden en el que iterar los independent (de route más a menos larga)
        independents = organize_independents(independents, family)
        # Agrupamos las actividades familiares por sus miembros
        level2_citizens = family.groupby(['agent'])
        level1_citizens = level1_schedule.groupby(['agent'])
        # Inicializamos el nuevo schedule
        new_family_schedule = pd.DataFrame()
        for c_name in independents:
            # Logramos el schedule especifico de este agente
            citizen_schedule = level2_citizens.get_group(c_name)
            # Sacamos sus datos de agente
            citizen_data = pop_citizen[pop_citizen['name'] == c_name].iloc[0]
            # Sacamos su ruta
            citizen_route = route_creation(citizen_schedule)
            # Eliminamos cualquier trip previamente considerado (no deberia pasar)
            citizen_route = [r for r in citizen_route if r not in eval_trips]
            # Añadimos los trips a la lista de trips evaluados
            eval_trips.append(citizen_route)
            # Calculamos la vehicle_score_matrix
            distime_matrix, past_dist_calculations = VSM_calculation(citizen_route, avail_vehicles, citizen_data, pop_archetypes_transport, pop_building, networks_map, past_dist_calculations)
            # Decidimos el transporte
            best_transport_distime_matrix = vehicle_chosing(distime_matrix)
            # Actualizamos el schedule 
            new_family_schedule = update_citizen_schedule(best_transport_distime_matrix, c_name, level1_schedule, family)
            # Lo sumamos al total
            new_level2_schedules = pd.concat([new_level2_schedules, new_family_schedule], ignore_index=True)
            # Sumamos el vehiculo al vehicles_actions
            vehicles_actions = update_vehicles_actions(vehicles_actions, new_family_schedule, best_transport_distime_matrix, avail_vehicles)
            # Actualizamos la lista de vehiculos disponibles, para que el resto no tomen este
            if not best_transport_distime_matrix['vehicle'].iloc[0] in ['walk', 'UB_diesel']:
                avail_vehicles.remove(best_transport_distime_matrix['vehicle'])
                
    vehicles_actions.to_excel(f"{paths['results']}/{study_area}_vehicles_actions.xlsx", index=False)
    new_level2_schedules.to_excel(f"{paths['results']}/{study_area}_new_level_2.xlsx", index=False)

    return vehicles_actions, new_level2_schedules

def main():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    
    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    file_management = system_management[['file_1', 'file_2', 'pre']]

    # Paso 2: Bucle sobre filas del mini DF
    for index, row in file_management.iterrows():
        
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        
        paths[file_2] = file_1 / file_2
        
        if not paths[file_2].exists():
            if row['pre'] == 'y':
                print(f"[Error] Critical file not detected:")
                print(f"{paths[file_2]}")
                print(f"Please solve the mentioned issue and reestart the model.")
                sys.exit()
            else:
                os.makedirs(paths[file_2], exist_ok=True)
                
    networks = ['drive', 'walk']
    networks_map = {}   
    for net_type in networks:           
        networks_map[net_type + "_map"] = ox.load_graphml(paths['maps'] / (net_type + '.graphml'))
    
    level_1_results = pd.read_excel(f"{paths['results']}/{study_area}_level_1.xlsx")
    level_2_results = pd.read_excel(f"{paths['results']}/{study_area}_level_2.xlsx")
    
    pop_citizen = pd.read_excel(f"{paths['population']}/pop_citizen.xlsx")
    pop_family = pd.read_excel(f"{paths['population']}/pop_family.xlsx")
    pop_building = pd.read_excel(f"{paths['population']}/pop_building.xlsx")
    pop_transport = pd.read_excel(f"{paths['population']}/pop_transport.xlsx")
    
    pop_archetypes_transport = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    
    ##############################################################################
    print(f'docs readed')

    # Mostrar el mapa
    ox.plot_graph(networks_map['drive_map'])
    
    
    vehicles_actions, new_level2_schedules = vehicle_choice_model(level_1_results, level_2_results, pop_transport, pop_citizen, paths, study_area, pop_archetypes_transport, pop_building, networks_map)
    
        
        
def update_citizen_schedule(best_transport_distime_matrix, c_name, level1_schedule, level2_families):
    """
    Bypassed:
    
    Deberiamos ver lo que hace el agente y cambiarle las entradas y salidas en base a 
    sus nuevos tiempos de conmutacion, diponibles en best_transport_distime_matrix
    """
    
    l2_citizen_schedule = level2_families[level2_families['agent'] == c_name].reset_index(drop=True)

    return l2_citizen_schedule

def update_vehicles_actions(vehicles_actions, new_family_schedule, best_transport_distime_matrix, avail_vehicles):
    # El objetivo es tener un df que de los datos de consumo relevante para cada actividad
    
    ''' Deshabilitado para poder evaluar el resto de funciones
    
    # Antes de iniciar, aseguramos que no sea walk o publico o walk_public
    if best_transport_distime_matrix['vehicle'].iloc[0] in ['walk', 'public']:
        # Devolvemos el df sin modificaciones
        return vehicles_actions
    '''
    
    # Simplificamos el 'new_family_schedule', para guardar la info como lo hariamos para el output
    simple_schedule = schedule_simplification(new_family_schedule)
    # Ahora duplicamos pero metemos el vehiculo en vez de la persona
    simple_schedule['agent'] = best_transport_distime_matrix['vehicle'].iloc[0]
    simple_schedule['archetype'] = best_transport_distime_matrix['archetype'].iloc[0]
    # Eliminamos las columnas innecesarias
    simple_schedule = simple_schedule.drop(['todo', 'todo_type', 'opening', 'closing', 'fixed', 'time2spend', 'conmutime'], axis=1)
    # Agregamos la nueva schedule a 'vehicles_actions'
    vehicles_actions = pd.concat([vehicles_actions, simple_schedule], ignore_index=True)
    # Devolvemos el df modificado
    return vehicles_actions    
    
def schedule_simplification(new_family_schedule):
    # Inicializamos el df de salida
    simple_schedule = pd.DataFrame()
    # Filtramos el df quitando las actividades 'not-time-related'
    filtered = new_family_schedule[abs(new_family_schedule['in']-new_family_schedule['out']) != 0]
    # Agrupamos por 'osm_id' para simplificar
    filtered_grouped = filtered.groupby('osm_id')
    # Evaluamos los grupos
    for name, group in filtered_grouped:
        # Si el grupo no tiene duplicados, no nos va a dar problemas
        if len(group) == 1:
            simple_schedule = pd.concat([simple_schedule, group], ignore_index=True)
            continue
        # Inicializamos 'def_in' (para tener como memoria del 'in' del grupo)
        def_in = float('inf')
        # Actualizamos los indices para evitar problematicas
        group_n = group.reset_index(drop=True)
        # En caso de tener más de una actividad 'time-related' en el mismo 'osm_id' 
        for idx, row in group_n.iterrows():
            # Nos saltamos el último row
            if idx == len(group_n)-1:
                continue
            # Evaluamos cuales estan concatenados
            if group_n.loc[idx, 'out'] == group_n.loc[idx+1, 'in']:
                # Asignamos las variables
                def_in = group_n.loc[idx, 'in']
                def_out = group_n.loc[idx+1, 'out']
        # Si no se han identificado concatenaciones
        if def_in == float('inf'):
            simple_schedule = pd.concat([simple_schedule, group_n], ignore_index=True)
            continue
        # Copiamos los datos del grupo
        new_row = group_n.iloc[0].copy()
        # Añadimos los nuevos datos
        new_row['in'] = def_in
        new_row['out'] = def_out
        # Anadimos la nueva fila al schedule simplificado
        simple_schedule = pd.concat([simple_schedule, pd.DataFrame([new_row])], ignore_index=True)
    return simple_schedule.sort_values(by='in', ascending=True).reset_index(drop=True)

def vehicle_chosing(vehicle_score_matrix):   
    # Sumamos los scores por transporte
    simplified_df = vehicle_score_matrix.groupby('vehicle', as_index=False)['score'].sum()
    # Sacamos el transporte con menos score
    best_transport = simplified_df.loc[simplified_df['score'].idxmin()]
    # Sacamos del original walk y public transport
    public_walk = vehicle_score_matrix[vehicle_score_matrix['vehicle'].isin(['walk', 'UB_diesel'])]
    # Índices de filas con mínimo score por trip
    idx = public_walk.groupby('trip')['score'].idxmin() 
    # Sacamos el score de elegir el minimo entre walk y public por cada trip (model split)
    score_public_walk = public_walk.loc[idx, ['trip', 'score']].reset_index(drop=True)
    # Evaluamos se el modal split es mejor que lo previo    
    if score_public_walk['score'].sum() < best_transport['score']:
        # Si es mejor, devolvemos esto como resultado
        return public_walk.loc[idx]
    # Si no es mejor, devuelve el anteriormente definido como mejor
    return vehicle_score_matrix[vehicle_score_matrix['vehicle'] == best_transport['vehicle']].reset_index(drop=True)

def VSM_calculation(citizen_route, avail_vehicles, citizen_data, pop_archetypes_transport, pop_building, networks_map, past_dist_calculations):
    # Inicializamos la vehicle_score_matrix
    vehicle_score_matrix = pd.DataFrame()
    # Inicializamos la full_distime_matrix
    full_distime_matrix = pd.DataFrame()
    # Añadimos a la matriz de vehiculos disponibles el publico y andar
    avail_transport = add_public_walk(avail_vehicles, citizen_data, pop_archetypes_transport)
    # Iteramos los distintos transportes disponibles
    for _, transport in avail_transport.iterrows():
        # Inicializamos transport_VSM
        transport_VSM = pd.DataFrame()
        # Inicializamos last_P (determina la posición donde se encontro por última vez el vehicle)
        last_P = transport['ubication']
        # Miramos todos los trips, de uno a uno, actualizando el transport_VSM (el VSM especifico para este medio de transporte)
        for trip in citizen_route:
            # Calculamos el score de este trip
            distime_matrix, last_P, past_dist_calculations = score_calculation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map, past_dist_calculations, citizen_data)
            # Añadimos info relevante
            distime_matrix['citizen'] = citizen_data['name']
            distime_matrix['vehicle'] = transport['name']
            # La añadimos al df de resultados
            transport_VSM = pd.concat([transport_VSM, pd.DataFrame([distime_matrix])], ignore_index=True)
            full_distime_matrix = pd.concat([full_distime_matrix, pd.DataFrame([distime_matrix])], ignore_index=True)
        # Añadimos el nuevo transport_VSM a vehicle_score_matrix
        vehicle_score_matrix = pd.concat([vehicle_score_matrix, transport_VSM], ignore_index=True)

    return full_distime_matrix, past_dist_calculations
            
def score_calculation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map, past_dist_calculations, citizen_data):
    # Completamos el trip en caso de que tenga que acudir a algún punto P
    complete_trip, last_P, past_dist_calculations = trip_completation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map, past_dist_calculations)
    # Calculamos la matriz de distime
    distime_matrix, past_dist_calculations = distime_calculation(networks_map, complete_trip, past_dist_calculations, pop_building, citizen_data, transport)
    # Sacamos el score en base al algoritmo especificado para ello
    distime_matrix = score_algorithm(distime_matrix)
    # Devolvemos score
    return distime_matrix, last_P, past_dist_calculations

def score_algorithm(distime_matrix):
    """
    Aqui definimos el algoritmo de toma de decisiones
    """
    
    distime_matrix = distime_matrix.iloc[0]
    # Asegurarse de que distime_matrix está desconectado de slices
    distime_matrix = distime_matrix.copy()
    # Calculamos el conmu_time
    conmu_time = distime_matrix['walk_time'] + distime_matrix['travel_time'] + distime_matrix['wait_time']
    # Calcular score de forma robusta
    distime_matrix['score'] = (
        conmu_time +
        distime_matrix['cost'] -
        distime_matrix['benefits'] +
        distime_matrix['emissions']
    )
    
    return distime_matrix


def distime_calculation(networks_map, complete_trip, past_dist_calculations, pop_building, citizen_data, transport):
    # Inicializamos el df de salida
    distime_matrix = pd.DataFrame()
    # Sacamos el trip
    trip = (complete_trip[0][0], complete_trip[-1][-1])
    # Inicializamos la variable map_type
    map_type = 'drive'
    for step_0, step_1 in complete_trip:
        if map_type == 'walk':
            map_type = 'drive'
        else:
            map_type = 'walk'            
            
        # Miramos si ya lo hemos calculado antes
        previously = past_dist_calculations[(past_dist_calculations['step'] == (step_0, step_1)) & (past_dist_calculations['map'] == map_type)]
        
        # Si no lo hemos hehco, lo hacemos ahora
        if previously.empty:
            lon_0, lat_0 = pop_building[pop_building['osm_id'] == step_0][['lon', 'lat']].iloc[0]
            lon_1, lat_1 = pop_building[pop_building['osm_id'] == step_1][['lon', 'lat']].iloc[0]
            
            # Encuentra el nodo más cercano en el mapa
            node_0 = ox.distance.nearest_nodes(networks_map[f'{map_type}_map'], lon_0, lat_0)
            node_1 = ox.distance.nearest_nodes(networks_map[f'{map_type}_map'], lon_1, lat_1)
            # Encuentra la ruta más corta entre poi_node y P_node
            route = ox.shortest_path(networks_map[f'{map_type}_map'], node_0, node_1, weight='length')
            
            if route is None:
                print(f"step_0: {step_0}, step_1: {step_1}")
                print(f"node_0: {node_0}, node_1: {node_1}")
                if node_0 not in networks_map[f'{map_type}_map'].nodes:
                    input("node_0 no existe")
                if node_1 not in networks_map[f'{map_type}_map'].nodes:
                    input("node_1 no existe")
                if not nx.has_path(networks_map[f'{map_type}_map'], node_0, node_1):
                    print("No hay camino entre los nodos")
                    G = networks_map[f'{map_type}_map']

                    pos = nx.get_node_attributes(G, 'pos')  # Asegúrate de que los nodos tengan atributo 'pos' (x, y)

                    plt.figure(figsize=(10, 10))
                    nx.draw(G, pos, node_size=10, node_color='lightgray', edge_color='gray', alpha=0.5)

                    # Dibujar node_0 en rojo y node_1 en azul
                    if node_0 in pos:
                        nx.draw_networkx_nodes(G, pos, nodelist=[node_0], node_color='red', node_size=100, label='node_0')
                    else:
                        print("Advertencia: node_0 no tiene posición asignada")

                    if node_1 in pos:
                        nx.draw_networkx_nodes(G, pos, nodelist=[node_1], node_color='blue', node_size=100, label='node_1')
                    else:
                        print("Advertencia: node_1 no tiene posición asignada")

                    plt.legend()
                    plt.title(f"Mapa sin camino entre {node_0} y {node_1}")
                    plt.axis("equal")
                    plt.show()
                    
                    
                    
                    
                input(f"node_0: {node_0}, node_1: {node_1}")
                
                # Maneja el error: puede ser que no haya camino disponible
                distance = float('inf')  # o 0, o algún valor sentinela, según el contexto
            else:
                distance = nx.path_weight(networks_map[f'{map_type}_map'], route, weight="length") / 1000
            
            
            distance = nx.path_weight(networks_map[f'{map_type}_map'], route, weight="length")/1000
            
            past_dist_calculations = pd.concat([past_dist_calculations, pd.DataFrame([{'step': (step_0, step_1), 'map': map_type, 'km': distance}])], ignore_index=True)
        # Si ya lo hemos hehco, lo copiamos
        else:
            distance = previously['km'].iloc[0]

        waiting_time = waiting_time_calculation(distance, step_1, transport)
        benefits = benefits_calculation(citizen_data, step_1)
        
        # Creamos nueva fila
        new_row = {
            'citizen': citizen_data['name'],
            'vehicle': transport['name'],
            'archetype': transport['archetype'],
            'trip': trip,
            'distance': distance,
            'walk_time': (citizen_data['walk_speed']/distance) if map_type == 'walk' and distance > 0 else 0,
            'travel_time': transport['v']/distance if map_type == 'drive'and distance > 0 else 0,
            'wait_time': waiting_time,
            'cost': transport['Ekm']*distance if map_type == 'drive'and distance > 0 else 0,
            'benefits': benefits, 
            'emissions': transport['CO2km']*distance if map_type == 'drive'and distance > 0 else 0,
            }
        
        distime_matrix = pd.concat([distime_matrix, pd.DataFrame([new_row])], ignore_index=True)
        # Sumar solo la parte numérica
        summed_numeric = distime_matrix.select_dtypes(include='number').sum().to_frame().T
        # Añadir columnas no numéricas de forma segura
        summed_df = summed_numeric.assign(
            citizen=distime_matrix.iloc[0]['citizen'],   
            vehicle=distime_matrix.iloc[0]['vehicle'],
            archetype=distime_matrix.iloc[0]['archetype'],
            trip=[distime_matrix.iloc[0]['trip']],
        ) 
    return summed_df, past_dist_calculations

def waiting_time_calculation(distance, step_1, transport):
    
    """
    Solo si es public y walk. si lo es que mire walk_time y luego que mire la hora de 
    salida original del agente y que calcule cuando llegaria a PP y de hay el tiempo hasta un muntipo de lo que sea
    """
    
    return 0

def benefits_calculation(citizen_data, step_1):
    """
    Con el nombre del agenet, puedo mirar en family_schedule el tiempo que este agente pretende estar en el step_1
    """
    
    return 0

def trip_completation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map, past_dist_calculations):
    # Sacamos los valores de inicio y fin del trip
    poi_A, poi_B = trip
    # Inicializamos la lista de nueva ruta (empieza con poi_A ya metido, porque eso es así fijo)
    steps = [poi_A]
    # Sacamos de los datos del arquetipo sus valores de P1P2
    p1, p2, p1s, p2s = pop_archetypes_transport[pop_archetypes_transport['name']==transport['archetype']][['P1', 'P2', 'P1s', 'P2s']].values[0]
    if p1 != 0:
        steps.append(last_P)
    if p2 != 0:
        p2_osm_id, p2_poib_dist = find_p2(poi_B, transport, p2s, pop_building, networks_map)
        steps.append(p2_osm_id)
        past_dist_calculations = pd.concat([past_dist_calculations, pd.DataFrame([{'step': (p2_osm_id, poi_B), 'map': 'walk', 'km': p2_poib_dist}])], ignore_index=True)
    else:
        p2_osm_id = poi_B
    # Por último añadimos la meta
    steps.append(poi_B)
    # Inicializamos la lista de resultados
    complete_trip = []
    # Adaptamos para que sea más comoda de usar
    for step in range(len(steps)-1):
        complete_trip.append((steps[step], steps[step+1]))    
    
    return complete_trip, p2_osm_id, past_dist_calculations

def find_p2(poi_B, transport, p2s, pop_building, networks_map):
    # Sacamos los datos de los posibles P
    available_P = pop_building[pop_building['archetype'] == p2s].copy()
    # En caso de que no se detecte ningun POI available para actuar como P, se notifica al usuario #ISSUE 32
    if available_P.empty:
        print(f"Ha ocurrido un error. No se ha detectado el servicio '{p2s}' entre los disponibles en pop_building.")
        print(f"Esto puede deberse a dos razones:")
        print(f"    1. No hay ningun POI con esta etiqueta en el territorio de analisis.")
        print(f"    2. Al generar la poblacion 'pop_archetypes_building' tenia la opción del servicio '{p2s}' deshabilitado.")
        input(f"Revisa ambas opciones y vuelve a correr el codigo. Gracias.")
    
    # Sacamos los datos del POI B
    poi_B_data = pop_building[pop_building['osm_id'] == poi_B].copy()
    # Cálculo vectorizado de distancia entre un punto fijo (poi_B_data) y todos los puntos en available_P
    available_P.loc[:, 'distance'] = haversine(poi_B_data['lat'].iloc[0], poi_B_data['lon'].iloc[0], available_P['lat'].values, available_P['lon'].values)
    # Ordenamos de mas cerca a mas lejos
    available_P = available_P.sort_values(by='distance', ascending=True) # ISSUE 30
    # Calculamos de distancia real y sacamos el mejor osm_id
    p2, p2_poib_dist = find_closest_service(poi_B_data, available_P, networks_map)
    # Devolvemos el osm_id y la distancia del menor valor
    return p2, p2_poib_dist

def find_closest_service(poi_data, feasible_services, networks_map):
    # Encontrar el nodo más cercano al POI
    poi_node = ox.distance.nearest_nodes(networks_map['walk_map'], poi_data['lon'].iloc[0], poi_data['lat'].iloc[0])
    # Cálculo de distancia real entre un punto fijo (nodo B) y todos los puntos en available_nodes
    results = real_dist_calulation(networks_map, poi_node, feasible_services)
    # Sacamos el osm_id más cercano y la distancia en km
    return results['osm_id'].iloc[0], results['km'].iloc[0]

def real_dist_calulation(networks_map, poi_node, feasible_services):
    # Inicializamos del df de resultados
    results = pd.DataFrame()
    # Inicializamos una pequeña variable de memoria
    min_distance = float('inf')
    # Iniciamos el counter (lo usamos para reducir los calculos realizados)
    counter = 0
    # Miramos todas las filas del feasible_services
    for _, fs_row in feasible_services.iterrows():
        # Encuentra el nodo más cercano en el mapa
        P_node = ox.distance.nearest_nodes(networks_map['walk_map'], fs_row['lon'], fs_row['lat'])
        # Encuentra la ruta más corta entre poi_node y P_node
        route = ox.shortest_path(networks_map['walk_map'], poi_node, P_node, weight='length')
        distance = nx.path_weight(networks_map['walk_map'], route, weight="length")
        # Creamos nueva fila
        new_row = {
            'osm_id': fs_row['osm_id'],
            'km': distance/1000,
        }
        # Se la agregamos al df de resultados
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        # Miramos si hemos obtenido mejor que la última vez
        if distance/1000 >= min_distance:
            # Si no es asi, aumentamos el counter
            counter += 1
        else:
            # Si es asi, reiniciamos el counter
            counter = 0
            # Actualizamos la variable
            min_distance = distance/1000
        # Al llegas a 5 iteraciones iguales, se acaba
        if counter >= 5:
            break
    # Devolvemos los datos ordenados de menos a más
    return results.sort_values(by='km', ascending=True)   

# Función de distancia haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c       

def add_public_walk(avail_vehicles, citizen_data, pop_archetypes_transport):
    ## Walk
    # Creamos la nueva fila
    new_row = {
        'name': 'walk',
        'archetype': 'walk',
        'family': citizen_data['family'],
        'ubication': citizen_data['Home'],
        'v': citizen_data['walk_speed'], # No es del citizen, si no del mas lento del grupo!!!! ISSUE 28
        'Ekm': 0,
        'mjkm': 0,
        'Emin': 0,
        'CO2km': 0,
        'SoC': 0,
    }
    # La añadimos al df de resultados
    avail_vehicles = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    ## Public transport
    # Sacamos las caracteristicas con valores estadisticos
    variables = ['v', 'Ekm', 'mjkm', 'Emin', 'CO2km', 'SoC']
    values = get_vehicle_stats('UB_diesel', pop_archetypes_transport, variables)
    
    # Creamos la nueva fila
    new_row = {
        'name': 'UB_diesel',
        'archetype': 'UB_diesel',
        'family': citizen_data['family'],
        'ubication': citizen_data['Home'], # ISSUE 29 esto deberia ser la parada de publico más cercana
        'v': values['v'],
        'Ekm': values['Ekm'],
        'mjkm': values['mjkm'],
        'Emin': values['Emin'],
        'CO2km': values['CO2km'],
        'SoC': values['SoC'],
    }
    # La añadimos al df de resultados
    avail_vehicles = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    # Devolvemos la version actualizada
    return avail_vehicles

def route_creation(citizen_schedule):
    # Inicializamos la lista de resultados
    route = []
    # Sacamos los osm_id en los que actua
    osm_id_route = citizen_schedule['osm_id']
    simpl_route = [k for k, _ in groupby(osm_id_route)]
    # Lo organizamos por trips
    for idx in range(len(simpl_route)):
        # Si es el último osm_id de la lista, no calcula nada
        if idx == len(simpl_route)-1:
            continue
        # Trabajamos con duplas
        route.append((simpl_route[idx], simpl_route[idx+1]))
    # Devolvemos la lista preparada
    return route
    
def get_independents(level1_schedule):
    return level1_schedule[(level1_schedule['todo_type'] == 0) & (level1_schedule['todo'] == 'WoS')]['agent'].unique()
    
def organize_independents(independents, family):
    # Inicializamos routes_data
    routes_data = pd.DataFrame()
    for agent in independents:
        # Sacamos la rute del agente
        new_route = family[family['agent'] == agent]['osm_id'].unique()
        # Creamos la nueva fila para routes_data
        new_row = {
            'agent': agent,
            'route': new_route,
            'len': len(new_route)
        }
        # Lo añadimos a routes_data
        routes_data = pd.concat([routes_data, pd.DataFrame([new_row])], ignore_index=True)
    # Ordenamos para tener el que más trips tenga el primero
    routes_data.sort_values(by='len', ascending=False)
    # Devolvemos solo la lista de los nombres
    return routes_data['agent'].tolist()
    
def get_vehicle_stats(archetype, transport_archetypes, variables):
    results = {}   
    
    # Filtrar la fila correspondiente al arquetipo
    row = transport_archetypes[transport_archetypes['name'] == archetype]
    if row.empty:
        print(f"Strange mistake happend")
        print(archetype)
        input(transport_archetypes)
        return {}

    row = row.iloc[0]  # Extrae la primera (y única esperada) fila como Series

    for variable in variables:
        mu = float(row[f'{variable}_mu'])
        sigma = float(row[f'{variable}_sigma'])
        try:
            max_var = float(row[f'{variable}_max'])
        except Exception as e:
            max_var = float('inf')
        try:
            min_var = float(row[f'{variable}_min'])
        except Exception as e:
            min_var = float(0)
        
        var_result = np.random.normal(mu, sigma)
        var_result = max(min(var_result, max_var), min_var)
        results[variable] = var_result

    return results

# Ejecución
if __name__ == '__main__':
    main()