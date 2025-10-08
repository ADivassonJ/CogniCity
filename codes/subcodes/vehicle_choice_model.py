import os
import sys
import math
from tqdm import tqdm
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
from pathlib import Path
from itertools import groupby
import matplotlib.pyplot as plt
import networkx as nx
import networkx as nx
import osmnx as ox
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed   # o ThreadPoolExecutor si es I/O-bound
from functools import partial
import pandas as pd
from tqdm import tqdm
import os

# ------- WORKER: procesa una familia completa -------
def _process_family(
    family_tuple,
    transport_families_dict,
    pop_citizen,
    pop_archetypes_transport,
    pop_building,
    networks_map
):
    f_name, family = family_tuple

    # Vehículos disponibles para esta familia (o vacío)
    avail_vehicles = transport_families_dict.get(f_name, pd.DataFrame())
    
    # Schedule level-1 de esta familia
    level1_schedule = family  # ya es el subset

    # Agrupar por ciudadano
    level1_citizens = level1_schedule.groupby('agent')
    family_members = family['agent'].unique()

    all_citizen_schedule = pd.DataFrame()
    all_vehicle_schedule = pd.DataFrame()

    for c_name in family_members:
        citizen_todolist = level1_citizens.get_group(c_name).sort_values(by='trip', ascending=True).reset_index(drop=True).copy()
        citizen_data = pop_citizen.loc[pop_citizen['name'] == c_name].iloc[0]
        
        # If 'citizen_todolist' only has one row, mean they did not left home, so no vehicle moving nether
        if len(citizen_todolist) == 1:
            continue
        
        # Ruta del ciudadano
        citizen_route = route_creation(citizen_todolist)
        
        # Calcula matriz dist/tiempo y actualiza cache local
        distime_matrix = VSM_calculation(
            citizen_route,
            avail_vehicles,
            citizen_data,
            pop_archetypes_transport,
            pop_building,
            networks_map
        )
        
        # Escoge vehículo
        best_transport_distime_matrix = vehicle_chosing(distime_matrix)
        
        # Actualiza schedule de la familia con la elección
        citizen_schedule = create_citizen_schedule(best_transport_distime_matrix, c_name, family)
        all_citizen_schedule = pd.concat([all_citizen_schedule, citizen_schedule], ignore_index=True)
        
        # Actualiza acciones de vehículos
        vehicle_schedule = create_vehicles_actions(citizen_schedule, best_transport_distime_matrix)
        all_vehicle_schedule = pd.concat([all_vehicle_schedule, vehicle_schedule], ignore_index=True)
        
        # “Consume” vehículo si no es compartible
        vehicle_name = best_transport_distime_matrix['vehicle'].iloc[0]
        if vehicle_name not in ('walk', 'UB_diesel') and not avail_vehicles.empty:
            avail_vehicles = avail_vehicles[avail_vehicles['name'] != vehicle_name]
    
    # aqui tenemos que añadir cualquier vehiculo no utilizado :)
    
    # sacamos los datos de family
    filtered = family[family['todo'].str.contains(r'Home|Home_in|Home_out', na=False)]
    
    family_home = filtered.iloc[0]['osm_id']
    family_node = filtered.iloc[0]['node']
    family_archetype = filtered.iloc[0]['family_archetype']

    for _, vehicle in avail_vehicles.iterrows():
      
        new_row = [{
            'agent': vehicle['name'],
            'archetype': vehicle['archetype'],
            'independent': 0,
            'osm_id': family_home,
            'node': family_node,
            'family': f_name,
            'family_archetype': family_archetype,
            'trip': 0,
            'in': 0,
            'out': 24*60,
            'user': None,
            'ETC [kWh]': 0,
        }]      
        
        all_vehicle_schedule = pd.concat([all_vehicle_schedule, pd.DataFrame(new_row)], ignore_index=True)

    return all_citizen_schedule, all_vehicle_schedule


def vehicle_choice_model(
    level_1_results,
    pop_transport,
    pop_citizen,
    paths,
    study_area,
    pop_archetypes_transport,
    pop_building,
    networks_map,
    day,
    n_jobs=None,        # None -> CPUs lógicas
    use_threads=False   # True si tu carga es I/O (lee/escribe mucho / red)
):
    # --- Agrupar por familia una sola vez ---
    level1_families = level_1_results.groupby('family')
    families = list(level1_families)  # [(f_name, df_sub)]
    transport_families = pop_transport.groupby('family') if not pop_transport.empty else None

    # Para acceso rápido desde procesos, convertir a dict {family_name: df}
    transport_families_dict = {}
    if transport_families is not None:
        for fam_name, fam_df in transport_families:
            transport_families_dict[fam_name] = fam_df 

    # --- Elige ejecutor ---
    Executor = ProcessPoolExecutor
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor
        Executor = ThreadPoolExecutor

    # --- Lanzar en paralelo por familia ---
    results_schedules = []
    results_actions = []
    cache_deltas = []
        
    
    '''for fam_tuple in families:
        fam_schedule, fam_actions= _process_family(fam_tuple,
                                                    transport_families_dict,
                                                    pop_citizen,
                                                    pop_archetypes_transport,
                                                    pop_building,
                                                    networks_map)'''
        
        
    worker = partial(
        _process_family,
        transport_families_dict=transport_families_dict,
        pop_citizen=pop_citizen,
        pop_archetypes_transport=pop_archetypes_transport,
        pop_building=pop_building,
        networks_map=networks_map,
    )
    
    with Executor(max_workers=n_jobs) as ex:
        futures = {ex.submit(worker, fam_tuple): fam_tuple[0] for fam_tuple in families}
        for fut in tqdm(as_completed(futures), total=len(families), desc=f"Transport Choice Modelling ({day})"):
            fam_name = futures[fut]
            try:
                fam_schedule, fam_actions = fut.result()
                if fam_schedule is not None and not fam_schedule.empty:
                    results_schedules.append(fam_schedule)
                if fam_actions is not None and not fam_actions.empty:
                    results_actions.append(fam_actions)
            except Exception as e:
                print(f"[ERROR] familia '{fam_name}': {e}")

    # --- Agregación en el proceso principal ---
    new_level1_schedules = pd.concat(results_schedules, ignore_index=True) if results_schedules else pd.DataFrame()
    vehicles_actions     = pd.concat(results_actions,   ignore_index=True) if results_actions   else pd.DataFrame()

    # --- Escrituras UNA sola vez ---
    out_actions = os.path.join(paths['results'], f"{study_area}_{day}_vehicles.xlsx")
    out_level1  = os.path.join(paths['results'], f"{study_area}_{day}_schedule.xlsx")
    vehicles_actions.to_excel(out_actions, index=False)
    new_level1_schedules.to_excel(out_level1, index=False)

    return vehicles_actions, new_level1_schedules

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
    
    pop_citizen = pd.read_parquet(f"{paths['population']}/pop_citizen.parquet")
    pop_family = pd.read_parquet(f"{paths['population']}/pop_family.parquet")
    pop_building = pd.read_parquet(f"{paths['population']}/pop_building.parquet")
    pop_transport = pd.read_parquet(f"{paths['population']}/pop_transport.parquet")
    
    pop_archetypes_transport = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    # Días que quieres considerar
    days = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}

    found_schedule = set()
    found_vehicles = set()
    found_todolist = set()

    for file in paths['results'].glob('*.xlsx'):
        name = file.stem  # sin extensión
        parts = name.split('_')
        if len(parts) < 3:
            continue
        # asumiendo formato: study_area_day_kind
        study, day, kind = parts[-3], parts[-2], parts[-1].lower()
        if day not in days:
            continue
        if kind == 'schedule':
            found_schedule.add(day)
        elif kind == 'vehicles':
            found_vehicles.add(day)
        elif kind == 'todolist':
            found_todolist.add(day)

    # Faltantes por tipo
    missing_schedule = days - found_schedule
    missing_vehicles = days - found_vehicles

    # Faltantes en general (en al menos uno)
    days_missing_schedules = missing_schedule | missing_vehicles
    
    # In case of having days to model
    if days_missing_schedules:
        # We act on each different day
        for day in days_missing_schedules:
            # Input reading
            level_1_results = pd.read_excel(f"{paths['results']}/{study_area}_{day}_todolist.xlsx")
            # Vehicle Choice Modeling
            vehicles_actions, new_level2_schedules = vehicle_choice_model(level_1_results, pop_transport, pop_citizen, paths, study_area, pop_archetypes_transport, pop_building, networks_map, day)
    else:
        print(f"All days' schedules already modeled.")
        
def create_citizen_schedule(best_transport_distime_matrix, c_name, todo_list_family):
    """
    Actualiza los tiempos de entrada/salida ('in'/'out') del agente `c_name`
    en base a los nuevos tiempos de conmutación contenidos en
    `best_transport_distime_matrix['conmu_time']`.

    Requisitos de columnas:
      - en `todo_list_family`: ['agent', 'trip', 'todo', 'opening', 'closing', 'time2spend']
      - en `best_transport_distime_matrix`: ['conmu_time']
    """
    
    # Filtrar y ordenar la lista del agente
    todo_list = (
        todo_list_family[todo_list_family['agent'] == c_name]
        .sort_values(by='trip', ascending=True)
        .reset_index(drop=True)
        .copy()
    )

    # Asegurar mismo largo (simple y directo)
    # Se usará el índice de la iteración para tomar el conmu_time
    commute = best_transport_distime_matrix.reset_index(drop=True)['conmu_time']
    
    out_time = None  # se actualizará en el bucle

    for idx, row in todo_list.iterrows():
        if row['todo'] == 'Home_out':
            # Sale de casa con antelación igual a la conmutación hasta la primera actividad
            in_time = row['opening']
            out_time = todo_list.iloc[idx+1]['opening'] - commute.iloc[idx]
            todo_list.loc[idx, 'in'] = in_time
            todo_list.loc[idx, 'out'] = out_time
            continue
        
        # tiempo de conmutación para este paso (si falta, tomamos 0)
        conmu_time = commute.iloc[idx-1]

        # Para el resto, llega tras la conmutación desde el punto anterior
        if out_time is None:
            # Si por lo que sea no pasó por 'Home_out', asumimos que parte en 'opening'
            out_time = row['opening']

        in_time = out_time + conmu_time
        out_time = (in_time + row['time2spend']) if row['time2spend'] != 0 else row['closing']

        todo_list.loc[idx, 'in'] = in_time
        todo_list.loc[idx, 'out'] = out_time
    
    todo_list['in'] = todo_list['in'].astype(int)
    todo_list['out'] = todo_list['out'].astype(int)
    
    citizen_schedule = todo_list.copy()
    
    return citizen_schedule


def create_vehicles_actions(new_family_schedule, best_transport_distime_matrix):
    # El objetivo es tener un df que de los datos de consumo relevante para cada actividad
       
    # Antes de iniciar, aseguramos que no sea walk o publico o walk_public
    if best_transport_distime_matrix['vehicle'].iloc[0] in ['walk', 'public']:
        # Devolvemos el df sin modificaciones
        return pd.DataFrame()
    
    # Simplificamos el 'new_family_schedule', para guardar la info como lo hariamos para el output
    simple_schedule = schedule_simplification(new_family_schedule.sort_values(by='trip', ascending=True).reset_index(drop=True).copy())
    
    # Ahora duplicamos pero metemos el vehiculo en vez de la persona
    simple_schedule['user'] = simple_schedule['agent']
    simple_schedule['agent'] = best_transport_distime_matrix['vehicle'].iloc[0]
    simple_schedule['archetype'] = best_transport_distime_matrix['archetype'].iloc[0]
    
    for idx, row in simple_schedule.iterrows():
        if idx == 0:
            simple_schedule.at[idx, 'ETC [kWh]'] = 0
            continue
        simple_schedule.at[idx, 'ETC [kWh]'] = best_transport_distime_matrix.at[idx-1,'mjkm'] + simple_schedule.at[idx-1, 'ETC [kWh]']

    # Eliminamos las columnas innecesarias
    simple_schedule = simple_schedule.drop(['todo', 'opening', 'closing', 'fixed', 'time2spend'], axis=1)
    
    return simple_schedule    
    
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
    simplified_df = vehicle_score_matrix.groupby('vehicle', as_index=False).sum()
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
        current_transport = public_walk.loc[idx]
    else:
        # Si no es mejor, devuelve el anteriormente definido como mejor
        current_transport = vehicle_score_matrix[vehicle_score_matrix['vehicle'] == best_transport['vehicle']].reset_index(drop=True)
    
    ####################################################
    # Aqui es donde debemos meter el modelo de Qiaochu #
    ####################################################
    '''
    Basicamente, su modelo, mira el transporte actual y evalua la probabilidad de que cambie del que tenia a 
    uno más V2G-related. Por lo tanto, y a menos de que me exprese alguna modificacion. Habría que añadir su
    modulo aqui y considerar el current_transport como el actual y evaluar si cambia o no.
    
    Parece que no se concibe la opcion de que el current sea V2G, por lo que
    
    '''
    
    return current_transport

def VSM_calculation(citizen_route, avail_vehicles, citizen_data, pop_archetypes_transport, pop_building, networks_map):
    # Inicializamos la vehicle_score_matrix
    vehicle_score_matrix = pd.DataFrame()
    # Inicializamos la full_distime_matrix
    full_distime_matrix = pd.DataFrame()
    
    if citizen_data['independent_type'] == 0:
        vehicles = pd.DataFrame()
    else:
        vehicles = avail_vehicles
    
    # Añadimos a la matriz de vehiculos disponibles el publico y andar
    avail_transport = add_public_walk(vehicles, citizen_data, pop_archetypes_transport)
    # Iteramos los distintos transportes disponibles
    for _, transport in avail_transport.iterrows():
        # Inicializamos transport_VSM
        transport_VSM = pd.DataFrame()
        # Inicializamos last_P (determina la posición donde se encontro por última vez el vehicle)
        last_P = transport['ubication']
        # Miramos todos los trips, de uno a uno, actualizando el transport_VSM (el VSM especifico para este medio de transporte)
        for trip in citizen_route:
            # Calculamos el score de este trip
            distime_matrix, last_P = score_calculation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map, citizen_data)
            # Añadimos info relevante
            distime_matrix['citizen'] = citizen_data['name']
            distime_matrix['vehicle'] = transport['name']
            # La añadimos al df de resultados
            transport_VSM = pd.concat([transport_VSM, pd.DataFrame([distime_matrix])], ignore_index=True)
            full_distime_matrix = pd.concat([full_distime_matrix, pd.DataFrame([distime_matrix])], ignore_index=True)
        # Añadimos el nuevo transport_VSM a vehicle_score_matrix
        vehicle_score_matrix = pd.concat([vehicle_score_matrix, transport_VSM], ignore_index=True)
    
    return full_distime_matrix
            
def score_calculation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map, citizen_data):
    # Completamos el trip en caso de que tenga que acudir a algún punto P
    complete_trip, last_P = trip_completation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map)
    # Calculamos la matriz de distime
    distime_matrix = distime_calculation(networks_map, complete_trip, pop_building, citizen_data, transport, standard=False)
    # Sacamos el score en base al algoritmo especificado para ello
    distime_matrix = score_algorithm(distime_matrix)
    # Devolvemos score
    return distime_matrix, last_P

def score_algorithm(distime_matrix):
    """
    Aqui definimos el algoritmo de toma de decisiones
    """
    distime_matrix = distime_matrix.iloc[0]
    # Asegurarse de que distime_matrix está desconectado de slices
    distime_matrix = distime_matrix.copy()
    # Calculamos el conmu_time
    conmu_time = distime_matrix['walk_time'] + distime_matrix['travel_time'] + distime_matrix['wait_time']
    
    distime_matrix['conmu_time'] = conmu_time
    
    # Calcular score de forma robusta
    distime_matrix['score'] = (
        conmu_time +
        distime_matrix['cost']*0.004 -
        distime_matrix['benefits']*0.4 +
        distime_matrix['emissions']*0.0001
    )
    
    return distime_matrix

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcula la distancia en km entre dos puntos usando la fórmula de Haversine.
    """
    R = 6371  # radio de la Tierra en km
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def distime_calculation(
    networks_map,
    complete_trip,
    pop_building: pd.DataFrame,
    citizen_data: dict,
    transport: dict,
    standard: bool = True,):
    
    # --- 1) Preparación rápida de lookups ---
    # osm_id -> {'lon':..., 'lat':...}
    df_coords = (pop_building[['osm_id', 'lon', 'lat']]
                .dropna(subset=['lon', 'lat'])
                .drop_duplicates(subset='osm_id', keep='first'))
    coord_map = df_coords.set_index('osm_id')[['lon', 'lat']].to_dict('index')

    # cache dict (para lectura rápida)
    cache = {}
    new_cache_rows = []  # acumulamos y concatenamos al final

    # grafos (atajo)
    G_drive = networks_map.get('drive_map')
    G_walk  = networks_map.get('walk_map')

    # --- 2) Precomputar nearest_nodes por osm_id y mapa ---
    # ids únicos implicados
    osmid_set = set([s0 for s0, _ in complete_trip] + [s1 for _, s1 in complete_trip])

    nearest = {'drive': {}, 'walk': {}}
    if standard:
        # Para cada mapa, calculamos nodos más cercanos en lote
        for mode, G in (('drive', G_drive), ('walk', G_walk)):
            if G is None: 
                continue
            # preparar listas de coords en el orden esperado por OSMnx (x=lon, y=lat)
            xs, ys, valid_ids = [], [], []
            for osm_id in osmid_set:
                c = coord_map.get(osm_id)
                if c is not None:
                    xs.append(c['lon'])
                    ys.append(c['lat'])
                    valid_ids.append(osm_id)
            if valid_ids:
                # vectorizado: devuelve lista de node_ids en el mismo orden
                node_ids = ox.distance.nearest_nodes(G, xs, ys)
                # mapear
                nearest[mode] = {osm: nid for osm, nid in zip(valid_ids, node_ids)}

    # --- 3) Bucle principal (ligero) ---
    rows = []
    trip = (complete_trip[0][0], complete_trip[-1][-1])

    # alternancia simple (conservamos tu lógica)
    map_type = 'drive'
    
    for step_0, step_1 in complete_trip:
        map_type = 'walk' if map_type == 'drive' else 'drive'
        cache_key = (step_0, step_1, map_type)

        # 3.a) Distancia (cache -> calcula -> guarda en new_cache_rows)
        if cache_key in cache:
            distance_km = cache[cache_key]
        elif step_0 == step_1:
            distance_km = 0
            # guardar en cache (memoria) y para persistir al final
            cache[cache_key] = distance_km
            new_cache_rows.append({'step': (step_0, step_1), 'map': map_type, 'km': distance_km})
        else:
            c0 = coord_map.get(step_0)
            c1 = coord_map.get(step_1)
            if c0 is None or c1 is None:
                # si faltan coords, imposible calcular
                distance_km = float('inf')
            else:
                if standard:
                    G = G_drive if map_type == 'drive' else G_walk
                    if G is None:
                        distance_km = float('inf')
                    else:
                        n0 = nearest[map_type].get(step_0)
                        n1 = nearest[map_type].get(step_1)
                        if n0 is None or n1 is None:
                            distance_km = float('inf')
                        else:
                            try:
                                # más rápido que construir la ruta completa
                                dist_m = nx.shortest_path_length(G, n0, n1, weight='length', method='dijkstra')
                                distance_km = dist_m / 1000.0
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                distance_km = float('inf')
                else:
                    distance_km = haversine(c0['lon'], c0['lat'], c1['lon'], c1['lat'])

            # guardar en cache (memoria) y para persistir al final
            cache[cache_key] = distance_km
            new_cache_rows.append({'step': (step_0, step_1), 'map': map_type, 'km': distance_km})

        # 3.b) Otras métricas
        waiting_time = waiting_time_calculation(cache[cache_key], step_1, transport)
        benefits     = benefits_calculation(citizen_data, step_1)        
        
        rows.append({
            'citizen':      citizen_data['name'],
            'vehicle':      transport['name'],
            'archetype':    transport['archetype'],
            'trip':         trip,
            'distance':     cache[cache_key],
            'walk_time':    (cache[cache_key] / citizen_data['walk_speed']) if (map_type == 'walk' and cache[cache_key] > 0) else 0,
            'travel_time':  (cache[cache_key] / transport['v']) if (map_type == 'drive' and cache[cache_key] > 0) else 0,
            'wait_time':    waiting_time,
            'cost':         (transport['Ekm']   * cache[cache_key]) if (map_type == 'drive' and cache[cache_key] > 0) else 0,
            'mjkm':         (transport['mjkm']   * cache[cache_key]) if (map_type == 'drive' and cache[cache_key] > 0) else 0,
            'benefits':     benefits,
            'emissions':    (transport['CO2km'] * cache[cache_key]) if (map_type == 'drive' and cache[cache_key] > 0) else 0,
        })

    # --- 5) Agregación final (igual que antes, pero sin overhead extra) ---
    distime_matrix = pd.DataFrame(rows)
    summed_numeric = distime_matrix.select_dtypes(include='number').sum().to_frame().T
    summed_df = summed_numeric.assign(
        citizen = distime_matrix.iloc[0]['citizen'],
        vehicle = distime_matrix.iloc[0]['vehicle'],
        archetype = distime_matrix.iloc[0]['archetype'],
        trip = [distime_matrix.iloc[0]['trip']],
    )
    return summed_df


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

def trip_completation(trip, transport, pop_archetypes_transport, last_P, pop_building, networks_map):
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
    else:
        p2_osm_id = poi_B
    # Por último añadimos la meta
    steps.append(poi_B)
    # Inicializamos la lista de resultados
    complete_trip = []
    # Adaptamos para que sea más comoda de usar
    for step in range(len(steps)-1):
        complete_trip.append((steps[step], steps[step+1]))    
    
    return complete_trip, p2_osm_id

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
        'name': 'Public_transport',
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

def route_creation(schedule):
    # Extrae la secuencia de osm_id
    osm_id_route = schedule['osm_id']
    # Elimina duplicados consecutivos
    simpl_route = [k for k, _ in groupby(osm_id_route)]
    # Construye las duplas consecutivas
    route = list(zip(simpl_route, simpl_route[1:]))
    return route
    
def get_independents(level1_schedule):
    return level1_schedule[(level1_schedule['todo'] == 'WoS') & (level1_schedule['fixed'] == False)]['agent'].unique()
    
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