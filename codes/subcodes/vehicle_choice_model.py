import os
import re
import sys
import copy
import math
from tqdm import tqdm
import numpy as np
import numbers
import random
import osmnx as ox
import pandas as pd
import networkx as nx
from collections import defaultdict
from pathlib import Path
from itertools import groupby
import matplotlib.pyplot as plt
import networkx as nx
import networkx as nx
import osmnx as ox
import pandas as pd
import multiprocessing
from haversine import Unit, haversine


from concurrent.futures import ProcessPoolExecutor, as_completed   # o ThreadPoolExecutor si es I/O-bound
from functools import partial
import pandas as pd
from tqdm import tqdm
import os

# ------- WORKER: procesa una familia completa -------
def _process_family(
    family_tuple,
    paths,
    transport_families_dict,
    agent_populations,
    pop_archetypes,
    networks_map
):
    f_name, family = family_tuple

    # Vehículos disponibles para esta familia (o vacío)
    df_avail_vehicles = transport_families_dict.get(f_name, pd.DataFrame())
    avail_vehicles = df_avail_vehicles.to_dict(orient='records')
    
    # Schedule level-1 de esta familia
    todo_list_family = family.to_dict(orient='records')

    family_members = list({row['agent'] for row in todo_list_family})

    all_citizen_schedule = []
    all_vehicle_schedule = []

    for c_name in family_members:

        citizen_todolist = [row for row in todo_list_family if row['agent'] == c_name]
        citizen_data = agent_populations['citizen'].loc[agent_populations['citizen']['name'] == c_name].iloc[0]
    
        # If 'citizen_todolist' only has one row, mean they did not left home, so no vehicle moving nether
        if len(citizen_todolist) == 1:
            continue
        
        # Ruta del ciudadano
        citizen_route = route_creation(citizen_todolist)

        # El agente se queda en casa
        if citizen_route == []:
            continue

        # Calcula matriz dist/tiempo y actualiza cache local
        distime_matrix = VSM_calculation(
            citizen_route,
            avail_vehicles,
            citizen_data,
            pop_archetypes['transport'],
            agent_populations['building'],
            networks_map
        )

        # Escoge vehículo
        best_transport_distime_matrix = vehicle_chosing(distime_matrix)

        # Actualiza schedule de la familia con la elección
        schedule = create_citizen_schedule(best_transport_distime_matrix, c_name, citizen_todolist)
        citizen_schedule = copy.deepcopy(schedule)
        all_citizen_schedule.extend(citizen_schedule)

        # Actualiza acciones de vehículos
        vehicle_schedule = create_vehicles_actions(schedule, best_transport_distime_matrix)
        all_vehicle_schedule.extend(vehicle_schedule)

        ####################################################
        # Aqui es donde debemos meter el modelo de Qiaochu #
        ####################################################
        '''
        Basicamente, su modelo, mira el transporte actual y evalua la probabilidad de que cambie del que tenia a 
        uno más V2G-related. Por lo tanto, y a menos de que me exprese alguna modificacion. Habría que añadir su
        modulo aqui y considerar el current_transport como el actual y evaluar si cambia o no.
        
        Parece que no se concibe la opcion de que el current sea V2G, por lo que
        
        '''  

        file_path = os.path.join(paths['new_POIs'], 'share_mob_hubs.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            choices3 = not df.empty
        else:
            choices3 = False
        
        results = WP3_parameters_simplified(paths, pop_archetypes, agent_populations, avail_vehicles, best_transport_distime_matrix, citizen_schedule, vehicle_schedule, choices3)
        
        new_row = citizen_data
        
        # “Consume” vehículo si no es compartible
        vehicle_name = best_transport_distime_matrix[0]['vehicle']
        if vehicle_name not in ('walk', 'Public_transport') and avail_vehicles != []:
            avail_vehicles = [v for v in avail_vehicles if v['name'] != vehicle_name]
       
    # sacamos los datos de family
    pattern = re.compile(r'Home|Home_in|Home_out')

    filtered = [row for row in todo_list_family if row.get('todo') and pattern.search(row['todo'])]
    
    family_home = filtered[0]['osm_id']
    family_node = filtered[0]['node']
    family_archetype = filtered[0]['family_archetype']

    for vehicle in avail_vehicles:

        if vehicle['name'] in ('walk', 'Public_transport'):
            continue
      
        new_row = {
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
        }     
        
        all_vehicle_schedule.append(new_row)
        
    return all_citizen_schedule, all_vehicle_schedule

def vehicle_choice_model(
    todolist,
    agent_populations,
    paths,
    study_area,
    pop_archetypes,
    networks_map,
    day,
    use_threads=False
):
    # --- Agrupar por familia una sola vez ---
    todolist_families = todolist.groupby('family')
    families = list(todolist_families)  # [(f_name, df_sub)]
    transport_families = agent_populations['transport'].groupby('family') if not agent_populations['transport'].empty else None

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
    citizen_schedules = []
    vehicle_schedules = []      
    
    for fam_tuple in tqdm(families, desc=f"/secuential/ Transport Choice Modelling ({day})"):
        fam_schedule, fam_actions= _process_family(fam_tuple,
                                                   paths,
                                                   transport_families_dict,
                                                   agent_populations,
                                                   pop_archetypes,
                                                   networks_map)
        if fam_schedule is not None and fam_schedule != []:
            citizen_schedules.extend(fam_schedule)
        if fam_actions is not None and fam_actions != []:
            vehicle_schedules.extend(fam_actions)
        
    '''worker = partial(
        _process_family,
        paths=paths,
        transport_families_dict=transport_families_dict,
        agent_populations=agent_populations,
        pop_archetypes=pop_archetypes,
        networks_map=networks_map,
    )
    
    n_jobs = multiprocessing.cpu_count() - 1 
    
    with Executor(max_workers=n_jobs) as ex:
        futures = {ex.submit(worker, fam_tuple): fam_tuple[0] for fam_tuple in families}
        for fut in tqdm(as_completed(futures), total=len(families), desc=f"Transport Choice Modelling ({day})"):
            fam_name = futures[fut]
            try:
                fam_schedule, fam_actions = fut.result()
                if fam_schedule is not None and fam_schedule != []:
                    citizen_schedules.extend(fam_schedule)
                if fam_actions is not None and fam_actions != []:
                    vehicle_schedules.extend(fam_actions)
            except Exception as e:
                print(f"[ERROR] familia '{fam_name}': {e}")'''

    # --- Agregación en el proceso principal ---
    df_citizen_schedules = pd.DataFrame(citizen_schedules)
    df_vehicle_schedules = pd.DataFrame(vehicle_schedules)

    # --- Escrituras UNA sola vez ---
    out_vehicle_schedules = os.path.join(paths['results'], f"{study_area}_{day}_schedule_vehicle.xlsx")
    out_citizen_schedules  = os.path.join(paths['results'], f"{study_area}_{day}_schedule_citizen.xlsx")
    df_vehicle_schedules.to_excel(out_vehicle_schedules, index=False)
    df_citizen_schedules.to_excel(out_citizen_schedules, index=False)

    return vehicle_schedules, citizen_schedules

def main():
    # Input
    population = 20
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
    
    agent_populations = {}
    pop_archetypes = {}
    
    agent_populations['citizen'] = pd.read_parquet(f"{paths['population']}/pop_citizen.parquet")
    agent_populations['family'] = pd.read_parquet(f"{paths['population']}/pop_family.parquet")
    agent_populations['building'] = pd.read_parquet(f"{paths['population']}/pop_building.parquet")
    agent_populations['transport'] = pd.read_parquet(f"{paths['population']}/pop_transport.parquet")
    
    pop_archetypes['transport'] = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    # Días que quieres considerar
    days = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}

    found_schedule = set()
    found_vehicles = set()

    for file in paths['results'].glob('*.xlsx'):
        name = file.stem  # sin extensión
        parts = name.split('_')
        if len(parts) < 4:
            continue
        # asumiendo formato: study_area_day_kind
        day, agent_type = parts[-3], parts[-1].lower()
        if day not in days:
            continue
        if agent_type == 'citizen':
            found_schedule.add(day)
        elif agent_type == 'vehicle':
            found_vehicles.add(day)

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
            todolist = pd.read_excel(f"{paths['results']}/{study_area}_{day}_todolist.xlsx")
            # Vehicle Choice Modeling
            vehicle_schedules, citizen_schedules = vehicle_choice_model(todolist, agent_populations, paths, study_area, pop_archetypes, networks_map, day)
    else:
        print(f"All days' schedules already modeled.")
        
def create_citizen_schedule(best_transport_distime_matrix, c_name, citizen_todolist):
    """
    Actualiza los tiempos de entrada/salida ('in'/'out') del agente `c_name`
    en base a los nuevos tiempos de conmutación contenidos en
    `best_transport_distime_matrix['conmu_time']`.

    Requisitos de columnas:
      - en `todo_list_family`: ['agent', 'trip', 'todo', 'opening', 'closing', 'time2spend']
      - en `best_transport_distime_matrix`: ['conmu_time']
    """

    # Filtrar y ordenar la lista del agente
    todo_list = sorted(citizen_todolist, key=lambda x: x['trip'])

    # Asegurar mismo largo (simple y directo)
    # Se usará el índice de la iteración para tomar el conmu_time
    commute = [row['conmu_time'] for row in best_transport_distime_matrix]

    out_time = None  # se actualizará en el bucle

    for idx, row in enumerate(todo_list):
        if row['todo'] == 'Home_out':
            # Sale de casa con antelación igual a la conmutación hasta la primera actividad
            in_time = row['opening']
            out_time = todo_list[idx+1]['opening'] - commute[idx]
            row['in'] = int(in_time)
            row['out'] = int(out_time)
            continue

        # tiempo de conmutación para este paso
        conmu_time = commute[idx-1]

        # Para el resto, llega tras la conmutación desde el punto anterior
        if out_time is None:
            # Si por lo que sea no pasó por 'Home_out', asumimos que parte en 'opening'
            out_time = row['opening']

        in_time = out_time + conmu_time
        out_time = (in_time + row['time2spend']) if row['time2spend'] != 0 else row['closing']

        row['in'] = int(in_time)
        row['out'] = int(out_time)
    
    citizen_schedule = todo_list.copy()
    
    return citizen_schedule

def create_vehicles_actions(citizen_schedule, best_transport_distime_matrix):
    '''El objetivo es tener un df que de los datos de consumo relevante para cada actividad'''

    # Antes de iniciar, aseguramos que no sea walk o publico o walk_public
    if best_transport_distime_matrix[0]['vehicle'] in ['walk', 'Public_transport']:
        # lista vacía = DF vacío
        return []
    
    # Simplificamos el 'new_family_schedule', para guardar la info como lo hariamos para el output
    simple_schedule = schedule_simplification(citizen_schedule)

    # Ahora duplicamos pero metemos el vehiculo en vez de la persona
    for row in simple_schedule:
        row['user'] = row['agent']
        row['agent'] = best_transport_distime_matrix[0]['vehicle']
        row['archetype'] = best_transport_distime_matrix[0]['archetype']

    for idx, row in enumerate(simple_schedule):
        if idx == 0:
            row['ETC [kWh]'] = 0
            continue
        row['ETC [kWh]'] = best_transport_distime_matrix[idx-1]['mjkm'] + simple_schedule[idx-1]['ETC [kWh]']

    return simple_schedule    
    
def schedule_simplification(citizen_schedule):

    ''' Este modulo es basicamente aplicable a los casos con relaciones intrafamiliares, 
    donde hay not time related activities y temas por el estilo, porque igual estan 
    esperando un rato en el mimo edificio, luego se piran, ...'''

    # Inicializamos el df de salida
    simple_schedule = []
    # Filtramos el df quitando las actividades 'not-time-related'
    filtered = [row for row in citizen_schedule if abs(row['in'] - row['out']) != 0]

    # Agrupamos por 'osm_id' para simplificar
    filtered_grouped = defaultdict(list)
    for row in filtered:
        filtered_grouped[row['osm_id']].append(row)
    # Evaluamos los grupos
    for _, group in filtered_grouped.items():
        # Si el grupo no tiene duplicados, no nos va a dar problemas
        if len(group) == 1:
            simple_schedule.extend(group)
            continue
        # Inicializamos 'def_in' (para tener como memoria del 'in' del grupo)
        def_in = float('inf')
        # En caso de tener más de una actividad 'time-related' en el mismo 'osm_id' 
        for idx, row in enumerate(group):
            # Nos saltamos el último row
            if idx == len(group)-1:
                continue
            # Evaluamos cuales estan concatenados
            if group[idx]['out'] == group[idx + 1]['in']:
                # Asignamos las variables
                def_in = group[idx]['in']
                def_out = group[idx + 1]['out']
        # Si no se han identificado concatenaciones
        if def_in == float('inf'):
            simple_schedule.extend(group)
            continue
        # Copiamos los datos del grupo
        new_row = group[0]
        # Añadimos los nuevos datos
        new_row['in'] = def_in
        new_row['out'] = def_out
        # Anadimos la nueva fila al schedule simplificado
        simple_schedule.extend(new_row)

    return sorted(simple_schedule, key=lambda x: x['trip'])

def WP3_parameters_simplified(paths: list, pop_archetypes: dict, agent_populations: dict, avail_vehicles: list, current_transport: list, 
                              citizen_schedule: list, vehicle_schedule: list, choices3: bool=True):
    def Mode_choice(choices3: bool,
                    IS_Gaso: bool, IS_EV: bool, IS_PT: bool, IS_Bike: bool,
                    EV_OWNERSHIP: bool, HAVING_KIDS: bool,
                    AGE: int, INCOME: int, 
                    COST: float, COST_EV2G: float, COST_CSV2G: float, 
                    WALK_TIME: float, WALK_TIME_EV2G: float, WALKING_TIME_CV2G: float, 
                    TRAVEL_TIME: float, TRAVEL_TIME_EV2G: float, TRAVEL_TIME_CSV2G: float,
                    WAIT_TIME: float=20.5, PARK_COST: float=0, SOC_PEV: float=0):
        
        B_Cost_all          = -0.05 if choices3 else -0.03
        B_Parkcost_GasoEV   = -0.03 
        B_TravelTime_PTBike = -0.02 if choices3 else -0.05
        B_WalkTime_all      = -0.04 if choices3 else -0.03
        B_WaitTime_PT       = -0.16 if choices3 else -0.14
        B_SOC_PEV           = 0.01
        ASC_EV2G_PTUser             = -1.27 if choices3 else 0
        ASC_EV2G_GasoEVUser         = -1.76 if choices3 else 0
        ASC_EV2G_BikeUser           = -2.66 if choices3 else 0
        B_TravelTime_V2G_PTBikeUser = -0.05 if choices3 else 0
        B_EV_OWNERSHIP_EV2G         = 0.52 if choices3 else 0
        B_HAVING_KIDS_EV2G          = -0.37 if choices3 else 0
        ASC_CSV2G_PTUser        = -1.27 if choices3 else -1.17 # SURE?
        ASC_CSV2G_GasoEVUser    = -1.76 if choices3 else 0 # SURE?
        ASC_CSV2G_BikeUser      = -2.66 if choices3 else -3.48
        B_EV_OWNERSHIP_CSV2G    = 0.52 if choices3 else 0.64
        B_HAVING_KIDS_CSV2G     = -0.2 if choices3 else 0
        B_AGE_YOUNG_CSV2G       = 0.21
        B_INCOME_LOW_CSV2G      = 0.35 if choices3 else 0.34


        V1 = (
            B_Cost_all * COST +
            B_Parkcost_GasoEV * PARK_COST * IS_Gaso +
            B_Parkcost_GasoEV * PARK_COST * IS_EV +
            B_TravelTime_PTBike * TRAVEL_TIME * IS_PT +
            B_TravelTime_PTBike * TRAVEL_TIME * IS_Bike +
            B_WalkTime_all * WALK_TIME +
            B_WaitTime_PT * WAIT_TIME * IS_PT +
            B_SOC_PEV * SOC_PEV * IS_EV
        )

        if choices3:
            V2 = (
                ASC_EV2G_PTUser * IS_PT +
                ASC_EV2G_GasoEVUser * IS_EV +
                ASC_EV2G_GasoEVUser * IS_Gaso +
                ASC_EV2G_BikeUser * IS_Bike +
                B_Cost_all * COST_EV2G +
                B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_EV2G * IS_PT +
                B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_EV2G * IS_Bike +
                B_WalkTime_all * WALK_TIME_EV2G + 
                B_EV_OWNERSHIP_EV2G * EV_OWNERSHIP + 
                B_HAVING_KIDS_EV2G * HAVING_KIDS 
            )
        else:
            V2 = 0

        V3 = (
            ASC_CSV2G_PTUser*IS_PT +
            ASC_CSV2G_GasoEVUser*IS_EV +
            ASC_CSV2G_GasoEVUser*IS_Gaso +
            ASC_CSV2G_BikeUser*IS_Bike +
            B_Cost_all * COST_CSV2G  +
            B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_CSV2G * IS_PT +
            B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_CSV2G * IS_Bike +
            B_WalkTime_all * WALKING_TIME_CV2G +
            B_EV_OWNERSHIP_CSV2G * EV_OWNERSHIP +
            (B_HAVING_KIDS_CSV2G * HAVING_KIDS if choices3 else 0) +
            B_AGE_YOUNG_CSV2G * (AGE <= 35) +
            B_INCOME_LOW_CSV2G * (INCOME <= 3000) 
        )

        #print(f"Utility for choosing the current mode: {V1:.3f}")
        #print(f"Utility for choosing private EV with V2G: {V2:.3f}")
        #print(f"Probability of choosing carsharing services with V2G: {V3:.3f}")

        P_1 = math.exp(V1) / (math.exp(V1) + (math.exp(V2) if choices3 else 0) + math.exp(V3))
        P_2 = (math.exp(V2) if choices3 else 0) / (math.exp(V1) + (math.exp(V2) if choices3 else 0) + math.exp(V3))
        P_3 = math.exp(V3) / (math.exp(V1) + (math.exp(V2) if choices3 else 0) + math.exp(V3))

        #print(f"Probability of choosing current mode: {P_1*100:.3f}%")
        #print(f"Probability of choosing private Evs with V2G: {P_2*100:.3f}%")
        #print(f"Probability of choosing carsharing services with V2G: {P_3*100:.3f}%")
        #print(f"Total: {(P_1+P_2+P_3):.3f}")
        
        modes = ["P_1", "P_2", "P_3"]
        chosen_mode = random.choices(modes, weights=[P_1, P_2, P_3], k=1)[0]
        
        return chosen_mode

    def Plug_in_choices(LOCATION_WORK: bool, LOCATION_SHOPPING: bool, EV_OWNERSHIP: bool, 
                        PARK_TIME: float, COST_SAVING: float,
                        INCOME: float,
                        WALK_TIME: float=0,
                        CYCLE: int=1, SOC_state: float=50, BATTERY_GUARANTEE: float=50, 
                        DISTANCE_NEXT: float=0):
        
        ASC_PLUGIN          = 0.845600726
        B_LOCATION_WORK     = -0.214813432
        B_LOCATION_SHOPPING = -0.349427058
        B_PARK_TIME         = 0.072730788
        B_DISTANCE_NEXT     = 0.005024964
        B_COST_SAVING       = 0.129340844
        B_WALK_TIME         = -0.082726453
        B_INCOME_LOW        = 0.182566091
        B_INCOME_HIGH       = 0.265787361
        B_CYCLE             = -0.052004313
        B_SOC_EVowner       = -0.004559245
        B_SOC_nonEVowner    = -0.009348043
        B_BATTERY_GUARANTEE_nonEVowner  = 0.004927057

        V_plugin  = (
            ASC_PLUGIN+
            B_LOCATION_WORK * LOCATION_WORK + B_LOCATION_SHOPPING * LOCATION_SHOPPING +
            B_PARK_TIME * PARK_TIME +
            B_DISTANCE_NEXT * DISTANCE_NEXT +
            B_COST_SAVING* COST_SAVING +
            B_WALK_TIME * WALK_TIME +
            B_INCOME_LOW * (INCOME<=3000)+ 
            B_INCOME_HIGH * (INCOME>6000) + 
            B_CYCLE *CYCLE +
            B_SOC_EVowner * (EV_OWNERSHIP==1) * SOC_state +
            B_SOC_nonEVowner * (EV_OWNERSHIP==0) * SOC_state + B_BATTERY_GUARANTEE_nonEVowner * (EV_OWNERSHIP==0) * BATTERY_GUARANTEE 
        )
        V_notplugin = 0

        P_plugin = math.exp(V_plugin)/(math.exp(V_plugin) + math.exp(V_notplugin))
        P_notplugin = 1-P_plugin

        #print(f"Probability of plug-in: {P_plugin}")
        #print(f"Probability of not plug-in: {P_notplugin}")

        modes = [True, False]
        plugin = random.choices(modes, weights=[P_plugin, P_notplugin], k=1)[0]
        
        return plugin

    def get_vehicle_stats(archetype: str, transport_archetypes: pd.DataFrame, variables: list):
        results = {}   
        
        # Filtrar la fila correspondiente al arquetipo
        row = transport_archetypes[transport_archetypes['name'] == archetype]
        if row.empty:
            print(f"The archetype '{archetype}' was not found in transport_archetypes:")
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
    
    def virtual_EV_generator(archetypes_transport: pd.DataFrame, CSEV: bool):

        archetype = 'CS_electric' if CSEV else 'PC_electric'
        
        # Sacamos las caracteristicas con valores estadisticos
        variables = [
                col.rsplit('_', 1)[0]
                for col in archetypes_transport.columns
                if col.endswith('_mu')
            ]

        virtual_EV = {
            'name':         'virtual',
            'archetype':    archetype,
            'family':       '-',
            'ubication':    'W448331296'
        }

        values = get_vehicle_stats(archetype, archetypes_transport, variables)
        row_updated = assign_data(variables, values, virtual_EV)
        
        return row_updated
    
    def closest_share_mob_hubs(home_lat, home_lon, hubs):

        # 2) Calcular la distancia
        # Si las distancias son cortas (misma ciudad), euclidiana es suficiente.
        hubs['dist'] = np.sqrt((hubs['lat'] - home_lat)**2 + (hubs['lon'] - home_lon)**2)

        # 3) Seleccionar el más cercano
        nearest = hubs.loc[hubs['dist'].idxmin(), ['lat', 'lon']]

        return float(nearest['lat']), float(nearest['lon'])
    
    def generate_ev_data(trip, pop_building, citizen_data, networks_map, CSEV):
        
        transport = virtual_EV_generator(pop_archetypes['transport'], CSEV)

        if CSEV:
            
            home_lat, home_lon = pop_building.loc[pop_building['osm_id'] == citizen_data['Home'], ['lat', 'lon']].iloc[0]

            # 1) Leer el Excel
            file_path = os.path.join(paths['new_POIs'], 'share_mob_hubs.xlsx')
            hubs = pd.read_excel(file_path)

            lat, lon = closest_share_mob_hubs(home_lat, home_lon, hubs)
            
            csev_hub = [{
                'building_type': 'share_mob_hubs',
                'osm_id': ''
                '',
                'lat': lat,
                'lon': lon,
                'archetype': 'share_mob_hubs',
                'WoS_opening': 0,
                'WoS_closing': 24*60,
                'Service_opening': 0,
                'Service_closing': 24*60, 
                'node': '-'
            }]
            
            new_pop_building = pd.concat([pop_building, pd.DataFrame(csev_hub)], ignore_index=True)
            
            last_P = 'share_mob_hubs'
        else:
            last_P = transport['ubication']
            new_pop_building = pop_building
        
        distime_matrix, last_P = score_calculation(trip, transport, pop_archetypes['transport'], last_P, new_pop_building, networks_map, citizen_data)
        
        TRAVEL_TIME = distime_matrix['travel_time']
        WALK_TIME = distime_matrix['walk_time']
        COST = distime_matrix['cost']
        
        return TRAVEL_TIME, WALK_TIME, COST
        
    
    def data_gathering(paths: list, pop_archetypes: dict, agent_populations: dict, 
                       avail_vehicles: list, current_transport: list, 
                       citizen_schedule: list, vehicle_schedule: list, choices3: bool):
        
        data = {}
        
        # 1) current_transport
        first_trip = current_transport[0]
        
        data['COST']                = first_trip['cost']
        data['TRAVEL_TIME']         = first_trip['conmu_time']
        data['WALK_TIME']           = first_trip['walk_time']
        
        data['IS_Gaso']             = False
        data['IS_EV']               = False
        data['IS_PT']               = False
        data['IS_Bike']             = False

        if first_trip['vehicle'] == 'Public_transport':
            data['IS_PT']           = True
        elif first_trip['archetype'] == 'PC_electric':
            data['IS_EV']           = True
        elif first_trip['archetype'] == 'PC_petrol':
            data['IS_Gaso']         = True        

        # 2) avail_vehicles
        if len(avail_vehicles) > 0:
            data['EV_OWNERSHIP'] = any(v['archetype'] == "PC_electric" for v in avail_vehicles)
        else:
            data['EV_OWNERSHIP'] = False

        # 3) citizen_schedule
        data["HAVING_KIDS"] = (citizen_schedule[0]['family_archetype'] in ["f_arch_1", "f_arch_2", "f_arch_3", "f_arch_6"])
        
        data['LOCATION_WORK']       = True if citizen_schedule[1]['todo'] == 'WoS' else False
        data['LOCATION_SHOPPING']   = True if citizen_schedule[1]['todo'] != 'WoS' else False
        
        data['PARK_TIME']           = (citizen_schedule[1]['out'] - citizen_schedule[1]['in'])/60
        
        if citizen_schedule[0]['archetype'] in ["c_arch_0", "c_arch_1", "c_arch_3"]:
            data['AGE'] = 50
        else:
            data['AGE'] = 12
        
        s_class = citizen_schedule[0]['s_class']
        if s_class == 'Salariat':
            data['INCOME'] = 7000
        elif s_class == 'Intermediate':
            data['INCOME'] = 4500
        else:
            data['INCOME'] = 2000
        
        # 4) trip
        trip = current_transport[0]['trip']
        
        # 5) agent_populations
        pop_citizen = agent_populations['citizen'] 
        pop_building = agent_populations['building']
        
        citizen_name = citizen_schedule[0]['agent']
        
        citizen_data = pop_citizen[pop_citizen['name'] == citizen_name].iloc[0]

        networks_map = None
        
        # 6) EV2G data
        data['TRAVEL_TIME_EV2G'], data['WALK_TIME_EV2G'], data['COST_EV2G'] = generate_ev_data(trip, pop_building, citizen_data, networks_map, False)
        
        if choices3:
            data['TRAVEL_TIME_CSV2G'], data['WALKING_TIME_CV2G'], data['COST_CSV2G']  = generate_ev_data(trip, pop_building, citizen_data, networks_map, True)
        else:
            # In this case this data will not be considered, but the tags must exists so '0' will do
            data['TRAVEL_TIME_CSV2G'] = data['WALKING_TIME_CV2G'] = data['COST_CSV2G'] = 0

        return data
    
    data = data_gathering(paths, pop_archetypes, agent_populations, avail_vehicles, current_transport, citizen_schedule, vehicle_schedule, choices3)

    chosen_mode = Mode_choice(choices3,
                              data['IS_Gaso'], data['IS_EV'], data['IS_PT'], data['IS_Bike'],
                              data['EV_OWNERSHIP'], data['HAVING_KIDS'],
                              data['AGE'], data['INCOME'], 
                              data['COST'], data['COST_EV2G'], data['COST_CSV2G'], 
                              data['WALK_TIME'], data['WALK_TIME_EV2G'], data['WALKING_TIME_CV2G'], 
                              data['TRAVEL_TIME'], data['TRAVEL_TIME_EV2G'], data['TRAVEL_TIME_CSV2G'])

    #print(f"chosen_mode: {chosen_mode}")

    if (chosen_mode == 'P_1') & (not data['IS_EV']):
        return chosen_mode, False

    if chosen_mode == 'P_2':
        COST_P = data['COST_EV2G']
    else:
        COST_P = data['COST_CSV2G']

    plugin = Plug_in_choices(data['LOCATION_WORK'], data['LOCATION_SHOPPING'], data['EV_OWNERSHIP'], 
                             data['PARK_TIME'], COST_P,
                             data['INCOME'])

    #print(f"plugin: {plugin}")

    return chosen_mode, plugin 

def vehicle_chosing(vehicle_score_matrix, simplified: bool=True): 

    if simplified:
        walk_rows = [row for row in vehicle_score_matrix if row['vehicle'] == 'walk']
        
        if all(row['conmu_time'] < 15 for row in walk_rows):
            return walk_rows
        
        rest_rows = [row for row in vehicle_score_matrix if row['vehicle'] not in ('walk', 'Public_transport')]

        if not rest_rows:
            current_transport = [row for row in vehicle_score_matrix if row['vehicle'] == 'Public_transport']
        else:
            chosen_vehicle = rest_rows[0]['vehicle']
            current_transport = [row for row in vehicle_score_matrix if row['vehicle'] == chosen_vehicle]
        
        return current_transport

    # Sumamos los scores por transporte
    simplified_df = vehicle_score_matrix.groupby('vehicle', as_index=False).sum()   
    # Sacamos el transporte con menos score
    best_transport = simplified_df.loc[simplified_df['score'].idxmin()]

    # Sacamos del original walk y public transport
    public_walk = vehicle_score_matrix[vehicle_score_matrix['vehicle'].isin(['walk', 'Public_transport'])]
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
    
    return current_transport

def VSM_calculation(citizen_route, avail_vehicles, citizen_data, pop_archetypes_transport, pop_building, networks_map):
    # Inicializamos la full_distime_matrix
    full_distime_matrix = []
    if citizen_data['independent_type'] == 0:
        vehicles = [v for v in avail_vehicles if v['name'] in ["walk", "Public_transport"]]
    else:
        vehicles = avail_vehicles

    # Añadimos a la matriz de vehiculos disponibles el publico y andar
    avail_transport = add_public_walk(vehicles, citizen_data, pop_archetypes_transport)
    # Iteramos los distintos transportes disponibles
    for transport in avail_transport:
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
            full_distime_matrix.append(distime_matrix)

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

def distime_calculation(
    networks_map,
    complete_trip,
    pop_building: pd.DataFrame,
    citizen_data: dict,
    transport: dict,
    standard: bool = True
):
    # --- 1) Preparación rápida de lookups ---
    # Idealmente esto debería venir precalculado de fuera
    df_coords = (
        pop_building[['osm_id', 'lon', 'lat']]
        .dropna(subset=['lon', 'lat'])
        .drop_duplicates(subset='osm_id', keep='first')
    )
    coord_map = df_coords.set_index('osm_id')[['lon', 'lat']].to_dict('index')

    # cache local (en tu código parece pensado para extenderlo a algo global)
    cache = {}
    new_cache_rows = []  # si luego lo vas a persistir en algún sitio

    # --- 2) Precomputar nearest_nodes por osm_id y mapa ---
    osmid_set = {s0 for s0, _ in complete_trip} | {s1 for _, s1 in complete_trip}

    nearest = {'drive': {}, 'walk': {}}
    if standard:
        G_drive = networks_map.get('drive_map')
        G_walk  = networks_map.get('walk_map')

        for mode, G in (('drive', G_drive), ('walk', G_walk)):
            if G is None:
                continue

            xs, ys, valid_ids = [], [], []
            for osm_id in osmid_set:
                c = coord_map.get(osm_id)
                if c is not None:
                    xs.append(c['lon'])
                    ys.append(c['lat'])
                    valid_ids.append(osm_id)

            if valid_ids:
                node_ids = ox.distance.nearest_nodes(G, xs, ys)
                nearest[mode] = {osm: nid for osm, nid in zip(valid_ids, node_ids)}

    # --- 3) Precomputos lógicos antes del bucle ---
    trip = (complete_trip[0][0], complete_trip[-1][-1])

    has_virtual = any("virtual" in elem for pair in complete_trip for elem in pair)
    is_dutties  = ("Dutties" in complete_trip[-1][-1]) if has_virtual else False

    rows = []
    map_type = 'drive'  # punto de partida

    # --- 4) Bucle principal ---
    for step_0, step_1 in complete_trip:
        # lógica de alternancia
        if transport['archetype'] == 'walk':
            map_type = 'walk'
        elif len(complete_trip) == 1:
            map_type = 'drive'
        else:
            map_type = 'walk' if map_type == 'drive' else 'drive'

        cache_key = (step_0, step_1, map_type)

        # 4.a) Distancia
        if cache_key in cache:
            distance_km = cache[cache_key]

        elif step_0 == step_1:
            distance_km = 0.0
            cache[cache_key] = distance_km
            new_cache_rows.append({'step': (step_0, step_1), 'map': map_type, 'km': distance_km})

        else:
            if has_virtual:
                if is_dutties:
                    distance_km = citizen_data['dist_poi'] / 1000.0
                else:
                    distance_km = citizen_data['dist_wos'] / 1000.0
            else:
                c0 = coord_map.get(step_0)
                c1 = coord_map.get(step_1)

                if c0 is None or c1 is None:
                    distance_km = float('inf')
                elif standard:
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
                                dist_m = nx.shortest_path_length(
                                    G, n0, n1, weight='length', method='dijkstra'
                                )
                                distance_km = dist_m / 1000.0
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                distance_km = float('inf')
                else:
                    distance_km = haversine(
                        (c0['lon'], c0['lat']),
                        (c1['lon'], c1['lat']),
                        unit=Unit.METERS
                    ) / 1000.0

            cache[cache_key] = distance_km
            new_cache_rows.append({'step': (step_0, step_1), 'map': map_type, 'km': distance_km})

        # 4.b) Otras métricas
        waiting_time = waiting_time_calculation(distance_km, step_1, transport)
        benefits     = benefits_calculation(citizen_data, step_1)

        rows.append({
            'citizen':      citizen_data['name'],
            'vehicle':      transport['name'],
            'archetype':    transport['archetype'],
            'trip':         trip,
            'distance':     distance_km,
            'walk_time':    (distance_km / citizen_data['walk_speed']) if (map_type == 'walk'  and distance_km > 0) else 0,
            'travel_time':  (distance_km / transport['v'])            if (map_type == 'drive' and distance_km > 0) else 0,
            'wait_time':    waiting_time,
            'cost':         (transport['price'] * transport['mjkm'] * distance_km) if (map_type == 'drive' and distance_km > 0) else 0,
            'mjkm':         (transport['mjkm']   * distance_km)                     if (map_type == 'drive' and distance_km > 0) else 0,
            'benefits':     benefits,
            'emissions':    (transport['CO2km']  * distance_km)                     if (map_type == 'drive' and distance_km > 0) else 0,
        })

    # --- 5) Agregación final ---
    first = rows[0]
    numeric_keys = [k for k, v in first.items() if isinstance(v, numbers.Number)]
    acc = {k: 0.0 for k in numeric_keys}

    for row in rows:
        for k in numeric_keys:
            acc[k] += row[k]

    result = {
        **acc,
        "citizen":   first["citizen"],
        "vehicle":   first["vehicle"],
        "archetype": first["archetype"],
        "trip":      first["trip"],
    }

    return result

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

def trip_completation(
    trip,
    transport,
    pop_archetypes_transport,  # ya con index = 'name'
    last_P,
    pop_building,
    networks_map
):
    """
    Devuelve la lista de tramos del viaje completo y el p2_osm_id final.
    """

    poi_A, poi_B = trip

    # Si alguno es virtual, no tocamos nada
    if poi_A.startswith("virtual") or poi_B.startswith("virtual"):
        return [trip], poi_B

    # Obtenemos parámetros del arquetipo
    p1, p2, p1s, p2s = pop_archetypes_transport[
        pop_archetypes_transport['name'] == transport['archetype']
    ][['P1', 'P2', 'P1s', 'P2s']].values[0]

    steps = [poi_A]

    if p1 != 0:
        steps.append(last_P)

    if p2 != 0:
        p2_osm_id, p2_poib_dist = find_p2(
            poi_B, transport, p2s, pop_building, networks_map
        )
        steps.append(p2_osm_id)
    else:
        p2_osm_id = poi_B

    # Siempre se añade la meta al final
    steps.append(poi_B)

    # Construimos los tramos consecutivos
    complete_trip = list(zip(steps, steps[1:]))

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
    available_P['distance'] = available_P.apply(
        lambda row: haversine(
            (poi_B_data['lat'].iloc[0], poi_B_data['lon'].iloc[0]),
            (row['lat'], row['lon']),
            unit=Unit.METERS
        ),
        axis=1
    )

    # Ordenamos de mas cerca a mas lejos
    best = available_P.sort_values(by='distance', ascending=True).iloc[0] # ISSUE 30

    p2 = best['osm_id']
    p2_poib_dist = best['distance']
    
    '''
    # Calculamos de distancia real y sacamos el mejor osm_id
    p2, p2_poib_dist = find_closest_service(poi_B_data, available_P, networks_map)
    '''
    
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

def assign_data(list_variables, list_values, row_old):
    """
    Asigna valores de list_values a una fila (Series o dict) según reglas:
      - *_type y *_amount → enteros redondeados
      - *_time → redondeado a múltiplos de 30
      - resto → tal cual
    """

    row = row_old.copy() 
    
    for variable in list_variables:
        val = list_values.get(variable, None)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue  # no asignamos nada si falta valor

        if variable.endswith(('_type', '_amount', '_fixed')):
            try:
                val = int(round(val))
            except Exception:
                pass  # por si val no es numérico
        elif variable.endswith('_time'):
            try:
                val = int(round(val / 30.0) * 30)
            except Exception:
                pass
        
        row[variable] = val

    return row

def add_public_walk(avail_vehicles, citizen_data, pop_archetypes_transport):

    # Sacamos las caracteristicas con valores estadisticos
    variables = [
            col.rsplit('_', 1)[0]
            for col in pop_archetypes_transport.columns
            if col.endswith('_mu')
        ]
    
    ## Walk
    # Creamos la nueva fila
    new_row = {
        'name': 'walk',
        'archetype': 'walk',
        'family': citizen_data['family'],
        'ubication': citizen_data['Home'],
    }
    # Creamos valores estocasticos
    values = get_vehicle_stats(new_row['archetype'], pop_archetypes_transport, variables)
    row_updated = assign_data(variables, values, new_row)
    # Cambiamos 'v' (esta mejor asignado en pop_citizen)
    row_updated['v'] = citizen_data['walk_speed'] # No es del citizen, si no del mas lento del grupo!!!! ISSUE 28
    # La añadimos al df de resultados
    avail_vehicles.append(row_updated)

    # Devolvemos la version actualizada
    return avail_vehicles

def route_creation(todo_list):
    # 1) ordenar por trip ascendente (equivalente a sort_values(by='trip'))
    todo_list_sorted = sorted(todo_list, key=lambda r: r['trip'])
    # 2) extraer la secuencia de osm_id (equivalente a df['osm_id'].tolist())
    simpl_route = [row['osm_id'] for row in todo_list_sorted]
    # 3) construir pares consecutivos (igual que antes)
    route = list(zip(simpl_route, simpl_route[1:]))
    return route
    
def get_independents(todo_list_all):
    return todo_list_all[(todo_list_all['todo'] == 'WoS') & (todo_list_all['fixed'] == False)]['agent'].unique()
    
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