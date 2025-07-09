import os
import sys
import numpy as np
import osmnx as ox
import pandas as pd
from pathlib import Path









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
    
    # Agrupamos los resultados por familias
    level1_families = level_1_results.groupby(['family'])
    level2_families = level_2_results.groupby(['family'])
    transport_families = pop_transport.groupby(['family'])
    
    # Inicializamos la memoria de trips ya evaluados
    eval_trips = []
    
    # Pasamos por los datos de cada una de las familias
    for f_name, family in level2_families:
        # Inicializamos distime_matrix de la familia
        distime_matrix = pd.DataFrame()
        # Logramos los vehiculos asignados a cada miembro familiar
        avail_vehicles = transport_families.get_group(f_name)
        # Sacamos el schedule de level1 tambien
        level1_schedule = level1_families.get_group(f_name)
        # Sacamos los nombres de los agentes independientes
        independents = get_independents(level1_schedule)
        # Obtenemos el orden en el que iterar los independent (de route más a menos larga)
        independents = organize_independents(independents, family)
        # Agrupamos las actividades familiares por sus miembros
        level2_citizens = family.groupby(['agent'])
        level1_citizens = level1_schedule.groupby(['agent'])
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
            vehicle_score_matrix = VSM_calculation(citizen_route, avail_vehicles, citizen_data, pop_archetypes_transport)
            
def VSM_calculation(citizen_route, avail_vehicles, citizen_data, pop_archetypes_transport):
    # Añadimos a la matriz de vehiculos disponibles el publico y andar
    avail_transport = add_public_walk(avail_vehicles, citizen_data, pop_archetypes_transport)
    
    input(avail_transport)
    


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
        'enkm': 0,
        'Emin': 0,
        'COkm': 0,
        'SoC': 0,
    }
    # La añadimos al df de resultados
    avail_vehicles = pd.concat([avail_vehicles, pd.DataFrame([new_row])], ignore_index=True)
    ## Public transport
    # Sacamos las caracteristicas con valores estadisticos
    variables = ['v', 'Ekm', 'enkm', 'Emin', 'COkm', 'SoC']
    values = get_vehicle_stats('conb_public', pop_archetypes_transport, variables)
    # Creamos la nueva fila
    new_row = {
        'name': 'public',
        'archetype': 'public',
        'family': citizen_data['family'],
        'ubication': citizen_data['Home'],
        'v': values['v'],
        'Ekm': values['Ekm'],
        'enkm': values['enkm'],
        'Emin': values['Emin'],
        'COkm': values['COkm'],
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
    simpl_route = citizen_schedule['osm_id'].unique()
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