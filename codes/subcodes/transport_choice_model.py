import os
import sys
import math
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm
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
    
    pop_citizen = pd.read_excel(f"{paths['population']}/pop_citizen.xlsx")
    pop_family = pd.read_excel(f"{paths['population']}/pop_family.xlsx")
    pop_building = pd.read_excel(f"{paths['population']}/pop_building.xlsx")
    pop_transport = pd.read_excel(f"{paths['population']}/pop_transport.xlsx")
    
    pop_archetypes_transport = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_transport.xlsx")
    
    ##############################################################################
    print(f'docs readed')
    
    output_for_tugraz = copy_vehicles(level_1_results, pop_transport, pop_citizen, pop_building)
    output_for_tugraz.to_excel(f"{paths['results']}/{study_area}_vehicles.xlsx", index=False)
    

def get_independents(level1_schedule):
    return level1_schedule[(level1_schedule['todo_type'] == 0) & (level1_schedule['todo'] == 'WoS')]['agent'].unique() 
    
def copy_vehicles(level_1_results, pop_citizen, pop_building):
    # Agrupamos los resultados por familias
    level1_families = level_1_results.groupby(['family'])
    
    vehicles_actions = pd.DataFrame()
    
    # Pasamos por los datos de cada una de las familias
    for f_name, family in tqdm(level1_families, desc="Procesando familias"):
        # Sacamos los nombres de los agentes independientes
        independents = get_independents(family)
        # Agrupamos las actividades familiares por sus miembros
        level1_citizens = family.groupby(['agent'])
        # Inicializamos el nuevo schedule
        new_family_schedule = pd.DataFrame()
        for c_name in independents:
            # Logramos el schedule especifico de este agente
            citizen_schedule = level1_citizens.get_group(c_name).reset_index(drop=True)
            # Sacamos sus datos de agente
            citizen_data = pop_citizen[pop_citizen['name'] == c_name].iloc[0]
            # Decidimos el transporte
            choosen_transport = vehicle_chosing()
            # modificamos el schedule 
            new_family_schedule = update_schedule(citizen_schedule, choosen_transport, pop_building)
            
            # Lo sumamos al total
            vehicles_actions = pd.concat([vehicles_actions, new_family_schedule], ignore_index=True)
            
    return vehicles_actions

def vehicle_chosing():
    
    # Ejemplo de DataFrame
    df = pd.DataFrame({
        'transport': ['walk', 'bicicle', 'bus', 'car', 'micro'],
        'prob': [21.5, 8, 21.5, 46, 3]
    })
    # Normalizamos las probabilidades
    probs_normalizadas = df['prob'] / df['prob'].sum()
    # Elegimos según la probabilidad
    eleccion = np.random.choice(df['transport'], p=probs_normalizadas)
    
    return eleccion

def update_schedule(citizen_schedule, choosen_transport, pop_building):
    
    vehicle_schedule = pd.DataFrame()
    
    if choosen_transport == 'car':
        # Ejemplo de DataFrame
        df = pd.DataFrame({
            'transport': ['conb', 'EV', 'V2G'],
            'prob': [8.86, 1, 0.14]
        })
        # Normalizamos las probabilidades
        probs_normalizadas = df['prob'] / df['prob'].sum()
        # Elegimos según la probabilidad
        eleccion = np.random.choice(df['transport'], p=probs_normalizadas)
        
        if eleccion == 'conb':
            return pd.DataFrame()
        
        s = citizen_schedule['agent'].iloc[0]
        numb = s.split("_")[1]
        
        #new
        agent = f"vehicle_{numb}"
        archetype = eleccion
        
        for num, row in citizen_schedule.iterrows():
            #old
            osm_id = row['osm_id']
            node = row['node']
            in_v = row['in']
            out_v = row['out']
            #Energy Total Cost
            if num == 0:
                ETC_v = 0
            else:
                ETC_v = last_ETC_v + calculate_energy_consumption(pop_building, osm_id, citizen_schedule['osm_id'][num-1])
            last_ETC_v = ETC_v
            
            new_row = [{
                'agent':agent,
                'archetype': archetype,
                'osm_id': osm_id,
                'node': node,
                'in': in_v,
                'out': out_v,
                'ETC [kWh]': ETC_v,
            }]
            vehicle_schedule = pd.concat([vehicle_schedule, pd.DataFrame(new_row)], ignore_index=True).reset_index(drop=True)
            
        return vehicle_schedule
    else:
        return pd.DataFrame()


def calculate_energy_consumption(pop_building, osm_id, prev_osm_id, consumo_kwh_por_km=3.927): #0.0927
    # Obtener info de cada punto
    b0_info = pop_building[pop_building['osm_id']==osm_id].iloc[0]
    b1_info = pop_building[pop_building['osm_id']==prev_osm_id].iloc[0]
    
    # Extraer latitudes y longitudes
    lat0, lon0 = b0_info['lat'], b0_info['lon']
    lat1, lon1 = b1_info['lat'], b1_info['lon']
    
    # Convertir a radianes
    lat0_rad, lon0_rad = math.radians(lat0), math.radians(lon0)
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    
    # Diferencias
    dlat = lat1_rad - lat0_rad
    dlon = lon1_rad - lon0_rad
    
    # Fórmula de Haversine
    a = math.sin(dlat/2)**2 + math.cos(lat0_rad) * math.cos(lat1_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Radio de la Tierra en metros
    R = 6371000
    distance_m = R * c
    
    # Convertir distancia a km
    distance_km = distance_m / 1000
    
    # Calcular consumo energético en kWh
    energy_kwh = distance_km * consumo_kwh_por_km
    
    return energy_kwh



if __name__ == '__main__':
    main()