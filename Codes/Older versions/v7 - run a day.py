import os
import math
import random
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from geopy.distance import great_circle

def obtener_centroide_o_punto(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        return centroid.y, centroid.x
    elif isinstance(geom, Point):
        return geom.y, geom.x
    return None, None

def obtener_edificios_ciudad(ciudad):
    try:
        print(f"Retrieving building data for {ciudad}...")
        edificios = ox.features_from_place(ciudad, tags={'building': True})
        building_ID = edificios.index.get_level_values('osmid').tolist()

        values = {'osmid': building_ID, 'coordenadas': []}
        names = ['osmid', 'coordenadas']

        for geom in edificios['geometry']:
            resultado = obtener_centroide_o_punto(geom)
            values['coordenadas'].append(resultado)

        variables = ['building', 'amenity', 'geometry']
        for vari in variables:
            if vari in edificios.columns:
                data_array = edificios[vari].tolist()
                names.append(vari)
                values[vari] = data_array

        buildings_data = pd.DataFrame(values, columns=names)
        return buildings_data
    except Exception as e:
        print(f"Error retrieving buildings data: {e}")
        return pd.DataFrame()

def type_service_ok(directory, df_area, osmid, service_type):
    try:
        to_study = df_area.loc[df_area['osmid'] == osmid, 'amenity']
        if pd.isna(to_study.iloc[0]) or to_study.iloc[0] == 'yes':
            to_study = df_area.loc[df_area['osmid'] == osmid, 'building']
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
        print(f"Error in type_service_ok: {e}")
        return False

def buildind_to_services(directory, type_services, service_building_dict, df):
    for service_type in type_services:
        ava_serv = []
        for i in range(len(df)):
            is_ok = type_service_ok(directory, df, df['osmid'][i], service_type)
            if is_ok:
                ava_serv.append(df['osmid'][i])
        service_building_dict[service_type] = ava_serv

def df_a_excel(directory, df, nombre):
    try:
        path = os.path.join(directory, f'{nombre}.xlsx')
        df.to_excel(path, index=False)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

def initialize_population(services_buildings, population):
    agents_names = [f'citizen_{i}' for i in range(population)]
    buildings_info = {osmid: [] for osmid in services_buildings.get('Living', [])}
    
    day = 0
    time = 0
    living_count = len(services_buildings.get('Living', []))
    residents_per_building = population // living_count if living_count else 0
    SoC = [0]*population
    i = 0

    for living in services_buildings.get('Living', []):
        for _ in range(residents_per_building):
            code = f"{day},{time},{agents_names[i]},in,citizen,0,{SoC[i]}"
            buildings_info[living].append(code)
            i += 1

    remaining_agents = population % living_count
    for living in services_buildings.get('Living', []):
        if remaining_agents == 0:
            break
        code = f"{day},{time},{agents_names[i]},in,citizen,0,{SoC[i]}"
        buildings_info[living].append(code)
        i += 1
        remaining_agents -= 1

    return buildings_info

def initialization(ciudad, population, directory, type_services):
    buildings_df = obtener_edificios_ciudad(ciudad)
    if buildings_df.empty:
        print("No data retrieved from the city.")
        return

    print("Data from selected area obtained.")
    print("Tagging all buildings to their services...")
    services_buildings = {}
    buildind_to_services(directory, type_services, services_buildings, buildings_df)
    print("Tagging completed.")

    buildings_info = initialize_population(services_buildings, population)

    data_for_df = []
    for osmid, agent_data_list in buildings_info.items():
        for agent_data in agent_data_list:
            day, time, agent_name, in_out, agent_type, archetype, SoC = agent_data.split(',')
            data_for_df.append({
                'osmid': osmid,
                'day': day,
                'time': time,
                'agent_name': agent_name,
                'in_out': in_out,
                'agent_type': agent_type,
                'archetype': archetype,
                'SoC': SoC
            })

    df_buildings_info = pd.DataFrame(data_for_df)

    df_a_excel(directory, df_buildings_info, 'buildings_info')
    
    return df_buildings_info, buildings_df

import random  # Ensure random module is imported

def next_move(data, buildings_df):
    building_name = data['osmid']
    building_coord = buildings_df.loc[buildings_df['osmid'] == building_name, 'coordenadas'].iloc[0]
    new_building = building_name

    while new_building == building_name:
        new_building = random.choice(buildings_df['osmid'])  # Use random.choice to select a new building
    new_building_coord = buildings_df.loc[buildings_df['osmid'] == new_building, 'coordenadas'].iloc[0]
    distance = great_circle(building_coord, new_building_coord).kilometers
    travel_time = math.ceil((distance / 3)*12)  # Assuming 3 km/hr speed for travel time calculation
    new_data = data.copy()
    new_data['osmid'] = new_building
    new_data['time'] += travel_time
    new_data['in_out'] = 'in'
    new_data['SoC'] = str(int(new_data['SoC']) - travel_time*10)
    
    return new_data

if __name__ == "__main__":
    ciudad = "Otxarkoaga"
    population = 10
    directory = 'C:/Users/asier.divasson/Downloads'
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']
    
    moving_agents = []  # Should be initialized as an empty list
    df_buildings_info, buildings_df = initialization(ciudad, population, directory, type_services)

    if not df_buildings_info.empty:
        time = 0
        day = 0
        agents_names = [f'citizen_{i}' for i in range(population)]

        while time < 7:
            time += 1
            print('-'*20, 'time:', time, '-'*20)
            
            for agent in moving_agents['agent_name']:
                row = moving_agents.loc[moving_agents['agent_name'] == agent].iloc[0]
                time_data = row['time']
                if time_data == time:
                    df_buildings_info.append(agent)
                    # add here a way to eliminate the data from moving_agent for this agent
                elif time_data < time:
                    print('ERROR ERROR ERROR; time_data < time')
            
            for agent in agents_names:
                # Check if the agent is currently moving
                if random.choice(['move', 'stay']) == 'move' and agent not in [a['agent_name'] for a in moving_agents]:
                    row = df_buildings_info.loc[df_buildings_info['agent_name'] == agent].iloc[0]
                    osmid_data = row['osmid']
                    day_data = day
                    time_data = time
                    agent_name_data = row['agent_name']
                    in_out_data = 'out'
                    agent_type_data = row['agent_type']
                    archetype_data = row['archetype']
                    SoC_data = row['SoC']
                    print(osmid_data, day_data, time_data, agent_name_data, in_out_data, agent_type_data, archetype_data, SoC_data)

                    new_row = pd.DataFrame([{
                        'osmid': osmid_data,
                        'day': day_data,
                        'time': time_data,
                        'agent_name': agent_name_data,
                        'in_out': in_out_data,
                        'agent_type': agent_type_data,
                        'archetype': archetype_data,
                        'SoC': SoC_data
                    }])

                    df_buildings_info = pd.concat([df_buildings_info, new_row], ignore_index=True)
                    
                    new_data = next_move(new_row.iloc[0], buildings_df)
                    
                    moving_agents.append(new_data)
                    
    print(moving_agents)