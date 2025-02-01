import os
import random
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon

def obtener_centroide_o_punto(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        return centroid.y, centroid.x
    elif isinstance(geom, Point):
        return geom.y, geom.x
    return None, None

def obtener_edificios_ciudad(ciudad):
    try:
        edificios = ox.features_from_place(ciudad, tags={'building': True})
        building_ID = edificios.index.get_level_values('osmid').tolist()

        values = {'osmid': building_ID, 'coordenadas': []}
        names = ['osmid', 'coordenadas']

        for geom in edificios['geometry']:
            resultado = obtener_centroide_o_punto(geom)
            values['coordenadas'].append(resultado)

        variables = ['building', 'amenity', 'geometry']

        for vari in variables:
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
    buildings_info = {osmid: [] for osmid in services_buildings['Living']}
    
    day = 0
    time = 0
    living_count = len(services_buildings['Living'])
    residents_per_building = population // living_count
    i = 0

    for living in services_buildings['Living']:
        for _ in range(residents_per_building):
            code = f"{day},{time},{agents_names[i]},in,citizen,0"
            buildings_info[living].append(code)
            i += 1

    remaining_agents = population % living_count
    for living in services_buildings['Living']:
        if remaining_agents == 0:
            break
        code = f"{day},{time},{agents_names[i]},in,citizen,0"
        buildings_info[living].append(code)
        i += 1
        remaining_agents -= 1

    return buildings_info

def main():
    ciudad = "Otxarkoaga"
    population = 1000
    directory = 'C:/Users/asier.divasson/Downloads'
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']

    print("Gathering data from selected area...")
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

    # Flatten the buildings_info dictionary into a list of dictionaries for DataFrame conversion
    data_for_df = []
    for osmid, agent_data_list in buildings_info.items():
        for agent_data in agent_data_list:
            day, time, agent_name, in_out, agent_type, archetype = agent_data.split(',')
            data_for_df.append({
                'osmid': osmid,
                'day': day,
                'time': time,
                'agent_name': agent_name,
                'in_out': in_out,
                'agent_type': agent_type,
                'archetype': archetype
            })

    df_buildings_info = pd.DataFrame(data_for_df)

    df_a_excel(directory, df_buildings_info, 'buildings_info')

if __name__ == "__main__":
    main()