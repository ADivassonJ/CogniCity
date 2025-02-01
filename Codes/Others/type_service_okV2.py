import os
import random
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon

# Function to get the centroid or point of a given geometry (to get approximate coordinates of the service location)
def obtener_centroide_o_punto(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        return centroid.y, centroid.x
    elif isinstance(geom, Point):
        return geom.y, geom.x
    return None, None

# Function to get buildings labeled as 'building' in the given area
def obtener_edificios_ciudad(ciudad):
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

# Function to compare a building ID with the type of service it should provide
def type_service_ok(directory, df_area, osmid, service_type):
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

# Function to cluster all buildings based on their services
def buildind_to_services(directory, type_services, service_building_dict, df):
    for service_type in type_services:
        ava_serv = []
        for i in range(len(df)):
            is_ok = type_service_ok(directory, df, df['osmid'][i], service_type)
            if is_ok:
                ava_serv.append(df['osmid'][i])
        service_building_dict[service_type] = ava_serv

def main():
    ciudad = "Otxarkoaga"
    population = 1000
    directory = 'C:/Users/asier.divasson/Downloads'
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']

    print("Gathering data from selected area...")
    buildings_df = obtener_edificios_ciudad(ciudad)
    print("Data from selected area obtained.")
    print("Tagging all buildings to their services...")
    services_buildings = {}
    buildind_to_services(directory, type_services, services_buildings, buildings_df)
    print("Tagging completed.")

    buildings_info = {osmid: [] for osmid in services_buildings['Living']}
    agents_names = [f'agent_{i}' for i in range(population)]

    living_count = len(services_buildings['Living'])
    residents_per_building = int(population / living_count)
    i = 0

    for living in services_buildings['Living']:
        for _ in range(residents_per_building):
            buildings_info[living].append(agents_names[i])
            i += 1

    remaining_agents = population - residents_per_building * living_count
    if remaining_agents != 0:
        for living in services_buildings['Living']:
            if remaining_agents == 0:
                break
            buildings_info[living].append(agents_names[i])
            i += 1
            remaining_agents -= 1

    # Find and display the building containing agent_999
    for osmid, agents in buildings_info.items():
        if 'agent_999' in agents:
            print(f"Building {osmid} contains agent_999.")
            break
    

if __name__ == "__main__":
    main()
