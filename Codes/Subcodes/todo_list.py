import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic

# Cargar datos
study_area = 'Kanaleneiland'
main_path = Path(__file__).resolve().parent.parent.parent
data_path = main_path / 'Data'
study_area_path = data_path / study_area
df_citizens = pd.read_excel(f'{study_area_path}/df_citizens.xlsx')
SG_relationship = pd.read_excel(f'{study_area_path}/SG_relationship.xlsx')

# Función de distancia haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

# Función que crea la matriz de distancias de cada familia
def StoW_matrix_creation(family_df, SG_relationship_unique):
    # Separar los dos grupos
    list_type_0 = family_df[family_df['WoS_type'] == 0]
    list_type_not_0 = family_df[family_df['WoS_type'] != 0]

    # Hacer merge para traer lat y lon
    list_type_0 = list_type_0.merge(SG_relationship_unique[['osm_id', 'lat', 'lon']], left_on='WoS', right_on='osm_id', how='left')
    list_type_not_0 = list_type_not_0.merge(SG_relationship_unique[['osm_id', 'lat', 'lon']], left_on='WoS', right_on='osm_id', how='left')

    # Si alguno está vacío, devolvemos None
    if list_type_0.empty:
        return None
    if list_type_not_0.empty:
#        print(f'Familia {family_df["family"].iloc[0]} no tiene responsables!!!!')
        return None

    # Crear combinaciones y calcular distancias
    rows = []
    for idx_0, row_0 in list_type_0.iterrows():
        for idx_n0, row_n0 in list_type_not_0.iterrows():
            if pd.notnull(row_0['lat']) and pd.notnull(row_0['lon']) and pd.notnull(row_n0['lat']) and pd.notnull(row_n0['lon']):
                distance = haversine(row_0['lat'], row_0['lon'], row_n0['lat'], row_n0['lon'])
                rows.append({
                    'family': family_df["family"].iloc[0],
                    'id_type_0': row_0['name'],
                    'id_type_not_0': row_n0['name'],
                    'distance_km': distance
                })

    # Crear DataFrame de resultados
    StoW_matrix = pd.DataFrame(rows)
    return StoW_matrix

# Función que asigna el responsable más cercano a cada dependiente en una familia
def assign_responsable(family_df, SG_relationship_unique):
    StoW_matrix = StoW_matrix_creation(family_df, SG_relationship_unique)

    # Si no hay datos (familia vacía), saltamos
    if StoW_matrix is None or StoW_matrix.empty:
        return None

    # Para cada id_type_0, encontrar el id_type_not_0 más cercano
    idx_min = StoW_matrix.groupby('id_type_0')['distance_km'].idxmin()
    df_min_distances = StoW_matrix.loc[idx_min].reset_index(drop=True)

    return df_min_distances

# Función principal
def main_td():
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')

    results = pd.DataFrame(columns=['agent', 'route'])

    # Recorrer cada familia
    for family_name in df_citizens['family'].unique():
        family_df = df_citizens[df_citizens['family'] == family_name]
        df_family_result = assign_responsable(family_df, SG_relationship_unique)
        if df_family_result is not None:
            results = pd.concat([results, df_family_result], ignore_index=True)
        else:
            continue
            
        for _, family_member in family_df.iterrows():
            id_type_not_0_list = df_family_result['id_type_not_0'].tolist() if df_family_result is not None else []

            if family_member['name'] in id_type_not_0_list:
                related_id = df_family_result[df_family_result['id_type_not_0'] == family_member['name']]['id_type_0'].iloc[0]
                # Buscamos los WoS relacionados al id_type_0
                related_wos = df_citizens[df_citizens['name'] == related_id]['WoS'].tolist()
                route = [family_member['home']] + related_wos + [family_member['WoS']]
                # aqui el agente mira que transporte mode tomar
                choice_modeling()

    # Unir todos los resultados
    if not results.empty:
        final_df = results
    else:
        print('No se encontró ningún dependiente con responsable.')
        final_df = None
    
    return final_df


def choice_modeling(citizen):
    acutime_matrix = acutime_matrix_creation(citizen)
    willinness_matrix = willinness_calculation(acutime_matrix)

    #seleccionar de forma estadistica cual elegir, en base a willinness_matrix que sera como:
    # archetype    wilinness
    # walk         1.2
    # E_car        5.2
    # E_micro      2.3
    #        ...

def acutime_matrix_creation(df_priv_vehicles, df_citizens, route, SG_relationship, transport_archetypes, networks_map, citizen):
    
    acutime_matrix = pd.DataFrame(columns=['archetype', 't_walk', 't_travel', 't_wait', 'cost', 'benefict', 'CO2'])
    transport_archetypes = transport_archetypes['name'].tolist()
    
    for archetype in transport_archetypes:
        value = transport_archetypes.loc[transport_archetypes['name'] == archetype, 'P_1']

        if not value.empty and value.iloc[0] == 'x':
            #### sumar los intermedios a route
            route_methods = []
            
        route_methods = []    # si tienes tres POIs, tendrias dos methods,, si pillas coche, seria walk entre poi home y poi P_1, 
        #                       drive entre P_1 y P_2, walk entre P_2 y WoS
            
        for idx in range(len(route)):
            if idx+1 == len(route):
                break
            # Coordenadas de origen y destino (lat, lon)
            lat1, lon1 = SG_relationship.loc[SG_relationship['osm_id'] == route[idx], {'lat', 'lon'}]
            lat2, lon2 = SG_relationship.loc[SG_relationship['osm_id'] == route[idx+1], {'lat', 'lon'}]

            graph = networks_map[f"{route_methods[idx]}_map"]
                
            # Encontrar los nodos más cercanos en el grafo
            orig_node = ox.distance.nearest_nodes(graph, X=lon1, Y=lat1)
            dest_node = ox.distance.nearest_nodes(graph, X=lon2, Y=lat2)

            # Calcular la ruta más corta en distancia
            route = nx.shortest_path(graph, orig_node, dest_node, weight='length')

            # Calcular la longitud total de la ruta (en metros)
            route_length = nx.path_weight(graph, route, weight='length')
            
            print(route_length)
    
    

# Ejecución
if __name__ == '__main__':
    main_td(df_citizens, SG_relationship)
