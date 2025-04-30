import pandas as pd
import numpy as np
from pathlib import Path

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
        print(f'Familia {family_df["family"].iloc[0]} no tiene responsables!!!!')
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
def main_td(df_citizens, SG_relationship):
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')

    results = pd.DataFrame(columns=['agent', 'route'])

    # Recorrer cada familia
    for family_name in df_citizens['family'].unique():
        family_df = df_citizens[df_citizens['family'] == family_name]
        df_family_result = assign_responsable(family_df, SG_relationship_unique)
        
        if df_family_result is not None:
            results = pd.concat([results, df_family_result], ignore_index=True)
        
        routes = []
            
        for _, family_member in family_df.iterrows():
            id_type_not_0_list = df_family_result['id_type_not_0'].tolist() if df_family_result is not None else []

            if family_member['name'] not in id_type_not_0_list:
                routes.append({'agent': family_member['name'], 'route': [family_member['WoS']]})
            else:
                related_id = df_family_result[df_family_result['id_type_not_0'] == family_member['name']]['id_type_0'].iloc[0]
                # Buscamos los WoS relacionados al id_type_0
                related_wos = df_citizens[df_citizens['name'] == related_id]['WoS'].tolist()
                route = [family_member['WoS']] + related_wos
                routes.append({'agent': family_member['name'], 'route': route})
        routes_df = pd.DataFrame(routes)
        print(routes_df)
    # Unir todos los resultados
    if not results.empty:
        final_df = results
    else:
        print('No se encontró ningún dependiente con responsable.')
        final_df = None
    
    return final_df

# Ejecución
if __name__ == '__main__':
    main_td(df_citizens, SG_relationship)
