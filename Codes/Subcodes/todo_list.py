import pandas as pd
import numpy as np
from pathlib import Path

# Supongamos que ya tienes df_citizens y SG_relationship cargados
study_area = 'Kanaleneiland'
main_path = Path(__file__).resolve().parent.parent.parent
data_path = main_path / 'Data'
study_area_path = data_path / study_area
df_citizens = pd.read_excel(f'{study_area_path}/df_citizens.xlsx')
SG_relationship = pd.read_excel(f'{study_area_path}/SG_relationship.xlsx')


def StoW_matrix_creation(df_citizens, SG_relationship):

    # 1. Separar los dos grupos
    list_type_0 = df_citizens[df_citizens['WoS_type'] == 0]
    list_type_not_0 = df_citizens[df_citizens['WoS_type'] != 0]

    # Limpiar duplicados antes del merge
    SG_relationship_unique = SG_relationship.drop_duplicates(subset='osm_id')

    # Hacer el merge limpio
    list_type_0 = list_type_0.merge(SG_relationship_unique[['osm_id', 'lat', 'lon']], left_on='WoS', right_on='osm_id', how='left')
    list_type_not_0 = list_type_not_0.merge(SG_relationship_unique[['osm_id', 'lat', 'lon']], left_on='WoS', right_on='osm_id', how='left')

    # 4. Crear combinaciones y calcular distancias
    rows = []

    for idx_0, row_0 in list_type_0.iterrows():
        for idx_n0, row_n0 in list_type_not_0.iterrows():
            # Verificar que ambos tengan lat y lon
            if pd.notnull(row_0['lat']) and pd.notnull(row_0['lon']) and pd.notnull(row_n0['lat']) and pd.notnull(row_n0['lon']):
                distance = haversine(row_0['lat'], row_0['lon'], row_n0['lat'], row_n0['lon'])
                rows.append({
                    'id_type_0': row_0['name'],
                    'id_type_not_0': row_n0['name'],
                    'distance_km': distance
                })

    # 5. Crear el DataFrame de resultados
    StoW_matrix = pd.DataFrame(rows)

    return StoW_matrix

def assign_reponsable(df_citizens, SG_relationship):   
    StoW_matrix = StoW_matrix_creation(df_citizens, SG_relationship) 
    
    # Para cada id_type_0, encontrar el id_type_not_0 con la distancia mínima
    idx_min = StoW_matrix.groupby('id_type_0')['distance_km'].idxmin()

    # Seleccionar las filas correspondientes
    df_min_distances = StoW_matrix.loc[idx_min].reset_index(drop=True)

    print(df_min_distances)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

if __name__ == '__main__':
    assign_reponsable(df_citizens, SG_relationship)